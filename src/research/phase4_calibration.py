"""Full eleven-fold synthetic calibration, using the frozen v3.1 seed panel."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from src.research.alpha_calibration import artificial_data_only
from src.research.alpha_pipeline import load_alpha_design
from src.research.phase4_engine import build_phase4_tables
from src.research.phase4_execution import execution_identity, write_outputs
from src.research.phase4_synthetic import make_history
from src.validation.protocol_v3_1 import load_config


def synthetic_run(root: Path, history: pd.DataFrame, output: Path):
    if history.attrs.get('phase4_data_kind')!='synthetic':
        raise ValueError('Synthetic entry accepts only explicitly marked artificial history.')
    if any(output.resolve().is_relative_to((root/folder).resolve()) for folder in ('artifacts','data')):
        raise ValueError('Synthetic output must remain separate from canonical artifacts and data.')
    with artificial_data_only(root):
        tables = build_phase4_tables(history,load_config(root),load_alpha_design(root))
        manifest = execution_identity(root,'synthetic')
        manifest.update(synthetic_seed=history.attrs.get('synthetic_seed'),synthetic_case=history.attrs.get('synthetic_case'))
        write_outputs(tables,output,manifest)
        return tables


def _case(root_string, workspace_string, kind, seed):
    root, workspace = Path(root_string), Path(workspace_string)
    start = time.monotonic()
    with artificial_data_only(root):
        history = make_history(root,seed,kind)
        tables = synthetic_run(root,history,workspace/f'{kind}-{seed}')
        selected = tables.selection.loc[tables.selection.selected].copy()
        selected['case']=kind; selected['seed']=seed
        result = tables.summary.copy(); result['case']=kind; result['seed']=seed
        result['best_apparent_inner_ic'] = result.pipeline.map(tables.selection.groupby('pipeline').ranking_score.max())
        folds = tables.folds.copy(); folds['case']=kind; folds['seed']=seed
        return {'summary':result.to_dict('records'),'selected':selected.to_dict('records'),
            'folds':folds.to_dict('records'),'case':kind,'seed':seed,'seconds':time.monotonic()-start}


def summarize_null(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pipeline, g in summary.loc[summary.case=='noise'].groupby('pipeline'):
        values = g.mean_fold_ic.to_numpy(float)
        if len(values)!=20 or not np.isfinite(values).all() or not (g.finite_folds==11).all():
            raise RuntimeError('Incomplete null calibration; do not freeze.')
        std = values.std(ddof=1)
        threshold = max(.02,3*std/np.sqrt(20))
        rows.append({'pipeline':pipeline,'seeds':len(values),'mean':values.mean(),'std':std,
            'p05':np.quantile(values,.05),'p50':np.quantile(values,.5),'p95':np.quantile(values,.95),
            'min':min(values),'max':max(values),'alarm_threshold':threshold,
            'engineering_alarm':bool(abs(values.mean())>threshold)})
    return pd.DataFrame(rows)


def recovery(summary: pd.DataFrame, selected: pd.DataFrame) -> dict:
    raw = selected.loc[selected.case=='raw']
    correct = ((raw.candidate_id=='raw/M1/Level')&(raw.sign==1) |
        (raw.candidate_id=='raw/E1/Level')&(raw.sign==-1) |
        (raw.candidate_id=='raw/P1/Level')&(raw.sign==1))
    transformed = selected.loc[(selected.case=='transformed')&(selected.pipeline=='B')&
        (selected.candidate_id=='raw/M1/Delta21')&(selected.sign==1)]
    interaction = selected.loc[(selected.case=='regime')&(selected.pipeline=='B')&
        (selected.candidate_id=='interaction/P/HighV')&(selected.sign==1)]
    means = summary.groupby(['case','pipeline']).mean_fold_ic.mean()
    raw_counts = raw.loc[correct].groupby('pipeline').size().to_dict()
    return {'raw_correct_source_selections':raw_counts,
        'raw_mean_ic':{p:float(means['raw',p]) for p in ('A','B')},
        'raw_recovered':bool(all(raw_counts.get(p,0)>0 and means['raw',p]>0 for p in ('A','B'))),
        'delta21_correct_selections':len(transformed),
        'transformed_mean_ic':{p:float(means['transformed',p]) for p in ('A','B')},
        'transformed_mean_B_minus_A':float(means['transformed','B']-means['transformed','A']),
        'transformed_recovered':bool(len(transformed)>0 and means['transformed','B']>means['transformed','A']),
        'P_HighV_correct_selections':len(interaction),
        'regime_mean_ic':{p:float(means['regime',p]) for p in ('A','B')},
        'regime_recovered':bool(len(interaction)>0 and means['regime','B']>0)}


def run_calibration(root: Path, workspace: Path, evidence: Path, workers: int = 3) -> dict:
    config = load_config(root)
    panel = config['synthetic_calibration']
    jobs = [('noise',s) for s in panel['noise_seeds']]+[(k,s) for k in ('raw','transformed','regime') for s in panel['planted_seeds']]
    # Child Python imports inherit these caps before numpy/BLAS initializes.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    workspace.mkdir(parents=True,exist_ok=False)
    evidence.mkdir(parents=True,exist_ok=False)
    identity = execution_identity(root,'synthetic')
    completed = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_case,str(root),str(workspace),kind,seed):(kind,seed) for kind,seed in jobs}
        for future in as_completed(futures):
            case = future.result(); completed.append(case)
            progress = [{'case':c['case'],'seed':c['seed'],'seconds':round(c['seconds'],2)} for c in completed]
            (workspace/'progress.json').write_text(json.dumps({'completed':len(progress),'total':len(jobs),'cases':progress},indent=2),encoding='utf-8')
            print(f"Completed {case['case']} seed {case['seed']} ({len(completed)}/{len(jobs)})",flush=True)
    summary = pd.DataFrame([r for c in completed for r in c['summary']]).sort_values(['case','seed','pipeline'])
    selection = pd.DataFrame([r for c in completed for r in c['selected']]).sort_values(['case','seed','pipeline','fold_id','ranking_position'])
    folds = pd.DataFrame([r for c in completed for r in c['folds']]).sort_values(['case','seed','pipeline','fold_id'])
    null = summarize_null(summary)
    recovered = recovery(summary,selection)
    summary.to_csv(evidence/'per_seed_summary.csv',index=False)
    null.to_csv(evidence/'noise_distribution.csv',index=False)
    selection.to_csv(evidence/'selected_candidates.csv',index=False)
    folds.to_csv(evidence/'outer_fold_results.csv',index=False)
    # The first preregistered seed demonstrates every future audit output.
    shutil.copytree(workspace/f'noise-{panel["noise_seeds"][0]}',evidence/'synthetic_audit_example')
    manifest = {**identity,'calibration':'Phase4A synthetic only','noise_seeds':panel['noise_seeds'],
        'planted_seeds':panel['planted_seeds'],'cases':panel['cases'],'runs':len(jobs),'outer_folds_each':11,
        'raw_candidates':97,'engineered_candidates':499,'no_parameter_retuning':True,
        'engineering_alarm':bool(null.engineering_alarm.any()),'recovery':recovered,
        'source_config':'configs/protocol_v3_1.yaml','market_data_open_blocked':True}
    (evidence/'calibration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    if manifest['engineering_alarm']:
        raise RuntimeError('Systematic null IC alarm. Investigate without changing frozen rules; do not freeze.')
    if not all(recovered[n] for n in ('raw_recovered','transformed_recovered','regime_recovered')):
        raise RuntimeError('A planted recovery check was not met. Record the result; do not retune the protocol.')
    return manifest
