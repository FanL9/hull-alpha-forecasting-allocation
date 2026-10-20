"""Execute the preregistered, full-budget artificial Phase 5 panels."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from src.research.phase5_execution import run_synthetic, synthetic_destination, verified_completed_synthetic
from src.research.phase5_synthetic import synthetic_project
from src.research.phase5_tasks import atomic, read
from src.utils.paths import project_root


def assess(case,summary):
    means=summary.set_index('pipeline').mean_fold_ic
    finite=summary.set_index('pipeline').finite_folds
    if case=='noise':
        passed=all(np.isfinite(means[n]) and abs(means[n])<=.10 for n in ('Ridge','GRU','TCN','C','D'))
    elif case=='linear':
        passed=means['Ridge']>.30 and means['GRU']>.15 and means['TCN']>.15 and all(finite[n]==11 for n in ('Ridge','GRU','TCN'))
    elif case=='nonlinear':
        passed=max(means['GRU'],means['TCN'])-means['Ridge']>.05 and all(finite[n]==11 for n in ('Ridge','GRU','TCN'))
    else:
        raise ValueError('Unknown preregistered panel.')
    return {'case':case,'passed':bool(passed),'mean_fold_ic':means.to_dict(),
            'nonlinear_minus_ridge':{n:float(means[n]-means['Ridge']) for n in ('GRU','TCN')}}


def run(output,workers):
    root=project_root(); output=synthetic_destination(output)
    output.mkdir(parents=True,exist_ok=True)
    specification=json.loads((root/'configs/phase5_calibration.json').read_text(encoding='utf-8'))
    results=[]
    for case,seed in specification['panels'].items():
        source=output/(case+'_source')
        if not source.exists():
            synthetic_project(source,case,seed)
        destination=output/(case+'_run')
        start=time.perf_counter()
        if (destination/'phase5_run_manifest.json').exists() and read(destination/'execution_state.json')['state']=='COMPLETE':
            summary=verified_completed_synthetic(source,destination)
        else:
            summary=run_synthetic(source,destination,workers)['summary']
        result={**assess(case,summary),'seed':seed,'invocation_seconds':time.perf_counter()-start}
        atomic(output/(case+'_assessment.json'),result)
        print(json.dumps(result),flush=True)
        results.append(result)
        if not result['passed']:
            raise RuntimeError('Predeclared '+case+' calibration criterion failed. Stop before freezing; no model retuning.')
    atomic(output/'calibration_summary.json',{'specification':specification,'results':results,'real_market_data_read':False})
    return results
