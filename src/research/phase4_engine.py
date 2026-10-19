"""Shared nested selector for A (97 Level) and B (499 engineered).

All selection and map fitting stops at a supplied purged reference end. Outer
labels enter only evaluate(), after forecasts for every block have been made.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.features.engineered_alpha import CandidateProvider, array_hash, eligibility, mid_ecdf
from src.research.alpha_pipeline import validate_research_history
from src.statistics.robust_inference import blockwise_rank_pairs
from src.validation.protocol_v3 import inner_splits
from src.validation.walk_forward import make_folds


def corr(x, y, rank=False) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    a, b = x[mask], y[mask]
    if len(a) < 3:
        return np.nan
    if rank:
        a, b = rankdata(a), rankdata(b)
    a, b = a-a.mean(), b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a @ b / den, -1, 1)) if den > 0 else np.nan


def column_ics(x: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank on complete pairs separately for each missingness pattern."""
    a = x.to_numpy(float)
    masks = np.isfinite(a) & np.isfinite(y[:,None])
    groups = {}
    for j in range(a.shape[1]):
        groups.setdefault(masks[:,j].tobytes(), []).append(j)
    rho = np.full(a.shape[1], np.nan)
    counts = masks.sum(axis=0)
    for ids in groups.values():
        mask = masks[:,ids[0]]
        if mask.sum() < 126 or mask.mean() < .5:
            continue
        ranks = rankdata(a[mask][:,ids], axis=0)
        target = rankdata(y[mask]); target -= target.mean()
        ranks -= ranks.mean(axis=0)
        den = np.linalg.norm(ranks, axis=0)*np.linalg.norm(target)
        rho[ids] = np.divide(target @ ranks, den, out=np.full(len(ids), np.nan), where=den > 0)
    return rho, counts


class PairRanks:
    """Exact pairwise deletion BEFORE ranking; cache by common mask."""
    def __init__(self, x: pd.DataFrame):
        self.x = x.to_numpy(float)
        self.mask = np.isfinite(self.x)
        self.cache = {}

    def pair(self, i: int, j: int) -> tuple[float, int]:
        mask = self.mask[:,i] & self.mask[:,j]
        n = int(mask.sum())
        if n < 252:
            return np.nan, n
        keymask = mask.tobytes()
        vectors = []
        for col in (i,j):
            key = col, keymask
            if key not in self.cache:
                v = rankdata(self.x[mask,col]); v -= v.mean()
                norm = np.linalg.norm(v)
                self.cache[key] = v/norm if norm else np.full(n, np.nan)
            vectors.append(self.cache[key])
        return float(np.clip(vectors[0] @ vectors[1], -1, 1)), n


def greedy_select(ranked: list[str], x: pd.DataFrame) -> tuple[list[str], list[dict]]:
    selected, audit = [], []
    pairs = PairRanks(x)
    lookup = {c:i for i,c in enumerate(x)}
    for name in ranked:
        rejected = False
        for partner in selected:
            rho, n = pairs.pair(lookup[name], lookup[partner])
            if not np.isfinite(rho) or abs(rho) >= .90:
                audit.append({'candidate_id': name, 'rejected_with': partner, 'rho': rho, 'pairs': n,
                    'reason': 'unknown_pair' if not np.isfinite(rho) else 'redundant'})
                rejected = True
                break
        if not rejected:
            selected.append(name)
        if len(selected) == 16:
            break
    return selected, audit


@dataclass
class FittedSelector:
    reference_end: int
    selected: tuple[str, ...]
    signs: dict[str, int]
    maps: dict[str, np.ndarray]
    candidates: pd.DataFrame
    redundancy: list[dict]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if len(self.selected) < 2:
            return np.zeros(len(frame))
        contributions = np.column_stack([self.signs[c]*(2*mid_ecdf(self.maps[c], frame[c].to_numpy(float))-1)
            for c in self.selected])
        observed = np.isfinite(contributions).sum(axis=1)
        # FIXED denominator; missing contributions become zero only after mapping.
        score = np.nansum(contributions, axis=1)/len(self.selected)
        score[observed < max(2, int(np.ceil(.5*len(self.selected))))] = np.nan
        return score

    def audit(self) -> dict:
        return {'reference_end': self.reference_end, 'selected': list(self.selected), 'signs': self.signs,
            'weights': {c: 1/len(self.selected) for c in self.selected},
            'map_hashes': {c: array_hash(v) for c,v in self.maps.items()},
            'status': 'active' if len(self.selected) >= 2 else 'no_alpha'}


class NestedSelector:
    """Single implementation; candidate provider is the only A/B difference."""
    def __init__(self, provider: CandidateProvider):
        self.provider = provider
        self.config = provider.config
        self.cache: dict[int, FittedSelector] = {}

    def fit(self, end: int) -> FittedSelector:
        if end in self.cache:
            return self.cache[end]
        p = self.provider
        final = p.frame(end, 0, end+1)
        audit = eligibility(final, self.config['availability']).set_index('feature')
        audit['final_eligible'] = audit.eligible
        for i, split in enumerate(inner_splits(end, self.config), 1):
            train = p.frame(split.training_end, 0, split.training_end+1)
            inner_availability = eligibility(train, self.config['availability']).set_index('feature')
            allowed = inner_availability.eligible
            validation = p.frame(split.training_end, split.validation_start, split.validation_end+1)
            ic, pairs = column_ics(validation, p.y.iloc[split.validation_start:split.validation_end+1].to_numpy(float))
            audit[f'inner_{i}_eligible'] = allowed
            audit[f'inner_{i}_reasons'] = inner_availability.reasons
            audit[f'inner_{i}_reference_end'] = split.training_end
            audit[f'inner_{i}_ic'] = ic
            audit[f'inner_{i}_pairs'] = pairs
        audit['eligible_all_references'] = audit[['final_eligible','inner_1_eligible','inner_2_eligible']].all(axis=1)
        a, b = audit.inner_1_ic, audit.inner_2_ic
        audit['same_sign'] = np.isfinite(a) & np.isfinite(b) & (a*b > 0)
        audit['inner_mean_ic'] = (a+b)/2
        audit['ranking_score'] = audit.inner_mean_ic.abs()
        pool = audit.loc[audit.eligible_all_references & audit.same_sign].reset_index()
        pool = pool.sort_values(['ranking_score','feature'], ascending=[False,True], kind='stable')
        selected, redundancy = greedy_select(pool.feature.tolist(), final)
        audit['selected'] = audit.index.isin(selected)
        audit['ranking_position'] = audit.index.map({c:i+1 for i,c in enumerate(pool.feature)})
        signs = {c: int(np.sign(audit.loc[c,'inner_mean_ic'])) for c in selected}
        maps = {c: np.sort(final[c].dropna().to_numpy(float)) for c in selected}
        state = FittedSelector(end, tuple(selected), signs, maps, audit.reset_index().rename(columns={'feature':'candidate_id'}), redundancy)
        self.cache[end] = state
        return state

    def forecast(self, end: int, start: int, stop: int) -> np.ndarray:
        if not end < start <= stop-1 <= self.config['research']['end']:
            raise ValueError('Forecast dates must follow the reference and stay in research.')
        return self.fit(end).predict(self.provider.frame(end, start, stop))

    def inner_oof_bounds(self, end: int) -> tuple[tuple[float,float], list[dict]]:
        forecasts, sources = [], []
        for split in inner_splits(end, self.config):
            # Complete earlier selector, including its own two screening blocks.
            forecasts.append(self.forecast(split.training_end, split.validation_start, split.validation_end+1))
            sources.append({'fit_end': split.training_end, 'start': split.validation_start, 'end': split.validation_end,
                'selector': self.fit(split.training_end).audit()})
        v = np.concatenate(forecasts); v = v[np.isfinite(v)]
        bounds = tuple(np.quantile(v, [.01,.99])) if len(v) else (np.nan,np.nan)
        return bounds, sources


@dataclass
class Phase4Tables:
    folds: pd.DataFrame
    selection: pd.DataFrame
    candidates: pd.DataFrame
    redundancy: pd.DataFrame
    transforms: dict
    summary: pd.DataFrame
    predictions: pd.DataFrame


def evaluate(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired primary metrics; structural availability is assessed without y."""
    active = []
    for pipeline, group in forecasts.groupby('pipeline'):
        blocks = [g for _,g in group.groupby('fold_id')]
        if all(g.status.iloc[0]=='active' and g.prediction.notna().sum()>=126 and g.prediction.nunique()>1 for g in blocks):
            active.append(pipeline)
    rows, pooled = [], {}
    for fold, block in forecasts.groupby('fold_id'):
        pivot = block.pivot(index='date_id', columns='pipeline', values='prediction')
        common = pivot[active].notna().all(axis=1) if active else pd.Series(False,index=pivot.index)
        for pipeline, g in block.groupby('pipeline'):
            g = g.sort_values('date_id')
            mask = common.loc[g.date_id].to_numpy() & np.isfinite(g.target)
            if pipeline not in active:
                mask[:] = False
            x = np.where(mask, g.prediction, np.nan)
            y = np.where(mask, g.target, np.nan)
            enough = int(mask.sum()) >= 126 and mask.mean() >= .5
            rank_ic = corr(x,y,True) if enough else np.nan
            clipped = np.clip(x, g.clip_lower.iloc[0], g.clip_upper.iloc[0])
            pearson = corr(clipped,y) if enough else np.nan
            individual = np.isfinite(g.prediction)&np.isfinite(g.target)
            own_ic = corr(g.prediction,g.target,True) if individual.sum()>=126 and individual.mean()>=.5 else np.nan
            rows.append({'fold_id': fold, 'pipeline': pipeline, 'outer_rank_ic': rank_ic,
                'outer_pearson_ic': pearson, 'paired_count': int(mask.sum()), 'paired_coverage': float(mask.mean()),
                'structurally_available_all_folds': pipeline in active,
                'individual_rank_ic': own_ic, 'individual_pairs': int(individual.sum())})
            pooled.setdefault(pipeline, []).append((x,y,np.full(len(g),fold)))
    folds = pd.DataFrame(rows)
    summaries = []
    for pipeline, g in folds.groupby('pipeline'):
        ic = g.outer_rank_ic.to_numpy(float)
        finite = ic[np.isfinite(ic)]
        x,y,labels = [np.concatenate([p[k] for p in pooled[pipeline]]) for k in range(3)]
        rx,ry = blockwise_rank_pairs(x,y,labels)
        complete = len(finite)==len(ic)
        summaries.append({'pipeline': pipeline, 'finite_folds': len(finite), 'expected_folds': len(ic),
            'blockwise_pooled_rank_ic': corr(rx,ry), 'mean_fold_ic': float(np.mean(finite)) if complete else np.nan,
            'median_fold_ic': float(np.median(finite)) if complete else np.nan,
            'ic_std': float(np.std(finite,ddof=1)) if complete and len(finite)>1 else np.nan,
            'positive_fold_ratio': float((ic>0).sum()/len(ic)) if complete else np.nan,
            'worst_fold_ic': float(min(finite)) if len(finite) else np.nan,
            'best_fold_ic': float(max(finite)) if len(finite) else np.nan})
    return folds, pd.DataFrame(summaries)


def build_phase4_tables(history: pd.DataFrame, config: dict, design) -> Phase4Tables:
    validate_research_history(history, design)
    folds = make_folds(np.arange(design.validation.development_end+1), design.validation)
    meta, selections, registry, rejections, predictions = [], [], [], [], []
    transforms = {'rules':{'transformations':config['transformations'],'availability':config['availability'],
        'selection':config['selection'],'regime':config['regime'],'family_composites':config['family_composites'],
        'maximum_source_offset':0,'lagged_outcome_source_offset':-1,'candidate_maximum':499}}
    for pipeline in ('A','B'):
        provider = CandidateProvider(history, config, pipeline)
        selector = NestedSelector(provider)
        registry.append(provider.registry())
        for fold in folds:
            end, start, stop = fold.train_ids[-1], fold.validation_ids[0], fold.validation_ids[-1]+1
            state = selector.fit(end)
            score = selector.forecast(end, start, stop)
            bounds, sources = selector.inner_oof_bounds(end)
            a = state.candidates
            meta.append({'fold_id': fold.number, 'pipeline': pipeline, 'train_start': 0, 'train_end': end,
                'purge_id': start-1, 'validation_start': start, 'validation_end': stop-1,
                'candidate_count': len(provider.ids), 'eligible_candidate_count': int(a.eligible_all_references.sum()),
                'same_sign_candidate_count': int((a.eligible_all_references&a.same_sign).sum()),
                'selected_count': len(state.selected), 'selected_features': json.dumps(state.selected),
                'selected_transformations': json.dumps([c.split('/')[-1] for c in state.selected]),
                'selected_signs': json.dumps(state.signs,sort_keys=True),
                'clip_lower': bounds[0], 'clip_upper': bounds[1]})
            selection = a.copy(); selection['fold_id']=fold.number; selection['pipeline']=pipeline
            selection['sign'] = selection.candidate_id.map(state.signs)
            selection['weight'] = np.where(selection.selected, 1/len(state.selected) if state.selected else 0, 0)
            selections.append(selection)
            rejections.extend([{'fold_id':fold.number,'pipeline':pipeline,**r} for r in state.redundancy])
            predictions.append(pd.DataFrame({'date_id':np.arange(start,stop),'fold_id':fold.number,'pipeline':pipeline,
                'prediction':score,'target':provider.y.iloc[start:stop].to_numpy(float),
                'status':'active' if len(state.selected)>=2 else 'no_alpha', 'clip_lower':bounds[0],'clip_upper':bounds[1]}))
            transforms[f'{pipeline}/{fold.number}'] = {'selector':state.audit(),'pearson_inner_oof':sources,
                'pearson_bounds':list(bounds),'candidate_context':provider.context(end)[1]}
        transforms[f'{pipeline}/all_reference_contexts'] = {str(k):v[1] for k,v in provider.contexts.items()}
    preds = pd.concat(predictions,ignore_index=True)
    metrics, summary = evaluate(preds)
    result = pd.DataFrame(meta).merge(metrics,on=['fold_id','pipeline'],validate='one_to_one')
    summary = summary.merge(result.groupby('pipeline').selected_count.agg(['mean','min','max']).rename(
        columns={'mean':'mean_selected_count','min':'min_selected_count','max':'max_selected_count'}),on='pipeline')
    return Phase4Tables(result,pd.concat(selections,ignore_index=True),pd.concat(registry,ignore_index=True),
        pd.DataFrame(rejections,columns=['fold_id','pipeline','candidate_id','rejected_with','rho','pairs','reason']),
        transforms,summary,preds)
