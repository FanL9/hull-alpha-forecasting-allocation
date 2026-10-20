"""Nested v3.1 A/B/Ridge/GRU/TCN/C/D experiment on an explicit prefix.

Fitting capsules have no subsequent block labels. Inner OOF forecasts are
complete earlier fits, including their OWN representative and stopping split.
No file loader or holdout evaluator lives here.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from itertools import product

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.features.deep_inputs import InputState, fit_inputs, windows, training_arrays
from src.features.engineered_alpha import CandidateProvider
from src.models.deep_alpha import neural_task, ridge_solutions, ridge_predict
from src.research.alpha_pipeline import validate_research_history
from src.research.phase4_engine import NestedSelector, corr, evaluate
from src.research.phase5_tasks import TaskStore, hash_object
from src.validation.protocol_v3 import inner_splits, stopping_split
from src.validation.walk_forward import make_folds

COMPONENTS = ('A','B','Ridge','GRU','TCN','C','D')


def primary_selection(folds):
    """Pure frozen rule; selecting here never unlocks any held-out data."""
    candidates = []
    for name in ('A','B','C','D'):
        ic = folds.loc[folds.pipeline==name,'outer_rank_ic'].to_numpy(float)
        if len(ic)==11 and np.isfinite(ic).all() and (ic>0).sum()>=7 and ic.mean()>0:
            candidates.append({'pipeline':name,'mean':float(ic.mean()),'std':float(ic.std(ddof=1)),
                               'worst':float(ic.min())})
    if not candidates:
        return {'selected':None,'status':'no_admissible_primary','holdout_unlocked':False}
    anchor = max(r['mean'] for r in candidates)
    near = [r for r in candidates if anchor-r['mean']<.01]
    winner = min(near,key=lambda r:(('A','B','C','D').index(r['pipeline']),r['std'],-r['worst'],r['pipeline']))
    return {'selected':winner['pipeline'],'anchor_mean':anchor,'admissible':candidates,
            'near_best':[r['pipeline'] for r in near],'holdout_unlocked':False,'status':'development_selection_only'}


def normalization(inner):
    """Common forecast rows only; no target argument by construction."""
    a = np.column_stack(inner)
    mask = np.isfinite(a).all(axis=1)
    if mask.sum()<252:
        return None
    means,scales = a[mask].mean(axis=0),a[mask].std(axis=0,ddof=0)
    if not np.isfinite(scales).all() or np.any(scales<=1e-12):
        return None
    return {'means':means,'scales':scales,'pairs':int(mask.sum()),'mask':mask}


def combined(inner,outer):
    state = normalization(inner)
    if state is None:
        return np.full(len(outer[0]),np.nan),None
    a = np.column_stack(outer)
    return ((a-state['means'])/state['scales']).mean(axis=1),state


def residualize(inner,outer):
    state = normalization(inner)
    if state is None:
        return np.full(len(outer[0]),np.nan),None
    a = (np.column_stack(inner)[state['mask']]-state['means'])/state['scales']
    regressors = np.column_stack([np.ones(len(a)),a[:,:2]])
    beta = np.linalg.pinv(regressors,rcond=1e-12)@a[:,2]
    b = (np.column_stack(outer)-state['means'])/state['scales']
    state['beta']=beta
    return b[:,2]-np.column_stack([np.ones(len(b)),b[:,:2]])@beta,state


def bounds(values):
    a = np.asarray(values,float); a=a[np.isfinite(a)]
    return tuple(np.quantile(a,[.01,.99])) if len(a) else (np.nan,np.nan)


class Engine:
    def __init__(self,history,config,design,store: TaskStore,test_budget=None):
        validate_research_history(history,design)
        if test_budget not in (None,1,2) or (test_budget is not None and store.identity['run_type']!='synthetic'):
            raise ValueError('Small epoch budget is restricted to explicit synthetic software tests.')
        self.history,self.config,self.design,self.store = history,config,design,store
        self.test_budget = test_budget
        self.raw = CandidateProvider(history,config,'A').raw
        self.y = history[config['target']].to_numpy(float)
        self.states,self.scores,self.neural,self.ridges = {},{},{},{}
        self.folds = make_folds(np.arange(design.validation.development_end+1),design.validation)
        self.outer = tuple(int(f.train_ids[-1]) for f in self.folds)
        first = {s.training_end for end in self.outer for s in inner_splits(end,config)}
        second = {s.training_end for end in first for s in inner_splits(end,config)}
        self.references = tuple(sorted(set(self.outer)|first|second))

    def input_state(self,end):
        if end not in self.states:
            task = {'kind':'inputs','reference_end':end,'model':'shared_Ridge_GRU_TCN',
                    'training_range':[0,end],'validation_range':None,'seed':None}
            saved = self.store.execute(task,lambda:asdict(fit_inputs(self.history,end,self.config)))
            saved['names']=tuple(saved['names']); saved['clusters']=tuple(map(tuple,saved['clusters']))
            self.states[end]=InputState(**saved)
        return self.states[end]

    def ab(self,name,end):
        key = name,end
        if key not in self.scores:
            start,stop=end+2,end+254
            task = {'kind':'selector_forecast','model':name,'reference_end':end,'seed':None,
                    'training_range':[0,end],'validation_range':[start,stop-1]}
            def compute():
                # Current/past lagged outcomes remain legal X. Subsequent y is
                # physically removed from the fitted selector's label series.
                provider = CandidateProvider(self.history.iloc[:stop].copy(),self.config,name)
                provider.y = provider.y.iloc[:end+1].copy()
                selector = NestedSelector(provider)
                state = selector.fit(end)
                return {'prediction':selector.forecast(end,start,stop),'selector':state.audit(),
                        'status':'active' if len(state.selected)>=2 else 'no_alpha'}
            self.scores[key] = self.store.execute(task,compute)
        return self.scores[key]['prediction']

    def job(self,end,name,seed):
        split = stopping_split(end,self.config)
        early,full=self.input_state(split.training_end),self.input_state(end)
        a = training_arrays(self.raw,self.y,early)
        b = training_arrays(self.raw,self.y,full)
        dates = np.arange(split.validation_start,split.validation_end+1)
        dates = dates[np.isfinite(self.y[dates])]
        valid = a is not None and b is not None and len(dates)>0
        consumers=[]
        for f in self.folds:
            outer_end=int(f.train_ids[-1])
            if end==outer_end:
                consumers.append({'outer_fold':f.number,'scenario':'outer'})
            for j,s in enumerate(inner_splits(outer_end,self.config),1):
                if end==s.training_end:
                    consumers.append({'outer_fold':f.number,'scenario':'inner_oof_'+str(j)})
                for k,t in enumerate(inner_splits(s.training_end,self.config),1):
                    if end==t.training_end:
                        consumers.append({'outer_fold':f.number,'scenario':f'D_inner_{j}_normalization_{k}'})
        return {'directory':str(self.store.directory),'identity':self.store.identity,
                'task':{'kind':'neural','reference_end':end,'model':name,'seed':seed,'outer_consumers':consumers,
                        'model_config_hash':hash_object({'model':self.config['models'][name.lower()],
                            'training':self.config['training'],'deep':self.config['deep']})},
                'stop_x':a[0] if valid else None,'stop_y':a[1] if valid else None,
                'validation_x':windows(self.raw,early,dates) if valid else None,
                'validation_y':(self.y[dates]-early.target_mean)/early.target_scale if valid else None,
                'refit_x':b[0] if valid else None,'refit_y':b[1] if valid else None,
                'forecast_x':windows(self.raw,full,np.arange(end+2,end+254)),
                'stop_range':[0,split.training_end],'stopping_range':[split.validation_start,split.validation_end],
                'refit_range':[0,end],'forecast_range':[end+2,end+253],
                'stop_feature_hash':hash_object(early.audit()),'refit_feature_hash':hash_object(full.audit()),
                'target_mean':full.target_mean,'target_scale':full.target_scale,'test_budget':self.test_budget}

    def run_neural(self,workers):
        if not 1<=workers<=8:
            raise ValueError('Engineering worker budget must be 1..8; each worker uses one thread.')
        specs = iter(product(self.references,('GRU','TCN'),self.config['deep']['seeds']))
        timings = []
        def collect(row):
            self.neural[row['reference_end'],row['model'],row['seed']]=row['result']
            timings.append({k:v for k,v in row.items() if k!='result'})
        if workers==1:
            for end,name,seed in specs:
                collect(neural_task(self.job(end,name,seed)))
        else:
            # At most workers materialized sequence capsules at any moment.
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending = set()
                exhausted = False
                while pending or not exhausted:
                    while len(pending)<workers and not exhausted:
                        try:
                            end,name,seed=next(specs)
                            pending.add(pool.submit(neural_task,self.job(end,name,seed)))
                        except StopIteration:
                            exhausted=True
                    if pending:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED)
                        for f in done:
                            collect(f.result())
        return sorted(timings,key=lambda r:(r['reference_end'],r['model'],r['seed']))

    def deep(self,name,end):
        if name=='C':
            return .5*self.deep('GRU',end)+.5*self.deep('TCN',end)
        rows=[self.neural[end,name,seed] for seed in self.config['deep']['seeds']]
        if any(row['status']!='active' for row in rows):
            return np.full(252,np.nan)
        return np.mean([row['prediction'] for row in rows],axis=0)

    def fixed_ridge(self,end):
        task = {'kind':'ridge_svd','reference_end':end,'model':'Ridge','seed':None,
                'training_range':[0,end],'validation_range':[end+2,end+253],
                'input_feature_hash':hash_object(self.input_state(end).audit())}
        def compute():
            state=self.input_state(end)
            arrays=training_arrays(self.raw,self.y,state)
            alphas=self.config['models']['ridge']['alphas']
            if arrays is None:
                return {str(a):{'prediction':np.full(252,np.nan),'status':'unavailable'} for a in alphas}
            x,y,_=arrays
            fits=ridge_solutions(x,y,alphas)
            forecast=windows(self.raw,state,np.arange(end+2,end+254))
            return {str(a):{'prediction':ridge_predict(fits[a],forecast)*state.target_scale+state.target_mean,
                            'state':fits[a],'status':'active'} for a in alphas}
        return self.store.execute(task,compute)

    def ridge(self,end):
        if end not in self.ridges:
            settings=[]
            splits=inner_splits(end,self.config)
            # True earlier fits, not a final-reference fitted design matrix.
            for alpha in self.config['models']['ridge']['alphas']:
                ics=[corr(self.fixed_ridge(s.training_end)[str(alpha)]['prediction'],
                          self.y[s.validation_start:s.validation_end+1],True) for s in splits]
                settings.append((alpha,float(np.mean(ics)) if np.isfinite(ics).all() else np.nan))
            valid=[r for r in settings if np.isfinite(r[1])]
            if not valid:
                self.ridges[end]={'prediction':np.full(252,np.nan),'alpha':None,'inner_ics':settings}
            else:
                best=max(r[1] for r in valid)
                alpha=max(r[0] for r in valid if best-r[1]<=1e-12)
                self.ridges[end]={'prediction':self.fixed_ridge(end)[str(alpha)]['prediction'],
                                  'alpha':alpha,'inner_ics':settings}
        return self.ridges[end]['prediction']

    def inner(self,name,end):
        return np.concatenate([self.forecast(name,s.training_end) for s in inner_splits(end,self.config)])

    def forecast(self,name,end):
        if name in ('A','B'):
            return self.ab(name,end)
        if name in ('GRU','TCN','C'):
            return self.deep(name,end)
        if name=='Ridge':
            return self.ridge(end)
        if name=='D':
            return combined([self.inner(n,end) for n in ('A','B','C')],
                            [self.forecast(n,end) for n in ('A','B','C')])[0]
        raise ValueError('Unknown frozen model.')

    def predictions(self):
        """Fix all fold forecasts and derived maps before evaluation consumes y."""
        rows,audit=[],{}
        for fold,end in zip(self.folds,self.outer):
            outer={n:self.forecast(n,end) for n in COMPONENTS}
            inner={n:self.inner(n,end) for n in COMPONENTS}
            _,dstate=combined([inner[n] for n in ('A','B','C')],[outer[n] for n in ('A','B','C')])
            detail={'reference_end':end,'inputs':self.input_state(end).audit(),'D_normalization':dstate,
                    'ridge':{k:v for k,v in self.ridges[end].items() if k!='prediction'},
                    'A_selector':self.scores['A',end]['selector'],'B_selector':self.scores['B',end]['selector'],
                    'inner_prediction_ranges':[asdict(s) for s in inner_splits(end,self.config)]}
            for name in ('GRU','TCN'):
                residual,state=residualize([inner[n] for n in ('A','B',name)],[outer[n] for n in ('A','B',name)])
                outer['residual_'+name]=residual
                detail['residual_'+name]=state
            dates=np.asarray(fold.validation_ids)
            for name,pred in outer.items():
                lo,hi=bounds(inner[name]) if name in inner else (np.nan,np.nan)
                rows.append(pd.DataFrame({'date_id':dates,'fold_id':fold.number,'pipeline':name,'prediction':pred,
                    'clip_lower':lo,'clip_upper':hi,'status':'active' if np.isfinite(pred).sum()>=126 and np.unique(pred[np.isfinite(pred)]).size>1 else 'unavailable'}))
            for name in ('historical_mean','zero_forecast'):
                value=float(np.nanmean(self.y[:end+1])) if name=='historical_mean' else 0.
                rows.append(pd.DataFrame({'date_id':dates,'fold_id':fold.number,'pipeline':name,
                    'prediction':value,'clip_lower':value,'clip_upper':value,'status':'constant_secondary'}))
            audit[str(fold.number)]=detail
        return pd.concat(rows,ignore_index=True),audit

    def evaluate(self,predictions):
        forecasts=predictions.copy()
        # The only path attaching each fold's target for metrics. Fitting and
        # prediction capsules above never accept that same fold's target block.
        forecasts['target']=self.y[forecasts.date_id.to_numpy(int)]
        primary,summary=evaluate(forecasts[forecasts.pipeline.isin(('A','B','C','D'))])
        secondary,secondary_summary=evaluate(forecasts[forecasts.pipeline.isin(('Ridge','GRU','TCN'))])
        constants,constant_summary=evaluate(forecasts[forecasts.pipeline.isin(('historical_mean','zero_forecast'))])
        diagnostics=[]
        for (fold,name),g in forecasts.groupby(['fold_id','pipeline']):
            if name.startswith('residual_'):
                deep=forecasts[(forecasts.fold_id==fold)&(forecasts.pipeline==name.removeprefix('residual_'))]
                mask=np.isfinite(g.prediction.to_numpy())&np.isfinite(deep.prediction.to_numpy())&np.isfinite(g.target.to_numpy())
                diagnostics.append({'fold_id':int(fold),'pipeline':name,'pairs':int(mask.sum()),
                    'deep_ic':corr(deep.prediction.to_numpy()[mask],g.target.to_numpy()[mask],True) if mask.sum()>=126 else np.nan,
                    'residual_ic':corr(g.prediction.to_numpy()[mask],g.target.to_numpy()[mask],True) if mask.sum()>=126 else np.nan})
        differences=[]
        for table,pairs in ((primary,[('B','A'),('C','A'),('D','A')]),
                            (secondary,[('GRU','Ridge'),('TCN','Ridge')])):
            pivot=table.pivot(index='fold_id',columns='pipeline',values='outer_rank_ic')
            for lhs,rhs in pairs:
                delta=pivot[lhs]-pivot[rhs]
                differences.append({'comparison':lhs+' - '+rhs,'fold_differences':delta.to_list(),
                                    'mean_difference':float(delta.mean()) if delta.notna().all() else np.nan,
                                    'folds_won':int((delta>0).sum()),'finite_folds':int(delta.notna().sum())})
        for table in (summary,secondary_summary,constant_summary):
            table['positive_fold_count']=[int((t.outer_rank_ic>0).sum()) for n in table.pipeline
                for t in [pd.concat([primary,secondary,constants]).query('pipeline == @n')]]
        return {'predictions':forecasts,'folds':pd.concat([primary,secondary,constants],ignore_index=True),
                'summary':pd.concat([summary,secondary_summary,constant_summary],ignore_index=True),
                'incremental':pd.DataFrame(diagnostics),'differences':differences,'primary_selection':primary_selection(primary)}
