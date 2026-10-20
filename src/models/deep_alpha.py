"""Exact v3.1 CPU architectures and deterministic chronological training."""
from __future__ import annotations

from dataclasses import dataclass
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

from src.research.phase5_tasks import TaskStore, hash_object


def deterministic(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            raise RuntimeError('Phase 5 requires one interop thread before any parallel work.') from exc
    torch.use_deterministic_algorithms(True)


class GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(16,8,num_layers=1,batch_first=True,dropout=0,bias=True)
        self.head = nn.Linear(8,1)

    def forward(self,x):
        _,h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class CausalBlock(nn.Module):
    def __init__(self, inputs, dilation):
        super().__init__()
        self.dilation = dilation
        self.first = nn.Conv1d(inputs,8,3,dilation=dilation)
        self.second = nn.Conv1d(8,8,3,dilation=dilation)
        self.projection = nn.Conv1d(inputs,8,1) if inputs!=8 else nn.Identity()

    def forward(self,x):
        h = F.relu(self.first(F.pad(x,(2*self.dilation,0))))
        h = self.second(F.pad(h,(2*self.dilation,0)))
        return F.relu(h+self.projection(x))


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(CausalBlock(16,1),CausalBlock(8,2))
        self.head = nn.Linear(8,1)

    def forward(self,x):
        return self.head(self.blocks(x.transpose(1,2)).mean(dim=2)).squeeze(-1)


def model(name):
    net = {'GRU':GRU,'TCN':TCN}[name]()
    expected = {'GRU':633,'TCN':1137}[name]
    assert sum(p.numel() for p in net.parameters())==expected < 10000
    return net


@dataclass
class Stopper:
    minimum: int = 10
    maximum: int = 100
    patience: int = 10
    delta: float = .0001
    best: float = np.inf
    epoch: int = 0
    stale: int = 0

    def update(self, epoch, loss):
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite stopping loss.')
        if epoch>=self.minimum:
            if self.best-loss>self.delta:
                self.best,self.epoch,self.stale = loss,epoch,0
            else:
                self.stale += 1
        return epoch>=self.maximum or self.stale>=self.patience


def predict(net,x):
    net.eval()
    with torch.no_grad():
        return torch.cat([net(x[i:i+64]) for i in range(0,len(x),64)]).numpy().astype(np.float64)


def train(name, seed, x, y, epochs, *, stopping=None, test_budget=None):
    """Fresh initialization for EACH stage, including the full-reference refit."""
    deterministic(seed)
    net = model(name)
    opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    x = torch.from_numpy(np.asarray(x,np.float32).copy())
    y = torch.from_numpy(np.asarray(y,np.float32).copy())
    stopper = Stopper() if test_budget is None else Stopper(minimum=1,maximum=test_budget)
    trace = []
    if stopping is not None:
        sx,sy = stopping
        sx = torch.from_numpy(np.asarray(sx,np.float32).copy())
        sy = np.asarray(sy,np.float32)
    for epoch in range(1,epochs+1):
        net.train()
        for i in range(0,len(x),64):
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(x[i:i+64]), y[i:i+64])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss.')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
            opt.step()
        if stopping is not None:
            mse = float(np.mean((predict(net,sx).astype(np.float32)-sy)**2,dtype=np.float32))
            trace.append(mse)
            if stopper.update(epoch,mse):
                break
    return net,stopper.epoch if stopping is not None else epochs,trace


def neural_task(job):
    """Independent seed fit: immutable stop stage, then fresh exact-epoch refit."""
    from src.research.alpha_calibration import artificial_data_only
    from src.utils.paths import project_root
    from contextlib import nullcontext
    start = time.perf_counter()
    store = TaskStore(job['directory'],job['identity'])
    task = job['task']
    context = artificial_data_only(project_root()) if job['identity']['run_type']=='synthetic' else nullcontext()
    with context, threadpool_limits(limits=1):
        def stopping():
            if job['stop_x'] is None or job['refit_x'] is None:
                return {'status':'unavailable','selected_epochs':None,'reason':'input_or_target_unavailable'}
            try:
                _,epochs,trace = train(task['model'],task['seed'],job['stop_x'],job['stop_y'],
                    job['test_budget'] or 100,stopping=(job['validation_x'],job['validation_y']),test_budget=job['test_budget'])
                return {'status':'active','selected_epochs':epochs,'stopping_mse':np.asarray(trace)}
            except FloatingPointError:
                return {'status':'unavailable','selected_epochs':None,'reason':'nonfinite_loss'}
        stop = store.execute({**task,'stage':'stopping','training_range':job['stop_range'],
            'validation_range':job['stopping_range'],'input_feature_hash':job['stop_feature_hash']},stopping)
        refit_task = {**task,'stage':'refit','training_range':job['refit_range'],
            'validation_range':job['forecast_range'],'input_feature_hash':job['refit_feature_hash'],
            'selected_epochs':stop['selected_epochs'],'stopping_output_hash':hash_object(stop)}
        def refitting():
            if stop['status']!='active':
                return {'status':'unavailable','selected_epochs':None,'prediction':np.full(252,np.nan)}
            try:
                net,epochs,_ = train(task['model'],task['seed'],job['refit_x'],job['refit_y'],stop['selected_epochs'])
                pred = predict(net,torch.from_numpy(np.asarray(job['forecast_x'],np.float32).copy()))
                pred = pred*job['target_scale']+job['target_mean']
                if not np.isfinite(pred).all():
                    raise FloatingPointError('Nonfinite refit forecast.')
                return {'status':'active','selected_epochs':epochs,'prediction':pred,
                        'weights':{k:v.detach().numpy() for k,v in net.state_dict().items()}}
            except FloatingPointError:
                return {'status':'unavailable','selected_epochs':stop['selected_epochs'],'prediction':np.full(252,np.nan)}
        result = store.execute(refit_task,refitting)
    return {'reference_end':task['reference_end'],'model':task['model'],'seed':task['seed'],
            'result':result,'seconds':time.perf_counter()-start}


def ridge_solutions(x,y,alphas):
    x = np.asarray(x,np.float64).reshape(len(x),-1)
    y = np.asarray(y,np.float64)
    xm,ym = x.mean(axis=0),float(y.mean())
    u,s,vt = np.linalg.svd(x-xm,full_matrices=False)
    uy = u.T@(y-ym)
    return {alpha:{'coef':vt.T@((s/(s*s+len(y)*alpha))*uy),'x_mean':xm,'y_mean':ym} for alpha in alphas}


def ridge_predict(state,x):
    x = np.asarray(x,np.float64).reshape(len(x),-1)
    return (x-state['x_mean'])@state['coef']+state['y_mean']
