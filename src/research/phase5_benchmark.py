"""Production-budget CPU timing on artificial 63x16 sequences only."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import argparse
import platform
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.models.deep_alpha import train
from src.research.alpha_calibration import artificial_data_only
from src.research.phase5_tasks import atomic
from src.utils.paths import project_root


def benchmark_task(spec):
    name,seed = spec
    with artificial_data_only(project_root()),threadpool_limits(limits=1):
        rng = np.random.default_rng(1701)
        # 5087-row first outer reference, 63 real observations per sequence.
        stream = rng.normal(size=(5087,16))
        x = stream[np.arange(62,5087)[:,None]+np.arange(-62,1)]
        y = stream[62:,0]+rng.normal(size=len(x))
        train_n = 4834-62
        sx,sy = x[-252:],y[-252:]
        start = time.perf_counter()
        _,epochs,trace = train(name,seed,x[:train_n],y[:train_n],100,stopping=(sx,sy))
        stopping_seconds = time.perf_counter()-start
        start = time.perf_counter()
        train(name,seed,x,y,epochs)
        refit_seconds = time.perf_counter()-start
        return {'model':name,'seed':seed,'stopping_epochs_run':len(trace),'selected_epochs':epochs,
                'stopping_seconds':stopping_seconds,'refit_seconds':refit_seconds,
                'training_sequences':train_n,'refit_sequences':len(x),'shape':[63,16]}


def run(output,workers=4):
    output = Path(output)
    if output.exists():
        raise ValueError('Benchmark output already exists.')
    specs = [('GRU',11),('TCN',11),('GRU',22),('TCN',22)]
    start = time.perf_counter()
    serial = [benchmark_task(spec) for spec in specs]
    serial_wall = time.perf_counter()-start
    atomic(output.with_suffix('.serial.json'),{'tasks':serial,'wall_seconds':serial_wall})
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parallel = list(pool.map(benchmark_task,specs))
    atomic(output,{'workers_1':serial,'workers_parallel':parallel,'parallel_workers':workers,
                  'serial_wall_seconds':serial_wall,'parallel_wall_seconds':time.perf_counter()-start,
                  'environment':{'python':platform.python_version(),'torch':torch.__version__},
                  'real_market_data_read':False})


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); run(a.output,a.workers)
