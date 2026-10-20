"""Standalone Phase 5 CLI; no action runs by default, no holdout mode exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    for mode in ('check','synthetic','calibrate','run-real-phase5'):
        g.add_argument('--'+mode,action='store_true')
    p.add_argument('--output',type=Path)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--case',choices=['noise','linear','nonlinear'],default='noise')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--synthetic-test-epochs',type=int,choices=[1,2],help='Software E2E only, never real or calibration')
    a=p.parse_args(argv)
    if not 1<=a.workers<=8:
        p.error('workers must be 1..8')
    if a.synthetic_test_epochs is not None and not a.synthetic:
        p.error('Test epoch budget requires explicit --synthetic mode.')
    from src.research import phase5_execution as execution
    if a.check:
        if a.output or a.data_root:
            p.error('Read-only check uses canonical paths.')
        print(json.dumps(execution.preflight(),indent=2)); return 0
    if a.run_real_phase5:
        if a.data_root is not None:
            p.error('Real entry requires canonical source.')
        execution.run_real(a.output,a.workers); return 0
    if a.output is None:
        p.error('Synthetic/calibration requires a separate --output.')
    a.output=execution.synthetic_destination(a.output)
    if a.data_root is not None:
        a.data_root=execution.synthetic_destination(a.data_root)
    if a.calibrate:
        if a.data_root is not None:
            p.error('Calibration generates its preregistered artificial sources.')
        from src.research.phase5_calibration import run
        run(a.output,a.workers); return 0
    from src.research.phase5_synthetic import synthetic_project
    data=a.data_root or a.output.with_name(a.output.name+'_source')
    if a.data_root is None:
        synthetic_project(data,a.case,a.seed)
    execution.run_synthetic(data,a.output,a.workers,a.synthetic_test_epochs)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
