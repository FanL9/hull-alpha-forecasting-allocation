"""Explicit standalone Phase 4 CLI. No action runs by default.

python -m src.research.phase4 --check                 # metadata only
python -m src.research.phase4 --synthetic --output ...
python -m src.research.phase4 --calibrate --output ... --workspace ...
Real execution requires separate human authorization, the frozen code tag, a
clean worktree and an unused canonical reservation; never run it in Phase 4A.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.utils.paths import project_root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('check','synthetic','calibrate','run-real-phase4'):
        group.add_argument('--'+mode,action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--workspace',type=Path)
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--case',choices=['noise','raw','transformed','regime'],default='noise')
    parser.add_argument('--workers',type=int,default=3)
    args = parser.parse_args(argv)
    root = project_root()
    if args.check:
        from src.research.phase4_execution import preflight
        print(json.dumps(preflight(root),indent=2)); return 0
    if args.run_real_phase4:
        from src.research.phase4_execution import run_real
        run_real(root,args.output); return 0
    if args.output is None:
        parser.error('Synthetic/calibration mode requires an explicit separate --output directory.')
    destination = args.output.resolve()
    if destination.is_relative_to((root/'artifacts').resolve()) or destination.is_relative_to((root/'data').resolve()):
        parser.error('Synthetic output must stay outside canonical artifacts and data.')
    from src.research.phase4_calibration import run_calibration, synthetic_run
    if args.calibrate:
        if args.workspace is None or not 1<=args.workers<=3:
            parser.error('Calibration requires --workspace and 1..3 workers.')
        workspace = args.workspace.resolve()
        if workspace.is_relative_to((root/'data').resolve()) or workspace.is_relative_to((root/'artifacts').resolve()):
            parser.error('Calibration workspace must be separate from data/artifacts.')
        run_calibration(root,workspace,destination,args.workers)
    else:
        from src.research.phase4_synthetic import make_history
        synthetic_run(root,make_history(root,args.seed,args.case),destination)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
