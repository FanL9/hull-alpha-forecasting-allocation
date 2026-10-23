"""Post-hoc winner's-curse check of A's nested selection, from committed reports only.

For every outer fold, compares the oriented inner score of the 16 selected raw
levels (|mean of two inner block Rank ICs|) with the same features' Phase 3
outer-fold Rank IC under the selected sign, against eligible same-sign
features that were not selected. Descriptive only; no data file is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.paths import project_root

SELECTION = 'reports/phase4_real_run/phase4_raw_control_selection.csv'
FOLD_IC = 'reports/phase3_real_run/fold_level_ic.csv'
OUTPUT = 'reports/phase6_posthoc/selection_shrinkage.csv'


def shrinkage(selection: pd.DataFrame, fold_ic: pd.DataFrame) -> pd.DataFrame:
    """Per fold and group (selected / not selected): mean oriented inner and outer IC."""
    s = selection.loc[selection['pipeline'].eq('A') & selection['eligible_all_references']
                      & selection['same_sign']].copy()
    s['feature'] = s['candidate_id'].str.split('/').str[1]
    f = fold_ic.loc[fold_ic['scope'].eq('single_features'), ['hypothesis', 'fold', 'fold_ic']]
    m = s.merge(f, left_on=['feature', 'fold_id'], right_on=['hypothesis', 'fold'], how='left', validate='1:1')
    if m['fold_ic'].isna().any():
        raise ValueError('Every eligible candidate needs a Phase 3 outer-fold IC.')
    sign = np.sign(m['inner_mean_ic'])
    m['inner_oriented'], m['outer_oriented'] = m['inner_mean_ic'].abs(), m['fold_ic']*sign
    m['group'] = np.where(m['selected'], 'selected', 'not_selected')
    rows = []
    for (fold, group), d in [*m.groupby(['fold_id', 'group']), *((('all', g), d) for g, d in m.groupby('group'))]:
        rows.append({'fold': fold, 'group': group, 'features': len(d), 'inner_oriented_ic': d['inner_oriented'].mean(),
                     'outer_oriented_ic': d['outer_oriented'].mean(),
                     'outer_positive_share': (d['outer_oriented'] > 0).mean(),
                     'retained_share': d['outer_oriented'].mean()/d['inner_oriented'].mean()})
    return pd.DataFrame(rows)


def main() -> int:
    root = project_root()
    table = shrinkage(pd.read_csv(root/SELECTION), pd.read_csv(root/FOLD_IC))
    (root/OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(root/OUTPUT, index=False, float_format='%.6f', lineterminator='\n')
    print(table.loc[table['fold'].eq('all')].to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
