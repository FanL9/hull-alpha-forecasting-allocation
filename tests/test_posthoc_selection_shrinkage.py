"""Winner's-curse table on artificial selection records."""
import pandas as pd
import pytest

from src.research.posthoc_selection_shrinkage import shrinkage


def test_orientation_and_grouping():
    sel = pd.DataFrame({'candidate_id': ['raw/X1/Level', 'raw/X2/Level', 'raw/X3/Level'], 'pipeline': 'A',
                        'eligible_all_references': [True, True, False], 'same_sign': True,
                        'inner_mean_ic': [-0.10, 0.04, 0.5], 'selected': [True, False, False], 'fold_id': 1})
    ic = pd.DataFrame({'hypothesis': ['X1', 'X2', 'X3'], 'scope': 'single_features', 'fold': 1,
                       'fold_ic': [-0.05, -0.02, 0.9]})
    t = shrinkage(sel, ic).set_index(['fold', 'group'])
    assert t.loc[(1, 'selected'), 'outer_oriented_ic'] == pytest.approx(0.05)
    assert t.loc[(1, 'selected'), 'retained_share'] == pytest.approx(0.5)
    assert t.loc[('all', 'not_selected'), 'outer_oriented_ic'] == pytest.approx(-0.02)
    assert t.loc[('all', 'not_selected'), 'features'] == 1  # ineligible X3 excluded
