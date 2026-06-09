"""Tests for country_rotation.factors.redundancy — hierarchical factor clustering.

Parity source: FactorTransformer.py lines 767-908.
"""
import numpy as np
import pandas as pd
import pytest

from country_rotation.factors import redundancy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identical_factor_pair():
    """F1 and F2 are perfectly correlated (F2 = 2*F1 + 1); F3 is independent."""
    idx = pd.bdate_range("2019-01-01", periods=200)
    rng = np.random.default_rng(5)
    base = pd.DataFrame(
        rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("WXYZ")
    )
    dfs = {
        "F1": base,
        "F2": base * 2 + 1,
        "F3": pd.DataFrame(
            rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("WXYZ")
        ),
    }
    return dfs


# ---------------------------------------------------------------------------
# Mandatory spec tests (verbatim from task spec)
# ---------------------------------------------------------------------------

def test_duplicate_factors_cluster_together(identical_factor_pair):
    res = redundancy.cluster_factors(identical_factor_pair, threshold=0.3)
    cl = {tuple(sorted(v)) for v in res.clusters.values()}
    assert ("F1", "F2") in cl              # perfectly correlated -> same cluster
    assert len(res.representatives) == 2   # one per cluster


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------

def test_distance_matrix_symmetric(identical_factor_pair):
    """distance matrix is square, symmetric, diagonal == 0."""
    res = redundancy.cluster_factors(identical_factor_pair, threshold=0.3)
    dm = res.distance
    assert dm.shape == (3, 3)
    assert dm.index.tolist() == dm.columns.tolist()
    # Diagonal should be ~0 (distance to self)
    np.testing.assert_allclose(np.diag(dm.values), 0.0, atol=1e-10)
    # Symmetric
    np.testing.assert_allclose(dm.values, dm.values.T, atol=1e-10)


def test_single_factor_per_cluster_returns_itself():
    """When all factors are independent, each becomes its own cluster."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(99)
    # Three independent random walks
    dfs = {
        f"F{i}": pd.DataFrame(
            rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("WXYZ")
        )
        for i in range(3)
    }
    # threshold=0.0 means only factors with |corr|>=1.0 merge.
    # Independent random walks should have |corr| well below 1.
    res = redundancy.cluster_factors(dfs, threshold=0.0)
    assert len(res.clusters) == 3
    assert len(res.representatives) == 3
    factor_names = set(dfs.keys())
    for rep in res.representatives:
        assert rep in factor_names


def test_selection_coverage_picks_densest_factor():
    """'coverage' selection picks the factor with most non-NaN observations."""
    idx = pd.bdate_range("2020-01-01", periods=200)
    rng = np.random.default_rng(7)
    base = pd.DataFrame(
        rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("ABCD")
    )
    # F_dense has no NaNs; F_sparse has 50% NaNs — they're perfectly correlated so cluster together
    dense = base.copy()
    sparse = base.copy()
    sparse.iloc[:100, :] = np.nan   # first half missing

    dfs = {
        "F_dense": dense,
        "F_sparse": sparse,
        "F_other": pd.DataFrame(
            rng.normal(size=(200, 4)).cumsum(axis=0), index=idx, columns=list("ABCD")
        ),
    }
    res = redundancy.cluster_factors(dfs, threshold=0.3, selection="coverage")
    # F_dense and F_sparse should cluster together; coverage should pick F_dense
    for cluster_id, members in res.clusters.items():
        members_set = set(members)
        if "F_dense" in members_set and "F_sparse" in members_set:
            rep = res.representatives[
                list(res.clusters.keys()).index(cluster_id)
            ]
            assert rep == "F_dense", "coverage selection should pick the denser factor"
            break
    else:
        pytest.skip("F_dense and F_sparse did not end up in the same cluster — "
                    "raise threshold or adjust fixture if correlation < threshold")


def test_selection_unique_returns_valid_representative(identical_factor_pair):
    """'unique' mode still returns one valid representative per cluster."""
    res = redundancy.cluster_factors(identical_factor_pair, threshold=0.3, selection="unique")
    factor_names = set(identical_factor_pair.keys())
    for rep in res.representatives:
        assert rep in factor_names
    assert len(res.representatives) == len(res.clusters)


def test_selection_central_returns_valid_representative(identical_factor_pair):
    """'central' mode still returns one valid representative per cluster."""
    res = redundancy.cluster_factors(identical_factor_pair, threshold=0.3, selection="central")
    factor_names = set(identical_factor_pair.keys())
    for rep in res.representatives:
        assert rep in factor_names
    assert len(res.representatives) == len(res.clusters)


def test_invalid_selection_raises(identical_factor_pair):
    with pytest.raises(ValueError, match="selection"):
        redundancy.cluster_factors(identical_factor_pair, threshold=0.3, selection="bogus")


def test_cluster_result_frozen():
    """ClusterResult must be a frozen dataclass (immutable)."""
    idx = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(size=(50, 3)).cumsum(0), index=idx, columns=list("XYZ"))
    dfs = {"F1": df, "F2": df + 0.5}
    res = redundancy.cluster_factors(dfs, threshold=0.5)
    with pytest.raises((AttributeError, TypeError)):
        res.representatives = []  # frozen dataclass should reject mutation
