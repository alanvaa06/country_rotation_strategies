"""Factor redundancy clustering — port from FactorTransformer.py lines 767-908.

Computes pairwise factor correlations averaged across countries (min 10 observations
per country pair), converts to distance = 1 - |corr|, runs scipy hierarchical
linkage, and cuts the dendrogram at a distance threshold.  One representative per
cluster is selected according to one of three criteria: 'coverage', 'unique', or
'central'.

Parity notes vs legacy:
  - Legacy default linkage_method = 'ward' (line 769).  Task spec signature default
    is 'average' — the public API exposes 'average' as default but callers can pass
    'ward' to match legacy exactly.
  - Legacy operated on self.weighted_average_scores (dict of DataFrames, same shape
    per factor, columns = countries).  Here the API accepts factor_dfs with the same
    semantic: dict[factor_name -> DataFrame(dates x countries)].
  - 'coverage' criterion: non-NaN fraction over full df (notna sum / size), matching
    legacy lines 855-860.
  - 'unique' criterion: lowest mean |corr| to all factors in *other* clusters,
    matching legacy lines 861-875.
  - 'central' criterion: highest mean |corr| to own cluster members, matching legacy
    lines 876-884.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class ClusterResult:
    """Container for hierarchical clustering output.

    Attributes
    ----------
    clusters : dict[int, list[str]]
        Maps cluster_id -> list of factor names belonging to that cluster.
    representatives : list[str]
        One selected representative per cluster, in cluster_id order.
    distance : pd.DataFrame
        Factor × factor distance matrix (1 - |corr|), shape (n, n).
    """

    clusters: dict          # cluster_id -> list[str] factor names
    representatives: list   # one factor per cluster
    distance: pd.DataFrame  # factor x factor distance matrix


def cluster_factors(
    factor_dfs: dict[str, pd.DataFrame],
    threshold: float,
    linkage_method: str = "average",
    selection: str = "coverage",
) -> ClusterResult:
    """Cluster factors by pairwise correlation redundancy.

    Parameters
    ----------
    factor_dfs : dict[str, pd.DataFrame]
        Mapping of factor_name -> DataFrame(dates × countries).  Values must
        share the same column labels (country names).
    threshold : float
        Distance threshold for fcluster (criterion='distance').  Distance is
        1 - |corr|, so threshold=0.3 merges factors whose |corr| >= 0.7.
    linkage_method : str
        Linkage method passed to scipy.cluster.hierarchy.linkage.
        Legacy default was 'ward'; this API defaults to 'average'.
    selection : str
        How to pick the representative factor from each cluster:
          'coverage' — highest non-NaN data density (notna / size).
          'unique'   — lowest mean |corr| to factors in all other clusters.
          'central'  — highest mean |corr| to other members of own cluster.

    Returns
    -------
    ClusterResult
        Frozen dataclass with clusters, representatives, and distance matrix.

    Raises
    ------
    ValueError
        If selection is not one of 'coverage', 'unique', 'central'.
    """
    if selection not in ("coverage", "unique", "central"):
        raise ValueError(
            f"Invalid selection '{selection}'. Must be 'coverage', 'unique', or 'central'."
        )

    factor_names = list(factor_dfs.keys())
    n_factors = len(factor_names)

    # ------------------------------------------------------------------
    # Step 1: Build the n×n correlation matrix.
    # Per-country pairwise correlations, averaged across countries that
    # have >= 10 overlapping observations.  Matches legacy lines 808-829.
    # ------------------------------------------------------------------
    corr_matrix = pd.DataFrame(
        np.zeros((n_factors, n_factors)),
        index=factor_names,
        columns=factor_names,
    )

    for i, f1 in enumerate(factor_names):
        for j, f2 in enumerate(factor_names):
            if i > j:
                # Already computed (symmetric)
                corr_matrix.loc[f1, f2] = corr_matrix.loc[f2, f1]
                continue

            df1 = factor_dfs[f1]
            df2 = factor_dfs[f2]

            country_corrs: list[float] = []
            for country in df1.columns:
                if country not in df2.columns:
                    continue
                combined = pd.DataFrame(
                    {"f1": df1[country], "f2": df2[country]}
                ).dropna()
                if len(combined) <= 10:
                    continue
                corr = combined["f1"].corr(combined["f2"])
                if not np.isnan(corr):
                    country_corrs.append(corr)

            avg_corr = float(np.mean(country_corrs)) if country_corrs else 0.0
            corr_matrix.loc[f1, f2] = avg_corr
            corr_matrix.loc[f2, f1] = avg_corr

    # ------------------------------------------------------------------
    # Step 2: Distance = 1 - |corr|; run linkage; cut at threshold.
    # Matches legacy lines 831-834.
    # ------------------------------------------------------------------
    dist_values = 1.0 - np.abs(corr_matrix.values)
    # Clip floating-point noise into valid [0, 1] range
    np.clip(dist_values, 0.0, 1.0, out=dist_values)

    distance_df = pd.DataFrame(dist_values, index=factor_names, columns=factor_names)

    condensed = squareform(dist_values, checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")

    # ------------------------------------------------------------------
    # Step 3: Build clusters dict.  Matches legacy lines 836-840.
    # ------------------------------------------------------------------
    clusters: dict[int, list[str]] = {}
    for factor, cid in zip(factor_names, labels):
        cid = int(cid)
        clusters.setdefault(cid, [])
        clusters[cid].append(factor)

    # ------------------------------------------------------------------
    # Step 4: Select representatives.  Matches legacy lines 849-888.
    # ------------------------------------------------------------------
    representatives: list[str] = []

    for cluster_id, members in clusters.items():
        if len(members) == 1:
            representatives.append(members[0])
            continue

        if selection == "coverage":
            # Factor with most non-NaN data points relative to total size
            scores = {
                f: factor_dfs[f].notna().sum().sum() / factor_dfs[f].size
                for f in members
            }
            selected = max(scores, key=scores.__getitem__)

        elif selection == "unique":
            # Factor with lowest mean |corr| to factors in other clusters
            other_factors = [
                f
                for cid2, mlist in clusters.items()
                if cid2 != cluster_id
                for f in mlist
            ]
            if not other_factors:
                selected = members[0]
            else:
                scores = {
                    f: float(np.mean([abs(corr_matrix.loc[f, o]) for o in other_factors]))
                    for f in members
                }
                selected = min(scores, key=scores.__getitem__)

        else:  # 'central'
            # Factor most correlated with its cluster peers (highest mean |corr|)
            scores = {
                f: float(
                    np.mean([abs(corr_matrix.loc[f, o]) for o in members if o != f])
                )
                if len(members) > 1
                else 0.0
                for f in members
            }
            selected = max(scores, key=scores.__getitem__)

        representatives.append(selected)

    return ClusterResult(
        clusters=clusters,
        representatives=representatives,
        distance=distance_df,
    )
