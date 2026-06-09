# Memory — Architecture & Decisions

- # decision: All market data (xlsx/csv) gitignored — repo is code-only, safe for public hosting.
- # decision: Strategy universe segments = World / DM / EM (Asia, Europe, LatAm as secondary regions).
- # decision: Score pipeline = raw factor -> 4 metrics (expanding zscore, abs pct, relative rank, 63d delta pct) -> weighted factor score -> category aggregate -> composite -> cross-sectional 0-1 normalization.
- # decision: Validation standard = Deflated Sharpe, PSR, anchored walk-forward, Monte-Carlo random-selection null, equal-weight buy-and-hold null (adopted from docs/superpowers/plans/2026-06-07 methodology; must be ported to this codebase).
- # decision: docs/superpowers plans pre-2026-06 reference a `momentum_strategy` package from another repo — methodology reusable, file paths are not.
- # decision: Package root is `country_rotation/` (v0.2.0); config entry-point is `load_config(path) -> PlatformConfig` (frozen dataclasses); canonical defaults live in `configs/default.json`.
- # decision: All yield/margin/reconstruction divisions in processing.py route through `_safe_div` (non-positive denominator -> NaN) — deliberate D8 fix over legacy raw division; consensus growth ratios are NOT guarded (legacy parity, forward multiples can be negative).
- # decision: Look-ahead detection uses input perturbation (multiply + shift future rows by perturb_scale); any output difference at dates <= cutoff indicates leakage. fillna(-9e9) before allclose so NaN-matching is consistent.
- # decision: Perturbation must fillna(0.0) before scaling future NaNs so NaN gaps after cutoff are forced to a sentinel and not silently preserved; column-set mismatch between base and perturbed outputs is itself treated as dirty.
- # decision: Pure transform functions do NOT apply .iloc[window:] slicing — they return full aligned DataFrames; callers decide which rows to consume (legacy class sliced for its own downstream use only).
- # decision: absolute_percentile uses method='min' (fraction <= self); delta_percentile uses method='average'; both match legacy FactorTransformer.py exactly.
