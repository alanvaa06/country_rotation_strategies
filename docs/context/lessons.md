# Lessons

- CLI `--help` smoke tests do NOT catch wiring bugs (wrong kwargs, missing positional args) — every script needs at least one end-to-end smoke test that executes the full main() path on tiny synthetic fixtures.
- On Windows + Python <3.15, subprocess children writing non-ASCII (e.g. arrows) to a piped stdout crash with cp1252 UnicodeEncodeError — set PYTHONIOENCODING=utf-8 in the test env.
- Do not duplicate library logic in CLI scripts (build_scores re-implemented the 4 transforms with divergent metric keys); call the package function and use its keys.
- Anchored expanding-fold mean t-tests are INFLATED, not conservative: overlapping folds make fold means nearly identical, shrinking their std. Test significance on the per-period IC series (df = n_periods−1); keep folds only for regime/sign-consistency gates.
- `bfill()` on time series is a look-ahead leak by construction; gap-filling must be `ffill(limit=...)` and proven by perturbation tests (perturb data after t, assert outputs ≤ t unchanged).
- Raw levels (Price, GDP, Market_Cap) are not cross-country comparable factors; only ratios, yields, spreads, and changes belong in a factor catalog.
- Untracked design docs can vanish between sessions (the 2026-06-07 methodology plan was deleted before commit); persist canonical formulas in committed references (docs/references/validation_formulas.md) before relying on them.
- Subagent background processes die with the subagent; long detached runs must be launched by the orchestrator session (harness-tracked background) or via committed runner scripts.
- Statistical significance must be measured on the finest honest grid: period-level stats (n≈65) discard the power available in the daily curve (n≈4,000) of the same strategy.
- Verify statistics against canonical sources with hand-computed tests at 1e-12 tolerance (caught a PSR kurtosis-term bug and a DSR mean-term bug written "from first principles").
- "Byte-deterministic" claims must be PROVEN by hashing two consecutive real runs: numpy/pandas reductions jitter 1 ULP across processes (memory-alignment-dependent pairwise summation) — scalars derived from long series need math.fsum + fixed-precision quantization at the artifact boundary; in-process unit tests of determinism cannot catch this.
