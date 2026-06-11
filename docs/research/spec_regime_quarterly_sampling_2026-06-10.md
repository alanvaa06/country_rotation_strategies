# Pre-Registered Spec — Test 3: EM Transitional Effect Under Quarterly State-Sampling (2026-06-10)

Committed BEFORE execution. One trial-ledger entry. Follows Test 2
(CONFIRMED in-sample, spec 8a32959): EM Cap-Tilt daily active return is
concentrated in the lagged JM `EM_Eq` Transitional state. This test asks the
only question that matters for tradeability: **does the effect survive when
the state can only be observed at the quarterly rebalance date?**

**Honest expectation, stated up front:** average transitional episode ≈ 25
days vs a 63-trading-day holding period — the state will often change
mid-quarter, diluting at-rebalance information. This test may well fail; a
failure means "true but untradeable at our certified cadence" (intra-quarter
action is already ruled out by the monthly-cadence evidence).

## Design (all fixed)

**Series.** EM book engine reconstruction (parity-locked). Period active
returns from `period_results.active_return` (engine's own quarterly grid,
~65 periods). Period i earns over (d[i−1], d[i]]; it is conditioned on the
**lagged JM `EM_Eq` label at d[i−1]** (the information a quarterly rule has
when the position is put on). First period dropped (no prior grid date).

**Primary statistic.** Δ = mean(period active | start-state = Transitional)
− mean(period active | start-state ≠ Transitional).

**Significance.** Block-bootstrap label null: N=500 stationary-bootstrap
resamples of the daily `EM_Eq` label sequence (mean block 25d, wrap-around,
seed 42); each resample re-sampled at the same grid dates, Δ* recomputed;
p = (#Δ* ≥ Δ + 1)/(N+1).

**Gates (pre-stated, both required to PASS):**
1. Δ > 0 (right sign), and
2. label-null p ≤ 0.10.

PASS ⇒ eligible for ONE pre-registered `active_share(state-at-rebalance)`
rule spec + paper-trade tracking. FAIL ⇒ effect untradeable at quarterly
cadence; park; no rule.

**Pre-declared diagnostics (reported, not gating):**
- Per-state period counts, means, simple t (n per state ~20 — power-limited
  by construction; that is why the bootstrap carries the inference).
- Contemporaneous-mix check: fraction of days INSIDE each period whose
  lagged label is Transitional vs that period's active return (Pearson +
  Spearman). Discriminates "predictive at rebalance" vs "merely
  contemporaneous" — if mix-correlation is strong while Δ is weak, the
  effect is real but arrives intra-quarter, where we cannot act.
- Same Δ for HMM labels (robustness, never selection).

**Ledger.** +1 trial (total regime-family: 3).
