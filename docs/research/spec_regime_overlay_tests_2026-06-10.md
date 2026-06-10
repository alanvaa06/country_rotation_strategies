# Pre-Registered Spec — RoRo Regime Tests (2026-06-10)

Committed BEFORE execution. Two tests, one trial-ledger entry each.
Everything below is fixed ex-ante; any deviation must be reported as a new trial.
Diagnostic context that motivated these tests: `docs/context/results.md` [R2]
(regime-conditioned attribution, 2026-06-10). Both tests reuse the exact daily
series reconstruction already parity-verified against the certified verdicts.

---

## Test 1 — Divergence-sized DM−ACWI spread overlay (Proposal 1)

**Hypothesis.** RoRo's DM_Eq/EM_Eq label divergence carries information about
the forward DM−ACWI spread; sizing the spread position by divergence improves
risk-adjusted return over constant exposure.

**Position.** Daily spread `s_t = r_DM,t − r_World,t` (vendor index columns,
identical to the forensics alpha-decomposition series).

**Signal (fixed).** JM classifier (persistence-enforced), segments `DM_Eq`
and `EM_Eq`, labels **lagged one trading day**. JM is primary because it was
declared primary throughout the program *before* this spec; HMM is a
pre-declared robustness re-run (reported, never used for selection — noted:
HMM looked stronger in the motivating diagnostic, so choosing it now would be
post-hoc selection).

**Mapping (fixed, three levels, no fitted parameters).**

| Lagged state pair | Weight w_t |
|---|---|
| DM_Eq = Risk-on AND EM_Eq = Risk-off | 1.0 |
| DM_Eq = Risk-off AND EM_Eq = Risk-on | 0.0 |
| all other combinations | 0.5 |

**Window.** 2010-03-16 → 2025-11-14 (book backtest window ∩ RoRo coverage).

**Costs.** 5 bps one-way on |Δw| (DM-tier index instruments). Switches/yr reported.

**Metrics.**
- Overlay (gross and net) vs two baselines: constant w=1 and constant w=w̄
  (the overlay's average weight — isolates timing from exposure reduction).
- **Primary statistic:** timing component `timing_t = (w_t − w̄)·s_t`;
  Newey-West t of its mean, and a **block-bootstrap label null**: N=500
  stationary-bootstrap resamples of the JOINT (DM_Eq, EM_Eq) daily label-pair
  sequence (mean block 25 days ≈ observed regime duration, wrap-around,
  seed 42), same mapping applied, p = (#null timing-mean ≥ actual + 1)/(N+1).
- Halves split reported (front-loading diagnostic).

**Pass bar (pre-stated).** NW t(timing) ≥ 2 AND label-null p ≤ 0.05 AND
net overlay Sharpe > constant-w=1 Sharpe. All three or the overlay is
NOT confirmed (suggestive at best). Note: all divergence states had positive
spread means in the motivating diagnostic, so the overlay may well lose to
constant exposure — that outcome is informative and will be reported as such.

**Ledger.** +1 trial (mapping + classifier + lag + costs = this one spec).

---

## Test 2 — EM transitional-state alpha concentration (Proposal 2)

**Hypothesis (post-hoc, found among ~36 diagnostic cells — this test is
CONFIRMATION, not discovery).** EM Cap-Tilt active return is concentrated in
the JM `EM_Eq` Transitional state (diagnostic: +4.9%/yr, NW t 3.6).

**Series.** EM book daily active return (engine reconstruction, parity-locked),
lagged JM `EM_Eq` labels (same classifier/segment the cell was found with —
switching classifiers now would be selection).

**Battery (all fixed).**
1. **Half-split:** conditional means/IRs per state in H1 (≤ 2018-01-09) and
   H2 (> 2018-01-09). Confirmation requires Transitional mean > 0 in BOTH
   halves AND Transitional = top-ranked state in BOTH halves.
2. **Thirds:** same by thirds (reported, not gating).
3. **Cross-classifier:** HMM and tercile by halves (reported, not gating).
4. **Episode view:** distinct Transitional episodes (≥5d), per-episode
   cumulative active return, sign count.
5. **Multiplicity-honest significance:** block-bootstrap label null — N=500
   stationary-bootstrap resamples of the `EM_Eq` label sequence (mean block
   25d, seed 42); for each, compute the MAX conditional NW |t| across the
   three states; p = (#null max-|t| ≥ 3.61 + 1)/(N+1). This prices "how often
   does a persistent 3-state label produce a cell this strong by chance".

**Pass bar (pre-stated).** Gates 1 AND 5 (p ≤ 0.10) ⇒ "confirmed in-sample;
eligible for ONE pre-registered active_share-scaling spec + live tracking."
Either fails ⇒ "not confirmed — park, revisit only with live data."

**Ledger.** +1 trial.

---

## Execution plan

1. Commit this spec (git timestamp = pre-registration).
2. One analysis script (`outputs/research/_regime_overlay_tests.py`,
   gitignored, exploratory tier) implementing both tests exactly as above.
3. Self-verification: spread series reconciles to forensics (+0.75%/yr,
   NW t ≈ 2.4); EM active series reconciles to verdict IR 0.2917; the
   motivating diagnostic cell (t 3.61) reproduces before halves are split;
   bootstrap nulls seeded and deterministic.
4. Report results against the pre-stated bars verbatim — pass or fail.
