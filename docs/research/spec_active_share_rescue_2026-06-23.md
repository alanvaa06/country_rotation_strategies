# Pre-registration — active_share rescue of DM full cap_tilt @21 vs ACWI

- **Date committed:** 2026-06-23 (BEFORE running the sweep)
- **Status:** PRE-REGISTERED. Editing after results = spec burn.
- **Parent:** the selected monthly book (grid_vsACWI_2026-06-23.md) +
  its forensics (overfit_forensics_DM_vsWorld_full_p21.md): NOT overfit, but
  IR = 100% DM−ACWI composition spread, **within-DM selection IR −0.06**,
  composite IC positive (rel +0.0283, t 1.66) but does NOT convert.

## Hypothesis

Cap_Tilt holds a cap-weight base and tilts ±active_share toward high-signal
/ away from low-signal countries. At the deployed active_share = 0.30 the
tilts are small and the bottom-N underweights are clipped at the base weight
(`min(active_share/N, base_i)`, long-only). **If the positive composite IC is
genuine selection skill currently muted by the small/clipped tilt, raising
active_share should lift the within-DM selection leg (book − DM cap index)
toward positive.**

## Prediction (declared)

**Does NOT rescue.** The selection-leg IR is ~scale-invariant in active_share
(the tilt scales both the mean and the vol of the selection return, leaving
IR ≈ constant), and it is already NEGATIVE (−0.06). Widening should leave
selection IR ≈ flat-to-worse and raise turnover. The only channel that could
surprise is the bottom-N clip asymmetry — hence the test.

**Falsification:** any active_share where within-DM **selection IR ≥ +0.10
with NW t ≥ 2** (the signal demonstrably adds value over the passive DM cap
index). A merely-higher *total* active IR does NOT count — that is the
composition spread, not the signal.

## Grid

DM, full (4-cat) composite, cap_tilt, @21 monthly, vs ACWI (`World`),
active basis. **active_share ∈ {0.30 (baseline), 0.50, 0.70, 1.00}.**
3 net-new trials (0.30 already measured) → ledger ~234 → ~237.

## Metrics (per active_share)

Primary: **within-DM selection IR (book − DM cap index) + NW t** (the skill
that must turn positive). Secondary: total active IR vs ACWI, segment-spread
IR, ann one-way turnover. Gross — costs only worsen it, so a gross-negative
selection leg settles the question.

## Stage gate

If (and only if) some active_share clears the falsification bar on the
deterministic selection IR, run the MC random-tilt null at that level to test
significance before any deployment talk. Otherwise the rescue is rejected.
