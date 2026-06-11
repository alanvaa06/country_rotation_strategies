# Pre-Registered Spec — Test 4: Transitional-Concentration Replication Across Sibling Books (2026-06-10)

Committed BEFORE execution. One trial-ledger entry (regime family entry #4).

**Motivation.** Test 2 confirmed (in-sample, multiplicity-honest) that the EM
Cap-Tilt book's daily active return concentrates in the lagged JM
Transitional state. The proposed mechanism — top-N *change*-selection earns
when cross-sectional leadership rotates — predicts the same pattern in the
sibling books that run the identical signal on different universes. This
test REPLICATES that one directional prediction; it does not scan for new
cells, and per Test 3 its outcome cannot unlock a trading rule at the
certified cadence (timescale wall). Stakes: whether the runbook's
regime-conditioned monitoring language is a platform property or an
EM-specific fragility.

## Replication targets (gating) — selection-only active returns

| Book | Reconstruction | Benchmark | Matched RoRo label (fixed) |
|---|---|---|---|
| World Cap-Tilt | `reconstruct("World", None)` | World vendor index (own cap index) | `Equity` |
| DM Cap-Tilt | `reconstruct("DM", None)` | DM vendor index (own cap index) | `DM_Eq` |

(DM-vs-ACWI is excluded from gating: its active return is dominated by the
passive spread, not selection — it appears in the descriptive panel only.)

**Conditioning.** Lagged (t−1) JM labels, matched segment per table.
Window: book daily curve ∩ label coverage (2010-03 → 2025-11).

**Gate per book (directional, pre-stated):** Transitional conditional mean
is (a) the maximum of the three states AND (b) > 0.

**Family verdict:**
- 2/2 PASS ⇒ REPLICATED — mechanism treated as a platform property;
  runbook monitoring language extended to all books.
- 1/2 ⇒ PARTIAL — runbook stays book-specific; report which book failed.
- 0/2 ⇒ NOT REPLICATED — EM finding downgraded to "EM-specific,
  mechanism unconfirmed"; runbook language weakened accordingly.

**Reported, not gating:** per-cell NW t (power differs across books — DM TE
≈ 2.6% vs EM 4.3%, so t-magnitudes are not comparable); halves split per
book; max-state-|t| label-null (N=500, mean block 25d, seed 42) per book for
context; HMM same tables as robustness.

## Descriptive panel (no gates, no ledger cost)

Same conditional table re-printed for the two known books (EM Cap-Tilt vs EM
index; DM Cap-Tilt vs ACWI) so all four books sit side by side as the
monitoring reference table. Equal-weight construction variants are out of
scope (different construction path; revisit only if this test replicates).

## Self-verification (pre-stated)

Each reconstructed book's full-sample active IR must reconcile to its
verdict JSON (`sharpe_ann` in verdict_{World|DM}_prior_vm_p63_active_captilt_capbmk.json,
EM capbmk 0.2917, DM vsWorld 0.3016) to ≤ 1e-3. Bootstrap seeded.

**Ledger.** +1 (regime family total: 4).
