# PRD — Country Rotation Strategy Platform

**Owner:** Alan Vazquez, CFA · **Status:** Active research · **Updated:** 2026-06-09

## Product
A reproducible research platform that designs, validates, and reports country equity rotation strategies for three universes — **World (34 countries), DM, EM** — and certifies any candidate strategy with out-of-sample statistical evidence before it is trusted.

## Users
- Primary: the PM/quant (Alan) running segment research and reading verdicts/reports.
- Secondary: reviewers of the public repo (code only; data proprietary, gitignored).

## Goals (acceptance criteria)
1. Leak-free data pipeline over vendor Excel inputs (perturbation-tested; ffill-only; publication-lag aware). ✅
2. Factor library with literature-grounded catalog (no raw levels; 12-1 momentum; corrected categories). ✅
3. OOS-honest factor selection: per-period IC t-stats, BH-FDR q≤0.10, HLZ weak labels, untouched lockbox. ✅
4. Backtest engine, parity-locked to legacy, with blend and benchmark-relative (active) modes, turnover-based costs. ✅
5. Validation scorecard: DSR≥0.95, PSR≥0.95, MC p≤0.05, WFE≥0.5, bootstrap CI>0, stability gates. ✅
6. HTML research report: performance abs/rel, risk abs/rel, IC analysis, building-block score decomposition, scorecard. ✅
7. Segment verdicts: a strategy is only "found" when the scorecard passes; honest negatives are first-class outputs. 🔄 in progress

## Non-goals
- Live trading/execution, intraday data, short books, derivatives overlays.
- Cap-weighted benchmark replication (vendor index levels exist in data but EW universe is the honest rotation null).

## Strategy envelope
- Long-only, 3–7 country sleeves, relative (top-N score-change) selection default; monthly–quarterly rebalance (21/63d); 2bps TC baseline with sensitivity; benchmark = equal-weight universe B&H (+ blend mode for mandates).

## Decision log pointers
See docs/context/memory.md and docs/superpowers/specs/2026-06-09-country-rotation-platform-design.md.
