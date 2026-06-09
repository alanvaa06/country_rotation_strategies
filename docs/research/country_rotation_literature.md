# Country Rotation Strategies — Literature Evidence Base

**Date:** 2026-06-09
**Method:** Deep-research harness — 5 search angles, 23 primary sources fetched, 114 claims extracted, top 25 adversarially verified (3-vote panels), 21 confirmed / 4 refuted, synthesized to 10 findings.
**Purpose:** Ground the platform's factor priors (spec §6 Stage 1), composite weighting, selection design, and validation thresholds in verified evidence.

---

## 1. Signal ranking (strongest verified OOS evidence first)

### 1.1 Country momentum (12-1) — STRONGEST standalone signal [HIGH confidence]
- 18 DM index futures, 1978–2011: long-short 12-1 momentum earned **8.7%/yr (t=4.14, Sharpe 0.73)** gross (Asness-Moskowitz-Pedersen 2013, *J. Finance*; Table I Panel B, verified verbatim). Corroborated independently by Balvers & Wu (2006) and the momentum cluster in Calice & Lin (2021).
- Canonical construction: **return over past 12 months skipping the most recent month** (MOM2-12), value = prior month BE/ME of MSCI country index (AMP 2013 p.936-937).
- → Platform: `Momentum_12_1`/`Momentum_6_1` added to catalog (decision D4) — directly validated.

### 1.2 Country value — second [HIGH confidence]
- Same universe: BE/ME long-short **6.0%/yr (t=3.45, Sharpe 0.61)** gross (AMP 2013).
- CAPE variant: long-horizon (5–15y) signal only; pooled R²≈48% for 10-15y real returns but per-country fit ranges Japan ~90% to Canada ~1%; EM forecast errors systematically higher (panel SE 6.5–9.0% vs DM 3.8–6.5%) (Keimling 2016; Klement 2012 — practitioner papers, overlapping-window R² inflation per Boudoukh-Israel-Richardson). [MEDIUM]
- → Valuation factors valid for quarterly rotation horizon mainly through cross-sectional ranks, not absolute levels.

### 1.3 Value + Momentum composite — THE construction result [HIGH confidence]
- Country value/momentum correlation **−0.34 to −0.37**; 50/50 combo Sharpe **1.16 (t=6.62)** vs 0.61/0.73 standalone (AMP 2013). Same conclusion in timing form (AIM 2017): Sharpe-maximizing weight ≈ 50/50 despite unequal standalone Sharpes, because correlation is negative.
- → Composite category weights: near-equal Value/Momentum split is evidence-based default; pure category bets (Scenario D/E style) are not.

### 1.4 Multi-factor country allocation implementable [HIGH core / MEDIUM details]
- Tilts toward value+momentum+size+low-risk country indexes beat world cap-weight after transaction costs and TE constraints, via ETFs/futures (Angelidis & Tessaromatis 2017, *FAJ*, 1980–2015). Per-factor magnitude numbers REFUTED — cite direction only.
- Counterweight: Zaremba-line replications find costs "largely lethal" to most **long-only** country anomaly portfolios. → TC sensitivity analysis is mandatory, not optional.

### 1.5 Momentum + mean reversion jointly [MEDIUM]
- Balvers & Wu (2006, *JEF*): combined momentum/mean-reversion 1.1–1.7%/month excess, survives 2%-per-switch cost (Max1-Min1 keeps 10.8%/yr net). Sample ends 1999; Malin & Bornholt (2013) find DM long-run reversal absent post-1989 (decay documented).

### 1.6 Breadth of other predictors [MEDIUM, uncorrected]
- Calice & Lin (2021, *JEF*; 90 variables, 45 countries, 2002–2018): significant predictors cluster into **size, profitability/value (EBIT/EV, earnings yield), momentum, default risk (Debt/Capital, Debt/Equity)**. NO Harvey-Liu-Zhu correction across 90 tests — expect a material fraction false positives. Default-risk factor unreplicated.
- → Our Quality (leverage) and Profitability blocks have tentative support; hold to stricter (FDR-adjusted) thresholds, as spec §6 already requires.

## 2. Typical IC magnitudes
- First-PC multi-factor signal: monthly OOS R² ≈ **0.30–0.42%** vs historical mean; 2nd/3rd PCs and placebo PCs negative (Calice-Lin Table 13, verified). Implies cross-sectional IC ≈ **0.05–0.06** (inference, flagged).
- → Country signals are weak per-period; Sharpe comes from breadth + compounding. A full-sample mean IC filter of 0.05 (current backtesting_tests threshold) is at the upper end of what literature supports — must be walk-forward, not full-sample.

## 3. Known failure modes (verified)
1. **Absolute valuation thresholds are in-sample artifacts.** Faber (2012) CAPE<15 / >30 rules improved in-sample results (4–7pp/yr real spread), but AIM 2017 (peer-reviewed, OOS 1900–2015, rolling 60y calibration): realistic CAPE contrarian timing **failed to beat buy-and-hold even gross** (Sharpe 0.37 vs 0.38); cheap/expensive quintiles defined on full sample + ex-post signal choice = hindsight bias; "early equals wrong". → Prefer relative (cross-sectional) selection; absolute-threshold strategies demand extra OOS skepticism.
2. **Transaction costs** can kill long-only country anomalies (Zaremba). → Net-of-cost results at realistic bps, plus cost-sensitivity sweep.
3. **Regime dependence:** most verified effect sizes are gross, in-sample, samples ending 1999–2018; post-2010 value winter and live underperformance of CAPE-cheap funds (GVAL post-2014) flagged. → Per-year regime breakdown required in reports.
4. **EM data quality:** higher forecast errors in EM; survivorship concerns. → Segment-level verdicts must carry EM caveat.

## 4. Refuted claims (do NOT cite)
- A&T per-factor outperformance 2.48–8.42% (1-2).
- Calice-Lin composite 0.507%/month long-short (0-3).
- Keimling "all 17 countries" CAPE universality (0-3; Denmark exception).
- Keimling "only CAPE and P/B reliable" (1-2).

## 5. Open gaps (validation arm)
No verified claims survived on IC methodology standards, Deflated Sharpe Ratio, walk-forward protocols, or HLZ t>3.0 thresholds (sources exist — Bailey & López de Prado 2014; Harvey-Liu-Zhu 2016 — but the harness verified none of their numeric content this run). Platform keeps its validation standards from the methodology spec (DSR≥0.95, PSR≥0.95, MC p≤0.05, WFE≥0.5, FDR q=0.10, HLZ t<3 ⇒ "weak" label) as **design policy**, attributed to those papers, acknowledged here as not re-verified.

## 6. Direct implications adopted by the platform
| Literature finding | Platform decision |
|---|---|
| 12-1 momentum strongest | D4: Momentum_12_1/6_1 in catalog; Momentum block first-class |
| 50/50 value-momentum near-optimal | Balanced Value/Momentum category weights as default scenario; scenario grid retained but DSR-penalized |
| IC ≈ 0.05 typical | IC screening thresholds set near 0.03–0.05, walk-forward only (spec §6 Stage 3) |
| Absolute thresholds = snooping risk | Relative selection is default; absolute mode kept but flagged for stricter OOS gates |
| Costs potentially lethal | TC sensitivity sweep in validation protocols |
| Leverage/profitability tentative | Quality/Profitability factors marked for FDR-adjusted screening; consensus-growth factors `exploratory=True` |
| EM noisier | EM verdict requires explicit caveat; segment-level validation separate |

## Sources (primary, verified-claim-bearing)
- Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere", *J. Finance* — doi:10.1111/jofi.12021
- Asness, Ilmanen, Maloney (2017) "Market Timing: Sin a Little", *JOIM* — AQR white paper PDF
- Angelidis, Tessaromatis (2017) "Global Equity Country Allocation: An Application of Factor Investing", *FAJ* 73(4)
- Balvers, Wu (2006) "Momentum and mean reversion across national equity markets", *J. Empirical Finance*
- Calice, Lin (2021) "Exploring risk premium factors for country equity returns", *J. Empirical Finance* (Essex repository manuscript)
- Keimling (2016) "Predicting Stock Market Returns Using the Shiller CAPE" — SSRN 2736423 (practitioner)
- Klement (2012) "Does the Shiller-PE Work in Emerging Markets?" — SSRN 2088140 (practitioner)
- Faber (2012) "Global Value: Building Trading Models with the 10 Year CAPE" — SSRN 2129474 (practitioner, commercial interest)
- Harvey, Liu, Zhu (2016) "…and the Cross-Section of Expected Returns" — policy reference (unverified this run)
- Bailey, López de Prado (2014) "The Deflated Sharpe Ratio" — SSRN 2460551 — policy reference (unverified this run)
