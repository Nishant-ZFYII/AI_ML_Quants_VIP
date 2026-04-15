# Investment Strategy Plan — Wheat Futures (Binary Buy/Sell/Flat)

## Context

**Why this exists.** Across the NYU AI/ML Quants VIP lab, ~50 model architectures × 12 students hit a ~50% AUC ceiling on daily wheat direction (see `alternative_data/processed/v6_final_results.csv`: best AUC 0.545). Last week we added alt data (COT, cross-commodity, Trends, weather, VIX/OVX — all in `alternative_data/raw/`) and it **did not improve returns**. The classifier is not the bottleneck and more data on the *primary* signal won't fix it.

**The bet.** Stop trying to make the classifier better and instead attack the "how do you trade a mediocre classifier profitably" problem. This is the **investment-strategy deliverable** (Part B of `alternative_data/RESEARCH_PLAN.md`) — a standalone track, no dataset-collection work bundled in. Teammates are doing RAG, random search, new architectures (horizontal exploration). Our edge is vertical: **meta-labeling + ensemble + ablation overlays**, using only data already in `alternative_data/raw/`.

**Intended outcome.** A single strategy module, driven by an ensemble of our 4 assigned models, gated by a meta-labeling classifier on alt data, with switchable overlays so we can ablate and report *which overlays actually add Sharpe*. Target: walk-forward Sharpe > 1.0 after costs on 2022–2025 test window.

**Binary output.** Every signal maps to one of {long, short, flat}. Aligned with the course requirement.

---

## Approach (recommended)

### Part 1 — Primary prediction: 4-model × 4-horizon ensemble
Train each of the assigned models (ARX, BiRNN+Attention, BiGRU+Skip+Attention, RCNN+Self-Attention) on 4 horizons (1d, 3d, 5d, 10d direction). Stack the 16 `predict_proba` outputs via logistic-regression meta-learner on out-of-fold predictions (time-series CV). Output: single `P(up)` per day.

Literature: weekly AUC typically beats daily by 5–10 pp (Lopez Gil 2024, Guida 2025).

### Part 2 — Meta-labeling (the novel contribution)
A **secondary** binary classifier (`P(act)`) whose job is **not to predict direction** but to predict *whether the primary ensemble's call will be correct*. Training target = 1 if primary was right on day t, 0 otherwise. Use XGBoost (interpretable via SHAP, tabular-friendly).

**Feature set — drawn entirely from `alternative_data/raw/` (already collected):**
- COT positioning: hedging-pressure z-score, speculator-net / open-interest, 4-week position-change momentum, 52-week extreme z-score.
- Cross-commodity: wheat/corn and wheat/soy ratios + 20-day momentum, crude oil level change, DXY change, gold return.
- Risk / attention: VIX level + change, OVX level + change, Google Trends `wheat price` z-score and spike indicator.
- Weather anomalies: temp/precip anomalies from `nasa_power_ag.csv` over Kansas and North Dakota.

Why this works when last week's alt data in the primary did not: alt data too weak to predict direction can still predict *trustworthiness of a directional call* — López de Prado AFML ch. 3. Reframes last week's "alt data didn't help" as "alt data was in the wrong layer."

**Signal rule:** Trade only when `P(up) > 0.55` AND `P(act) > 0.60` (or `P(up) < 0.45` for shorts). Else flat.

### Part 3 — Position sizing
Volatility-target overlay on top of the gated signal:
```
size_t = direction × clip(target_vol / realized_vol_20d, 0, 2) × P(act)
```
Target annualized vol ≈ 10%. `P(act)` acts as a soft-Kelly multiplier. Binary output preserved as `sign(size_t)`; continuous size only affects $ PnL, not the buy/sell/flat label.

### Part 4 — Ablation overlays (switchable via config)
Each overlay toggleable on/off so we can report the Sharpe delta per overlay — this *is* the scientific contribution.

| Overlay | Rule | Data need |
|---|---|---|
| `vol_target` | size ∝ 1/realized vol | OHLCV (have) |
| `cot_gate` | require COT commercials z-score agrees with direction | COT (have) |
| `tsmom_regime` | only trade with 12-mo price-return sign | OHLCV (have) |
| `wasde_blackout` | flat 1 day before / 1 day after WASDE release | **new: WASDE calendar** |
| `seasonality` | reduce size in harvest window (Jun 15–Jul 31) | date (have) |

### Part 5 — Walk-forward backtest
Reuse the existing `SimpleBacktester` in `Quants data/Quants /research_grade_forecasting.py` (lines ~2959–3032). It already supports binary/proportional sizing, configurable transaction cost, Sharpe, max drawdown, win rate. Extend it with:
- vol-target sizing mode
- overlay-toggle config
- per-overlay ablation report

Train window: 2010–2021. Test: 2022–2025 walk-forward. Costs: \$12.50 round-trip/contract (CBOT wheat e-mini) + 1-tick slippage. Target metrics: Sharpe > 1.0, max DD < 25%, win rate > 52%.

---

## Files to create / modify

**Repository.** First rename the unpushed folder `alternative_data-20260415T010050Z-3-001/alternative_data/` → `alternative_data/` at the repo root, then commit.

**New module:** `alternative_data/strategy/`
- `strategy/ensemble.py` — stacks the 4 models × 4 horizons via `LogisticRegression` meta-learner. Inherits `BaseForecastModel` from `VIP_Abstract_Class.ipynb` so the lab's abstract-class requirement is met.
- `strategy/meta_label.py` — XGBoost `P(act)` classifier; exposes `fit(primary_preds, alt_features, primary_correct)` and `predict_proba`.
- `strategy/overlays.py` — one function per overlay (`apply_vol_target`, `apply_cot_gate`, `apply_tsmom_regime`, `apply_wasde_blackout`, `apply_seasonality`); each takes `(signal_series, config) → gated_signal_series`.
- `strategy/backtest.py` — thin wrapper around the `SimpleBacktester` class adapted from `research_grade_forecasting.py`. Adds vol-target sizing + ablation loop.
- `strategy/run_ablation.py` — driver script: loops overlay on/off combinations, reports Sharpe table.

**One new data artifact:** `alternative_data/raw/wasde_calendar.csv` — WASDE release dates scraped from usda.gov/oce/commodity-markets/wasde (one-shot scrape, ~15 rows/year back to 2010). Needed only for the `wasde_blackout` overlay.

**Notebook deliverable:** `strategy_ablation.ipynb` at repo root — runs end-to-end, reports the ablation table + equity curves, satisfies the weekly lab-notes format.

---

## Reused existing code (do not rewrite)

- `VIP_Abstract_Class.ipynb` — `BaseForecastModel` (ensemble inherits this).
- `Quants data/Quants /research_grade_forecasting.py` lines 2959–3032 — `SimpleBacktester` class: equity curve, Sharpe (√252 annualization), max drawdown, win rate, execution-delay shift. Reuse as base class for `strategy/backtest.py`.
- `daily_bigru_classification.ipynb` ~line 180–190 — reference for target definition: `y = 1 if close_t > close_{t-1} else 0`. Extend to multi-horizon: `y_h = 1 if close_{t+h} > close_t else 0`.
- `alternative_data/phase_test_v6_final.ipynb` — reference data-loading + 80-column feature pipeline. Feed the same tensor shape into the 4 primary models and the meta-labeler.
- `alternative_data/raw/cot_wheat_disaggregated.csv` — already contains commercial/noncommercial/open-interest columns needed for `cot_gate` overlay.

---

## Verification

End-to-end sanity path:

1. **Ensemble sanity.** Run `strategy/ensemble.py` on 2010–2021 train. Confirm stacked AUC beats each base model's OOF AUC on validation. Expected: 0.58–0.65 (per Celik & Celik 2025 stacking results).
2. **Meta-label sanity.** Check `P(act)` calibration: bucket by predicted decile, confirm realized primary-accuracy is monotonically increasing across deciles. If flat, meta-label is not learning — fall back to raw confidence gate.
3. **Ablation run.** Execute `run_ablation.py` across 2^5 = 32 overlay on/off combinations (or a Latin-hypercube subset). Output: CSV of (config, Sharpe, MaxDD, WinRate, TradeCount).
4. **Baseline comparison.** Confirm the gated + overlayed strategy beats (a) buy-and-hold wheat, (b) raw-classifier sign strategy, (c) the no-overlay `SimpleBacktester` on the same test window.
5. **Cost sensitivity.** Re-run with 2× costs; strategy should still have Sharpe > 0.5. If it collapses, we're trading too often — tighten confidence thresholds.
6. **Binary-output check.** Assert every bar's signal ∈ {-1, 0, +1} before sizing is applied — confirms course-requirement alignment.
7. **Notebook runs cold.** `strategy_ablation.ipynb` must execute Run-All with no manual edits on a fresh checkout.

## Out of scope

- New data collection (FinBERT news sentiment, MODIS NDVI, USDA crop progress) — that's the separate "Dataset Collection" track, which we are explicitly *not* doing.
- Carry / roll-yield signal (would require front/back contract data we don't have).
- Cross-sectional grain momentum (requires multi-instrument portfolio — scope creep).
- Re-tuning the primary models beyond what's already in the classification notebooks.
