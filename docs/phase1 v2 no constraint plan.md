# Phase 1 v2 — No-Constraint (Minimum-Constraint) Accuracy Plan

## Context

**Current state.** The `phase1_v2_classification_all_models.ipynb` pipeline (v2.3) reached **52.2 % accuracy / AUC 0.526** on next-day wheat direction using the TA-mandated setup: 31 FRED-MD variables + lagged close, 30-day rolling window, 5-fold `TimeSeriesSplit` over 2008+, BiGRU with 5-seed ensembling as the headline model. Phase 2's full `+ABC` alt-data stack independently reached **52.1 % / AUC 0.535** on BiRNN+Attention. The TA-compliant ceiling on this dataset clusters in the 51–52.2 % band.

**Why this plan.** The TA brief forced several choices that bound accuracy from above: next-day target horizon (lowest SNR of any horizon we could pick), static 5-fold CV instead of walk-forward, no price transformations beyond log-return, no regime-aware training. Separately, the user asked "if we relaxed the TA constraints, how much more is there?"

**Reality check on expected accuracy.** Earlier drafts of this plan floated numbers like "70–75 % on 5-day" and "~80 % with no constraints." Those are not realistic for out-of-sample commodity direction forecasting. Consensus from the literature and practitioner discussions (Quant SE, published commodity-forecasting papers, reddit/algotrading and r/Commodities surveys):

- **Sophisticated quant firms achieve 51–58 % directional accuracy out-of-sample** on liquid commodity futures. 60 %+ is usually a red flag for overfitting or look-ahead leakage, not a genuinely stronger model.
- **Alternative-data lift is typically +2–5 pp**, not +10–20 pp. Weather, sentiment, and satellite features move the needle in that range when honestly validated.
- **In-sample vs out-of-sample gap is large.** Studies routinely show 78 %+ in-sample accuracy collapsing to 50–55 % out-of-sample. Phase0_v1's 74 % wavelet number was in that trap (acausal DWT leak). The honest number was much lower.
- **Regime matters more than model.** Calm regimes give 1.5–4 % MAPE on price levels; volatile regimes blow out to 10–20 % error. A single model averaged across regimes sits in the middle and looks mediocre on both.

**Revised honest ceiling under these constraints:**

| Horizon | v2.3 baseline | Realistic ceiling with this plan | Why not higher |
|---|---|---|---|
| Next-day direction | 52.2 % | **53–55 %** | Daily SNR is near-zero on wheat; +1–3 pp is the practitioner consensus for a good-but-not-magic model |
| 5-day direction | ~54 % (not yet measured honestly) | **55–58 %** | 5-day has more SNR than daily but is still in the 51–58 % band that liquid-commodity forecasters sit in |
| Daily, conviction-gated (top-50 % confidence) | n/a | **56–60 % on kept subset** | Honest framing: accuracy on days worth acting on. This is where firms actually live |

The earlier 73–75 % and 80 % numbers are **withdrawn** — they were theoretical upper bounds that required ingredients (leaky features, in-sample evaluation, target-definition changes, broader model classes) this plan excludes.

**This plan's constraints.** We keep the **31 FRED-MD macros** (fixed feature source) and the **three existing RNN architectures** (BiGRU, BiRNN+Attention, BiRNN+Skip, plus ARX logistic as linear baseline). Everything else the TA prescribed — target horizon, window length, CV protocol, price-channel transformations, training-time augmentations, ensembling strategy — is negotiable. Outcome: a research path that honestly estimates the ceiling of this **model+data combination** without changing the problem's ingredient list.

---

## Tier 1 — Target horizon (single biggest lever)

Changing the forecast horizon is the single biggest accuracy lever available. 5-day returns have ~5× the SNR of daily returns; most of the 52 % → 70 %+ jump comes from this alone.

| # | Variant | Expected BiGRU accuracy | Notes |
|---|---|---|---|
| 1.1 | 5-day forward direction | **70–72 %** | Baseline horizon shift. phase0_v1 saw 74 % with leaky wavelet DWT; 70 % is the honest floor |
| 1.2 | 10-day forward direction | ~73–75 % | Signal becomes drift-dominated; less reactive to macro |
| 1.3 | Weekly Friday-to-Friday direction | ~72–74 % | Aligns with COT release cadence (useful if we ever re-add COT) |
| 1.4 | Conviction-gated daily (drop days with \|expected move\| < 0.5σ) | ~58–62 % on kept subset | Honest framing: "accuracy on days worth trading" |

**Recommendation:** run the plan at two horizons — **daily** (to stay comparable with prior work) and **5-day** (to show the honest no-constraint ceiling).

## Tier 2 — Feature engineering on wheat close

Close-only transformations. Adds derived columns to the price channel without introducing any new data source.

| # | Feature group | Columns | Expected lift (daily / 5-day) |
|---|---|---|---|
| 2.1 | Realized volatility | 5d, 10d, 20d std of log-returns | +0.3–0.6 pp / +0.5–1 pp |
| 2.2 | Higher moments | 20d rolling skewness + kurtosis | +0.2–0.4 pp / +0.3–0.6 pp |
| 2.3 | MA distance | `(close − MA_k) / MA_k` for k ∈ {5, 10, 20, 50} | +0.3–0.5 pp / +0.5–1 pp |
| 2.4 | Long memory | 60d rolling Hurst exponent | +0.1–0.3 pp / +0.2–0.5 pp |
| 2.5 | Price-only momentum | RSI(14), MACD(12,26,9) built from close (not OHLC) | +0.3–0.5 pp / +0.5–1 pp |

## Tier 3 — Feature engineering on the 31 FRED-MD

The RNN currently sees 30 near-constant copies of each macro inside every 30-day window — the sequence branch has almost nothing to chew on. These engineer the macro panel into something temporally richer.

| # | Feature group | Expected lift |
|---|---|---|
| 3.1 | Within-window deltas: `macro_t − macro_{t-30}` as 31 extra columns | +0.3–0.6 pp |
| 3.2 | Macro surprise z-scores: `(macro_t − μ_12mo) / σ_12mo` per macro | +0.3–0.5 pp |
| 3.3 | Cross-macro interactions: Optuna picks top-10 pairwise products from 31×31 grid | +0.2–0.5 pp |
| 3.4 | Per-fold PCA: 31 macros → top 8 PCs + residual variance explained | +0.1–0.3 pp (sometimes −0.2 pp) |
| 3.5 | Longer lookback window: 30 → 60 or 90 days | +0.2–0.5 pp |
| 3.6 | Multi-resolution window: 30-day daily stream + 12-month monthly stream, merged in head | +0.3–0.8 pp |

## Tier 4 — Training protocol changes

CV / loss / augmentation — every item here is different from what v2 through v2.3 used.

| # | Change | Expected lift |
|---|---|---|
| 4.1 | Walk-forward retraining (retrain monthly on all-to-date) instead of 5 static folds | +0.3–0.8 pp |
| 4.2 | Purged + embargoed k-fold (drop train rows whose 30-day window overlaps val, add 5-day gap) | Leakage-honesty fix; may reduce headline accuracy by 0.3 pp but trustworthy |
| 4.3 | Focal loss (γ=2) replacing class-weighted BCE | +0.2–0.4 pp, especially on Attn (class-biased in v2.3) |
| 4.4 | Temporal mixup: `λ·x_i + (1−λ)·x_j` for adjacent windows | +0.2–0.5 pp |
| 4.5 | Gaussian input noise (σ=0.01 on standardized features) during training | +0.1–0.3 pp |
| 4.6 | Label smoothing (0.9 / 0.1 targets) | +0.1–0.2 pp |
| 4.7 | Regime-conditional models: 2 BiGRUs (low-vol / high-vol by 20d ATR-%), gate at inference | +1–2 pp on fold 5 specifically |

## Tier 5 — Ensemble / inference changes

Only uses the 3 existing RNN architectures — no new model classes.

| # | Change | Expected lift |
|---|---|---|
| 5.1 | **5-seed ensembling on BiGRU** (Phase 2 deliberately turned this off for fair ablation; turning it back on is free money) | +0.3–0.5 pp |
| 5.2 | **MC Dropout at inference** on Attn/Skip (20 forward passes, average probs) — seed ensembling hurts them in v2.3, but MC Dropout is orthogonal | +0.1–0.3 pp |
| 5.3 | Snapshot ensembling within one run (cyclic LR, 5 restarts per seed, average snapshots) | +0.3–0.5 pp |
| 5.4 | **Logistic meta-stacker** on OOF probs of {ARX, BiGRU, Attn, Skip}, fit per-fold — can learn negative weights for systematically-wrong models | +0.3–0.6 pp |
| 5.5 | Pooling ensemble: 3 BiGRUs with `pool ∈ {last, mean, max}`, average probs | +0.2–0.4 pp |
| 5.6 | Per-fold temperature scaling on each model's logits before stacking | +0.1–0.3 pp |

---

## Recommended execution path

Three experiments, stacked in order. Each row is measured independently so we can attribute lift.

### Experiment A — daily horizon, richer features

Goal: push the honest daily-direction number past 54 %.

1. Add Tier 2 (close transformations) + Tier 3.1, 3.2 (FRED deltas + surprise z-scores) to the feature set
2. Keep 5-fold TS-CV, 2008+, 30-day window — so results are directly comparable to v2.3
3. Re-run Optuna (100 trials per model, widened search space)
4. Turn 5-seed ensembling back on for BiGRU; add MC Dropout for Attn/Skip (5.1, 5.2)
5. Add logistic stacker on top (5.4)

**Expected:** BiGRU ~53.5 %, stacker ~54–55 %. Wall time ~6 h on A100.

### Experiment B — 5-day horizon, same features

Goal: honestly estimate the no-constraint ceiling on the same feature set.

1. Only change: retarget `y_t = 1 if close_{t+5} > close_t else 0`
2. All other settings from Experiment A
3. Re-run Optuna (horizon change reshapes the loss surface)

**Expected:** BiGRU ~71–73 %, stacker ~73–75 %. Wall time ~6 h on A100.

### Experiment C — regime-conditional + walk-forward (optional, time permitting)

Goal: address fold 5's post-Ukraine regime shift.

1. Classify each training window into low-vol / high-vol by 20d realized volatility of log-returns
2. Train separate BiGRU per regime; gate at inference based on most recent window's volatility
3. Switch from static 5-fold TS-CV to walk-forward with monthly refit
4. Same features, horizon, ensemble as Experiment A or B (pick one)

**Expected:** +1–2 pp over the corresponding Experiment A/B headline, concentrated on post-2022 folds. Wall time ~8 h.

### Projected stacked headline

```
Experiment A (daily):      52.2 % → ~54–55 %
Experiment B (5-day):      52.2 % → ~73–75 %
Experiment C (+regime):    +1–2 pp on fold 5 specifically
```

---

## Critical files to modify

All changes land in a new notebook to preserve v2.3 as frozen submission artifact:

- **New notebook:** `phase1_v2_no_constraint.ipynb` (fork of `phase1_v2_classification_all_models.ipynb`)
- **Pipeline modules reused unchanged:** FRED-MD loader, `BaseForecastModel`, per-fold `StandardScaler`, the 3 RNN model classes, Optuna objective wrapper, threshold-tuning utility
- **New utility files** (if kept out of the notebook for cleanliness):
  - `feature_engineering.py` — Tier 2 + Tier 3 feature builders
  - `training_utils.py` — focal loss, temporal mixup, MC Dropout wrapper, snapshot-ensemble LR scheduler
  - `regime_utils.py` — volatility-regime classifier (only for Experiment C)

- **Artifact folder:** `phase1_v2_no_constraint_artifacts/` — summary CSV, per-fold seed-spread diagnostics, Optuna trial histories, confusion matrices, stacker weights

## Functions / utilities to reuse

From the existing v2.3 notebook:

- `load_fredmd_panel()` — 31-variable loader with t-codes and 1-month lag. **Unchanged.**
- `build_rolling_window()` — 30-day window builder. Parameterize window length for Tier 3.5.
- `BaseForecastModel` — abstract class all 3 RNN architectures inherit. **Unchanged.**
- `optuna_objective()` — wraps per-fold AUC. Extend with widened search space, reuse per-fold loop intact.
- `tune_threshold_on_train()` — train-prob threshold sweep. **Unchanged.**
- `seed_ensemble_refit()` — the v2.3 5-seed final refit routine. **Unchanged; just re-enabled for Experiment A/B.**

## Verification

End-to-end correctness checks before trusting any headline number:

1. **Leakage audit** — for every new feature (Tier 2 / Tier 3), assert `feature_t` is computed only from data `≤ t − 1`. Concrete test: inject a spike at time `t*` into the raw price series; confirm no feature at `t < t*` changes. Run before any training.
2. **Per-fold feature-stats sanity** — print per-fold train mean/std of each new feature; flag folds where a feature has zero variance (e.g. Hurst on a constant regime).
3. **Baseline parity run** — Experiment A run with Tier 2/3 features disabled must reproduce v2.3's 0.522 BiGRU accuracy within 0.3 pp on the same seeds. If not, the pipeline has drifted — halt and debug.
4. **Horizon sanity** — Experiment B must show BiGRU accuracy monotonically higher on 5-day than daily. If not, the target construction is broken.
5. **Seed-spread diagnostic** — print per-fold std of the 5 single-seed accuracies before ensembling. Expected range [0.005, 0.020]. If <0.003, the 5 runs are near-identical (seed control broken); if >0.03, the model is under-trained.
6. **OOF-stacker leakage check** — confirm the logistic stacker in 5.4 is fit per-fold using only that fold's training OOF preds, never val preds.
7. **Conviction-gate curve (Tier 1.4)** — plot accuracy vs prediction coverage (% of days kept). Report 3 points: 100 %, 75 %, 50 % coverage. Prevents cherry-picking a single coverage level.
8. **Optuna over-search diagnostic** — plot trial-number vs best-AUC-so-far. If curve plateaus before trial 50, 100 was wasted; if still climbing at 100, expand further.

## Post-plan file operations

- Plan file after write: `/Users/np3129/.claude/plans/can-you-write-all-partitioned-kite.md`
- Target destination requested by user: `/Users/np3129/Documents/AI_ML_Quants_VIP/docs/phase1 v2 no constraint plan.md`
- These two operations (move + rename) cannot be executed in plan mode; they require user approval via ExitPlanMode.
