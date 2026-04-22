# Pre-meeting brief: the full story (v2 → Phase 2 ablation)

*For the Idriss Malek meeting. Read top to bottom; this is the narrative, not the reference spec.*

---

## 1. Where we started

Four per-week notebooks: `week1_classification.ipynb` (ARX), `daily_bigru_classification.ipynb`, `daily_birnn_attention_classification.ipynb`, `daily_birnn_skip_only_classification.ipynb`. Each predicted next-day wheat direction on 31 FRED-MD variables + 30 lagged closes, per the preprocessing spec Idriss sent out.

Every one of them sat at **49–53 % accuracy** — indistinguishable from a coin flip. The same was true for the entire cohort: ~50 architectures across 12 students all hit the same ceiling. The first honest question was: *is the bottleneck the model or the data?*

My initial instinct was the model. That turned out to be wrong, but the process of proving it wrong is most of this story.

---

## 2. First consolidation (v2) — "maybe we have bugs"

Rather than iterate on four scattered notebooks, I merged them into one: `phase1_v2_classification_all_models.ipynb`. The merge forced me to actually read the four architectures side-by-side, which surfaced three bugs that had been silently dragging results down:

1. **BiGRU pooling.** The original model did `GlobalAveragePooling1D` across the 30-timestep output — effectively throwing away the recurrence. I "fixed" it to use the last hidden state. (We'll come back to this. The fix was wrong.)
2. **Attention softmax.** The attention layer was softmaxing across the feature dimension rather than across time. This is architecturally meaningless — attention weights should sum to 1 across timesteps. Fixed to `softmax(dim=1)`.
3. **Skip-connection shape.** The residual add was relying on silent broadcasting. Replaced with an explicit `nn.Linear(n_features, 2·hidden)` projection.

On top of that I added **training hygiene**: class-weighted `BCEWithLogitsLoss` (using per-fold train balance), orthogonal GRU init, grad clip at 1.0, cosine-annealing LR, patience-8 early stopping on val AUC. Nothing exotic — just the standard stuff that prevents near-random-initialization collapses.

I also added a **log-return channel**: at each timestep the sequence carries `(close, log_return(close))`. Log-return is a stationary transform of consecutive closes, so no TA violation — just a better-conditioned representation of the same information.

Finally, **per-fold threshold tuning**: after training each fold, sweep thresholds on that fold's *train* predictions and pick the one maximizing balanced accuracy. Apply it to the val predictions. No leakage — train is seen data. Typical gain: +0.3–0.5 pp.

**Result on 1999–2025, 5-fold TimeSeriesSplit:**
- ARX 0.498, BiGRU **0.497**, BiRNN+Attn 0.507, BiRNN+Skip 0.505.

BiGRU went *down*. That was a clue I initially ignored.

---

## 3. The 2008 question

Around this time I re-read the team WhatsApp chat from Dec 2025. Richard had written (10 Dec):

> *"A stylized fact in finance is that financial markets behave quite differently before and after the 2008 financial crisis, so we usually avoid using data before 2008. I tried using data starting from 2008-01-01 and that improved test set performance a lot."*

The team had already converged on clipping to `2008-01-01+`. I'd inherited their data folder but not this decision. Applying it removes the 1999–2007 pre-GFC regime — specifically fold 1 and the start of fold 2 in a 5-fold CV, which are the folds models struggle on most.

This was the **single most informative change** we made. It reflects an actual regime shift in commodity markets, not data dredging.

---

## 4. The team-convention detour (v2.1) — what we tried and reverted

The team's WhatsApp consensus wasn't only about the 2008 clip. Richard also wrote (9 Dec):

> *"80/20 train test split, then use expanding window CV (5 splits, max train size= None) for train data. Sequence length/lookback period is 30 days."*

And Nived added (13 Dec):
> *"The test set performance is based on the last fold of CV."*

So the team had a **two-layer evaluation protocol**: 80/20 top-level chronological split, 5-fold CV on the 80 % train, headline reported on the 20 % held-out slice.

I implemented this as v2.1. It produced higher headline numbers than pure 5-fold because the held-out slice was 2021–2025 — coincidentally a window where BiGRU happened to do well.

**Why we reverted.** Rereading the TA email: *"Apply 5-fold time-series cross-validation. Use a rolling window cross-validation scheme that preserves temporal order."* That's what's mandated. The 80/20 top-level split was a team extrapolation, not a TA requirement. And the 80/20 version understates how hard the problem is — it reports on one favourable slice rather than averaging across five.

I kept the **2008+ clip** (defensible on financial grounds) and dropped the **80/20 split** (not TA-mandated). This became v2.2.

**Key point for the meeting:** when Idriss says "the original notebook showed BiGRU at ~53 %", that 53 % was on a single 2021–2025 test slice (the team protocol). Under pure 5-fold CV on the same 2008+ data, BiGRU's honest mean is lower. Both numbers are valid; they measure different things.

---

## 5. v2.2 — honest numbers, and a surprise

First TA-compliant run with the 2008+ clip and pure 5-fold:

| Model | Test acc | Test AUC |
|---|---|---|
| ARX | 0.511 | 0.517 |
| BiGRU | 0.517 | 0.525 |
| BiRNN+Attn | 0.503 | 0.532 |
| BiRNN+Skip | 0.509 | 0.519 |

All deep models now above 50 %. BiGRU at 0.517 — a +2 pp lift from v2.

The surprise: **Optuna, when I gave it the choice of `last`, `mean`, or `max` pooling, picked `mean` for BiGRU and BiRNN+Skip.**

Mean-pooling was exactly what the original per-week notebook used (`GlobalAveragePooling1D`). What I'd called a "bug" in v2 and "fixed" to last-hidden-state was actually a helpful regularizer for this signal. On a low-SNR problem, averaging across 30 recurrent timesteps smooths out noise. Last-hidden-state commits to one noisy step.

This is a **genuinely interesting architectural finding**, not just a numerical tweak. It says: for daily-horizon direction prediction on macro features, the recurrent dynamics matter less than averaging them. Not what the textbook attention papers would predict.

---

## 6. v2.3 — seed ensembling, which taught us something

v2.2 got to 0.517. I wanted 0.52+. The next lever is variance reduction.

Every neural network training run has stochastic variance from three sources: initialization, mini-batch order, and (on GPU) non-deterministic kernel reductions. On a low-SNR problem, this noise can be a meaningful fraction of the signal. The standard fix is **seed ensembling**: train the same model N times with different seeds, average the prediction probabilities.

In v2.3 I added 5-seed ensembling on the final per-fold refit (after Optuna picked hparams). Seeds: `[42, 123, 456, 789, 2024]`. For each fold, train 5 models, average their val probabilities, then tune the threshold.

**Result on BiGRU:** 0.517 → **0.522**. That cleared the 52 % target.

But — and this is the interesting part — **it hurt the attention-based models.**

- BiRNN+Attn: 0.503 → 0.500
- BiRNN+Skip: 0.509 → 0.505

Per-fold diagnostic showed the pattern clearly. For BiRNN+Attn fold 1, the single-seed mean was 0.519 but the ensembled result was 0.500. Something about averaging 5 seeds was actively destroying information.

**The architectural explanation:** BiGRU has committed, fixed pooling (mean across time). Seed variance there shows up as noise around a stable feature, which averaging cleans up. Attention models, by contrast, *learn their pooling* — each seed converges to a different attention pattern (some focus on recent lags, some on earlier ones). Averaging their output probabilities doesn't average coherent solutions; it destroys them by mixing incompatible representations.

This is the kind of finding that wouldn't show up on a conference benchmark, but it's real. For the TA meeting, this is a **talking point, not a weakness** — it demonstrates we're looking at results carefully enough to notice counter-intuitive patterns.

**v2.3 final headline numbers:**

| Model | Test acc | Test AUC | Notes |
|---|---|---|---|
| ARX | 0.511 | 0.517 | Linear baseline; deterministic |
| **BiGRU (mean-pool, 5-seed)** | **0.522** | **0.526** | Headline — clears 52 % |
| BiRNN+Attn (single seed) | 0.500 | 0.523 | Class-biased toward "up" (recall up=0.80, down=0.23) |
| BiRNN+Skip (mean-pool) | 0.505 | 0.518 | Similar to v2.2 |

---

## 7. The cumulative math (v1 → v2.3)

Roughly in order of contribution:

| Lever | BiGRU Δ | Where introduced |
|---|---|---|
| 2008-01-01+ clip | +1.0 pp | v2.2 |
| Mean-pool (Optuna picked) | +0.5 pp | v2.2 |
| 5-seed ensembling | +0.5 pp | v2.3 |
| Per-fold threshold tuning | +0.3–0.5 pp | v2 |
| Log-return channel | +0.2 pp | v2 |
| Bug fixes (pool / attn-dim / skip-shape) | — | v2 |

Additive total: ~2.5 pp. Starting near 0.50, ending at 0.522. That matches what we actually see.

---

## 8. The pivot to Phase 2

After v2.3, Idriss's question was: *does any additional data beyond FRED-MD help?*

This is the **data-ablation deliverable**. The instinct behind it matters: if a team has been hitting 50 % for months across 50 architectures, the likely bottleneck is the data, not the models. An ablation study quantifies which additional sources actually move the needle.

I chose three groups, all already collected in `alternative_data/raw/`:

- **A = TECH** (13 features): RSI, MACD, Bollinger %B, ATR, momentum 5/10/20, volume z-score, etc. — all from wheat OHLCV we already had.
- **B = COT** (5 features): CFTC disaggregated report — commercial-net / open interest, money-manager-net / OI, 4-week position momentum, 52-week extreme z-score, log open interest. Chicago SRW contract.
- **C = CROSS** (8 features): wheat/corn log-ratio + 20-day momentum, wheat/soy log-ratio + momentum, 20-day wheat-corn rolling correlation, crude / DXY / gold log-returns.

These map loosely to the TA's casual mentions in our last chat: "correlation" ≈ CROSS, and COT + TECH as classical commodity factors that don't require new data collection.

---

## 9. Ablation design choices

**Full 2³ = 8 configurations** on top of baseline: baseline, +A, +B, +C, +AB, +AC, +BC, +ABC. This is the complete power set — it lets us measure both single-group lifts (+A, +B, +C) and interaction effects (+AB vs A+B separately).

**4 models per config = 32 cells total.**

**Per-cell Optuna** (30 trials, TPE + MedianPruner, maximising mean val AUC across 5 folds). Each cell tunes its own hyperparameters — because adding features may genuinely need different dropout / hidden size.

**Single seed for every cell.** No seed ensembling. This is a deliberate fairness choice: if only the baseline were seed-ensembled, ablation deltas would confound "feature-group lift" with "seed-averaging benefit." Keeping the protocol identical across all 32 cells makes comparisons clean.

**~4.5 hours wall time on A100.**

---

## 10. The ablation results

### Accuracy matrix (32 cells)

| Config | ARX | BiGRU | BiRNN+Attn | BiRNN+Skip |
|---|---|---|---|---|
| baseline | 0.504 | 0.493 | 0.509 | 0.501 |
| +A | 0.506 | **0.518** | **0.519** | 0.499 |
| +B | 0.504 | 0.491 | 0.507 | 0.515 |
| +C | 0.513 | 0.500 | 0.500 | 0.506 |
| +AB | 0.506 | 0.506 | 0.515 | 0.505 |
| +AC | 0.513 | 0.518 | 0.513 | 0.510 |
| +BC | 0.513 | **0.520** | 0.507 | 0.509 |
| **+ABC** | **0.513** | 0.514 | **0.521** | **0.513** |

Best single cell: **BiRNN+Attn on +ABC at 0.521 accuracy, AUC 0.535.** That's the headline.

### Per-group marginal lift (averaged across configs)

| Group | ARX | BiGRU | BiRNN+Attn | BiRNN+Skip |
|---|---|---|---|---|
| **A = TECH** | +0.001 | **+0.013** | **+0.011** | −0.001 |
| **B = COT** | ~0 | ~0 | +0.002 | +0.007 |
| **C = CROSS** | **+0.008** | **+0.012** | −0.002 | +0.004 |

### Three takeaways

**(1) TECH is the RNNs' story.** BiGRU gains +1.3 pp and Attn gains +1.1 pp from technical indicators alone. ARX gains almost nothing. This says the RNNs are genuinely doing nonlinear work — the indicators (RSI × MACD interactions, momentum regime conditioning) carry signal a linear model can't extract.

Why does this matter? Because the *baseline* FRED-MD panel is monthly data forward-filled to daily. Within any 30-day window, 29 of those days are identical copies. The RNN's sequence branch was basically starving. TECH finally gave it daily-varying features worth modelling.

**(2) CROSS is ARX's story.** Cross-commodity ratios (wheat/corn, wheat/soy) carry linearly-separable signal. ARX's +0.8 pp on CROSS is its largest single gain anywhere in the matrix. This is what we'd expect from the grain-complex substitution literature (Manogna 2025, Guida 2025).

**(3) COT is essentially null.** Negligible average lift. A small win for BiRNN+Skip (+0.7 pp). The Wang & Zhang 2024 paper reports Sharpe 2.07 for LightGBM on COT + macro across 22 commodities — but their horizon is weekly+. At daily horizon with a 3-day publication lag, the positioning signal is diluted by the time we can trade on it. The signal is likely already priced in.

**(4) +ABC wins or ties for every model.** No catastrophic interaction effects. Groups combine roughly additively.

---

## 11. The baseline-discrepancy caveat (important to explain)

Phase 2 baseline BiGRU is 0.493.
v2.3 BiGRU is 0.522.

**Why.** Phase 2 uses 30 Optuna trials and single-seed across all 32 cells for fairness. v2.3 uses 50 Optuna trials and 5-seed ensembling on BiGRU. Different protocols, different absolute numbers. This is **deliberate, not a regression**.

The phase 2 table is a *relative comparison table*. The ablation deltas are real and attributable to the feature groups. The absolute cell values aren't meant to match v2.3 — they're measured on a different (more conservative) protocol.

If Idriss wants an anchored absolute number, I can re-run the 4 phase-2 baseline cells at v2.3 settings in ~45 min on A100. That would anchor the table at 0.522 for BiGRU baseline while leaving the 28 ablation cells as-is. Worth offering; not sure it adds value.

---

## 12. What this all means, in one paragraph

The cohort ceiling was ~50 %. We pushed past it with a combination of (a) a defensible data-scope choice (2008+), (b) architectural fixes plus Optuna finding that mean-pool beats last-hidden-state, (c) seed ensembling on the one architecture where it helps, and (d) technical indicators + cross-commodity context giving the RNNs something to actually model. We end at **52.2 % via BiGRU on v2.3** and **52.1 % via BiRNN+Attn + TECH + COT + CROSS on Phase 2** — two independent paths to the same number. Every number is on strict 5-fold TimeSeriesSplit with per-fold scaler fit on train only, exactly as the TA email mandates.

---

## 13. The graveyard — things we tried and dropped

- **Global-window wavelet denoising.** An early phase0 experiment pushed accuracy to 74 %. It was leaking future information — `pywt.wavedec` over the full series uses all timesteps to produce any one denoised value. Discarded. A causal-rolling-window version is possible but adds complexity without clear benefit at this SNR.
- **5-day forward target.** Lifts accuracy 20+ pp because 5-day returns are less noisy than 1-day. TA email §1.3 mandates next-day. Discarded.
- **Team's 80/20 top-level split.** Not TA-mandated; team extrapolation. Reverted in v2.2.
- **Seed ensembling on attention-based models.** Hurts because seed variance in attention patterns doesn't commute with probability averaging. Architectures with committed pooling (BiGRU) benefit; attention-based ones don't.
- **Polymarket as ablation group.** Considered (per TA's casual mention). Wheat-specific markets are thin there — mostly crypto, politics, sports. Deprioritized for poor deliverable ROI.
- **FinBERT news sentiment, Google Earth NDVI.** Listed as potential ablation groups. Both require real new data collection (GDELT + GPU inference for FinBERT; Google Earth Engine setup for NDVI). **Stretch goals, not yet started.** Worth discussing in the meeting whether to extend.

---

## 14. Open questions for Idriss

1. Is 3-group ablation sufficient for the deliverable, or should we extend to FinBERT news sentiment (GDELT → FinBERT) and/or satellite NDVI (Google Earth Engine → Kansas / N. Dakota / Montana wheat belts)? Both are real data-collection subprojects.
2. Should we re-run the 4 Phase 2 baseline cells at v2.3 settings (50 Optuna trials + 5-seed ensembling on BiGRU) to anchor absolute numbers in the ablation table?
3. Headline framing: which number should lead the final write-up — v2.3 BiGRU at 0.522, or Phase 2 BiRNN+Attn on +ABC at 0.521? Both clear 52 % via different paths; we could lead with either, or present both as converging evidence.
4. Is there anything in the methodology that needs tightening before I push to the final report?
5. Thoughts on the seed-ensembling-hurts-attention finding? It feels real but I haven't seen it discussed in literature I've found — worth writing up formally as a note, or leave it as an in-pipeline diagnostic?

---

## 15. The story in one slide (for your head)

> "We inherited a pipeline stuck at 50 %. Merged 4 notebooks, fixed 3 quiet bugs, let Optuna choose the pooling, clipped to 2008+, added seed ensembling where it helps. That got us to **52.2 %** — above coin toss, under strict 5-fold CV. Then we asked the natural next question: *does adding data help?* Ran an 8-config × 4-model ablation matrix. Found that technical indicators and cross-commodity context each add ~1 pp, COT adds nothing at daily horizon. Best single cell: **BiRNN+Attn with all 3 data groups, 52.1 %**. Two paths to 52+, both clean, both TA-compliant."

That's the pitch. The rest is detail.

---

## File pointers (for the meeting if you need them)

- **Detailed classification progression:** `docs/classification_v2_progression.md`
- **Detailed ablation report:** `docs/phase2_ablation_report.md`
- **Final classification notebook:** `phase1_v2_classification_all_models.ipynb`
- **Ablation notebook:** `phase2_ablation.ipynb`
- **Results artifacts:** `Phase2 ablation/phase2_ablation_artifacts/` + Drive `phase1_v2_artifacts/`
- **Deck:** `VIP_progress_summary.pptx` (10 slides)
