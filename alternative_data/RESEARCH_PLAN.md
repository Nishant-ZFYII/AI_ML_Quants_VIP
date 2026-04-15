# Research Roadmap: Wheat Futures Classification with Alternative Data

## Context

**Problem:** All models across the lab (~50 architectures, 12 students) achieve ~50% accuracy on daily wheat direction classification using FRED-MD + price lags. The data is the bottleneck, not the models.

**Lab structure:** Each student implements assigned model architectures (RNNs, LSTMs, GRUs, CNNs, Transformers, tree-based, etc.) on a SHARED data pipeline. The `BaseForecastModel` abstract class ensures all models use the same input format.

**Nishant's deliverables:**
1. **Dataset (shared)** — Collect, clean, and integrate alternative data into the shared pipeline. All 50+ models across 12 students benefit.
2. **Investment strategy (individual)** — Build a trading strategy using predictions from assigned models (ARX, BiRNN+Attention, BiGRU+Attention+Skip, RCNN+Self-Attention).

**Goal:** Enrich the shared feature set so that ANY model in the lab can improve, then use those improvements to build a profitable trading strategy.

**Scope:** Ongoing VIP lab project — no fixed timeline.

---

# PART A: SHARED DATASET

The alternative data integrates into the existing pipeline as additional feature columns alongside the 31 FRED-MD variables and 30 price lags. The `(N, 30, F)` input tensor simply grows from F=32 to F=32+new_features. Every model in the lab automatically gets the enriched data.

---

## Phase 0: Extract Signal from Existing Data (No New Collection)

**Objective:** Before collecting anything new, squeeze maximum value from the OHLCV data we already have.

### Technical Indicators

| Indicator | What it captures | Literature |
|-----------|-----------------|-----------|
| Momentum (5/10/20-day returns) | Trend strength & direction | Strongest predictor across all horizons (Guida 2025, CFA Institute) |
| RSI (14-day) | Overbought/oversold conditions | Classic reversal signal |
| MACD (12/26/9) | Trend acceleration/deceleration | Trend-following momentum |
| Bollinger Band %B | Price relative to volatility bands | Mean-reversion signal |
| ATR (Average True Range) | Volatility regime | Risk/position sizing signal |
| Volume + Volume-weighted change | Conviction behind price moves | Confirms or contradicts direction |

These are computed from the Investing.com CSVs (which already contain Open/High/Low/Close/Volume). Adds ~8-10 feature columns. Zero new data collection.

### Wavelet Denoising
Apply Discrete Wavelet Transform to price series before feature computation. Lopez Gil et al. (2024) lifted daily direction F1 from ~50% to ~73% with wavelet denoising. Uses `pywt` library.

### XGBoost / LightGBM as Model Baselines
Tree-based models often outperform RNNs on tabular features (Wang & Zhang 2024). Add these to the lab's model roster to test whether model choice matters as much as data choice.

**Expected impact:** AUC 0.52-0.55. Confirms marginal signal exists, motivates external data.

---

## Phase 1: CFTC Commitment of Traders (COT)

**What it is:** Weekly report of who is long/short on wheat futures — hedge funds, farmers, swap dealers.

**Why chosen:** Wang & Zhang (2024), J. Futures Markets — LightGBM on COT + macro data achieved **Sharpe 2.07, 35% annualized return** across 22 commodities. Hedging pressure ranked among top features via Shapley analysis. Zhu et al. (2024) confirmed XGBoost + COT factors outperform linear models.

**Source:** https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm
- Free CSV download, no API key
- Weekly (Tuesday snapshot, Friday release)
- History: 2006–present (disaggregated), 1986–present (legacy)
- Filter: CBOT Wheat, commodity code 001602

**Features added to shared pipeline:**

| Feature | What it captures |
|---------|-----------------|
| Hedging pressure: `commercial_net / open_interest` | Are farmers/grain elevators bearish? |
| Speculator sentiment: `noncommercial_net / open_interest` | Are hedge funds bullish or bearish? |
| Position change momentum (4-week rolling) | Is positioning shifting? |
| Extreme positioning z-score (52-week window) | Are we at an extreme? (contrarian signal) |
| Open interest level | Overall market participation |

Forward-fill weekly → daily with 1-week publication lag.

**Expected impact:** AUC 0.55-0.60

---

## Phase 2: Cross-Commodity Prices

**What it is:** Daily prices of commodities that directly influence wheat.

**Why chosen:**
- Manogna et al. (2025), Scientific Reports: GRU best for ag commodities with cross-commodity features
- Guida (2025), CFA Institute: Cross-commodity momentum among strongest predictors
- These are the supply/demand linkages the survey paper (Section 2.2.4) already identified but never implemented

**Source:** Yahoo Finance via `yfinance` Python library — free, no API key, daily

| Asset | Why it matters for wheat |
|-------|------------------------|
| Corn (ZC) | Feed substitution — wheat-corn spread drives demand shifts |
| Soybeans (ZS) | Competing crop for acreage allocation |
| Crude Oil (CL) | Input cost: fertilizer, transportation, drying |
| US Dollar Index (DX) | Wheat traded globally in USD — strong dollar = lower demand |
| Gold (GC) | Inflation/safe-haven proxy; correlated in crisis periods |

**Features added to shared pipeline:**

| Feature | What it captures |
|---------|-----------------|
| Wheat-corn price ratio + 20-day momentum | Feed substitution pressure |
| Wheat-soybean price ratio | Acreage competition |
| Rolling 20-day cross-correlations | Are commodities moving together or diverging? |
| Relative strength (wheat return minus corn/soy) | Is wheat outperforming or lagging? |
| DXY level change | Dollar impact on global wheat demand |

**Expected impact:** AUC 0.57-0.62

---

## Phase 3: Google Trends + News Sentiment

### 3a. Google Trends

**What it is:** Weekly search volume for wheat-related terms.

**Why chosen:** Gutierrez & Taieb (2021), Economics of Agriculture — Granger causality confirmed from Google search volume → corn/wheat prices. Built a profitable attention-driven trading system.

**Source:** `pytrends` Python library (free, weekly)

**Search terms:** "wheat price", "wheat futures", "drought", "crop failure", "USDA report"

**Features:** Normalized search volume, week-over-week change, spike indicator (z-score > 2 over 52 weeks)

### 3b. News Sentiment

**What it is:** NLP-derived sentiment scores from wheat/agriculture news articles.

**Why chosen:** Ghali et al. (2024), arXiv — Dual-stream LSTM + news achieved **AUC 0.94, Accuracy 91%**. Critically: **removing news collapsed AUC from 0.94 to 0.46**. News is the single highest-impact feature in the literature. Kaplan et al. (2023) showed FinBERT-based sentiment outperformed RavenPack by 8.1x.

**Source options:**

| Source | Cost | Coverage | Best for |
|--------|------|----------|----------|
| GDELT (gdeltproject.org) | Free | Daily, 1979–present | Massive coverage, coarser sentiment |
| NewsAPI + FinBERT | Free tier | 30-day history | Higher quality sentiment scoring |
| Kaggle commodity news datasets | Free | Varies | Historical backfill |

**Features added to shared pipeline:**

| Feature | What it captures |
|---------|-----------------|
| Daily average sentiment (-1 to +1) | Is news about wheat positive or negative? |
| Sentiment momentum (5-day rolling change) | Is sentiment shifting? |
| News volume (articles/day) | Market attention / uncertainty level |
| Sentiment dispersion (std dev across articles) | Disagreement / confusion |
| Sentiment surprise (deviation from 20-day mean) | Sudden shift from normal |

**Expected impact:** AUC 0.62-0.68

---

## Phase 4 (Stretch): Supply-Side Data

Only if Phases 0-3 are complete and delivering results.

### 4a. USDA Crop Progress & Condition
- **Source:** quickstats.nass.usda.gov (free API key)
- **Feature:** Good+Excellent % weekly during growing season
- **Why:** Isengildina-Massa (2023) — crop report surprise explains >70% of futures price variance

### 4b. USDA WASDE Reports
- **Source:** usda.gov/ers (free Excel/CSV, monthly)
- **Feature:** Ending stocks, stocks-to-use ratio, month-over-month revision surprise
- **Why:** Milacek (2017) — profitable trading using WASDE projections. Combining WASDE + futures reduces errors by 12-16%

### 4c. Weather Data
- **Source:** Open-Meteo API (free) or NOAA Climate Data Online
- **Variables:** Temperature, precipitation, Snow Water Equivalent (SWE), Palmer Drought Severity Index
- **Coverage:** Kansas, North Dakota, Montana + Black Sea region
- **Why:** Wang et al. (2023) — SWE ranked 3rd-7th in feature importance for grain prices, 7-8 places above precipitation

### 4d. Satellite NDVI
- **Source:** NASA MODIS via Google Earth Engine (free academic) or Copernicus (free)
- **Feature:** Mean NDVI over wheat regions, anomaly from historical average
- **Why:** Teste (2025) — 24.12% annual return on wheat futures from satellite imagery. Adama et al. (2024) — NDVI + climate achieved 92.1% accuracy, AUC 0.935

---

## How the Shared Pipeline Changes

**Before (current):**
```
Input tensor: (N, 30, 32)  →  30 days × [31 FRED-MD + 1 price]
```

**After all phases:**
```
Input tensor: (N, 30, 32 + ~25 new features)

New features breakdown:
  Phase 0:  ~8-10  (technical indicators from OHLCV)
  Phase 1:  ~5     (COT positioning features)
  Phase 2:  ~8     (cross-commodity prices + ratios)
  Phase 3:  ~7     (Google Trends + news sentiment)
  Phase 4:  ~5+    (USDA + weather + NDVI, if reached)
```

All models in the lab (RNNs, LSTMs, CNNs, Transformers, tree-based, etc.) automatically receive the enriched features. No model code changes needed — only the data loader changes.

---

# PART B: INVESTMENT STRATEGY (Nishant's Individual Deliverable)

Built on top of predictions from assigned models: ARX, BiRNN+Attention, BiGRU+Attention+Skip, RCNN+Self-Attention.

---

## Phase 5: Multi-Horizon Prediction

**Why:** Weekly predictions gain 5-10 percentage points over daily (Lopez Gil et al. 2024; Guida 2025). Multi-horizon ensembles reduce model risk.

### Targets
Train each of Nishant's 4 assigned models on multiple horizons:
- 1-day direction
- 3-day direction (sign of 3-day return)
- 5-day direction (weekly)
- 10-day direction (bi-weekly)

### Ensemble
- Use predictions from all 4 models × all 4 horizons
- Stacking meta-learner (Logistic Regression on out-of-fold predictions)
- Stacking consistently outperforms individual models: 90-100% accuracy range in ensemble literature (Celik & Celik 2025)

**Expected:** 1-day AUC 0.63-0.68, 5-day AUC 0.67-0.73

---

## Phase 6: Trading Strategy Backtest

**Literature benchmarks:**
- Wang & Zhang (2024): Sharpe 2.07 with LightGBM on commodity futures
- Teste (2025): 24.12% annual return on wheat from satellite signals
- Celik & Celik (2025): GBM = "best balance of forecast precision and risk-adjusted returns"

### Signal Generation
- Use ensemble predicted probability
- High-confidence only: trade when P(Up) > 0.6 (go long) or P(Up) < 0.4 (go short)
- Skip uncertain signals → fewer trades, higher win rate

### Backtest Rules
- Walk-forward out-of-sample on test period (2022-2025)
- Transaction costs: $12.50/round-trip/contract (CBOT wheat e-mini)
- Slippage: 1 tick = $12.50
- No lookahead — same temporal discipline as classification pipeline

### Metrics

| Metric | Target |
|--------|--------|
| Annualized return | > 0% net of costs |
| Sharpe ratio | > 1.0 (good), > 1.5 (excellent) |
| Max drawdown | < 25% |
| Win rate | > 52% |
| Profit factor (gross profit / gross loss) | > 1.2 |

### Regime Analysis
Performance split by: trending vs mean-reverting periods, high-vol vs low-vol, monthly breakdown.

---

## Evaluation Framework (All Phases)

| Metric | Method | Good | Excellent |
|--------|--------|------|-----------|
| Accuracy | 5-fold TimeSeriesSplit CV | > 55% | > 62% |
| AUC-ROC | 5-fold TimeSeriesSplit CV | > 0.58 | > 0.65 |
| F1 Score | Macro-averaged | > 0.55 | > 0.62 |
| Statistical significance | DeLong test for AUC | p < 0.05 | p < 0.01 |
| Feature importance | SHAP values for tree models | Top-10 identified | Ablation per source |
| Economic value | Walk-forward backtest Sharpe | > 1.0 | > 1.5 |

**Rule:** Each phase must show measurable improvement. If not, document why and proceed.

---

## Expected Cumulative Trajectory

| Phase | Key Addition | Expected AUC | Expected Acc |
|-------|-------------|-------------|-------------|
| Current | FRED-MD + price lags only | 0.50-0.54 | ~50% |
| 0 | Technical indicators + XGBoost + wavelet | 0.52-0.55 | 51-54% |
| 1 | CFTC COT positioning data | 0.55-0.60 | 54-58% |
| 2 | Cross-commodity prices | 0.57-0.62 | 55-60% |
| 3 | Google Trends + News sentiment | 0.62-0.68 | 58-64% |
| 4 | USDA + weather + NDVI (stretch) | 0.65-0.72 | 60-67% |
| 5-6 | Multi-horizon ensemble + strategy | Sharpe > 1.0 | — |

---

## Key Literature References

1. Wang & Zhang (2024), J. Futures Markets — COT + LightGBM, Sharpe 2.07
2. Zhu et al. (2024), Pacific-Basin Finance J. — XGBoost for commodity factors
3. Wang et al. (2023), J. Sustainable Agric. & Environ. — Snow Water Equivalent for grain prices
4. Bora & Katchova (2024), Agric. Finance Review — LSTM beats USDA at 1-4yr horizons
5. Teste (2025), PhD Thesis — Satellite NDVI → 24.12% return on wheat
6. Adama et al. (2024), Int. J. Digital Earth — NDVI + climate, 92.1% accuracy
7. Ghali et al. (2024), arXiv — News sentiment AUC 0.94; removal → AUC 0.46
8. Kaplan et al. (2023), Cognitive Computation — CrudeBERT+ 8.1x improvement
9. Gutierrez & Taieb (2021), Economics of Agriculture — Google Trends → commodity prices
10. Deng et al. (2023), N. Am. J. Econ. & Finance — XGBoost + sentiment features
11. Manogna et al. (2025), Scientific Reports — GRU best for 23 ag commodities
12. Isengildina-Massa (2023) — Crop report surprise → >70% price variance
13. Lopez Gil et al. (2024), arXiv — Wavelet denoising, xLSTM-TS F1=73%
14. Guida (2025), CFA Institute — Momentum strongest predictor; multi-horizon ensembles
15. Celik & Celik (2025), Borsa Istanbul Review — Hybrid ARIMA+LSTM+XGBoost
16. Oktoviany et al. (2021), Decisions in Econ. & Finance — K-means + classification for futures
17. Hao (2025), J. Futures Markets — CNN on candlestick images
18. Cao (2022), J. Futures Markets — USDA announcement surprise effects

---

## Critical Files

- `/Users/np3129/Documents/AI_ML_Quants_VIP/VIP_Abstract_Class.ipynb` — base class all models inherit
- `/Users/np3129/Documents/AI_ML_Quants_VIP/daily_bigru_classification.ipynb` — reference pipeline
- `/Users/np3129/Documents/AI_ML_Quants_VIP/classification/Copy_of_week1_classification.ipynb` — ARX baseline with full data pipeline
- `/Users/np3129/Documents/AI_ML_Quants_VIP/create_presentation.py` — presentation builder
- `/Users/np3129/Documents/AI_ML_Quants_VIP/Agriculture Markets.xlsx` — lab model assignments

## Verification

After each phase:
1. Run notebooks end-to-end without errors
2. Compare 5-fold CV metrics against previous phase
3. SHAP analysis to confirm new features contribute
4. Ablation: remove feature group, confirm AUC drops
5. After Phase 6: verify backtest equity curve, Sharpe > 1.0 after costs
