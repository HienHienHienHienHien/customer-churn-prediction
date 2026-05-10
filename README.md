# Customer Churn Prediction and Campaign Analytics for Music Subscription Platforms

> **Course:** Data-Driven Marketing | **Institution:** National Economics University
> **Class:** DSEB 65B | **Supervisor:** Dr. Nguyen Tuan Long | **Hanoi, April 2026**

---

## Group 6

| Name | Student ID |
|---|---|
| Doan Tung Lam | 11230555 |
| Nguyen Manh Cuong | 112305xx |
| Tran Thu Hien | 11230534 |
| Do Ha Linh | 11230557 |
| Pham Minh Bao Ngoc | 112305xx |

---

## Project Overview

This project builds an end-to-end **churn prediction and retention optimization system** for **KKBox**, a music streaming platform operating in Taiwan, Hong Kong, Japan, and Southeast Asia. The dataset comes from the [WSDM Cup 2018 Kaggle competition](https://www.kaggle.com/c/kkbox-churn-prediction-challenge).

The primary objective is not competition leaderboard performance but a **deployable business retention framework** — one that translates raw churn probabilities into per-customer voucher decisions grounded in Customer Lifetime Value (CLV) and economic break-even logic. The methodology is inspired by Bryan Gregory's first-place solution (LogLoss 0.07974).

### Business motivation

KKBox loses approximately 9% of its subscriber base every month. At a median CLV of **$1,434 per user**, every 1,000 unnecessary churners represents $1.43M of lost future revenue. The platform faces a churn rate 2–4× higher than comparable Western streaming services, driven by competitive free-tier alternatives and lower willingness-to-pay in emerging Asian markets.

---

## Dataset

**Source:** [KKBox WSDM Cup 2018](https://www.kaggle.com/c/kkbox-churn-prediction-challenge) — place raw files in `Data/` before running.

| File | Rows | Description |
|---|---|---|
| `transactions.parquet` | 547,746 | Subscriptions Jan 2015 – Jan 2017 |
| `transactions_v2.csv` | 1,431,009 | Additional transactions Jan – Mar 2017 |
| `members_v3.csv` | 6,769,473 | User demographics and registration |
| `user_logs.parquet` | 106,543 | Historical daily listening (22,443 users) |
| `user_logs_v2.parquet` | 396,362 | March 2017 listening (316,345 users) |

**Key data characteristics:**
- 80% of registered accounts never transacted — a large "dark user" population
- Log data covers only 1.1% of training users — most listening features are zero-filled
- Dates stored as YYYYMMDD integers requiring parsing
- 8,983 rows removed with negative expiry gaps (data entry errors)

---

## Pipeline

```
01_Preprocessing
        ↓
02_FeatureEngineering
        ↓
03_Modeling
        ↓
04_RetentionDecision
        ↓
05_AB_Testing
```

### Step 1 — Preprocessing (`01_Preprocessing.ipynb`)

Loads and cleans all five raw files. Key decisions:

- Both transaction files merged into 1,969,768 clean rows after removing corrupted entries
- Age (`bd`) clipped to [10, 80]; missing values imputed and flagged
- Log files kept as **two separate branches** — only 18% user overlap confirms they are distinct cohorts, not continuous history
- `registration_init_time` decomposed into year/month/day components

**Output:** 5 clean parquet files + preprocessing metadata JSON

---

### Step 2 — Feature Engineering (`02_FeatureEngineering.ipynb`)

Constructs 54 behavioral and transactional features using strict temporal constraints to prevent data leakage.

**Time-based split:**

| Split | Month | Users | Churn rate | Labels |
|---|---|---|---|---|
| Train | Jan 2017 | 992,931 | 6.39% | `train.csv` (official) |
| Validation | Feb 2017 | 970,960 | 8.99% | `train_v2.csv` (official) |
| Inference | Mar 2017 | 907,471 | ~6.7% (implied) | `sample_submission_v2.csv` |

**Why time-based splitting?** A random split would allow the model to learn from future behavior (e.g. a user's March transactions) to predict January churn — leaking future data into training and producing unrealistically high accuracy metrics.

**Label construction (Scala-aligned):** Each user's effective membership state is reconstructed at the cutoff using `sort → groupby().last()`. A user is labelled `is_churn = 1` if they do not renew within 30 days of expiry.

**Feature groups:**

| Group | Count | Key examples |
|---|---|---|
| Member | 10 | `registered_via`, `days_since_reg`, `registration_date_abs` |
| Transaction | 30 | `cancel_rate`, `last_is_cancel`, `auto_renew_rate`, `days_last_txn_to_expire` |
| Log | 13 | `total_secs_played`, `delta_secs_30d_vs_prior`, `days_since_last_login` |

**Two temporal engineering methods (Bryan Gregory):**
- *Relative refactoring*: dates converted to days elapsed since prediction period start — ensures features are comparable across Jan/Feb/Mar splits
- *Absolute method*: raw YYYYMMDD integer retained when calendar date carries independent signal

**Critical bug fixed:** The column `num_uniq` (unique songs played) was incorrectly referenced as `num_uniq_songs`, causing all 5 song-based features to be identically zero. This single error caused LGBM to stop at `best_iteration = 6` (out of 214) and compressed all predictions near the population mean. Fixing it improved campaign ROI from 33% to 44.7%.

**Output:** `master_model_table.parquet` (1,963,891 × 59), `inference_snapshot.parquet` (907,471 × 55)

---

### Step 3 — Modeling (`03_Modeling.ipynb`)

Three models trained and evaluated on the February 2017 validation set.

**Results:**

| Model | Val AUC | Val LogLoss | Notes |
|---|---|---|---|
| Logistic Regression | 0.702 | 0.606 | Linear baseline — confirms feature signal |
| LightGBM | 0.766 | 0.290 | Leaf-wise growth, best LogLoss calibration |
| XGBoost | 0.766 | 0.551 | Level-wise growth, Bryan's `tree_method=hist` |
| **Ensemble (95/5)** | **0.769** | — | Grid-searched optimal blend |
| After isotonic calibration | 0.769 | **0.257** | Mean prediction = 0.0899 (exact val churn rate) |

**Why LightGBM + XGBoost?** The two models use different tree growth strategies — leaf-wise (LGBM) vs level-wise (XGB) — meaning their errors are not perfectly correlated. Blending reduces variance. A grid search over blend weights in 5% steps found 95% LGBM + 5% XGB as optimal because LGBM's probability calibration (LogLoss 0.290) is far better than XGBoost's (LogLoss 0.551).

**Why isotonic calibration, not the 0.75 multiplier?** Top competition solutions multiplied predictions by 0.75 because the March test churn rate (~6.7%) is lower than the validation rate (8.99%). However, this requires knowing the test churn rate in advance — unavailable in real deployment. Isotonic regression learns the optimal mapping from raw predictions to actual outcomes using the validation set alone, producing a calibrated mean of exactly 8.99%.

**Hyperparameter tuning:** Optuna TPE sampler, 100 trials per model (best LGBM trial found at trial 98, confirming 50 trials was insufficient).

**Top features (LGBM gain %):** `registered_via` (42.0%), `registration_date_abs` (12.6%), `total_amount_paid` (9.9%), `days_since_reg` (7.3%), `last_is_auto_renew` (7.0%)

**Output:** `submission.csv` (calibrated), model artifacts, `modeling_metadata.json`

---

### Step 4 — Retention Decision (`04_RetentionDecision.ipynb`)

Translates churn probabilities into per-customer economic decisions.

**Data-driven behavioral weights** — computed from actual churn rate differences in training data (not assumed multipliers):

| Behavior | Churn (flag=1) | Weight | Interpretation |
|---|---|---|---|
| Auto-renew active | 5.1% | +0.125 | Committed subscriber |
| Recent cancellation | 14.6% | −0.225 | About to leave |
| Active, no cancel | **62.1%** | −0.250 | Passive churner — counterintuitive |
| Long plan user (≥90d) | **72.4%** | −0.250 | Promotional buyer, not loyal |

The two counterintuitive findings — long-plan users and active-no-cancel users churning far above average — are only discoverable through data-driven analysis. Assumed multipliers would have gotten both backwards.

**Quantile-based risk tiers** (adapts to any model output distribution):

| Tier | P_churn range | Users | Voucher | Base save rate |
|---|---|---|---|---|
| Stable | < Q1 = 0.026 | 173,442 | 0% | 0% |
| Medium | Q1 – Q2 = 0.040 | 210,125 | 5% | 8% |
| High | Q2 – Q3 = 0.060 | 119,384 | 10% | 15% |
| Critical | > Q3 = 0.060 | 404,520 | 20% | 7% |

**CLV formula:**
```
monthly_revenue   = avg_amount_paid / (avg_plan_days / 30)
expected_lifetime = 1 / monthly_churn_rate  =  11.1 months
CLV               = monthly_revenue × expected_lifetime
Median CLV        = $1,434  |  Max CLV = $21,881
```

**Per-customer break-even:**
```
retention_cost = min(avg_amount_paid × voucher_pct, $200)
p_threshold    = retention_cost / (CLV × adjusted_save_rate)
Send voucher only if: P_churn > p_threshold  AND  expected_profit > 0
```

**Campaign results:**

| Metric | Value |
|---|---|
| Users targeted | 84,420 |
| Campaign cost | $2,517,531 |
| Expected revenue retained | $3,643,789 |
| Net benefit | $1,126,258 |
| ROI | **44.7%** |

**Strategic segments:**

| Segment | Users | Net benefit | Action |
|---|---|---|---|
| Critical Selective | 75,787 | $1,105,002 | Max voucher (20%) |
| High (profitable) | 6,199 | $19,989 | Standard voucher (10%) |
| Medium (profitable) | 2,434 | $1,267 | Light voucher (5%) |

**Output:** `retention_decision_table.csv`, `retention_targets.csv`, `campaign_tracking_template.csv`, `retention_metadata.json`

---

### Step 5 — A/B Testing (`05_AB_Testing.ipynb`)

Two-part experiment: a retrospective backtest on February 2017 (real labels) and a Monte Carlo simulation on March 2017 (synthetic labels).

**Why February, not March?** March users' membership expiry dates fall in April 2017. Observing renewal outcomes requires April transaction data — which has 0 rows in the public dataset (confirmed: 1,025,980 memberships expire in April but outcomes are unobservable).

**Experimental design (February backtest):**

| Arm | Share | Users | Treatment |
|---|---|---|---|
| Control (C) | 20% | 163,304 | No voucher |
| Treatment A (T_A) | 40% | 326,896 | Tier-based voucher (5/10/20%) |
| Treatment B (T_B) | 40% | 327,188 | Flat 10% voucher |

Arm assignment via MD5 hash of `campaign_id:msno` — deterministic and reproducible across runs.

**Backtest results:**

| Arm | Renewal rate | Lift vs control | p-value | Significant? |
|---|---|---|---|---|
| Control | 89.34% | — | — | — |
| Treatment A | 89.33% | −0.01% | 0.560 | **No** |
| Treatment B | 89.29% | −0.05% | 0.706 | **No** |

**Why the null result is correct and informative:** Medium and High tier users renew at 99.4–99.5% naturally. No voucher can improve on 99.4% — the ceiling is 100%. The model assigned users with P_churn = 0.03–0.06 to these tiers, but their actual churn is only 0.5%. The null result is a model discrimination diagnostic, not a campaign failure.

**Monte Carlo simulation (March 2017):** Anchored to 6.7% implied March churn rate (from the leaderboard finding that top solutions multiplied predictions by 0.75). Treatment effect applied as Bernoulli draws with Critical-tier save rate = 15%.

| Seed | Simulated churn | T_A lift | p-value | Significant? |
|---|---|---|---|---|
| 42 | 6.67% | +0.96% | 0.00000 | Yes |
| 123 | 6.70% | +1.09% | 0.00000 | Yes |
| 999 | 6.70% | +1.09% | 0.00000 | Yes |
| 2017 | 6.72% | +1.09% | 0.00000 | Yes |
| 7 | 6.70% | +1.00% | 0.00000 | Yes |

Significant across all 5 seeds. Minimum viable Critical save rate: **5%** (lowest tested). The retention framework is economically sound — model discrimination is the only remaining gap.

**Output:** `ab_tracking_table.csv`, `ab_backtest_results.csv`, `ab_test_metadata.json`

---

## Key Findings

1. **Long-plan users churn at 72.4%** — users who purchase 90-day plans are promotional buyers, not loyal subscribers. Long-plan promotions may be attracting the wrong customer cohort.

2. **Active-no-cancel users churn at 62.1%** — these are passive churners approaching natural membership lapse without explicitly cancelling. They look healthy by conventional metrics but are silently departing.

3. **A single column name error (`num_uniq` vs `num_uniq_songs`) was the highest-leverage fix** — it caused LGBM convergence to jump from best_iteration=6 to 214, widened the P_churn distribution from max=0.134 to max=0.181, and improved campaign ROI from 33.0% to 44.7%.

4. **The A/B null result is a diagnostic** — the model places safe users (99.4% natural renewal) in at-risk tiers. The simulation shows the framework produces +1.05% lift when the model correctly identifies churners.

5. **ROI is 44.7% at full reach and 48% at the highest-confidence threshold** — fewer but better-targeted users produces both lower cost and higher net benefit per dollar spent.

---

## Repository Structure

```
.
├── 01_Preprocessing.ipynb
├── 02_FeatureEngineering.ipynb
├── 03_Modeling.ipynb
├── 04_RetentionDecision.ipynb
├── 05_AB_Testing.ipynb
├── Data/                        # Raw files (not tracked — download from Kaggle)
├── Models/                      # Saved model artifacts (not tracked)
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `lightgbm`, `xgboost`, `optuna`, `scikit-learn`, `pyarrow`

---

## How to Run

```bash
# 1. Place raw data files in Data/
# 2. Run notebooks in order
jupyter nbconvert --to notebook --execute 01_Preprocessing.ipynb
jupyter nbconvert --to notebook --execute 02_FeatureEngineering.ipynb
jupyter nbconvert --to notebook --execute 03_Modeling.ipynb
jupyter nbconvert --to notebook --execute 04_RetentionDecision.ipynb
jupyter nbconvert --to notebook --execute 05_AB_Testing.ipynb
```

---

## References

- B. Gregory, *Predicting Customer Churn: Extreme Gradient Boosting with Temporal Data*, arXiv:1802.03396, 2018.
- T. Chen and C. Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD 2016.
- G. Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*, NeurIPS 2017.
- T. Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework*, KDD 2019.
- KKBox and Kaggle, *WSDM Cup 2018 Churn Prediction Challenge*, 2018.
