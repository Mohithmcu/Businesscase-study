# 📦 Kestrel Home — Returns Risk Prediction & Fulfillment Screening Engine
## Task 2 V1: Kestrel Home — Returns Risk (Variant A)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20v4.7-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI%200.122-teal.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit%201.52-red.svg)](https://streamlit.io/)
[![Validation](https://img.shields.io/badge/ROC--AUC-0.7572-success.svg)](#3-evidence-that-it-works)
[![AP](https://img.shields.io/badge/Average_Precision-0.3648-blueviolet.svg)](#3-evidence-that-it-works)
[![Cost](https://img.shields.io/badge/Cost_per_Prediction-Rs_0.00-informational.svg)](#6-cost-arithmetic)

---

## 🎯 Deliverables Summary

| # | Deliverable | Location | Description |
|:---:|:---|:---|:---|
| **1** | **`predictions.csv`** | [`predictions.csv`](./predictions.csv) | 2,096 test order predictions across Q3 2026 (July to September). Continuous probability scores ($p \in [0, 1]$). |
| **2** | **A working service** | [`service.py`](./service.py) & [`app.py`](./app.py) | **Endpoint:** `POST /predict` returns score + plain-English reasons. <br>**Screen:** `GET /` on `http://localhost:8000` is an interactive screening UI calling the endpoint. Runs locally without paid keys. |
| **3** | **Evidence that it works** | [`solution/model_card.txt`](./solution/model_card.txt) | Strict time-based validation holdout (`ROC-AUC 0.7572`, `AP 0.3648`, `0.8105` on dispatch-time rows, confusion matrix, threshold sweep). |
| **4** | **One-page memo to Ritu** | Submitted via Portal / Drive | Non-technical executive memo covering **The Decision, The Number, The Rupees, and What she should do next week**. |
| **5** | **Screen recording** | Submitted via Portal / Drive | 3-minute walkthrough: What we tried, what we changed, what we threw away. No slides. |
| **6** | **The submission form** | Submitted via Portal / Drive | All 11 questions on the assessment portal answered thoroughly and truthfully. |

---

## 🚀 Quickstart: Running on a Clean Machine (Zero Paid Keys)

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/Mohithmcu/Businesscase-study.git
cd Businesscase-study

# Install dependencies (all open-source, no paid API keys required)
pip install -r requirements.txt
```

### 2. Launch the Working Screening Service & UI Screen (Deliverable 2)
```bash
python service.py
```
* **Interactive Screening Screen:** Open browser at [`http://localhost:8000/`](http://localhost:8000/)
  * Includes quick presets (High-Risk COD, Shield VIP Subscriber, Low-Risk Repeat Buyer).
  * Evaluates order and displays the return risk score, recommended action, and **bulleted reasons for Kestrel warehouse staff**.
* **JSON API Endpoint:** `POST http://localhost:8000/predict`
  * Example request:
    ```bash
    curl -X POST http://localhost:8000/predict \
      -H "Content-Type: application/json" \
      -d '{"order_id": "KO2610504", "customer_id": "KC105196", "sku": "KH-IC-03", "order_value_inr": 3521.76, "payment_mode": "prepaid_upi", "discount_pct": 13, "customer_prior_orders": 2, "customer_prior_returns": 1, "promised_delivery_days": 7, "is_shield": false, "delivery_note": "Office address, weekdays only"}'
    ```
  * Example response:
    ```json
    {
      "order_id": "KO2610504",
      "score": 0.1241,
      "risk_tier": "Medium Risk",
      "recommended_action": "PRE_DISPATCH_CONFIRMATION_CALL",
      "reasons": [
        "Customer has prior return history (1 return(s) across 2 orders, 50% return rate).",
        "Extended delivery commitment (7 days) increases buyer transit remorse.",
        "Office delivery address — risk of failed weekend delivery attempt."
      ],
      "details": {
        "order_value_inr": 3521.76,
        "payment_mode": "prepaid_upi",
        "is_shield": false,
        "policy_rule": "Standard dispatch policy"
      }
    }
    ```

### 3. Launch the Executive Analytics Dashboard (Streamlit)
```bash
streamlit run app.py
```
* Runs on [`http://localhost:8501/`](http://localhost:8501/) with interactive threshold sliders, rupee call ROI calculation, and test set exploration.

### 4. Reproduce Model Training & Predictions from Scratch
```bash
python train_model.py
```
* Re-trains the leak-free LightGBM pipeline, outputs `solution/predictions.csv`, mirrors to `predictions.csv`, and writes `solution/model_card.txt`.

---

## 🏗️ Architecture & Critical Leakage Remediation

### ⚠️ Target Leakage Audit & Fix
In historical CRM exports, `last_service_event_type` and `pickup_scheduled_at` were captured as of *export date* rather than point-of-dispatch:
* `REVERSE_PICKUP` in training had a **100% return rate** (the return event itself).
* `INSTALL_DONE` and `DEMO_DONE` had a **0% return rate**.
* Retaining `svc_none` or any service event feature leaks post-dispatch outcomes (inflating AUC to 0.86–0.99).

We permanently eliminated all service event features from the model. The production model relies strictly on authentic pre-dispatch signals (customer history, order economics, product age, pricing ratios, SLA, geography, delivery notes).

### ⚙️ October 2025 Gateway Correction
All 700 orders from October 2025 had `order_value_inr` recorded in paise (100× inflated; median value ratio was 92.0 vs 0.92 elsewhere). This was corrected by dividing October 2025 order values by 100.

---

## 📊 Evidence That It Works (And How Often It Does Not)

### 1. Validation Design
We implemented a strict **time-based temporal split** (85% historical train: 8,928 orders / 15% future holdout: 1,576 orders; cutoff `2026-04-24`). Random shuffling was strictly prohibited to prevent look-ahead bias.

* **ROC-AUC (All Validation Rows):** `0.7572`
* **ROC-AUC on Dispatch-Time (`NONE`-only) Validation Rows:** `0.8105`
* **Average Precision:** `0.3648` (**3.2× lift** over the 11.3% base rate).
* **Log-Loss:** `0.3026`.

### 2. Confusion Matrix on Validation Holdout (1,576 Orders, Threshold = 0.15)

```
                        Predicted: No Return    Predicted: Return
Actual: No Return              1,189 (TN)           209 (FP)
Actual: Return                    87 (FN)            91 (TP)
```
* **Precision:** `30.3%` (nearly 1 in 3 flagged orders is an actual return).
* **Recall:** `51.1%` (intercepts over half of all prospective returns).
* **Specificity:** `85.0%`.

### 3. Empirical Threshold Sweep Across Business Operating Points (700 Orders / Month)

| Threshold | Precision | Recall | F1 Score | % Orders Flagged | True Pos (TP) | False Pos (FP) | False Neg (FN) | Calls / Mo | Returns Prev / Mo | Net P&L Savings / Mo |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.08** | 18.7% | 77.0% | 0.301 | 46.5% | 137 | 596 | 41 | 326 | 21.3 | Rs 9,823 |
| **0.10** | 22.2% | 66.9% | 0.333 | 34.1% | 119 | 418 | 59 | 238 | 18.5 | Rs 10,525 |
| **0.12** | 25.3% | 59.6% | 0.355 | 26.6% | 106 | 313 | 72 | 186 | 16.5 | Rs 10,561 |
| **0.15** | 30.3% | 51.1% | 0.381 | 19.0% | 91 | 209 | 87 | 133 | 14.1 | **Rs 10,260** |
| **0.18** | 33.2% | 42.7% | 0.373 | 14.5% | 76 | 153 | 102 | 102 | 11.8 | Rs 8,999 |
| **0.20** | 35.6% | 41.0% | 0.381 | 13.0% | 73 | 132 | 105 | 91 | 11.3 | Rs 8,943 |
| **0.25** | 40.4% | 32.0% | 0.357 | 8.9% | 57 | 84 | 121 | 63 | 8.9 | Rs 7,364 |
| **0.30** | 44.7% | 23.6% | 0.309 | 6.0% | 42 | 52 | 136 | 42 | 6.5 | Rs 5,624 |

---

## 💰 Operational Economics & Policy

Using the cost figures from Kestrel's Operations Policy (§4 & §7):
* **Cost of an unmanaged return (§4):** **Rs 1,150** (reverse pickup, inspection, repacking, markdown).
* **Cost of a pre-dispatch confirmation call (§4):** **Rs 45** per completed call.
* **Pre-dispatch call efficacy (§7):** Confirmation calls prevent **~35% of returns** on called orders with **zero cancellation risk**.
* **Holding orders friction (§7):** Orders held over 24 hours trigger customer cancellation on **12% of held orders**.
* **Shield VIP Subscribers (§6):** Account for ~20% of orders and highest customer LTV. They receive free 30-day returns and must **never** be subjected to physical dispatch holds.

**At Kestrel's volume of 700 orders/month (Threshold 0.15):**
* Flagged for confirmation calls: ~133 calls/month.
* Returns prevented: ~14.1 returns/month.
* Gross savings: $14.1 \times \text{Rs } 1,150 = \mathbf{\text{Rs } 16,215 \text{ / month}}$.
* Net P&L savings (after deducting Rs 5,985 call costs): **~Rs 10,230 / month** ($\mathbf{\text{Rs } 1.23 \text{ Lakhs / year}}$ EBITDA gain).
* Prediction compute cost: **Rs 0.00**.

---

## 🗂️ File Directory Structure

```
Businesscase-study/
├── predictions.csv            # Official submission predictions (2,096 rows, order_id + score)
├── service.py                 # FastAPI service (POST /predict + GET / screening UI)
├── app.py                     # Streamlit Decision Support & ROI Dashboard
├── train_model.py             # End-to-end LightGBM training & prediction script
├── requirements.txt           # Project dependencies
├── README.md                  # Complete technical documentation
├── solution/
│   ├── model.joblib           # Trained LightGBM model artifact
│   ├── model_card.txt         # Detailed model card & threshold metrics
│   └── predictions.csv        # Mirrored prediction deliverable
└── .gitignore                 # Excludes raw data and assessment writeups
```

> **🔒 Submission & Compliance Notice:**
> 1. **Data Privacy (Ops Policy §10):** Proprietary transaction data (`train.csv`, `customers.csv`, `test_unlabelled.csv`, `ops-policy.pdf`) are excluded from this public repository via `.gitignore`.
> 2. **Assessment Documents:** `memo_to_ritu.md` and `submission-form.md` are submitted directly via the candidate application portal / Google Drive upload.

---

## 📋 Handoff Protocol: Three Things to Know on Monday

1. **`predictions.csv` is the final deliverable. Use threshold 0.15 for confirmation calls.** Running threshold $\ge 0.15$ through Meenal's phone confirmation queue prevents ~14 returns per month with zero cancellation friction.
2. **Do NOT re-introduce `last_service_event_type` or `pickup_scheduled_at` into the model.** Historical data contains post-dispatch events (`REVERSE_PICKUP` = 100% returns) that do not exist at dispatch time. Retraining with raw service fields re-introduces catastrophic target leakage.
3. **Protect Kestrel Shield members (`is_shield = 1`).** Never place a hard dispatch hold on a Shield member. They return more frequently (18.6% across all history; 7.8% on dispatch-time rows) because their plan guarantees free 30-day returns (Policy §6), but they buy 3+ appliances a year and represent Kestrel's highest lifetime value.
