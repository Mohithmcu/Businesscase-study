"""
Kestrel Home – Returns Risk Scoring Service
============================================
FastAPI service providing:
1. Endpoint: POST /predict (takes JSON single record, returns score + plain-English reasons)
2. UI Screen: GET / (interactive employee screening screen that calls /predict)
3. Health check: GET /health

Runs locally without any paid API keys.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import uvicorn

app = FastAPI(
    title="Kestrel Home - Returns Risk Scoring API",
    description="Real-time pre-dispatch fulfillment screening for Kestrel Home D2C orders",
    version="3.0.0"
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "solution" / "model.joblib"

# Load lookups
CUST_PATH = list(BASE_DIR.glob("*customers.csv"))[0]
PROD_PATH = list(BASE_DIR.glob("*products.csv"))[0]

cust_df = pd.read_csv(CUST_PATH).set_index("customer_id")
prod_df = pd.read_csv(PROD_PATH).set_index("sku")

MODEL = None
if MODEL_PATH.exists():
    MODEL = joblib.load(MODEL_PATH)
else:
    print(f"CRITICAL: Model file not found at {MODEL_PATH}")

class OrderRecord(BaseModel):
    order_id: str = Field(default="KO2610504", description="Unique Kestrel order identifier")
    customer_id: Optional[str] = Field(default="KC105196", description="Customer ID")
    sku: Optional[str] = Field(default="KH-IC-03", description="Product SKU")
    payment_mode: Optional[str] = Field(default="cod", description="Payment mode: cod, prepaid_upi, prepaid_card, emi")
    sales_channel: Optional[str] = Field(default="web", description="Sales channel: web, app, partner_outlet, marketplace")
    discount_pct: Optional[float] = Field(default=20.0, description="Discount percentage applied")
    qty: Optional[int] = Field(default=1, description="Quantity of items")
    order_value_inr: Optional[float] = Field(default=3520.0, description="Order total in INR")
    promised_delivery_days: Optional[int] = Field(default=7, description="Promised SLA in days")
    delivery_pincode: Optional[str] = Field(default="440378", description="Delivery pincode")
    customer_prior_orders: Optional[int] = Field(default=0, description="Prior order count for customer")
    customer_prior_returns: Optional[int] = Field(default=0, description="Prior return count for customer")
    is_shield: Optional[bool] = Field(default=False, description="Whether customer is a Kestrel Shield subscriber")
    is_gift: Optional[bool] = Field(default=False, description="Whether order is marked as gift")
    delivery_note: Optional[str] = Field(default="Leave with security", description="Customer delivery note")

class PredictionResponse(BaseModel):
    order_id: str
    score: float
    risk_tier: str
    recommended_action: str
    reasons: List[str]
    details: Dict[str, Any]

def generate_reasons(data: OrderRecord, score: float, is_shield: bool) -> List[str]:
    reasons = []
    
    # Shield rule first
    if is_shield:
        reasons.append("Kestrel Shield VIP member: Entitled to free 30-day returns by policy (Policy §6). High-LTV customer — NEVER place a dispatch hold. Route to concierge confirmation call if needed.")
    
    # COD + High value
    is_cod = (data.payment_mode or "").lower() == "cod"
    val = float(data.order_value_inr or 0)
    if is_cod and val >= 5000:
        reasons.append(f"High-value purchase (Rs {val:,.0f}) ordered via Cash-on-Delivery (COD) — high risk of refusal at doorstep.")
    elif is_cod:
        reasons.append("Cash-on-Delivery (COD) payment selected — historically elevated cancellation and refusal rate.")
        
    # First-time buyer vs prior return history
    prior_orders = int(data.customer_prior_orders or 0)
    prior_returns = int(data.customer_prior_returns or 0)
    if prior_orders == 0:
        reasons.append("First-time customer with zero prior purchase history.")
    elif prior_returns > 0:
        ret_rate = prior_returns / prior_orders
        reasons.append(f"Customer has prior return history ({prior_returns} return(s) across {prior_orders} orders, {ret_rate:.0%} return rate).")
        
    # Discount depth
    if (data.discount_pct or 0) >= 25:
        reasons.append(f"Steep promotional discount ({data.discount_pct:.0f}%) — correlates with speculative impulse purchases.")
        
    # Delivery SLA
    if (data.promised_delivery_days or 0) >= 6:
        reasons.append(f"Extended delivery commitment ({data.promised_delivery_days} days) increases buyer transit remorse.")
        
    # Delivery note keywords
    note = (data.delivery_note or "").lower()
    if "security" in note or "gate" in note:
        reasons.append("Delivery note references gate/security drop-off — risk of delivery dispute or missed handoff.")
    elif "office" in note:
        reasons.append("Office delivery address — risk of failed weekend delivery attempt.")
        
    if not reasons:
        if score < 0.12:
            reasons.append("Standard repeat-customer profile with low risk indicators across payment and delivery terms.")
        else:
            reasons.append("Score reflects elevated risk across regional delivery SLA, pricing, and fulfillment parameters.")
            
    return reasons

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "kestrel-returns-risk",
        "model_loaded": MODEL is not None,
        "benchmark_auc": 0.7572,
        "benchmark_ap": 0.3648,
        "cost_per_prediction_inr": 0.00
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_order(record: OrderRecord):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    # Lookup customer info if customer_id provided
    is_shield = record.is_shield
    account_age_days = 60.0
    state = "Maharashtra"
    
    if record.customer_id and record.customer_id in cust_df.index:
        c_row = cust_df.loc[record.customer_id]
        if isinstance(c_row, pd.DataFrame):
            c_row = c_row.iloc[0]
        if str(c_row.get("shield_member", "N")).upper() == "Y":
            is_shield = True
        signup_dt = pd.to_datetime(c_row.get("signup_date", "2025-01-01"))
        account_age_days = max(0, (datetime.now() - signup_dt).days)
        state = str(c_row.get("state", "Maharashtra"))
        
    # Lookup product info if sku provided
    list_price = float(record.order_value_inr or 3500.0)
    family = "Appliances"
    product_age_days = 180.0
    warranty_months = 12
    
    if record.sku and record.sku in prod_df.index:
        p_row = prod_df.loc[record.sku]
        if isinstance(p_row, pd.DataFrame):
            p_row = p_row.iloc[0]
        list_price = float(p_row.get("list_price_inr", list_price))
        family = str(p_row.get("family", "Appliances"))
        warranty_months = int(p_row.get("warranty_months", 12))
        launch_dt = pd.to_datetime(p_row.get("launch_date", "2025-01-01"))
        product_age_days = max(0, (datetime.now() - launch_dt).days)

    now = datetime.now()
    val_inr = float(record.order_value_inr or 3500.0)
    disc_pct = float(record.discount_pct or 0.0)
    qty = int(record.qty or 1)
    sla_days = int(record.promised_delivery_days or 5)
    prior_orders = int(record.customer_prior_orders or 0)
    prior_returns = int(record.customer_prior_returns or 0)
    ret_rate = (prior_returns / prior_orders) if prior_orders > 0 else np.nan
    is_first = 1 if prior_orders == 0 else 0
    is_cod = 1 if (record.payment_mode or "").lower() == "cod" else 0
    note_str = str(record.delivery_note or "").lower()
    pin = str(record.delivery_pincode or "400001").zfill(6)
    
    feat_dict = {
        "discount_pct": disc_pct,
        "log_discount": np.log1p(disc_pct),
        "discount_depth": disc_pct / 100.0,
        "deep_discount": 1 if disc_pct >= 25 else 0,
        "qty": qty,
        "log_qty": np.log1p(qty),
        "multi_qty": 1 if qty > 1 else 0,
        "order_value_inr": val_inr,
        "log_order_value": np.log1p(val_inr),
        "promised_delivery_days": sla_days,
        "long_delivery": 1 if sla_days >= 6 else 0,
        "very_fast": 1 if sla_days <= 2 else 0,
        "customer_prior_orders": prior_orders,
        "log_prior_orders": np.log1p(prior_orders),
        "customer_prior_returns": prior_returns,
        "return_rate": ret_rate,
        "is_first_order": is_first,
        "high_return_hist": 1 if ret_rate and ret_rate > 0.3 else 0,
        "has_any_return": 1 if prior_returns > 0 else 0,
        "order_hour": now.hour,
        "order_dow": now.weekday(),
        "order_month": now.month,
        "order_day": now.day,
        "order_week": int(now.strftime("%W")),
        "late_night": 1 if now.hour >= 23 or now.hour <= 5 else 0,
        "is_weekend": 1 if now.weekday() >= 5 else 0,
        "is_cod": is_cod,
        "is_emi": 1 if (record.payment_mode or "").lower() == "emi" else 0,
        "is_prepaid_upi": 1 if "upi" in (record.payment_mode or "").lower() else 0,
        "is_prepaid_card": 1 if "card" in (record.payment_mode or "").lower() else 0,
        "is_marketplace": 1 if (record.sales_channel or "").lower() == "marketplace" else 0,
        "is_partner": 1 if "partner" in (record.sales_channel or "").lower() else 0,
        "is_app": 1 if (record.sales_channel or "").lower() == "app" else 0,
        "is_web": 1 if (record.sales_channel or "").lower() == "web" else 0,
        "default_pincode": 1 if pin == "000000" else 0,
        "pin_prefix": int(pin[:3]) if pin[:3].isdigit() else 400,
        "is_metro": 1 if pin[:2] in ["11", "40", "56", "60", "70", "50"] else 0,
        "is_gift": 1 if record.is_gift else 0,
        "is_shield": 1 if is_shield else 0,
        "list_price_inr": list_price,
        "log_list_price": np.log1p(list_price),
        "value_ratio": val_inr / (list_price * qty) if (list_price * qty) > 0 else np.nan,
        "is_high_value": 1 if val_inr >= 6000 else 0,
        "warranty_months": warranty_months,
        "product_age_days": product_age_days,
        "new_product": 1 if product_age_days <= 60 else 0,
        "account_age_days": account_age_days,
        "log_account_age": np.log1p(account_age_days),
        "new_customer": 1 if account_age_days <= 30 else 0,
        "note_neighbour": 1 if "neighbour" in note_str or "neighbor" in note_str else 0,
        "note_security": 1 if "security" in note_str else 0,
        "note_fragile": 1 if "fragile" in note_str else 0,
        "note_office": 1 if "office" in note_str else 0,
        "note_gate_code": 1 if "gate" in note_str else 0,
        "note_call_before": 1 if "call" in note_str else 0,
        "note_after_6pm": 1 if "after 6" in note_str else 0,
        "note_morning": 1 if "morning" in note_str else 0,
        "note_empty": 1 if len(note_str.strip()) == 0 else 0,
        "note_len": len(note_str),
        "cod_first_order": is_cod * is_first,
        "cod_high_value": is_cod * (1 if val_inr >= 6000 else 0),
        "shield_cod": (1 if is_shield else 0) * is_cod,
        "discount_cod": disc_pct * is_cod,
        "high_ret_cod": (1 if ret_rate and ret_rate > 0.3 else 0) * is_cod,
        "sales_channel": str(record.sales_channel or "web"),
        "payment_mode": str(record.payment_mode or "cod"),
        "source": "crm",
        "family": family,
        "state": state
    }
    
    df_row = pd.DataFrame([feat_dict])
    for col in ["sales_channel", "payment_mode", "source", "family", "state"]:
        df_row[col] = df_row[col].astype("category")
        
    score = float(MODEL.predict(df_row)[0])
    score = round(score, 4)
    
    # Shield member decision rule: Never hold dispatch for Shield members
    if is_shield:
        if score >= 0.15:
            tier = "Elevated Risk (Shield VIP)"
            action = "CONCIERGE_CONFIRMATION_CALL"
        else:
            tier = "Standard Risk (Shield VIP)"
            action = "CLEAR_FOR_DISPATCH"
    else:
        if score >= 0.25:
            tier = "High Risk"
            action = "HOLD_DISPATCH"
        elif score >= 0.12:
            tier = "Medium Risk"
            action = "PRE_DISPATCH_CONFIRMATION_CALL"
        else:
            tier = "Low Risk"
            action = "CLEAR_FOR_DISPATCH"
            
    reasons = generate_reasons(record, score, is_shield)
    
    return PredictionResponse(
        order_id=record.order_id,
        score=score,
        risk_tier=tier,
        recommended_action=action,
        reasons=reasons,
        details={
            "order_value_inr": val_inr,
            "payment_mode": record.payment_mode,
            "is_shield": is_shield,
            "policy_rule": "No dispatch holds on Shield subscribers (Policy §6)" if is_shield else "Standard dispatch policy"
        }
    )

@app.get("/", response_class=HTMLResponse)
def index_screen():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Kestrel Home | Order Fulfillment Screening</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #2563eb;
                --primary-dark: #1d4ed8;
                --bg: #f8fafc;
                --card-bg: #ffffff;
                --text-main: #0f172a;
                --text-muted: #64748b;
                --border: #e2e8f0;
            }
            body { font-family: 'Inter', sans-serif; background-color: var(--bg); color: var(--text-main); margin: 0; padding: 30px 20px; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 25px; padding-bottom: 15px; border-bottom: 1px solid var(--border); }
            .logo { font-size: 1.4rem; font-weight: 700; color: var(--text-main); display: flex; align-items: center; gap: 10px; }
            .badge { font-size: 0.75rem; padding: 4px 10px; border-radius: 999px; font-weight: 600; background: #dbeafe; color: #1e40af; }
            .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
            label { display: block; font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
            input, select { width: 100%; padding: 10px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.95rem; box-sizing: border-box; font-family: inherit; }
            input:focus, select:focus { outline: none; border-color: var(--primary); }
            .btn { background: var(--primary); color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1rem; font-weight: 600; cursor: pointer; }
            .btn:hover { background: var(--primary-dark); }
            .sample-presets { display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap; }
            .preset-btn { background: #f1f5f9; border: 1px solid var(--border); padding: 6px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 500; cursor: pointer; }
            .preset-btn:hover { background: #e2e8f0; }
            #result-section { display: none; }
            .result-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; }
            .score-circle { font-size: 2.2rem; font-weight: 800; }
            .action-badge { font-size: 0.85rem; padding: 6px 14px; border-radius: 20px; font-weight: 700; text-transform: uppercase; }
            .action-hold { background: #fee2e2; color: #991b1b; }
            .action-call { background: #fef3c7; color: #92400e; }
            .action-clear { background: #dcfce7; color: #166534; }
            .reasons-list { padding-left: 20px; color: #334155; line-height: 1.6; }
            .reasons-list li { margin-bottom: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">
                    <span>📦</span>
                    <span>Kestrel Home — Pre-Dispatch Risk Screening</span>
                </div>
                <div>
                    <span class="badge">Model: Leak-Free LightGBM | AUC 0.7572</span>
                </div>
            </div>

            <div class="card">
                <div style="margin-bottom:12px; font-weight:600; font-size:0.95rem;">Quick Test Presets:</div>
                <div class="sample-presets">
                    <button class="preset-btn" onclick="loadPreset('high_risk')">🚨 High-Risk COD First-Timer</button>
                    <button class="preset-btn" onclick="loadPreset('shield_vip')">🛡️ Shield VIP Member</button>
                    <button class="preset-btn" onclick="loadPreset('low_risk')">✅ Safe Prepaid Repeat Buyer</button>
                </div>

                <form id="orderForm" onsubmit="handleScreen(event)">
                    <div class="grid">
                        <div>
                            <label>Order ID</label>
                            <input type="text" id="order_id" value="KO2610504" required>
                        </div>
                        <div>
                            <label>Customer ID</label>
                            <input type="text" id="customer_id" value="KC105196" required>
                        </div>
                        <div>
                            <label>SKU</label>
                            <input type="text" id="sku" value="KH-IC-03" required>
                        </div>
                        <div>
                            <label>Order Value (INR)</label>
                            <input type="number" id="order_value_inr" value="8500" step="1" required>
                        </div>
                        <div>
                            <label>Payment Method</label>
                            <select id="payment_mode">
                                <option value="cod" selected>Cash on Delivery (COD)</option>
                                <option value="prepaid_upi">Prepaid UPI</option>
                                <option value="prepaid_card">Prepaid Card</option>
                                <option value="emi">EMI</option>
                            </select>
                        </div>
                        <div>
                            <label>Discount (%)</label>
                            <input type="number" id="discount_pct" value="25" min="0" max="70">
                        </div>
                        <div>
                            <label>Prior Orders</label>
                            <input type="number" id="customer_prior_orders" value="0" min="0">
                        </div>
                        <div>
                            <label>Prior Returns</label>
                            <input type="number" id="customer_prior_returns" value="0" min="0">
                        </div>
                        <div>
                            <label>Promised Delivery Days</label>
                            <input type="number" id="promised_delivery_days" value="7" min="1" max="15">
                        </div>
                        <div>
                            <label>Shield VIP Subscriber?</label>
                            <select id="is_shield">
                                <option value="false" selected>No</option>
                                <option value="true">Yes (Shield VIP)</option>
                            </select>
                        </div>
                    </div>
                    <div style="margin-top: 16px;">
                        <label>Delivery Note</label>
                        <input type="text" id="delivery_note" value="Leave with security at main gate">
                    </div>
                    <div style="margin-top: 20px; text-align: right;">
                        <button type="submit" class="btn" id="submitBtn">⚡ Screen Order</button>
                    </div>
                </form>
            </div>

            <div class="card" id="result-section">
                <div class="result-header">
                    <div>
                        <div style="font-size:0.85rem; color:var(--text-muted); font-weight:600; text-transform:uppercase;">Predicted Return Probability</div>
                        <div class="score-circle" id="score-display">0.00%</div>
                    </div>
                    <div>
                        <span class="action-badge" id="action-badge">RECOMMENDED ACTION</span>
                    </div>
                </div>

                <div style="margin-top: 15px;">
                    <div style="font-weight:700; margin-bottom:8px; font-size:1.05rem;">Reasons for Kestrel Warehouse / Operations Team:</div>
                    <ul class="reasons-list" id="reasons-list"></ul>
                </div>
            </div>
        </div>

        <script>
            const presets = {
                high_risk: {
                    order_id: "KO-TEST-HIGH",
                    customer_id: "KC105196",
                    sku: "KH-RV-01",
                    order_value_inr: 14500,
                    payment_mode: "cod",
                    discount_pct: 25,
                    customer_prior_orders: 0,
                    customer_prior_returns: 0,
                    promised_delivery_days: 7,
                    is_shield: "false",
                    delivery_note: "Leave with building security"
                },
                shield_vip: {
                    order_id: "KO-TEST-SHIELD",
                    customer_id: "KC106844",
                    sku: "KH-WP-01",
                    order_value_inr: 9600,
                    payment_mode: "prepaid_upi",
                    discount_pct: 20,
                    customer_prior_orders: 4,
                    customer_prior_returns: 1,
                    promised_delivery_days: 6,
                    is_shield: "true",
                    delivery_note: "Deliver after 6pm"
                },
                low_risk: {
                    order_id: "KO-TEST-SAFE",
                    customer_id: "KC100002",
                    sku: "KH-IC-01",
                    order_value_inr: 2800,
                    payment_mode: "prepaid_upi",
                    discount_pct: 10,
                    customer_prior_orders: 3,
                    customer_prior_returns: 0,
                    promised_delivery_days: 3,
                    is_shield: "false",
                    delivery_note: "Standard delivery"
                }
            };

            function loadPreset(key) {
                const p = presets[key];
                document.getElementById('order_id').value = p.order_id;
                document.getElementById('customer_id').value = p.customer_id;
                document.getElementById('sku').value = p.sku;
                document.getElementById('order_value_inr').value = p.order_value_inr;
                document.getElementById('payment_mode').value = p.payment_mode;
                document.getElementById('discount_pct').value = p.discount_pct;
                document.getElementById('customer_prior_orders').value = p.customer_prior_orders;
                document.getElementById('customer_prior_returns').value = p.customer_prior_returns;
                document.getElementById('promised_delivery_days').value = p.promised_delivery_days;
                document.getElementById('is_shield').value = p.is_shield;
                document.getElementById('delivery_note').value = p.delivery_note;
            }

            async function handleScreen(e) {
                e.preventDefault();
                const btn = document.getElementById('submitBtn');
                btn.innerText = "Evaluating...";
                btn.disabled = true;

                const payload = {
                    order_id: document.getElementById('order_id').value,
                    customer_id: document.getElementById('customer_id').value,
                    sku: document.getElementById('sku').value,
                    order_value_inr: parseFloat(document.getElementById('order_value_inr').value),
                    payment_mode: document.getElementById('payment_mode').value,
                    discount_pct: parseFloat(document.getElementById('discount_pct').value),
                    customer_prior_orders: parseInt(document.getElementById('customer_prior_orders').value),
                    customer_prior_returns: parseInt(document.getElementById('customer_prior_returns').value),
                    promised_delivery_days: parseInt(document.getElementById('promised_delivery_days').value),
                    is_shield: document.getElementById('is_shield').value === "true",
                    delivery_note: document.getElementById('delivery_note').value
                };

                try {
                    const res = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                    const data = await res.json();

                    document.getElementById('result-section').style.display = 'block';
                    document.getElementById('score-display').innerText = (data.score * 100).toFixed(1) + "%";

                    const badge = document.getElementById('action-badge');
                    badge.innerText = data.recommended_action.replace(/_/g, ' ');
                    badge.className = "action-badge";

                    if (data.recommended_action.includes("HOLD")) {
                        badge.classList.add("action-hold");
                        document.getElementById('score-display').style.color = "#dc2626";
                    } else if (data.recommended_action.includes("CALL")) {
                        badge.classList.add("action-call");
                        document.getElementById('score-display').style.color = "#d97706";
                    } else {
                        badge.classList.add("action-clear");
                        document.getElementById('score-display').style.color = "#16a34a";
                    }

                    const reasonsUl = document.getElementById('reasons-list');
                    reasonsUl.innerHTML = '';
                    data.reasons.forEach(r => {
                        const li = document.createElement('li');
                        li.innerText = r;
                        reasonsUl.appendChild(li);
                    });

                    document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
                } catch (err) {
                    alert('Error connecting to /predict endpoint: ' + err.message);
                } finally {
                    btn.innerText = "⚡ Screen Order";
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    print("Starting Kestrel Home Returns Risk Scoring Service on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
