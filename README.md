# 💎 Smart Sales & Customer Insight System
### AI-Powered Analytics Platform for Small Online Jewellery Businesses

---

## 🗂 Project Structure

```
smart_jewellery_ai/
├── data/
│   ├── generate_dataset.py       # Generates synthetic sales CSV
│   └── jewellery_sales.csv       # 300-row dataset (auto-generated)
├── utils/
│   └── data_loader.py            # Load, clean, feature-engineer data
├── modules/
│   ├── sales_analytics.py        # Revenue, trends, city demand
│   ├── customer_analysis.py      # RFM scoring, CLV, demographics
│   └── emotion_recommender.py    # ✨ UNIQUE: Emotion-based recommender
├── models/
│   └── ml_models.py              # Return risk, demand forecast, loyalty, discount optimizer
├── agents/
│   └── ai_agents.py              # 5 AI agents including Gen AI Advisor
├── outputs/                      # Auto-generated charts & reports
├── app.py                        # Streamlit dashboard
├── main.py                       # Run full pipeline without Streamlit
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Generate dataset
```bash
python data/generate_dataset.py
```

### Step 3 — Run full pipeline (terminal)
```bash
python main.py
```

### Step 4 — Launch Streamlit dashboard
```bash
streamlit run app.py
```

### Step 5 — Enable Gen AI (optional)
Set your Anthropic API key for live AI responses:
```bash
export ANTHROPIC_API_KEY=your_key_here
```
Without this, the Business Advisor uses intelligent rule-based fallback.

---

## 🌟 Features

### Standard Features
| Feature | Description |
|---|---|
| Sales Analytics | Category/material revenue, monthly trends |
| Customer Segmentation | RFM scoring (Champions → Lost) |
| Return Risk Prediction | RandomForest classifier |
| Festive Demand Forecast | GBM time-series regression |
| Customer Loyalty Prediction | ML-based loyalty scoring |
| Smart Discount Optimizer | Elasticity-based optimal discount |
| City × Category Heatmap | Geolocation demand analysis |

### ✨ Unique Features (Not in Existing Systems)
| Feature | Why It's Unique |
|---|---|
| **Emotion-Based Recommender** | Maps customer emotion (joy/love/celebration) to jewellery type |
| **Festive Trend Prediction Engine** | Predicts demand spikes for Indian festivals specifically |
| **Gen AI Business Advisor Chatbot** | Conversational AI business consultant using Claude |
| **Hidden CLV Detector** | Identifies future high-value customers before they churn |
| **Occasion-Aware Marketing Agent** | Generates campaign ideas per occasion automatically |

### 🤖 AI Agents
1. **DataAnalysisAgent** — Auto-reads and summarizes dataset
2. **SalesPredictionAgent** — Forecasts demand and raises alerts
3. **MarketingStrategyAgent** — Generates occasion-specific campaigns
4. **CustomerIntelligenceAgent** — Segments and scores customers
5. **BusinessAdvisorAgent (Gen AI)** — Conversational Claude-powered consultant

---

## 🛠 Tech Stack
- **Python** — Core language
- **Pandas / NumPy** — Data processing
- **Scikit-learn** — ML models
- **Matplotlib / Seaborn** — Visualizations
- **Anthropic Claude API** — Generative AI
- **Streamlit** — Interactive dashboard

---

## 📊 ML Models & Accuracy
| Model | Algorithm | Purpose |
|---|---|---|
| Return Risk | RandomForest (balanced) | Predict return probability |
| Demand Forecast | GradientBoosting | Monthly demand prediction |
| Loyalty Score | RandomForest | Identify future loyal customers |

---

## 💡 Sample Questions for Business Advisor
- *Which jewellery should I promote next month?*
- *How can I reduce return rates?*
- *Which customers should I target for loyalty rewards?*
- *What inventory should I stock before Diwali?*

---

## 📌 Notes
- Dataset is synthetic (300 orders). Replace with real business data for production use.
- Gen AI fallback ensures the system works without an API key.
- All charts are saved to `/outputs/` automatically.
