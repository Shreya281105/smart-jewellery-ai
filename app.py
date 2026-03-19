"""
app.py — Smart Sales & Customer Insight System
Fixes applied:
  1. Chatbot hallucination  — richer pre-computed context
  2. Cross-validation       — shown in model metrics
  3. Return risk form       — AgeGroup + Gender now user-selectable
  4. Market Basket Analysis — new tab in AI Predictions
  5. CSV upload             — business owners can upload real data
  6. Interactive charts     — matplotlib with annotations (plotly fallback)
  7. Joblib model saving    — models cached, no retrain on reload
  8. Error handling         — try/except everywhere with friendly messages
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys, traceback

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_data, get_summary
from modules.sales_analytics import (top_categories, top_materials, city_demand,
                                      plot_category_revenue, plot_monthly_trend,
                                      plot_city_heatmap)
from modules.customer_analysis import (compute_rfm, age_gender_summary,
                                        new_vs_returning, plot_rfm_segments,
                                        plot_age_spending)
from modules.emotion_recommender import EmotionRecommender
from models.ml_models import (ReturnRiskPredictor, FestiveDemandForecaster,
                               LoyaltyPredictor, optimize_discount,
                               run_market_basket)
from agents.ai_agents import (DataAnalysisAgent, SalesPredictionAgent,
                               MarketingStrategyAgent, CustomerIntelligenceAgent,
                               BusinessAdvisorAgent)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="💎 Smart Jewellery AI", page_icon="💎",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stMetric label {font-size:0.85rem;color:#a0a0c0;}
    section[data-testid="stSidebar"] {background:#0f0f1a;}
    .cv-box {background:#1a1a2e;border-radius:8px;padding:12px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)


# ── FIX 5: CSV Upload + Data Loading ──────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data(file_hash=None):
    return load_data()

def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load and validate user-uploaded CSV."""
    try:
        df = pd.read_csv(uploaded_file, parse_dates=['OrderDate'])
        required = ['OrderID', 'CustomerID', 'TotalAmount', 'Category',
                    'Material', 'FeedbackRating', 'ReturnStatus', 'Discount',
                    'Season', 'Occasion', 'Location', 'Gender', 'AgeGroup',
                    'CustomerType', 'Price', 'Quantity', 'ProfitMargin',
                    'DeliveryTime', 'Weight', 'OrderSource']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            return None
        from utils.data_loader import _clean, _engineer
        df = _clean(df)
        df = _engineer(df)
        return df
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None


# ── FIX 7: Model loading with joblib cache ─────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_models(data_hash):
    with st.spinner("🔄 Training AI models (first load only — saved for next time)..."):
        try:
            df  = get_data()
            rrp = ReturnRiskPredictor()
            if not rrp.load():
                rrp_metrics = rrp.train(df)
            else:
                rrp_metrics = {'accuracy': 'cached', 'cv_mean': 'cached', 'cv_std': '-'}

            fdf = FestiveDemandForecaster()
            if not fdf.load():
                fdf_metrics = fdf.train(df)
            else:
                fdf_metrics = {'r2': 'cached', 'cv_r2': 'cached'}

            rfm = compute_rfm(df)
            lp  = LoyaltyPredictor()
            if not lp.load():
                lp_metrics = lp.train(rfm)
            else:
                lp_metrics = {'accuracy': 'cached', 'cv_mean': 'cached'}

            loyal_df = lp.predict(rfm)
            er       = EmotionRecommender().fit(df)
            return rrp, fdf, rfm, lp, loyal_df, er, rrp_metrics, fdf_metrics, lp_metrics
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.code(traceback.format_exc())
            st.stop()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/diamond.png", width=64)
    st.title("💎 Jewellery AI")

    # ── FIX 5: CSV Upload ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📂 Upload Your Data")
    uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'],
                                 help="Upload your own jewellery sales CSV. Must match the required column format.")
    if uploaded is not None:
        df_up = load_uploaded_data(uploaded)
        if df_up is not None:
            st.session_state['df']        = df_up
            st.session_state['data_hash'] = uploaded.name + str(len(df_up))
            import shutil
            saved_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
            if os.path.exists(saved_dir):
                shutil.rmtree(saved_dir)
                os.makedirs(saved_dir)
            get_models.clear()
            st.session_state.pop('models', None)
            st.success(f"✅ Loaded {len(df_up)} rows! Models retraining...")
            st.rerun()

    if 'df' in st.session_state:
        st.info(f"📊 Using uploaded data ({len(st.session_state['df'])} rows)")
        if st.button("🔄 Reset to Original Data"):
            st.session_state.pop('df', None)
            st.session_state.pop('data_hash', None)
            st.session_state.pop('models', None)
            st.rerun()
    st.markdown("---")

    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "📊 Sales Analytics",
        "👥 Customer Insights",
        "🤖 AI Predictions",
        "✨ Emotion Recommender",
        "🤖 AI Agents",
        "💬 Business Advisor",
    ])

# ── Cached model trainer for uploaded data ────────────────────────────────────
# ── Load data & train models ───────────────────────────────────────────────────
def train_models(df):
    rrp   = ReturnRiskPredictor()
    rrp_m = rrp.train(df)
    fdf   = FestiveDemandForecaster()
    fdf_m = fdf.train(df)
    rfm   = compute_rfm(df)
    lp    = LoyaltyPredictor()
    lp_m  = lp.train(rfm)
    loyal_df = lp.predict(rfm)
    er    = EmotionRecommender().fit(df)
    return rrp, fdf, rfm, lp, loyal_df, er, rrp_m, fdf_m, lp_m

try:
    if 'df' in st.session_state:
        df = st.session_state['df']
        # Only retrain if not already done for this upload
        if 'models' not in st.session_state:
            with st.spinner("⚙️ Training models on your data (one time only)..."):
                st.session_state['models'] = train_models(df)
        (rrp, fdf, rfm, lp, loyal_df, er,
         rrp_m, fdf_m, lp_m) = st.session_state['models']
    else:
        df = get_data()
        (rrp, fdf, rfm, lp, loyal_df, er,
         rrp_m, fdf_m, lp_m) = get_models(str(len(df)))
    summary = get_summary(df)
except Exception as e:
    st.error(f"❌ Failed to load data or models: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("💎 Smart Sales & Customer Insight System")
    st.caption("AI-powered analytics platform for jewellery businesses")

    # FIX 8: wrapped in try/except
    try:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Orders",     f"{summary['total_orders']:,}")
        c2.metric("Total Revenue",    f"₹{summary['total_revenue']:,.0f}")
        c3.metric("Avg Order Value",  f"₹{summary['avg_order_value']:,.0f}")
        c4.metric("Return Rate",      f"{summary['return_rate']}%")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Unique Customers", f"{summary['unique_customers']}")
        c6.metric("Total Profit",     f"₹{summary['total_profit']:,.0f}")
        c7.metric("Avg Rating",       f"⭐ {summary['avg_rating']}")
        c8.metric("Champions",        f"{(rfm['Segment']=='Champions').sum()}")
    except Exception as e:
        st.error(f"Metrics error: {e}")

    try:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Revenue by Category")
            st.image(plot_category_revenue(df), use_container_width=True)
        with col2:
            st.subheader("Customer Segments")
            st.image(plot_rfm_segments(rfm), use_container_width=True)
        st.subheader("Monthly Revenue Trend")
        st.image(plot_monthly_trend(df), use_container_width=True)
    except Exception as e:
        st.error(f"Chart error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – SALES ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Sales Analytics":
    st.title("📊 Sales Analytics")
    try:
        tab1, tab2, tab3 = st.tabs(["Categories", "Materials", "City Demand"])
        with tab1:
            st.subheader("Top Jewellery Categories")
            st.dataframe(top_categories(df), use_container_width=True)
        with tab2:
            st.subheader("Material Performance")
            st.dataframe(top_materials(df), use_container_width=True)
        with tab3:
            st.subheader("City × Category Revenue Heatmap")
            st.image(plot_city_heatmap(df), use_container_width=True)
            st.dataframe(city_demand(df), use_container_width=True)
    except Exception as e:
        st.error(f"Analytics error: {e}")
        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – CUSTOMER INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 Customer Insights":
    st.title("👥 Customer Insights")
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("RFM Customer Segments")
            st.image(plot_rfm_segments(rfm), use_container_width=True)
            st.dataframe(rfm[['CustomerID','Segment','CLV_Score',
                               'RFM_Total']].head(20), use_container_width=True)
        with col2:
            st.subheader("Avg Spending by Age Group")
            st.image(plot_age_spending(df), use_container_width=True)

        st.subheader("New vs Returning Customers")
        nvr  = new_vs_returning(df)
        cols = st.columns(len(nvr))
        for col, (k, v) in zip(cols, nvr.items()):
            col.metric(k, f"₹{v['revenue']:,.0f}",
                       f"{v['revenue_pct']}% of revenue")

        st.subheader("Top Loyal Customers (Predicted)")
        st.dataframe(
            loyal_df[['CustomerID','Segment','LoyaltyProb',
                       'CLV_Score']].head(15),
            use_container_width=True)
    except Exception as e:
        st.error(f"Customer insights error: {e}")
        st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – AI PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Predictions":
    st.title("🤖 AI Predictions")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Return Risk", "Festive Demand",
        "Discount Optimizer", "Market Basket", "Model Metrics"])  # ← NEW tabs

    # ── Return Risk ────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("🔴 Return Risk Prediction")
        try:
            # FIX 3: AgeGroup and Gender are now user-selectable
            c1, c2, c3 = st.columns(3)
            cat    = c1.selectbox("Category",   df['Category'].unique())
            mat    = c2.selectbox("Material",   df['Material'].unique())
            occ    = c3.selectbox("Occasion",   df['Occasion'].unique())
            c4, c5, c6 = st.columns(3)
            price  = c4.number_input("Price (₹)", 500, 200000, 5000)
            qty    = c5.number_input("Quantity", 1, 10, 1)
            disc   = c6.slider("Discount %", 0, 50, 10)
            c7, c8, c9 = st.columns(3)
            rating = c7.slider("Expected Rating", 1.0, 5.0, 4.0, 0.1)
            deliv  = c8.number_input("Delivery Days", 1, 15, 5)
            season = c9.selectbox("Season", df['Season'].unique())
            # FIX 3: these two now actually affect the prediction
            c10, c11 = st.columns(2)
            age_grp = c10.selectbox("Age Group",
                                     df['AgeGroup'].unique())
            gender  = c11.selectbox("Gender",
                                     df['Gender'].unique())

            if st.button("🔍 Predict Return Risk"):
                record = {
                    'Category': cat, 'Material': mat, 'Season': season,
                    'Occasion': occ, 'AgeGroup': age_grp, 'Gender': gender,
                    'OrderSource': 'Instagram', 'Price': price,
                    'Quantity': qty, 'Discount': disc,
                    'FeedbackRating': rating, 'DeliveryTime': deliv,
                    'Weight': 10.0, 'ProfitMargin': 25.0
                }
                risk  = rrp.predict_proba_single(record)
                color = "🔴" if risk > 40 else ("🟡" if risk > 20 else "🟢")
                st.metric(f"{color} Return Probability", f"{risk}%")
                if risk > 40:
                    st.error("High return risk. Improve product photos, "
                             "add weight/size details, and reduce discount.")
                elif risk > 20:
                    st.warning("Moderate risk. Verify product descriptions "
                               "and delivery time.")
                else:
                    st.success("Low return risk. Good customer satisfaction signals.")

            img_path = os.path.join(os.path.dirname(__file__),
                                    'outputs', 'return_risk_features.png')
            if os.path.exists(img_path):
                st.subheader("Feature Importance")
                st.image(img_path, use_container_width=True)
        except Exception as e:
            st.error(f"Return risk error: {e}")

    # ── Festive Demand ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("🎊 Festive Demand Forecast")
        try:
            months   = st.slider("Months to forecast", 3, 12, 6)
            forecast = fdf.predict_next_months(months)
            st.image(fdf.plot_forecast(forecast), use_container_width=True)
            st.dataframe(forecast.sort_values('PredictedDemand', ascending=False),
                         use_container_width=True)
            agent2 = SalesPredictionAgent(fdf)
            agent2.run(months)
            alerts = agent2.get_alerts(forecast)
            if alerts:
                st.subheader("⚠ Demand Alerts")
                for a in alerts:
                    st.warning(a)
        except Exception as e:
            st.error(f"Forecast error: {e}")

    # ── Discount Optimizer ─────────────────────────────────────────────────────
    with tab3:
        st.subheader("💰 Smart Discount Optimizer")
        try:
            optimal = optimize_discount(df)
            st.dataframe(optimal, use_container_width=True)
            st.bar_chart(optimal.set_index('Category')['OptimalDiscount'])
        except Exception as e:
            st.error(f"Discount optimizer error: {e}")

    # ── FIX 4: Market Basket Analysis ─────────────────────────────────────────
    with tab4:
        st.subheader("🛒 Market Basket Analysis")
        st.caption("Finds which jewellery categories customers frequently buy together")
        try:
            min_sup  = st.slider("Min Support",    0.01, 0.3, 0.05, 0.01)
            min_conf = st.slider("Min Confidence", 0.1,  0.8, 0.2,  0.05)
            basket   = run_market_basket(df, min_sup, min_conf)
            if basket.empty:
                st.info("No strong associations found. "
                        "Try lowering support or confidence thresholds.")
            else:
                st.success(f"Found {len(basket)} product associations!")
                st.dataframe(basket, use_container_width=True)
                st.subheader("💡 Bundle Recommendations")
                for _, row in basket.head(5).iterrows():
                    st.markdown(
                        f"🔗 Customers who buy **{row['If Customer Buys']}** "
                        f"also buy **{row['They Also Buy']}** "
                        f"— Confidence: {row['Confidence']:.0%}, "
                        f"Lift: {row['Lift']:.2f}x"
                    )
        except Exception as e:
            st.error(f"Market basket error: {e}")

    # ── FIX 2: Model CV Metrics ────────────────────────────────────────────────
    with tab5:
        st.subheader("📈 Model Performance & Cross-Validation")
        st.caption("5-Fold Cross-Validation results for each ML model")
        try:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🔴 Return Risk Predictor**")
                st.metric("Test Accuracy",  f"{rrp_m.get('accuracy')}%")
                st.metric("CV Mean Acc",    f"{rrp_m.get('cv_mean')}%")
                st.metric("CV Std Dev",     f"±{rrp_m.get('cv_std')}%")
                if 'cv_scores' in rrp_m:
                    for i, s in enumerate(rrp_m['cv_scores'], 1):
                        st.caption(f"Fold {i}: {s}%")
            with col2:
                st.markdown("**📈 Demand Forecaster**")
                st.metric("Train R²",   str(fdf_m.get('r2')))
                st.metric("CV R²",      str(fdf_m.get('cv_r2')))
                st.metric("CV MAE",     str(fdf_m.get('cv_mae')))
            with col3:
                st.markdown("**💛 Loyalty Predictor**")
                st.metric("Test Accuracy", f"{lp_m.get('accuracy')}%")
                st.metric("CV Mean Acc",   f"{lp_m.get('cv_mean')}%")
                st.metric("CV Std Dev",    f"±{lp_m.get('cv_std')}%")

            st.info("💡 CV Mean Accuracy shows how the model performs on unseen data "
                    "across 5 different train/test splits — more reliable than "
                    "a single split accuracy.")
        except Exception as e:
            st.error(f"Metrics display error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – EMOTION RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✨ Emotion Recommender":
    st.title("✨ Emotion-Based Jewellery Recommender")
    st.caption("Recommends jewellery based on emotional context & occasion")
    try:
        c1, c2, c3 = st.columns(3)
        emotion  = c1.selectbox("😊 How are you feeling?",
                                  ['joy', 'love', 'celebration', 'casual',
                                   'gifting', 'spiritual', 'glamour'])
        occasion = c2.selectbox("🎊 Occasion",
                                  ['Wedding', 'Anniversary', 'Festival',
                                   'Casual', 'Birthday', 'Valentine'])
        budget   = c3.selectbox("💳 Budget", ['budget', 'mid', 'premium'])

        if st.button("✨ Get Recommendations"):
            recs = er.recommend(emotion, occasion, budget, top_n=5)
            expl = er.explain(emotion, occasion, budget)
            st.info(expl)
            st.subheader("🏅 Top Recommendations")
            st.dataframe(recs, use_container_width=True)
            for _, row in recs.iterrows():
                st.markdown(
                    f"💍 **{row['RecommendedProduct']}** | "
                    f"⭐ {row['AvgRating']} | "
                    f"💰 {row['AvgPrice']} | "
                    f"Occasion: {row['Occasion']}"
                )
    except Exception as e:
        st.error(f"Recommender error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – AI AGENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Agents":
    st.title("🤖 AI Agents Dashboard")
    try:
        agent1   = DataAnalysisAgent()
        insights = agent1.run(df)
        st.subheader("Agent 1: Data Analysis Agent")
        st.code(agent1.format_report(insights))
    except Exception as e:
        st.error(f"Agent 1 error: {e}")

    try:
        agent3     = MarketingStrategyAgent()
        occ_choice = st.selectbox("Select Upcoming Occasion",
                                   df['Occasion'].unique())
        mkt        = agent3.run(df, occ_choice)
        st.subheader("Agent 3: Marketing Strategy Agent")
        st.code(agent3.format_report(mkt))
    except Exception as e:
        st.error(f"Agent 3 error: {e}")

    try:
        agent4 = CustomerIntelligenceAgent()
        ci     = agent4.run(rfm, loyal_df)
        st.subheader("Agent 4: Customer Intelligence Agent")
        st.code(agent4.format_report(ci))
    except Exception as e:
        st.error(f"Agent 4 error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 – BUSINESS ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Business Advisor":
    st.title("💬 AI Business Advisor")
    st.caption("Ask any business question — powered by Gen AI")

    try:
        agent5 = BusinessAdvisorAgent()

        # FIX 1: Rich pre-computed context to reduce hallucinations
        gender_rev    = df.groupby('Gender')['TotalAmount'].sum()
        age_rev       = df.groupby('AgeGroup')['TotalAmount'].sum()
        nvr           = new_vs_returning(df)
        cat_rev       = df.groupby('Category')['TotalAmount'].sum().to_dict()
        mat_rev       = df.groupby('Material')['TotalAmount'].sum().to_dict()
        city_rev      = df.groupby('Location')['TotalAmount'].sum().to_dict()
        occ_rev       = df.groupby('Occasion')['TotalAmount'].sum().to_dict()
        ret_by_mat    = df.groupby('Material')['IsReturned'].mean().round(3).to_dict()
        ret_by_cat    = df.groupby('Category')['IsReturned'].mean().round(3).to_dict()
        avg_del_ret   = round(df[df['IsReturned']==1]['DeliveryTime'].mean(), 1)
        avg_del_noret = round(df[df['IsReturned']==0]['DeliveryTime'].mean(), 1)

        ctx = get_summary(df)
        ctx.update({
            'top_category':              top_categories(df).index[0],
            'top_city':                  city_demand(df).index[0],
            'top_material':              top_materials(df).index[0],
            'top_occasion':              df.groupby('Occasion')['TotalAmount'].sum().idxmax(),
            'peak_season':               df.groupby('Season')['TotalAmount'].sum().idxmax(),
            'champion_count':            int((rfm['Segment']=='Champions').sum()),
            'at_risk_count':             int((rfm['Segment']=='At Risk').sum()),
            'lost_count':                int((rfm['Segment']=='Lost / Inactive').sum()),
            'female_revenue':            float(gender_rev.get('Female', 0)),
            'male_revenue':              float(gender_rev.get('Male', 0)),
            'new_customer_pct':          nvr.get('New', {}).get('revenue_pct', 44),
            'returning_customer_pct':    nvr.get('Returning', {}).get('revenue_pct', 56),
            'top_age_group':             age_rev.idxmax(),
            'age_group_revenue':         age_rev.round(0).to_dict(),
            'category_revenue':          {k: round(v,0) for k,v in cat_rev.items()},
            'material_revenue':          {k: round(v,0) for k,v in mat_rev.items()},
            'city_revenue':              {k: round(v,0) for k,v in city_rev.items()},
            'occasion_revenue':          {k: round(v,0) for k,v in occ_rev.items()},
            'return_rate_by_material':   ret_by_mat,
            'return_rate_by_category':   ret_by_cat,
            'avg_delivery_returned':     avg_del_ret,
            'avg_delivery_not_returned': avg_del_noret,
            'highest_return_material':   max(ret_by_mat, key=ret_by_mat.get),
            'lowest_return_material':    min(ret_by_mat, key=ret_by_mat.get),
        })

        # Show which AI provider is active
        provider_labels = {
            'anthropic': '🟢 Claude AI (Anthropic)',
            'gemini':    '🟢 Gemini AI (Google)',
            'groq':      '🟢 LLaMA AI (Groq)',
            None:        '🟡 Smart Rule-Based System'
        }
        st.caption(f"AI Provider: {provider_labels.get(agent5._provider, '🟡 Rule-Based')}")

        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        if not st.session_state.chat_history:
            st.subheader("💡 Try asking:")
            q_cols = st.columns(3)
            suggestions = [
                "Which jewellery should I promote next month?",
                "How can I reduce return rates?",
                "Which customers should I target for loyalty?",
            ]
            for col, q in zip(q_cols, suggestions):
                if col.button(q):
                    st.session_state._auto_question = q

            st.markdown("---")
            q_cols2 = st.columns(3)
            suggestions2 = [
                "What's the correlation between delivery time and returns?",
                "Should I invest more in Gold or Diamond?",
                "What is my biggest revenue opportunity?",
            ]
            for col, q in zip(q_cols2, suggestions2):
                if col.button(q):
                    st.session_state._auto_question = q

        auto_q   = st.session_state.pop('_auto_question', None)
        question = st.chat_input("Ask your business question...") or auto_q

        if question:
            st.session_state.chat_history.append(
                {'role': 'user', 'content': question})
            with st.chat_message('user'):
                st.write(question)
            with st.chat_message('assistant'):
                with st.spinner("Thinking..."):
                    try:
                        answer = agent5.ask(question, ctx)
                    except Exception as e:
                        answer = f"Sorry, I encountered an error: {str(e)}. Please try again."
                st.write(answer)
            st.session_state.chat_history.append(
                {'role': 'assistant', 'content': answer})

        if st.session_state.chat_history:
            if st.button("🗑 Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    except Exception as e:
        st.error(f"Business Advisor error: {e}")
        st.code(traceback.format_exc())