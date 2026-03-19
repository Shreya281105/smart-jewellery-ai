"""
main.py — Run the full Smart Jewellery AI pipeline (no Streamlit needed)
Generates all insights, model results, and charts to /outputs/
"""
import os, sys
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
                               LoyaltyPredictor, optimize_discount)
from agents.ai_agents import (DataAnalysisAgent, SalesPredictionAgent,
                               MarketingStrategyAgent, CustomerIntelligenceAgent,
                               BusinessAdvisorAgent)

DIVIDER = "=" * 55

def section(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def main():
    print("\n💎 Smart Sales & Customer Insight System")
    print("   AI-Powered Analytics for Jewellery Businesses")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    section("1. LOADING & PREPROCESSING DATA")
    df = load_data()
    summary = get_summary(df)
    print(f"  Rows loaded : {summary['total_orders']}")
    print(f"  Revenue     : ₹{summary['total_revenue']:,.2f}")
    print(f"  Return Rate : {summary['return_rate']}%")
    print(f"  Avg Order   : ₹{summary['avg_order_value']:,.2f}")

    # ── 2. Sales Analytics ────────────────────────────────────────────────────
    section("2. SALES ANALYTICS")
    print("\n-- Top Categories --")
    print(top_categories(df).to_string())
    print("\n-- Top Materials --")
    print(top_materials(df).to_string())
    print("\n-- City Demand --")
    print(city_demand(df).to_string())

    print("\n  Generating charts...")
    plot_category_revenue(df)
    plot_monthly_trend(df)
    plot_city_heatmap(df)
    print("  Charts saved to /outputs/")

    # ── 3. Customer Analysis ──────────────────────────────────────────────────
    section("3. CUSTOMER BEHAVIOUR ANALYSIS")
    rfm = compute_rfm(df)
    print("\n-- RFM Segments --")
    print(rfm['Segment'].value_counts().to_string())
    print("\n-- New vs Returning --")
    for k, v in new_vs_returning(df).items():
        print(f"  {k}: {v['orders']} orders, ₹{v['revenue']:,.0f} ({v['revenue_pct']}%)")
    plot_rfm_segments(rfm)
    plot_age_spending(df)

    # ── 4. ML Models ──────────────────────────────────────────────────────────
    section("4. MACHINE LEARNING MODELS")

    # Return Risk
    print("\n[A] Return Risk Predictor")
    rrp = ReturnRiskPredictor()
    result = rrp.train(df)
    print(f"  Model Accuracy: {result['accuracy']}%")
    # Example prediction
    sample = {'Category': 'Necklace', 'Material': 'Silver', 'Season': 'Festive',
              'Occasion': 'Wedding', 'AgeGroup': '26-35', 'Gender': 'Female',
              'OrderSource': 'Instagram', 'Price': 3500.0, 'Quantity': 1,
              'Discount': 15, 'FeedbackRating': 3.5, 'DeliveryTime': 7,
              'Weight': 12.0, 'ProfitMargin': 20.0}
    risk = rrp.predict_proba_single(sample)
    print(f"  Sample Prediction (Silver Necklace) → Return Risk: {risk}%")

    # Festive Forecaster
    print("\n[B] Festive Demand Forecaster")
    fdf = FestiveDemandForecaster()
    fm  = fdf.train(df)
    print(f"  R² Score : {fm['r2']}  |  MAE : {fm['mae']}")
    forecast = fdf.predict_next_months(6)
    print("\n-- 6-Month Demand Forecast (Top 10 rows) --")
    print(forecast.nlargest(10, 'PredictedDemand').to_string(index=False))
    fdf.plot_forecast(forecast)

    # Loyalty Prediction
    print("\n[C] Customer Loyalty Predictor")
    lp       = LoyaltyPredictor()
    lp_res   = lp.train(rfm)
    print(f"  Accuracy: {lp_res['accuracy']}%")
    loyal_df = lp.predict(rfm)
    print("\n-- Top 5 Future-Loyal Customers --")
    print(loyal_df[['CustomerID','Segment','LoyaltyProb','CLV_Score']].head(5).to_string(index=False))

    # Discount Optimizer
    print("\n[D] Smart Discount Optimizer")
    opt = optimize_discount(df)
    print(opt.to_string(index=False))

    # ── 5. Emotion Recommender ────────────────────────────────────────────────
    section("5. EMOTION-BASED RECOMMENDER (UNIQUE FEATURE)")
    er = EmotionRecommender().fit(df)
    recs = er.recommend(emotion='love', occasion='Anniversary',
                        budget='mid', top_n=5)
    print("\nQuery: Emotion=love | Occasion=Anniversary | Budget=mid")
    print(recs[['RecommendedProduct','AvgRating','AvgPrice']].to_string(index=False))
    print("\nExplanation:")
    print(er.explain('love', 'Anniversary', 'mid'))

    # ── 6. AI Agents ──────────────────────────────────────────────────────────
    section("6. AI AGENTS")

    print("\n[Agent 1] Data Analysis Agent")
    a1 = DataAnalysisAgent()
    i1 = a1.run(df)
    print(a1.format_report(i1))

    print("\n[Agent 2] Sales Prediction Agent")
    a2 = SalesPredictionAgent(fdf)
    fc = a2.run(3)
    for alert in a2.get_alerts(fc):
        print(" ", alert)

    print("\n[Agent 3] Marketing Strategy Agent")
    a3  = MarketingStrategyAgent()
    m3  = a3.run(df, 'Wedding')
    print(a3.format_report(m3))

    print("\n[Agent 4] Customer Intelligence Agent")
    a4 = CustomerIntelligenceAgent()
    c4 = a4.run(rfm, loyal_df)
    print(a4.format_report(c4))

    # ── 7. Gen AI Business Advisor ────────────────────────────────────────────
    section("7. BUSINESS ADVISOR AGENT (GEN AI)")
    ctx = get_summary(df)
    ctx.update({
        'top_category': top_categories(df).index[0],
        'top_city':     city_demand(df).index[0],
        'peak_season':  df.groupby('Season')['TotalAmount'].sum().idxmax(),
    })
    a5 = BusinessAdvisorAgent()
    questions = [
        "Which jewellery should I promote next month?",
        "How can I reduce return rates?",
    ]
    for q in questions:
        print(f"\n❓ {q}")
        print(f"💡 {a5.ask(q, ctx)}")

    section("PIPELINE COMPLETE — All outputs saved to /outputs/")


if __name__ == '__main__':
    main()
