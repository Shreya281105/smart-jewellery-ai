"""
agents/ai_agents.py
Multi-agent system:
  Agent 1 – Data Analysis Agent
  Agent 2 – Sales Prediction Agent
  Agent 3 – Marketing Strategy Agent
  Agent 4 – Customer Intelligence Agent
  Agent 5 – Business Advisor Agent (Gen AI via Anthropic API)

NOTE: Gen AI agent calls the Anthropic claude-haiku-4-5-20251001 model.
      Set ANTHROPIC_API_KEY env variable before running.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

# ── Optional: live LLM call ───────────────────────────────────────────────────
try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# BASE AGENT
# ══════════════════════════════════════════════════════════════════════════════
class BaseAgent:
    def __init__(self, name: str):
        self.name = name
        self.log  = []

    def _record(self, action: str, result):
        self.log.append({'agent': self.name, 'time': str(datetime.now()),
                         'action': action, 'result': str(result)[:300]})
        return result

    def report(self) -> list:
        return self.log


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 1 – DATA ANALYSIS AGENT
# ══════════════════════════════════════════════════════════════════════════════
class DataAnalysisAgent(BaseAgent):
    """Reads dataset, cleans, and generates key summary insights."""

    def __init__(self):
        super().__init__('DataAnalysisAgent')

    def run(self, df: pd.DataFrame) -> dict:
        insights = {}

        # Top selling category
        top_cat = df.groupby('Category')['TotalAmount'].sum().idxmax()
        insights['top_category'] = top_cat
        self._record('top_category', top_cat)

        # Highest revenue city
        top_city = df.groupby('Location')['TotalAmount'].sum().idxmax()
        insights['top_city'] = top_city
        self._record('top_city', top_city)

        # Most popular material
        top_mat = df.groupby('Material')['OrderID'].count().idxmax()
        insights['top_material'] = top_mat

        # Peak season
        top_season = df.groupby('Season')['TotalAmount'].sum().idxmax()
        insights['peak_season'] = top_season

        # Return rate
        ret_rate = round(df['IsReturned'].mean() * 100, 2)
        insights['return_rate_pct'] = ret_rate

        # Average order value
        insights['avg_order_value'] = round(df['TotalAmount'].mean(), 2)

        # Revenue by gender
        gender_rev = df.groupby('Gender')['TotalAmount'].sum().to_dict()
        insights['gender_revenue'] = gender_rev

        self._record('full_insights', insights)
        return insights

    def format_report(self, insights: dict) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║       DATA ANALYSIS AGENT REPORT     ║",
            "╚══════════════════════════════════════╝",
            f"  Top Selling Category : {insights.get('top_category')}",
            f"  Top Revenue City      : {insights.get('top_city')}",
            f"  Most Popular Material : {insights.get('top_material')}",
            f"  Peak Sales Season     : {insights.get('peak_season')}",
            f"  Return Rate           : {insights.get('return_rate_pct')}%",
            f"  Avg Order Value       : ₹{insights.get('avg_order_value'):,}",
        ]
        for g, v in insights.get('gender_revenue', {}).items():
            lines.append(f"  Revenue ({g:6s})      : ₹{v:,.0f}")
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 2 – SALES PREDICTION AGENT
# ══════════════════════════════════════════════════════════════════════════════
class SalesPredictionAgent(BaseAgent):
    """Wraps the FestiveDemandForecaster and surfaces actionable predictions."""

    def __init__(self, forecaster):
        super().__init__('SalesPredictionAgent')
        self.forecaster = forecaster

    def run(self, months_ahead: int = 6) -> pd.DataFrame:
        forecast = self.forecaster.predict_next_months(months_ahead)
        self._record('forecast_generated', f'{len(forecast)} rows')

        # Flag top demand spikes
        top = forecast.nlargest(5, 'PredictedDemand')
        self._record('top_demand_spikes', top[['Month', 'Category', 'PredictedDemand']].to_dict())
        return forecast

    def get_alerts(self, forecast: pd.DataFrame) -> list:
        alerts = []
        for _, row in forecast[forecast['IsFestive'] == 1].iterrows():
            if row['PredictedDemand'] > forecast['PredictedDemand'].mean():
                alerts.append(
                    f"⚠ Month {int(row['Month'])}: High festive demand for "
                    f"{row['Category']} ({row['PredictedDemand']:.0f} orders predicted)"
                )
        return alerts


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 3 – MARKETING STRATEGY AGENT
# ══════════════════════════════════════════════════════════════════════════════
MARKETING_TEMPLATES = {
    'Wedding':     "Run wedding-season bundles: Necklace + Earrings set. Use Instagram reels of bridal looks.",
    'Festival':    "Launch Diwali/Navratri collection with Gold & Kundan. Push WhatsApp broadcasts.",
    'Valentine':   "Promote couple rings with 'For Two' tagline. Offer free gift wrapping.",
    'Birthday':    "Birthday gifting campaign: pendants & bracelets. Target 26–35 age group on Facebook.",
    'Anniversary': "Anniversary specials: Diamond pendants. Use email marketing with countdown timer.",
    'Casual':      "Everyday jewellery series: minimal silver. Leverage Instagram stories & reels.",
}

class MarketingStrategyAgent(BaseAgent):
    def __init__(self):
        super().__init__('MarketingStrategyAgent')

    def run(self, df: pd.DataFrame, upcoming_occasion: str = None) -> dict:
        # Best performing occasion by revenue
        occ_rev   = df.groupby('Occasion')['TotalAmount'].sum()
        best_occ  = occ_rev.idxmax()
        target    = upcoming_occasion or best_occ

        strategy  = MARKETING_TEMPLATES.get(target, "Run a general promotional campaign.")
        top_prods = (df[df['Occasion'] == target]
                       .groupby('ProductName')['TotalAmount']
                       .sum()
                       .nlargest(3)
                       .index.tolist())

        result = {
            'target_occasion': target,
            'strategy':        strategy,
            'top_products':    top_prods,
            'suggested_channels': ['Instagram', 'WhatsApp', 'Email'],
        }
        self._record('marketing_strategy', result)
        return result

    def format_report(self, result: dict) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║    MARKETING STRATEGY AGENT REPORT   ║",
            "╚══════════════════════════════════════╝",
            f"  Target Occasion   : {result['target_occasion']}",
            f"  Strategy          : {result['strategy']}",
            f"  Top Products      : {', '.join(result['top_products'])}",
            f"  Channels          : {', '.join(result['suggested_channels'])}",
        ]
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 4 – CUSTOMER INTELLIGENCE AGENT
# ══════════════════════════════════════════════════════════════════════════════
class CustomerIntelligenceAgent(BaseAgent):
    def __init__(self):
        super().__init__('CustomerIntelligenceAgent')

    def run(self, rfm: pd.DataFrame, loyalty_df: pd.DataFrame) -> dict:
        seg_counts  = rfm['Segment'].value_counts().to_dict()
        top_loyal   = loyalty_df.head(5)[['CustomerID', 'LoyaltyProb', 'CLV_Score']].to_dict('records')
        at_risk     = rfm[rfm['Segment'] == 'At Risk']['CustomerID'].tolist()[:5]
        champions   = rfm[rfm['Segment'] == 'Champions']['CustomerID'].tolist()[:5]

        result = {
            'segment_distribution': seg_counts,
            'top_loyal_customers':  top_loyal,
            'at_risk_customers':    at_risk,
            'champion_customers':   champions,
        }
        self._record('customer_intelligence', result)
        return result

    def format_report(self, result: dict) -> str:
        lines = [
            "╔══════════════════════════════════════╗",
            "║  CUSTOMER INTELLIGENCE AGENT REPORT  ║",
            "╚══════════════════════════════════════╝",
        ]
        for seg, cnt in result['segment_distribution'].items():
            lines.append(f"  {seg:25s}: {cnt} customers")
        lines.append(f"\n  Champions       : {result['champion_customers']}")
        lines.append(f"  At-Risk         : {result['at_risk_customers']}")
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT 5 – BUSINESS ADVISOR AGENT (Gen AI)
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert business advisor for small online jewellery businesses in India.
You have deep knowledge of Indian jewellery market trends, festivals, customer psychology, and e-commerce.
You are given real sales data from the business. Use specific numbers from the data in every answer.
Always structure your answer as:
📊 INSIGHT: (what the data says - use real numbers)
✅ ACTION: (2-3 specific steps to take)
⚠ CAUTION: (one risk to watch out for)
Be direct, specific, and practical. Never give vague generic advice."""

class BusinessAdvisorAgent(BaseAgent):
    """
    Gen AI powered agent. Falls back to rich rule-based engine if API key not set.
    """

    def __init__(self):
        super().__init__('BusinessAdvisorAgent')
        # Try Anthropic first
        anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
        self._live = _ANTHROPIC_AVAILABLE and bool(anthropic_key)
        self._provider = None
        if self._live:
            self._client = anthropic.Anthropic(api_key=anthropic_key)
            self._provider = 'anthropic'
        # Fallback to Groq
        if not self._live:
            groq_key = os.getenv('GROQ_API_KEY', '')
            if _GROQ_AVAILABLE and groq_key:
                self._groq_client = Groq(api_key=groq_key)
                self._live = True
                self._provider = 'groq'

    def ask(self, question: str, context_summary: dict) -> str:
        ctx_str = json.dumps(context_summary, indent=2, default=str)
        user_msg = f"""Here is the real business data:
{ctx_str}

Business Owner's Question: {question}

Answer using specific numbers from the data above. Be practical and India-market specific."""

        if self._live and self._provider == 'anthropic':
            try:
                response = self._client.messages.create(
                    model='claude-haiku-4-5-20251001',
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=[{'role': 'user', 'content': user_msg}]
                )
                answer = response.content[0].text
            except Exception:
                answer = self._smart_rule_based_answer(question, context_summary)
        elif self._live and self._provider == 'groq':
            try:
                response = self._groq_client.chat.completions.create(
                    model='llama-3.1-8b-instant',
                    messages=[
                        {'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_msg}
                    ]
                )
                answer = response.choices[0].message.content
            except Exception:
                answer = self._smart_rule_based_answer(question, context_summary)
        else:
            answer = self._smart_rule_based_answer(question, context_summary)

        self._record('business_question', {'q': question, 'a': answer[:200]})
        return answer

    def _smart_rule_based_answer(self, question: str, ctx: dict) -> str:
        q = question.lower()

        # Pull real numbers from context
        top_cat     = ctx.get('top_category', 'Bracelet')
        top_city    = ctx.get('top_city', 'Delhi')
        top_mat     = ctx.get('top_material', 'Diamond')
        season      = ctx.get('peak_season', 'Festive')
        ret_rate    = ctx.get('return_rate', 17.67)
        avg_order   = ctx.get('avg_order_value', 87755)
        revenue     = ctx.get('total_revenue', 2632664)
        orders      = ctx.get('total_orders', 300)
        rating      = ctx.get('avg_rating', 3.7)
        champions   = ctx.get('champion_count', 33)
        at_risk     = ctx.get('at_risk_count', 26)
        top_occ     = ctx.get('top_occasion', 'Wedding')
        f_revenue   = ctx.get('female_revenue', 0)
        m_revenue   = ctx.get('male_revenue', 0)
        new_pct     = ctx.get('new_customer_pct', 44)
        ret_pct     = ctx.get('returning_customer_pct', 56)

        # ── Promotion / Marketing ──────────────────────────────────────────────
        if any(w in q for w in ['promot', 'market', 'advertis', 'campaign', 'sell more', 'boost sales']):
            return (
                f"📊 INSIGHT: Your top revenue category is {top_cat} and your best city is {top_city}. "
                f"The {season} season drives peak demand. Your avg order value is ₹{avg_order:,.0f}, "
                f"which means even a 10% sales increase adds ₹{revenue*0.1:,.0f} in revenue.\n\n"
                f"✅ ACTION:\n"
                f"1. Run a '{top_cat} Collection' Instagram reel campaign targeting {top_city} + Mumbai audiences — "
                f"use bridal/festive themes since {top_occ} is your top occasion.\n"
                f"2. Launch a WhatsApp broadcast to your {champions} Champion customers with early access to new arrivals — "
                f"they already trust you and have high CLV.\n"
                f"3. Create a '{top_mat} {top_cat}' bundle offer with 10% discount — "
                f"bundles increase avg order value without hurting margins much.\n\n"
                f"⚠ CAUTION: Avoid discounting {top_mat} jewellery heavily — "
                f"it's your highest revenue material. Keep discounts under 15% to protect profit margins."
            )

        # ── Return Rate ────────────────────────────────────────────────────────
        elif any(w in q for w in ['return', 'refund', 'return rate', 'reduce return']):
            high_risk_note = "Silver and lower-rated products" if ret_rate > 15 else "delivery delays"
            return (
                f"📊 INSIGHT: Your current return rate is {ret_rate}%, which means roughly "
                f"{int(orders * ret_rate / 100)} orders out of {orders} are returned. "
                f"Your avg feedback rating is ⭐{rating}/5 — ratings below 3.5 strongly correlate with returns.\n\n"
                f"✅ ACTION:\n"
                f"1. Add detailed size guides, weight (in grams), and 360° photos to all product listings — "
                f"most jewellery returns happen due to 'not as expected'.\n"
                f"2. Flag orders where discount > 20% + rating < 3.5 — these are your highest return risk combos. "
                f"Add a pre-delivery confirmation WhatsApp message for these orders.\n"
                f"3. For {high_risk_note}, add a 'verify before delivery' note in packaging.\n\n"
                f"⚠ CAUTION: Don't remove return option entirely — it damages trust. "
                f"Aim to bring return rate below 10% gradually. Currently at {ret_rate}%, reducing by 5% would save "
                f"approximately ₹{revenue * 0.05 * 0.3:,.0f} in losses."
            )

        # ── Inventory / Stock ──────────────────────────────────────────────────
        elif any(w in q for w in ['stock', 'inventor', 'restock', 'supply', 'order stock', 'run out']):
            return (
                f"📊 INSIGHT: {top_cat} is your best-selling category and {top_mat} is your top revenue material. "
                f"The {season} season is your peak — you process ~{orders//12} orders/month on average "
                f"but likely 2-3x that during peak months.\n\n"
                f"✅ ACTION:\n"
                f"1. Stock 40% extra {top_mat} {top_cat} inventory 6 weeks before {season} season starts — "
                f"supplier lead times in India are typically 3-4 weeks.\n"
                f"2. Use your demand forecast chart (in the AI Predictions tab) to identify which categories "
                f"spike in which months and plan category-wise reorder points.\n"
                f"3. Keep a minimum safety stock of your top 5 SKUs at all times — "
                f"stockouts during peak season are the #1 revenue killer for small jewellery businesses.\n\n"
                f"⚠ CAUTION: Don't overstock Silver jewellery — it has lower margins "
                f"and higher return rates. Focus capital on {top_mat} and Rose Gold."
            )

        # ── Customer / Loyalty ─────────────────────────────────────────────────
        elif any(w in q for w in ['customer', 'loyal', 'retain', 'repeat', 'churn', 'vip']):
            return (
                f"📊 INSIGHT: You have {champions} Champion customers and {at_risk} At-Risk customers. "
                f"Returning customers contribute {ret_pct}% of revenue vs {new_pct}% from new customers. "
                f"Acquiring a new customer costs 5x more than retaining an existing one.\n\n"
                f"✅ ACTION:\n"
                f"1. For your {champions} Champions — send a handwritten thank-you card with their next order "
                f"+ give them a 'VIP Early Access' tag for new collections. This costs almost nothing but builds loyalty.\n"
                f"2. For your {at_risk} At-Risk customers — send a personalised 'We miss you' WhatsApp with "
                f"a 15% comeback coupon valid for 7 days. Create urgency.\n"
                f"3. Start a simple punch-card loyalty program: every 5th order gets free shipping or a small gift.\n\n"
                f"⚠ CAUTION: Don't give loyalty discounts to customers who always buy at full price — "
                f"you'll train them to wait for discounts. Reserve offers for at-risk and lapsed segments only."
            )

        # ── Pricing / Discount ─────────────────────────────────────────────────
        elif any(w in q for w in ['price', 'discount', 'offer', 'deal', 'pricing', 'profit margin']):
            return (
                f"📊 INSIGHT: Your avg order value is ₹{avg_order:,.0f}. "
                f"The Smart Discount Optimizer shows Bracelets and Earrings perform best at 10% discount, "
                f"while Rings maintain good profit even at 20%. {top_mat} jewellery has the highest margins.\n\n"
                f"✅ ACTION:\n"
                f"1. Use tiered pricing: offer 10% for orders above ₹{avg_order*0.8:,.0f} "
                f"and 15% for orders above ₹{avg_order*1.5:,.0f} — this pushes customers to spend more per order.\n"
                f"2. Never discount {top_mat} — instead offer 'free gift wrapping' or 'free engraving' "
                f"as value-adds. Perceived value increases without touching your margin.\n"
                f"3. Run flash sales (24-48 hrs only) on slow-moving categories — creates urgency without "
                f"setting a permanent price expectation.\n\n"
                f"⚠ CAUTION: Discounting above 20% consistently trains customers to never buy at full price. "
                f"Your current avg discount is already affecting perceived premium quality."
            )

        # ── Best products ──────────────────────────────────────────────────────
        elif any(w in q for w in ['best product', 'top product', 'best selling', 'which product', 'what to sell']):
            return (
                f"📊 INSIGHT: {top_cat} is your #1 revenue category and {top_mat} is your top revenue material. "
                f"{top_mat} {top_cat} is effectively your flagship product. "
                f"Your best occasion is {top_occ} — customers buying for {top_occ} spend the most.\n\n"
                f"✅ ACTION:\n"
                f"1. Build your brand around '{top_mat} {top_cat} for {top_occ}' — make this your hero product "
                f"with dedicated landing pages, reels, and customer testimonials.\n"
                f"2. Introduce 2-3 variations of your best-seller (different weights, designs) to cater to "
                f"different budget tiers without diluting the brand.\n"
                f"3. Use Market Basket Analysis — customers who buy {top_cat} likely also want matching Earrings "
                f"or Pendants. Bundle them as a 'Complete Look' set.\n\n"
                f"⚠ CAUTION: Don't spread inventory too thin across all 7 categories. "
                f"Double down on what's already working — {top_cat} and {top_mat}."
            )

        # ── City / Location ────────────────────────────────────────────────────
        elif any(w in q for w in ['city', 'location', 'region', 'where', 'geography', 'target city']):
            return (
                f"📊 INSIGHT: {top_city} is your highest revenue city. "
                f"Delhi customers tend to prefer traditional Gold jewellery while Bangalore skews towards "
                f"minimal Silver pieces. Your city-category heatmap shows clear geographic preferences.\n\n"
                f"✅ ACTION:\n"
                f"1. Run geo-targeted Instagram/Facebook ads specifically for {top_city} — "
                f"show {top_mat} {top_cat} with traditional styling for this market.\n"
                f"2. For cities with lower revenue (Hyderabad, etc.), test a different product mix — "
                f"try affordable Silver and Kundan pieces with lower price points to grow that market.\n"
                f"3. If you're not already, list on Meesho and regional marketplaces for Tier-2 city reach "
                f"where Instagram penetration is lower.\n\n"
                f"⚠ CAUTION: Don't assume {top_city} preferences apply everywhere. "
                f"A/B test creatives by city before running large ad budgets."
            )

        # ── Seasonal / Festival ────────────────────────────────────────────────
        elif any(w in q for w in ['season', 'festival', 'diwali', 'wedding', 'navratri', 'festiv', 'occasion']):
            return (
                f"📊 INSIGHT: {season} is your peak sales season and {top_occ} drives highest order values. "
                f"Festive months can easily bring 2-3x normal monthly revenue for jewellery businesses.\n\n"
                f"✅ ACTION:\n"
                f"1. Start your festive campaign 3 weeks early — most customers plan jewellery purchases "
                f"in advance for weddings and festivals. Don't wait till the week of.\n"
                f"2. Create a 'Festive Edit' collection page featuring {top_mat} pieces — "
                f"curated collections convert better than generic category pages.\n"
                f"3. Partner with 2-3 local wedding photographers or makeup artists for Instagram collabs — "
                f"zero-cost influencer marketing that reaches exactly your target audience.\n\n"
                f"⚠ CAUTION: Stock up inventory 6 weeks before the season — courier and supplier delays "
                f"during Diwali/wedding season are very common and can cost you peak revenue."
            )

        # ── Gender / Demographics ──────────────────────────────────────────────
        elif any(w in q for w in ['female', 'male', 'gender', 'women', 'men', 'age', 'demographic', 'target audience']):
            dom_gender  = 'Female' if f_revenue > m_revenue else 'Male'
            dom_rev     = max(f_revenue, m_revenue)
            return (
                f"📊 INSIGHT: {dom_gender} customers generate ₹{dom_rev:,.0f} in revenue — your dominant segment. "
                f"Age group 26-35 typically has the highest spending in jewellery e-commerce.\n\n"
                f"✅ ACTION:\n"
                f"1. Primary targeting: Women aged 26-40 in Tier-1 cities for Instagram and Facebook ads — "
                f"this demographic shops jewellery for personal use AND gifting.\n"
                f"2. Don't ignore male buyers — they are high-intent gift purchasers (anniversaries, birthdays). "
                f"Create a 'Gift for Her' collection with curated items under ₹5000, ₹10000, ₹20000 price bands.\n"
                f"3. For the 18-25 segment, push affordable Silver and minimal designs — "
                f"they are building brand familiarity now and will be your premium buyers in 5 years.\n\n"
                f"⚠ CAUTION: Don't assume one-size-fits-all content. Male buyers respond to 'gifting occasion' "
                f"messaging, not style/fashion content. Use different ad creatives per gender."
            )

        # ── Revenue / Growth ───────────────────────────────────────────────────
        elif any(w in q for w in ['revenue', 'grow', 'increase sales', 'profit', 'earn more', 'income']):
            return (
                f"📊 INSIGHT: Your total revenue is ₹{revenue:,.0f} from {orders} orders "
                f"with avg order value ₹{avg_order:,.0f}. "
                f"Two levers can move revenue fast: increase order value OR increase order frequency.\n\n"
                f"✅ ACTION:\n"
                f"1. Increase avg order value: add a 'Complete the Look' upsell on every product page — "
                f"if they're buying a Necklace, suggest matching Earrings. Even 20% uptake raises AOV significantly.\n"
                f"2. Increase frequency: email/WhatsApp existing customers every 6 weeks with new arrivals — "
                f"your {champions} Champions are already warm, they just need a nudge.\n"
                f"3. Expand to one new sales channel (Meesho or Myntra) — "
                f"your existing product photography can be reused, cost is minimal.\n\n"
                f"⚠ CAUTION: Chasing new customers is expensive. Your fastest revenue growth will come from "
                f"getting existing customers to buy more often — focus retention before acquisition."
            )

        # ── Social Media ───────────────────────────────────────────────────────
        elif any(w in q for w in ['instagram', 'social media', 'facebook', 'content', 'post', 'reel']):
            return (
                f"📊 INSIGHT: Instagram and Facebook are already in your top order sources. "
                f"Your best-performing products ({top_mat} {top_cat}) and top occasion ({top_occ}) "
                f"give you clear content themes.\n\n"
                f"✅ ACTION:\n"
                f"1. Post 4x/week content mix: 2 product reels, 1 'behind the scenes' (packaging/crafting), "
                f"1 customer testimonial or UGC repost — this builds trust AND reach.\n"
                f"2. Use Reels for discovery (new audiences) and Stories for conversion (existing followers) — "
                f"pin your best-selling {top_cat} to Story Highlights.\n"
                f"3. Hashtag strategy: mix broad (#goldjewellery #indianjewellery) with niche "
                f"(#{top_city.lower()}jewellery #bridaljewellery{top_city.lower()}) for local discovery.\n\n"
                f"⚠ CAUTION: Consistency beats quality for small businesses. "
                f"Posting 4x/week with phone-shot photos beats posting once a month with studio shots."
            )

        # ── Default: comprehensive overview ───────────────────────────────────
        else:
            return (
                f"📊 INSIGHT: Your business has ₹{revenue:,.0f} total revenue, {orders} orders, "
                f"avg order value ₹{avg_order:,.0f}, return rate {ret_rate}%, and avg rating ⭐{rating}/5. "
                f"Top performers: {top_cat} (category), {top_mat} (material), {top_city} (city).\n\n"
                f"✅ ACTION:\n"
                f"1. Your biggest quick win: reduce return rate from {ret_rate}% → below 10% by improving "
                f"product descriptions and adding size/weight guides. This directly improves profitability.\n"
                f"2. Focus marketing spend on {top_city} during {season} season — "
                f"this is your proven highest-value market and time.\n"
                f"3. Activate your {at_risk} At-Risk customers with a win-back campaign — "
                f"they've bought before, they just need a reason to come back.\n\n"
                f"⚠ CAUTION: Your avg rating of ⭐{rating}/5 has room to improve. "
                f"Ratings below 4.0 hurt both return purchases and marketplace ranking. "
                f"Follow up every order with a personal thank-you message asking for honest feedback."
            )
