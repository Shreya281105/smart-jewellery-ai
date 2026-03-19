"""
modules/emotion_recommender.py
UNIQUE FEATURE: Emotion + Occasion aware jewellery recommender.
Uses collaborative signals from historical data + rule-based emotional profiling.
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


# ── Emotion → Occasion → Category mapping ────────────────────────────────────
EMOTION_MAP = {
    'joy':         ['Wedding', 'Anniversary', 'Birthday'],
    'love':        ['Anniversary', 'Valentine', 'Wedding'],
    'celebration': ['Festival', 'Birthday', 'Wedding'],
    'casual':      ['Casual'],
    'gifting':     ['Birthday', 'Anniversary', 'Valentine'],
    'spiritual':   ['Festival'],
    'glamour':     ['Wedding', 'Festival'],
}

OCCASION_CATEGORY_WEIGHTS = {
    'Wedding':     {'Necklace': 5, 'Bangle': 4, 'Earrings': 4, 'Ring': 3},
    'Anniversary': {'Ring': 5, 'Pendant': 4, 'Bracelet': 3},
    'Festival':    {'Bangle': 5, 'Earrings': 4, 'Necklace': 3},
    'Casual':      {'Earrings': 4, 'Bracelet': 3, 'Anklet': 3},
    'Birthday':    {'Pendant': 4, 'Bracelet': 4, 'Ring': 3},
    'Valentine':   {'Ring': 5, 'Pendant': 4, 'Bracelet': 3},
}

MATERIAL_BUDGET = {
    'budget':    ['Silver', 'Kundan'],
    'mid':       ['Rose Gold', 'Gold'],
    'premium':   ['Diamond', 'Platinum'],
}


class EmotionRecommender:
    """
    Recommends jewellery based on:
      - Emotion keyword (joy, love, celebration…)
      - Occasion
      - Budget tier
      - Customer history (optional)
    """

    def __init__(self):
        self._product_profiles = None

    def fit(self, df: pd.DataFrame):
        """Build product popularity profiles from historical data."""
        prod = (df.groupby(['Category', 'Material', 'Occasion'])
                  .agg(Popularity=('OrderID', 'count'),
                       AvgRating=('FeedbackRating', 'mean'),
                       AvgPrice=('Price', 'mean'),
                       ReturnRate=('IsReturned', 'mean'))
                  .reset_index()
                  .round(3))
        prod['Score'] = (
            prod['Popularity'] * 0.4
            + prod['AvgRating'] * 3 * 0.3
            - prod['ReturnRate'] * 10 * 0.3
        ).round(3)
        self._product_profiles = prod
        return self

    def recommend(self,
                  emotion: str = 'joy',
                  occasion: str = 'Wedding',
                  budget: str = 'mid',
                  top_n: int = 5) -> pd.DataFrame:
        """
        Returns top_n jewellery recommendations.
        """
        if self._product_profiles is None:
            raise RuntimeError("Call .fit(df) before .recommend()")

        # Step 1 – expand emotion to related occasions
        related_occasions = EMOTION_MAP.get(emotion.lower(), [occasion])
        if occasion not in related_occasions:
            related_occasions.append(occasion)

        # Step 2 – filter by related occasions
        mask = self._product_profiles['Occasion'].isin(related_occasions)
        pool = self._product_profiles[mask].copy()

        if pool.empty:
            pool = self._product_profiles.copy()

        # Step 3 – apply category weight boost
        cat_weights = {}
        for occ in related_occasions:
            for cat, w in OCCASION_CATEGORY_WEIGHTS.get(occ, {}).items():
                cat_weights[cat] = cat_weights.get(cat, 0) + w
        pool['CategoryBoost'] = pool['Category'].map(cat_weights).fillna(1)
        pool['FinalScore']    = pool['Score'] * pool['CategoryBoost']

        # Step 4 – budget filter
        allowed_materials = MATERIAL_BUDGET.get(budget.lower(),
                                                 MATERIAL_BUDGET['mid'])
        pool = pool[pool['Material'].isin(allowed_materials)]

        if pool.empty:
            pool = self._product_profiles[mask].copy()
            pool['FinalScore'] = pool['Score']

        result = (pool.sort_values('FinalScore', ascending=False)
                      .drop_duplicates(subset=['Category', 'Material'])
                      .head(top_n)
                      [['Category', 'Material', 'Occasion',
                         'AvgRating', 'AvgPrice', 'FinalScore']]
                      .reset_index(drop=True))

        result['RecommendedProduct'] = result['Material'] + ' ' + result['Category']
        result['AvgPrice'] = result['AvgPrice'].apply(lambda x: f'₹{x:,.0f}')
        result['AvgRating'] = result['AvgRating'].round(2)
        result['FinalScore'] = result['FinalScore'].round(3)
        return result

    def explain(self, emotion: str, occasion: str, budget: str) -> str:
        """Returns a human-readable explanation for the recommendation logic."""
        related = EMOTION_MAP.get(emotion.lower(), [occasion])
        mats    = MATERIAL_BUDGET.get(budget.lower(), ['Gold', 'Rose Gold'])
        return (
            f"Emotion '{emotion}' maps to occasions: {', '.join(related)}.\n"
            f"Budget tier '{budget}' suggests materials: {', '.join(mats)}.\n"
            f"Category scores are boosted for occasion '{occasion}' context."
        )
