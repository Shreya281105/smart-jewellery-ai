"""
models/ml_models.py
1. Return Risk Prediction       (Classification + 5-Fold CV)
2. Festive Demand Forecasting   (Regression + 5-Fold CV)
3. Customer Loyalty Prediction  (Classification + 5-Fold CV)
4. Smart Discount Optimizer
5. Market Basket Analysis       (Apriori / Co-occurrence fallback)
All models saved/loaded with joblib for faster Streamlit startup.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings, joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     StratifiedKFold, KFold)
from sklearn.metrics import (classification_report,
                              mean_absolute_error, r2_score)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. RETURN RISK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
class ReturnRiskPredictor:
    CAT_COLS = ['Category', 'Material', 'Season', 'Occasion',
                'AgeGroup', 'Gender', 'OrderSource']
    NUM_COLS = ['Price', 'Quantity', 'Discount', 'FeedbackRating',
                'DeliveryTime', 'Weight', 'ProfitMargin']
    MODEL_PATH = os.path.join(MODEL_DIR, 'return_risk.joblib')

    def __init__(self):
        self.encoders = {}
        self._n_trees = None  # set dynamically
        self.model    = None
        self.trained  = False

    def _encode(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        d = df.copy()
        for col in self.CAT_COLS:
            if fit:
                le = LabelEncoder()
                d[col] = le.fit_transform(d[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                d[col] = d[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0]
                    if x in le.classes_ else -1)
        return d

    def train(self, df: pd.DataFrame) -> dict:
        n_trees = 50 if len(df) < 100 else 150
        self.model = RandomForestClassifier(n_estimators=n_trees, random_state=42, class_weight='balanced')
        data = self._encode(df, fit=True)
        X    = data[self.CAT_COLS + self.NUM_COLS]
        y    = data['IsReturned']

        cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                    random_state=42, stratify=y)
        self.model.fit(X_tr, y_tr)
        self.trained = True
        report = classification_report(self.model.predict(X_te), y_te,
                                       output_dict=True)
        joblib.dump({'model': self.model, 'encoders': self.encoders},
                    self.MODEL_PATH)
        self._plot_feature_importance(X.columns)
        return {
            'accuracy': round(report['accuracy'] * 100, 2),
            'cv_mean':  round(cv_scores.mean() * 100, 2),
            'cv_std':   round(cv_scores.std()  * 100, 2),
            'cv_scores': [round(s * 100, 2) for s in cv_scores],
        }

    def load(self) -> bool:
        if os.path.exists(self.MODEL_PATH):
            data          = joblib.load(self.MODEL_PATH)
            self.model    = data['model']
            self.encoders = data['encoders']
            self.trained  = True
            return True
        return False

    def predict_proba_single(self, record: dict) -> float:
        row = pd.DataFrame([record])
        row = self._encode(row, fit=False)
        X   = row[self.CAT_COLS + self.NUM_COLS]
        return round(self.model.predict_proba(X)[0][1] * 100, 1)

    def _plot_feature_importance(self, feature_names):
        imp = pd.Series(self.model.feature_importances_,
                        index=feature_names).sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        imp.plot(kind='barh', ax=ax, color='coral')
        ax.set_title('Return Risk – Feature Importance',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'return_risk_features.png'), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. FESTIVE DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
FESTIVE_MONTHS = {
    'Diwali':         [10, 11],
    'Wedding Season': [11, 12, 1, 2],
    "Valentine's":    [2],
    'Summer Sale':    [5, 6],
}

class FestiveDemandForecaster:
    MODEL_PATH = os.path.join(MODEL_DIR, 'demand_forecast.joblib')

    def __init__(self):
        self.model    = GradientBoostingRegressor(n_estimators=200, random_state=42)
        self.scaler   = StandardScaler()
        self.trained  = False
        self._cat_map = {}

    def train(self, df: pd.DataFrame) -> dict:
        # Auto-scale trees based on data size
        n_trees = 50 if len(df) < 100 else 200
        self.model = GradientBoostingRegressor(n_estimators=n_trees, random_state=42)
        monthly = (df.groupby(['Year', 'Month', 'Category'])
                     .agg(Demand=('OrderID', 'count'),
                          AvgRevenue=('TotalAmount', 'mean'))
                     .reset_index())
        cats = monthly['Category'].unique()
        self._cat_map       = {c: i for i, c in enumerate(cats)}
        monthly['CatCode']  = monthly['Category'].map(self._cat_map)
        monthly['IsFestive'] = monthly['Month'].apply(
            lambda m: 1 if any(m in v for v in FESTIVE_MONTHS.values()) else 0)

        feats = ['Year', 'Month', 'CatCode', 'IsFestive', 'AvgRevenue']
        X     = monthly[feats].values
        y     = monthly['Demand'].values
        X_sc  = self.scaler.fit_transform(X)

        cv     = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2  = cross_val_score(self.model, X_sc, y, cv=cv, scoring='r2')
        cv_mae = cross_val_score(self.model, X_sc, y, cv=cv,
                                  scoring='neg_mean_absolute_error')

        self.model.fit(X_sc, y)
        self.trained  = True
        self._monthly = monthly

        joblib.dump({'model': self.model, 'scaler': self.scaler,
                     'cat_map': self._cat_map, 'monthly': monthly},
                    self.MODEL_PATH)
        return {
            'r2':     round(r2_score(y, self.model.predict(X_sc)), 3),
            'mae':    round(mean_absolute_error(y, self.model.predict(X_sc)), 2),
            'cv_r2':  round(cv_r2.mean(), 3),
            'cv_mae': round(-cv_mae.mean(), 2),
        }

    def load(self) -> bool:
        if os.path.exists(self.MODEL_PATH):
            data          = joblib.load(self.MODEL_PATH)
            self.model    = data['model']
            self.scaler   = data['scaler']
            self._cat_map = data['cat_map']
            self._monthly = data['monthly']
            self.trained  = True
            return True
        return False

    def predict_next_months(self, months_ahead: int = 6) -> pd.DataFrame:
        last      = self._monthly['Month'].max()
        last_year = self._monthly['Year'].max()
        rows = []
        for i in range(1, months_ahead + 1):
            m = (last + i - 1) % 12 + 1
            y = last_year + (last + i - 1) // 12
            for cat, code in self._cat_map.items():
                is_f  = 1 if any(m in v for v in FESTIVE_MONTHS.values()) else 0
                avg_r = self._monthly[self._monthly['Category'] == cat]['AvgRevenue'].mean()
                rows.append({'Year': y, 'Month': m, 'Category': cat,
                             'CatCode': code, 'IsFestive': is_f,
                             'AvgRevenue': avg_r})
        pred_df = pd.DataFrame(rows)
        X = self.scaler.transform(pred_df[['Year', 'Month', 'CatCode',
                                            'IsFestive', 'AvgRevenue']])
        pred_df['PredictedDemand'] = np.maximum(0, self.model.predict(X)).round(1)
        return pred_df[['Year', 'Month', 'Category', 'IsFestive', 'PredictedDemand']]

    def plot_forecast(self, forecast_df: pd.DataFrame) -> str:
        pivot = forecast_df.pivot_table(index='Month', columns='Category',
                                         values='PredictedDemand', aggfunc='sum')
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(ax=ax, marker='o', linewidth=2)
        ax.set_xlabel('Month')
        ax.set_ylabel('Predicted Demand (Orders)')
        ax.set_title('Festive Demand Forecast – Next 6 Months',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, 'festive_forecast.png')
        plt.savefig(path, dpi=150)
        plt.close()
        return path


# ══════════════════════════════════════════════════════════════════════════════
# 3. CUSTOMER LOYALTY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
class LoyaltyPredictor:
    MODEL_PATH = os.path.join(MODEL_DIR, 'loyalty.joblib')

    def __init__(self):
        self.model   = None
        self.trained = False
        self._feats  = ['Recency', 'Frequency', 'Monetary', 'CLV_Score']

    def train(self, rfm: pd.DataFrame) -> dict:
        n_trees = 30 if len(rfm) < 100 else 100
        self.model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
        rfm = rfm.copy()
        rfm['IsLoyal'] = rfm['Segment'].isin(
            ['Champions', 'Loyal Customers']).astype(int)
        X = rfm[self._feats].fillna(0)
        y = rfm['IsLoyal']

        cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                    random_state=42)
        self.model.fit(X_tr, y_tr)
        self.trained = True
        joblib.dump({'model': self.model}, self.MODEL_PATH)
        return {
            'accuracy': round(self.model.score(X_te, y_te) * 100, 2),
            'cv_mean':  round(cv_scores.mean() * 100, 2),
            'cv_std':   round(cv_scores.std()  * 100, 2),
        }

    def load(self) -> bool:
        if os.path.exists(self.MODEL_PATH):
            self.model   = joblib.load(self.MODEL_PATH)['model']
            self.trained = True
            return True
        return False

    def predict(self, rfm: pd.DataFrame) -> pd.DataFrame:
        X   = rfm[self._feats].fillna(0)
        rfm = rfm.copy()
        rfm['LoyaltyProb'] = (self.model.predict_proba(X)[:, 1] * 100).round(1)
        rfm['WillBeLoyal'] = rfm['LoyaltyProb'] >= 60
        return rfm.sort_values('LoyaltyProb', ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4. SMART DISCOUNT OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
def optimize_discount(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for cat, grp in df.groupby('Category'):
        for disc in range(0, 30, 5):
            mask = grp['Discount'].between(disc, disc + 4)
            if mask.sum() < 3:
                continue
            avg_pm   = grp.loc[mask, 'ProfitMargin'].mean()
            avg_qty  = grp.loc[mask, 'Quantity'].mean()
            exp_rev  = grp.loc[mask, 'TotalAmount'].mean()
            exp_prof = exp_rev * avg_pm / 100
            results.append({'Category': cat, 'Discount': disc,
                             'AvgProfitMargin': round(avg_pm, 1),
                             'AvgQty': round(avg_qty, 2),
                             'ExpectedProfit': round(exp_prof, 2)})
    res_df   = pd.DataFrame(results)
    best_idx = res_df.groupby('Category')['ExpectedProfit'].idxmax()
    optimal  = res_df.loc[best_idx].reset_index(drop=True)
    optimal.columns = ['Category', 'OptimalDiscount', 'AvgProfitMargin',
                        'AvgQty', 'MaxExpectedProfit']
    return optimal


# ══════════════════════════════════════════════════════════════════════════════
# 5. MARKET BASKET ANALYSIS  ← NEW
# ══════════════════════════════════════════════════════════════════════════════
def run_market_basket(df: pd.DataFrame,
                      min_support: float = 0.05,
                      min_confidence: float = 0.2) -> pd.DataFrame:
    """
    Finds which jewellery categories are frequently bought together
    by the same customer using Apriori (or co-occurrence fallback).
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        from mlxtend.preprocessing import TransactionEncoder

        transactions = df.groupby('CustomerID')['Category'].apply(list).tolist()
        te     = TransactionEncoder()
        te_arr = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_arr, columns=te.columns_)

        frequent = apriori(basket, min_support=min_support, use_colnames=True)
        if frequent.empty:
            return _cooccurrence_basket(df)

        rules = association_rules(frequent, metric='confidence',
                                  min_threshold=min_confidence)
        if rules.empty:
            return _cooccurrence_basket(df)

        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        result = (rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
                    .sort_values('lift', ascending=False)
                    .round(3)
                    .reset_index(drop=True))
        result.columns = ['If Customer Buys', 'They Also Buy',
                           'Support', 'Confidence', 'Lift']
        return result

    except ImportError:
        return _cooccurrence_basket(df)


def _cooccurrence_basket(df: pd.DataFrame) -> pd.DataFrame:
    """Manual co-occurrence fallback when mlxtend is not installed."""
    co = {}
    cat_counts = df.groupby('Category')['CustomerID'].nunique()
    total      = df['CustomerID'].nunique()

    for _, grp in df.groupby('CustomerID'):
        cats = grp['Category'].unique().tolist()
        for i in range(len(cats)):
            for j in range(len(cats)):
                if i != j:
                    key = (cats[i], cats[j])
                    co[key] = co.get(key, 0) + 1

    rows = []
    for (a, b), cnt in co.items():
        sup  = round(cnt / total, 3)
        conf = round(cnt / cat_counts.get(a, 1), 3)
        lift = round(conf / (cat_counts.get(b, 1) / total), 3)
        if sup >= 0.05 and conf >= 0.2:
            rows.append({'If Customer Buys': a, 'They Also Buy': b,
                         'Support': sup, 'Confidence': conf, 'Lift': lift})

    result = (pd.DataFrame(rows)
                .sort_values('Lift', ascending=False)
                .drop_duplicates(subset=['If Customer Buys', 'They Also Buy'])
                .head(15)
                .reset_index(drop=True))
    return result if not result.empty else pd.DataFrame(
        columns=['If Customer Buys', 'They Also Buy',
                 'Support', 'Confidence', 'Lift'])