"""
modules/customer_analysis.py
Customer segmentation, RFM scoring, and CLV detection.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── RFM Scoring ────────────────────────────────────────────────────────────────
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df['OrderDate'].max() + pd.Timedelta(days=1)

    rfm = (df.groupby('CustomerID')
             .agg(
                 Recency   = ('OrderDate',   lambda x: (snapshot - x.max()).days),
                 Frequency = ('OrderID',     'count'),
                 Monetary  = ('TotalAmount', 'sum'),
             )
             .reset_index())

    for col, label in [('Recency', 'R'), ('Frequency', 'F'), ('Monetary', 'M')]:
        ascending = col == 'Recency'          # lower recency = better
        rfm[label + '_Score'] = pd.qcut(
            rfm[col].rank(method='first'),
            q=4,
            labels=[4, 3, 2, 1] if ascending else [1, 2, 3, 4]
        ).astype(int)

    rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    rfm['Segment']   = rfm['RFM_Total'].apply(_segment_label)
    rfm['CLV_Score'] = (rfm['Monetary'] * rfm['Frequency'] / (rfm['Recency'] + 1)).round(2)
    return rfm


def _segment_label(score: int) -> str:
    if score >= 10:
        return 'Champions'
    elif score >= 8:
        return 'Loyal Customers'
    elif score >= 6:
        return 'Potential Loyalists'
    elif score >= 4:
        return 'At Risk'
    else:
        return 'Lost / Inactive'


# ── Demographic Analysis ───────────────────────────────────────────────────────
def age_gender_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(['AgeGroup', 'Gender'])
              .agg(Orders=('OrderID', 'count'),
                   Revenue=('TotalAmount', 'sum'))
              .reset_index()
              .round(2))


def new_vs_returning(df: pd.DataFrame) -> dict:
    grp = df.groupby('CustomerType')['TotalAmount'].agg(['sum', 'count'])
    total_rev = grp['sum'].sum()
    return {
        t: {
            'orders':      int(grp.loc[t, 'count']),
            'revenue':     round(float(grp.loc[t, 'sum']), 2),
            'revenue_pct': round(float(grp.loc[t, 'sum']) / total_rev * 100, 1)
        }
        for t in grp.index
    }


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_rfm_segments(rfm: pd.DataFrame) -> str:
    seg_counts = rfm['Segment'].value_counts()
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22', '#e74c3c']
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        seg_counts, labels=seg_counts.index, autopct='%1.1f%%',
        colors=colors[:len(seg_counts)], startangle=140,
        pctdistance=0.82, wedgeprops=dict(width=0.55))
    for t in autotexts:
        t.set_fontsize(10)
    ax.set_title('Customer Segments (RFM)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'rfm_segments.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_age_spending(df: pd.DataFrame) -> str:
    data = (df.groupby('AgeGroup')['TotalAmount']
              .mean()
              .reset_index()
              .sort_values('TotalAmount', ascending=False))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(data['AgeGroup'], data['TotalAmount'],
                  color=sns.color_palette('pastel'))
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Avg Order Value (₹)')
    ax.set_title('Average Spending by Age Group', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, data['TotalAmount']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                f'₹{val:,.0f}', ha='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'age_spending.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path
