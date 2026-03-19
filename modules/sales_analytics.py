"""
modules/sales_analytics.py
Core sales analytics: top products, revenue trends, city-wise demand.
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

PALETTE = 'Set2'


def top_categories(df: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    return (df.groupby('Category')
              .agg(TotalRevenue=('TotalAmount', 'sum'),
                   OrderCount=('OrderID', 'count'),
                   AvgRating=('FeedbackRating', 'mean'))
              .sort_values('TotalRevenue', ascending=False)
              .head(n)
              .round(2))


def top_materials(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby('Material')
              .agg(TotalRevenue=('TotalAmount', 'sum'),
                   AvgProfit=('ProfitMargin', 'mean'))
              .sort_values('TotalRevenue', ascending=False)
              .round(2))


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(['Year', 'Month'])
              .agg(Revenue=('TotalAmount', 'sum'),
                   Orders=('OrderID', 'count'))
              .reset_index()
              .sort_values(['Year', 'Month']))


def city_demand(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby('Location')
              .agg(Revenue=('TotalAmount', 'sum'),
                   Orders=('OrderID', 'count'),
                   AvgOrderValue=('TotalAmount', 'mean'))
              .sort_values('Revenue', ascending=False)
              .round(2))


def plot_category_revenue(df: pd.DataFrame) -> str:
    data = top_categories(df)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(data.index, data['TotalRevenue'],
                   color=sns.color_palette(PALETTE, len(data)))
    ax.set_xlabel('Total Revenue (₹)')
    ax.set_title('Revenue by Jewellery Category', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, data['TotalRevenue']):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
                f'₹{val:,.0f}', va='center', fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'category_revenue.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_monthly_trend(df: pd.DataFrame) -> str:
    data = monthly_trend(df)
    data['Period'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data['Period'], data['Revenue'], marker='o', linewidth=2, color='steelblue')
    ax.fill_between(data['Period'], data['Revenue'], alpha=0.15, color='steelblue')
    ax.set_xticklabels(data['Period'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Revenue (₹)')
    ax.set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'monthly_trend.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_city_heatmap(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(values='TotalAmount', index='Location',
                           columns='Category', aggfunc='sum', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, annot_kws={'size': 8})
    ax.set_title('City × Category Revenue Heatmap (₹)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'city_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path
