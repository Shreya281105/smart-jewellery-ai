"""
utils/data_loader.py
Loads and preprocesses the jewellery sales dataset.
"""
import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'jewellery_sales.csv')


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['OrderDate'])
    df = _clean(df)
    df = _engineer(df)
    return df


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df.dropna(subset=['OrderID', 'CustomerID', 'TotalAmount'], inplace=True)
    df['FeedbackRating'] = df['FeedbackRating'].clip(1, 5)
    df['Discount']       = df['Discount'].clip(0, 50)
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df['Month']         = df['OrderDate'].dt.month
    df['DayOfWeek']     = df['OrderDate'].dt.dayofweek
    df['Quarter']       = df['OrderDate'].dt.quarter
    df['Year']          = df['OrderDate'].dt.year
    df['IsWeekend']     = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['IsReturned']    = (df['ReturnStatus'] == 'Returned').astype(int)
    df['IsReturning']   = (df['CustomerType'] == 'Returning').astype(int)
    df['RevenuePerUnit'] = df['TotalAmount'] / df['Quantity'].replace(0, 1)
    df['ActualProfit']  = df['TotalAmount'] * df['ProfitMargin'] / 100
    return df


def get_summary(df: pd.DataFrame) -> dict:
    return {
        'total_orders':    len(df),
        'total_revenue':   round(df['TotalAmount'].sum(), 2),
        'total_profit':    round(df['ActualProfit'].sum(), 2),
        'unique_customers': df['CustomerID'].nunique(),
        'avg_order_value': round(df['TotalAmount'].mean(), 2),
        'return_rate':     round(df['IsReturned'].mean() * 100, 2),
        'avg_rating':      round(df['FeedbackRating'].mean(), 2),
    }
