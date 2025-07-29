"""feature_engineering.py – 100+ feature skeleton"""
import pandas as pd, numpy as np, pandas_ta as ta

def build_feature_matrix(df_raw: pd.DataFrame, window:int=3)->pd.DataFrame:
    df=df_raw.copy()
    df['mid']=(df['best_bid']+df['best_ask'])/2
    bb=ta.bbands(df['mid'], length=20)
    df=pd.concat([df, bb], axis=1)
    feats=df.rolling(window).agg(['mean','std']).fillna(method='bfill')
    feats.columns=['_'.join(c) for c in feats.columns]
    return feats.reset_index(drop=True)