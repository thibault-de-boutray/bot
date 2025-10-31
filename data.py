import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import AverageTrueRange

def _ensure_single_ticker_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalise la sortie yfinance en colonnes simples: Open, High, Low, Close, Volume.
    Gère le cas MultiIndex (ex: ('Close','SPY')).
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Essaye de sélectionner le niveau correspondant au ticker
        lvls = [lev.astype(str).tolist() for lev in df.columns.levels]
        try:
            # le ticker est souvent au niveau 1: ('Open','SPY')
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        except Exception:
            try:
                # sinon au niveau 0: ('SPY','Open')
                df = df.xs(ticker, axis=1, level=0, drop_level=True)
            except Exception:
                # dernier recours: si un seul symbole, aplatis le premier niveau
                df = df.droplevel(0, axis=1)

    # Ne garder que les colonnes standard si présentes
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if not keep:
        raise ValueError(f"Colonnes OHLCV introuvables après normalisation. Colonnes: {df.columns}")
    df = df[keep].copy()
    return df

def download_5m(ticker: str = "SPY", period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("Empty dataframe from yfinance. Try another ticker/period/interval.")

    df = _ensure_single_ticker_ohlcv(df, ticker)

    # Assure l'index temporel avec timezone, puis convertit en Amérique/New_York
    if df.index.tz is None:
        df.index = pd.to_datetime(df.index, utc=True)
    else:
        df.index = df.index.tz_convert("UTC")
    df.index = df.index.tz_convert("America/New_York")

    # Filtre plage horaire (éviter ouverture/finale trop volatiles)
    df = df.between_time("09:35", "15:55")
    df = df.dropna().copy()
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # S'assurer que 'Close' est une Series float
    close = pd.to_numeric(out["Close"], errors="coerce")
    high  = pd.to_numeric(out.get("High", close), errors="coerce")
    low   = pd.to_numeric(out.get("Low", close), errors="coerce")
    vol   = pd.to_numeric(out.get("Volume", 0), errors="coerce")

    # Retours
    out["ret_1"] = close.pct_change().fillna(0.0)
    out["ret_5"] = close.pct_change(5).fillna(0.0)
    out["ret_20"] = close.pct_change(20).fillna(0.0)

    # Moyennes mobiles / ratios
    out["ma_10"] = close.rolling(10).mean()
    out["ma_20"] = close.rolling(20).mean()
    out["ma_ratio_10"] = close / out["ma_10"]
    out["ma_ratio_20"] = close / out["ma_20"]

    # RSI
    rsi = RSIIndicator(close, window=14)
    out["rsi_14"] = rsi.rsi()

    # MACD
    macd = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    out["macd"] = macd.macd()
    out["macd_sig"] = macd.macd_signal()
    out["macd_diff"] = macd.macd_diff()

    # ATR (volatilité)
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    out["atr"] = atr.average_true_range()

    # Volume z-score intra-période
    vol_roll_mean = vol.rolling(50).mean()
    vol_roll_std = vol.rolling(50).std()
    out["vol_z"] = (vol - vol_roll_mean) / (vol_roll_std + 1e-8)

    # Drop nan au début des indicateurs
    out = out.dropna().copy()

    # Normalisation robuste (médiane / IQR) colonne par colonne
    feats = ["ret_1","ret_5","ret_20","ma_ratio_10","ma_ratio_20","rsi_14","macd","macd_sig","macd_diff","atr","vol_z"]
    for f in feats:
        med = out[f].median()
        iqr = (out[f].quantile(0.75) - out[f].quantile(0.25)) + 1e-9
        out[f] = (out[f] - med) / iqr

    return out

def load_dataset(ticker="SPY"):
    df = download_5m(ticker=ticker, period="60d", interval="5m")
    df = add_features(df)
    # Marquer les journées pour les épisodes
    df["date"] = df.index.date
    return df
