import os, numpy as np, pandas as pd
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST
from stable_baselines3 import PPO
import yfinance as yf
from data import add_features

TICKER="SPY"; WINDOW=48; QTY=1
TAKE_PROFIT_PCT=0.012; STOP_LOSS_PCT=0.008
FEATURES=["ret_1","ret_5","ret_20","ma_ratio_10","ma_ratio_20","rsi_14","macd","macd_sig","macd_diff","atr","vol_z"]

def fetch_latest_5m(t="SPY", lookback="3d"):
    df = yf.download(t, period=lookback, interval="5m", auto_adjust=True, progress=False)
    if df is None or df.empty: return None
    if df.index.tz is None: df.index = pd.to_datetime(df.index, utc=True)
    return df.tz_convert("America/New_York").dropna()

def build_obs(df):
    df_feat = add_features(df)
    if len(df_feat) < WINDOW: return None, None
    x = df_feat[FEATURES].values[-WINDOW:, :].reshape(-1).astype(np.float32)
    return x, float(df_feat["Close"].iloc[-1])

def main():
    load_dotenv()
    api = REST(os.getenv("ALPACA_API_KEY_ID"), os.getenv("ALPACA_API_SECRET_KEY"),
               base_url=os.getenv("ALPACA_BASE_URL","https://paper-api.alpaca.markets"))
    try:
        if not api.get_clock().is_open:
            print("Market closed → skip"); return
    except Exception:
        pass
    model = PPO.load("ppo_intraday_longflat", device="cpu")
    df = fetch_latest_5m(TICKER, "3d")
    if df is None or df.empty: print("No data"); return
    obs, last_price = build_obs(df)
    if obs is None: print("Not enough history"); return
    has_long=False
    try:
        pos = api.get_position(TICKER); has_long = float(pos.qty) > 0
    except Exception: pass
    action,_ = model.predict(obs, deterministic=True)  # 0 hold, 1 long, 2 flat
    if action == 1 and not has_long:
        tp=round(last_price*(1+TAKE_PROFIT_PCT),2); sl=round(last_price*(1-STOP_LOSS_PCT),2)
        print(f"BUY {QTY} {TICKER} @~{last_price:.2f} TP={tp} SL={sl}")
        api.submit_order(symbol=TICKER, qty=QTY, side="buy", type="market",
                         time_in_force="day", order_class="bracket",
                         take_profit={"limit_price": tp}, stop_loss={"stop_price": sl})
    elif action == 2 and has_long:
        print("CLOSE position"); api.close_position(TICKER)
    else:
        print("HOLD")

if __name__ == "__main__":
    main()
