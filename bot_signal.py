#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HIGH-FREQUENCY HIGH-ACCURACY MATRIX BOT - XAUUSD
Features: Dynamic XGBoost/LightGBM Loading + ADX Sideways Filter + Multi-Target TP
"""

import os
import json
import logging
import requests
import websocket
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

DERIV_APP_ID = "1089"
DERIV_API_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"
SYMBOL = "frxXAUUSD"

# Parameter Risiko & Probabilitas AI
PROB_THRESHOLD = 0.62   # Keyakinan AI minimal 62%
SL_ATR_MULT = 1.5       # SL Rapat = 1.5 x ATR
TP1_ATR_MULT = 1.5      # TP1 (Scalp Risk Free) = 1.5 x ATR
TP2_ATR_MULT = 3.0      # TP2 (Trend Runner) = 3.0 x ATR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n--- NOTIFIKASI TELEGRAM ---")
        print(message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        logging.error(f"Error Telegram: {e}")

def fetch_candles(granularity=900, count=100):
    ws = websocket.create_connection(DERIV_API_URL, timeout=20)
    req = {
        "ticks_history": SYMBOL,
        "adjust_start_time": 1,
        "count": count,
        "end": "latest",
        "style": "candles",
        "granularity": granularity
    }
    ws.send(json.dumps(req))
    resp = ws.recv()
    ws.close()
    
    data = json.loads(resp)
    if "candles" not in data:
        raise RuntimeError(f"Gagal mengambil data candle: {data}")
    
    df = pd.DataFrame(data["candles"])
    df['time'] = pd.to_datetime(df['epoch'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def extract_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # RSI Wilder
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR & EMA Ratios
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema_ratio'] = df['ema_50'] / df['ema_200']
    
    # ADX Calculation
    up_move = df['high'].diff()
    down_move = df['low'].diff().abs()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    
    tr = pd.concat([df['high'] - df['low'], 
                    (df['high'] - df['close'].shift(1)).abs(), 
                    (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df['adx'] = dx.rolling(14).mean()

    df.dropna(inplace=True)
    return df

def main():
    model_file = "model_xauusd.pkl"
    if not os.path.exists(model_file):
        # Fallback jika nama file menggunakan versi 99
        if os.path.exists("model_xauusd_99.pkl"):
            model_file = "model_xauusd_99.pkl"
        else:
            logging.error("File model .pkl tidak ditemukan di repository!")
            return

    bundle = joblib.load(model_file)
    
    # Deteksi struktur bundle (Single model atau Multi-model XGBoost/LGBM)
    if "model" in bundle:
        model = bundle["model"]
    elif "lgb_model" in bundle:
        model = bundle["lgb_model"]
    elif "xgb_model" in bundle:
        model = bundle["xgb_model"]
    else:
        logging.error("Struktur bundle model tidak dikenali.")
        return

    scaler = bundle["scaler"]
    features = bundle["features"]

    # Ambil Data & Ekstrak Fitur M15
    df_m15 = fetch_candles(granularity=900, count=100)
    df_feat = extract_features(df_m15)
    
    last_row = df_feat[features].iloc[[-1]]
    last_close = df_m15['close'].iloc[-1]
    last_atr = df_feat['atr'].iloc[-1]
    last_adx = df_feat['adx'].iloc[-1]

    # Prediksi Probabilitas
    X_scaled = scaler.transform(last_row.values)
    prob_up = model.predict_proba(X_scaled)[0][1]

    signal = "HOLD"
    sl, tp1, tp2 = 0.0, 0.0, 0.0

    # Logika Eksekusi Sinyal
    if prob_up >= PROB_THRESHOLD and last_adx >= 20.0:
        signal = "BUY"
        sl = last_close - (last_atr * SL_ATR_MULT)
        tp1 = last_close + (last_atr * TP1_ATR_MULT)
        tp2 = last_close + (last_atr * TP2_ATR_MULT)
    elif prob_up <= (1.0 - PROB_THRESHOLD) and last_adx >= 20.0:
        signal = "SELL"
        sl = last_close + (last_atr * SL_ATR_MULT)
        tp1 = last_close - (last_atr * TP1_ATR_MULT)
        tp2 = last_close - (last_atr * TP2_ATR_MULT)

    icon = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")
    
    msg = f"⚡ <b>AI MATRIX SIGNAL (XAUUSD)</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"{icon} Sinyal Eksekusi: <b>{signal}</b>\n"
    msg += f"💵 Harga Saat Ini: <code>{last_close:.2f}</code>\n"
    msg += f"🧠 Keyakinan AI: <b>{max(prob_up, 1-prob_up) * 100:.1f}%</b>\n"
    msg += f"📈 ADX Trend Strength: <b>{last_adx:.1f}</b>\n"
    msg += f"-------------------------------------\n"
    
    if signal != "HOLD":
        msg += f"🛑 Stop Loss (SL): <code>{sl:.2f}</code>\n"
        msg += f"🎯 Target TP 1 (Scalp): <code>{tp1:.2f}</code>\n"
        msg += f"🚀 Target TP 2 (Runner): <code>{tp2:.2f}</code>\n"
    else:
        msg += f"💡 <i>Status: WAIT & SEE (Kondisi belum ideal)</i>\n"
        
    msg += f"-------------------------------------\n"
    msg += f"⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB</i>"

    send_telegram(msg)

if __name__ == "__main__":
    main()
