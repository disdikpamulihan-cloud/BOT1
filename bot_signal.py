#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRODUCTION SIGNAL BOT - KHUSUS XAUUSD (GOLD) M15
Fungsi: Membaca 'model_xauusd.pkl', menganalisis candle real-time Deriv,
        menghitung SL/TP ATR, dan mengirim notifikasi ke Telegram.
"""

import os
import json
import time
import logging
import requests
import websocket
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Credential Telegram dari Environment Variables / Secrets
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# Konfigurasi Deriv API
DERIV_APP_ID = "1089"
DERIV_API_URL = f"wss://ws.binaryws.com/websockets/v3?app_id={DERIV_APP_ID}"

# Konfigurasi Khusus XAUUSD
SYMBOL = "frxXAUUSD"
GRANULARITY = 900       # Timeframe M15 (900 detik)
SL_ATR_MULT = 2.0       # Jarak Stop Loss = 2.0 x ATR
TP_RR_MULT = 2.0        # Risk-to-Reward Ratio = 1:2 (Take Profit = 2 x SL)
PROB_THRESHOLD = 0.58   # Ambang batas probabilitas AI (58%)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def send_telegram(message):
    """Mengirim pesan notifikasi ke Telegram"""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n--- NOTIFIKASI TELEGRAM (MODE TESTING LOKAL) ---")
        print(message)
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        response = requests.post(url, data=payload, timeout=10)
        if response.status_code == 200:
            logging.info("Notifikasi sinyal XAUUSD berhasil dikirim ke Telegram!")
        else:
            logging.error(f"Gagal mengirim ke Telegram. Response: {response.text}")
    except Exception as e:
        logging.error(f"Error saat mengirim notifikasi Telegram: {e}")

def fetch_live_candles(symbol=SYMBOL, granularity=GRANULARITY, count=60):
    """Mengambil candle real-time terbaru dari Deriv WebSocket API"""
    ws = websocket.create_connection(DERIV_API_URL, timeout=15)
    req = {
        "ticks_history": symbol,
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
        raise RuntimeError(f"Gagal mendapatkan data live dari Deriv: {data}")
    
    df = pd.DataFrame(data["candles"])
    df['time'] = pd.to_datetime(df['epoch'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def extract_features(df):
    """Menghitung indikator teknikal sesuai fitur pelatihan model XAUUSD"""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # RSI Wilder
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR & MA Ratio
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['sma_fast'] = df['close'].rolling(7).mean()
    df['sma_slow'] = df['close'].rolling(25).mean()
    df['ma_ratio'] = df['sma_fast'] / df['sma_slow']
    
    df.dropna(inplace=True)
    return df

def main():
    logging.info("Memulai analisis sinyal AI Khusus XAUUSD...")
    
    model_file = "model_xauusd.pkl"
    if not os.path.exists(model_file):
        logging.error(f"File '{model_file}' tidak ditemukan! Unggah file .pkl ke repository.")
        return

    # Load Model Bundle
    bundle = joblib.load(model_file)
    model = bundle["model"]
    scaler = bundle["scaler"]
    features = bundle["features"]

    # Ambil Data & Extrak Fitur
    df = fetch_live_candles()
    df_feat = extract_features(df)
    
    last_row = df_feat[features].iloc[[-1]]
    last_close = df['close'].iloc[-1]
    last_atr = df_feat['atr'].iloc[-1]

    # Prediksi Probabilitas menggunakan Model AI
    X_scaled = scaler.transform(last_row.values)
    prob_up = model.predict_proba(X_scaled)[0][1]

    signal = "HOLD"
    sl, tp = 0.0, 0.0

    # Logika Eksekusi Sinyal & Kalkulasi Risk Management
    if prob_up >= PROB_THRESHOLD:
        signal = "BUY"
        sl = last_close - (last_atr * SL_ATR_MULT)
        tp = last_close + (last_atr * SL_ATR_MULT * TP_RR_MULT)
    elif prob_up <= (1.0 - PROB_THRESHOLD):
        signal = "SELL"
        sl = last_close + (last_atr * SL_ATR_MULT)
        tp = last_close - (last_atr * SL_ATR_MULT * TP_RR_MULT)

    # Susun Pesan Telegram
    icon = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "⚪")
    
    msg = f"🥇 <b>AI SIGNAL XAUUSD (GOLD) M15</b>\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"{icon} Sinyal Eksekusi: <b>{signal}</b>\n"
    msg += f"💵 Harga Saat Ini: <code>{last_close:.2f}</code>\n"
    msg += f"📊 Probabilitas Naik (Bullish): <b>{prob_up * 100:.1f}%</b>\n"
    msg += f"📊 Probabilitas Turun (Bearish): <b>{(1.0 - prob_up) * 100:.1f}%</b>\n"
    msg += f"-------------------------------------\n"
    
    if signal != "HOLD":
        msg += f"🛑 Stop Loss (SL): <code>{sl:.2f}</code>\n"
        msg += f"🎯 Take Profit (TP): <code>{tp:.2f}</code>\n"
        msg += f"⚖️ Risk/Reward Ratio: 1:{TP_RR_MULT}\n"
    else:
        msg += f"💡 <i>Kondisi pasar belum memenuhi ambang batas ({PROB_THRESHOLD*100:.0f}%). Disarankan WAIT & SEE.</i>\n"
        
    msg += f"-------------------------------------\n"
    msg += f"⏰ <i>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB</i>"

    # Kirim Pesan
    send_telegram(msg)

if __name__ == "__main__":
    main()
