import joblib
import pandas as pd
import numpy as np
import logging
import os
import requests
import json
import websocket
import ssl
from datetime import datetime, time
import pytz
import time as time_module

# Set up logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SniperXAUUSDBot:
    """
    SNIPER XAUUSD BOT: Dedicated High-Precision Signal Generator via Deriv WebSocket.
    Dilengkapi Startup Notification, Anti-State Memory, & Reversal Alert.
    """
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.state_file = "sniper_state.json"
        self.startup_file = "sniper_startup.json"

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                model = joblib.load(path)
                logging.info(f"✅ Sukses memuat model AI XAUUSD dari {path}")
                return model
            except Exception as e:
                logging.warning(f"⚠️ Gagal memuat model dari {path}: {e}")
        return None

    def fetch_market_data(self, count: int = 100) -> pd.DataFrame:
        """Menarik data candles XAUUSD real-time tina WebSocket Deriv (Akurat & Presisi)."""
        symbols = ["frxXAUUSD", "XAUUSD", "gold"]
        app_id = "1089"
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        for s in symbols:
            ws = None
            try:
                ws = websocket.create_connection(ws_url, timeout=8, sslopt={"cert_reqs": ssl.CERT_NONE})
                req = {
                    "ticks_history": s,
                    "count": count,
                    "end": "latest",
                    "granularity": 300, # TF 5 Menit
                    "style": "candles"
                }
                ws.send(json.dumps(req))
                res = json.loads(ws.recv())
                ws.close()
                
                if "candles" in res and len(res["candles"]) > 0:
                    df = pd.DataFrame(res["candles"])
                    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
                    for col in ['Close', 'Open', 'High', 'Low']:
                        df[col] = df[col].astype(float)
                    
                    current_p = df['Close'].iloc[-1]
                    logging.info(f"✅ Sukses tarik data XAUUSD via Deriv WS ({s}) | Harga Terkini: {current_p:.2f}")
                    return df
            except Exception as e:
                logging.warning(f"⚠️ Gagal narik data {s} via WS: {e}")
            finally:
                if ws:
                    try: ws.close()
                    except: pass
                    
        logging.error("❌ Gagal total narik data pasar via WebSocket Deriv!")
        return pd.DataFrame()

    def load_last_signal(self) -> str:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f).get("signal", None)
            except: pass
        return None

    def save_last_signal(self, signal: str):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"signal": signal}, f)
        except: pass

    def calculate_sniper_strategy(self) -> dict:
        df = self.fetch_market_data()
        if df.empty or len(df) < 40:
            return {"valid": False}

        close = df['Close'].values
        high = df['High'].values
        low = df['Low'].values
        open_p = df['Open'].values
        
        current_price = close[-1]

        # A. Volatility & Range Filter (ATR 14)
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high[-1] - low[-1]))
        
        # B. Momentum Candle Body
        body_size = abs(close[-1] - open_p[-1])
        avg_body = np.mean(abs(close[-10:] - open_p[-10:]))

        # C. Donchian / Breakout Channel (5 Candle Terakhir)
        highest_high_5 = np.max(high[-6:-1])
        lowest_low_5 = np.min(low[-6:-1])

        # D. RSI Calculation (14)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])

        # Syarat Eksekusi Sniper Pro
        is_buy_signal = (
            (current_price > highest_high_5) and 
            (body_size > (avg_body * 1.2)) and 
            (atr >= 1.5) and 
            (current_rsi < 68) and
            (close[-1] > open_p[-1])
        )

        is_sell_signal = (
            (current_price < lowest_low_5) and 
            (body_size > (avg_body * 1.2)) and 
            (atr >= 1.5) and 
            (current_rsi > 32) and
            (close[-1] < open_p[-1])
        )

        if is_buy_signal:
            return {
                "valid": True, "signal": "BUY", "price": current_price,
                "atr": atr, "rsi": current_rsi,
                "sl": current_price - 35.0, "tp": current_price + 75.0
            }
        elif is_sell_signal:
            return {
                "valid": True, "signal": "SELL", "price": current_price,
                "atr": atr, "rsi": current_rsi,
                "sl": current_price + 35.0, "tp": current_price - 75.0
            }

        return {"valid": False}

def send_telegram(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logging.warning("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID tidak terdeteksi.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        if res.status_code == 200:
            logging.info("Notifikasi Telegram berhasil terkirim.")
        else:
            logging.error(f"Gagal kirim pesan Telegram: {res.text}")
    except Exception as e:
        logging.error(f"Error Telegram API: {e}")

def format_sniper_card(res: dict, is_reversal: bool = False) -> str:
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    badge = "🟢🟢 **[SNIPER PRO: LAWAN ARAH / REVERSAL BUY]** 🟢🟢" if res['signal'] == "BUY" and is_reversal else \
            ("🔴🔴 **[SNIPER PRO: LAWAN ARAH / REVERSAL SELL]** 🔴🔴" if res['signal'] == "SELL" and is_reversal else \
            ("🟢🟢 **[SNIPER PRO EXCLUSIVE: STRONG BUY]** 🟢🟢" if res['signal'] == "BUY" else "🔴🔴 **[SNIPER PRO EXCLUSIVE: STRONG SELL]** 🔴🔴"))

    return (
        f"🔥 *[XAUUSD SNIPER SYSTEM - KARYA UTAMA]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *EKSEKUSI*: {badge}\n"
        f"💵 *Harga Masuk*: `{res['price']:.2f}`\n"
        f"📊 *ATR / RSI*: `{res['atr']:.2f} / RSI: {res['rsi']:.1f}`\n"
        "-------------------------------------\n"
        f"🛑 *Stop Loss*: `{res['sl']:.2f}`\n"
        f"🎯 *Take Profit*: `{res['tp']:.2f}`\n"
        "-------------------------------------\n"
        f"⏰ *WAKTU*: `{wib}`"
    )

if __name__ == "__main__":
    bot = SniperXAUUSDBot(model_path='model_xauusd.pkl')
    now_wib = datetime.now(bot.wib_tz)

    # 1. Startup Notification (Sakali)
    if not os.path.exists(bot.startup_file):
        df_s = bot.fetch_market_data(count=5)
        p_s = float(df_s['Close'].iloc[-1]) if not df_s.empty else 0.0
        send_telegram(f"🚀 *[SNIPER BOT AKTIF]*\n✅ Sistem Eksklusif Siap Tempur!\n💵 Harga XAUUSD: `{p_s:.2f}`\n⏰ `{now_wib.strftime('%H:%M:%S WIB')}`")
        try:
            with open(bot.startup_file, "w") as f: json.dump({"ok": True}, f)
        except: pass

    last_daily_report_date = None

    # Loop utama monitoring
    while True:
        now_wib = datetime.now(bot.wib_tz)
        
        # 2. Laporan Subuh 06:00 WIB
        if now_wib.hour == 6 and now_wib.minute == 0:
            if last_daily_report_date != now_wib.date():
                df_m = bot.fetch_market_data(count=5)
                p_m = float(df_m['Close'].iloc[-1]) if not df_m.empty else 0.0
                send_telegram(f"🌅 *[LAPORAN SUBUH SNIPER]*\n🟢 Bot Siaga Senén-Jumaah\n💵 Harga: `{p_m:.2f}`")
                last_daily_report_date = now_wib.date()

        # 3. Eksekusi Sinyal Berkelanjutan
        signal = bot.calculate_sniper_strategy()
        if signal["valid"]:
            curr_sig = signal["signal"]
            last_sig = bot.load_last_signal()
            
            if curr_sig != last_sig:
                is_rev = last_sig is not None
                msg = format_sniper_card(signal, is_reversal=is_rev)
                send_telegram(msg)
                bot.save_last_signal(curr_sig)
                logging.info(f"✅ Sinyal {curr_sig} berhasil dikirim!")
            else:
                logging.info(f"⏳ Sinyal masih {curr_sig} (Aman, teu ngirim duplikat).")
        else:
            logging.info("⏳ Market XAUUSD di-skip (Teu acan nyumponan sarat rumus Sniper).")

        # Jeda 60 detik sateuacan mariksa deui pasar
        time_module.sleep(60)
