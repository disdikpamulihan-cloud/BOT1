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
    SNIPER XAUUSD AI BOT: Dedicated High-Precision Signal Generator via Deriv WebSocket.
    Dilengkapi AI Model Integration, Anti-State Memory, & Reversal Alert.
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

    def fetch_market_data(self, count: int = 250) -> pd.DataFrame:
        """Menarik data candles XAUUSD real-time tina WebSocket Deriv kalayan sistem Retry."""
        symbols = ["frxXAUUSD", "XAUUSD", "gold"]
        app_id = "1089"
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        for attempt in range(3):
            for s in symbols:
                ws = None
                try:
                    ws = websocket.create_connection(ws_url, timeout=10, sslopt={"cert_reqs": ssl.CERT_NONE})
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
                    logging.warning(f"⚠️ Gagal narik data {s} (Attempt {attempt+1}): {e}")
                    time_module.sleep(2)
                finally:
                    if ws:
                        try: ws.close()
                        except: pass
            if attempt < 2:
                time_module.sleep(3)
                    
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
        df = self.fetch_market_data(count=250)
        if df.empty or len(df) < 205:
            return {"valid": False}

        # 1. Anti-Repainting: Ngan ukur ngolah candle anu geus tutup
        df_closed = df.iloc[:-1] 
        
        close = df_closed['Close'].values
        high = df_closed['High'].values
        low = df_closed['Low'].values
        open_p = df_closed['Open'].values
        current_price = df['Close'].iloc[-1]

        # 2. Filter Trend MA-200
        ma200 = pd.Series(close).rolling(window=200).mean().iloc[-1]

        # 3. Volatility & ATR (14)
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high[-1] - low[-1]))
        
        # 4. Momentum & Donchian Channel
        body_size = abs(close[-1] - open_p[-1])
        avg_body = np.mean(abs(close[-10:] - open_p[-10:]))
        highest_high_5 = np.max(high[-6:-1])
        lowest_low_5 = np.min(low[-6:-1])

        # Syarat Eksekusi Sniper + Filter Trend MA-200
        is_buy_signal = (
            (current_price > ma200) and
            (current_price > highest_high_5) and 
            (body_size > (avg_body * 1.2)) and 
            (atr >= 0.5) and 
            (close[-1] > open_p[-1])
        )

        is_sell_signal = (
            (current_price < ma200) and
            (current_price < lowest_low_5) and 
            (body_size > (avg_body * 1.2)) and 
            (atr >= 0.5) and 
            (close[-1] < open_p[-1])
        )

        # Integrasi Model AI Baru (Aman & Sinkron sareng format train_ai.py)
        if self.model_xauusd is not None:
            try:
                features = np.array([[atr, body_size, current_price - ma200]])
                ai_pred = self.model_xauusd.predict(features)[0]
                if ai_pred == 0: # Upami model AI nyatakeun turun / teu layak
                    is_buy_signal = False
                elif ai_pred == 1 and not is_buy_signal and current_price > ma200:
                    # AI nguatkeun sinyal buy upami tren ngarojong
                    pass
            except Exception as e:
                logging.warning(f"⚠️ Catetan prediksi AI: {e}")

        if is_buy_signal:
            return {
                "valid": True, "signal": "BUY", "price": current_price,
                "atr": atr, "sl": current_price - (atr * 1.5), "tp": current_price + (atr * 3.0)
            }
        elif is_sell_signal:
            return {
                "valid": True, "signal": "SELL", "price": current_price,
                "atr": atr, "sl": current_price + (atr * 1.5), "tp": current_price - (atr * 3.0)
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
    
    badge = "🟢🟢 **[AI SNIPER: LAWAN ARAH / REVERSAL BUY]** 🟢🟢" if res['signal'] == "BUY" and is_reversal else \
            ("🔴🔴 **[AI SNIPER: LAWAN ARAH / REVERSAL SELL]** 🔴🔴" if res['signal'] == "SELL" and is_reversal else \
            ("🟢🟢 **[AI SNIPER EXCLUSIVE: STRONG BUY]** 🟢🟢" if res['signal'] == "BUY" else "🔴🔴 **[AI SNIPER EXCLUSIVE: STRONG SELL]** 🔴🔴"))

    return (
        f"🔥 *[XAUUSD AI SNIPER SYSTEM]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *EKSEKUSI*: {badge}\n"
        f"💵 *Harga Masuk*: `{res['price']:.2f}`\n"
        f"📊 *ATR*: `{res['atr']:.2f}`\n"
        "-------------------------------------\n"
        f"🛑 *Stop Loss (Dynamic)*: `{res['sl']:.2f}`\n"
        f"🎯 *Take Profit (Dynamic)*: `{res['tp']:.2f}`\n"
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
        send_telegram(f"🚀 *[AI SNIPER BOT AKTIF]*\n✅ Sistem Siap Tempur!\n💵 Harga XAUUSD: `{p_s:.2f}`\n⏰ `{now_wib.strftime('%H:%M:%S WIB')}`")
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
                send_telegram(f"🌅 *[LAPORAN SUBUH AI SNIPER]*\n🟢 Bot Siaga Senén-Jumaah\n💵 Harga: `{p_m:.2f}`")
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
            logging.info("⏳ Market XAUUSD di-skip (Teu acan nyumponan sarat rumus AI Sniper).")

        # Jeda 60 detik sateuacan mariksa deui pasar
        time_module.sleep(60)
