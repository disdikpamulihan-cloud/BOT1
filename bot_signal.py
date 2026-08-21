import joblib
import pandas as pd
import numpy as np
import logging
import os
import requests
import json
import websocket
import ssl
from datetime import datetime
import pytz
import time as time_module

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SniperXAUUSDBot:
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.state_file = "sniper_state.json"

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
                        return df
                except Exception:
                    time_module.sleep(2)
                finally:
                    if ws:
                        try: ws.close()
                        except: pass
            if attempt < 2:
                time_module.sleep(3)
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

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def analyze_market_condition(self) -> dict:
        df = self.fetch_market_data(count=250)
        if df.empty or len(df) < 205:
            return {"valid": False, "conf": 0.0}

        df_closed = df.iloc[:-1]
        close = df_closed['Close']
        high = df_closed['High']
        low = df_closed['Low']
        open_p = df_closed['Open']
        current_price = df['Close'].iloc[-1]

        ma200 = close.rolling(window=200).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1]
        
        tr = np.maximum(high.values[1:] - low.values[1:], np.maximum(abs(high.values[1:] - close.values[:-1]), abs(low.values[1:] - close.values[:-1])))
        atr_series = pd.Series(tr).rolling(window=14).mean()
        atr = float(atr_series.iloc[-1]) if not atr_series.empty else 1.0
        avg_atr = float(atr_series.rolling(20).mean().iloc[-1]) if len(atr_series) >= 20 else atr
        
        rsi_series = self.calculate_rsi(close, 14)
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0
        
        body_size = float(abs(close.iloc[-1] - open_p.iloc[-1]))

        confidence = 0.0
        ai_pred = None
        if self.model_xauusd is not None:
            try:
                features = np.array([[
                    atr, 
                    body_size, 
                    float(current_price - ma200), 
                    float(current_price - ma50), 
                    rsi
                ]])
                
                if hasattr(self.model_xauusd, "predict_proba"):
                    probs = self.model_xauusd.predict_proba(features)[0]
                    ai_pred = int(np.argmax(probs))
                    confidence = float(np.max(probs))
                else:
                    ai_pred = int(self.model_xauusd.predict(features)[0])
                    confidence = 0.85

                # FILTER KETAT: Minimal akurasi 80% (0.80)
                if confidence >= 0.80:
                    # --- RUMUS SL & TP DINAMIS (Nyesuaikeun Pasar Real) ---
                    volatility_ratio = atr / avg_atr if avg_atr > 0 else 1.0
                    dynamic_multiplier = np.clip(volatility_ratio, 0.7, 1.6)

                    sl_distance = atr * (1.1 * dynamic_multiplier)
                    tp_distance = atr * (2.8 * dynamic_multiplier)

                    if ai_pred == 1:
                        return {
                            "valid": True, "signal": "BUY", "price": current_price, "conf": confidence,
                            "sl": current_price - sl_distance, "tp": current_price + tp_distance
                        }
                    else:
                        return {
                            "valid": True, "signal": "SELL", "price": current_price, "conf": confidence,
                            "sl": current_price + sl_distance, "tp": current_price - tp_distance
                        }
            except Exception as e:
                logging.warning(f"AI Prediction Error: {e}")

        return {"valid": False, "conf": confidence, "price": current_price}

def send_telegram(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except: pass

if __name__ == "__main__":
    bot = SniperXAUUSDBot(model_path='model_xauusd.pkl')
    
    market = bot.analyze_market_condition()
    if market.get("valid"):
        curr_sig = market["signal"]
        last_sig = bot.load_last_signal()
        
        if curr_sig != last_sig:
            if curr_sig == "BUY":
                sig_display = "🟢 `BUY`"
            else:
                sig_display = "🔴 `SELL`"

            card = (
                f"🚨 **BOT SNIPER XAUUSD UPDATED!** 🚨\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"• **EKSEKUSI** : {sig_display}\n"
                f"• **HARGA REAL** : `{market['price']:.2f}`\n"
                f"• **AKURASI LIVE** : `{market['conf']*100:.2f}%`\n"
                f"• **OPEN POSISI** : `{market['price']:.2f}`\n"
                f"• **TP** : `{market['tp']:.2f}`\n"
                f"• **SL** : `{market['sl']:.2f}`\n"
                f"• **WAKTU** : `{datetime.now(bot.wib_tz).strftime('%Y-%m-%d %H:%M:%S')} WIB`"
            )
            send_telegram(card)
            bot.save_last_signal(curr_sig)
            logging.info(f"✅ Sinyal {curr_sig} suksés dikirim ka Telegram!")
        else:
            logging.info(f"ℹ️ Sinyal masih tetep {curr_sig}, teu dikirim (Anti-spam aktif).")
    else:
        # Notif status live disumputkeun tina Telegram (mung kacatet di log console)
        live_conf = market.get("conf", 0.0) * 100
        live_price = market.get("price", 0.0)
        logging.info(f"ℹ️ Bot Sniper Aktif | Harga: {live_price:.2f} | Akurasi AI: {live_conf:.2f}% (Menunggu akurasi >= 80%)")
