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
            return {"valid": False}

        close = df['Close']
        high = df['High']
        low = df['Low']
        current_price = close.iloc[-1]

        ma200 = close.rolling(window=200).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1]
        
        tr = np.maximum(high.values[1:] - low.values[1:], np.maximum(abs(high.values[1:] - close.values[:-1]), abs(low.values[1:] - close.values[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high.iloc[-1] - low.iloc[-1]))
        
        rsi_series = self.calculate_rsi(close, 14)
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        # Logika Sinyal Sederhana tapi Pasti (Trend Following + RSI + AI)
        is_buy = (current_price > ma200) and (rsi < 70)
        is_sell = (current_price < ma200) and (rsi > 30)

        if self.model_xauusd is not None:
            try:
                features = np.array([[float(atr), float(abs(close.iloc[-1] - df['Open'].iloc[-1])), float(current_price - ma200), float(current_price - ma50), float(rsi)]])
                ai_pred = self.model_xauusd.predict(features)[0]
                if ai_pred == 1:
                    return {"valid": True, "signal": "BUY", "price": current_price, "sl": current_price - (atr * 1.5), "tp": current_price + (atr * 3.0)}
                else:
                    return {"valid": True, "signal": "SELL", "price": current_price, "sl": current_price + (atr * 1.5), "tp": current_price - (atr * 3.0)}
            except: pass

        # Fallback lamun AI teu aktip
        if is_buy:
            return {"valid": True, "signal": "BUY", "price": current_price, "sl": current_price - (atr * 1.5), "tp": current_price + (atr * 3.0)}
        elif is_sell:
            return {"valid": True, "signal": "SELL", "price": current_price, "sl": current_price + (atr * 1.5), "tp": current_price - (atr * 3.0)}

        return {"valid": False}

def send_telegram(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except: pass

def format_sniper_card(res: dict) -> str:
    wib = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M:%S WIB')
    badge = "🟢🟢 **[AI SNIPER: BUY SIGNAL]** 🟢🟢" if res['signal'] == "BUY" else "🔴🔴 **[AI SNIPER: SELL SIGNAL]** 🔴🔴"
    return (
        f"🔥 *[SINYAL EKSEKUSI XAUUSD]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *STATUS*: {badge}\n"
        f"💵 *Harga Masuk (OP)*: `{res['price']:.2f}`\n"
        f"🛑 *Stop Loss*: `{res['sl']:.2f}`\n"
        f"🎯 *Take Profit*: `{res['tp']:.2f}`\n"
        "-------------------------------------\n"
        f"⏰ *WAKTU*: `{wib}`"
    )

if __name__ == "__main__":
    bot = SniperXAUUSDBot(model_path='model_xauusd.pkl')
    
    market = bot.analyze_market_condition()
    if market.get("valid"):
        curr_sig = market["signal"]
        last_sig = bot.load_last_signal()
        
        # NGAN KODE 📉 NGAN KIRIM NOTIF PAS SINYALNA BÉDA (ANTI-SPAM UTAMA)
        if curr_sig != last_sig:
            msg = format_sniper_card(market)
            send_telegram(msg)
            bot.save_last_signal(curr_sig)
            logging.info(f"✅ Sinyal {curr_sig} suksés dikirim ka Telegram!")
        else:
            logging.info(f"ℹ️ Sinyal masih tetep {curr_sig}, teu dikirim (Anti-spam aktif).")
    else:
        logging.info("ℹ️ Belum ada sinyal valid.")
