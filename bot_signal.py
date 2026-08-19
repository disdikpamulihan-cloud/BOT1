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
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.state_file = "sniper_state.json"
        self.startup_file = "sniper_startup.json"
        self.warning_state_file = "sniper_warning_state.json"

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
                except Exception as e:
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
        """Fungsi pikeun nganalisis pasar sarta ngadeteksi potensi sinyal sateuacanna (Persiapan OP)."""
        df = self.fetch_market_data(count=250)
        if df.empty or len(df) < 205:
            return {"valid": False, "warning": False}

        df_closed = df.iloc[:-1] 
        close = df_closed['Close']
        high = df_closed['High']
        low = df_closed['Low']
        open_p = df_closed['Open']
        current_price = df['Close'].iloc[-1]

        ma200 = close.rolling(window=200).mean().iloc[-1]
        ma50 = close.rolling(window=50).mean().iloc[-1]
        
        tr = np.maximum(high.values[1:] - low.values[1:], np.maximum(abs(high.values[1:] - close.values[:-1]), abs(low.values[1:] - close.values[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high.iloc[-1] - low.iloc[-1]))
        
        rsi_series = self.calculate_rsi(close, 14)
        rsi = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

        body_size = abs(close.iloc[-1] - open_p.iloc[-1])
        avg_body = np.mean(abs(close.iloc[-10:] - open_p.iloc[-10:]))
        highest_high_5 = np.max(high.values[-6:-1])
        lowest_low_5 = np.min(low.values[-6:-1])

        is_buy = (current_price > ma200) and (current_price > highest_high_5) and (atr >= 0.5)
        is_sell = (current_price < ma200) and (current_price < lowest_low_5) and (atr >= 0.5)

        # Cek Prediksi AI lamun aya
        ai_approval = True
        if self.model_xauusd is not None:
            try:
                features = np.array([[float(atr), float(body_size), float(current_price - ma200), float(current_price - ma50), float(rsi)]])
                ai_pred = self.model_xauusd.predict(features)[0]
                if is_buy and ai_pred == 0: ai_approval = False
                if is_sell and ai_pred == 1: ai_approval = False
            except: pass

        # Validasi Sinyal Matang (Full OP)
        is_buy_signal = is_buy and ai_approval and (body_size > (avg_body * 1.2)) and (close.iloc[-1] > open_p.iloc[-1])
        is_sell_signal = is_sell and ai_approval and (body_size > (avg_body * 1.2)) and (close.iloc[-1] < open_p.iloc[-1])

        if is_buy_signal:
            return {"valid": True, "warning": False, "signal": "BUY", "price": current_price, "atr": atr, "sl": current_price - (atr * 1.5), "tp": current_price + (atr * 3.0)}
        elif is_sell_signal:
            return {"valid": True, "warning": False, "signal": "SELL", "price": current_price, "atr": atr, "sl": current_price + (atr * 1.5), "tp": current_price - (atr * 3.0)}

        # Deteksi Persiapan (Warning 5 Menit sateuacan matang)
        is_warning_buy = is_buy and ai_approval and (close.iloc[-1] > open_p.iloc[-1])
        is_warning_sell = is_sell and ai_approval and (close.iloc[-1] < open_p.iloc[-1])

        if is_warning_buy:
            return {"valid": False, "warning": True, "signal": "BUY", "price": current_price, "atr": atr}
        elif is_warning_sell:
            return {"valid": False, "warning": True, "signal": "SELL", "price": current_price, "atr": atr}

        return {"valid": False, "warning": False}

def send_telegram(message: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
    except: pass

def format_warning_card(res: dict) -> str:
    wib = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%H:%M:%S WIB')
    sig = res['signal']
    return (
        f"⚠️ *[PERSIAPAN OP XAUUSD]* ⚠️\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔔 *Aba-aba*: Tendensi pasar nuju nyiapkeun sinyal *{sig}*!\n"
        f"💵 *Harga Pantau*: `{res['price']:.2f}`\n"
        f"📊 *ATR*: `{res['atr']:.2f}`\n"
        "💡 *Saran*: Siapkeun posisi, kantosan konfirmasi 1-2 candle deui!\n"
        f"⏰ *WAKTU*: `{wib}`"
    )

def format_sniper_card(res: dict, is_reversal: bool = False) -> str:
    wib = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%Y-%m-%d %H:%M:%S WIB')
    badge = "🟢🟢 **[AI SNIPER: STRONG BUY]** 🟢🟢" if res['signal'] == "BUY" else "🔴🔴 **[AI SNIPER: STRONG SELL]** 🔴🔴"
    return (
        f"🔥 *[EKSEKUSI OP XAUUSD]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *STATUS*: {badge}\n"
        f"💵 *Harga Masuk (OP)*: `{res['price']:.2f}`\n"
        f"🛑 *Stop Loss*: `{res['sl']:.2f}`\n"
        f"🎯 *Take Profit*: `{res['tp']:.2f}`\n"
        "-------------------------------------\n"
        f"⏰ *WAKTU EKSEKUSI*: `{wib}`"
    )

if __name__ == "__main__":
    bot = SniperXAUUSDBot(model_path='model_xauusd.pkl')
    
    # Loop utama monitoring
    while True:
        now_wib = datetime.now(bot.wib_tz)
        
        # Analisis Pasar
        market = bot.analyze_market_condition()
        
        if market["valid"]:
            curr_sig = market["signal"]
            last_sig = bot.load_last_signal()
            if curr_sig != last_sig:
                is_rev = last_sig is not None
                msg = format_sniper_card(market, is_reversal=is_rev)
                send_telegram(msg)
                bot.save_last_signal(curr_sig)
                logging.info(f"✅ Sinyal Eksekusi {curr_sig} dikirim!")
        elif market["warning"]:
            # Kirim aba-aba persiapan (bisa diatur supaya teu ngirim spam unggal menit ku simpen state samentawis)
            logging.info(f"⚠️ Market nuju masihan aba-aba persiapan {market['signal']}")
            send_telegram(format_warning_card(market))
            # Jeda rada lila sakedik sangkan teu spam
            time_module.sleep(120)

        time_module.sleep(60)
