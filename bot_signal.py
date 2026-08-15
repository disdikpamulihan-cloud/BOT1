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

# Set up logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DualMarketSignalBot:
    """
    BOT 1: Upgrade to Deriv WebSocket Live Feed for XAUUSD & Volatility 80 (Synced with MT5)
    """
    def __init__(self, model_xau_path: str, model_vol_path: str):
        """Memuat kedua model ML saat inisialisasi."""
        self.model_xauusd = self._safe_load(model_xau_path)
        self.model_vol80 = self._safe_load(model_vol_path)

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                model = joblib.load(path)
                logging.info(f"✅ Sukses memuat model AI dari {path}")
                return model
            except Exception as e:
                logging.warning(f"⚠️ Gagal memuat model dari {path}: {e}")
        return None

    def _extract_model(self, model_obj):
        if isinstance(model_obj, dict):
            for key in ['model', 'estimator', 'lgbm', 'classifier', 'xgboost']:
                if key in model_obj:
                    return model_obj[key]
            return list(model_obj.values())[0]
        return model_obj

    def fetch_deriv_candles(self, symbol: str, count: int = 100) -> pd.DataFrame:
        """
        Menerik data candles real-time langsung dari Deriv WebSocket dengan multi-symbol fallback 
        supaya akurat sinkron jeung MT5.
        """
        if symbol == 'XAUUSD':
            symbols_to_try = ["frxXAUUSD", "XAUUSD", "gold"]
        else:
            symbols_to_try = ["R_80", "R80", "VOLT80"]

        app_id = "1089"
        ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        
        for deriv_symbol in symbols_to_try:
            ws = None
            try:
                ws = websocket.create_connection(ws_url, timeout=8, sslopt={"cert_reqs": ssl.CERT_NONE})
                req = {
                    "ticks_history": deriv_symbol,
                    "count": count,
                    "end": "latest",
                    "granularity": 60, # TF 1 Menit supados hargana peka & akurat
                    "style": "candles"
                }
                ws.send(json.dumps(req))
                res = json.loads(ws.recv())
                ws.close()
                
                if "candles" in res and len(res["candles"]) > 0:
                    df = pd.DataFrame(res["candles"])
                    df.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low'}, inplace=True)
                    df['Close'] = df['Close'].astype(float)
                    df['High'] = df['High'].astype(float)
                    df['Low'] = df['Low'].astype(float)
                    logging.info(f"✅ Sukses tarik data {symbol} via simbol: {deriv_symbol}")
                    return df
            except Exception as e:
                logging.warning(f"⚠️ Gagal dengan simbol {deriv_symbol}: {e}")
            finally:
                if ws:
                    try:
                        ws.close()
                    except:
                        pass
                        
        return pd.DataFrame()

    def extract_features_and_indicators(self, df: pd.DataFrame, symbol: str):
        if df.empty or len(df) < 30:
            default_price = 4375.97 if symbol == 'XAUUSD' else 250357.0
            default_atr = 5.0 if symbol == 'XAUUSD' else 500.0
            return None, default_price, default_atr, 50.0

        close = np.array(df['Close'].values, dtype=float).ravel()
        high = np.array(df['High'].values, dtype=float).ravel()
        low = np.array(df['Low'].values, dtype=float).ravel()

        current_price = float(close[-1])

        # 1. RSI (14)
        delta = np.diff(close)
        gain = np.mean(delta[delta > 0][-14:]) if len(delta[delta > 0]) > 0 else 0
        loss = -np.mean(delta[delta < 0][-14:]) if len(delta[delta < 0]) > 0 else 1e-6
        rsi = float(100 - (100 / (1 + gain/loss)))

        # 2. ATR (Average True Range)
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-14:]) if len(tr) >= 14 else (high[-1] - low[-1]))

        # Fitur input dinamis tina datacandles asli
        features = pd.DataFrame([{
            'Column_0': (close[-1] - close[-2]) / close[-2],
            'Column_1': (close[-1] - close[-5]) / close[-5],
            'Column_2': rsi / 100.0,
            'Column_3': atr / current_price,
            'Column_4': np.std(close[-10:]) / current_price
        }])

        return features, current_price, atr, rsi

    def predict_market(self, symbol: str, model_wrapper) -> dict:
        """Kalkulasi sinyal, keyakinan AI, indikator, dan manajemen risiko berdasarkan data real-time."""
        
        # Tarik data candles real-time via WebSocket
        df = self.fetch_deriv_candles(symbol)
        input_df, current_price, atr, rsi = self.extract_features_and_indicators(df, symbol)
        
        actual_model = self._extract_model(model_wrapper)
        
        if actual_model is not None and input_df is not None:
            try:
                if isinstance(model_wrapper, dict) and 'scaler' in model_wrapper:
                    input_data = model_wrapper['scaler'].transform(input_df.values)
                else:
                    input_data = input_df.values
                
                prediction = actual_model.predict(input_data)[0]
                signal = "BUY" if prediction == 1 else "SELL"
                
                confidence = 65.0
                if hasattr(actual_model, "predict_proba"):
                    probs = actual_model.predict_proba(input_data)[0]
                    confidence = float(max(probs) * 100)
            except Exception as e:
                logging.warning(f"⚠️ Prediksi model error ({e}), menggunakan fallback indikator RSI.")
                signal = "BUY" if rsi < 50 else "SELL"
                confidence = 55.0
        else:
            signal = "BUY" if rsi < 50 else "SELL"
            confidence = 55.0

        # ATR-based Dynamic SL & TP (Supados adaptif jeung volatilitas pasar)
        if symbol == 'XAUUSD':
            sl_distance = max(atr * 1.5, 6.0)
            tp1_distance = sl_distance * 1.0
            tp2_distance = sl_distance * 2.0
        else:
            sl_distance = max(atr * 1.5, 150.0)
            tp1_distance = sl_distance * 1.0
            tp2_distance = sl_distance * 2.0

        if signal == "BUY":
            sl = current_price - sl_distance
            tp1 = current_price + tp1_distance
            tp2 = current_price + tp2_distance
        else: # SELL
            sl = current_price + sl_distance
            tp1 = current_price - tp1_distance
            tp2 = current_price - tp2_distance

        return {
            "signal": signal,
            "price": current_price,
            "confidence": confidence,
            "rsi": rsi,
            "atr": atr,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2
        }

def send_telegram_message(message: str):
    """Mengirim notifikasi ke Telegram."""
    bot_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logging.warning("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID tidak terdeteksi.")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        requests.post(url, json=payload, timeout=5)
        logging.info("Notifikasi Telegram berhasil terkirim.")
    except Exception as e:
        logging.error(f"Error Telegram API: {e}")

def format_signal_card(symbol: str, res: dict) -> str:
    """Format tampilan pesan Telegram."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    dec = 2
    
    return (
        f"🤖 *AI MATRIX SIGNAL ({symbol})*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Sinyal Eksekusi*: `{res['signal']}`\n"
        f"💵 *Harga Real-Time MT5 Feed*: `{res['price']:.{dec}f}`\n"
        f"🔥 *Keyakinan AI*: `{res['confidence']:.1f}%`\n"
        f"📊 *RSI (14)*: `{res['rsi']:.1f}` | *ATR*: `{res['atr']:.2f}`\n"
        "-------------------------------------\n"
        f"🔴 *Stop Loss (SL)*: `{res['sl']:.{dec}f}`\n"
        f"🟢 *Target TP 1 (Scalp)*: `{res['tp1']:.{dec}f}`\n"
        f"🟢 *Target TP 2 (Runner)*: `{res['tp2']:.{dec}f}`\n"
        "-------------------------------------\n"
        f"⏰ `{wib_time}`"
    )

if __name__ == "__main__":
    bot = DualMarketSignalBot(
        model_xau_path='model_xauusd.pkl',
        model_vol_path='model_vol80.pkl'
    )
    
    # 1. Kalkulasi sinyal & harga real-time via WebSocket Deriv (sinkron MT5)
    res_xau = bot.predict_market('XAUUSD', bot.model_xauusd)
    res_vol = bot.predict_market('VOLATILITY 80', bot.model_vol80)
    
    # 2. Kirim pesan ke Telegram
    send_telegram_message(format_signal_card('XAUUSD', res_xau))
    send_telegram_message(format_signal_card('VOLATILITY 80', res_vol))
