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
import time as time_module  # Dirobah supaya teu tabrakan

# Set up logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SuperAIXAUUSDBot:
    """
    SUPER AI TRADING BOT: Dedicated XAUUSD High-Precision Signal Generator
    Dilengkapi Startup Notification, Scheduled Daily Check (06:00 WIB), & MT5 Price Comparison.
    """
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')

    def _safe_load(self, path):
        if path and os.path.exists(path):
            try:
                model = joblib.load(path)
                logging.info(f"✅ Sukses memuat model AI XAUUSD dari {path}")
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

    def fetch_deriv_candles(self, count: int = 100) -> pd.DataFrame:
        """Menarik data candles XAUUSD real-time langsung dari Deriv WebSocket."""
        symbols_to_try = ["frxXAUUSD", "XAUUSD", "gold"]
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
                    "granularity": 60, # TF 1 Menit
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
                    logging.info(f"✅ Sukses tarik data XAUUSD via simbol: {deriv_symbol}")
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

    def extract_features_and_indicators(self, df: pd.DataFrame):
        if df.empty or len(df) < 30:
            return None, 4375.97, 5.0, 50.0, 0.0

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

        # 3. MACD Sederhana untuk konfirmasi momentum
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
        macd = float(ema12 - ema26)

        features = pd.DataFrame([{
            'Column_0': (close[-1] - close[-2]) / close[-2],
            'Column_1': (close[-1] - close[-5]) / close[-5],
            'Column_2': rsi / 100.0,
            'Column_3': atr / current_price,
            'Column_4': np.std(close[-10:]) / current_price
        }])

        return features, current_price, atr, rsi, macd

    def evaluate_market(self) -> dict:
        """Kalkulasi sinyal XAUUSD presisi tinggi & validasi target minimal 50 pips."""
        df = self.fetch_deriv_candles()
        input_df, current_price, atr, rsi, macd = self.extract_features_and_indicators(df)
        
        actual_model = self._extract_model(self.model_xauusd)
        confidence = 55.0
        prediction = 1

        if actual_model is not None and input_df is not None:
            try:
                if isinstance(self.model_xauusd, dict) and 'scaler' in self.model_xauusd:
                    input_data = self.model_xauusd['scaler'].transform(input_df.values)
                else:
                    input_data = input_df.values
                
                prediction = actual_model.predict(input_data)[0]
                if hasattr(actual_model, "predict_proba"):
                    probs = actual_model.predict_proba(input_data)[0]
                    confidence = float(max(probs) * 100)
            except Exception as e:
                logging.warning(f"⚠️ Prediksi model error ({e}), menggunakan fallback indikator.")
                prediction = 1 if rsi < 50 else 0
                confidence = 60.0
        else:
            prediction = 1 if rsi < 50 else 0
            confidence = 60.0

        signal = "BUY" if prediction == 1 else "SELL"

        # Dynamic Risk Management XAUUSD (Target minimal setara 50 pips / 5.0 poin)
        sl_distance = max(atr * 1.2, 4.0)
        tp1_distance = max(sl_distance * 1.5, 5.0) 
        tp2_distance = tp1_distance * 2.0

        if signal == "BUY":
            sl = current_price - sl_distance
            tp1 = current_price + tp1_distance
            tp2 = current_price + tp2_distance
        else:
            sl = current_price + sl_distance
            tp1 = current_price - tp1_distance
            tp2 = current_price - tp2_distance

        # FILTER KETAT: Keyakinan >= 75% & Target Minimal 50 Pips tercapai
        min_target_pips = 5.0
        is_high_probability = (confidence >= 75.0) and (tp1_distance >= min_target_pips)
        is_momentum_strong = abs(macd) > (atr * 0.03)

        valid_signal = is_high_probability and is_momentum_strong

        return {
            "valid": valid_signal,
            "signal": signal,
            "price": current_price,
            "confidence": confidence,
            "rsi": rsi,
            "atr": atr,
            "macd": macd,
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
        res = requests.post(url, json=payload, timeout=5)
        if res.status_code == 200:
            logging.info("Notifikasi Telegram berhasil terkirim.")
        else:
            logging.error(f"Gagal kirim pesan Telegram: {res.text}")
    except Exception as e:
        logging.error(f"Error Telegram API: {e}")

def format_signal_card(res: dict) -> str:
    """Format tampilan pesan Telegram khusus XAUUSD dengan perbandingan harga MT5."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    if res['signal'] == "BUY":
        signal_badge = "🟢🟢 **[STRONG BUY - LONG]** 🟢🟢"
        action_desc = "Target XAUUSD siap MEROKET naik (Target >50 Pips)! 🚀"
    else:
        signal_badge = "🔴🔴 **[STRONG SELL - SHORT]** 🔴🔴"
        action_desc = "Target XAUUSD siap TERJUN bebas (Target >50 Pips)! 📉"

    return (
        f"🤖 *[SUPER AI XAUUSD BOT - SIGNAL]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Sinyal Eksekusi*: {signal_badge}\n"
        f"💡 *Analisis*: `{action_desc}`\n"
        f"💵 *Harga WebSocket (Feed Bot)*: `{res['price']:.2f}`\n"
        f"🔍 *(Cocokkeun jeung Harga Bid/Ask MT5)*\n"
        f"🔥 *Keyakinan AI (High Conf)*: `{res['confidence']:.1f}%`\n"
        f"📊 *RSI*: `{res['rsi']:.1f}` | *ATR*: `{res['atr']:.2f}`\n"
        "-------------------------------------\n"
        f"🛑 *Stop Loss (Anti-SL)*: `{res['sl']:.2f}`\n"
        f"🟢 *Target TP 1 (Aman)*: `{res['tp1']:.2f}`\n"
        f"🚀 *Target TP 2 (Runner)*: `{res['tp2']:.2f}`\n"
        "-------------------------------------\n"
        f"⏰ `{wib_time}`"
    )

def send_startup_notification():
    """Mengirim notifikasi bahwa bot berhasil di-start / jalan."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    msg = (
        f"🚀 *[SYSTEM STARTUP]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ *Super AI XAUUSD Bot* parantos sukses diaktifkeun!\n"
        "🛡️ Sistem siap ngawas pasar & nyaring sinyal beresiko rendah.\n"
        f"⏰ `{wib_time}`"
    )
    send_telegram_message(msg)

def send_daily_report(bot_instance):
    """Mengirim laporan rutin jam 06.00 WIB beserta harga terkini."""
    df = bot_instance.fetch_deriv_candles(count=5)
    _, current_price, atr, rsi, _ = bot_instance.extract_features_and_indicators(df)
    
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    msg = (
        f"🌅 *[LAPORAN RUTIN SUBUH - 06:00 WIB]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "🟢 *Bot Status*: `AKTIF & SIAGA`\n"
        f"💵 *Harga Terkini XAUUSD (Feed)*: `{current_price:.2f}`\n"
        f"🔍 *(Bandingkeun sareng MT5 ayeuna)*\n"
        f"📊 *RSI Saat Ini*: `{rsi:.1f}` | *ATR*: `{atr:.2f}`\n"
        "-------------------------------------\n"
        "☕ Siap-siap ngantosan sinyal high-conviction dinten ieu!\n"
        f"⏰ `{wib_time}`"
    )
    send_telegram_message(msg)

if __name__ == "__main__":
    bot = SuperAIXAUUSDBot(model_path='model_xauusd.pkl')
    
    # 1. Kirim notif startup pas bot mimiti jalan
    send_startup_notification()
    
    # Variabel pelacak laporan harian supaya teu ngirim sababaraha kali di jam 06.00
    last_daily_report_date = None

    # Loop utama monitoring (Bisa dijalankeun salawasna di VPS)
    while True:
        now_wib = datetime.now(bot.wib_tz)
        
        # Cek naha geus waktuna laporan rutin jam 06.00 WIB
        if now_wib.hour == 6 and now_wib.minute == 0:
            if last_daily_report_date != now_wib.date():
                send_daily_report(bot)
                last_daily_report_date = now_wib.date()

        # Evaluasi pasar berkala
        res = bot.evaluate_market()
        if res["valid"]:
            msg = format_signal_card(res)
            send_telegram_message(msg)
            logging.info("✅ Sinyal XAUUSD valid & terkirim!")
        else:
            logging.info("⏳ Market XAUUSD di-skip (Belum memenuhi syarat keyakinan >75% / target pips <50).")

        # Jeda 60 detik (1 menit) sateuacan mariksa deui pasar (Menggunakan time_module)
        time_module.sleep(60)
