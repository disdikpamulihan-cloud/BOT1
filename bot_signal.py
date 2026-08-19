import joblib
import pandas as pd
import numpy as np
import logging
import os
import requests
import json
import yfinance as yf
from datetime import datetime, time
import pytz
import time as time_module

# Set up logging profesional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SuperAIXAUUSDBot:
    """
    SUPER AI TRADING BOT: Dedicated XAUUSD High-Precision Signal Generator
    Dilengkapi Startup Notification, Anti-Spam State Memory, & Reversal Alert.
    """
    def __init__(self, model_path: str = "model_xauusd.pkl"):
        self.model_xauusd = self._safe_load(model_path)
        self.wib_tz = pytz.timezone('Asia/Jakarta')
        self.last_sent_signal = None  # Pelacak sinyal sateuacanna supados henteu spam

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

    def fetch_deriv_candles(self, count: int = 150) -> pd.DataFrame:
        """Menarik data candles XAUUSD real-time tina Yahoo Finance (Akurat & Tembus Firewall GitHub)."""
        try:
            # Tarik data XAUUSD tina Yahoo Finance (ticker: GC=F atanapi XAU-USD)
            ticker = "GC=F" 
            df = yf.download(ticker, period="5d", interval="5m", progress=False)
            
            if df.empty:
                ticker = "XAU-USD"
                df = yf.download(ticker, period="5d", interval="5m", progress=False)

            if not df.empty:
                # Ngaberesihkeun multi-index kolom ti yfinance upami aya
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                df = df.tail(count).copy()
                df.reset_index(inplace=True)
                
                # Ngarobah ngaran kolom janten standar bot
                rename_dict = {}
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'close' in col_lower:
                        rename_dict[col] = 'Close'
                    elif 'open' in col_lower:
                        rename_dict[col] = 'Open'
                    elif 'high' in col_lower:
                        rename_dict[col] = 'High'
                    elif 'low' in col_lower:
                        rename_dict[col] = 'Low'
                
                df.rename(columns=rename_dict, inplace=True)
                
                if {'Close', 'Open', 'High', 'Low'}.issubset(df.columns):
                    df['Close'] = df['Close'].astype(float)
                    df['High'] = df['High'].astype(float)
                    df['Low'] = df['Low'].astype(float)
                    df['Open'] = df['Open'].astype(float)
                    
                    current_p = df['Close'].iloc[-1]
                    logging.info(f"✅ Sukses tarik data XAUUSD via Yahoo Finance | Harga Terkini: {current_p:.2f}")
                    return df
                    
        except Exception as e:
            logging.warning(f"⚠️ Gagal tarik data Yahoo Finance: {e}")
            
        logging.error("❌ Gagal total narik data pasar!")
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
        """Kalkulasi sinyal XAUUSD presisi tinggi & validasi target minimal 100 pips (10 poin)."""
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

        # Dynamic Risk Management XAUUSD (Target minimal setara 100 pips / 10.0 poin)
        sl_distance = max(atr * 1.2, 5.0)
        tp1_distance = max(sl_distance * 1.5, 10.0)  # Minimal 10.0 poin (100 pips)
        tp2_distance = tp1_distance * 2.0

        if signal == "BUY":
            sl = current_price - sl_distance
            tp1 = current_price + tp1_distance
            tp2 = current_price + tp2_distance
        else:
            sl = current_price + sl_distance
            tp1 = current_price - tp1_distance
            tp2 = current_price - tp2_distance

        # FILTER KETAT: Keyakinan >= 75% & Target Minimal 100 Pips tercapai
        min_target_pips = 10.0
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

def format_signal_card(res: dict, is_reversal: bool = False) -> str:
    """Format tampilan pesan Telegram."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    if is_reversal:
        if res['signal'] == "BUY":
            alert_title = "🚨🚨 **[ALERT: CLOSE SELL & REVERSE TO BUY!]** 🚨🚨"
            desc = "Tren pasar ngadadak males! Tutup posisi SELL ayeuna, siap-siap pindah BUY!"
        else:
            alert_title = "🚨🚨 **[ALERT: CLOSE BUY & REVERSE TO SELL!]** 🚨🚨"
            desc = "Tren pasar ngadadak turun! Tutup posisi BUY ayeuna, siap-siap pindah SELL!"
        
        return (
            f"⚠️ *[SUPER AI XAUUSD - COUNTER SIGNAL]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *Aksi Squel/Close*: {alert_title}\n"
            f"💡 *Katerangan*: `{desc}`\n"
            f"💵 *Harga Pasar (TF 5M)*: `{res['price']:.2f}`\n"
            f"🔥 *Keyakinan AI*: `{res['confidence']:.1f}%`\n"
            "-------------------------------------\n"
            f"🛑 *Stop Loss Baru*: `{res['sl']:.2f}`\n"
            f"🟢 *Target TP 1*: `{res['tp1']:.2f}`\n"
            "-------------------------------------\n"
            f"⏰ `{wib_time}`"
        )
    else:
        if res['signal'] == "BUY":
            signal_badge = "🟢🟢 **[STRONG BUY - LONG]** 🟢🟢"
            action_desc = "Target XAUUSD siap MEROKET naik (Target >100 Pips)! 🚀"
        else:
            signal_badge = "🔴🔴 **[STRONG SELL - SHORT]** 🔴🔴"
            action_desc = "Target XAUUSD siap TERJUN bebas (Target >100 Pips)! 📉"

        return (
            f"🤖 *[SUPER AI XAUUSD BOT - SIGNAL]*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 *Sinyal Eksekusi*: {signal_badge}\n"
            f"💡 *Analisis*: `{action_desc}`\n"
            f"💵 *Harga Pasar (TF 5M)*: `{res['price']:.2f}`\n"
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

def send_startup_notification(bot_instance):
    """Mengirim notifikasi startup lengkap dengan harga real-time terkini."""
    df = bot_instance.fetch_deriv_candles(count=5)
    _, current_price, atr, rsi, _ = bot_instance.extract_features_and_indicators(df)
    
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    msg = (
        f"🚀 *[SYSTEM STARTUP]*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ *Super AI XAUUSD Bot* parantos sukses diaktifkeun!\n"
        f"💵 *Harga Real-Time XAUUSD (TF 5M)*: `{current_price:.2f}`\n"
        f"🔍 *(Bandingkeun sareng MT5 ayeuna)*\n"
        f"📊 *RSI*: `{rsi:.1f}` | *ATR*: `{atr:.2f}`\n"
        "🛡️ Mode Anti-Spam Aktif: Notifikasi dikirim ngan sakali per sinyal.\n"
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
        f"💵 *Harga Terkini XAUUSD (TF 5M)*: `{current_price:.2f}`\n"
        f"🔍 *(Bandingkeun sareng MT5 ayeuna)*\n"
        f"📊 *RSI Saat Ini*: `{rsi:.1f}` | *ATR*: `{atr:.2f}`\n"
        "-------------------------------------\n"
        "☕ Siap-siap ngantosan sinyal high-conviction dinten ieu!\n"
        f"⏰ `{wib_time}`"
    )
    send_telegram_message(msg)

if __name__ == "__main__":
    bot = SuperAIXAUUSDBot(model_path='model_xauusd.pkl')
    
    # 1. Kirim notif startup
    send_startup_notification(bot)
    
    last_daily_report_date = None

    # Loop utama monitoring
    while True:
        now_wib = datetime.now(bot.wib_tz)
        
        # 2. Laporan rutin jam 06.00 WIB
        if now_wib.hour == 6 and now_wib.minute == 0:
            if last_daily_report_date != now_wib.date():
                send_daily_report(bot)
                last_daily_report_date = now_wib.date()

        # Evaluasi pasar berkala
        res = bot.evaluate_market()
        
        if res["valid"]:
            current_signal = res["signal"]
            
            # 3. Logika Anti-Spam Sinyal
            if bot.last_sent_signal is None:
                msg = format_signal_card(res, is_reversal=False)
                send_telegram_message(msg)
                bot.last_sent_signal = current_signal
                logging.info(f"✅ Sinyal awal {current_signal} terkirim!")
                
            elif bot.last_sent_signal != current_signal:
                msg = format_signal_card(res, is_reversal=True)
                send_telegram_message(msg)
                bot.last_sent_signal = current_signal
                logging.info(f"🚨 Sinyal berbalik arah! Alert close & reverse dikirim: {current_signal}")
                
            else:
                logging.info(f"⏳ Sinyal masih {current_signal} (Aman, teu ngirim notif ulang).")
        else:
            logging.info("⏳ Market XAUUSD di-skip (Teu acan nyumponan sarat keyakinan >75% / TP <100 pips).")

        # Jeda 60 detik sateuacan mariksa deui pasar
        time_module.sleep(60)
