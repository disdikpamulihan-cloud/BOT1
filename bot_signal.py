import joblib
import pandas as pd
import logging
import os
import requests
from datetime import datetime
import pytz

# Set up logging untuk memantau eksekusi bot
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DualMarketSignalBot:
    def __init__(self, model_xau_path: str, model_vol_path: str):
        """Memuat kedua model ML saat inisialisasi."""
        try:
            self.model_xauusd = joblib.load(model_xau_path)
            self.model_vol80 = joblib.load(model_vol_path)
            logging.info("Berhasil memuat model XAUUSD dan Volatility 80.")
        except Exception as e:
            logging.error(f"Gagal memuat model: {e}")
            raise e

    def _extract_model(self, model_obj):
        """Mengambil estimator dari dictionary jika file .pkl berupa dictionary."""
        if isinstance(model_obj, dict):
            for key in ['model', 'estimator', 'lgbm', 'classifier', 'xgboost']:
                if key in model_obj:
                    return model_obj[key]
            return list(model_obj.values())[0]
        return model_obj

    def predict_market(self, df: pd.DataFrame, model_wrapper, symbol: str) -> dict:
        """Melakukan kalkulasi prediksi, confidence score, dan manajemen risiko (SL/TP)."""
        features = ['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4']
        input_data = df[features].tail(1)
        
        actual_model = self._extract_model(model_wrapper)
        
        if isinstance(model_wrapper, dict) and 'scaler' in model_wrapper:
            input_data = model_wrapper['scaler'].transform(input_data.values)
        
        # Prediksi arah sinyal & probabilitas (confidence score)
        prediction = actual_model.predict(input_data)[0]
        signal = "BUY" if prediction == 1 else "SELL"
        
        confidence = 50.0
        if hasattr(actual_model, "predict_proba"):
            probs = actual_model.predict_proba(input_data)[0]
            confidence = float(max(probs) * 100)
            
        # Mengambil harga saat ini dari DataFrame input
        current_price = float(df['close'].iloc[-1]) if 'close' in df.columns else 2400.00
        
        # Kalkulasi SL & TP otomatis berdasarkan jarak harga (contoh: 6 pip SL, 6 pip TP1, 12 pip TP2)
        pip_offset = 6.03 if symbol == 'XAUUSD' else 15.0
        
        if signal == "BUY":
            sl = current_price - pip_offset
            tp1 = current_price + pip_offset
            tp2 = current_price + (pip_offset * 2)
        else: # SELL
            sl = current_price + pip_offset
            tp1 = current_price - pip_offset
            tp2 = current_price - (pip_offset * 2)

        # ADX nilai simulasi/indikator
        adx = float(df['adx'].iloc[-1]) if 'adx' in df.columns else 60.3

        return {
            "signal": signal,
            "price": current_price,
            "confidence": confidence,
            "adx": adx,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2
        }

def send_telegram_message(message: str):
    """Mengirim pesan notifikasi ke Telegram."""
    bot_token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logging.warning("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID tidak ditemukan di environment variables!")
        return

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info("Notifikasi Telegram berhasil dikirim!")
        else:
            logging.error(f"Gagal mengirim Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Error Telegram API: {e}")

def format_signal_card(symbol: str, res: dict) -> str:
    """Memformat output pesan agar persis seperti AI MATRIX SIGNAL."""
    wib_tz = pytz.timezone('Asia/Jakarta')
    wib_time = datetime.now(wib_tz).strftime('%Y-%m-%d %H:%M:%S WIB')
    
    card = (
        f"🤖 *AI MATRIX SIGNAL ({symbol})*\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🎯 *Sinyal Eksekusi*: `{res['signal']}`\n"
        f"💵 *Harga Saat Ini*: `{res['price']:.2f}`\n"
        f"🔥 *Keyakinan AI*: `{res['confidence']:.1f}%`\n"
        f"📊 *ADX Trend Strength*: `{res['adx']:.1f}`\n"
        "-------------------------------------\n"
        f"🔴 *Stop Loss (SL)*: `{res['sl']:.2f}`\n"
        f"🟢 *Target TP 1 (Scalp)*: `{res['tp1']:.2f}`\n"
        f"🟢 *Target TP 2 (Runner)*: `{res['tp2']:.2f}`\n"
        "-------------------------------------\n"
        f"⏰ `{wib_time}`"
    )
    return card


# ==========================================
# EXECUTION LOGIC
# ==========================================
if __name__ == "__main__":
    bot = DualMarketSignalBot(
        model_xau_path='model_xauusd.pkl',
        model_vol_path='model_vol80.pkl'
    )
    
    # Simulation Data Payload (Memuat kolom fitur ML + kolom harga 'close' & 'adx')
    df_xau = pd.DataFrame({
        'Column_0': [-1.2], 'Column_1': [0.5], 
        'Column_2': [1.1], 'Column_3': [-0.4], 'Column_4': [1.86],
        'close': [4376.05], 'adx': [60.3]
    })
    
    df_vol = pd.DataFrame({
        'Column_0': [-0.5], 'Column_1': [1.2], 
        'Column_2': [-1.8], 'Column_3': [0.9], 'Column_4': [0.1],
        'close': [8025.50], 'adx': [45.2]
    })
    
    # 1. Kalkulasi Hasil Sinyal Lengkap
    res_xau = bot.predict_market(df_xau, bot.model_xauusd, 'XAUUSD')
    res_vol = bot.predict_market(df_vol, bot.model_vol80, 'VOL80')
    
    # 2. Format Kartu Sinyal Telegram
    card_xau = format_signal_card('XAUUSD', res_xau)
    card_vol = format_signal_card('VOLATILITY 80', res_vol)
    
    # 3. Kirim Ke Telegram
    send_telegram_message(card_xau)
    send_telegram_message(card_vol)
