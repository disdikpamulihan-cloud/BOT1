import joblib
import pandas as pd
import logging
import os
import requests

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

    def predict_xauusd(self, df_xau: pd.DataFrame) -> str:
        """Proses dan prediksi untuk XAUUSD."""
        features_xau = ['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4']
        input_data = df_xau[features_xau].tail(1)
        
        actual_model = self._extract_model(self.model_xauusd)
        
        if isinstance(self.model_xauusd, dict) and 'scaler' in self.model_xauusd:
            input_data = self.model_xauusd['scaler'].transform(input_data.values)
        
        prediction = actual_model.predict(input_data)[0]
        return "BUY" if prediction == 1 else "SELL"

    def predict_vol80(self, df_vol: pd.DataFrame) -> str:
        """Proses dan prediksi untuk Volatility 80 Index."""
        features_vol = ['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4']
        input_data = df_vol[features_vol].tail(1)
        
        actual_model = self._extract_model(self.model_vol80)
        
        if isinstance(self.model_vol80, dict) and 'scaler' in self.model_vol80:
            input_data = self.model_vol80['scaler'].transform(input_data.values)
        
        prediction = actual_model.predict(input_data)[0]
        return "BUY" if prediction == 1 else "SELL"

    def get_signals(self, market_data: dict) -> dict:
        """Menghasilkan sinyal gabungan untuk kedua pasar."""
        signals = {}
        
        if 'XAUUSD' in market_data:
            signals['XAUUSD'] = self.predict_xauusd(market_data['XAUUSD'])
            
        if 'VOL80' in market_data:
            signals['VOL80'] = self.predict_vol80(market_data['VOL80'])
            
        return signals

def send_telegram_message(message: str):
    """Mengirim pesan notifikasi ke Telegram."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        logging.warning("TELEGRAM_BOT_TOKEN atau TELEGRAM_CHAT_ID belum diatur di Secrets!")
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
            logging.info("Notifikasi Telegram berhasil dikirim.")
        else:
            logging.error(f"Gagal mengirim notifikasi Telegram: {response.text}")
    except Exception as e:
        logging.error(f"Error saat menghubungi Telegram API: {e}")


# ==========================================
# CONTOH PENGGUNAAN (Execution Logic)
# ==========================================
if __name__ == "__main__":
    bot = DualMarketSignalBot(
        model_xau_path='model_xauusd.pkl',
        model_vol_path='model_vol80.pkl'
    )
    
    dummy_df_xau = pd.DataFrame({
        'Column_0': [-1.2], 'Column_1': [0.5], 
        'Column_2': [1.1], 'Column_3': [-0.4], 'Column_4': [1.86]
    })
    
    dummy_df_vol = pd.DataFrame({
        'Column_0': [-0.5], 'Column_1': [1.2], 
        'Column_2': [-1.8], 'Column_3': [0.9], 'Column_4': [0.1]
    })
    
    market_payload = {
        'XAUUSD': dummy_df_xau,
        'VOL80': dummy_df_vol
    }
    
    hasil_sinyal = bot.get_signals(market_payload)
    
    # Format pesan Telegram
    pesan = (
        "📊 **SINYAL TRADING HARI INI** 📊\n\n"
        f"🟡 **XAUUSD**: `{hasil_sinyal.get('XAUUSD', 'N/A')}`\n"
        f"📈 **VOL80**: `{hasil_sinyal.get('VOL80', 'N/A')}`"
    )
    
    # Kirim notifikasi ke Telegram
    send_telegram_message(pesan)
