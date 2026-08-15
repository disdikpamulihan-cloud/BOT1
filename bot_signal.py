import joblib
import pandas as pd
import logging

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

    def predict_xauusd(self, df_xau: pd.DataFrame) -> str:
        """Proses dan prediksi untuk XAUUSD."""
        # Pastikan kolom fitur sesuai dengan model XAUUSD
        features_xau = ['Column_0', 'Column_1', 'Column_2', 'Column_3', 'Column_4']
        input_data = df_xau[features_xau].tail(1)
        
        prediction = self.model_xauusd.predict(input_data)[0]
        
        # Contoh pemetaan output model (1 = BUY, 0 = SELL)
        return "BUY" if prediction == 1 else "SELL"

    def predict_vol80(self, df_vol: pd.DataFrame) -> str:
        """Proses dan prediksi untuk Volatility 80 Index."""
        # Sesuaikan nama kolom fitur dengan model Volatility 80 milikmu
        features_vol = ['vol_feature_0', 'vol_feature_1', 'vol_feature_2']
        input_data = df_vol[features_vol].tail(1)
        
        prediction = self.model_vol80.predict(input_data)[0]
        
        # Contoh pemetaan output model
        return "BUY" if prediction == 1 else "SELL"

    def get_signals(self, market_data: dict) -> dict:
        """Menghasilkan sinyal gabungan untuk kedua pasar."""
        signals = {}
        
        if 'XAUUSD' in market_data:
            signals['XAUUSD'] = self.predict_xauusd(market_data['XAUUSD'])
            
        if 'VOL80' in market_data:
            signals['VOL80'] = self.predict_vol80(market_data['VOL80'])
            
        return signals


# ==========================================
# CONTOH PENGGUNAAN (Execution Logic)
# ==========================================
if __name__ == "__main__":
    # 1. Inisialisasi Bot dengan path model masing-masing
    bot = DualMarketSignalBot(
        model_xau_path='model_xauusd.pkl',
        model_vol_path='model_vol80.pkl'
    )
    
    # 2. Simulasi/Pengambilan Data Real-time (replace dengan API MT5 / Deriv kamu)
    dummy_df_xau = pd.DataFrame({
        'Column_0': [-1.2], 'Column_1': [0.5], 
        'Column_2': [1.1], 'Column_3': [-0.4], 'Column_4': [1.86]
    })
    
    dummy_df_vol = pd.DataFrame({
        'vol_feature_0': [102.4], 'vol_feature_1': [0.03], 'vol_feature_2': [-0.8]
    })
    
    market_payload = {
        'XAUUSD': dummy_df_xau,
        'VOL80': dummy_df_vol
    }
    
    # 3. Eksekusi Prediksi
    hasil_sinyal = bot.get_signals(market_payload)
    
    print("\n--- HASIL SINYAL HARI INI ---")
    print(f"Sinyal XAUUSD : {hasil_sinyal.get('XAUUSD')}")
    print(f"Sinyal VOL80  : {hasil_sinyal.get('VOL80')}")
