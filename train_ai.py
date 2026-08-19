import websocket
import json
import pandas as pd
import numpy as np
import joblib
import os
import ssl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_historical_data_for_training(count: int = 1000) -> pd.DataFrame:
    """Narik data sajarah loba (contona 1000 candle) pikeun bahan latihan AI."""
    symbols = ["frxXAUUSD", "XAUUSD", "gold"]
    app_id = "1089"
    ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
    
    for s in symbols:
        try:
            ws = websocket.create_connection(ws_url, timeout=15, sslopt={"cert_reqs": ssl.CERT_NONE})
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
                logging.info(f"✅ Berhasil tarik {len(df)} data history tina {s} pikeun latihan AI.")
                return df
        except Exception as e:
            logging.warning(f"⚠️ Gagal narik data tina {s}: {e}")
            
    return pd.DataFrame()

def train_new_ai_model():
    """Prosés ngalatih model AI anyar anu hampang tur akurat."""
    logging.info("🔄 Mimiti narik data pasar pikeun latihan AI...")
    df = fetch_historical_data_for_training(count=1000)
    
    if df.empty or len(df) < 200:
        logging.error("❌ Data teu cukup pikeun melatih model AI!")
        return False

    # 1. Ekstraksi Fitur (Fitur anu dibaca ku AI)
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    open_p = df['Open'].values

    # Hitung Indikator Pendukung (ATR & MA-200)
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else 1.0
    
    ma200 = pd.Series(close).rolling(window=200).mean().values

    # Siapkeun Dataset pikeun Machine Learning
    X = []
    y = []

    for i in range(200, len(close) - 1):
        # Fitur: [ATR, Ukuran Body Candle, Jarak Harga kana MA200]
        b_size = abs(close[i] - open_p[i])
        dist_ma = close[i] - ma200[i]
        
        features = [float(atr), float(b_size), float(dist_ma)]
        
        # Target/Labél: Naha candle payunna naék (1) atawa turun (0)?
        future_return = close[i+1] - close[i]
        label = 1 if future_return > 0 else 0
        
        X.append(features)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # 2. Paké Algoritma Modern & Hampang (Logistic Regression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    logging.info("🧠 Sedang melatih model AI...")
    model.fit(X_train, y_train)

    # Evaluasi akurasi sakedik
    score = model.score(X_test, y_test)
    logging.info(f"✨ Model AI suksés dilatih! Akurasi test: {score * 100:.2f}%")

    # 3. Simpen kana file .pkl anu anyar
    model_filename = "model_xauusd.pkl"
    joblib.dump(model, model_filename)
    logging.info(f"💾 Model anyar suksés disimpen kana {model_filename}!")
    return True

if __name__ == "__main__":
    train_new_ai_model()
