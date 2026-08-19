import websocket
import json
import pandas as pd
import numpy as np
import joblib
import os
import ssl
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_historical_data_for_training(count: int = 2500) -> pd.DataFrame:
    """Narik data history langkung seueur (2500 candle) supados AI diajar tina sagala kondisi market emas."""
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
                logging.info(f"✅ Berhasil tarik {len(df)} data history tina {s} pikeun latihan AI tingkat luhur.")
                return df
        except Exception as e:
            logging.warning(f"⚠️ Gagal narik data tina {s}: {e}")
            
    return pd.DataFrame()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_new_ai_model():
    logging.info("🔄 Mimiti narik data pasar pikeun latihan AI Sniper Jitu...")
    df = fetch_historical_data_for_training(count=2500)
    
    if df.empty or len(df) < 300:
        logging.error("❌ Data teu cukup pikeun melatih model AI!")
        return False

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_p = df['Open']

    # Indikator Komplit: MA-200, MA-50, ATR, RSI
    ma200 = close.rolling(window=200).mean()
    ma50 = close.rolling(window=50).mean()
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(window=14).mean()
    rsi = calculate_rsi(close, 14)

    df_feat = pd.DataFrame({
        'Close': close,
        'MA200': ma200,
        'MA50': ma50,
        'ATR': atr,
        'RSI': rsi,
        'BodySize': abs(close - open_p)
    }).dropna()

    X = []
    y = []

    # Filter Ekstrim Ketat: AI ngan diajar tina pergerakan anu bener-bener akurat (Anti-SL)
    for i in range(200, len(df_feat) - 3):
        row = df_feat.iloc[i]
        
        # Fitur multi-dimensi supaya AI terang struktur pasar
        features = [
            float(row['ATR']), 
            float(row['BodySize']), 
            float(row['Close'] - row['MA200']), 
            float(row['Close'] - row['MA50']),
            float(row['RSI'])
        ]
        
        # Tingali pergerakan 3 candle ka hareup (supaya tangtu moal gampang kena SL sakedapan)
        future_prices = df_feat['Close'].iloc[i+1 : i+4]
        max_future = future_prices.max()
        min_future = future_prices.min()
        current_c = row['Close']

        # Label ketat: 1 (BUY) lamun naék luhur tanpa nyolok ka handapheula, 0 (SELL) lamun sabalikna
        if (max_future - current_c) > (row['ATR'] * 1.5) and (current_c - min_future) < (row['ATR'] * 0.8):
            X.append(features)
            y.append(1)
        elif (current_c - min_future) > (row['ATR'] * 1.5) and (max_future - current_c) < (row['ATR'] * 0.8):
            X.append(features)
            y.append(0)

    if len(X) < 50:
        logging.error("❌ Data sampel teuing saeutik akibat filter teuing ketat! Nyobaan melonggarkeun sakedik...")
        # Fallback leuwih longgar sakedik lamun data kirang
        for i in range(200, len(df_feat) - 1):
            row = df_feat.iloc[i]
            features = [float(row['ATR']), float(row['BodySize']), float(row['Close'] - row['MA200']), float(row['Close'] - row['MA50']), float(row['RSI'])]
            future_ret = df_feat['Close'].iloc[i+1] - row['Close']
            if future_ret > 0.3:
                X.append(features); y.append(1)
            elif future_ret < -0.3:
                X.append(features); y.append(0)

    X = np.array(X)
    y = np.array(y)

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Kombinasi AI Tingkat Tinggi (Ensemble Model)
    model1 = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42)
    model2 = GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, random_state=42)

    ensemble_model = VotingClassifier(
        estimators=[('rf', model1), ('gb', model2)],
        voting='soft' # Soft voting ngamungkinkeun AI milih dumasar probabilitas persentase kepastian
    )

    logging.info("🧠 Sedang melatih Ensemble AI High-Precision...")
    ensemble_model.fit(X_train, y_train)

    score = ensemble_model.score(X_test, y_test)
    logging.info(f"✨ Model AI Sniper Jitu suksés dilatih! Akurasi test: {score * 100:.2f}%")

    model_filename = "model_xauusd.pkl"
    joblib.dump(ensemble_model, model_filename)
    logging.info(f"💾 Model AI Jitu suksés disimpen kana {model_filename}!")
    return True

if __name__ == "__main__":
    train_new_ai_model()
