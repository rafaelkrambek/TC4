# preprocessing.py
import yfinance as yf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
import config

def get_data():
    data = yf.download(config.TICKER, start=config.START_DATE)
    return data[['Close']].values

def prepare_sequences(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(config.DAYS_BEHIND, len(scaled_data)):
        X.append(scaled_data[i-config.DAYS_BEHIND:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, np.array(y), scaler