# train.py
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import config
import preprocessing

def train():
    raw_data = preprocessing.get_data()
    X, y, scaler = preprocessing.prepare_sequences(raw_data)

    # Definir Arquitetura
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    
    print("Iniciando treinamento...")
    model.fit(X, y, batch_size=32, epochs=10) # 10 épocas para um bom equilíbrio inicial

    # Salvar Artefatos
    model.save(config.MODEL_PATH)
    joblib.dump(scaler, config.SCALER_PATH)
    print(f"✅ Modelo e Scaler salvos com sucesso!")

if __name__ == "__main__":
    train()