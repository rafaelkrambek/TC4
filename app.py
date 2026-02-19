# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib
import config
import os

app = FastAPI(title="Stock Predictor 5-Day API")

# Carregar modelos
if os.path.exists(config.MODEL_PATH):
    model = tf.keras.models.load_model(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    print(f"✅ Modelo de {config.DAYS_BEHIND} dias carregado!")
else:
    model = None

class PriceData(BaseModel):
    prices: list[float]

@app.post("/predict")
async def predict(data: PriceData):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo não treinado.")

    # Se o usuário mandar mais que 5, pegamos os 5 mais recentes
    # Se mandar menos, retornamos erro
    if len(data.prices) < config.DAYS_BEHIND:
        raise HTTPException(
            status_code=400, 
            detail=f"Dados insuficientes. Envie exatamente {config.DAYS_BEHIND} preços."
        )
    
    # Preparação dos dados
    input_list = data.prices[-config.DAYS_BEHIND:] # Garante que temos apenas os últimos 5
    input_array = np.array(input_list).reshape(-1, 1)
    
    # Normalização e Reshape (1, 5, 1)
    scaled_input = scaler.transform(input_array)
    final_input = scaled_input.reshape(1, config.DAYS_BEHIND, 1)

    # Executa a previsão
    prediction_scaled = model.predict(final_input)
    prediction_real = scaler.inverse_transform(prediction_scaled)

    return {
        "ticker": config.TICKER,
        "input_used": input_list,
        "prediction_next_day": float(prediction_real[0][0])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)