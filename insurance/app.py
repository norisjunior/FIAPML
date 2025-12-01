from fastapi import FastAPI
import pandas as pd
import joblib

# Carregar modelo treinado
modelo = joblib.load("insurance_pipeline.pkl")["model"]

# Inicializar API
app = FastAPI(title="API de Predição de Custos de Seguro")

# Endpoint raiz
@app.get("/")
def home():
    return {"mensagem": "API de Custos de Seguro no ar! Use /predict"}

# Endpoint de predição
@app.post("/predict")
def predict(age: int, sex: str, bmi: float, children: int, smoker: str, region: str):
    dados = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    pred = modelo.predict(dados)[0]
    return {"predicted_charges": round(float(pred), 2)}
