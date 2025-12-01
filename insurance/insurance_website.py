import streamlit as st
import requests

st.title("Predição de Custos de Seguro")

# Campos de entrada
age = st.number_input("Idade", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Nº de filhos", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Fumante?", ["yes", "no"])
region = st.selectbox("Região", ["southeast", "southwest", "northeast", "northwest"])

# Quando o botão for clicado
if st.button("Prever custo"):
    # URL da API (ajuste se rodar local ou no ngrok)
    url = "http://127.0.0.1:8000/predict"
    
    # Fazer requisição GET/POST
    params = {
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }
    response = requests.post(url, params=params)

    if response.status_code == 200:
        resultado = response.json()
        st.success(f"Custo previsto: US$ {resultado['predicted_charges']}")
    else:
        st.error("Erro ao consultar API")
