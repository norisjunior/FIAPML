"""
Streamlit App - Previsão de Churn
=================================

Este aplicativo demonstra como carregar um **modelo serializado** (pipeline .pkl) e
usar para prever se um cliente vai churnar (fechar conta) ou não.

Pré-requisitos de execução
-------------------------
1. Instale dependências (no terminal ou dentro do Colab):
   pip install streamlit scikit-learn pandas joblib lightgbm xgboost
2. Faça upload do arquivo do modelo (ex.: `churn_pipeline.pkl`).
3. Rode o app:
   streamlit run streamlit_churn_app.py

Observação
----------
O modelo foi salvo como **Pipeline sklearn** - ele já inclui as etapas de:
- LabelEncoder (`Gender`)
- OneHotEncoder (`Geography`)
- StandardScaler (atributos numéricos)
- Estimador final (KNN, DecisionTree, RandomForest, XGBoost, LightGBM ou outro)

Portanto, o Streamlit precisa apenas coletar os dados, organizá-los em um
`pandas.DataFrame` e chamar `predict()`.
"""

import streamlit as st
import pickle, pandas as pd, numpy as np

st.title("Previsão de Churn")

# === 1) Carrega artefatos ===
@st.cache_data
def load_artifacts():
    with open("churn_pipeline.pkl", 'rb') as f:
        return pickle.load(f)

art = load_artifacts()

# === 2) Entradas do usuário (campos vazios) ===
geo = st.selectbox("Geography", [""] + art["meta"]["cat_options"]["Geography"])
gender = st.selectbox("Gender", [""] + art["meta"]["cat_options"]["Gender"])

credit = st.number_input("CreditScore", min_value=0, value=0, step=1)
age    = st.number_input("Age", min_value=0, value=0, step=1)
tenure = st.number_input("Tenure", min_value=0, value=0, step=1)
balance= st.number_input("Balance", min_value=0.0, value=0.0)
nprod  = st.number_input("NumOfProducts", min_value=0, value=0, step=1)
hascc  = st.selectbox("HasCrCard", ["", 0, 1])
active = st.selectbox("IsActiveMember", ["", 0, 1])
salary = st.number_input("EstimatedSalary", min_value=0.0, value=0.0)

# === 3) Prever ===
if st.button("Prever"):
    # Validação simples
    if not all([geo, gender, credit, age, tenure, nprod, hascc != "", active != "", salary]):
        st.error("Preencha todos os campos!")
    else:
        # Monta entrada
        entrada = pd.DataFrame([{
            "CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age,
            "Tenure": tenure, "Balance": balance, "NumOfProducts": nprod,
            "HasCrCard": hascc, "IsActiveMember": active, "EstimatedSalary": salary
        }])[art["feature_order"]]

        # Preprocessa
        num_data = entrada[art["num_cols"]]
        cat_data = entrada[art["cat_cols"]]
        
        num_transformed = art["num_transformer"].transform(num_data)
        cat_transformed = art["cat_transformer"].transform(cat_data)
        
        X_processed = np.concatenate([num_transformed, cat_transformed], axis=1)
        
        # Predição
        pred = art["classifier"].predict(X_processed)[0]
        proba = art["classifier"].predict_proba(X_processed)[0][1]

        st.success(f"**Previsão:** {'Vai sair' if pred == 1 else 'Vai ficar'}")
        st.info(f"**Probabilidade de sair:** {proba:.1%}")