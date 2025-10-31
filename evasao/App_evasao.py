import streamlit as st
import pandas as pd
import joblib

# ================================================
# CONFIGURAÇÃO INICIAL
# ================================================
st.set_page_config(page_title="Preditor de Evasão Escolar", page_icon="🎓", layout="centered")
st.title("App Preditor de Evasão Escolar")
st.write("Preencha as informações do estudante e veja a previsão com base no modelo treinado")

# ================================================
# CARREGA O MODELO COMPLETO
# ================================================
obj = joblib.load("evasao_pipeline_completo.pkl")
modelo = obj["model"]
le = obj["label_encoder"]
scaler = obj["scaler_pca"]["scaler"]
pca = obj["scaler_pca"]["pca"]

# ================================================
# ENTRADAS DO USUÁRIO
# ================================================
# --- Dicionários de mapeamento para as categorias ---
mapa_marital = {
    "Solteiro(a)": 1, "Casado(a)": 2, "Viúvo(a)": 3,
    "Divorciado(a)": 4, "União estável": 5, "Separado(a)": 6
}

mapa_application = {
    "Online": 1, "Preferência Nacional": 17, "Transferência": 18,
    "Outra Instituição": 39, "Estudante Internacional": 42
}

mapa_prev_qual = {
    "Ensino Médio": 1, "Curso Técnico": 2, "Licenciatura": 3,
    "Bacharelado": 4, "Outro": 5
}

mapa_gender = {"Feminino": 0, "Masculino": 1}
mapa_binario = {"Não": 0, "Sim": 1}

# --- Entradas amigáveis ---
st.subheader("Informações do estudante")

marital_status = mapa_marital[st.selectbox("Estado civil", list(mapa_marital.keys()))]
application_mode = mapa_application[st.selectbox("Modo de inscrição", list(mapa_application.keys()))]
application_order = st.slider("Ordem de preferência de curso", 1, 10, 1)
course = st.number_input("Código do curso", 1, 9999, 9773)
daytime_attendance = mapa_binario[st.selectbox("Frequência diurna?", list(mapa_binario.keys()))]
previous_qualification = mapa_prev_qual[st.selectbox("Formação anterior", list(mapa_prev_qual.keys()))]
previous_qualification_grade = st.number_input("Nota na formação anterior", 0.0, 200.0, 120.0)
mother_qualification = st.slider("Escolaridade da mãe (1 a 50)", 1, 50, 37)
father_occupation = st.slider("Ocupação do pai (1 a 50)", 1, 50, 9)
admission_grade = st.number_input("Nota de admissão", 0.0, 200.0, 120.0)
displaced = mapa_binario[st.selectbox("Estudante deslocado?", list(mapa_binario.keys()))]
special_needs = mapa_binario[st.selectbox("Necessidades especiais?", list(mapa_binario.keys()))]
debtor = mapa_binario[st.selectbox("Está inadimplente?", list(mapa_binario.keys()))]
tuition_fees_up_to_date = mapa_binario[st.selectbox("Mensalidades em dia?", list(mapa_binario.keys()))]
gender = mapa_gender[st.selectbox("Gênero", list(mapa_gender.keys()))]
scholarship_holder = mapa_binario[st.selectbox("Possui bolsa de estudos?", list(mapa_binario.keys()))]
age = st.slider("Idade ao ingressar", 15, 70, 20)
international = mapa_binario[st.selectbox("Estudante internacional?", list(mapa_binario.keys()))]

campos_basicos = {
    "Marital status": marital_status,
    "Application mode": application_mode,
    "Application order": application_order,
    "Course": course,
    "Daytime attendance": daytime_attendance,
    "Previous qualification": previous_qualification,
    "Previous qualification (grade)": previous_qualification_grade,
    "Mother's qualification": mother_qualification,
    "Father's occupation": father_occupation,
    "Admission grade": admission_grade,
    "Displaced": displaced,
    "Educational special needs": special_needs,
    "Debtor": debtor,
    "Tuition fees up to date": tuition_fees_up_to_date,
    "Gender": gender,
    "Scholarship holder": scholarship_holder,
    "Age at enrollment": age,
    "International": international,
}


st.subheader("Notas e desempenho acadêmico")

col1, col2 = st.columns(2)
with col1:
    curr1 = [
        st.number_input("1st sem - credited", 0.0, 10.0, 0.0),
        st.number_input("1st sem - enrolled", 0.0, 10.0, 6.0),
        st.number_input("1st sem - evaluations", 0.0, 10.0, 6.0),
        st.number_input("1st sem - without evaluations", 0.0, 10.0, 0.0),
        st.number_input("1st sem - approved", 0.0, 10.0, 6.0),
        st.number_input("1st sem - grade", 0.0, 20.0, 12.0),
    ]
with col2:
    curr2 = [
        st.number_input("2nd sem - credited", 0.0, 10.0, 0.0),
        st.number_input("2nd sem - enrolled", 0.0, 10.0, 6.0),
        st.number_input("2nd sem - evaluations", 0.0, 10.0, 6.0),
        st.number_input("2nd sem - without evaluations", 0.0, 10.0, 0.0),
        st.number_input("2nd sem - approved", 0.0, 10.0, 6.0),
        st.number_input("2nd sem - grade", 0.0, 20.0, 12.0),
    ]

# ================================================
# CÁLCULO DO PCA (usando scaler e PCA treinados)
# ================================================
colunas_pca = [
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (without evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)'
]

df_pca = pd.DataFrame([curr1 + curr2], columns=colunas_pca)
dados_scaled = scaler.transform(df_pca)
pca_result = pca.transform(dados_scaled)
PCA1_curricular, PCA2_curricular = pca_result[0]

# ================================================
# DATAFRAME FINAL (igual ao do treino)
# ================================================
# **campos_basicos: "desempacota" todo o dicionário e traz as variáveis que serão
# unidas com PCA1_curricular e PCA2_curricular
entrada = pd.DataFrame([{**campos_basicos,
                         "PCA1_curricular": PCA1_curricular,
                         "PCA2_curricular": PCA2_curricular}])

# ================================================
# PREDIÇÃO + PROBABILIDADE
# ================================================
if st.button("Prever Evasão"):
    pred_enc = modelo.predict(entrada)
    prob = modelo.predict_proba(entrada)[0][1]  # probabilidade de evasão
    pred = le.inverse_transform(pred_enc)[0]
    prob_pct = prob * 100

    st.markdown("---")
    st.write(f"**Probabilidade de evasão:** `{prob_pct:.2f}%`")

    if pred:
        st.error("ALTA probabilidade de **EVASÃO ESCOLAR**")
    else:
        st.success("BAIXA probabilidade de evasão (aluno tende a permanecer)")
