# streamlit_app_cyberpunk.py
import streamlit as st
import pickle, pandas as pd, numpy as np

# === CONFIGURAÇÃO CYBERPUNK ===
st.set_page_config(
    page_title="🔮 PREVISÃO DE CHURN",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS Cyberpunk
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #00ffff;
        font-family: 'Orbitron', monospace;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #ff0080, #00ffff, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(255, 0, 128, 0.5);
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: brightness(1); }
        to { filter: brightness(1.2); }
    }
    
    .cyber-container {
        background: rgba(0, 255, 255, 0.05);
        border: 2px solid #00ffff;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox label, .stNumberInput label {
        color: #00ffff !important;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
    }
    
    .stSelectbox > div > div, .stNumberInput > div > div {
        background-color: rgba(0, 0, 0, 0.7) !important;
        border: 1px solid #00ffff !important;
        border-radius: 8px !important;
        color: #ffffff !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff0080, #00ffff) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        box-shadow: 0 0 20px rgba(255, 0, 128, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 0 30px rgba(255, 0, 128, 0.6) !important;
    }
    
    .prediction-success {
        background: linear-gradient(45deg, #00ff00, #00cc00) !important;
        color: #000000 !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        text-align: center !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
        box-shadow: 0 0 25px rgba(0, 255, 0, 0.5) !important;
        animation: pulse-green 1.5s ease-in-out !important;
        margin: 1rem 0 !important;
    }
    
    .prediction-danger {
        background: linear-gradient(45deg, #ff0000, #cc0000) !important;
        color: #ffffff !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        text-align: center !important;
        font-size: 1.5rem !important;
        font-weight: bold !important;
        box-shadow: 0 0 25px rgba(255, 0, 0, 0.5) !important;
        animation: pulse-red 1.5s ease-in-out !important;
        margin: 1rem 0 !important;
    }
    
    @keyframes pulse-green {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes pulse-red {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .probability-box {
        background: rgba(0, 255, 255, 0.1);
        border: 1px solid #00ffff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
        font-size: 1.2rem;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    }
    
    .error-box {
        background: linear-gradient(45deg, #ff4444, #cc0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.4);
    }
    
    .cyber-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# === TÍTULO PRINCIPAL ===
st.markdown('<h1 class="main-title">🔮 PREVISÃO DE CHURN 🔮</h1>', unsafe_allow_html=True)

# === 1) Carrega artefatos ===
@st.cache_data
def load_artifacts():
    with open("churn_pipeline.pkl", 'rb') as f:
        return pickle.load(f)

try:
    art = load_artifacts()
    
    # === 2) INTERFACE CYBERPUNK ===
    st.markdown('<div class="cyber-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 style="color: white; margin-top: 0;">👤 <strong>DADOS PESSOAIS</strong></h3>', unsafe_allow_html=True)
        geo = st.selectbox("🌍 País", [""] + art["meta"]["cat_options"]["Geography"])
        #gender = st.selectbox("⚧ Sexo", [""] + art["meta"]["cat_options"]["Gender"])
        gender = st.selectbox("⚧ Sexo", ["", "Masculino", "Feminino"])
        age = st.number_input("🎂 Idade", min_value=0, value=0, step=1)
        tenure = st.number_input("⏱️ Relacionamento (anos)", min_value=0, value=0, step=1)
        salary = st.number_input("💰 Salário estimado", min_value=0.0, value=0.0, step=1000.0)
    
    with col2:
        st.markdown('<h3 style="color: white; margin-top: 0;">💳 <strong>DADOS BANCÁRIOS</strong></h3>', unsafe_allow_html=True)
        credit = st.number_input("📊 Credit Score", min_value=0, value=0, step=1)
        balance = st.number_input("💵 Saldo na conta", min_value=0.0, value=0.0, step=100.0)
        nprod = st.number_input("🛍️ Número de produtos", min_value=0, value=0, step=1)
        hascc = st.selectbox("💳 Tem cartão de crédito?", ["", "Não", "Sim"])
        active = st.selectbox("🟢 É um membro ativo?", ["", "Não", "Sim"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # === 3) BOTÃO DE PREDIÇÃO ===
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🚀 INICIAR MACHINE LEARNING 🚀")
    
    # === 4) PREDIÇÃO ===
    if predict_button:
        # Converte Sim/Não para 1/0
        hascc_val = {"Sim": 1, "Não": 0}.get(hascc, "")
        active_val = {"Sim": 1, "Não": 0}.get(active, "")
        
        # Converte Masculino/Feminino para Male/Female
        gender_val = {"Masculino": "Male", "Feminino": "Female"}.get(gender, "")
        
        # Validação
        if not all([geo, gender != "", credit, age, tenure, nprod, hascc != "", active != "", salary]):
            st.markdown('<div class="error-box">⚠️ ERRO: Todos os campos devem ser preenchidos para previsão usando ML!</div>', unsafe_allow_html=True)
        else:
            with st.spinner('🔄 Processando dados através da rede neural...'):
                # Monta entrada
                entrada = pd.DataFrame([{
                    "CreditScore": credit, "Geography": geo, "Gender": gender_val, "Age": age,
                    "Tenure": tenure, "Balance": balance, "NumOfProducts": nprod,
                    "HasCrCard": hascc_val, "IsActiveMember": active_val, "EstimatedSalary": salary
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

                # Resultado com cores
                if pred == 1:  # Vai sair - VERMELHO
                    st.markdown(f'''
                    <div class="prediction-danger">
                        🚨 ALERTA CRÍTICO 🚨<br>
                        <strong>CLIENTE IRÁ SAIR</strong><br>
                        Status: CHURN DETECTADO
                    </div>
                    ''', unsafe_allow_html=True)
                else:  # Vai ficar - VERDE
                    st.markdown(f'''
                    <div class="prediction-success">
                        ✅ STATUS SEGURO ✅<br>
                        <strong>CLIENTE IRÁ PERMANECER</strong><br>
                        Status: RETENÇÃO CONFIRMADA
                    </div>
                    ''', unsafe_allow_html=True)

                # Probabilidade
                st.markdown(f'''
                <div class="probability-box">
                    🎯 <strong>PROBABILIDADE DE CHURN:</strong> {proba:.1%}<br>
                    📈 Confiança da IA: {max(proba, 1-proba):.1%}
                </div>
                ''', unsafe_allow_html=True)

except FileNotFoundError:
    st.markdown('''
    <div class="error-box">
        🚫 ERRO CRÍTICO: Arquivo 'churn_pipeline.pkl' não encontrado!<br>
        Verifique se o modelo foi treinado e salvo corretamente.
    </div>
    ''', unsafe_allow_html=True)

# === FOOTER CYBERPUNK ===
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #00ffff; opacity: 0.7; font-size: 0.9rem;">
    🤖 Powered by ML • 2077 Technology • 🔮
</div>
""", unsafe_allow_html=True)