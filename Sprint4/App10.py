import streamlit as st
import joblib
import pandas as pd
import json
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Configuração da página
st.set_page_config(
    page_title="Preditor de Risco de Acidentes",
    page_icon="🚗",
    layout="wide"
)

# Função para carregar o modelo
@st.cache_resource
def carregar_modelo():
    """Carrega o modelo salvo em .pkl"""
    try:
        modelo_artefatos = joblib.load("risco_acidentes_pipeline.pkl")
        return modelo_artefatos
    except FileNotFoundError:
        st.error("❌ Arquivo do modelo não encontrado! Certifique-se de que 'risco_acidentes_pipeline.pkl' está na pasta.")
        return None

# Função para buscar coordenadas de um endereço
@st.cache_data
def buscar_coordenadas(endereco):
    """Busca latitude e longitude de um endereço usando geocoding"""
    try:
        geolocator = Nominatim(user_agent="risk_predictor_app")
        location = geolocator.geocode(endereco, timeout=10)
        
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except GeocoderTimedOut:
        return None, None, None

# Função para mapear BR para Delegacia
def obter_delegacia_por_br(br_selecionada, metadados):
    """Mapeia uma BR para uma delegacia (simplificado para demo)"""
    # Mapeamento simplificado BR -> Estado -> Delegacia
    mapeamento_br_estado = {
        "BR-101": "SC", "BR-116": "PR", "BR-277": "PR", "BR-290": "RS",
        "BR-153": "GO", "BR-230": "PB", "BR-376": "PR", "BR-381": "MG",
        "BR-262": "MG", "BR-280": "SC", "BR-70": "DF", "BR-135": "BA"
    }
    
    estado = mapeamento_br_estado.get(br_selecionada, "MG")
    
    # Busca uma delegacia do estado correspondente
    delegacias_estado = [d for d in metadados["delegacia"] if estado in d]
    
    if delegacias_estado:
        return delegacias_estado[0]
    else:
        return metadados["delegacia"][0]  # Fallback

# Função para obter valor dos postos PRF baseado na BR
def obter_postos_prf_por_br(br_selecionada):
    """Retorna o nível de postos PRF baseado na BR (simplificado)"""
    # BRs principais têm mais postos
    brs_alta_densidade = ["BR-101", "BR-116", "BR-381", "BR-277", "BR-153"]
    brs_media_densidade = ["BR-230", "BR-290", "BR-262", "BR-376", "BR-280"]
    
    if br_selecionada in brs_alta_densidade:
        return "Alta"
    elif br_selecionada in brs_media_densidade:
        return "Média"
    else:
        return "Baixa"

# Interface principal
def main():
    # Título da aplicação
    st.title("🚗 Preditor de Risco de Acidentes em Rodovias")
    st.markdown("---")
    
    # Descrição
    st.markdown("""
    ### 📋 Como funciona:
    1. **Insira um endereço** → O sistema busca automaticamente latitude e longitude
    2. **Selecione as características da viagem** → BR, dia da semana, condições, etc.
    3. **Obtenha a predição** → O modelo de Machine Learning calcula o risco
    """)
    
    # Carregar modelo
    modelo_artefatos = carregar_modelo()
    if modelo_artefatos is None:
        st.stop()
    
    pipeline = modelo_artefatos["pipeline"]
    label_encoder = modelo_artefatos["label_encoder"]
    metadados = modelo_artefatos["meta"]
    
    st.markdown("---")
    
    # Layout em colunas
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📍 Localização")
        
        # Campo de endereço
        endereco = st.text_input(
            "Digite o endereço:",
            placeholder="Ex: Avenida Paulista, 1000, São Paulo, SP",
            help="Digite um endereço completo para buscar as coordenadas automaticamente"
        )
        
        # Buscar coordenadas quando endereço é inserido
        latitude, longitude, endereco_completo = None, None, None
        
        if endereco:
            with st.spinner("🔍 Buscando coordenadas..."):
                latitude, longitude, endereco_completo = buscar_coordenadas(endereco)
            
            if latitude and longitude:
                st.success(f"✅ Endereço encontrado!")
                st.info(f"📍 **Coordenadas:** {latitude:.4f}, {longitude:.4f}")
                if endereco_completo:
                    st.caption(f"📋 {endereco_completo}")
            else:
                st.error("❌ Não foi possível encontrar as coordenadas. Tente um endereço mais específico.")
    
    with col2:
        st.subheader("🛣️ Características da Rodovia")
        
        # Seleção da BR
        br_selecionada = st.selectbox(
            "Rodovia (BR):",
            options=metadados["cat_options"]["br"],
            help="Selecione a rodovia federal"
        )
        
        # Quando BR é selecionada, automaticamente define delegacia e postos PRF
        if br_selecionada:
            delegacia_automatica = obter_delegacia_por_br(br_selecionada, metadados)
            postos_prf_automatico = obter_postos_prf_por_br(br_selecionada)
            
            st.info(f"🏢 **Delegacia:** {delegacia_automatica}")
            st.info(f"🚓 **Postos PRF:** {postos_prf_automatico}")
        
        # KM da rodovia
        km = st.number_input(
            "Quilômetro (KM):",
            min_value=0.0,
            max_value=1000.0,
            value=50.0,  # Valor padrão simples
            step=0.1,
            help="Quilometragem da rodovia"
        )
    
    st.markdown("---")
    
    # Características adicionais
    st.subheader("🕐 Características Temporais e Ambientais")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        dia_semana = st.selectbox(
            "Dia da Semana:",
            options=metadados["cat_options"]["dia_semana"]
        )
        
        fase_dia = st.selectbox(
            "Fase do Dia:",
            options=metadados["cat_options"]["fase_dia"]
        )
    
    with col4:
        condicao_metereologica = st.selectbox(
            "Condição Meteorológica:",
            options=metadados["cat_options"]["condicao_metereologica"]
        )
        
        sentido_via = st.selectbox(
            "Sentido da Via:",
            options=metadados["cat_options"]["sentido_via"]
        )
    
    with col5:
        tipo_pista = st.selectbox(
            "Tipo de Pista:",
            options=metadados["cat_options"]["tipo_pista"]
        )
        
        tracado_via = st.selectbox(
            "Traçado da Via:",
            options=metadados["cat_options"]["tracado_via"]
        )
    
    st.markdown("---")
    
    # Botão de predição
    if st.button("🔮 Prever Risco de Acidente", type="primary", use_container_width=True):
        
        # Validar se temos coordenadas
        if latitude is None or longitude is None:
            st.error("❌ Por favor, insira um endereço válido para obter as coordenadas!")
            return
        
        # Preparar dados para predição
        dados_entrada = {
            'dia_semana': dia_semana,
            'br': br_selecionada,
            'km': km,
            'fase_dia': fase_dia,
            'sentido_via': sentido_via,
            'condicao_metereologica': condicao_metereologica,
            'tipo_pista': tipo_pista,
            'tracado_via': tracado_via,
            'latitude': latitude,
            'longitude': longitude,
            'delegacia': delegacia_automatica,
            'postos_policiais_PRF': postos_prf_automatico
        }
        
        # Converter para DataFrame
        df_predicao = pd.DataFrame([dados_entrada])
        
        # Fazer predição
        with st.spinner("🤖 Processando predição..."):
            predicao_encoded = pipeline.predict(df_predicao)
            predicao_original = label_encoder.inverse_transform(predicao_encoded)
            
            # Obter probabilidades
            probabilidades = pipeline.predict_proba(df_predicao)[0]
            classes = label_encoder.classes_
        
        # Exibir resultado
        st.markdown("---")
        st.subheader("📊 Resultado da Predição")
        
        # Resultado principal
        risco_previsto = predicao_original[0]
        
        # Definir cor baseada no risco
        if risco_previsto == "Risco Alto":
            cor = "🔴"
            st.error(f"{cor} **RISCO ALTO** - Atenção redobrada necessária!")
        elif risco_previsto == "Risco Médio":
            cor = "🟡"
            st.warning(f"{cor} **RISCO MÉDIO** - Cuidado moderado recomendado")
        else:
            cor = "🟢"
            st.success(f"{cor} **RISCO BAIXO** - Condições relativamente seguras")
        
        # Mostrar probabilidades
        st.subheader("📈 Probabilidades por Classe")
        
        for i, classe in enumerate(classes):
            prob = probabilidades[i] * 100
            st.metric(
                label=classe,
                value=f"{prob:.1f}%",
                delta=None
            )
        
        # Resumo dos dados utilizados
        with st.expander("📋 Dados utilizados na predição"):
            st.json(dados_entrada)

if __name__ == "__main__":
    main()