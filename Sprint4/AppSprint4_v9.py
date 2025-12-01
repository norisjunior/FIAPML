# app.py
# Requisitos: streamlit, pandas, joblib, geopy, requests
# pip install streamlit pandas joblib geopy requests

import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path

# ========= TÍTULO =========
st.title("Risco de Rota • Sprint 4 (Streamlit)")

# ========= CARREGAR MODELO =========
pkl_path = st.text_input("Caminho do arquivo .pkl do modelo", "risco_acidentes_pipeline.pkl")

@st.cache_resource
def load_artifacts(path):
    return joblib.load(path)

art = None
if Path(pkl_path).exists():
    try:
        art = load_artifacts(pkl_path)
        st.success("Modelo carregado.")
    except Exception as e:
        st.error(f"Falha ao carregar o .pkl: {e}")
else:
    st.info("Informe o caminho do .pkl (arquivo unificado salvo na Sprint 3).")

if art:
    pipe = art["pipeline"]
    le   = art["label_encoder"]
    meta = art["meta"]

    # ========= FUNÇÕES AUXILIARES =========
    def mapear_br_por_regiao(latitude, longitude, opcoes_br_disponiveis):
        """Mapeia coordenadas para BR usando apenas opções do modelo"""
        # Mapeamento por regiões do Brasil
        if latitude > -16:  # Norte
            candidata = "BR-364"
        elif latitude > -23.5:  # Centro-Oeste/Nordeste
            if longitude < -47:
                candidata = "BR-153"
            else:
                candidata = "BR-116"
        else:  # Sudeste/Sul
            if longitude < -48:
                candidata = "BR-101"
            elif longitude < -45:
                candidata = "BR-381"
            else:
                candidata = "BR-116"
        
        # Verifica se existe no modelo
        if candidata in opcoes_br_disponiveis:
            return candidata
        
        # Fallback para primeira BR disponível
        brs_comuns = ["BR-116", "BR-101", "BR-381", "BR-153"]
        for br in brs_comuns:
            if br in opcoes_br_disponiveis:
                return br
                
        return opcoes_br_disponiveis[0] if opcoes_br_disponiveis else "BR-116"

    def obter_valores_categoricos_padroes(opcoes_categoricas):
        """Obtém valores padrão das categorias do modelo"""
        delegacia = opcoes_categoricas.get("delegacia", ["DEL001-UF"])[0]
        postos = opcoes_categoricas.get("postos_policiais_PRF", ["Baixa"])[0]
        return delegacia, postos

    # ========= GEOCODIFICAÇÃO =========
    @st.cache_resource
    def criar_geocodificador():
        try:
            from geopy.geocoders import Nominatim
            return Nominatim(user_agent="fiap_sprint4_app")
        except Exception:
            return None

    geocoder = criar_geocodificador()

    def geocodificar_endereco(endereco):
        if (geocoder is None) or (not endereco.strip()):
            return None
        try:
            localizacao = geocoder.geocode(endereco, timeout=10)
            if localizacao:
                return float(localizacao.latitude), float(localizacao.longitude)
        except Exception:
            pass
        return None

    # ========= MAPA DE RISCO =========
    risco_para_numero = {
        "Risco Baixo": 0, "Baixo": 0,
        "Risco Médio": 1, "Médio": 1, "Medio": 1,
        "Risco Alto":  2, "Alto":  2
    }
    numero_para_risco = {0: "Risco Baixo", 1: "Risco Médio", 2: "Risco Alto"}

    # ========= SESSION STATE =========
    if "lat_o" not in st.session_state:
        st.session_state.lat_o = float(meta["num_defaults"].get("latitude", -23.561))
    if "lon_o" not in st.session_state:
        st.session_state.lon_o = float(meta["num_defaults"].get("longitude", -46.655))
    if "lat_d" not in st.session_state:
        st.session_state.lat_d = float(meta["num_defaults"].get("latitude", -23.551))
    if "lon_d" not in st.session_state:
        st.session_state.lon_d = float(meta["num_defaults"].get("longitude", -46.634))

    # ========= INTERFACE - ENDEREÇOS =========
    st.subheader("Entradas da Rota")

    with st.form("form_endercos"):
        colA, colB = st.columns(2)
        with colA:
            endereco_origem = st.text_input("Endereço de Origem", "Avenida Paulista, 1578, São Paulo")
        with colB:
            endereco_destino = st.text_input("Endereço de Destino", "Praça da Sé, São Paulo")

        botao_geocodificar = st.form_submit_button("Tentar geocodificar endereços")

    # Processar geocodificação
    if botao_geocodificar:
        with st.spinner("Geocodificando endereços..."):
            coords_origem = geocodificar_endereco(endereco_origem)
            coords_destino = geocodificar_endereco(endereco_destino)
            
            if coords_origem:
                st.session_state.lat_o, st.session_state.lon_o = coords_origem
                st.success(f"Origem geocodificada: {coords_origem}")
            
            if coords_destino:
                st.session_state.lat_d, st.session_state.lon_d = coords_destino
                st.success(f"Destino geocodificado: {coords_destino}")
                
            if coords_origem or coords_destino:
                st.rerun()

    # ========= INTERFACE - COORDENADAS =========
    st.caption("Se a geocodificação falhar, edite manualmente lat/long.")
    col1, col2, col3, col4 = st.columns(4)
    
    lat_origem = col1.number_input("Latitude Origem", value=st.session_state.lat_o)
    lon_origem = col2.number_input("Longitude Origem", value=st.session_state.lon_o)
    lat_destino = col3.number_input("Latitude Destino", value=st.session_state.lat_d)
    lon_destino = col4.number_input("Longitude Destino", value=st.session_state.lon_d)

    # Atualizar session state
    st.session_state.lat_o = lat_origem
    st.session_state.lon_o = lon_origem
    st.session_state.lat_d = lat_destino
    st.session_state.lon_d = lon_destino

    # ========= OBTER DADOS DO MODELO =========
    opcoes_categoricas = meta["cat_options"]
    valores_numericos_padroes = meta["num_defaults"]
    
    # BR automática baseada em coordenadas
    brs_disponiveis = opcoes_categoricas.get("br", ["BR-116", "BR-101"])
    br_origem = mapear_br_por_regiao(lat_origem, lon_origem, brs_disponiveis)
    br_destino = mapear_br_por_regiao(lat_destino, lon_destino, brs_disponiveis)

    # Delegacias e postos padrão
    delegacia_origem, postos_origem = obter_valores_categoricos_padroes(opcoes_categoricas)
    delegacia_destino, postos_destino = obter_valores_categoricos_padroes(opcoes_categoricas)

    # ========= EXIBIR PARÂMETROS DO MODELO =========
    st.subheader("Parâmetros do Modelo")
    
    colA, colB = st.columns(2)
    with colA:
        st.write(f"**BR (Origem) [auto]:** {br_origem}")
        st.write(f"Delegacia (Origem) [auto]: **{delegacia_origem}**")
        st.write(f"Postos PRF (Origem) [auto]: **{postos_origem}**")
    with colB:
        st.write(f"**BR (Destino) [auto]:** {br_destino}")
        st.write(f"Delegacia (Destino) [auto]: **{delegacia_destino}**")
        st.write(f"Postos PRF (Destino) [auto]: **{postos_destino}**")

    # ========= INTERFACE - VARIÁVEIS CATEGÓRICAS =========
    def criar_selectbox_categoria(nome_categoria, primeira_opcao=True):
        opcoes = opcoes_categoricas.get(nome_categoria, [])
        if not opcoes:
            return st.text_input(nome_categoria, "")
        
        indice = 0 if primeira_opcao else (opcoes.index("Pleno dia") if "Pleno dia" in opcoes else 0)
        return st.selectbox(nome_categoria, opcoes, index=min(indice, len(opcoes)-1))

    st.markdown("**Variáveis Categóricas**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        dia_semana = criar_selectbox_categoria("dia_semana")
        sentido_via = criar_selectbox_categoria("sentido_via")
    with col2:
        fase_dia = criar_selectbox_categoria("fase_dia")
        condicao_meteorologica = criar_selectbox_categoria("condicao_metereologica")
    with col3:
        tipo_pista = criar_selectbox_categoria("tipo_pista")
        tracado_via = criar_selectbox_categoria("tracado_via")

    st.markdown("**Variáveis Numéricas**")
    km_origem = st.number_input("km (Origem)", value=float(valores_numericos_padroes.get("km", 0.0)))
    km_destino = st.number_input("km (Destino)", value=float(valores_numericos_padroes.get("km", 0.0)))

    # ========= PREPARAR DADOS PARA PREDIÇÃO =========
    def criar_linha_dados(dia_sem, br, km, fase, sentido, condicao_met, tipo_p, tracado,
                         latitude, longitude, delegacia, postos_prf):
        dados = {
            "dia_semana": dia_sem,
            "br": br,
            "km": float(km),
            "fase_dia": fase,
            "sentido_via": sentido,
            "condicao_metereologica": condicao_met,
            "tipo_pista": tipo_p,
            "tracado_via": tracado,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "delegacia": delegacia,
            "classificacao_acidente": "",  # placeholder
            "postos_policiais_PRF": postos_prf
        }
        
        # Manter ordem das features do treino
        features_ordenadas = meta["features"]
        linha_ordenada = {coluna: dados.get(coluna, None) for coluna in features_ordenadas}
        return pd.DataFrame([linha_ordenada])

    # ========= PREDIÇÃO =========
    if st.button("Calcular Risco"):
        # Preparar dados de origem e destino
        dados_origem = criar_linha_dados(
            dia_semana, br_origem, km_origem, fase_dia, sentido_via, 
            condicao_meteorologica, tipo_pista, tracado_via,
            lat_origem, lon_origem, delegacia_origem, postos_origem
        )
        
        dados_destino = criar_linha_dados(
            dia_semana, br_destino, km_destino, fase_dia, sentido_via,
            condicao_meteorologica, tipo_pista, tracado_via, 
            lat_destino, lon_destino, delegacia_destino, postos_destino
        )

        try:
            # Fazer predições
            risco_origem = pipe.predict(dados_origem)[0]
            risco_destino = pipe.predict(dados_destino)[0]
            
            # Converter códigos para texto se necessário
            if hasattr(le, "inverse_transform"):
                try:
                    risco_origem = le.inverse_transform([risco_origem])[0]
                    risco_destino = le.inverse_transform([risco_destino])[0]
                except Exception:
                    pass

            # Exibir resultados
            st.success(f"Risco Origem: **{risco_origem}**")
            st.success(f"Risco Destino: **{risco_destino}**")

            # Regra A: Conservadora (maior risco)
            risco_max = max(risco_para_numero.get(str(risco_origem), 0), 
                           risco_para_numero.get(str(risco_destino), 0))
            regra_a = numero_para_risco[risco_max]
            st.info(f"Regra A (conservadora): **{regra_a}**")

            # Regra B: Com ponto médio
            lat_medio = (lat_origem + lat_destino) / 2.0
            lon_medio = (lon_origem + lon_destino) / 2.0
            br_medio = br_origem
            
            dados_medio = criar_linha_dados(
                dia_semana, br_medio, (km_origem + km_destino) / 2.0, fase_dia, sentido_via,
                condicao_meteorologica, tipo_pista, tracado_via,
                lat_medio, lon_medio, delegacia_origem, postos_origem
            )
            
            risco_medio = pipe.predict(dados_medio)[0]
            if hasattr(le, "inverse_transform"):
                try:
                    risco_medio = le.inverse_transform([risco_medio])[0]
                except Exception:
                    pass

            st.write(f"Ponto Médio (lat {lat_medio:.5f}, lon {lon_medio:.5f}) ⇒ **{risco_medio}**")

            # Votação por maioria
            votos = [risco_para_numero.get(str(risco_origem), 0),
                    risco_para_numero.get(str(risco_destino), 0),
                    risco_para_numero.get(str(risco_medio), 0)]
            
            voto_majoritario = max(set(votos), key=votos.count)
            if votos.count(voto_majoritario) == 1:  # empate triplo
                regra_b = regra_a
            else:
                regra_b = numero_para_risco[voto_majoritario]

            st.success(f"Regra B (maioria com ponto médio): **{regra_b}**")

            # Conclusão
            st.markdown(
                "- **Regra A** é mais conservadora (toma o pior entre origem e destino).  \n"
                "- **Regra B** considera também um ponto intermediário simples (média das coordenadas), "
                "o que pode reduzir falsos alarmes ou captar risco no meio do caminho"
            )

        except Exception as e:
            st.error(f"Erro ao prever: {e}")