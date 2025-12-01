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

    # ========= DADOS AUXILIARES PRF (BR -> postos/ delegacia) =========
    URL_PRF = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-dados_dos_postos_prfs.csv"

    @st.cache_data
    def carregar_prf(url):
        df = pd.read_csv(url, sep=";", encoding="latin1")
        # Normaliza BR
        if "rodovia" in df.columns:
            df["br"] = df["rodovia"].astype(str).str.upper().str.replace(" ", "")
            df["br"] = df["br"].apply(lambda x: "BR-" + x.split("-")[-1] if not x.startswith("BR-") else x)
        else:
            df["br"] = "BR-000"

        # Contagem por BR e tercis -> postos_policiais_PRF
        cont = df["br"].value_counts().rename_axis("br").reset_index(name="contagem")
        if len(cont) > 0:
            bins = [0, cont["contagem"].quantile(0.33), cont["contagem"].quantile(0.66), cont["contagem"].max()]
            labels = ["Baixa", "Média", "Alta"]
            cont["postos_policiais_PRF"] = pd.cut(cont["contagem"], bins=bins, labels=labels, include_lowest=True)
        else:
            cont["postos_policiais_PRF"] = "Baixa"

        postos_map = cont.set_index("br")["postos_policiais_PRF"].astype(str).to_dict()

        # Delegacia mais frequente por BR, se existir coluna correspondente
        del_col = "delegacia" if "delegacia" in df.columns else None
        if del_col:
            del_map = df.groupby("br")[del_col].agg(lambda x: x.value_counts().idxmax()).to_dict()
        else:
            del_map = {}

        return postos_map, del_map

    try:
        postos_map, del_map = carregar_prf(URL_PRF)
    except Exception as e:
        st.warning(f"Não foi possível carregar dados PRF online ({e}). Usando padrões.")
        postos_map, del_map = {}, {}

    def inferir_delegacia_e_postos(br):
        postos = str(postos_map.get(br, "Baixa"))
        # fallback simples caso não haja no CSV
        deleg = del_map.get(br, f"DEL{br.split('-')[-1]}-UF")
        return deleg, postos

    # ========= GEOCODIFICAÇÃO (Nominatim) =========
    @st.cache_resource
    def make_geocoder():
        try:
            from geopy.geocoders import Nominatim
            return Nominatim(user_agent="fiap_sprint4_app")
        except Exception:
            return None

    geocoder = make_geocoder()

    def geocode_addr(addr):
        if (geocoder is None) or (not addr.strip()):
            return None
        try:
            loc = geocoder.geocode(addr, timeout=10)
            if loc:
                return float(loc.latitude), float(loc.longitude)
        except Exception:
            pass
        return None

    # ========= MAPA DE RISCO PARA NÚMERO (para decisões) =========
    risco2num = {
        "Risco Baixo": 0, "Baixo": 0,
        "Risco Médio": 1, "Médio": 1, "Medio": 1,
        "Risco Alto":  2, "Alto":  2
    }
    num2risco = {0: "Risco Baixo", 1: "Risco Médio", 2: "Risco Alto"}

    # ========= FORM ENTRADAS =========
    st.subheader("Entradas da Rota")

    with st.form("form_inputs"):
        colA, colB = st.columns(2)
        with colA:
            end_origem = st.text_input("Endereço de Origem", "Avenida Paulista, 1578, São Paulo")
        with colB:
            end_dest   = st.text_input("Endereço de Destino", "Praça da Sé, São Paulo")

        st.caption("Se a geocodificação falhar, edite manualmente lat/long.")
        col1, col2, col3, col4 = st.columns(4)
        lat_o = col1.number_input("Latitude Origem", value=float(meta["num_defaults"].get("latitude", -23.561)))
        lon_o = col2.number_input("Longitude Origem", value=float(meta["num_defaults"].get("longitude", -46.655)))
        lat_d = col3.number_input("Latitude Destino", value=float(meta["num_defaults"].get("latitude", -23.551)))
        lon_d = col4.number_input("Longitude Destino", value=float(meta["num_defaults"].get("longitude", -46.634)))

        # Tenta geocodificar (sobrepõe os valores numéricos se der certo)
        if st.form_submit_button("Tentar geocodificar endereços (opcional)"):
            g1 = geocode_addr(end_origem)
            g2 = geocode_addr(end_dest)
            if g1:
                lat_o, lon_o = g1
            if g2:
                lat_d, lon_d = g2
            st.experimental_rerun()

    # Seletores com base nas opções do próprio modelo (`meta["cat_options"]`)
    st.subheader("Parâmetros do Modelo")
    cat = meta["cat_options"]
    num_defaults = meta["num_defaults"]

    # BR -> delegacia e postos automáticos
    br_opts = cat.get("br", ["BR-116", "BR-101"])
    colA, colB = st.columns(2)
    with colA:
        br_o = st.selectbox("BR (Origem)", br_opts, index=0)
    with colB:
        br_d = st.selectbox("BR (Destino)", br_opts, index=min(1, len(br_opts)-1))

    deleg_o, postos_o = inferir_delegacia_e_postos(br_o)
    deleg_d, postos_d = inferir_delegacia_e_postos(br_d)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Delegacia (Origem) [auto]: **{deleg_o}**")
        st.write(f"Postos PRF (Origem) [auto]: **{postos_o}**")
    with col2:
        st.write(f"Delegacia (Destino) [auto]: **{deleg_d}**")
        st.write(f"Postos PRF (Destino) [auto]: **{postos_d}**")

    # Demais campos categóricos básicos (com defaults simples)
    def pick(name, default_first=True):
        opts = cat.get(name, [])
        idx = 0 if default_first else (opts.index("Pleno dia") if "Pleno dia" in opts else 0)
        return st.selectbox(name, opts, index=min(idx, len(opts)-1)) if opts else st.text_input(name, "")

    st.markdown("**Variáveis Categóricas**")
    col1, col2, col3 = st.columns(3)
    with col1:
        dia_semana  = pick("dia_semana")
        sentido_via = pick("sentido_via")
    with col2:
        fase_dia    = pick("fase_dia")
        cond_met    = pick("condicao_metereologica")
    with col3:
        tipo_pista  = pick("tipo_pista")
        tracado_via = pick("tracado_via")

    st.markdown("**Variáveis Numéricas**")
    km_o = st.number_input("km (Origem)", value=float(num_defaults.get("km", 0.0)))
    km_d = st.number_input("km (Destino)", value=float(num_defaults.get("km", 0.0)))

    # ========= PREPARO DA LINHA DE ENTRADA =========
    features = meta["features"]  # ordem original de treino

    def montar_linha(dia_semana, br, km, fase_dia, sentido_via, cond_met, tipo_pista, tracado_via,
                     lat, lon, delegacia, postos):
        d = {
            "dia_semana": dia_semana,
            "br": br,
            "km": float(km),
            "fase_dia": fase_dia,
            "sentido_via": sentido_via,
            "condicao_metereologica": cond_met,
            "tipo_pista": tipo_pista,
            "tracado_via": tracado_via,
            "latitude": float(lat),
            "longitude": float(lon),
            "delegacia": delegacia,
            "classificacao_acidente": "",  # não usado em predição; placeholder se existir no features
            "postos_policiais_PRF": postos
        }
        # mantém exatamente a ordem do treino
        row = {col: d.get(col, None) for col in features}
        return pd.DataFrame([row])

    # ========= PREDIÇÃO =========
    if st.button("Calcular Risco"):
        Xo = montar_linha(dia_semana, br_o, km_o, fase_dia, sentido_via, cond_met, tipo_pista, tracado_via,
                          lat_o, lon_o, deleg_o, postos_o)
        Xd = montar_linha(dia_semana, br_d, km_d, fase_dia, sentido_via, cond_met, tipo_pista, tracado_via,
                          lat_d, lon_d, deleg_d, postos_d)

        try:
            yo = pipe.predict(Xo)[0]
            yd = pipe.predict(Xd)[0]
            # Se o pipeline devolve codificado, faz inverse_transform
            if hasattr(le, "inverse_transform"):
                try:
                    yo = le.inverse_transform([yo])[0]
                    yd = le.inverse_transform([yd])[0]
                except Exception:
                    pass

            st.success(f"Risco Origem: **{yo}**")
            st.success(f"Risco Destino: **{yd}**")

            # ===== Opção A: Risco da rota = maior risco entre origem/destino =====
            rA = num2risco[max(risco2num.get(str(yo), 0), risco2num.get(str(yd), 0))]
            st.info(f"Regra A (conservadora): **{rA}**")

            # ===== Opção B: usa um ponto médio para desempatar =====
            lat_m = (lat_o + lat_d) / 2.0
            lon_m = (lon_o + lon_d) / 2.0
            # Se BRs diferentes, por simplicidade usamos a BR de origem no ponto médio (poderia ser uma escolha do usuário)
            bm = br_o
            del_m, postos_m = inferir_delegacia_e_postos(bm)
            Xm = montar_linha(dia_semana, bm, (km_o + km_d) / 2.0, fase_dia, sentido_via, cond_met,
                              tipo_pista, tracado_via, lat_m, lon_m, del_m, postos_m)
            ym = pipe.predict(Xm)[0]
            if hasattr(le, "inverse_transform"):
                try:
                    ym = le.inverse_transform([ym])[0]
                except Exception:
                    pass

            st.write(f"Ponto Médio (lat {lat_m:.5f}, lon {lon_m:.5f}) ⇒ **{ym}**")

            votos = [risco2num.get(str(yo), 0), risco2num.get(str(yd), 0), risco2num.get(str(ym), 0)]
            # maioria simples; se empatar, cai na conservadora
            maj = max(set(votos), key=votos.count)
            if votos.count(maj) == 1:  # todos diferentes
                rB = rA
            else:
                rB = num2risco[maj]

            st.success(f"Regra B (maioria com ponto médio): **{rB}**")

            # Pequena conclusão
            st.markdown(
                "- **Regra A** é mais conservadora (toma o pior entre origem e destino).  \n"
                "- **Regra B** considera também um ponto intermediário simples (média das coordenadas), "
                "o que pode reduzir falsos alarmes ou captar risco no “meio do caminho”."
            )

        except Exception as e:
            st.error(f"Erro ao prever: {e}")
