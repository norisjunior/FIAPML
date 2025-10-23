# ============================================================
# 03_app_streamlit.py
# ============================================================
# Executar com: streamlit run 03_app_streamlit.py
# Recomendador simples com K-Means + Similaridade de Cosseno
# ============================================================

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# 1. Configuração da página
# ------------------------------------------------------------
st.set_page_config(page_title="Spotify Recommender", page_icon="🎧", layout="wide")
st.title("🎧 Recomendador de Músicas — K-Means + Similaridade de Cosseno")

st.markdown("""
Digite o nome de uma música e receba recomendações com a mesma "vibe sonora".  
O modelo usa **K-Means** para segmentar músicas e **Cosseno** para medir similaridade dentro do cluster.
""")

# ------------------------------------------------------------
# 2. Carregar artefatos
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('spotify_clustered.csv')
    model = joblib.load('spotify_kmeans.pkl')
    scaler = joblib.load('spotify_scaler.pkl')
    return df, model, scaler

df, model, scaler = load_data()

features = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo'
]

# ------------------------------------------------------------
# 3. Funções utilitárias
# ------------------------------------------------------------
def localizar_musica(df, nome: str) -> pd.DataFrame:
    nome = nome.strip().lower()
    exata = df['track_name'].str.lower() == nome
    parcial = df['track_name'].str.lower().str.contains(nome, na=False)
    return df[exata | parcial]

def similares_por_cosseno(df, scaler, track_row, topn=5):
    cluster = int(track_row['cluster'])
    sub_df = df[df['cluster'] == cluster].copy()
    sub_df = sub_df.drop_duplicates(subset=['track_name','track_artist']).reset_index(drop=True)

    X_scaled = scaler.transform(sub_df[features])
    sim = cosine_similarity(X_scaled)

    idx_ref = sub_df.index[sub_df['track_name'].str.lower() == track_row['track_name'].lower()]
    if idx_ref.empty:
        return cluster, pd.DataFrame()
    idx_ref = idx_ref[0]

    scores = pd.Series(sim[idx_ref], index=sub_df.index).drop(index=idx_ref)
    scores = scores.sort_values(ascending=False).head(topn)

    out = sub_df.loc[scores.index, ['track_name','track_artist','playlist_name']].copy()
    out['similaridade'] = scores.values
    return cluster, out

# ------------------------------------------------------------
# 4. Interface
# ------------------------------------------------------------
musica = st.text_input("🎵 Digite o nome da música (ex.: Before the Start):")
num_recos = st.slider("Quantas recomendações deseja?", 3, 10, 5, step=1)

if st.button("Gerar Recomendações"):
    if not musica:
        st.warning("Por favor, digite o nome de uma música.")
    else:
        resultados = localizar_musica(df, musica)
        if resultados.empty:
            st.error("Música não encontrada (nem exata nem parcial).")
        else:
            ref = resultados.iloc[0]
            st.success(f"Base: **{ref['track_name']}** — {ref['track_artist']}")
            st.caption(f"Playlist original: {ref['playlist_name']}")
            cluster, similares = similares_por_cosseno(df, scaler, ref, topn=num_recos)
            st.info(f"Cluster identificado: **{cluster}**")
            if similares.empty:
                st.warning("Nenhuma recomendação disponível para este cluster.")
            else:
                st.subheader(f"🎧 {num_recos} músicas semelhantes (Cosseno):")
                st.dataframe(similares.reset_index(drop=True))

st.markdown("---")
st.caption("Machine Learning & Modeling | Exemplo didático: K-Means + Similaridade de Cosseno")
