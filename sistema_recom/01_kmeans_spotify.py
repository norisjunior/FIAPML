# ============================================================
# 01_kmeans_spotify.py
# ============================================================
# Treina um modelo K-Means para agrupar músicas do Spotify
# por "vibe sonora" e salva:
#   - spotify_kmeans.pkl
#   - spotify_scaler.pkl
#   - spotify_clustered.csv
# ============================================================

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# === 1. Carregar dataset ===
url = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-spotify_songs.csv"
df = pd.read_csv(url)
print("|> Dataset carregado:", df.shape)

# === 2. Selecionar colunas ===
features = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo'
]
cols_keep = ['track_id','track_name','track_artist','playlist_name'] + features
df_model = df[cols_keep].dropna().copy()

# === 3. Remover duplicatas ===
df_model = df_model.drop_duplicates(subset=['track_name','track_artist']).reset_index(drop=True)
print(f"|> Após remover duplicatas: {len(df_model)} músicas únicas.")

# === 4. Normalizar ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model[features])

# === 5. Treinar K-Means ===
k = 6
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
df_model['cluster'] = kmeans.fit_predict(X_scaled)
print(f"|> K-Means treinado com K={k}")

# === 6. Salvar artefatos ===
joblib.dump(kmeans, 'spotify_kmeans.pkl')
joblib.dump(scaler, 'spotify_scaler.pkl')
df_model.to_csv('spotify_clustered.csv', index=False)

print("\n|> Artefatos salvos com sucesso:")
print("- spotify_kmeans.pkl")
print("- spotify_scaler.pkl")
print("- spotify_clustered.csv")

# === 7. Sanity checks ===
print("\nDistribuição de músicas por cluster:")
print(df_model['cluster'].value_counts())

print("\nMédias das features por cluster:")
print(df_model.groupby('cluster')[features].mean().round(2))
