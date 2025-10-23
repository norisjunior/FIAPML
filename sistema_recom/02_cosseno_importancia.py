# ============================================================
# 02_cosseno_importancia.py
# ============================================================
# Demonstra a importância da Similaridade de Cosseno.
# O usuário digita uma música e recebe recomendações únicas.
# ============================================================

import pandas as pd
import joblib
import random
from sklearn.metrics.pairwise import cosine_similarity

# === 1. Carregar dados e modelos ===
print("|> Carregando modelos e dados...")
df = pd.read_csv('spotify_clustered.csv')
kmeans = joblib.load('spotify_kmeans.pkl')
scaler = joblib.load('spotify_scaler.pkl')

features = [
    'danceability','energy','loudness','speechiness','acousticness',
    'instrumentalness','liveness','valence','tempo'
]

print(f"|> Dataset carregado com {len(df)} músicas.")

# === 2. Exibir amostra de músicas ===
amostras = df[['track_name','track_artist','playlist_name']].sample(5, random_state=random.randint(0,1000))
print("\n|> Exemplos de músicas disponíveis:\n")
for i, row in amostras.iterrows():
    print(f"- {row['track_name']} — {row['track_artist']}  ({row['playlist_name']})")

# === 3. Entrada do usuário ===
musica_input = input("\n>> Digite o nome de uma música (exata ou parte do nome): ").strip().lower()

mask_eq = df['track_name'].str.lower() == musica_input
mask_ct = df['track_name'].str.lower().str.contains(musica_input, na=False)
candidatos = df[mask_eq | mask_ct]

if candidatos.empty:
    print("\n |X| Nenhuma música encontrada. Tente outro nome.")
    exit()

ref = candidatos.iloc[0]
print(f"\n||>> Música selecionada: {ref['track_name']} — {ref['track_artist']}")
print(f"|> Playlist original: {ref['playlist_name']}")
print(f"|> Cluster: {ref['cluster']}")

# === 4. Filtrar cluster e remover duplicatas ===
sub_df = df[df['cluster'] == ref['cluster']].copy()
sub_df = sub_df.drop_duplicates(subset=['track_name','track_artist']).reset_index(drop=True)

# === 5. Calcular Similaridade de Cosseno ===
X_scaled = scaler.transform(sub_df[features])
sim_matrix = cosine_similarity(X_scaled)

# Localizar música base
idx_ref = sub_df.index[sub_df['track_name'].str.lower() == ref['track_name'].lower()]
if idx_ref.empty:
    print("\n ! Música base não encontrada após limpeza.")
    exit()
idx_ref = idx_ref[0]

sim_scores = pd.Series(sim_matrix[idx_ref], index=sub_df.index)
sim_scores = sim_scores.drop(idx_ref).sort_values(ascending=False).head(5)

# === 6. Exibir recomendações ===
print("\n|> Músicas mais similares (Cosseno):\n")
for idx, score in sim_scores.items():
    linha = sub_df.loc[idx]
    print(f"- {linha['track_name']} — {linha['track_artist']} | Similaridade: {score:.3f}")

print("\n||>>Explicação:")
print("O K-Means agrupa músicas por 'vibe sonora' (vibe sonora = features).")
print("A Similaridade de Cosseno refina as recomendações dentro de cada grupo.")
print("--> K-Means = Segmentação/agrupamento")
print("--> Similaridade de Cosseno = Mais similares dentro do grupo")
