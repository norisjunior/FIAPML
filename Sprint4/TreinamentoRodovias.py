import pandas as pd
df = pd.read_csv("brazil_car_accidents/accidents_2017_to_2023_portugues.csv")
df.head()

df.info()

"""## Pré-processamento de variáveis

### Feature Selection e informações gerais do dataset
"""

df['classificacao_acidente'].unique()

df.shape

df_cs = df[['dia_semana', 'br', 'km', 'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'latitude', 'longitude', 'delegacia', 'classificacao_acidente']]
df_cs.sample(7)

"""### Tratamento de nulos"""

df_cs.isnull().sum()

df_cs = df_cs.dropna()
df_cs

"""### Tratamento de duplicatas"""

duplicadas = df_cs.duplicated().sum()
print("Total de duplicadas:", duplicadas)

# Remover duplicatas
df_cs = df_cs.drop_duplicates()
duplicadas = df_cs.duplicated().sum()
print("Depois do tratamento - Há duplicadas?", duplicadas)

# Selecionar apenas uma amostra aleatória de 20000 observações:
df_cs = df_cs.sample(50000, random_state=255)

"""### Tratamento da variável target"""

#1, a)
# Ajustando os nomes da variável target:
df_cs['classificacao_acidente'] = df_cs['classificacao_acidente'].replace(['Com Vítimas Feridas', 'Com Vítimas Fatais', 'Sem Vítimas'], ['Risco Médio', 'Risco Alto', 'Risco Baixo'])
df_cs.sample(10)

"""## Geração de dataset da Sprint 3

### Transformação das variáveis BR (para string) e KM (para float)
"""

# Trata variáveis: br para string e km trocando "," por "."
# Salva arquivo da Sprint 3
df_cs['br'] = 'BR-' + df_cs['br'].astype(int).astype(str)
df_cs['km'] = df_cs['km'].astype(str).str.replace(',', '.').astype(float)
df_cs.info()

"""### Feature Engineering - acréscimo da coluna "postos_policiais_PRF"

Essa informação veio de uma fonte de dados do Governo Federal que contém o número de postos policiais da PRF nas rodovias brasileiras. Estipulou-se que o número de postos por rodovia segue os tercis: 0 a 33% (baixo), 33% a 66% (médio) e acima de 66% (alto).

#### Obtenção dos postos policiais da PRF e merge com o dataset
"""

# Informações sobre postos da PRF:
# https://dados.antt.gov.br/dataset/prf/resource/361bdeca-2f4a-49f0-a97b-449caac0b7ca
url = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-dados_dos_postos_prfs.csv"
df_postos_prf = pd.read_csv(url, sep=";", encoding="latin1")

# Contagem de postos por BR
contagem_rodovias = df_postos_prf['rodovia'].value_counts().reset_index()
contagem_rodovias.columns = ['br', 'contagem']

# Atribuição de valores nos tercis por BR
bins = [0, contagem_rodovias['contagem'].quantile(0.33), contagem_rodovias['contagem'].quantile(0.66), contagem_rodovias['contagem'].max()]
labels = ['Baixa', 'Média', 'Alta']
contagem_rodovias['postos_policiais_PRF'] = pd.cut(contagem_rodovias['contagem'], bins=bins, labels=labels, include_lowest=True)

# Merge do dataset de contagem_rodovias no dataset de acidentes
df_cs = df_cs.merge(contagem_rodovias[['br', 'postos_policiais_PRF']], on='br', how='left')
df_cs['postos_policiais_PRF'] = df_cs['postos_policiais_PRF'].fillna('Baixa')

"""#### Geração do .csv"""

# Geração do .csv da Sprint 3
filename = 'dataset-Sprint3-acidentes.csv'
df_cs.to_csv(filename, index=False)

"""#### Hash do .csv"""

# Função de hash de arquivo
import hashlib
from pathlib import Path

def sha256sum(path, block_size=8192):
    """Retorna o hash SHA-256 hexadecimal de um arquivo."""
    h = hashlib.sha256()                       # construtor da família hashlib
    with open(path, "rb") as f:                # ler em modo binário
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)                    # alimenta o objeto de hash
    return h.hexdigest()

# Mostra o hash (SHA-256) do arquivo .csv
hash_hex = sha256sum(filename)
print("SHA-256:", hash_hex)

"""#### Geração dos metadados"""

# Geração dos metadados do modelo
import json
var_categoricas = ['dia_semana', 'br', 'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'delegacia', 'postos_policiais_PRF']
metadata = {}
for col in var_categoricas:
    metadata[col] = df_cs[col].unique().tolist()

with open('metadados_model.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print("Arquivo 'metadados_model.json' gerado com sucesso.")

"""## Geração do modelo (pipeline)

### Pré-processamento (separação features / target), scaling e encoding
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# X (Matriz de features), y (vetor da target)
# definição das features
features = ['dia_semana', 'br', 'km', 'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'latitude', 'longitude', 'delegacia', 'postos_policiais_PRF']
X = df_cs.loc[:, features]
y = df_cs['classificacao_acidente'].values

# Separação entre de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=255)

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

cat_cols = ['dia_semana', 'br', 'fase_dia', 'sentido_via', 'condicao_metereologica', 'tipo_pista', 'tracado_via', 'delegacia', 'postos_policiais_PRF']  # serão tratadas com OneHot
num_cols = [ 'km', 'latitude', 'longitude']

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)
    ]
)

"""### Treinamento usando GridSearchCV"""

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("\n===== NÍVEL 3 – Pipeline + GridSearch (multi‑model) =====")

# Pipeline base (X -> prep -> clf)
pipe_nivel3 = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier())  # placeholder, será trocado no grid
])

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=255)

param_grid_multi = [
    # RandomForest
    {
        "clf": [RandomForestClassifier(random_state=255)],
        "clf__n_estimators": [10, 20],
        "clf__max_depth": [1, 10],
    },
    # XGBoost
    {
        "clf": [XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=255)],
        "clf__n_estimators": [20, 40],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 5],
    }
]

grid_multi = GridSearchCV(
    pipe_nivel3,
    param_grid=param_grid_multi,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_multi.fit(X_train, y_train_enc)

print("Melhor modelo (nível 3):", type(grid_multi.best_estimator_.named_steps["clf"]).__name__)
print("Melhor acurácia CV: {:.4f}".format(grid_multi.best_score_))
print("Acurácia de teste: {:.4f}".format(grid_multi.score(X_test, y_test_enc)))

# Predição com rótulo original quando precisar
y_pred_enc = grid_multi.predict(X_test)
y_pred = le.inverse_transform(y_pred_enc)

"""---

---

## Serialização

### Geração dos metadados
"""

# === Metadados para embutir no .pkl ===
from datetime import datetime
import sklearn

# mesmas listas já definidas acima:
# features, cat_cols, num_cols, df_cs

meta = {
    "features": features,
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    # opções para preencher selects no Streamlit:
    "cat_options": {c: sorted(df_cs[c].dropna().astype(str).unique().tolist()) for c in cat_cols},
    # defaults numéricos simples para inputs:
    "num_defaults": {c: float(df_cs[c].median()) for c in num_cols},
    # classes originais do problema (para exibição)
    "target_classes": None,   # será preenchido após o LabelEncoder aprender as classes
    # rastreabilidade
    "sklearn_version": sklearn.__version__,
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

"""### Serialização e versionamento"""

# === Serialização em um ÚNICO .pkl contendo tudo que a app precisa ===
import joblib, hashlib
from pathlib import Path

# pegue somente o melhor pipeline já treinado (prep + melhor clf)
best_pipe = grid_multi.best_estimator_

# guarde as classes originais no meta
meta["target_classes"] = le.classes_.tolist()

# objeto único de artefatos
art = {
    "pipeline": best_pipe,     # Pipeline(prep -> melhor clf)
    "label_encoder": le,       # para inverse_transform no app
    "meta": meta,              # cat_options, num_defaults, etc.
}

filename = "risco_acidentes_pipeline.pkl"   # ajuste o nome se quiser
joblib.dump(art, filename)
print(f"Arquivo salvo: {filename}")

# (opcional) hash para conferência
def sha256sum(path, block_size=8192):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

print("SHA-256:", sha256sum(filename))

art2 = joblib.load(filename)
pipe2 = art2["pipeline"]
le2   = art2["label_encoder"]
meta2 = art2["meta"]

print("Exemplo de opções para selects no app:")
for c in meta2["cat_cols"][:3]:
    print(c, "→", meta2["cat_options"][c][:5], "...")  # primeiras 5

print("Defaults numéricos:", meta2["num_defaults"])
print("Classes do target:", meta2["target_classes"])