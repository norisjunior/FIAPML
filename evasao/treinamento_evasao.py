# -*- coding: utf-8 -*-
"""
Treinamento Local - Evasão Escolar
"""

# =========================================================
# 1. Importações
# =========================================================
print("\nImportando bibliotecas...\n")
import numpy as np
import pandas as pd
import joblib
import datetime

# Pré-processamento
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Modelos
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# =========================================================
# 2. Leitura do dataset
# =========================================================
url = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-StudentsDropoutAcademicSuccess.csv"
df = pd.read_csv(url, sep=";")

print("\nDataset carregado:", df.shape, "linhas x colunas")

# =========================================================
# 3. Criação das features PCA
# =========================================================
colunas_desempenho = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (without evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)'
]

print("\nPCA...\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[colunas_desempenho])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PCA1_curricular"] = pca_result[:, 0]
df["PCA2_curricular"] = pca_result[:, 1]

# Remove colunas redundantes
df.drop([
    'Nationality', "Mother's occupation", "Father's qualification",
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (without evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Inflation rate', 'GDP', 'Unemployment rate'
], axis=1, inplace=True)

print("Número de features:", df.shape[1])

# =========================================================
# 4. Target e separação treino/teste
# =========================================================
# Cria dummy binária (Dropout = 1, demais = 0)
df["Target"] = (df["Target"] == "Dropout").astype(int)

X = df.drop(columns=["Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=255
)

# =========================================================
# 5. Treinamento com GridSearchCV
# =========================================================
print("\nTreinamento - GridSearchCV...\n")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=255))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=255)

param_grid = {
    "clf__n_estimators": [60, 100],
    "clf__learning_rate": [0.05, 0.1],
    "clf__max_depth": [3, 5],
}

grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train_enc)

print("\n===== RESULTADOS =====")
print("Melhor modelo:", type(grid.best_estimator_.named_steps["clf"]).__name__)
print("Melhores parâmetros:", grid.best_params_)
print("Melhor acurácia (CV): {:.4f}".format(grid.best_score_))
print("Acurácia teste: {:.4f}".format(grid.score(X_test, y_test_enc)))

# =========================================================
# 6. Serialização do modelo completo
# =========================================================
print("\nSerialização (.pkl)\n")

filename = f"evasao_pipeline_completo.pkl"
joblib.dump({
    "model": grid,
    "label_encoder": le,
    "scaler_pca": {
        "scaler": scaler,
        "pca": pca
    }
}, filename)

print("\nModelo salvo com sucesso:", filename)
