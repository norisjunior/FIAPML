import warnings, datetime, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import sklearn


url = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-Churn_Modelling.csv"
df = pd.read_csv(url)

target = "Exited"
cols_to_drop = ["RowNumber", "CustomerId", "Surname"]
df = df.drop(columns=cols_to_drop)

print(f"Dataset: {df.shape[0]} linhas | {df.shape[1]} colunas")



# X (features), y (target)
X = df.drop(columns=[target])
y = df[target]

# Split (estratificado)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=255)

# Separação das colunas categóricas das colunas numéricas
cat_cols = ["Geography", "Gender"]
num_cols = ["CreditScore","Age","Tenure","Balance","NumOfProducts",
            "HasCrCard","IsActiveMember","EstimatedSalary"]


# Compatibilidade sklearn (sparse_output em versões novas; sparse em antigas)
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", drop=None, sparse=False)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", make_ohe(),       cat_cols),
])


pipe_nivel3 = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(random_state=255))  # placeholder trocado no grid
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=255)

param_grid_multi = [
    # KNN
    {
        "clf": [KNeighborsClassifier()],
        "clf__n_neighbors": [5, 10, 15],
        "clf__weights": ["uniform", "distance"],
    },
    # DecisionTree
    {
        "clf": [DecisionTreeClassifier(random_state=255)],
        "clf__max_depth": [1, 6, 10],
        "clf__min_samples_split": [2, 4],
    },
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
    },
    # LightGBM
    {
        "clf": [LGBMClassifier(random_state=255)],
        "clf__n_estimators": [21, 41],
        "clf__learning_rate": [0.05, 0.1],
        "clf__num_leaves": [20, 50],
    },
]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

grid_multi = GridSearchCV(
    pipe_nivel3,
    param_grid=param_grid_multi,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_multi.fit(X_train, y_train_enc)

best_pipe = grid_multi.best_estimator_

print("Melhor modelo:", type(best_pipe.named_steps["clf"]).__name__)
print("Melhor acurácia (CV): {:.4f}".format(grid_multi.best_score_))
print("Acurácia (teste):     {:.4f}".format(best_pipe.score(X_test, y_test_enc)))



filename = f"churn_pipeline.pkl"

# Extrai apenas o classificador já treinado (após preprocessamento)
# Vamos treinar o preprocessor separadamente para extrair os transformadores já ajustados
preprocessor_fitted = best_pipe.named_steps["prep"]
classifier_trained = best_pipe.named_steps["clf"]

# Extrai os transformadores ajustados
num_transformer = preprocessor_fitted.named_transformers_["num"]  # StandardScaler ajustado
cat_transformer = preprocessor_fitted.named_transformers_["cat"]  # OneHotEncoder ajustado

import pickle

art = {
    "classifier": classifier_trained,
    "num_transformer": num_transformer,  # StandardScaler já ajustado
    "cat_transformer": cat_transformer,  # OneHotEncoder já ajustado
    "target_encoder": le,
    "feature_order": list(X.columns),
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "meta": {
        "cat_options": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols},
        "num_defaults": {c: float(df[c].median()) for c in num_cols}
    },
    "cv": {
        "best_score": float(grid_multi.best_score_),
        "best_params": grid_multi.best_params_
    },
    "versions": {
        "sklearn": sklearn.__version__,
        "xgboost": __import__("xgboost").__version__,
        "lightgbm": __import__("lightgbm").__version__,
    }
}

with open(filename, 'wb') as f:
    pickle.dump(art, f, protocol=4)

print("Salvo em:", filename)



