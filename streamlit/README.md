# Passo a Passo para Configurar e Rodar no VSCode

## 1. Configurar o ambiente no VSCode

**1. Abra o VSCode e crie uma pasta para o projeto:**
- Crie uma nova pasta (ex.: `streamlit_churn`) e abra-a no VSCode (File > Open Folder).
- Salve os scripts Streamlit (ex.: `app_v1.py`, `app_v2.py`, `app_v3.py`) nessa pasta.

**2. Crie um ambiente virtual (opcional, mas recomendado):**
- Abra o terminal integrado do VSCode (Terminal > New Terminal).
- Crie o ambiente virtual:
```
python -m venv meuappstreamlit
```

- Ative o ambiente virtual:
    - No Windows: `venv\Scripts\activate`
    - No macOS/Linux: `source venv/bin/activate`

Após ativar, você verá `(venv)` no terminal.
--> Comando para criar o venv:
PS D:\GitHub\FIAPML> python -m venv meuappstreamlit

--> Comando para ativar o venv:
PS D:\GitHub\FIAPML> .\meuappstreamlit\Scripts\activate

--> venv ativo:
(meuappstreamlit) PS D:\GitHub\FIAPML> 

**3. Instale as dependências:**
pip install "numpy==1.26.4" "scipy==1.11.4" "pandas==2.2.2" "joblib==1.4.2" "threadpoolctl==3.4.0" "scikit-learn==1.4.2" "xgboost==2.0.3" "lightgbm==4.3.0"

pip install "streamlit==1.37.1"

**4. Entre no diretório streamlit `cd streamlit`**

**5. Execute o treinamento**
`python treinamento.py`

- O treinamento gerará o arquivo "churn_pipeline.pkl"

**6. Execute o streamlit**

`streamlit run .\App1_simples.py`

ou

`streamlit run .\App2_modelo_ML.py`

ou

`streamlit run .\App3_prod.py`
