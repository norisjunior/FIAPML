# Passo a Passo para Configurar e Rodar no VSCode

## 1. Configurar o ambiente no VSCode

**1. Abra o VSCode (norisjunior/FIAPML) e crie uma pasta para o projeto:**
- Crie uma nova pasta (ex.: `insurance`).

**2. Crie um ambiente virtual (opcional, mas recomendado):**
- Abra o terminal integrado do VSCode (Terminal > New Terminal).
- Crie o ambiente virtual:
```
python -m venv insuranceEnv
```

- Ative o ambiente virtual:
    - No Windows: `.\insuranceEnv\Scripts\Activate.ps1`
    - No macOS/Linux: `source insuranceEnv/bin/activate`

Após ativar, você verá `(insurance)` no terminal.
--> venv ativo:
(insurance) PS D:\GitHub\FIAPML> 

**3. Instale as dependências:**
pip install pandas scikit-learn joblib fastapi uvicorn streamlit requests

**4. Entre no diretório streamlit `cd insurance`**

**5. Insira o arquivo com o modelo .pkl treinado**
- Exemplo: insurance_pipeline.pkl

**6. Execute a API**
uvicorn app:app --reload

**7. Execute o website**
streamlit run website.py

