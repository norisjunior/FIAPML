# Sistema de Recomendação Musical - dataset Spotify

> **Descubra músicas com a mesma "vibe sonora" usando Machine Learning!**

Este sistema utiliza **K-Means** para agrupar músicas por características sonoras e **Similaridade de Cosseno** para encontrar as mais similares dentro de cada grupo.

## Instalação

### Clone o repositório (ou baixe os arquivos)

```bash
# Se quiser clonar o repositório da disciplina
git clone https://github.com/norisjunior/FIAPML
cd sistema_recom

# Ou simplesmente crie a pasta e coloque os arquivos
mkdir sistema_recom
cd sistema_recom
```

### Crie e ative o ambiente virtual

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

> **Dica:** Você saberá que o ambiente está ativo quando ver `(venv)` no início do terminal

### Instale as dependências

```bash
pip install -r requirements.txt
```

---

## Como Usar

### Etapa 1: Treinamento do Modelo

Execute o script de treinamento para criar os modelos e processar os dados:

```bash
python 01_kmeans_spotify.py
```

**O que acontece:**
- Baixa o dataset do Spotify (~32k músicas)
- Remove duplicatas e limpa os dados
- Treina o modelo K-Means com 6 clusters
- Salva 3 arquivos essenciais:
  - `spotify_kmeans.pkl` (modelo)
  - `spotify_scaler.pkl` (normalizador)
  - `spotify_clustered.csv` (dados processados)

**Saída esperada:**
```
|> Dataset carregado: (32833, 23)
|> Após remover duplicatas: 28356 músicas únicas.
|> K-Means treinado com K=6
|> Artefatos salvos com sucesso!
```

### Etapa 2: Teste no Console (Modo Interativo)

Para testar rapidamente o sistema via terminal:

```bash
python 02_cosseno_importancia.py
```

**Como usar:**
1. O sistema mostrará 5 músicas aleatórias como exemplo
2. Digite o nome (ou parte) de uma música
3. Receba 5 recomendações similares!

### Etapa 3: Interface Gráfica (Streamlit)

Para uma experiência mais rica e visual:

```bash
streamlit run 03_app.py
```

**O que você verá:**
- O navegador abrirá automaticamente em `http://localhost:8501`
- Campo de busca inteligente
- Slider para escolher quantidade de recomendações (3-10)
- Tabela interativa com resultados

**Interface incluirá:**
- Nome e artista das músicas recomendadas
- Playlist de origem
- Score de similaridade (0-1)
- Cluster identificado

---

## Como Funciona

### Algoritmo em 2 Etapas:

#### 1. **K-Means Clustering**
```python
Features analisadas:
├── danceability    # Quão dançante é a música
├── energy          # Intensidade e atividade
├── loudness        # Volume médio
├── speechiness     # Presença de palavras faladas
├── acousticness    # Probabilidade de ser acústica
├── instrumentalness # Ausência de vocais
├── liveness        # Presença de audiência
├── valence         # Positividade musical
└── tempo           # BPM (batidas por minuto)
```

#### 2. **Similaridade de Cosseno**
- Calcula o ângulo entre vetores de features
- Quanto menor o ângulo, mais similar (0 = idêntico, 1 = oposto)
- Aplicada apenas dentro do mesmo cluster

### Fluxo Completo:
```
Música Input → Identificar Cluster → Filtrar músicas do cluster → 
Calcular Cosseno → Ordenar por similaridade → Top N recomendações
```
