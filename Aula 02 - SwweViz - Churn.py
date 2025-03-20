# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:07:39 2025

@author: noris
"""

# Importar bibliotecas
import sweetviz as sv
import pandas as pd

url = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-Churn_Modelling.csv"
df = pd.read_csv(url)
df.head(10)

#Variável que armazena as features
features = df.columns[:-1]

# Gerar o relatório
relatorio = sv.analyze(df, target_feat='Exited')

# Exibir o relatório
relatorio.show_html()