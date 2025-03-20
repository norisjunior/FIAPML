# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:07:39 2025

@author: noris
"""

# Importar bibliotecas
import sweetviz as sv
import pandas as pd

url = "https://raw.githubusercontent.com/norisjunior/FIAPML/refs/heads/main/datasets/dataset-AQI-updated_pollution_dataset.csv"
df = pd.read_csv(url)
df.head(10)

#Variável que armazena as features
features = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
#cria um array mapeando a qualidade do ar em texto para números
air_quality_mapping = {"Good": 0, "Moderate": 1, "Poor": 2, "Hazardous": 3}
#cria nova variável "AQI"
df['AQI'] = df['Air Quality'].astype(str).map(air_quality_mapping)
df = df.drop('Air Quality', axis=1) #ou del df['Air Quality']

# Gerar o relatório
feature_config = sv.FeatureConfig(force_num=['AQI'])
relatorio = sv.analyze(df, target_feat='AQI', feat_cfg=feature_config)

# Exibir o relatório
relatorio.show_html()