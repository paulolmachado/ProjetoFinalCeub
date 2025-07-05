# API/Modelo/modelo.py

import joblib
import pandas as pd
from pathlib import Path



# Constrói o caminho absoluto para a pasta do modelo
BASE_DIR = Path(__file__).resolve().parent

# Constrói o caminho para o artefato do modelo
MODELO_PATH = BASE_DIR / "Artefatos/modelo.bin"

def carregar_modelo():

    if not MODELO_PATH.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {MODELO_PATH}")
    
    # Agora estamos carregando um objeto CatBoostRegressor padrão
    modelo = joblib.load(MODELO_PATH)
    return modelo

def prever(modelo, dados_entrada):

    df_entrada = pd.DataFrame([dados_entrada])
    
    # O modelo já sabe como lidar com as features categóricas pois foi treinado assim
    predicao = modelo.predict(df_entrada)
    
    return predicao[0]