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
    try:
        # Verificação de segurança: garantir que é um DataFrame 2D
        if not isinstance(dados_entrada, pd.DataFrame):
            raise TypeError("Esperado pandas.DataFrame como entrada.")
        if dados_entrada.ndim != 2:
            raise ValueError(f"Entrada deve ter 2 dimensões. Recebido shape={dados_entrada.shape}")

        # Predição
        predicao = modelo.predict(dados_entrada)

        return predicao[0]  # retorna apenas o valor previsto
    
    except Exception as e:
        return {"erro": str(e)}