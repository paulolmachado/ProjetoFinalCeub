import pandas as pd
from typing import List, Union

class Preprocessador:
    def __init__(self):
        self.colunas_desejadas = ['sgUF', 'sgPartido', 'txtDescricao', 'numAno']

    def preprocessar(self, dados: Union[pd.DataFrame, List]) -> pd.DataFrame:
        # Converte se necessário
        if isinstance(dados, list):
            df_raw = pd.DataFrame([d.dict() for d in dados])
        else:
            df_raw = dados.copy()

        # Aplicação das regras
        df = df_raw.copy()       
        df = df[df['sgUF'].notnull()]
        df = df[df['sgPartido'].notnull()]
        df = df[self.colunas_desejadas].copy()
        df['numAno'] = df['numAno'].astype(int).astype(str)
        
        return df
