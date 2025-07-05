# API/main.py

from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from Features.preprocessamento import Preprocessador
from Modelo.modelo import carregar_modelo, prever
import pandas as pd

app = FastAPI(
    title="API de Predição de Despesas Parlamentares",
    description="Prevê o valor líquido total de despesas com base na UF, partido, tipo de despesa e ano.",
    version="1.0.0"
)


modelo_catboost = carregar_modelo()

pre = Preprocessador()

class DadosEntrada(BaseModel):
    txNomeParlamentar: str
    cpf: str
    ideCadastro: int
    nuCarteiraParlamentar: Optional[int]
    nuLegislatura: int
    sgUF: str
    sgPartido: str
    codLegislatura: int
    numSubCota: Optional[int]
    txtDescricao: str
    numEspecificacaoSubCota: Optional[int]
    txtDescricaoEspecificacao: Optional[str]
    txtFornecedor: Optional[str]
    txtCNPJCPF: Optional[str]
    txtNumero: Optional[str]
    indTipoDocumento: Optional[int]
    datEmissao: Optional[str]
    vlrDocumento: float
    vlrGlosa: Optional[float]
    vlrLiquido: float
    numMes: int
    numAno: int
    numParcela: Optional[int]
    txtPassageiro: Optional[str]
    txtTrecho: Optional[str]
    numLote: Optional[int]
    numRessarcimento: Optional[int]
    datPagamentoRestituicao: Optional[str]
    vlrRestituicao: Optional[float]
    nuDeputadoId: Optional[int]
    ideDocumento: int
    urlDocumento: Optional[str]

# Define o endpoint /predict
@app.post("/predict")
def fazer_predicao(dados_entrada: DadosEntrada):
    try:
        df_tratado = pre.preprocessar(pd.DataFrame([dados_entrada.dict()]))
    
        # 1. Realiza a predição (retorna um float)
        predicao = prever(modelo_catboost, df_tratado)
        
        # 2. Formata o número para o padrão R$
        valor_formatado = f"R$ {predicao:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        
        # 3. Retorna o resultado já formatado
        return {
            "previsao_vlr_liquido": valor_formatado,
           # "dados_preprocessados": df_tratado.to_dict(orient="records")
        }
    except Exception as e:
        return {"erro": str(e)}