# API/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from Modelo.modelo import carregar_modelo, prever


app = FastAPI(
    title="API de Predição de Despesas Parlamentares",
    description="Prevê o valor líquido total de despesas com base na UF, partido, tipo de despesa e ano.",
    version="1.0.0"
)


modelo_catboost = carregar_modelo()


# Isso garante que a API receba os dados no formato correto
class DadosEntrada(BaseModel):
    sgUF: str
    sgPartido: str
    txtDescricao: str
    numAno: str

# Define o endpoint /predict
@app.post("/predict")
def fazer_predicao(dados_entrada: DadosEntrada):

    dados_dict = dados_entrada.dict()
    
    # 1. Realiza a predição (retorna um float)
    predicao = prever(modelo_catboost, dados_dict)
    
    # 2. Formata o número para o padrão R$
    valor_formatado = f"R$ {predicao:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    
    # 3. Retorna o resultado já formatado
    return {
        "previsao_vlr_liquido": valor_formatado,
        
    }