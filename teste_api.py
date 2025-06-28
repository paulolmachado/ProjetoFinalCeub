import requests


# Para fins de comparação, vamos guardar o valor real (original)
valor_real = 1.0  # Nunca fumou (classe 0)

# Criando a amostra (com o valor real incluído apenas para referência)
amostra = {
    "ano": 2023,
    "mes": 5,
    "nome_parlamentar": "Joaquim Silva",
    "partido": "PSDB",
    "estado":   "SP",  
    "tipo_despesa": "Combustível",
    "valor_reembolsado": 
}

# Envia para a API
url = "http://127.0.0.1:8000/predict/"
res = requests.post(url, json=amostra)

print("Status:", res.status_code)

try:
    resultado = res.json()
   
    print("Probabilidades:", resultado.get("probabilidades"))

except Exception as e:
    print("❌ Erro ao interpretar resposta:")
    print(res.text)
