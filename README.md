# ProjetoFinalCeub
Trabalho final do curso de ciência de dados do UniCeub.

# =====================================
# REQUISITOS E COMANDOS PARA RODAR O PROJETO
# =====================================

# ✅ 1. Crie o ambiente virtual (usando conda ou venv)
`conda create -n api-ml`
`conda activate api-ml`
# OU
`python -m venv venv`
`venv\\Scripts\\activate (Windows) ou source venv/bin/activate (Linux/Mac)`

# ✅ 2. Instale as dependências do projeto:
`pip install -r requirements.txt`

# ✅ 3. (Opcional) Treine o modelo e gere os artefatos:
`python treinar_modelo.py`

# ✅ 4. Inicie a API:
`uvicorn servico:app --reload`

# ✅ 5. Acesse a API no navegador:
http://127.0.0.1:8000/docs

# ✅ 6. Testar e verificar a Probabilidade:
`python teste_api.py`

# =====================================