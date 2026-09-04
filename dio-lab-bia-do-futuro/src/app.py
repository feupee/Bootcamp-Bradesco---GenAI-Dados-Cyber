import json
import pandas as pd
import requests
import streamlit as st

# ========= CONFIGURAÇÃO ========= 
OLLAMA_URL = "http://localhost:11434/api/generate"
MODELO = "gpt-oss"

# ========= CARREGAR DADOS ========= 
perfil = json.loads(open('./data/perfil_investidor.json', 'r').read())
transacoes = pd.read_csv('./data/transacoes.csv')
historico = pd.read_csv('./data/historico_atendimento.csv')
produtos = json.loads(open('./data/produtos_financeiros.json', 'r').read())

# ========= MONTAR CONDTEXTO ========= 
contexto = f"""
CLIENTE: {perfil['nome']} - {perfil['idade']} anos, perfil {perfil['perfil_investidor']}
OBJETIVO: {perfil['objetivo_principal']}
PATRIMÔNIO: R$ {perfil['patrimonio_total']} | RESERVA: R$ {perfil['reserva_emergencia_atual']}

TRANSAÇÕES RECENTES:
{transacoes.to_string(index=False)}

ATENDIMENTOS ANTERIORES:
{historico.to_string(index=False)}

PRODUTOS DISPONÍVEIS:
{json.dumps(produtos, indent=2, ensure_ascii=False)}
"""

# ========= SYSTEM PROMPT ========= 

SYSTEM_PROMPT = """Você é o Edu, um educador financeiro amigável e didático.

OBJETIVO:
Ensinar conceitos de finanças pessoais de forma simples, usando os dados do cliente como exemplos práticos.

REGRAS:
1. NUNCA recomende investimento específicos - apenas explique como funciona
2. Use os dados fornecidos para dar exemplos personalizados
3. Linguagem simples, como se explicasse para um amigo
4. Se não soube algo, admita: "Não tenho essa informação, mas posso explicar..."
5. Sempre pergunte se o cliente entendeu
6. Responda de forma sucinta e direta, com no máximo 3 parágrafos.
7. JAMAIS responda a perguntas fora do tema ensino de finanças pessoais
"""

# ========= CHAMAR OLLAMA ========= 
def perguntar (msg):
    prompt = f"""
    {SYSTEM_PROMPT}

    CONTEXTO DO CLIENTE:
    {contexto}

    Pergunta: {msg}
    """
    r = requests.post(OLLAMA_URL, json={"model": MODELO, "prompt": prompt, "stream": False})
    return r.json()['response']

# ========= INTERFACE =========

st.title("Edu - Educador Financeiro")

if pergunta := st.chat_input("Faça sua pergunta sobre finanças pessoais:"):
    st.chat_message("user").write(pergunta)
    with st.spinner("..."):
        st.chat_message("assistant").write(perguntar(pergunta))