# 🤖 Edu — Educador Financeiro Inteligente

Agente financeiro inteligente desenvolvido durante o **Bootcamp Bradesco — GenAI & Dados**, utilizando **Inteligência Artificial Generativa**, **Ollama** e **Streamlit**.

O projeto apresenta um assistente capaz de responder dúvidas relacionadas à educação financeira utilizando informações contextualizadas de um cliente, como perfil de investidor, transações, histórico de atendimento e produtos financeiros disponíveis.

> ⚠️ O agente possui finalidade educacional. Ele não realiza recomendações específicas de investimentos.

---

## 📌 Sobre o Projeto

O **Edu** é um agente de Inteligência Artificial voltado para educação financeira.

Seu objetivo é transformar dados financeiros estruturados em um contexto que possa ser utilizado por um modelo de linguagem local, permitindo que o usuário faça perguntas e receba respostas personalizadas.

A aplicação combina:

- 🧠 **LLM local** através do Ollama
- 🖥️ **Interface web** através do Streamlit
- 🐍 **Python** como linguagem principal
- 📊 **Pandas** para manipulação dos dados
- 📁 Arquivos CSV e JSON como base de conhecimento

O projeto utiliza dados mockados para representar informações de um cliente, evitando a necessidade de utilizar dados financeiros reais.

---

## 🏗️ Arquitetura

O funcionamento do agente pode ser representado pelo seguinte fluxo:

```text
                    ┌──────────────────────┐
                    │      Usuário         │
                    │  Pergunta financeira │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Streamlit       │
                    │   Interface Web      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       app.py         │
                    │                      │
                    │ System Prompt        │
                    │ + Contexto Cliente   │
                    │ + Pergunta           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Ollama         │
                    │    LLM Local         │
                    │                      │
                    │      gpt-oss         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       Resposta       │
                    │      do Agente       │
                    └──────────────────────┘
```
