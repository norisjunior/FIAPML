# app_v1.py
import streamlit as st

# Título do aplicativo
st.title("Minha Primeira Aplicação Streamlit")

# Descrição para os alunos
st.write("Este é um exemplo simples de aplicativo com Streamlit. Digite valores, selecione opções e clique no botão!")

# Campo 1: Entrada numérica
idade = st.number_input("Digite sua idade:", min_value=18, max_value=100, value=30)

# Campo 2: Seleção de uma lista (dropdown)
pais = st.selectbox("Selecione seu país:", ["França", "Alemanha", "Espanha"])

# Campo 3: Entrada de texto
nome = st.text_input("Digite seu nome:", "João")

# Botão para ação
if st.button("Enviar"):
    # Exibe os valores inseridos
    st.write(f"Nome: {nome}")
    st.write(f"Idade: {idade}")
    st.write(f"País: {pais}")
    st.success("Dados enviados com sucesso!")