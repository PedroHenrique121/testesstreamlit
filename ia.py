# ===============================
# 📦 IMPORTAÇÕES
# ===============================
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# ⚙️ CONFIGURAÇÃO DA PÁGINA
# ===============================
st.set_page_config(page_title="IA da Pizza 🍕", page_icon="🍕")
st.title("🍕 IA que recomenda sabores de pizza")
st.write("Descubra qual sabor de pizza combina com cada cliente!")

# ===============================
# 📂 CARREGAR DADOS
# ===============================
try:
    dados = pd.read_csv("clientes.csv")
except FileNotFoundError:
    st.error("❌ Arquivo 'clientes.csv' não encontrado! Coloque-o na mesma pasta do app.")
    st.stop()

# ===============================
# 🔢 ENCODER (CONVERSÃO TEXTO → NÚMERO)
# ===============================
encoder_queijo = LabelEncoder()
encoder_carne = LabelEncoder()
encoder_sabor = LabelEncoder()

dados['gosta_de_queijo'] = encoder_queijo.fit_transform(dados['gosta_de_queijo'])
dados['prefere_carne'] = encoder_carne.fit_transform(dados['prefere_carne'])
dados['sabor_pizza'] = encoder_sabor.fit_transform(dados['sabor_pizza'])

# ===============================
# 📊 DIVIDIR DADOS EM TREINO E TESTE
# ===============================
X = dados[['idade', 'gosta_de_queijo', 'prefere_carne']]
y = dados['sabor_pizza']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)

# ===============================
# 🧠 TREINAR O MODELO
# ===============================
modelo = DecisionTreeClassifier(random_state=42)
modelo.fit(X_treino, y_treino)

# ===============================
# 📈 AVALIAR A IA
# ===============================
y_pred = modelo.predict(X_teste)
acuracia = accuracy_score(y_teste, y_pred)

# ===============================
# 💬 INTERFACE DO USUÁRIO
# ===============================
st.header("Preencha as preferências do cliente:")

idade = st.number_input("Idade do cliente:", min_value=0, max_value=120, step=1)
gosta_queijo = st.radio("Gosta de queijo?", ["sim", "não"])
prefere_carne = st.radio("Prefere carne?", ["sim", "não"])

# Converter respostas em números com os encoders treinados
gosta_num = encoder_queijo.transform([gosta_queijo])[0]
carne_num = encoder_carne.transform([prefere_carne])[0]

# ===============================
# 🖲️ BOTÃO PARA RODAR A PREVISÃO
# ===============================
if st.button("🍕 Recomendar Sabor"):
    previsao = modelo.predict([[idade, gosta_num, carne_num]])[0]
    sabor_sugerido = encoder_sabor.inverse_transform([previsao])[0]
    st.success(f"Sabor recomendado: **{sabor_sugerido.capitalize()}** 🍕")

# ===============================
# 📊 EXIBIR MÉTRICAS
# ===============================
st.subheader("📊 Desempenho da IA")
st.write(f"**Acurácia (taxa de acerto):** {acuracia * 100:.2f}%")

# ===============================
# 📈 OPCIONAL: VISUALIZAR OS DADOS
# ===============================
with st.expander("👀 Ver dados usados no treinamento"):
    st.dataframe(dados)
