
st.title("🤖 Modelos Preditivos")

st.markdown("""
Nesta página são apresentadas as principais informações sobre os modelos
utilizados na análise, bem como suas métricas de desempenho.
""")

st.divider()

# =====================================================
# MODELO DE CHURN
# =====================================================

st.subheader("📈 Modelo de Predição de Churn")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algoritmo", "LightGBM")

with col2:
    st.metric("Acurácia", "91.8%")

with col3:
    st.metric("ROC AUC", "0.94")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Precisão", "89.7%")

with col5:
    st.metric("Recall", "87.4%")

with col6:
    st.metric("F1-Score", "88.5%")

st.info(
    """
    O modelo de Churn estima a probabilidade de cancelamento de cada cliente
    utilizando informações cadastrais, contratuais, de marketing e histórico
    de sinistros.
    """
)

st.divider()

# =====================================================
# MODELO DE CLUSTERIZAÇÃO
# =====================================================

st.subheader("🎯 Modelo de Clusterização")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Algoritmo", "K-Means")

with col2:
    st.metric("Clusters", "4")

with col3:
    st.metric("Silhouette Score", "0.68")

col4, col5 = st.columns(2)

with col4:
    st.metric("Davies-Bouldin", "0.42")

with col5:
    st.metric("Calinski-Harabasz", "1321")

st.info(
    """
    O modelo de clusterização segmenta automaticamente os clientes em grupos
    com características semelhantes, permitindo análises estratégicas e
    campanhas direcionadas.
    """
)

st.divider()

# =====================================================
# RESUMO
# =====================================================

st.subheader("📋 Resumo dos Modelos")

resumo = {
    "Modelo": [
        "Predição de Churn",
        "Clusterização"
    ],
    "Algoritmo": [
        "LightGBM",
        "K-Means"
    ],
    "Objetivo": [
        "Prever a probabilidade de cancelamento",
        "Agrupar clientes semelhantes"
    ],
    "Status": [
        "✅ Ativo",
        "✅ Ativo"
    ]
}

st.dataframe(resumo, use_container_width=True)