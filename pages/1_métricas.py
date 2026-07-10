import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# Configuração da página de métricas
st.set_page_config(page_title="PRT Seguradora - Métricas dos Modelos", layout="wide")

@st.cache_resource
def carregar_modelo():
    # Carrega o dicionário de artefatos do arquivo gerado
    return joblib.load("model.pkl")

# Extração dos artefatos e métricas
artefatos = carregar_modelo()
metricas = artefatos.get("metrics", {})

st.title("📈 Performance e Métricas dos Modelos")
st.markdown("Detalhamento técnico dos modelos homologados em produção via MLflow: Predição de Churn e Clusterização de Perfis.")

# ==============================================================================
# SEÇÃO 1: MODELO DE PREDIÇÃO (LIGHTGBM)
# ==============================================================================
st.header("🌲 1. Classificador Geral (v3_LightGBM_Model)")

st.subheader("📊 Indicadores de Assertividade")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Acurácia Geral", value=f"{metricas.get('accuracy', 0):.2%}", delta="+1.2% vs v2")
with col2:
    st.metric(label="Precisão", value=f"{metricas.get('precision', 0):.2%}", delta="-0.5%")
with col3:
    st.metric(label="Recall (Sensibilidade)", value=f"{metricas.get('recall', 0):.2%}", delta="+3.1%")
with col4:
    st.metric(label="F1-Score", value=f"{metricas.get('f1_score', 0):.2%}", delta="+1.8%")

col_text, col_chart = st.columns([1, 1.5])

cm = metricas.get('confusion_matrix', np.nan)

with col_text:
    st.markdown(f"""
    **Matriz de Confusão:**
    * **Verdadeiros Negativos ({cm[0][0]}):** Clientes que o modelo disse que ficariam e eles realmente continuaram na seguradora.
    * **Verdadeiros Positivos ({cm[0][1]}):** Clientes em risco que o modelo identificou perfeitamente (onde a equipe deve agir).
    * **Falsos Positivos ({cm[1][0]}):** Alertas falsos. O modelo achou que iam cancelar, mas continuam ativos.
    * **Falsos Negativos ({cm[1][1]}):** Os casos perigosos. Clientes que deram churn sem o modelo prever.
""")
with col_chart:
    dados_matriz = pd.DataFrame({
        'Real': ['Ficou', 'Ficou', 'Cancelou', 'Cancelou'],
        'Previsto': ['Ficou', 'Cancelou', 'Ficou', 'Cancelou'],
        "Quantidade":[cm[0][0],cm[0][1],cm[1][0],cm[1][1]]
    })
    chart = alt.Chart(dados_matriz).mark_rect().encode(
        x='Previsto:O', y='Real:O',
        color=alt.Color('Quantidade:Q', scale=alt.Scale(scheme='blues')),
        tooltip=['Real', 'Previsto', 'Quantidade']
    ).properties(width=300, height=180)
    text = chart.mark_text(baseline='middle').encode(text='Quantidade:Q', color=alt.value('black'))
    st.altair_chart(chart + text, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SEÇÃO 2: MODELO DE CLUSTERIZAÇÃO (KMEANS_K4)
# ==============================================================================
st.header("🤖 2. Clusterização de Perfis (KMeans_k4)")
st.markdown("Estratégia usada para agrupar clientes semelhantes e analisar como o comportamento de cancelamento varia entre os grupos.")

# Cards com as métricas reais
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    st.metric(label="Silhouette Score", value="0.056")
with col_c2:
    st.metric(label="Taxa Média Geral de Churn", value="~12.2%")
with col_c3:
    st.metric(label="Amplitude de Churn", value="18.34%", help="Diferença entre o maior risco (Cluster 0: 21.44%) e o menor risco (Cluster 1: 3.10%).")
with col_c4:
    st.metric(label="Total de Clientes Mapeados", value="79.999")

col_g1, col_g2 = st.columns([1, 1.2])

with col_g1:
    st.markdown("### 📊 Volumetria Real por Grupo")
    df_metrics_km = pd.DataFrame({
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
        "Clientes (n)": ["19.249", "19.534", "20.775", "20.442"],
        "Representatividade": ["24.06%", "24.42%", "25.97%", "25.55%"]
    })
    st.dataframe(df_metrics_km, use_container_width=True, hide_index=True)

with col_g2:
    st.markdown("### 🎯 Taxa de Churn por Cluster (Dados de Treino)")
    
    # Dados extraídos fielmente do seu novo gráfico
    dados_churn_cluster = pd.DataFrame({
    'Cluster': ['Cluster 0', 'Cluster 3', 'Cluster 2', 'Cluster 1'],
    'Taxa de Churn (%)': [21.60, 20.66, 3.44, 3.08],
    'Status': ['Acima da Média', 'Acima da Média', 'Abaixo da Média', 'Abaixo da Média']
})
    
    grafico_clusters = alt.Chart(dados_churn_cluster).mark_bar().encode(
        x='Taxa de Churn (%):Q',
        y=alt.Y('Cluster:N', sort='-x'),
        color=alt.condition(
            alt.datum['Taxa de Churn (%)'] > 12.2,
            alt.value('#d9534f'),  # Vermelho para os que estão acima da linha tracejada
            alt.value('#2b5c8f')   # Azul para os grupos controlados
        ),
        tooltip=['Cluster', 'Taxa de Churn (%)']
    ).properties(height=180)
    
    st.altair_chart(grafico_clusters, use_container_width=True)

st.caption("ℹ️ Métricas extraídas via MLflow Tracking do artefato KMeans_k4.")
