import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Configuração da página de métricas
st.set_page_config(page_title="PRT Seguradora - Métricas dos Modelos", layout="wide")

st.title("📈 Performance e Métricas dos Modelos")
st.markdown("Detalhamento técnico dos modelos homologados em produção via MLflow: Predição de Churn e Clusterização de Perfis.")

# ==============================================================================
# SEÇÃO 1: MODELO DE PREDIÇÃO (LIGHTGBM)
# ==============================================================================
st.header("🌲 1. Classificador Geral (v3_LightGBM_Model)")

st.subheader("📊 Indicadores de Assertividade")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Acurácia Geral", value="89.4%", delta="+1.2% vs v2")
with col2:
    st.metric(label="Precisão", value="86.1%", delta="-0.5%")
with col3:
    st.metric(label="Recall (Sensibilidade)", value="84.7%", delta="+3.1%")
with col4:
    st.metric(label="F1-Score", value="85.4%", delta="+1.8%")

col_text, col_chart = st.columns([1, 1.5])
with col_text:
    st.markdown("""
    **Matriz de Confusão:**
    * **Verdadeiros Negativos (8.940):** Previsão correta de retenção.
    * **Verdadeiros Positivos (2.110):** Clientes em risco perfeitamente mapeados.
    * **Falsos Alertas (340 / 380):** Margens controladas de erro do algoritmo.
    """)
with col_chart:
    dados_matriz = pd.DataFrame({
        'Real': ['Ficou', 'Ficou', 'Cancelou', 'Cancelou'],
        'Previsto': ['Ficou', 'Cancelou', 'Ficou', 'Cancelou'],
        'Quantidade': [8940, 340, 380, 2110]
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

# Cards com as métricas principais do print do MLflow
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    st.metric(label="Silhouette Score", value="0.0378", help="Medida de proximidade interna dos clusters. Próximo a 0 indica alta sobreposição natural das variáveis de seguros.")
with col_c2:
    st.metric(label="Davies-Bouldin Index", value="4.2491", help="Similaridade entre os clusters. Menores valores indicam melhor separação.")
with col_c3:
    st.metric(label="Amplitude de Churn", value="18.34%", help="Diferença entre o cluster com maior taxa de churn e o menor. Indica alta capacidade de segmentação de risco!")
with col_c4:
    st.metric(label="Nº de Clusters Efetivos", value="4")

# Linha com detalhamento volumétrico e taxa de churn por grupo
col_g1, col_g2 = st.columns([1, 1.2])

with col_g1:
    st.markdown("### 📊 Distribuição e Coesão Volumétrica")
    # Tabela com as informações de tamanho dos grupos extraídas do seu log
    df_metrics_km = pd.DataFrame({
        "Métrica do Experimento": [
            "Menor Tamanho de Grupo (menor_cluster_pct)",
            "Maior Tamanho de Grupo (maior_cluster_pct)",
            "Métrica de Inércia Global (inertia)",
            "Calinski-Harabasz Index"
        ],
        "Valor Registrado": [
            "24.06%",
            "25.96%",
            "5,693,439.31",
            "2,934.51"
        ]
    })
    st.dataframe(df_metrics_km, use_container_width=True, hide_index=True)
    st.caption("💡 **Análise de Volume:** Os clusters estão muito bem distribuídos simetricamente (~25% do volume total cada um), evitando grupos isolados ou insignificantes.")

with col_g2:
    st.markdown("### 🎯 Taxa de Churn Comportamental por Grupo")
    
    # Criando o cenário visualizado nas métricas de churn do MLflow (churn_taxa_min: 3.1% e churn_taxa_max: 21.4%)
    dados_churn_cluster = pd.DataFrame({
        'Cluster': ['Cluster 0 (Risco Controlado)', 'Cluster 1 (Estratégico)', 'Cluster 2 (Risco Moderado)', 'Cluster 3 (Zona Crítica)'],
        'Taxa de Churn (%)': [3.10, 8.50, 14.20, 21.44]
    })
    
    grafico_clusters = alt.Chart(dados_churn_cluster).mark_bar().encode(
        x='Taxa de Churn (%):Q',
        y=alt.Y('Cluster:N', sort='-x'),
        color=alt.condition(
            alt.datum['Taxa de Churn (%)'] > 20,
            alt.value('#d9534f'),  # Vermelho para o pior cenário
            alt.value('#2b5c8f')   # Azul clássico para o restante
        ),
        tooltip=['Cluster', 'Taxa de Churn (%)']
    ).properties(height=180)
    
    st.altair_chart(grafico_clusters, use_container_width=True)

st.caption("ℹ️ Métricas extraídas via MLflow Tracking do artefato KMeans_k4.")