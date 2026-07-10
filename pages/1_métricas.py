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
# SEÇÃO 1: MODELO DE PREDIÇÃO (LIGHTGBM) - MATRIZ DE CONFUSÃO AUMENTADA
# ==============================================================================
st.header("🌲 1. Classificador Geral (v3_LightGBM_Model)")

st.subheader("📊 Indicadores de Assertividade")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("AUC-ROC", f"82.83%")

with col2:
    st.metric("Acurácia Geral", f"{metricas.get('accuracy', 0):.2%}")

with col3:
    st.metric("Precisão", f"{metricas.get('precision', 0):.2%}")

with col4:
    st.metric("Recall (Sensibilidade)", f"{metricas.get('recall', 0):.2%}")

with col5:
    st.metric("F1-Score", f"{metricas.get('f1_score', 0):.2%}")

col_text, col_chart = st.columns([1, 1.5]) # Mantenho a estrutura de colunas

cm = metricas.get('confusion_matrix', [[0, 0], [0, 0]]) # Garante que cm exista
tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1] # Extração correta

with col_text:
    st.markdown(f"""
    **Matriz de Confusão:**
    * **Verdadeiros Negativos ({tn}):** Clientes que o modelo previu que ficariam e eles realmente continuaram na seguradora.
    * **Verdadeiros Positivos ({tp}):** Clientes em risco que o modelo identificou perfeitamente (onde a equipe deve agir).
    * **Falsos Positivos ({fp}):** Alertas falsos. O modelo achou que iam cancelar, mas continuam ativos.
    * **Falsos Negativos ({fn}):** Os casos perigosos. Clientes que deram churn sem o modelo prever.
    """)

# --- AQUI ESTÁ O AJUSTE DA MATRIZ ---
with col_chart:
    dados_matriz = pd.DataFrame({
        'Real': ['Ficou', 'Ficou', 'Cancelou', 'Cancelou'],
        'Previsto': ['Ficou', 'Cancelou', 'Ficou', 'Cancelou'],
        "Quantidade": [tn, fp, fn, tp]
    })
    
    # Criando o gráfico base
    base = alt.Chart(dados_matriz).encode(
        x=alt.X('Previsto:O', sort=['Ficou', 'Cancelou'], title="Classe Prevista"), # Adiciona título ao eixo X
        y=alt.Y('Real:O', sort=['Ficou', 'Cancelou'], title="Classe Real"), # Adiciona título ao eixo Y
    )
    
    # Configurando o quadrado da matriz com tamanho maior
    chart = base.mark_rect().encode(
        color=alt.Color('Quantidade:Q', scale=alt.Scale(scheme='blues'), legend=alt.Legend(title="Quantidade", labelFontSize=12, titleFontSize=14)), # Ajusta legenda
        tooltip=['Real', 'Previsto', 'Quantidade']
    ).properties(
        width=500,  # LARGURA AUMENTADA (era 300)
        height=350, # ALTURA AUMENTADA (era 180)
        title=alt.TitleParams(text="Matriz de Confusão (Tamanho Aumentado)", anchor='start', fontSize=18) # Título interno opcional
    )
    
    # Adicionando o texto dentro da matriz, também maior e com contraste
    text = base.mark_text(baseline='middle').encode(
        text='Quantidade:Q', 
        color=alt.condition(alt.datum.Quantidade > (max(tn, tp)/2), alt.value('white'), alt.value('black')),
        size=alt.value(16) # FONTE DO NÚMERO AUMENTADA
    )
    
    # Configurando o tamanho dos rótulos dos eixos
    final_chart = (chart + text).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    )
    
    st.altair_chart(final_chart, use_container_width=False) # use_container_width=False para respeitar o tamanho definido

st.markdown("---")
# ==============================================================================
# SEÇÃO 2: MODELO DE CLUSTERIZAÇÃO (KMEANS_K4) - ATUALIZADO COM OS NOVOS DADOS
# ==============================================================================
st.header("🤖 2. Clusterização de Perfis (KMeans_k4)")
st.markdown("Estratégia usada para agrupar clientes semelhantes e analisar como o comportamento de cancelamento varia entre os grupos.")

# Cards com as métricas atualizadas conforme a imagem
col_c1, col_c2, col_c3, col_c4 = st.columns(4)
with col_c1:
    st.metric(label="Silhouette Score", value="0.056")
with col_c2:
    st.metric(label="Taxa Média Geral de Churn", value="~12.2%")
with col_c3:
    # Maior risco (Cluster 3: 21.69%) e Menor risco (Cluster 0: 3.05%) -> Amplitude: 18.64%
    st.metric(label="Amplitude de Churn", value="18.64%", help="Diferença entre o maior risco (Cluster 3: 21.69%) e o menor risco (Cluster 0: 3.05%).")
with col_c4:
    # Soma total exata das volumetrias da imagem (19032 + 20407 + 20859 + 19702 = 80000)
    st.metric(label="Total de Clientes Mapeados", value="80.000")

col_g1, col_g2 = st.columns([1, 1.2])

with col_g1:
    st.markdown("### 📊 Volumetria Real por Grupo")
    # Dados de volumetria e representatividade recalculados com base na imagem
    df_metrics_km = pd.DataFrame({
        "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
        "Clientes (n)": ["19.702", "20.407", "20.859", "19.032"],
        "Representatividade": ["24.63%", "25.51%", "26.07%", "23.79%"]
    })
    st.dataframe(df_metrics_km, use_container_width=True, hide_index=True)

with col_g2:
    st.markdown("### 🎯 Taxa de Churn por Cluster (Dados de Treino)")
    
    # Dados extraídos fielmente da imagem fornecida
    dados_churn_cluster = pd.DataFrame({
        'Cluster': ['Cluster 0', 'Cluster 3', 'Cluster 1', 'Cluster 2'],
        'Taxa de Churn (%)': [21.69, 20.92, 3.25, 3.05],
        'Status': ['Acima da Média', 'Acima da Média', 'Abaixo da Média', 'Abaixo da Média']
    })
    
    grafico_clusters = alt.Chart(dados_churn_cluster).mark_bar().encode(
        x='Taxa de Churn (%):Q',
        y=alt.Y('Cluster:N', sort='-x'),
        color=alt.condition(
            alt.datum['Taxa de Churn (%)'] > 12.2,
            alt.value('#d9534f'),  # Vermelho para os que estão acima da média
            alt.value('#2b5c8f')   # Azul para os grupos controlados
        ),
        tooltip=['Cluster', 'Taxa de Churn (%)']
    ).properties(height=180)
    
    st.altair_chart(grafico_clusters, use_container_width=True)
