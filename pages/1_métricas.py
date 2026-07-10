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
        'Cluster': ['Cluster 3', 'Cluster 1', 'Cluster 2', 'Cluster 0'],
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

st.caption("ℹ️ Métricas extraídas via MLflow Tracking do artefato KMeans_k4.")