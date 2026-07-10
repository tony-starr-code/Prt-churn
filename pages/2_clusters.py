import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Configuração da página
st.set_page_config(page_title="PRT Seguradora - Perfis de Clientes", layout="wide")

st.title("👥 Análise Comportamental dos Clusters (KMeans_k4)")
st.markdown("Interpretação estratégica dos 4 ecossistemas de clientes gerados pela nossa inteligência artificial baseada em dados de Contratos, Sinistros e Marketing.")

# --- COMPONENTE VISUAL: ABAS PARA NAVEGAÇÃO ---
aba0, aba1, aba2, aba3 = st.tabs([
    "🚨 Cluster 0: Detratores Críticos", 
    "✅ Cluster 1: Clientes Modelo / VIP", 
    "⚠️ Cluster 2: Inativos Digitais com Atrito", 
    "🔄 Cluster 3: Recém-Chegados Engajados"
])

# ==============================================================================
# CLUSTER 0
# ==============================================================================
with aba0:
    st.error("### Perfil: Detratores com Alto Atrito e Risco Imediato")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        **Análise de Comportamento:**
        Este grupo representa o cenário mais alarmante para a seguradora. São clientes com altíssimo volume de reclamações, ligações para o suporte e sinistros registados recentemente. Apesar de usarem muito a aplicação móvel, o nível de satisfação (**NPS**) é extremamente baixo. Eles têm muitas apólices ativas e descontos altos aplicados, mas o risco de churn é crítico devido à insatisfação acumulada.

        * **Pontos Críticos:** * Altíssimo número de reclamações (`num_reclamacoes_12m` e `num_ligacoes_suporte_12m`).
            * NPS muito baixo (`satisfacao_nps`).
            * Tempo de resposta lento no último sinistro.
        """)
    with col2:
        st.metric(label="Risco Estimado de Churn", value="Alta Vulnerabilidade", delta="Ação Urgente", delta_color="inverse")
        st.info("💡 **Ação Recomendada:** Direcionar para uma equipa de retenção dedicada (ouvidoria) para resolver os problemas de suporte pendentes antes da data de renovação.")

# ==============================================================================
# CLUSTER 1
# ==============================================================================
with aba1:
    st.success("### Perfil: Clientes VIP / Promotores de Longo Prazo")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        **Análise de Comportamento:**
        O cliente ideal da PRT Seguradora. São indivíduos com alto tempo de casa, maior estabilidade residencial, altos valores de rendimento anual e que mantêm o pagamento rigorosamente em dia. Eles indicam novos clientes de forma ativa, possuem alto índice de relacionamento com a marca e um excelente nível de satisfação (NPS).

        * **Pontos Fortes:**
            * Alta renda anual, tempo de residência elevado e idade madura.
            * Maior número de produtos contratados e renovações consecutivas.
            * Pagamento 100% em dia.
        """)
    with col2:
        st.metric(label="Risco Estimado de Churn", value="Praticamente Nulo (Estável)", delta="Zona Segura")
        st.info("💡 **Ação Recomendada:** Campanhas de *Cross-selling* (oferecer novos produtos premium) e programas de fidelidade/recompensas por indicações (*Member Get Member*).")

# ==============================================================================
# CLUSTER 2
# ==============================================================================
with aba2:
    st.warning("### Perfil: Tradicionais Inativos Digitais e Alto Prémio")
    col1, col2 = columns = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        **Análise de Comportamento:**
        Este grupo possui o maior valor de prémio anual cobrado pela seguradora, tornando-o financeiramente muito valioso, mas o relacionamento é digitalmente inexistente. Apresentam a maior taxa de clientes que **nunca fizeram login no portal** (`nunca_logou`). Além disso, sofrem com o maior tempo médio de resposta da nossa parte em dias, gerando um risco moderado silencioso.

        * **Características:**
            * Alto prémio anual (`valor_premio_anual`).
            * Altíssima inatividade no portal/app.
            * Tempo médio de resposta a sinistros muito elevado.
        """)
    with col2:
        st.metric(label="Risco Estimado de Churn", value="Moderado / Silencioso", delta="Atenção Operacional", delta_color="off")
        st.info("💡 **Ação Recomendada:** Forçar o contacto humano direto através de corretores ou gestores de conta para acelerar a resolução de processos pendentes e reduzir o tempo de resposta operacional.")

# ==============================================================================
# CLUSTER 3
# ==============================================================================
with aba3:
    st.info("### Perfil: Clientes Novos, Digitais e em Expansão")
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("""
        **Análise de Comportamento:**
        Clientes mais jovens e recém-chegados (baixo tempo como cliente). Apresentam um excelente comportamento digital, utilizam muito o portal, têm boa pontuação de relacionamento e dão muitas indicações. Não possuem quase nenhum registo de sinistros ou atritos de suporte. Contudo, recebem poucos descontos e têm apólices mais baratas por enquanto.

        * **Características:**
            * Idade mais baixa e baixo tempo de casa.
            * Excelente engajamento digital (`score_engajamento_digital`).
            * Quase zero sinistros ou reclamações históricas.
        """)
    with col2:
        st.metric(label="Risco Estimado de Churn", value="Baixo / Monitorizado", delta="Potencial de Crescimento")
        st.info("💡 **Ação Recomendada:** Régua de comunicação automatizada com foco em educação de seguros e ofertas progressivas de desconto à medida que o tempo de contrato avança.")

# --- GRÁFICO COMPARATIVO COMPLEMENTAR ---
st.markdown("---")
st.subheader("📊 Resumo Executivo para a Direção")

dados_resumo = pd.DataFrame({
    'Cluster': ['0: Detratores Críticos', '1: Clientes VIP', '2: Inativos com Atrito', '3: Novos Engajados'],
    'Volume de Clientes (%)': [24.06, 25.96, 25.00, 24.98],
    'Prioridade de Atuação': ['Imediata (Urgente)', 'Baixa (Manter)', 'Média (Operacional)', 'Baixa (Nutrir)']
})

st.table(dados_resumo)