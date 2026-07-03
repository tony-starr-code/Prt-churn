import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuração da página do Streamlit
st.set_page_config(page_title="Predição de Churn - PRT Seguradora", layout="wide")

# ── 1. CARREGAR O MODELO E AS COLUNAS SALVAS ─────────────────────────────────
@st.cache_resource
def carregar_artefatos():
    modelo = joblib.load('modelo_xgb_churn.pkl')
    colunas = joblib.load('colunas_modelo.pkl')
    return modelo, colunas

try:
    modelo, colunas_treino = carregar_artefatos()
    status_modelo = True
except FileNotFoundError:
    status_modelo = False

# ── 2. INTERFACE VISUAL ──────────────────────────────────────────────────────
st.title("📊 Painel de Predição de Churn de Clientes")
st.subheader("PRT Seguradora — Análise de Risco em Tempo Real")

if not status_modelo:
    st.error("⚠️ Arquivos 'modelo_xgb_churn.pkl' ou 'colunas_modelo.pkl' não foram encontrados. Certifique-se de exportá-los no notebook primeiro.")
    st.stop()

# Barra lateral informativa
st.sidebar.header("Instruções")
st.sidebar.markdown("""
1. Carregue uma base de dados de clientes no formato **.csv**.
2. A base deve conter as colunas cadastrais e de sinistros unificadas (ou o arquivo bruto unificado).
3. O modelo processará as variáveis categóricas e numéricas automaticamente.
4. O resultado mostrará a probabilidade de Churn de cada cliente.
""")

# Área de Upload do Arquivo
arquivo_carregado = st.file_uploader("Selecione o arquivo CSV dos clientes para análise:", type=["csv"])

if arquivo_carregado is not None:
    # Lendo os dados que o usuário enviou
    df_usuario = pd.read_csv(arquivo_carregado)
    
    st.write(f"### 📋 Dados Carregados ({df_usuario.shape[0]} clientes encontrados)")
    st.dataframe(df_usuario.head(10)) # Mostra as 10 primeiras linhas para validação visual
    
    # Botão para iniciar o processamento
    if st.button("🚀 Calcular Probabilidade de Churn"):
        with st.spinner("Processando dados e aplicando inteligência artificial..."):
            
            # Guardamos o ID do cliente para o relatório final
            if 'id_cliente' in df_usuario.columns:
                ids = df_usuario['id_cliente'].astype(str)
            else:
                ids = [f"Cliente_{i}" for i in range(len(df_usuario))]
            
            # ── 3. PRÉ-PROCESSAMENTO IDÊNTICO AO TREINO ──────────────────────────
            df_proc = df_usuario.copy()
            
            # Remove colunas que não vão para o modelo matemático se elas existirem
            colunas_remover = ['id_cliente', 'churned', 'score_propensao_churn', 'cluster_sugerido_crm']
            df_proc = df_proc.drop(columns=[c for c in colunas_remover if c in df_proc.columns], errors='ignore')
            
            # Tratamento de nulos estruturais pós-merge (igual ao seu script de modelagem)
            if 'teve_sinistro' in df_proc.columns:
                df_proc['teve_sinistro'] = df_proc['teve_sinistro'].fillna(0)
            if 'nunca_logou' in df_proc.columns:
                df_proc['nunca_logou'] = df_proc['nunca_logou'].fillna(1)
            
            # Dummificação das variáveis textuais
            df_encoded = pd.get_dummies(df_proc, drop_first=True)
            
            # ── 4. ALINHAMENTO DE COLUNAS ────────────────────────────────────────
            # Garante que o DataFrame final tenha exatamente as mesmas colunas do treino, na mesma ordem.
            # Se faltar alguma coluna dummificada (ex: nenhum cliente 'Gênero_Outro' na base nova), ele cria com 0.
            # Se houver colunas a mais, ele descarta.
            df_final = pd.DataFrame(0, index=np.arange(len(df_encoded)), columns=colunas_treino)
            for col in colunas_treino:
                if col in df_encoded.columns:
                    df_final[col] = df_encoded[col].values
            
            # ── 5. PREDIÇÃO ──────────────────────────────────────────────────────
            # Calcula as probabilidades de pertencer à classe 1 (Churn)
            probabilidades = modelo.predict_proba(df_final)[:, 1]
            
            # Definimos o risco baseado em faixas de probabilidade
            resultado_final = pd.DataFrame({
                'ID do Cliente': ids,
                'Probabilidade de Churn': probabilidades,
                'Nível de Risco': np.where(probabilities > 0.7, '🚨 Alto Risco', 
                                   np.where(probabilities > 0.4, '⚠️ Médio Risco', '✅ Baixo Risco'))
            })
            
            # Formata a probabilidade para exibição em porcentagem
            resultado_final['Probabilidade de Churn'] = (resultado_final['Probabilidade de Churn'] * 100).round(2).astype(str) + '%'
            
            # Ordena pelos clientes com maior risco primeiro
            resultado_final = resultado_final.sort_values(by='Probabilidade de Churn', ascending=False)
            
            # ── 6. EXIBIÇÃO DOS RESULTADOS ───────────────────────────────────────
            st.success("✨ Análise concluída com sucesso!")
            
            # Métricas resumidas na tela
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Analisado", f"{len(resultado_final)} clientes")
            col2.metric("🚨 Alto Risco (>70%)", f"{len(resultado_final[resultado_final['Nível de Risco'] == '🚨 Alto Risco'])} clientes")
            col3.metric("✅ Baixo Risco (<40%)", f"{len(resultado_final[resultado_final['Nível de Risco'] == '✅ Baixo Risco'])} clientes")
            
            st.write("### 📈 Relatório de Risco de Evasão")
            st.dataframe(resultado_final, use_container_width=True)
            
            # Permitir que o usuário baixe o resultado em CSV
            csv = resultado_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Baixar Relatório de Risco (CSV)",
                data=csv,
                file_name="predicoes_churn_seguradora.csv",
                mime="text/csv",
            )