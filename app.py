"""
App Streamlit - Etapa 1: Upload e Tratamento dos 4 CSVs brutos
------------------------------------------------------------
Próximas etapas (a implementar depois):
  2) Clusterização (modelo .pkl)
  3) Predição de churn usando o cluster como feature (modelo .pkl)

Como rodar:
  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Pipeline Churn", layout="wide")
st.title("Pipeline: Tratamento de Dados → Clusterização → Churn")

# ----------------------------------------------------------------
# Estado da sessão (guarda os dataframes entre interações)
# ----------------------------------------------------------------
if "df_tratado" not in st.session_state:
    st.session_state.df_tratado = None

NOMES_BASES = ["Base 1", "Base 2", "Base 3", "Base 4"]

# ==================================================================
# 1. UPLOAD DOS 4 ARQUIVOS
# ==================================================================
st.header("1. Upload dos arquivos")

col1, col2, col3, col4 = st.columns(4)
uploaders = {}
colunas_ui = [col1, col2, col3, col4]

for nome, col in zip(NOMES_BASES, colunas_ui):
    with col:
        uploaders[nome] = st.file_uploader(nome, type="csv", key=f"upload_{nome}")

arquivos_prontos = all(uploaders[n] is not None for n in NOMES_BASES)

if not arquivos_prontos:
    st.info("Envie os 4 arquivos CSV para continuar.")
    st.stop()

# ----------------------------------------------------------------
# Leitura com detecção simples de separador/encoding
# ----------------------------------------------------------------
def ler_csv_robusto(arquivo):
    """Tenta ler o CSV com combinações comuns de separador e encoding."""
    tentativas = [
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]
    for params in tentativas:
        try:
            arquivo.seek(0)
            df = pd.read_csv(arquivo, **params)
            if df.shape[1] > 1:  # se leu só 1 coluna, provavelmente sep errado
                return df
        except Exception:
            continue
    arquivo.seek(0)
    return pd.read_csv(arquivo)  # última tentativa, deixa o erro estourar se falhar


dfs_brutos = {nome: ler_csv_robusto(uploaders[nome]) for nome in NOMES_BASES}

with st.expander("Pré-visualização dos dados brutos"):
    for nome in NOMES_BASES:
        st.subheader(nome)
        st.write(f"Shape: {dfs_brutos[nome].shape}")
        st.dataframe(dfs_brutos[nome].head())

# ==================================================================
# 2. TRATAMENTO INDIVIDUAL DE CADA BASE
# ==================================================================
st.header("2. Tratamento de cada base")

dfs_tratados = {}

for nome in NOMES_BASES:
    st.subheader(nome)
    df = dfs_brutos[nome].copy()

    c1, c2, c3 = st.columns(3)

    # --- 2.1 Padronizar nomes de colunas ---
    with c1:
        padronizar_colunas = st.checkbox(
            "Padronizar nomes de colunas (minúsculo, sem espaço)",
            value=True, key=f"pad_{nome}",
        )
    if padronizar_colunas:
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("utf-8")
            .str.replace(" ", "_", regex=False)
        )

    # --- 2.2 Remover duplicatas ---
    with c2:
        remover_dup = st.checkbox("Remover linhas duplicadas", value=True, key=f"dup_{nome}")
    if remover_dup:
        antes = len(df)
        df = df.drop_duplicates()
        st.caption(f"Duplicatas removidas: {antes - len(df)}")

    # --- 2.3 Estratégia para valores faltantes ---
    with c3:
        estrategia_na = st.selectbox(
            "Valores faltantes",
            ["Não tratar", "Remover linhas com NA", "Preencher numéricas com mediana",
             "Preencher categóricas com 'desconhecido'", "Preencher numéricas e categóricas"],
            key=f"na_{nome}",
        )

    if estrategia_na == "Remover linhas com NA":
        df = df.dropna()
    elif estrategia_na == "Preencher numéricas com mediana":
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    elif estrategia_na == "Preencher categóricas com 'desconhecido'":
        cat_cols = df.select_dtypes(include="object").columns
        df[cat_cols] = df[cat_cols].fillna("desconhecido")
    elif estrategia_na == "Preencher numéricas e categóricas":
        num_cols = df.select_dtypes(include="number").columns
        cat_cols = df.select_dtypes(include="object").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].fillna("desconhecido")

    # --- 2.4 Conversão manual de tipos (opcional) ---
    with st.expander(f"Ajustar tipos de coluna — {nome}"):
        colunas_para_numero = st.multiselect(
            "Forçar conversão para número", df.columns.tolist(), key=f"tonum_{nome}"
        )
        for col in colunas_para_numero:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        colunas_para_data = st.multiselect(
            "Forçar conversão para data", df.columns.tolist(), key=f"todate_{nome}"
        )
        for col in colunas_para_data:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    st.dataframe(df.head())
    st.divider()

    dfs_tratados[nome] = df

# ==================================================================
# 3. MERGE DAS 4 BASES
# ==================================================================
st.header("3. Unificação das bases (merge)")

st.markdown("Selecione a coluna-chave (ex: `id_cliente`) em cada base para o merge.")

chaves = {}
cols_merge = st.columns(4)
for nome, col in zip(NOMES_BASES, cols_merge):
    with col:
        chaves[nome] = st.selectbox(
            f"Chave — {nome}", dfs_tratados[nome].columns.tolist(), key=f"chave_{nome}"
        )

tipo_merge = st.radio(
    "Tipo de merge", ["inner (só quem existe em todas)", "outer (mantém todos, gera NA)",
                       "left (mantém tudo da Base 1)"],
    horizontal=True,
)
tipo_merge_map = {"inner (só quem existe em todas)": "inner",
                  "outer (mantém todos, gera NA)": "outer",
                  "left (mantém tudo da Base 1)": "left"}

if st.button("Executar tratamento e unificar bases", type="primary"):
    try:
        df_final = dfs_tratados["Base 1"].rename(columns={chaves["Base 1"]: "chave_merge"})
        for nome in NOMES_BASES[1:]:
            df_aux = dfs_tratados[nome].rename(columns={chaves[nome]: "chave_merge"})
            df_final = df_final.merge(
                df_aux, on="chave_merge", how=tipo_merge_map[tipo_merge], suffixes=("", f"_{nome}")
            )

        st.session_state.df_tratado = df_final
        st.success(f"Bases unificadas com sucesso! Shape final: {df_final.shape}")

    except Exception as e:
        st.error(f"Erro ao unificar as bases: {e}")

# ==================================================================
# 4. RESULTADO E DOWNLOAD
# ==================================================================
if st.session_state.df_tratado is not None:
    st.header("4. Base tratada e unificada")
    df_final = st.session_state.df_tratado

    st.dataframe(df_final.head(50))

    st.write("Resumo de valores faltantes por coluna:")
    st.dataframe(df_final.isna().sum().rename("qtd_nulos").to_frame())

    csv_saida = df_final.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Baixar base tratada (CSV)", csv_saida, "base_tratada.csv", "text/csv"
    )

    st.info(
        "Próximo passo: essa base (`st.session_state.df_tratado`) será usada na etapa "
        "de clusterização assim que os arquivos .pkl estiverem prontos."
    )