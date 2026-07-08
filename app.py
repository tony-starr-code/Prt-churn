import time
import re
import os
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import cloudpickle
import joblib
from dateutil import parser

import new_pipeline

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(
    page_title="PRT Seguradora - Analytics Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .stAlert {
        border-radius: 8px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    div[data-testid="stMetricValue"] {
        color: #1E3A8A;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# CABEÇALHO
col_logo, col_titulo = st.columns([1, 5])

with col_logo:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=130)
    else:
        st.markdown(
            "<h2 style='margin-top:0; color:#1E3A8A;'>🏢 PRT</h2>"
            "<p style='font-size:12px; margin-top:-15px;'><i>Seguradora</i></p>",
            unsafe_allow_html=True,
        )

with col_titulo:
    st.title("Painel Integrado de Inteligência de Clientes")
    st.caption("Hub Estratégico: Predição de Churn & Segmentação de Carteira • PRT Seguradora")

st.markdown("---")

# CARREGAMENTO DOS ARTEFATOS
@st.cache_resource
def carregar_artefatos():
    artefatos = {
        "modelo_churn": None,
        "colunas_churn": None,
        "cluster_dict": None,
        "pipeline_cluster": None,
        "avisos": [],
    }

    try:
        with open("model.pkl", "rb") as f:
            artefatos["modelo_churn"] = cloudpickle.load(f)
        artefatos["colunas_churn"] = joblib.load("colunas_modelo.pkl")
    except Exception as e:
        artefatos["avisos"].append(f"Erro ao carregar modelo de Churn: {e}")

    if os.path.exists("pipeline_clusterizacao_k4.pkl"):
        try:
            artefatos["cluster_dict"] = joblib.load("pipeline_clusterizacao_k4.pkl")
        except Exception as e:
            artefatos["avisos"].append(f"Erro ao carregar clusterização: {e}")
    else:
        artefatos["avisos"].append("Arquivo 'pipeline_clusterizacao_k4.pkl' não encontrado.")

    try:
        artefatos["pipeline_cluster"] = new_pipeline.load_fitted_pipeline()
        if artefatos["cluster_dict"]:
            for chave in ("pipeline", "preprocessor"):
                if chave in artefatos["cluster_dict"]:
                    artefatos["pipeline_cluster"] = artefatos["cluster_dict"][chave]
                    break
    except Exception as e:
        artefatos["avisos"].append(f"Erro ao carregar pipeline de new_pipeline: {e}")

    return artefatos

artefatos = carregar_artefatos()
artefatos_carregados = artefatos["modelo_churn"] is not None and artefatos["colunas_churn"] is not None

st.sidebar.header("⚙️ Status dos Modelos")
if artefatos["avisos"]:
    for aviso in artefatos["avisos"]:
        st.sidebar.info(aviso)
else:
    st.sidebar.success("✅ Modelos carregados com sucesso!")

# --- FUNÇÕES GLOBAIS DE TRATAMENTO ---
NULOS_DISFARÇADOS = ["#n/d", "-", "", "?", "n/a", "na", "null", "none"]

def limpar_nulos(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(NULOS_DISFARÇADOS, np.nan)
    return df

def parse_data(val):
    try:
        return parser.parse(str(val), dayfirst=True)
    except Exception:
        return pd.NaT

def normalizar_texto_mkt(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip()
    if v.upper() in ["", "-", "?", "#N/D", "NAN"]:
        return np.nan
    return v.title()

def normalizar_categoria_con(valor):
    if pd.isna(valor):
        return np.nan
    return str(valor).strip().lower().replace(".", "")

def normalizar_canal_con(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip()
    if v.lower() in ["", "-", "?", "#n/d", "nan"]:
        return np.nan
    return v.title()

def normalizar_metodo_con(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", v)

def limpar_valor_monetario_con(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().replace("R$", "").replace(" ", "").strip()
    tem_ponto = "." in v
    tem_virgula = "," in v
    if tem_ponto and tem_virgula:
        if v.rfind(",") > v.rfind("."):
            v = v.replace(".", "").replace(",", ".")
        else:
            v = v.replace(",", "")
    elif tem_virgula:
        v = v.replace(".", "").replace(",", ".")
    elif tem_ponto:
        partes = v.split(".")
        if len(partes[-1]) != 2:
            v = v.replace(".", "")
    try:
        return float(v)
    except ValueError:
        return np.nan

# ENTRADA DE DADOS
st.sidebar.markdown("---")
st.sidebar.header("📁 Entrada de Dados")
st.sidebar.markdown("Arraste as **4 bases brutas** (.csv) simultaneamente:")
arquivos_carregados = st.sidebar.file_uploader(
    "Bases brutas (Cadastro, Sinistros, Marketing, Contratos)",
    type=["csv"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if arquivos_carregados and len(arquivos_carregados) == 4 and artefatos_carregados:
    st.info("As 4 bases brutas foram detectadas! Iniciando a esteira de Engenharia de Dados...")

    tabelas = {}
    hoje = pd.Timestamp.today()

    for arquivo in arquivos_carregados:
        amostra = arquivo.read(2048).decode("utf-8")
        arquivo.seek(0)
        sep = ";" if ";" in amostra and amostra.count(";") > amostra.count(",") else ","

        df_temp = pd.read_csv(arquivo, sep=sep)
        colunas_temp = [c.lower() for c in df_temp.columns]

        if "data_nascimento" in colunas_temp or "escolaridade" in colunas_temp:
            tabelas["cadastro"] = df_temp
        elif "customer_key" in colunas_temp or "num_sinistros_historico" in colunas_temp:
            tabelas["sinistros"] = df_temp
        elif "score_engajamento_digital" in colunas_temp or "km_anual_estimado" in colunas_temp:
            tabelas["marketing"] = df_temp
        elif "cod_individuo" in colunas_temp or "valor_premio_anual" in colunas_temp or "tipo_cobertura" in colunas_temp:
            tabelas["contratos"] = df_temp

    if len(tabelas) < 4:
        st.error(
            "Erro no mapeamento. Certifique-se de fazer o upload de todas as 4 bases distintas "
            "(Cadastro, Sinistros, Marketing e Contratos)."
        )
    else:
        try:
            # DIAGNÓSTICO DE NULOS: Passo 1 - Captura do Estado Bruto
            diagnostico_nulos_antes = {}
            for nome_aba, df_bruto in tabelas.items():
                for col in df_bruto.columns:
                    col_normalizada = col.lower().replace("id_cliente", "id_cliente").replace("customer_key", "id_cliente").replace("cod_individuo", "id_cliente").replace("id", "id_cliente")
                    # Contabiliza nulos tradicionais + nulos disfarçados textuais
                    nulos_iniciais = df_bruto[col].isnull().sum()
                    if df_bruto[col].dtype == 'object':
                        nulos_iniciais += df_bruto[col].astype(str).str.strip().str.lower().isin(NULOS_DISFARÇADOS).sum()
                    diagnostico_nulos_antes[col_normalizada] = int(nulos_iniciais)

            # 1. TRATAMENTO — CADASTRO DOS CLIENTES
            df_cad = tabelas["cadastro"].copy()
            df_cad = limpar_nulos(df_cad)
            df_cad.rename(columns={"Id_cliente": "id_cliente", "ID_Cliente": "id_cliente", "Id_Cliente": "id_cliente"}, errors="ignore", inplace=True)

            df_cad["idade"] = pd.to_numeric(df_cad["idade"], errors="coerce").astype("Int64")
            df_cad["data_nascimento"] = df_cad["data_nascimento"].apply(parse_data)
            mask_data = df_cad["idade"].isnull() & df_cad["data_nascimento"].notnull()
            df_cad.loc[mask_data, "idade"] = df_cad.loc[mask_data, "data_nascimento"].apply(lambda x: int((hoje - x).days / 365.25))
            df_cad.drop(columns="data_nascimento", errors="ignore", inplace=True)
            df_cad["idade"] = df_cad["idade"].astype(float).where(df_cad["idade"].between(18, 100), other=np.nan)
            df_cad["idade"] = df_cad["idade"].astype("Int64")

            mapa_genero = {"masc": "M", "m": "M", "masculino": "M", "f": "F", "fem": "F", "feminino": "F"}
            df_cad["genero"] = df_cad["genero"].astype(str).str.strip().str.lower().map(mapa_genero)

            mapa_ec = {"c": "casado", "casado": "casado", "married": "casado", "casado(a)": "casado", "s": "solteiro", "solt": "solteiro", "single": "solteiro", "solteiro(a)": "solteiro"}
            df_cad["estado_civil"] = df_cad["estado_civil"].astype(str).str.strip().str.lower().map(mapa_ec)

            mapa_filhos = {"sim": 1, "true": 1, "s": 1, "1": 1, "nao": 0, "não": 0, "n": 0, "false": 0, "0": 0}
            df_cad["tem_filhos"] = df_cad["tem_filhos"].astype(str).str.strip().str.lower().map(mapa_filhos)
            df_cad["tem_filhos"] = pd.to_numeric(df_cad["tem_filhos"], errors="coerce").astype("Int64")

            df_cad["qtd_dependentes"] = pd.to_numeric(df_cad["qtd_dependentes"], errors="coerce").astype("Int64")
            df_cad["escolaridade"] = df_cad["escolaridade"].astype(str).str.strip().str.lower().str.capitalize()
            df_cad["escolaridade"] = df_cad["escolaridade"].replace("Nan", np.nan)

            for col in ["renda_anual", "valor_imovel"]:
                if col in df_cad.columns:
                    df_cad[col] = df_cad[col].astype(str).str.strip().str.replace(r"r\$", "", regex=True).str.replace(r"\s", "", regex=True).str.replace(r"\.(?=\d{3})", "", regex=True).str.replace(",", ".", regex=False)
                    df_cad[col] = pd.to_numeric(df_cad[col], errors="coerce")

            df_cad["possui_imovel"] = df_cad["possui_imovel"].astype(str).str.strip().str.lower().replace(["nan"], np.nan)
            df_cad["possui_imovel"] = pd.to_numeric(df_cad["possui_imovel"], errors="coerce").astype("Int64")
            df_cad["tempo_residencia_anos"] = pd.to_numeric(df_cad["tempo_residencia_anos"], errors="coerce")

            df_cad["id_cliente"] = df_cad["id_cliente"].astype(str).str.strip()
            df_cad.drop_duplicates(subset="id_cliente", keep="first", inplace=True)
            df_cad.loc[(df_cad["tem_filhos"] == 0) & (df_cad["qtd_dependentes"] > 0), "tem_filhos"] = 1

            # 2. TRATAMENTO — ATENDIMENTO / SINISTROS
            df_sin = tabelas["sinistros"].copy()
            df_sin = limpar_nulos(df_sin)
            df_sin.rename(columns={"customer_key": "id_cliente", "ID": "id_cliente"}, errors="ignore", inplace=True)
            df_sin["id_cliente"] = df_sin["id_cliente"].astype(float).astype(int).astype(str).str.strip()

            cols_num_sin = ["num_reclamacoes_12m", "num_sinistros_historico", "dias_ultimo_contato", "tempo_medio_resposta_dias", "num_ligacoes_suporte_12m", "num_acessos_app_mes", "satisfacao_nps"]
            for col in cols_num_sin:
                df_sin[col] = pd.to_numeric(df_sin[col], errors="coerce")

            df_sin["tempo_resolucao_ultimo_sinistro"] = pd.to_numeric(df_sin["tempo_resolucao_ultimo_sinistro"], errors="coerce")
            df_sin.loc[df_sin["tempo_resolucao_ultimo_sinistro"].isnull() & df_sin["data_ultimo_sinistro"].isnull(), "tempo_resolucao_ultimo_sinistro"] = 0

            df_sin["data_ultimo_sinistro"] = pd.to_datetime(df_sin["data_ultimo_sinistro"], errors="coerce", format="mixed", dayfirst=True)
            df_sin.loc[df_sin["data_ultimo_sinistro"] > hoje, "data_ultimo_sinistro"] = pd.NaT

            df_sin.loc[~df_sin["satisfacao_nps"].between(0, 10), "satisfacao_nps"] = np.nan
            df_sin["satisfacao_nps"] = df_sin["satisfacao_nps"].astype("Int64")

            for col in ["num_reclamacoes_12m", "num_sinistros_historico", "num_ligacoes_suporte_12m", "num_acessos_app_mes"]:
                df_sin[col] = df_sin[col].astype("Int64")

            df_sin["dias_desde_ultimo_sinistro"] = (hoje - df_sin["data_ultimo_sinistro"]).dt.days.astype("Int64")
            df_sin["teve_sinistro"] = (df_sin["num_sinistros_historico"] > 0).astype("Int64")
            df_sin.drop_duplicates(subset="id_cliente", keep="first", inplace=True)

            # 3. TRATAMENTO — ENGAJAMENTO MARKETING
            df_mkt = tabelas["marketing"].copy()
            df_mkt["ID"] = df_mkt["ID"].astype(str).str.strip()
            df_mkt.rename(columns={"ID": "id_cliente"}, inplace=True)

            numeric_cols_mkt = ["score_engajamento_digital", "indicou_clientes", "renovacoes_consecutivas", "indice_relacionamento", "ano_veiculo", "km_anual_estimado", "ultimo_login_portal_dias", "score_propensao_churn", "cluster_sugerido_crm"]
            for col in numeric_cols_mkt:
                df_mkt[col] = pd.to_numeric(df_mkt[col], errors="coerce")

            for col in ["tipo_veiculo", "segmento_marketing", "regiao_vendas"]:
                df_mkt[col] = df_mkt[col].apply(normalizar_texto_mkt)

            df_mkt["regiao_vendas"] = df_mkt["regiao_vendas"].replace({"Oeste": "Centro-Oeste", "Regiao Oeste": "Centro-Oeste", "Centro": "Centro-Oeste"})
            df_mkt["nunca_logou"] = df_mkt["ultimo_login_portal_dias"].isna().astype(int)

            df_mkt["id_cliente"] = df_mkt["id_cliente"].astype(str).str.strip()
            df_mkt.drop_duplicates(subset="id_cliente", keep="first", inplace=True)

            # 4. TRATAMENTO — CONTRATOS E APÓLICES
            df_con = tabelas["contratos"].copy()
            df_con["cod_individuo"] = df_con["cod_individuo"].astype(str).str.replace("IND-", "", regex=False).str.strip()
            df_con.rename(columns={"cod_individuo": "id_cliente"}, inplace=True)

            mapa_cobertura = {"premium": "Premium", "prem": "Premium", "básica": "Básica", "basica": "Básica", "basic": "Básica", "padrão": "Padrão", "padrao": "Padrão", "std": "Padrão", "plus": "Premium"}
            df_con["tipo_cobertura"] = df_con["tipo_cobertura"].apply(normalizar_categoria_con).map(mapa_cobertura)
            df_con["canal_aquisicao"] = df_con["canal_aquisicao"].apply(normalizar_canal_con)

            mapa_metodo = {"boleto": "Boleto", "bol": "Boleto", "boleto bancario": "Boleto", "cartao": "Cartao", "cartão": "Cartao", "cc": "Cartao", "cartao credito": "Cartao", "debito auto": "Debito", "debito automatico": "Debito", "debito_auto": "Debito", "debito": "Debito", "deb auto": "Debito", "pix": "Pix"}
            df_con["metodo_pagamento"] = df_con["metodo_pagamento"].apply(normalizar_metodo_con).map(mapa_metodo)

            mapa_pagamento = {"em dia": 1, "ok": 1, "sim": 1, "s": 1, "1": 1, "nao": 0, "não": 0, "n": 0, "0": 0, "atrasado": 0}
            df_con["pagamento_em_dia"] = df_con["pagamento_em_dia"].apply(lambda x: mapa_pagamento.get(str(x).strip().lower(), np.nan)).astype("float64")

            colunas_monetarias = ["valor_premio_anual", "valor_cobertura_total", "franquia_media"]
            for col in colunas_monetarias:
                df_con[col] = df_con[col].apply(limpar_valor_monetario_con)

            df_con["data_primeira_apolice"] = pd.to_datetime(df_con["data_primeira_apolice"], format="mixed", errors="coerce")

            for col in ["num_apolices_ativas", "tempo_cliente_dias", "num_produtos_contratados", "desconto_aplicado_pct"]:
                df_con[col] = pd.to_numeric(df_con[col], errors="coerce")
            df_con["desconto_aplicado_pct"] = df_con["desconto_aplicado_pct"] * 100

            for col, (min_val, max_val) in {"valor_premio_anual": (0, 500_000), "valor_cobertura_total": (0, 2_000_000), "tempo_cliente_dias": (0, 10_950)}.items():
                df_con.loc[(df_con[col] < min_val) | (df_con[col] > max_val), col] = np.nan

            DATA_REFERENCIA = pd.Timestamp("2026-06-01")
            df_con["tempo_cliente_dias"] = df_con["tempo_cliente_dias"].fillna((DATA_REFERENCIA - df_con["data_primeira_apolice"]).dt.days)
            df_con["data_primeira_apolice"] = df_con["data_primeira_apolice"].fillna(DATA_REFERENCIA - pd.to_timedelta(df_con["tempo_cliente_dias"], unit="D"))

            df_con["id_cliente"] = df_con["id_cliente"].astype(str).str.strip()
            df_con.drop_duplicates(subset="id_cliente", keep="first", inplace=True)

            # 5. CONSOLIDAÇÃO DA BASE ÚNICA (MERGES)
            for df in [df_cad, df_sin, df_mkt, df_con]:
                if "id_cliente" in df.columns:
                    df["id_cliente"] = df["id_cliente"].astype(str).str.strip()

            df_final = (
                df_con
                .merge(df_mkt, on="id_cliente", how="outer")
                .merge(df_cad, on="id_cliente", how="outer")
                .merge(df_sin, on="id_cliente", how="outer")
                .copy()
            )

            st.success(f"🎉 Processamento completo! Base unificada com {df_final.shape[0]} clientes e {df_final.shape[1]} variáveis.")

            # DIAGNÓSTICO DE NULOS: Passo 2 - Consolidação da Tabela Comparativa
            total_linhas = len(df_final)
            dados_diagnostico = []
            
            for col in df_final.columns:
                nulos_antes = diagnostico_nulos_antes.get(col, 0)
                nulos_depois = int(df_final[col].isnull().sum())
                
                dados_diagnostico.append({
                    "Variável": col,
                    "Nulos (Antes)": nulos_antes,
                    "% Nulos (Antes)": round((nulos_antes / total_linhas) * 100, 1) if total_linhas > 0 else 0,
                    "Nulos (Depois)": nulos_depois,
                    "% Nulos (Depois)": round((nulos_depois / total_linhas) * 100, 1) if total_linhas > 0 else 0,
                })
            
            df_diagnostico = pd.DataFrame(dados_diagnostico)

            # EXPANDER DO DIAGNÓSTICO DE QUALIDADE DE DADOS
            with st.expanders("📊 Diagnóstico Técnico: Qualidade de Dados (Antes vs Depois)"):
                st.markdown("A tabela abaixo detalha o volume de valores ausentes ou corrompidos tratados pela nossa esteira automatizada.")
                
                col_diag_tab, col_diag_graf = st.columns([1.2, 1])
                
                with col_diag_tab:
                    st.dataframe(
                        df_diagnostico.sort_values(by="Nulos (Antes)", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                        height=350
                    )
                
                with col_diag_graf:
                    df_melt = df_diagnostico.melt(
                        id_vars=["Variável"], 
                        value_vars=["% Nulos (Antes)", "% Nulos (Depois)"],
                        var_name="Momento", 
                        value_name="Percentual"
                    )
                    fig_nulos = px.bar(
                        df_melt,
                        x="Percentual",
                        y="Variável",
                        color="Momento",
                        barmode="group",
                        title="Redução/Evolução Qualitativa de Nulos (%)",
                        orientation="h",
                        color_discrete_map={"% Nulos (Antes)": "#EF4444", "% Nulos (Depois)": "#3B82F6"}
                    )
                    fig_nulos.update_layout(yaxis={'categoryorder':'total ascending'}, height=350, margin=dict(t=30, b=10, l=10, r=10))
                    st.plotly_chart(fig_nulos, use_container_width=True)

            # PREDIÇÃO DE CHURN
            modelo = artefatos["modelo_churn"]
            colunas_treino = artefatos["colunas_churn"]

            X_scoring = df_final.copy()
            for col in colunas_treino:
                if col not in X_scoring.columns:
                    X_scoring[col] = 0
            X_scoring = X_scoring[colunas_treino].fillna(0)

            t0 = time.time()
            probabilidades = modelo.predict_proba(X_scoring)[:, 1]
            tempo_predicao = time.time() - t0

            df_resultado = df_final[["id_cliente"]].copy()
            df_resultado["Risco Churn (%)"] = (probabilidades * 100 if probabilidades.max() <= 1.0 else probabilidades).round(2)
            df_resultado["Status de Risco"] = np.where(
                df_resultado["Risco Churn (%)"] > 70.0,
                "🚨 Alto Risco",
                np.where(df_resultado["Risco Churn (%)"] > 30.0, "⚠️ Risco Moderado", "✅ Estável"),
            )

            # CLUSTERIZAÇÃO VIA NEW_PIPELINE
            pipeline_cluster = artefatos["pipeline_cluster"]
            if pipeline_cluster is not None:
                try:
                    dados_transformados = pipeline_cluster.transform(df_final)
                    
                    if "pipeline_cluster" in artefatos and hasattr(artefatos["pipeline_cluster"], "predict"):
                        clusters = artefatos["pipeline_cluster"].predict(df_final)
                    elif artefatos["cluster_dict"] and "model" in artefatos["cluster_dict"]:
                        clusters = artefatos["cluster_dict"]["model"].predict(dados_transformados)
                    else:
                        clusters = None
                    
                    if clusters is not None:
                        df_resultado["Grupo Cluster"] = [f"Grupo {c + 1}" for c in clusters]
                except Exception as e:
                    st.sidebar.error(f"Erro ao computar os clusters em tempo de execução: {e}")

            # ============================================================
            # DASHBOARD — CHURN + CLUSTERIZAÇÃO
            # ============================================================
            col_churn, col_cluster = st.columns([1, 1.1])

            with col_churn:
                st.subheader("🚨 Diagnóstico de Risco de Churn")
                st.markdown("Identificação proativa de clientes com propensão de cancelamento de apólices.")
                st.caption(f"Predições concluídas em {tempo_predicao:.2f} segundos.")

                m1, m2, m3 = st.columns(3)
                m1.metric("Alto Risco", len(df_resultado[df_resultado["Status de Risco"] == "🚨 Alto Risco"]))
                m2.metric("Risco Moderado", len(df_resultado[df_resultado["Status de Risco"] == "⚠️ Risco Moderado"]))
                m3.metric("Estável", len(df_resultado[df_resultado["Status de Risco"] == "✅ Estável"]))

                st.markdown("#### Lista de Clientes Prioritários")
                st.dataframe(
                    df_resultado.sort_values(by="Risco Churn (%)", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=380,
                )

                csv_saida = df_resultado.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Baixar Painel Consolidado (.CSV)",
                    data=csv_saida,
                    file_name="relatorio_final_prt.csv",
                    mime="text/csv",
                )

            with col_cluster:
                st.subheader("🎯 Segmentação Estratégica da Carteira")
                if "Grupo Cluster" in df_resultado.columns:
                    st.markdown("Distribuição volumétrica atualizada de clientes por agrupamento comportamental.")
                    
                    fig_cluster = px.histogram(
                        df_resultado, 
                        x="Grupo Cluster", 
                        color="Grupo Cluster",
                        title="Distribuição de Clientes por Cluster",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_cluster.update_layout(height=435)
                    st.plotly_chart(fig_cluster, use_container_width=True)
                else:
                    st.warning("⚠️ Dados de segmentação indisponíveis devido a problemas no carregamento do pipeline.")

        except Exception as e:
            st.error(f"Ocorreu um erro crítico durante o processamento dos dados: {e}")