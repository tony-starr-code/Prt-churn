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
from churn_pipe import EngenhariaDeFeatures, RemovedorDeColunas, ImputadorDistribuicao, CriadorFaixaEtaria, RemovedorDeColunas

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
        "pipeline_churn": None,
        "colunas_churn": None,
        "cluster_dict": None,
        "pipeline_cluster": None,
        "avisos": [],
    }

    try:
        dicionario_modelo = joblib.load("model.pkl")
        artefatos["modelo_churn"] = dicionario_modelo["model"]
        artefatos["pipeline_churn"] = dicionario_modelo["pipeline"]
        artefatos["colunas_churn"] = dicionario_modelo["columns"]

    except Exception as e:
        artefatos["avisos"].append(f"Erro ao carregar dicionário/modelo de Churn: {e}")

    if os.path.exists("pipeline_clusterizacao_k4.pkl"):
        try:
            artefatos["cluster_dict"] = joblib.load("pipeline_clusterizacao_k4.pkl")
        except Exception as e:
            artefatos["avisos"].append(f"Erro ao carregar clusterização: {e}")
    else:
        artefatos["avisos"].append("Arquivo 'pipeline_clusterizacao_k4.pkl' não encontrado.")

    try:
        artefatos["pipeline_cluster"] = new_pipeline.load_fitted_pipeline()
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

# --- FUNÇÕES GLOBAIS DE TRATAMENTO (inalteradas) ---
NULOS_DISFARÇADOS = ["#n/d", "-", "", "?", "n/a", "na", "null", "none", "-", "Nan"]

def limpar_nulos(df):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace(NULOS_DISFARÇADOS, np.nan)
    return df

def imputar_amostra(serie):
    nulos = serie.isnull()
    serie = serie.copy()
    if nulos.sum() > 0:
        serie.loc[nulos] = serie.dropna().sample(nulos.sum(), replace=True).values
    return serie

def imputar_categorica(serie):
    nulos = serie.isnull()
    serie = serie.copy()
    if nulos.sum() > 0:
        frequencias = serie.value_counts(normalize=True)
        serie.loc[nulos] = np.random.choice(frequencias.index, size=nulos.sum(), p=frequencias.values)
    return serie

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

def moda_segura(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan

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
            LIMITES_OUTLIERS = {
                # Cadastro
                "qtd_dependentes": (0, 15),
                "renda_anual": (0, 2_000_000),
                "valor_imovel": (0, 15_000_000),
                "tempo_residencia_anos": (0, 80),
                
                # Sinistros
                "num_reclamacoes_12m": (0, 50),
                "num_sinistros_historico": (0, 30),
                "dias_ultimo_contato": (0, 1825), # até ~5 anos
                "tempo_medio_resposta_dias": (0, 180),
                "num_ligacoes_suporte_12m": (0, 365),
                "num_acessos_app_mes": (0, 500),
                "tempo_resolucao_ultimo_sinistro": (0, 730),
                
                # Marketing
                "score_engajamento_digital": (0, 100),
                "indicou_clientes": (0, 100),
                "renovacoes_consecutivas": (0, 50),
                "indice_relacionamento": (0, 100),
                "ano_veiculo": (1950, 2027), # Ajustável via hoje.year + 1
                "km_anual_estimado": (100.0, 300_000),
                "ultimo_login_portal_dias": (0, 1825),
                "score_propensao_churn": (0.0, 1.0),
                
                # Contratos (os já fixados em np.nan foram mantidos na lógica de negócio mais abaixo)
                "franquia_media": (300.0, 200_000),
                "num_apolices_ativas": (0, 50),
                "num_produtos_contratados": (0, 50),
                "desconto_aplicado_pct": (0, 100),
                "valor_premio_anual": (300.0, 25_000.0)
            }

            # ============================================================
            # 1. TRATAMENTO — CADASTRO DOS CLIENTES
            # ============================================================
            df_cad = tabelas["cadastro"].copy()
            df_cad = limpar_nulos(df_cad)
            df_cad.rename(columns={"Id_cliente": "id_cliente", "ID_Cliente": "id_cliente", "Id_Cliente": "id_cliente"}, errors="ignore", inplace=True)

            df_cad["idade"] = pd.to_numeric(df_cad["idade"], errors="coerce").astype("Int64")
            df_cad["data_nascimento"] = df_cad["data_nascimento"].apply(parse_data)
            mask_data = df_cad["idade"].isnull() & df_cad["data_nascimento"].notnull()
            df_cad.loc[mask_data, "idade"] = df_cad.loc[mask_data, "data_nascimento"].apply(lambda x: int((hoje - x).days / 365.25))
            df_cad.drop(columns="data_nascimento", errors="ignore", inplace=True)
            # idade fora da faixa plausível -> NaN (regra de negócio fixa, não estatística do lote: mantido)
            df_cad["idade"] = df_cad["idade"].astype(float).where(df_cad["idade"].between(18, 100), other=np.nan)
            df_cad["idade"] = df_cad["idade"].astype("Int64")  # MUDANÇA: sem imputar_amostra aqui — NaN segue pro pipeline treinado

            mapa_genero = {"masc": "M", "m": "M", "masculino": "M", "f": "F", "fem": "F", "feminino": "F"}
            df_cad["genero"] = df_cad["genero"].astype(str).str.strip().str.lower().map(mapa_genero)

            mapa_ec = {"c": "casado", "casado": "casado", "married": "casado", "casado(a)": "casado", "s": "solteiro", "solt": "solteiro", "single": "solteiro", "solteiro(a)": "solteiro"}
            df_cad["estado_civil"] = df_cad["estado_civil"].astype(str).str.strip().str.lower().map(mapa_ec)

            mapa_filhos = {"sim": 1, "true": 1, "s": 1, "1": 1, "nao": 0, "não": 0, "n": 0, "false": 0, "0": 0}
            df_cad["tem_filhos"] = df_cad["tem_filhos"].astype(str).str.strip().str.lower().map(mapa_filhos)
            df_cad["tem_filhos"] = pd.to_numeric(df_cad["tem_filhos"], errors="coerce").astype("Int64")  # MUDANÇA: Int64 nullable (antes virava int só depois do fillna via imputar_categorica)

            df_cad["qtd_dependentes"] = pd.to_numeric(df_cad["qtd_dependentes"], errors="coerce").astype("Int64")
            df_cad["qtd_dependentes"] = df_cad["qtd_dependentes"].clip(lower=LIMITES_OUTLIERS["qtd_dependentes"][0], upper=LIMITES_OUTLIERS["qtd_dependentes"][1]).astype("Int64")
            
            df_cad["escolaridade"] = df_cad["escolaridade"].astype(str).str.strip().str.lower().str.capitalize()
            df_cad["escolaridade"] = df_cad["escolaridade"].replace("Nan", np.nan)

            for col in ["renda_anual", "valor_imovel"]:
                if col in df_cad.columns:
                    df_cad[col] = df_cad[col].astype(str).str.strip().str.replace(r"r\$", "", regex=True).str.replace(r"\s", "", regex=True).str.replace(r"\.(?=\d{3})", "", regex=True).str.replace(",", ".", regex=False)
                    df_cad[col] = pd.to_numeric(df_cad[col], errors="coerce")
                    df_cad[col] = df_cad[col].clip(lower=LIMITES_OUTLIERS[col][0], upper=LIMITES_OUTLIERS[col][1])

            df_cad["possui_imovel"] = df_cad["possui_imovel"].astype(str).str.strip().str.lower().replace(NULOS_DISFARÇADOS + ["nan"], np.nan)
            df_cad["possui_imovel"] = pd.to_numeric(df_cad["possui_imovel"], errors="coerce").astype("Int64")

            df_cad["tempo_residencia_anos"] = pd.to_numeric(df_cad["tempo_residencia_anos"], errors="coerce")
            df_cad["tempo_residencia_anos"] = df_cad["tempo_residencia_anos"].clip(lower=LIMITES_OUTLIERS["tempo_residencia_anos"][0], upper=LIMITES_OUTLIERS["tempo_residencia_anos"][1])
            df_cad["id_cliente"] = df_cad["id_cliente"].astype(str).str.strip()
            df_cad.drop_duplicates(subset="id_cliente", keep="first", inplace=True)  

            df_cad.loc[(df_cad["tem_filhos"] == 0) & (df_cad["qtd_dependentes"] > 0), "tem_filhos"] = 1


            # ============================================================
            # 2. TRATAMENTO — ATENDIMENTO / SINISTROS
            # ============================================================
            df_sin = tabelas["sinistros"].copy()
            df_sin = limpar_nulos(df_sin)
            df_sin.rename(columns={"customer_key": "id_cliente", "ID": "id_cliente"}, errors="ignore", inplace=True)
            df_sin["id_cliente"] = df_sin["id_cliente"].astype(float).astype(int).astype(str).str.strip()

            cols_num_sin = ["num_reclamacoes_12m", "num_sinistros_historico", "dias_ultimo_contato", "tempo_medio_resposta_dias", "num_ligacoes_suporte_12m", "num_acessos_app_mes", "satisfacao_nps"]
            for col in cols_num_sin:
                df_sin[col] = pd.to_numeric(df_sin[col], errors="coerce")
                if col in LIMITES_OUTLIERS:
                    df_sin[col] = df_sin[col].clip(lower=LIMITES_OUTLIERS[col][0], upper=LIMITES_OUTLIERS[col][1])
                #

            df_sin["tempo_resolucao_ultimo_sinistro"] = pd.to_numeric(df_sin["tempo_resolucao_ultimo_sinistro"], errors="coerce")
            df_sin["tempo_resolucao_ultimo_sinistro"] = df_sin["tempo_resolucao_ultimo_sinistro"].clip(lower=LIMITES_OUTLIERS["tempo_resolucao_ultimo_sinistro"][0], upper=LIMITES_OUTLIERS["tempo_resolucao_ultimo_sinistro"][1])
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


            # ============================================================
            # 3. TRATAMENTO — ENGAJAMENTO MARKETING
            # ============================================================
            df_mkt = tabelas["marketing"].copy()
            df_mkt["ID"] = df_mkt["ID"].astype(str).str.strip()
            df_mkt.rename(columns={"ID": "id_cliente"}, inplace=True)

            numeric_cols_mkt = ["score_engajamento_digital", "indicou_clientes", "renovacoes_consecutivas", "indice_relacionamento", "ano_veiculo", "km_anual_estimado", "ultimo_login_portal_dias", "score_propensao_churn", "cluster_sugerido_crm"]
            for col in numeric_cols_mkt:
                df_mkt[col] = pd.to_numeric(df_mkt[col], errors="coerce")
                if col in LIMITES_OUTLIERS:
                    df_mkt[col] = df_mkt[col].clip(lower=LIMITES_OUTLIERS[col][0], upper=LIMITES_OUTLIERS[col][1])

            for col in ["tipo_veiculo", "segmento_marketing", "regiao_vendas"]:
                df_mkt[col] = df_mkt[col].apply(normalizar_texto_mkt)

            df_mkt["regiao_vendas"] = df_mkt["regiao_vendas"].replace({"Oeste": "Centro-Oeste", "Regiao Oeste": "Centro-Oeste", "Centro": "Centro-Oeste"})
            df_mkt["nunca_logou"] = df_mkt["ultimo_login_portal_dias"].isna().astype(int)  # mantido: é a flag que o pipeline espera (ORIGENS/fantasma_mkt não cobre isso, é feature própria)

            df_mkt["id_cliente"] = df_mkt["id_cliente"].astype(str).str.strip()  # MUDANÇA: padronização que faltava no rename original (id_cliente vindo de "ID" já tratado acima, mantido explícito por clareza)
            df_mkt.drop_duplicates(subset="id_cliente", keep="first", inplace=True)


            # ============================================================
            # 4. TRATAMENTO — CONTRATOS E APÓLICES
            # ============================================================
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
                # TRATAMENTO OUTLIER: Aplicar o dicionário se existir (para franquia_media)
                if col in LIMITES_OUTLIERS:
                    df_con[col] = df_con[col].clip(lower=LIMITES_OUTLIERS[col][0], upper=LIMITES_OUTLIERS[col][1])

            df_con["data_primeira_apolice"] = pd.to_datetime(df_con["data_primeira_apolice"], format="mixed", errors="coerce")

            for col in ["num_apolices_ativas", "tempo_cliente_dias", "num_produtos_contratados", "desconto_aplicado_pct"]:
                df_con[col] = pd.to_numeric(df_con[col], errors="coerce")
            df_con["desconto_aplicado_pct"] = df_con["desconto_aplicado_pct"] * 100

            # Regras de negócio fixas (valores absolutos, não dependem do lote) — mantidas como estavam:
            for col, (min_val, max_val) in {"valor_premio_anual": (0, 500_000), "valor_cobertura_total": (0, 2_000_000), "tempo_cliente_dias": (0, 10_950)}.items():
                df_con.loc[(df_con[col] < min_val) | (df_con[col] > max_val), col] = np.nan

            DATA_REFERENCIA = pd.Timestamp("2026-06-01")
            df_con["tempo_cliente_dias"] = df_con["tempo_cliente_dias"].fillna((DATA_REFERENCIA - df_con["data_primeira_apolice"]).dt.days)
            df_con["data_primeira_apolice"] = df_con["data_primeira_apolice"].fillna(DATA_REFERENCIA - pd.to_timedelta(df_con["tempo_cliente_dias"], unit="D"))

            df_con["id_cliente"] = df_con["id_cliente"].astype(str).str.strip()
            df_con.drop_duplicates(subset="id_cliente", keep="first", inplace=True)



            # ============================================================
            # 5. CONSOLIDAÇÃO DA BASE ÚNICA (MERGES)
            # ============================================================
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

            colunas_vazadas = ["data_ultimo_sinistro", "data_primeira_apolice", "data_nascimento"]
            df_final.drop(columns=[col for col in colunas_vazadas if col in df_final.columns], inplace=True)

            st.success(f"🎉 Processamento completo! Base unificada com {df_final.shape[0]} clientes e {df_final.shape[1]} variáveis.")

            modelo = artefatos["modelo_churn"]
            pipeline_churn = artefatos["pipeline_churn"]
            colunas_treino = artefatos["colunas_churn"]

            for col in colunas_treino:
                if col not in df_final.columns:
                    df_final[col] = np.nan

            X_raw = df_final[colunas_treino]

            try:
                X_scoring = pipeline_churn.transform(X_raw)
            except Exception as e:
                st.error(f"Erro ao aplicar o pipeline de preparação do modelo CatBoost: {e}")
                st.stop()

            t0 = time.time()

            # 2. Predição com o modelo CatBoost
            probabilidades = modelo.predict_proba(X_scoring)[:, 1]
            tempo_predicao = time.time() - t0

            df_resultado = df_final[["id_cliente"]].copy()
            df_resultado["Risco Churn (%)"] = (probabilidades * 100 if probabilidades.max() <= 1.0 else probabilidades).round(2)
            df_resultado["Status de Risco"] = np.where(
                df_resultado["Risco Churn (%)"] > 70.0,
                "🚨 Alto Risco",
                np.where(df_resultado["Risco Churn (%)"] > 30.0, "⚠️ Risco Moderado", "✅ Estável"),
            )

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
                    label="📥 Baixar Painel Consolidado de Churn (.CSV)",
                    data=csv_saida,
                    file_name="relatorio_final_churn_prt.csv",
                    mime="text/csv",
                )

            with col_cluster:
                st.subheader("🎯 Segmentação Estratégica da Carteira")
                st.markdown("Agrupamento comportamental de segurados via PCA e clusterização K=4.")

                cluster_dict = artefatos["cluster_dict"]
                pipeline_proc = artefatos["pipeline_cluster"]

                if cluster_dict is not None and pipeline_proc is not None:
                    try:
                        modelo_kmeans = cluster_dict["clusterer"]
                        modelo_pca = cluster_dict["pca"]

                        df_features = df_final.copy()

                        dados_processados = pipeline_proc.transform(df_features)

                        dados_processados = pipeline_proc.transform(df_features)

                        
                        clusters_preditos = modelo_kmeans.predict(dados_processados)
   
                        componentes_calculadas = np.asarray(modelo_pca.transform(dados_processados))

                        df_visualizacao_pca = pd.DataFrame(
                            componentes_calculadas[:, :2],
                            columns=["PC1", "PC2"],
                        )
                        df_visualizacao_pca["Cluster"] = [f"Grupo {c}" for c in clusters_preditos]
                        df_visualizacao_pca["id_cliente"] = df_final["id_cliente"].values

                        st.markdown("#### Mapa de Dispersão de Clientes")

                        mapa_cores_cluster = {
                            "Grupo 0": "#1B2A4A",  # azul-marinho escuro
                            "Grupo 1": "#4C7C59",  # verde musgo
                            "Grupo 2": "#A9AFAF",  # cinza-prata
                            "Grupo 3": "#3E9FB0",  # azul-petróleo / teal
                        }

                        fig_pca = px.scatter(
                            df_visualizacao_pca,
                            x="PC1",
                            y="PC2",
                            color="Cluster",
                            hover_data=["id_cliente"],
                            title=None,
                            color_discrete_map=mapa_cores_cluster,  
                            category_orders={"Cluster": ["Grupo 0", "Grupo 1", "Grupo 2", "Grupo 3"]},  
                            template="plotly_white",
                        )
                        fig_pca.update_layout(
                            margin=dict(l=10, r=10, t=10, b=10),
                            height=340,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            xaxis_title="Componente Principal 1 (PC1)",
                            yaxis_title="Componente Principal 2 (PC2)",
                        )
                        st.plotly_chart(fig_pca, use_container_width=True)
                        # ============================================================
                        # 📥 EXPORTAÇÃO DOS RESULTADOS DA CLUSTERIZAÇÃO
                        # ============================================================
                        st.markdown("---")
                        st.markdown("### 📥 Exportar Segmentação")
                        st.info("Baixe a lista completa contendo o ID dos clientes e seus respectivos segmentos (clusters).")

                        # Separa apenas as duas colunas solicitadas
                        df_download_cluster = df_visualizacao_pca[["id_cliente", "Cluster"]].copy()

                        # Converte para CSV
                        csv_cluster = df_download_cluster.to_csv(index=False).encode("utf-8")

                        # Renderiza o botão de download no Streamlit
                        st.download_button(
                            label="📥 Baixar Mapeamento de Clusters (.CSV)",
                            data=csv_cluster,
                            file_name="clientes_clusters_prt.csv",
                            mime="text/csv",
                        )

                    except Exception as e:
                        st.error(f"Falha no processamento da clusterização: {e}")
                else:
                    st.warning(
                        "Artefatos de clusterização indisponíveis. "
                        "Certifique-se de que 'pipeline_clusterizacao_k4.pkl' está na pasta do projeto."
                    )

        except Exception as e:
            st.error(f"Erro inesperado durante a execução da esteira de dados: {e}")
            st.exception(e)

elif arquivos_carregados:
    st.warning(
        f"Aguardando o carregamento dos arquivos restantes. "
        f"Você inseriu apenas {len(arquivos_carregados)} de 4 bases necessárias."
    )
elif not artefatos_carregados:
    st.error("Não foi possível carregar os artefatos do modelo de Churn. Verifique 'model.pkl' e 'colunas_modelo.pkl'.")
else:
    st.info("Faça o upload das 4 bases brutas (.csv) na barra lateral para iniciar a análise.")
