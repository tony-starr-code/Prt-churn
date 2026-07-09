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
            # MUDANÇA: bloco inteiro removido —
            #   for col in ["genero", "estado_civil", "tem_filhos", "escolaridade"]:
            #       df_cad[col] = imputar_categorica(df_cad[col])
            #   for col in ["idade", "renda_anual", "valor_imovel", "qtd_dependentes", "possui_imovel"]:
            #       df_cad[col] = imputar_amostra(df_cad[col])
            # Essas duas funções amostram/imputam a partir da distribuição do PRÓPRIO df_cad carregado.
            # No teste, isso usa estatística do teste; no treino, usava estatística do treino — são
            # transformações DIFERENTES aplicadas com o mesmo nome de função. É a causa mais provável
            # da divergência de distribuição no PCA. Essas colunas ("idade", "genero", "estado_civil",
            # "tem_filhos", "escolaridade", "renda_anual", "valor_imovel", "qtd_dependentes",
            # "possui_imovel") já estão listadas em COLUNAS_NUMERICAS_SENTINELA_NEG1 /
            # COLUNAS_CATEGORICAS do new_pipeline.py — o ImputadorUniversal (fit no treino) é quem
            # deve preencher os NaN, com a moda/distribuição do treino, não do teste.

            df_cad["id_cliente"] = df_cad["id_cliente"].astype(str).str.strip()
            df_cad.drop_duplicates(subset="id_cliente", keep="first", inplace=True)  # mantido: é por linha, não por estatística do lote

            # MUDANÇA: removido — tem_filhos/possui_imovel não são mais forçados a astype(int) aqui,
            # porque isso quebraria com NaN ainda presente. Ficam Int64 (nullable) até o pipeline imputar.
            df_cad.loc[(df_cad["tem_filhos"] == 0) & (df_cad["qtd_dependentes"] > 0), "tem_filhos"] = 1
            # nota: essa regra de negócio (quem tem dependente > 0 tem filho) é determinística e foi mantida.
            # Ela só não roda para linhas onde tem_filhos ou qtd_dependentes ainda são NaN — o que é o
            # comportamento correto: não se deve inferir uma regra de negócio sobre um dado ausente.


            # ============================================================
            # 2. TRATAMENTO — ATENDIMENTO / SINISTROS
            # ============================================================
            df_sin = tabelas["sinistros"].copy()
            df_sin = limpar_nulos(df_sin)
            df_sin.rename(columns={"customer_key": "id_cliente", "ID": "id_cliente"}, errors="ignore", inplace=True)
            df_sin["id_cliente"] = df_sin["id_cliente"].astype(float).astype(int).astype(str).str.strip()

            # MUDANÇA: removida a imputação categórica do canal de contato aqui —
            #   df_sin["canal_preferencial_contato"] = imputar_categorica(df_sin["canal_preferencial_contato"])
            # "canal_preferencial_contato" está em COLUNAS_CATEGORICAS no new_pipeline.py: o
            # ImputadorUniversal treinado já imputa pela moda DO TREINO.

            cols_num_sin = ["num_reclamacoes_12m", "num_sinistros_historico", "dias_ultimo_contato", "tempo_medio_resposta_dias", "num_ligacoes_suporte_12m", "num_acessos_app_mes", "satisfacao_nps"]
            for col in cols_num_sin:
                df_sin[col] = pd.to_numeric(df_sin[col], errors="coerce")
                if col in LIMITES_OUTLIERS:
                    df_sin[col] = df_sin[col].clip(lower=LIMITES_OUTLIERS[col][0], upper=LIMITES_OUTLIERS[col][1])
                # MUDANÇA: removido imputar_amostra(df_sin[col]) — idem ao comentário acima, essas colunas
                # também estão em COLUNAS_NUMERICAS_SENTINELA_NEG1.

            df_sin["tempo_resolucao_ultimo_sinistro"] = pd.to_numeric(df_sin["tempo_resolucao_ultimo_sinistro"], errors="coerce")
            df_sin["tempo_resolucao_ultimo_sinistro"] = df_sin["tempo_resolucao_ultimo_sinistro"].clip(lower=LIMITES_OUTLIERS["tempo_resolucao_ultimo_sinistro"][0], upper=LIMITES_OUTLIERS["tempo_resolucao_ultimo_sinistro"][1])
            df_sin.loc[df_sin["tempo_resolucao_ultimo_sinistro"].isnull() & df_sin["data_ultimo_sinistro"].isnull(), "tempo_resolucao_ultimo_sinistro"] = 0
            # MUDANÇA: removido imputar_amostra(...) restante — o que sobrar de NaN aqui (sinistro existiu
            # mas o tempo de resolução não foi preenchido) é um NaN genuíno e deve ir pro pipeline treinado.

            df_sin["data_ultimo_sinistro"] = pd.to_datetime(df_sin["data_ultimo_sinistro"], errors="coerce", format="mixed", dayfirst=True)
            df_sin.loc[df_sin["data_ultimo_sinistro"] > hoje, "data_ultimo_sinistro"] = pd.NaT

            # MUDANÇA: removido o preenchimento de data por amostragem do PRÓPRIO df_sin carregado:
            #   df_sin.loc[mask_data_sin, "data_ultimo_sinistro"] = df_sin["data_ultimo_sinistro"].dropna().sample(...)
            # Isso sorteava uma data de dentro do lote de teste — instável com poucos registros e
            # inconsistente com o que foi feito no treino. "dias_desde_ultimo_sinistro" (calculado abaixo)
            # é a feature que realmente importa pro modelo; para linhas sem data, ele fica NaN e é
            # imputado pelo pipeline treinado, igual às outras numéricas.

            df_sin.loc[~df_sin["satisfacao_nps"].between(0, 10), "satisfacao_nps"] = np.nan
            # MUDANÇA: removido .astype(int) após imputar_amostra — mantido como Int64 nullable.
            df_sin["satisfacao_nps"] = df_sin["satisfacao_nps"].astype("Int64")

            # MUDANÇA: removido o clipping de "dias_ultimo_contato" por IQR calculado no lote:
            #   Q1_dias = df_sin["dias_ultimo_contato"].quantile(0.25)  # <- estatística do teste
            #   Q3_dias = df_sin["dias_ultimo_contato"].quantile(0.75)
            #   df_sin["dias_ultimo_contato"] = df_sin["dias_ultimo_contato"].clip(...)
            # Mesma razão do clipping removido em df_cad: os limites têm que vir do treino, não do teste.

            for col in ["num_reclamacoes_12m", "num_sinistros_historico", "num_ligacoes_suporte_12m", "num_acessos_app_mes"]:
                df_sin[col] = df_sin[col].astype("Int64")  # MUDANÇA: Int64 nullable em vez de int (sem fillna prévio, pode ter NaN)

            df_sin["dias_desde_ultimo_sinistro"] = (hoje - df_sin["data_ultimo_sinistro"]).dt.days.astype("Int64")
            # MUDANÇA: removido o .fillna(0) aqui. "Nunca teve sinistro" != "sinistro há 0 dias" — eram
            # valores semanticamente diferentes sendo colapsados no mesmo número. Deixa NaN e o pipeline
            # treinado decide o que fazer (ou usa a flag "teve_sinistro" abaixo, que já existe pra isso).
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

            # MUDANÇA: bloco de imputação inteiro removido daqui —
            #   df_mkt["ultimo_login_portal_dias"] = df_mkt["ultimo_login_portal_dias"].fillna(mediana do lote)
            #   df_mkt["indicou_clientes"] = df_mkt["indicou_clientes"].fillna(0)
            #   df_mkt["renovacoes_consecutivas"] = df_mkt["renovacoes_consecutivas"].fillna(0)
            #   df_mkt["regiao_vendas"] = df_mkt["regiao_vendas"].fillna(moda do lote)
            #   df_mkt["segmento_marketing"] = df_mkt["segmento_marketing"].fillna(moda do lote)
            #   df_mkt["tipo_veiculo"] = df_mkt["tipo_veiculo"].fillna(moda por grupo do lote) ...
            #   for col in ["ano_veiculo", "km_anual_estimado"]: fillna(mediana por grupo do lote)
            #   for col in ["score_engajamento_digital", "indice_relacionamento"]: fillna(mediana do lote)
            #   df_mkt["score_propensao_churn"] = ...fillna(mediana do lote)
            #   df_mkt["cluster_sugerido_crm"] = ...fillna(moda do lote)
            # Todas essas colunas (score_engajamento_digital, indicou_clientes, renovacoes_consecutivas,
            # indice_relacionamento, ultimo_login_portal_dias, nunca_logou, segmento_marketing,
            # regiao_vendas, tipo_veiculo) estão cobertas pelo ImputadorUniversal do new_pipeline.py
            # (COLUNAS_NUMERICAS_SENTINELA_NEG1 / COLUNAS_CATEGORICAS). "indicou_clientes" e
            # "renovacoes_consecutivas" não precisam de fillna(0) manual: 0 é o valor real de "não indicou
            # ninguém" só quando a linha realmente tem 0 informado — quando é NaN (não respondeu), deve
            # seguir NaN pro imputador, não virar 0 artificialmente.
            # "cluster_sugerido_crm" nem chega a entrar no modelo: é removida em
            # RemovedorDeColunas(colunas_para_remover) dentro do próprio new_pipeline.py — não há motivo
            # pra gastar lógica de imputação com ela aqui.

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
            # nota: mantido "plus" -> "Premium". O CodificadorOrdinalManual do new_pipeline.py só conhece
            # {"Básica": 1, "Padrão": 2, "Premium": 3} — se "Plus" ficasse como categoria própria (como no
            # notebook), o .map(encoding_maps) geraria NaN silencioso pra essa categoria no ordinal encoding.
            # Esse comportamento do app.py já está correto e compatível com o pipeline treinado.

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

            # MUDANÇA: removidos todos os fillna(mediana/moda do lote) desta seção —
            #   df_con["num_apolices_ativas"] = ...fillna(mediana do lote)
            #   df_con["num_produtos_contratados"] = ...fillna(mediana do lote)
            #   df_con["desconto_aplicado_pct"] = ...fillna(mediana do lote)
            #   df_con["tipo_cobertura"] = ...fillna(moda do lote)
            #   df_con["canal_aquisicao"] = ...fillna(moda do lote)
            #   df_con["metodo_pagamento"] = ...fillna(moda do lote)
            #   df_con["pagamento_em_dia"] = ...fillna(moda do lote)
            #   for col in colunas_monetarias: fillna(mediana por grupo tipo_cobertura do lote)
            # Todas essas colunas estão em COLUNAS_NUMERICAS_SENTINELA_NEG1 / COLUNAS_CATEGORICAS /
            # ORIGENS["fantasma_contratos"] do new_pipeline.py — é o ImputadorUniversal treinado quem
            # deve preencher, com a moda/distribuição do treino.

            DATA_REFERENCIA = pd.Timestamp("2026-06-01")
            df_con["tempo_cliente_dias"] = df_con["tempo_cliente_dias"].fillna((DATA_REFERENCIA - df_con["data_primeira_apolice"]).dt.days)
            # MUDANÇA: removido o dropna(subset=["tempo_cliente_dias"]).
            # Descartar a linha inteira quando os dois (data e tempo_cliente_dias) estão ausentes é
            # aceitável ao construir a base de TREINO (você decide fora do fluxo de produção quais
            # linhas entram no treino). Em INFERÊNCIA isso é perigoso: silenciosamente sumir com
            # clientes do lote de teste (ele nem aparece no resultado final, o que é diferente de
            # "aparecer como outlier"). Deixe o NaN seguir para o pipeline treinado.
            df_con["data_primeira_apolice"] = df_con["data_primeira_apolice"].fillna(DATA_REFERENCIA - pd.to_timedelta(df_con["tempo_cliente_dias"], unit="D"))

            df_con["id_cliente"] = df_con["id_cliente"].astype(str).str.strip()
            df_con.drop_duplicates(subset="id_cliente", keep="first", inplace=True)

            # MUDANÇA: bloco de winsorização por IQR removido inteiramente —
            #   CONFIG_OUTLIERS = {...}
            #   for col, fator in CONFIG_OUTLIERS.items():
            #       Q1 = df_con[col].quantile(0.25)   # <- estatística do LOTE DE TESTE
            #       Q3 = df_con[col].quantile(0.75)
            #       ... clip(lower=..., upper=...)
            # Esse é o ponto mais direto de divergência treino x teste: no notebook, Q1/Q3 foram
            # calculados uma vez sobre a base de treino inteira para gerar bases_tratadas/. Aqui, a
            # MESMA fórmula estava recalculando Q1/Q3 sobre qualquer lote novo carregado na interface —
            # com poucos registros, esses quantis ficam instáveis e os limites de corte mudam a cada
            # execução, o que desloca a escala das colunas monetárias antes de chegarem ao
            # StandardScaler (que foi ajustado com a escala do treino). Se quiser manter algum corte de
            # sanidade aqui, salve os limites (limite_inf/limite_sup) UMA VEZ a partir do treino (ex.:
            # num JSON) e aplique os mesmos números fixos sempre, em vez de repetir o .quantile() a
            # cada carga.


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




            # MUDANÇA: removido o bloco final —
            #   for col in df_final.columns:
            #       if df_final[col].dtype in ["float64", "int64", "Int64"]:
            #           df_final[col] = df_final[col].fillna(0)
            #
            # Esse era o bug mais grave da cadeia: depois do merge outer, um cliente ausente em uma
            # das 4 bases fica NaN em TODAS as colunas daquela origem — e esse fillna(0) zerava idade,
            # renda, índice de relacionamento etc. em vez de tratar isso como "dado ausente por
            # origem". Isso cria registros com valores fisicamente impossíveis (idade 0, renda 0) que,
            # depois de padronizados com a escala aprendida no treino, aparecem exatamente como os
            # pontos extremos isolados no PCA.
            #
            # O new_pipeline.py já resolve esse cenário de propósito: dentro de ImputadorUniversal,
            # o dicionário ORIGENS (fantasma_contratos / fantasma_mkt / fantasma_cadastro /
            # fantasma_sinistros) cria uma flag por origem ANTES de imputar, e "qtd_dados_ausentes" /
            # "eh_fantasma" resumem isso — exatamente para marcar "esse cliente não tinha dado nessa
            # tabela" em vez de mascarar com zero. Deixe os NaN em df_final como estão e entregue ao
            # pipeline treinado:
            #
            #     pipeline = joblib.load("caminho/para/pipeline_treinado.pkl")   # já fit() no treino
            #     df_pronto_para_kmeans = pipeline.transform(df_final)
            #
            # Não chame fit() nem fit_transform() aqui — isso re-calcularia distribuições/modas em cima
            # do teste (voltando ao Bug 1), e no caso do CodificadorAlvoManual/OneHotEncoder mudaria o
            # próprio espaço de features usado pelo K-means treinado.

            st.success(f"🎉 Processamento completo! Base unificada com {df_final.shape[0]} clientes e {df_final.shape[1]} variáveis.")

            modelo = artefatos["modelo_churn"]
            colunas_treino = artefatos["colunas_churn"]

            for col in colunas_treino:
                if col not in df_final.columns:
                    df_final[col] = 0

            X_scoring = df_final[colunas_treino]

            t0 = time.time()
            st.write("Quantidade total de NaNs:", X_scoring.isna().sum().sum())
            st.write(X_scoring.isna().sum()[X_scoring.isna().sum() > 0])
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

                        st.write("NaNs antes do pipeline:")
                        st.write(df_features.isna().sum()[df_features.isna().sum() > 0])

                        dados_processados = pipeline_proc.transform(df_features)

                        dados_processados = pipeline_proc.transform(df_features)

                        if isinstance(dados_processados, pd.DataFrame):
                            nans = dados_processados.isna().sum()
                            colunas_com_nan = nans[nans > 0]

                            st.write("NaNs após o pipeline:")
                            st.write(colunas_com_nan)

                            if len(colunas_com_nan) > 0:
                                st.write("Valores únicos das colunas problemáticas (antes do pipeline):")
                                for col in colunas_com_nan.index:
                                    # remove o prefixo do ColumnTransformer
                                    col_original = col.split("__")[-1]

                                    if col_original in df_features.columns:   # dados = dataframe original
                                        st.write(f"### {col_original}")
                                        st.write(sorted(df_features[col_original].dropna().unique()))
                        else:
                            st.write("NaNs:", np.isnan(dados_processados).sum())

                        clusters_preditos = modelo_kmeans.predict(dados_processados)

                        st.write(type(pipeline_proc))
                        st.write(pipeline_proc)
   
                        componentes_calculadas = np.asarray(modelo_pca.transform(dados_processados))

                        df_visualizacao_pca = pd.DataFrame(
                            componentes_calculadas[:, :2],
                            columns=["PC1", "PC2"],
                        )
                        df_visualizacao_pca["Cluster"] = [f"Grupo {c}" for c in clusters_preditos]
                        df_visualizacao_pca["id_cliente"] = df_final["id_cliente"].values

                        st.markdown("#### Mapa de Dispersão de Clientes")
                        fig_pca = px.scatter(
                            df_visualizacao_pca,
                            x="PC1",
                            y="PC2",
                            color="Cluster",
                            hover_data=["id_cliente"],
                            title=None,
                            color_discrete_sequence=px.colors.qualitative.Bold,
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
                        # 🛠️ MÓDULO DE DEPURAÇÃO DE OUTLIERS DO PCA
                        # ============================================================
                        st.markdown("---")
                        st.markdown("### 🔍 Investigador de Anomalias do PCA")
                        st.info("Ferramenta de engenharia reversa para identificar qual variável (feature) está distorcendo a projeção de clientes no mapa espacial.")
                        
                        # 1. Identificar os clientes mais distantes da origem (0,0)
                        df_visualizacao_pca['Distancia_Origem'] = np.sqrt(df_visualizacao_pca['PC1']**2 + df_visualizacao_pca['PC2']**2)
                        df_outliers = df_visualizacao_pca.sort_values(by='Distancia_Origem', ascending=False)
                        
                        st.markdown("**Top 5 Clientes mais extremos na projeção atual:**")
                        st.dataframe(df_outliers.head(5)[['id_cliente', 'Cluster', 'PC1', 'PC2', 'Distancia_Origem']], use_container_width=True)

                        # 2. Seletor para investigação profunda
                        cliente_alvo = st.selectbox(
                            "Selecione um ID de Cliente para realizar a decomposição matemática:", 
                            df_outliers['id_cliente'].head(20).tolist()
                        )

                        if cliente_alvo:
                            st.markdown(f"#### Raio-X do Cliente: `{cliente_alvo}`")
                            
                            # A. Dados Originais
                            st.markdown("**1. O que entrou no Pipeline (Base Unificada Bruta):**")
                            raw_row = df_final[df_final['id_cliente'] == cliente_alvo]
                            st.dataframe(raw_row)
                            
                            # B. Dados Transformados
                            idx_cliente = raw_row.index[0]
                            if isinstance(dados_processados, pd.DataFrame):
                                transformed_row = dados_processados.iloc[idx_cliente:idx_cliente+1]
                                features_names = dados_processados.columns
                            else:
                                features_names = pipeline_proc[:-1].get_feature_names_out()
                                transformed_row = pd.DataFrame(
                                    dados_processados[idx_cliente:idx_cliente+1], 
                                    columns=features_names
                                )
                            
                            st.markdown("**2. O que saiu do Pipeline (Após Imputação, Engenharia e Scaling):**")
                            st.dataframe(transformed_row)
                            
                            # C. Reconstrução do PCA
                            pesos_pc1 = modelo_pca.components_[0]
                            pesos_pc2 = modelo_pca.components_[1]
                            valores_z = transformed_row.values[0]
                            
                            # Calcular o impacto de cada feature
                            impacto_pc1 = pesos_pc1 * valores_z
                            impacto_pc2 = pesos_pc2 * valores_z
                            
                            df_impacto = pd.DataFrame({
                                'Feature': features_names,
                                'Valor Pós-Pipeline (Z)': valores_z,
                                'Peso no PC1': pesos_pc1,
                                'Impacto Real no PC1': impacto_pc1,
                                'Impacto Real no PC2': impacto_pc2
                            })
                            
                            # Ordenar pelo maior impacto absoluto no PC1
                            df_impacto['Impacto Absoluto PC1'] = df_impacto['Impacto Real no PC1'].abs()
                            df_impacto = df_impacto.sort_values(by='Impacto Absoluto PC1', ascending=False).drop(columns=['Impacto Absoluto PC1'])
                            
                            st.markdown("**3. Decomposição do Componente: Qual feature causou a explosão?**")
                            st.error("🚨 **Atenção à primeira linha desta tabela.** Ela mostra a variável exata que puxou o ponto para longe. Olhe a coluna 'Valor Pós-Pipeline (Z)'. Se este valor for maior que 10 ou menor que -10, o StandardScaler falhou ou uma divisão por zero ocorreu na engenharia de features.")
                            st.dataframe(df_impacto, use_container_width=True, hide_index=True)



                        st.markdown("#### Distribuição de Clientes por Segmento")
                        dist_cluster = df_visualizacao_pca["Cluster"].value_counts().reset_index()
                        dist_cluster.columns = ["Segmento Identificado", "Volume de Clientes"]
                        st.dataframe(
                            dist_cluster.sort_values(by="Segmento Identificado"),
                            use_container_width=True,
                            hide_index=True,
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
