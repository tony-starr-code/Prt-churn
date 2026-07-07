import time
import re

import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import joblib
from dateutil import parser

# CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="PRT Seguradora - Churn", layout="wide")

# CACHE DOS ARTEFATOS DO MODELO (AGORA UTILIZANDO CLOUDPICKLE PARA O MLFLOW)
@st.cache_resource
def carregar_artefatos():
    # Carrega o modelo binário do MLflow usando cloudpickle para evitar o travamento
    with open('model.pkl', 'rb') as f:
        modelo = cloudpickle.load(f)
    colunas_treino = joblib.load('colunas_modelo.pkl')
    return modelo, colunas_treino

try:
    modelo, colunas_treino = carregar_artefatos()
    artefatos_carregados = True
except Exception as e:
    artefatos_carregados = False
    st.error(f"Erro ao carregar os artefatos do modelo: {e}")

# --- SUAS FUNÇÕES GLOBAIS DE TRATAMENTO (inalteradas) ---
NULOS_DISFARÇADOS = ['#n/d', '-', '', '?', 'n/a', 'na', 'null', 'none', '-']

def limpar_nulos(df):
    for col in df.select_dtypes(include=['object', 'string']).columns:
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
    except:
        return pd.NaT

def normalizar_texto_mkt(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip()
    if v.upper() in ['', '-', '?', '#N/D', 'NAN']:
        return np.nan
    return v.title()

def moda_segura(x):
    m = x.mode()
    return m.iloc[0] if not m.empty else np.nan

def normalizar_categoria_con(valor):
    if pd.isna(valor):
        return np.nan
    return str(valor).strip().lower().replace('.', '')

def normalizar_canal_con(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip()
    if v.lower() in ['', '-', '?', '#n/d', 'nan']:
        return np.nan
    return v.title()

def normalizar_metodo_con(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().lower().replace('_', ' ').replace('-', ' ')
    return re.sub(r'\s+', ' ', v)

def limpar_valor_monetario_con(valor):
    if pd.isna(valor):
        return np.nan
    v = str(valor).strip().replace('R$', '').replace(' ', '').strip()
    tem_ponto = '.' in v
    tem_virgula = ',' in v
    if tem_ponto and tem_virgula:
        if v.rfind(',') > v.rfind('.'):
            v = v.replace('.', '').replace(',', '.')
        else:
            v = v.replace(',', '')
    elif tem_virgula:
        v = v.replace('.', '').replace(',', '.')
    elif tem_ponto:
        partes = v.split('.')
        if len(partes[-1]) != 2:
            v = v.replace('.', '')
    try:
        return float(v)
    except ValueError:
        return np.nan

# --- INTERFACE DO STREAMLIT ---
st.title("📂 Painel de Predição de Churn — PRT Seguradora")
st.markdown("### Processamento de Bases Brutas & Análise de Churn em Lote")

arquivos_carregados = st.file_uploader(
    "Arraste as 4 bases brutas (.csv) simultaneamente aqui:",
    type=["csv"],
    accept_multiple_files=True
)

if arquivos_carregados and len(arquivos_carregados) == 4 and artefatos_carregados:
    st.info("As 4 bases brutas foram detectadas! Iniciando a esteira de Engenharia de Dados...")

    tabelas = {}
    hoje = pd.Timestamp.today()

    # Mapeamento dinâmico baseado na estrutura interna de colunas
    for arquivo in arquivos_carregados:
        amostra = arquivo.read(2048).decode('utf-8')
        arquivo.seek(0)
        sep = ';' if ';' in amostra and amostra.count(';') > amostra.count(',') else ','

        df_temp = pd.read_csv(arquivo, sep=sep)
        colunas_temp = [c.lower() for c in df_temp.columns]

        if 'data_nascimento' in colunas_temp or 'escolaridade' in colunas_temp:
            tabelas['cadastro'] = df_temp
        elif 'customer_key' in colunas_temp or 'num_sinistros_historico' in colunas_temp:
            tabelas['sinistros'] = df_temp
        elif 'score_engajamento_digital' in colunas_temp or 'km_anual_estimado' in colunas_temp:
            tabelas['marketing'] = df_temp
        elif 'cod_individuo' in colunas_temp or 'valor_premio_anual' in colunas_temp or 'tipo_cobertura' in colunas_temp:
            tabelas['contratos'] = df_temp

    if len(tabelas) < 4:
        st.error("Erro no mapeamento. Certifique-se de fazer o upload de todas as 4 bases distintas (Cadastro, Sinistros, Marketing e Contratos).")
    else:
        try:
            # ============================================================
            # 1. TRATAMENTO — CADASTRO DOS CLIENTES
            # ============================================================
            df_cad = tabelas['cadastro'].copy()
            df_cad = limpar_nulos(df_cad)
            df_cad.rename(columns={'Id_cliente': 'id_cliente', 'ID_Cliente': 'id_cliente', 'Id_Cliente': 'id_cliente'}, errors='ignore', inplace=True)

            df_cad['idade'] = pd.to_numeric(df_cad['idade'], errors='coerce').astype('Int64')
            df_cad['data_nascimento'] = df_cad['data_nascimento'].apply(parse_data)
            mask_data = df_cad['idade'].isnull() & df_cad['data_nascimento'].notnull()
            df_cad.loc[mask_data, 'idade'] = df_cad.loc[mask_data, 'data_nascimento'].apply(lambda x: int((hoje - x).days / 365.25))
            df_cad.drop(columns='data_nascimento', errors='ignore', inplace=True)

            mapa_genero = {'masc': 'M', 'm': 'M', 'masculino': 'M', 'f': 'F', 'fem': 'F', 'feminino': 'F'}
            df_cad['genero'] = df_cad['genero'].astype(str).str.strip().str.lower().map(mapa_genero)

            mapa_ec = {'c': 'casado', 'casado': 'casado', 'married': 'casado', 'casado(a)': 'casado', 's': 'solteiro', 'solt': 'solteiro', 'single': 'solteiro', 'solteiro(a)': 'solteiro'}
            df_cad['estado_civil'] = df_cad['estado_civil'].astype(str).str.strip().str.lower().map(mapa_ec)

            mapa_filhos = {'sim': 1, 'true': 1, 's': 1, '1': 1, 'nao': 0, 'não': 0, 'n': 0, 'false': 0, '0': 0}
            df_cad['tem_filhos'] = df_cad['tem_filhos'].astype(str).str.strip().str.lower().map(mapa_filhos)

            df_cad['qtd_dependentes'] = pd.to_numeric(df_cad['qtd_dependentes'], errors='coerce').astype('Int64')
            df_cad['escolaridade'] = df_cad['escolaridade'].astype(str).str.strip().str.lower().str.capitalize()

            for col in ['renda_anual', 'valor_imovel']:
                if col in df_cad.columns:
                    df_cad[col] = df_cad[col].astype(str).str.strip().str.replace(r'r\$', '', regex=True).str.replace(r'\s', '', regex=True).str.replace(r'\.(?=\d{3})', '', regex=True).str.replace(',', '.', regex=False)
                    df_cad[col] = pd.to_numeric(df_cad[col], errors='coerce')

            df_cad['possui_imovel'] = df_cad['possui_imovel'].astype(str).str.strip().str.lower().replace(NULOS_DISFARÇADOS + ['nan'], np.nan)
            df_cad['possui_imovel'] = pd.to_numeric(df_cad['possui_imovel'], errors='coerce').astype('Int64')

            df_cad['tempo_residencia_anos'] = pd.to_numeric(df_cad['tempo_residencia_anos'], errors='coerce')
            df_cad['tempo_residencia_anos'] = df_cad['tempo_residencia_anos'].fillna(df_cad['tempo_residencia_anos'].median()).astype(int)

            for col in ['genero', 'estado_civil', 'tem_filhos', 'escolaridade']:
                df_cad[col] = imputar_categorica(df_cad[col])
            for col in ['idade', 'renda_anual', 'valor_imovel', 'qtd_dependentes', 'possui_imovel']:
                df_cad[col] = imputar_amostra(df_cad[col])

            # >>> CORREÇÃO: padroniza id_cliente ANTES do drop_duplicates <<<
            df_cad['id_cliente'] = df_cad['id_cliente'].astype(str).str.strip()
            df_cad.drop_duplicates(subset='id_cliente', keep='first', inplace=True)

            for col in ['renda_anual', 'valor_imovel']:
                Q1 = df_cad[col].quantile(0.25)
                Q3 = df_cad[col].quantile(0.75)
                IQR = Q3 - Q1
                df_cad[col] = df_cad[col].clip(lower=max(0, Q1 - 1.5 * IQR), upper=Q3 + 1.5 * IQR)

            df_cad['idade'] = df_cad['idade'].astype(float).where(df_cad['idade'].between(18, 100), other=np.nan)
            df_cad['idade'] = imputar_amostra(df_cad['idade'])
            df_cad['tem_filhos'] = df_cad['tem_filhos'].astype(int)
            df_cad['possui_imovel'] = df_cad['possui_imovel'].astype(int)
            df_cad.loc[(df_cad['tem_filhos'] == 0) & (df_cad['qtd_dependentes'] > 0), 'tem_filhos'] = 1

            # ============================================================
            # 2. TRATAMENTO — ATENDIMENTO / SINISTROS
            # ============================================================
            df_sin = tabelas['sinistros'].copy()
            df_sin = limpar_nulos(df_sin)
            df_sin.rename(columns={'customer_key': 'id_cliente', 'ID': 'id_cliente'}, errors='ignore', inplace=True)
            df_sin['id_cliente'] = df_sin['id_cliente'].astype(float).astype(int).astype(str).str.strip()

            df_sin['canal_preferencial_contato'] = imputar_categorica(df_sin['canal_preferencial_contato'])

            cols_num_sin = ['num_reclamacoes_12m', 'num_sinistros_historico', 'dias_ultimo_contato', 'tempo_medio_resposta_dias', 'num_ligacoes_suporte_12m', 'num_acessos_app_mes', 'satisfacao_nps']
            for col in cols_num_sin:
                df_sin[col] = pd.to_numeric(df_sin[col], errors='coerce')
                df_sin[col] = imputar_amostra(df_sin[col])

            df_sin['tempo_resolucao_ultimo_sinistro'] = pd.to_numeric(df_sin['tempo_resolucao_ultimo_sinistro'], errors='coerce')
            df_sin.loc[df_sin['tempo_resolucao_ultimo_sinistro'].isnull() & df_sin['data_ultimo_sinistro'].isnull(), 'tempo_resolucao_ultimo_sinistro'] = 0
            df_sin['tempo_resolucao_ultimo_sinistro'] = imputar_amostra(df_sin['tempo_resolucao_ultimo_sinistro'])

            df_sin['data_ultimo_sinistro'] = pd.to_datetime(df_sin['data_ultimo_sinistro'], errors='coerce', format='mixed', dayfirst=True)
            df_sin.loc[df_sin['data_ultimo_sinistro'] > hoje, 'data_ultimo_sinistro'] = pd.NaT

            mask_data_sin = df_sin['data_ultimo_sinistro'].isna() & (df_sin['tempo_resolucao_ultimo_sinistro'] > 0)
            if mask_data_sin.sum() > 0 and not df_sin['data_ultimo_sinistro'].dropna().empty:
                df_sin.loc[mask_data_sin, 'data_ultimo_sinistro'] = df_sin['data_ultimo_sinistro'].dropna().sample(mask_data_sin.sum(), replace=True).values

            df_sin.loc[~df_sin['satisfacao_nps'].between(0, 10), 'satisfacao_nps'] = np.nan
            df_sin['satisfacao_nps'] = imputar_amostra(df_sin['satisfacao_nps']).astype(int)

            Q1_dias = df_sin['dias_ultimo_contato'].quantile(0.25)
            Q3_dias = df_sin['dias_ultimo_contato'].quantile(0.75)
            df_sin['dias_ultimo_contato'] = df_sin['dias_ultimo_contato'].clip(lower=max(0, Q1_dias - 1.5 * (Q3_dias - Q1_dias)), upper=Q3_dias + 1.5 * (Q3_dias - Q1_dias))

            for col in ['num_reclamacoes_12m', 'num_sinistros_historico', 'num_ligacoes_suporte_12m', 'num_acessos_app_mes']:
                df_sin[col] = df_sin[col].astype(int)

            df_sin['dias_desde_ultimo_sinistro'] = (hoje - df_sin['data_ultimo_sinistro']).dt.days.fillna(0).astype(int)
            df_sin['teve_sinistro'] = np.where(df_sin['num_sinistros_historico'] > 0, 1, 0)
            # id_cliente já padronizado logo após o rename, então o dedup abaixo já é seguro
            df_sin.drop_duplicates(subset='id_cliente', keep='first', inplace=True)

            # ============================================================
            # 3. TRATAMENTO — ENGAJAMENTO MARKETING
            # ============================================================
            df_mkt = tabelas['marketing'].copy()
            df_mkt['ID'] = df_mkt['ID'].astype(str).str.strip()
            df_mkt.rename(columns={"ID": "id_cliente"}, inplace=True)

            numeric_cols_mkt = ['score_engajamento_digital', 'indicou_clientes', 'renovacoes_consecutivas', 'indice_relacionamento', 'ano_veiculo', 'km_anual_estimado', 'ultimo_login_portal_dias', 'score_propensao_churn', 'cluster_sugerido_crm']
            for col in numeric_cols_mkt:
                df_mkt[col] = pd.to_numeric(df_mkt[col], errors='coerce')

            for col in ['tipo_veiculo', 'segmento_marketing', 'regiao_vendas']:
                df_mkt[col] = df_mkt[col].apply(normalizar_texto_mkt)

            df_mkt['regiao_vendas'] = df_mkt['regiao_vendas'].replace({'Oeste': 'Centro-Oeste', 'Regiao Oeste': 'Centro-Oeste', 'Centro': 'Centro-Oeste'})
            df_mkt['nunca_logou'] = df_mkt['ultimo_login_portal_dias'].isna().astype(int)
            df_mkt['ultimo_login_portal_dias'] = df_mkt['ultimo_login_portal_dias'].fillna(df_mkt['ultimo_login_portal_dias'].median())

            df_mkt['indicou_clientes'] = df_mkt['indicou_clientes'].fillna(0)
            df_mkt['renovacoes_consecutivas'] = df_mkt['renovacoes_consecutivas'].fillna(0)

            df_mkt['regiao_vendas'] = df_mkt['regiao_vendas'].fillna(df_mkt['regiao_vendas'].mode()[0] if not df_mkt['regiao_vendas'].mode().empty else 'Ignorado')
            df_mkt['segmento_marketing'] = df_mkt['segmento_marketing'].fillna(df_mkt['segmento_marketing'].mode()[0] if not df_mkt['segmento_marketing'].mode().empty else 'Ignorado')

            df_mkt['tipo_veiculo'] = df_mkt['tipo_veiculo'].fillna(df_mkt.groupby('segmento_marketing')['tipo_veiculo'].transform(moda_segura))
            df_mkt['tipo_veiculo'] = df_mkt['tipo_veiculo'].fillna(df_mkt['tipo_veiculo'].mode()[0] if not df_mkt['tipo_veiculo'].mode().empty else 'Ignorado')

            for col in ['ano_veiculo', 'km_anual_estimado']:
                df_mkt[col] = df_mkt[col].fillna(df_mkt.groupby('tipo_veiculo')[col].transform('median')).fillna(df_mkt[col].median())

            for col in ['score_engajamento_digital', 'indice_relacionamento']:
                df_mkt.loc[df_mkt['nunca_logou'] == 1, col] = df_mkt.loc[df_mkt['nunca_logou'] == 1, col].fillna(0)
                df_mkt[col] = df_mkt[col].fillna(df_mkt[col].median())

            df_mkt['score_propensao_churn'] = df_mkt['score_propensao_churn'].fillna(df_mkt['score_propensao_churn'].median())
            df_mkt['cluster_sugerido_crm'] = df_mkt['cluster_sugerido_crm'].fillna(df_mkt['cluster_sugerido_crm'].mode()[0] if not df_mkt['cluster_sugerido_crm'].mode().empty else 0)
            # id_cliente já padronizado (str+strip) logo no início do bloco
            df_mkt.drop_duplicates(subset='id_cliente', keep='first', inplace=True)

            # ============================================================
            # 4. TRATAMENTO — CONTRATOS E APÓLICES
            # ============================================================
            df_con = tabelas['contratos'].copy()
            df_con['cod_individuo'] = df_con['cod_individuo'].astype(str).str.replace('IND-', '', regex=False).str.strip()
            df_con.rename(columns={"cod_individuo": "id_cliente"}, inplace=True)

            mapa_cobertura = {'premium': 'Premium', 'prem': 'Premium', 'básica': 'Básica', 'basica': 'Básica', 'basic': 'Básica', 'padrão': 'Padrão', 'padrao': 'Padrão', 'std': 'Padrão', 'plus': 'Plus'}
            df_con['tipo_cobertura'] = df_con['tipo_cobertura'].apply(normalizar_categoria_con).map(mapa_cobertura)
            df_con['canal_aquisicao'] = df_con['canal_aquisicao'].apply(normalizar_canal_con)

            mapa_metodo = {'boleto': 'Boleto', 'bol': 'Boleto', 'boleto bancario': 'Boleto', 'cartao': 'Cartao', 'cartão': 'Cartao', 'cc': 'Cartao', 'cartao credito': 'Cartao', 'debito auto': 'Debito', 'debito automatico': 'Debito', 'debito_auto': 'Debito', 'debito': 'Debito', 'deb auto': 'Debito', 'pix': 'Pix'}
            df_con['metodo_pagamento'] = df_con['metodo_pagamento'].apply(normalizar_metodo_con).map(mapa_metodo)

            mapa_pagamento = {'em dia': 1, 'ok': 1, 'sim': 1, 's': 1, '1': 1, 'nao': 0, 'não': 0, 'n': 0, '0': 0, 'atrasado': 0}
            df_con['pagamento_em_dia'] = df_con['pagamento_em_dia'].apply(lambda x: mapa_pagamento.get(str(x).strip().lower(), np.nan)).astype('float64')

            colunas_monetarias = ['valor_premio_anual', 'valor_cobertura_total', 'franquia_media']
            for col in colunas_monetarias:
                df_con[col] = df_con[col].apply(limpar_valor_monetario_con)

            df_con['data_primeira_apolice'] = pd.to_datetime(df_con['data_primeira_apolice'], format='mixed', errors='coerce')

            for col in ['num_apolices_ativas', 'tempo_cliente_dias', 'num_produtos_contratados', 'desconto_aplicado_pct']:
                df_con[col] = pd.to_numeric(df_con[col], errors='coerce')
            df_con["desconto_aplicado_pct"] = df_con["desconto_aplicado_pct"] * 100

            # Limites Realistas e Imputações
            for col, (min_val, max_val) in {'valor_premio_anual': (0, 500_000), 'valor_cobertura_total': (0, 2_000_000), 'tempo_cliente_dias': (0, 10_950)}.items():
                df_con.loc[(df_con[col] < min_val) | (df_con[col] > max_val), col] = np.nan

            df_con["num_apolices_ativas"] = df_con["num_apolices_ativas"].fillna(df_con["num_apolices_ativas"].median())
            df_con["num_produtos_contratados"] = df_con["num_produtos_contratados"].fillna(df_con["num_produtos_contratados"].median())
            df_con["desconto_aplicado_pct"] = df_con["desconto_aplicado_pct"].fillna(df_con["desconto_aplicado_pct"].median())

            df_con["tipo_cobertura"] = df_con["tipo_cobertura"].fillna(df_con["tipo_cobertura"].mode()[0])
            df_con["canal_aquisicao"] = df_con["canal_aquisicao"].fillna(df_con["canal_aquisicao"].mode()[0])
            df_con["metodo_pagamento"] = df_con["metodo_pagamento"].fillna(df_con["metodo_pagamento"].mode()[0])
            df_con["pagamento_em_dia"] = df_con["pagamento_em_dia"].fillna(df_con["pagamento_em_dia"].mode()[0])

            for col in colunas_monetarias:
                df_con[col] = df_con[col].fillna(df_con.groupby("tipo_cobertura")[col].transform("median"))

            DATA_REFERENCIA = pd.Timestamp("2026-06-01")
            df_con["tempo_cliente_dias"] = df_con["tempo_cliente_dias"].fillna((DATA_REFERENCIA - df_con["data_primeira_apolice"]).dt.days)
            df_con = df_con.dropna(subset=["tempo_cliente_dias"])
            df_con["data_primeira_apolice"] = df_con["data_primeira_apolice"].fillna(DATA_REFERENCIA - pd.to_timedelta(df_con["tempo_cliente_dias"], unit="D"))

            # >>> CORREÇÃO: padroniza id_cliente ANTES do drop_duplicates <<<
            df_con['id_cliente'] = df_con['id_cliente'].astype(str).str.strip()
            df_con.drop_duplicates(subset='id_cliente', keep='first', inplace=True)

            # Winsorização de Outliers
            CONFIG_OUTLIERS = {'valor_premio_anual': 2.5, 'valor_cobertura_total': 2.5, 'franquia_media': 1.5, 'tempo_cliente_dias': 1.5, 'desconto_aplicado_pct': 1.5}
            for col, fator in CONFIG_OUTLIERS.items():
                Q1 = df_con[col].quantile(0.25)
                Q3 = df_con[col].quantile(0.75)
                lim_inf = max(0, Q1 - fator * (Q3 - Q1))
                lim_sup = Q3 + fator * (Q3 - Q1)
                if col == 'desconto_aplicado_pct':
                    lim_sup = min(100, lim_sup)
                df_con[col] = df_con[col].clip(lower=lim_inf, upper=lim_sup)

            # ============================================================
            # 5. CONSOLIDAÇÃO DA BASE ÚNICA (MERGES)
            # ============================================================
            # Padronização final (defensiva, redundante mas inofensiva) antes do merge
            df_cad['id_cliente'] = df_cad['id_cliente'].astype(str).str.strip()
            df_sin['id_cliente'] = df_sin['id_cliente'].astype(str).str.strip()
            df_mkt['id_cliente'] = df_mkt['id_cliente'].astype(str).str.strip()
            df_con['id_cliente'] = df_con['id_cliente'].astype(str).str.strip()

            # >>> DIAGNÓSTICO: mostra shape e duplicatas de cada base antes do merge <<<
            with st.expander("🔍 Diagnóstico pré-merge (linhas e duplicatas de id_cliente)"):
                for nome, df_ in [("cadastro", df_cad), ("sinistros", df_sin),
                                   ("marketing", df_mkt), ("contratos", df_con)]:
                    dups = df_['id_cliente'].duplicated().sum()
                    st.write(f"**{nome}**: {df_.shape[0]} linhas | {dups} id_cliente duplicados")

            # >>> CORREÇÃO: validate='one_to_one' faz o merge falhar rápido (com erro claro)
            # em vez de gerar um produto cartesiano silencioso que trava o app depois <<<
            df_final = df_cad.merge(df_sin, on='id_cliente', how='left', validate='one_to_one')
            df_final = df_final.merge(df_mkt, on='id_cliente', how='left', validate='one_to_one')
            df_final = df_final.merge(df_con, on='id_cliente', how='left', validate='one_to_one')

            st.write(f"Shape final após merges: {df_final.shape}")

            # Preenchimento de nulos remanescentes
            for col in df_final.columns:
                if df_final[col].dtype in ['float64', 'int64', 'Int64']:
                    df_final[col] = df_final[col].fillna(0)
                else:
                    df_final[col] = df_final[col].fillna('ignorado')

            st.success("🎉 Processamento completo! Todas as 4 bases brutas foram unificadas via Pipeline.")

            # ============================================================
            # 6. EXECUÇÃO DAS PREDIÇÕES DO MODELO
            # ============================================================
            # Garante a existência de todas as colunas esperadas pelo modelo original
            for col in colunas_treino:
                if col not in df_final.columns:
                    df_final[col] = 0

            X_scoring = df_final[colunas_treino]

            # >>> USANDO DUMMY PREDICTIONS (aleatórias) POIS O MODELO.PKL NÃO FUNCIONA <<<
            st.error("⚠️ AVISO: O arquivo model.pkl está com problema (não consegue desserializar ou tem dependência faltando).")
            st.error("O pipeline está rodando com PREDIÇÕES ALEATÓRIAS (dummy) pra você testar o resto.")
            st.info("Próximos passos: revise o model.pkl (versão do scikit-learn/XGBoost usada no treino vs agora)")
            
            st.write(f"Gerando predições dummy para {X_scoring.shape[0]} linhas...")
            t0 = time.time()
            probabilidades = np.random.uniform(0, 1, len(X_scoring))
            st.write(f"✅ Predições geradas em {time.time() - t0:.2f}s")

            df_resultado = df_final[['id_cliente']].copy()
            df_resultado['Risco Churn (%)'] = (probabilidades * 100 if probabilidades.max() <= 1.0 else probabilidades).round(2)
            df_resultado['Status de Risco'] = np.where(df_resultado['Risco Churn (%)'] > 70.0, '🚨 Alto Risco', np.where(df_resultado['Risco Churn (%)'] > 30.0, '⚠️ Risco Moderado', '✅ Estável'))

            st.markdown("### 📊 Relatório Técnico de Risco de Churn")
            st.dataframe(df_resultado.sort_values(by='Risco Churn (%)', ascending=False), use_container_width=True)

            csv_saida = df_resultado.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Baixar Painel Consolidado de Churn (.CSV)", data=csv_saida, file_name="relatorio_final_churn_prt.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Erro inesperado durante a execução da esteira de dados: {e}")
            st.exception(e)

elif arquivos_carregados:
    st.warning(f"Aguardando o carregamento dos arquivos restantes. Você inseriu apenas {len(arquivos_carregados)} de 4 bases necessárias.")
