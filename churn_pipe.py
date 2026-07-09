import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class ImputadorDistribuicao(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if 'dias_desde_ultimo_sinistro' in X.columns:
            valores_validos = X['dias_desde_ultimo_sinistro'].replace(-1, np.nan).dropna()
            self.distribuicao_treino_ = valores_validos.values if len(valores_validos) > 0 else np.array([0])
        else:
            self.distribuicao_treino_ = np.array([0])
        return self

    def transform(self, X):
        X = X.copy()
        if 'dias_desde_ultimo_sinistro' in X.columns:
            X['dias_desde_ultimo_sinistro'] = X['dias_desde_ultimo_sinistro'].replace(-1, np.nan)
            nulos_mask = X['dias_desde_ultimo_sinistro'].isna()
            num_nulos = nulos_mask.sum()

            if num_nulos > 0:
                amostra = np.random.choice(self.distribuicao_treino_, size=num_nulos, replace=True)
                X.loc[nulos_mask, 'dias_desde_ultimo_sinistro'] = amostra
            
            X['dias_desde_ultimo_sinistro'] = X['dias_desde_ultimo_sinistro'].fillna(-1)
        return X

        
class EngenhariaDeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.media_desc_ = X["desconto_aplicado_pct"].mean() if "desconto_aplicado_pct" in X.columns else 0
        return self

    def transform(self, X):
        X = X.copy()
        
        premio = X.get('valor_premio_anual', 0).fillna(0)
        cobertura = X.get('valor_cobertura_total', 1).fillna(1).replace(0, 1)
        renda = X.get('renda_anual', 1).fillna(1).replace(0, 1)
        apolices = X.get('num_apolices_ativas', 1).fillna(1).replace(0, 1)
        km = X.get('km_anual_estimado', 1).fillna(1).replace(0, 1)
        
        X['custo_beneficio'] = premio / cobertura
        
        if 'metodo_pagamento' in X.columns and 'pagamento_em_dia' in X.columns:
            X['friccao_pagamento'] = ((X['metodo_pagamento'].astype(str).str.lower() == 'boleto') & (X['pagamento_em_dia'] == 0)).astype(int)
        else:
            X['friccao_pagamento'] = 0
            
        X["cliente_novo_alto_desconto"] = ((X.get("renovacoes_consecutivas", 0).fillna(0) <= 1) & (X.get("desconto_aplicado_pct", 0).fillna(0) > self.media_desc_)).astype(int)

        X['reclamacoes_s_resposta'] = ((X.get('num_reclamacoes_12m', 0).fillna(0) > 0) & (X.get('dias_ultimo_contato', 0).fillna(0) > 90) & (X.get('satisfacao_nps', 10).fillna(10) <= 6)).astype(int)
        
        X['comprometimento_renda'] = premio / renda
        X['premio_por_apolice'] = premio / apolices
        
        tempo_anos = (X.get('tempo_cliente_dias', 365).fillna(365) / 365).replace(0, 0.1)
        X['frequencia_sinistros_tempo'] = X.get('num_sinistros_historico', 0).fillna(0) / tempo_anos
        
        X['isolamento_digital'] = ((X.get('nunca_logou', 0).fillna(0) == 1) | (X.get('ultimo_login_portal_dias', 0).fillna(0) > 180)).astype(int)
        X['renda_per_capita'] = X.get('renda_anual', 0).fillna(0) / (X.get('qtd_dependentes', 0).fillna(0) + 1)
        X['custo_por_km'] = premio / km
        X['peso_franquia_premio'] = X.get('franquia_media', 0).fillna(0) / premio.replace(0, 1)
        
        X['idade_ingresso'] = X.get('idade', 30).fillna(30) - (X.get('tempo_cliente_dias', 0).fillna(0) / 365)

        # Regras de Negócio baseadas no Insight Gráfico de Faixa Etária (Idade)
        X['jovem_baixa_renda'] = ((X.get('idade', 30).fillna(30) <= 35) & (X['comprometimento_renda'] > 0.05)).astype(int)
        X['senior_com_sinistro'] = ((X.get('idade', 30).fillna(30) >= 56) & (X.get('teve_sinistro', 0).fillna(0) == 1)).astype(int)

        return X


class CriadorFaixaEtaria(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if 'idade' in X.columns:
            idade_preenchida = X['idade'].fillna(30)
            bins = [0, 25, 35, 45, 55, 65, np.inf]
            labels = ['Até 25', '26-35', '36-45', '46-55', '56-65', 'Mais de 65']
            X['faixa_etaria'] = pd.cut(idade_preenchida, bins=bins, labels=labels, right=True)
            X['faixa_etaria'] = X['faixa_etaria'].astype(str)
        return X


class CodificadorOrdinalManual(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dicts):
        self.mapping_dicts = mapping_dicts

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col, map_dict in self.mapping_dicts.items():
            if col in X.columns:
                X[col] = X[col].map(map_dict)
                X[col] = X[col].fillna(-1)
        return X


class RemovedorDeColunas(BaseEstimator, TransformerMixin):
    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.colunas if c in X.columns], errors='ignore')


# ==============================================================================
# 2. FUNÇÃO CENTRAL DE CONSTRUÇÃO DO DATA PIPELINE
# ==============================================================================

def build_pipeline(
    data_dir: str = "../bases_tratadas",
) -> dict:
    
    encoding_maps = {
        "tipo_cobertura":      {"Básica": 1, "Padrão": 2, "Premium": 3},
        "segmento_marketing":  {"Bronze": 1, "Prata": 2, "Ouro": 3, "Diamante": 4},
        "escolaridade":        {"Fundamental": 1, "Medio": 2, "Superior": 3, "Pos": 4},
        "faixa_etaria":        {"Até 25": 1, "26-35": 2, "36-45": 3, "46-55": 4, "56-65": 5, "Mais de 65": 6}
    }

    colunas_para_remover = ['id_cliente', 'score_propensao_churn', 'cluster_sugerido_crm']

    # Mapeamento completo e explícito das variáveis categóricas (Remove erros com 'carta' ou 'whatsapp')
    one_hot_cols = [
        'estado_civil', 'genero', 'canal_aquisicao', 'metodo_pagamento', 
        'regiao_vendas', 'tipo_veiculo', 'canal_preferencial_contato'
    ]

    numeric_cols = [
        'idade', 'tempo_cliente_dias', 'qtd_dependentes', 'renda_anual', 
        'valor_premio_anual', 'valor_cobertura_total', 'desconto_aplicado_pct', 
        'franquia_media', 'num_apolices_ativas', 'num_reclamacoes_12m', 
        'num_sinistros_historico', 'num_ligacoes_suporte_12m', 'num_acessos_app_mes', 
        'ultimo_login_portal_dias', 'dias_ultimo_contato', 'satisfacao_nps',
        'score_engajamento_digital', 'indice_relacionamento', 'ano_veiculo', 'km_anual_estimado'
    ]
    
    features_engenharia = [
        'custo_beneficio', 'friccao_pagamento', 'cliente_novo_alto_desconto',
        'reclamacoes_s_resposta', 'comprometimento_renda', 'premio_por_apolice',
        'frequencia_sinistros_tempo', 'isolamento_digital', 'renda_per_capita',
        'custo_por_km', 'peso_franquia_premio', 'idade_ingresso',
        'jovem_baixa_renda', 'senior_com_sinistro'
    ]
    
    ordinais_mapeadas = ['tipo_cobertura', 'segmento_marketing', 'escolaridade', 'faixa_etaria']
    
    num_cols_total = numeric_cols + features_engenharia + ordinais_mapeadas

    # ColumnTransformer blindado: remainder='drop' ignora qualquer vazamento de texto residual
    transformador_colunas = ColumnTransformer(
        transformers=[
            ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), one_hot_cols),
            ('num', StandardScaler(), [c for c in num_cols_total if c in num_cols_total and c not in colunas_para_remover])
        ],
        remainder='drop'
    )

    pipeline_preparacao = Pipeline(steps=[
        ('imputador',             ImputadorDistribuicao()),
        ('engenharia',            EngenhariaDeFeatures()),
        ('criar_faixas',          CriadorFaixaEtaria()),
        ('ordinal_encoding',      CodificadorOrdinalManual(encoding_maps)),
        ('remover_colunas',       RemovedorDeColunas(colunas_para_remover)),
        ('pre_processador_final', transformador_colunas),
    ])

    return {
        "pipeline":      pipeline_preparacao,
    }
