import pandas as pd
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold


# Configuração para o scikit-learn devolver DataFrames em vez de arrays Numpy
sklearn.set_config(transform_output="pandas")


COLUNAS_NUMERICAS_SENTINELA_NEG1 = [
    "num_apolices_ativas", "valor_premio_anual", "tempo_cliente_dias",
    "num_produtos_contratados", "valor_cobertura_total", "franquia_media",
    "pagamento_em_dia", "desconto_aplicado_pct", "score_engajamento_digital",
    "indicou_clientes", "renovacoes_consecutivas", "indice_relacionamento",
    "ano_veiculo", "km_anual_estimado", "ultimo_login_portal_dias",
    "nunca_logou", "idade", "tem_filhos", "qtd_dependentes", "renda_anual",
    "possui_imovel", "valor_imovel", "tempo_residencia_anos",
    "num_reclamacoes_12m", "num_sinistros_historico", "dias_ultimo_contato",
    "tempo_medio_resposta_dias", "num_ligacoes_suporte_12m",
    "tempo_resolucao_ultimo_sinistro", "num_acessos_app_mes",
    "satisfacao_nps", "teve_sinistro", "dias_desde_ultimo_sinistro",
    "score_propensao_churn"
]

COLUNAS_CATEGORICAS = [
    "tipo_cobertura", "canal_aquisicao", "metodo_pagamento",
    "segmento_marketing", "regiao_vendas", "genero", "estado_civil",
    "escolaridade", "canal_preferencial_contato", "tipo_veiculo",
]

ORIGENS = {
    "fantasma_contratos": [
        "num_apolices_ativas", "tipo_cobertura", "valor_premio_anual",
        "tempo_cliente_dias", "num_produtos_contratados",
        "valor_cobertura_total", "franquia_media", "pagamento_em_dia",
        "desconto_aplicado_pct", "metodo_pagamento",
    ],
    "fantasma_mkt": [
        "score_engajamento_digital", "indicou_clientes",
        "renovacoes_consecutivas", "indice_relacionamento",
        "ultimo_login_portal_dias", "nunca_logou", "num_acessos_app_mes",
    ],
    "fantasma_cadastro": [
        "idade", "genero", "estado_civil", "tem_filhos", "qtd_dependentes",
        "escolaridade", "renda_anual", "possui_imovel", "valor_imovel",
        "tempo_residencia_anos", "regiao_vendas",
    ],
    "fantasma_sinistros": [
        "num_reclamacoes_12m", "num_sinistros_historico", "dias_ultimo_contato",
        "tempo_medio_resposta_dias", "num_ligacoes_suporte_12m",
        "tempo_resolucao_ultimo_sinistro", "satisfacao_nps", "teve_sinistro",
        "dias_desde_ultimo_sinistro",
    ],
}


# ==============================================================================
# TRANSFORMERS CUSTOMIZADOS
# ==============================================================================

class ImputadorUniversal(BaseEstimator, TransformerMixin):
    """
    Substitui o dropna() + o script de imputação separado do Kaggle por UM
    único transformer, ajustado apenas no treino e aplicado (transform) de
    forma idêntica em treino/validação/teste.

    - Cria flags de "fantasma" por origem ANTES de imputar (o sinal mais forte
      que a EDA encontrou não pode ser jogado fora).
    - Imputa numéricas por amostragem da distribuição empírica de treino
      (mesma lógica que você já validou para dias_desde_ultimo_sinistro,
      generalizada para todas as colunas com sentinela -1).
    - Imputa categóricas pela moda do treino.
    """

    def __init__(
        self,
        colunas_numericas=COLUNAS_NUMERICAS_SENTINELA_NEG1,
        colunas_categoricas=COLUNAS_CATEGORICAS,
        origens=ORIGENS,
        random_state=42,
    ):
        self.colunas_numericas = colunas_numericas
        self.colunas_categoricas = colunas_categoricas
        self.origens = origens
        self.random_state = random_state

    def fit(self, X, y=None):
        X = X.copy()

        cols_num = [c for c in self.colunas_numericas if c in X.columns]
        cols_cat = [c for c in self.colunas_categoricas if c in X.columns]

        # Distribuições empíricas de treino, por coluna numérica
        self.distribuicoes_ = {}
        for col in cols_num:
            validos = pd.to_numeric(X[col], errors="coerce").replace(-1, np.nan).dropna()
            self.distribuicoes_[col] = validos.values if len(validos) > 0 else np.array([0.0])

        # Moda de treino, por coluna categórica
        self.modas_ = {}
        for col in cols_cat:
            moda = X[col].mode(dropna=True)
            self.modas_[col] = moda.iloc[0] if len(moda) > 0 else "Sem_Registro_Origem"

        self._cols_num_fit = cols_num
        self._cols_cat_fit = cols_cat
        return self

    def transform(self, X):
        X = X.copy()
        rng = np.random.RandomState(self.random_state)

        # --- 1. Flags de "fantasma" por origem (calculadas ANTES de imputar) ---
        for nome_flag, cols in self.origens.items():
            cols_presentes = [c for c in cols if c in X.columns]
            if not cols_presentes:
                continue
            ausente_num = pd.DataFrame({
                c: (pd.to_numeric(X[c], errors="coerce") == -1) | X[c].isna()
                for c in cols_presentes if c in self._cols_num_fit
            })
            ausente_cat = pd.DataFrame({
                c: X[c].isna() for c in cols_presentes if c in self._cols_cat_fit
            })
            partes = [df for df in [ausente_num, ausente_cat] if not df.empty]
            if partes:
                X[nome_flag] = pd.concat(partes, axis=1).any(axis=1).astype(int)
            else:
                X[nome_flag] = 0

        X["qtd_dados_ausentes"] = X[list(self.origens.keys())].sum(axis=1)
        X["eh_fantasma"] = (X["qtd_dados_ausentes"] > 0).astype(int)

        # --- 2. Imputação numérica por amostragem da distribuição de treino ---
        for col in self._cols_num_fit:
            X[col] = pd.to_numeric(X[col], errors="coerce").replace(-1, np.nan)
            mask_nulos = X[col].isna()
            n_nulos = int(mask_nulos.sum())
            if n_nulos > 0:
                amostra = rng.choice(self.distribuicoes_[col], size=n_nulos, replace=True)
                X.loc[mask_nulos, col] = amostra

        # --- 3. Imputação categórica pela moda de treino ---
        for col in self._cols_cat_fit:
            X[col] = X[col].fillna(self.modas_[col])

        return X


class EngenhariaDeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # O fit aprende a média de desconto APENAS no X_train
        self.media_desc_ = X["desconto_aplicado_pct"].mean()
        return self

    def transform(self, X):
        X = X.copy()

        X['custo_beneficio'] = X['valor_premio_anual'] / X['valor_cobertura_total'].replace(0, 1)

        X['friccao_pagamento'] = (
            (X['metodo_pagamento'].str.lower() == 'boleto') &
            (X['pagamento_em_dia'] == 0)
        ).astype(int)

        X["cliente_novo_alto_desconto"] = (
            (X["renovacoes_consecutivas"] <= 1) &
            (X["desconto_aplicado_pct"] > self.media_desc_)
        ).astype(int)

        X['reclamacoes_s_resposta'] = (
            (X['num_reclamacoes_12m'] > 0) &
            (X['dias_ultimo_contato'] > 90) &
            (X['satisfacao_nps'] <= 6)
        ).astype(int)

        X['comprometimento_renda'] = X['valor_premio_anual'] / X['renda_anual'].replace(0, 1)
        X['premio_por_apolice'] = X['valor_premio_anual'] / X['num_apolices_ativas'].replace(0, 1)

        tempo_anos = (X['tempo_cliente_dias'] / 365).replace(0, 0.1)
        X['frequencia_sinistros_tempo'] = X['num_sinistros_historico'] / tempo_anos

        X['isolamento_digital'] = (
            (X['nunca_logou'] == 1) |
            (X['ultimo_login_portal_dias'] > 180)
        ).astype(int)

        X['renda_per_capita'] = X['renda_anual'] / (X['qtd_dependentes'] + 1)
        X['custo_por_km'] = X['valor_premio_anual'] / X['km_anual_estimado'].replace(0, 1)
        X['peso_franquia_premio'] = X['franquia_media'] / X['valor_premio_anual'].replace(0, 1)

        tempo_anos_exato = X['tempo_cliente_dias'] / 365
        X['idade_ingresso'] = X['idade'] - tempo_anos_exato

        # Interações com NPS (relação quase monotônica e uma das mais fortes na EDA)
        X['score_insatisfacao'] = 11 - X['satisfacao_nps']
        X['friccao_aguda_nps'] = X['score_insatisfacao'] * (X['num_reclamacoes_12m'] + 1)
        X['custo_da_frustracao'] = X['score_insatisfacao'] * X['valor_premio_anual']
        
        # Clientes entre o 11º e o 13º mês
        X['risco_atrito_score'] = (
            X['isolamento_digital'] +
            X['reclamacoes_s_resposta'] +
            (X['satisfacao_nps'] <= 6).astype(int) +
            (X['num_ligacoes_suporte_12m'] > X['num_ligacoes_suporte_12m'].median()).astype(int)
        )
        
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
        return X


class RemovedorDeColunas(BaseEstimator, TransformerMixin):
    def __init__(self, colunas):
        self.colunas = colunas

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.colunas if c in X.columns], errors='ignore')

class CodificadorAlvoManual(BaseEstimator, TransformerMixin):
    """
    Target encoding escrito à mão (sem category_encoders / sem OneHotEncoder).
 
    Por que não é só "mapear categoria -> média de churn direto":
    - Se a média fosse calculada e aplicada nas MESMAS linhas que a geraram,
      cada linha "vazaria" um pouco do seu próprio y na própria feature
      (overfitting, principalmente em categorias com poucas amostras).
    - Por isso, fit_transform (usado só no treino) calcula a codificação de
      cada linha usando K-fold: a média da categoria é calculada com os
      OUTROS folds, nunca com o fold que contém a própria linha.
    - Já fit() (chamado internamente) guarda o mapeamento final -- calculado
      com o treino inteiro -- para ser usado depois em transform() (val/teste),
      sem novo K-fold, já que ali não existe mais risco de a linha vazar
      informação sobre si mesma.
    - "smoothing" evita que uma categoria com poucas linhas vire um valor
      extremo: quanto menos exemplos a categoria tiver, mais o valor final
      puxa pra média global em vez da média isolada da categoria.
    """
 
    def __init__(self, colunas, n_splits=5, smoothing=10, random_state=42):
        self.colunas = colunas
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
 
    def _medias_suavizadas(self, coluna, y):
        media_global = y.mean()
        stats = y.groupby(coluna).agg(['mean', 'count'])
        suavizada = (
            stats['count'] * stats['mean'] + self.smoothing * media_global
        ) / (stats['count'] + self.smoothing)
        return suavizada, media_global
 
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("CodificadorAlvoManual precisa de y — passe y no fit/fit_transform do Pipeline.")
        y = pd.Series(np.asarray(y), index=X.index)
 
        self.mapas_ = {}
        self.medias_globais_ = {}
        for col in self.colunas:
            if col not in X.columns:
                continue
            mapa, media_global = self._medias_suavizadas(X[col], y)
            self.mapas_[col] = mapa
            self.medias_globais_[col] = media_global
        return self
 
    def transform(self, X):
        # Usado em validação/teste: aplica o mapeamento já fixado no fit(),
        # sem recalcular nada a partir de X.
        X = X.copy()
        for col in self.colunas:
            if col not in X.columns or col not in self.mapas_:
                continue
            media_global = self.medias_globais_[col]
            X[col] = X[col].map(self.mapas_[col]).astype(float)
            X[col] = X[col].fillna(media_global)
        return X
 
    def fit_transform(self, X, y=None):
        if y is None:
            raise ValueError("CodificadorAlvoManual precisa de y — passe y no fit/fit_transform do Pipeline.")
 
        # 1. Guarda o mapeamento final (treino inteiro) para uso posterior em transform()
        self.fit(X, y)
 
        # 2. Gera a versão K-fold (sem vazamento) para a SAÍDA do treino
        y_serie = pd.Series(np.asarray(y), index=X.index)
        X_saida = X.copy()
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
 
        for col in self.colunas:
            if col not in X.columns:
                continue
            codificado = pd.Series(index=X.index, dtype=float)
            for idx_tr, idx_ho in skf.split(X, y_serie):
                mapa_fold, media_fold = self._medias_suavizadas(
                    X[col].iloc[idx_tr], y_serie.iloc[idx_tr]
                )
                codificado.iloc[idx_ho] = (
                    X[col].iloc[idx_ho].map(mapa_fold).fillna(media_fold)
                )
            X_saida[col] = codificado
 
        return X_saida




# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def build_pipeline() -> dict:

    # Mapas ordinais — agora NENHUMA dessas colunas é removida depois,
    # diferente da versão anterior (o bug estava aqui).
    encoding_maps = {
        "tipo_cobertura":      {"Básica": 1, "Padrão": 2, "Premium": 3},
        "segmento_marketing":  {"Bronze": 1, "Prata": 2, "Ouro": 3, "Diamante": 4},
        "escolaridade":        {"Fundamental": 1, "Medio": 2, "Superior": 3, "Pos": 4},
    }

    # Só removemos: identificador e as duas colunas com risco claro de
    # vazamento (score/cluster vindos de um modelo de churn anterior).
    colunas_para_remover = [
        'id_cliente',
        'cluster_sugerido_crm',
        
        # --- Colunas Originais (Denominadores) Removidas p/ Evitar Multicolinearidade ---
        'valor_cobertura_total', # Agora o modelo focará no 'custo_beneficio'
        'renda_anual',           # Agora o modelo focará no 'comprometimento_renda' e 'renda_per_capita'
        'num_apolices_ativas',   # Agora o modelo focará no 'premio_por_apolice'
        'qtd_dependentes',       # Embutido na 'renda_per_capita'
        'km_anual_estimado',     # Agora o modelo focará no 'custo_por_km'
    ]

    # Categóricas restantes (que antes eram descartadas inteiras) agora
    # entram via one-hot em vez de serem jogadas fora.
    one_hot_cols = [
        'estado_civil', 'canal_aquisicao', 'metodo_pagamento',
        'canal_preferencial_contato', 'genero', 'regiao_vendas',
        'tipo_veiculo',
    ]

    colunas_target_encoding = [
        'estado_civil',
        'canal_aquisicao',
        'metodo_pagamento',
        'canal_preferencial_contato',
        'genero',
        'regiao_vendas',
        'tipo_veiculo',
    ]
    
    transformador_colunas = ColumnTransformer(
        transformers=[
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), one_hot_cols),
        ],
        remainder="passthrough"
        )


    pipeline = Pipeline([
    ("imputador", ImputadorUniversal()),
    ("engenharia_features", EngenhariaDeFeatures()),
    ("ordinal_encoding", CodificadorOrdinalManual(encoding_maps)),
    ("remover_colunas", RemovedorDeColunas(colunas_para_remover)),
    ("codificacao", transformador_colunas),
    ("padronizacao", StandardScaler())
    ])


    return {"pipeline": pipeline}