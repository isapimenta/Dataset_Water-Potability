"""
Testes Automatizados com PyTest — Modelo de Potabilidade da Água
Engenharia de Software para Sistemas Inteligentes

Objetivo: Verificar se o modelo atende aos requisitos mínimos de desempenho
definidos para implantação em produção. Em caso de substituição do modelo,
estes testes evitam a implantação de um modelo inferior ao estabelecido.

Requisitos de desempenho (thresholds):
  - Acurácia    : ≥ 0.60  (60% de acertos mínimos)
  - F1-Score    : ≥ 0.55  (considerando desbalanceamento das classes)
  - ROC-AUC     : ≥ 0.62  (capacidade discriminativa mínima)
  - Precisão    : ≥ 0.55  (taxa de falsos positivos aceitável)
  - Recall      : ≥ 0.50  (mínimo de potabilidade verdadeira identificada)

Como executar:
  pip install pytest
  pytest tests/test_model.py -v
"""

import pytest
import joblib
import numpy as np
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# ── Configuração dos caminhos ────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'backend', 'modelo_potabilidade.pkl')

# ── URL do dataset (mesma usada no notebook — garante mesma distribuição) ────
DATA_URL = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/water_potability.csv"

# ── Thresholds de desempenho (requisitos mínimos para implantação) ────────────
MIN_ACCURACY  = 0.60  # 60%
MIN_F1_SCORE  = 0.55  # 55%
MIN_ROC_AUC   = 0.62  # 62%
MIN_PRECISION = 0.55  # 55%
MIN_RECALL    = 0.50  # 50%

# ── Features esperadas pelo modelo ───────────────────────────────────────────
EXPECTED_FEATURES = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def modelo():
    """Carrega o modelo uma única vez para todos os testes do módulo."""
    assert os.path.exists(MODEL_PATH), (
        f"Arquivo de modelo não encontrado: {MODEL_PATH}\n"
        "Execute o notebook para gerar 'modelo_potabilidade.pkl'."
    )
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope='module')
def dados_teste():
    """
    Carrega o dataset e retorna o conjunto de teste.
    Utiliza a mesma semente e mesma proporção da separação original (20% teste).
    """
    df = pd.read_csv(DATA_URL)
    X = df.drop('Potability', axis=1)
    y = df['Potability']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test, y_test


@pytest.fixture(scope='module')
def predicoes(modelo, dados_teste):
    """Realiza as predições uma única vez e disponibiliza para os testes de métricas."""
    X_test, y_test = dados_teste
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]
    return y_test, y_pred, y_prob


# ── Testes de carregamento e estrutura ────────────────────────────────────────

class TestCarregamentoModelo:
    """Verifica se o modelo pode ser carregado e possui a estrutura correta."""

    def test_arquivo_existe(self):
        """O arquivo do modelo deve existir no caminho configurado."""
        assert os.path.exists(MODEL_PATH), \
            f"Modelo não encontrado em: {MODEL_PATH}"

    def test_modelo_carrega_sem_erro(self, modelo):
        """O modelo deve ser carregado sem lançar exceções."""
        assert modelo is not None, "O modelo foi carregado como None"

    def test_modelo_tem_metodo_predict(self, modelo):
        """O modelo deve implementar o método predict() (interface sklearn)."""
        assert hasattr(modelo, 'predict'), \
            "O modelo não possui método predict()"

    def test_modelo_tem_metodo_predict_proba(self, modelo):
        """O modelo deve suportar predições probabilísticas."""
        assert hasattr(modelo, 'predict_proba'), \
            "O modelo não possui método predict_proba()"

    def test_modelo_e_pipeline(self, modelo):
        """O modelo deve ser um Pipeline (contendo pré-processamento integrado)."""
        from sklearn.pipeline import Pipeline
        assert isinstance(modelo, Pipeline), \
            f"Esperado Pipeline, obtido: {type(modelo).__name__}"


# ── Testes de formato de saída ────────────────────────────────────────────────

class TestFormatoSaida:
    """Verifica o formato das saídas do modelo."""

    def test_predicao_retorna_array(self, modelo):
        """O método predict deve retornar um array numpy."""
        amostra = pd.DataFrame([{
            'ph': 7.0, 'Hardness': 200.0, 'Solids': 20000.0,
            'Chloramines': 7.0, 'Sulfate': 350.0, 'Conductivity': 500.0,
            'Organic_carbon': 10.0, 'Trihalomethanes': 80.0, 'Turbidity': 3.0
        }])
        predicao = modelo.predict(amostra)
        assert isinstance(predicao, np.ndarray), \
            f"predict() deve retornar ndarray, obtido: {type(predicao)}"

    def test_predicao_retorna_classe_valida(self, modelo):
        """As classes preditas devem ser 0 (Não Potável) ou 1 (Potável)."""
        amostra = pd.DataFrame([{
            'ph': 7.0, 'Hardness': 200.0, 'Solids': 20000.0,
            'Chloramines': 7.0, 'Sulfate': 350.0, 'Conductivity': 500.0,
            'Organic_carbon': 10.0, 'Trihalomethanes': 80.0, 'Turbidity': 3.0
        }])
        predicao = modelo.predict(amostra)
        assert predicao[0] in [0, 1], \
            f"Classe predita deve ser 0 ou 1, obtida: {predicao[0]}"

    def test_probabilidades_somam_um(self, modelo):
        """As probabilidades para todas as classes devem somar 1.0."""
        amostra = pd.DataFrame([{
            'ph': 7.0, 'Hardness': 200.0, 'Solids': 20000.0,
            'Chloramines': 7.0, 'Sulfate': 350.0, 'Conductivity': 500.0,
            'Organic_carbon': 10.0, 'Trihalomethanes': 80.0, 'Turbidity': 3.0
        }])
        proba = modelo.predict_proba(amostra)
        soma  = proba[0].sum()
        assert abs(soma - 1.0) < 1e-6, \
            f"Probabilidades devem somar 1.0, obtido: {soma}"

    def test_probabilidades_formato(self, modelo):
        """predict_proba deve retornar shape (n_amostras, 2) para classificação binária."""
        amostra = pd.DataFrame([{
            'ph': 7.0, 'Hardness': 200.0, 'Solids': 20000.0,
            'Chloramines': 7.0, 'Sulfate': 350.0, 'Conductivity': 500.0,
            'Organic_carbon': 10.0, 'Trihalomethanes': 80.0, 'Turbidity': 3.0
        }])
        proba = modelo.predict_proba(amostra)
        assert proba.shape == (1, 2), \
            f"Shape esperado (1, 2), obtido: {proba.shape}"

    def test_aceita_valores_nan(self, modelo):
        """O pipeline deve tratar NaN (imputação) sem lançar exceções."""
        amostra = pd.DataFrame([{
            'ph': np.nan, 'Hardness': 200.0, 'Solids': 20000.0,
            'Chloramines': np.nan, 'Sulfate': np.nan, 'Conductivity': 500.0,
            'Organic_carbon': 10.0, 'Trihalomethanes': 80.0, 'Turbidity': 3.0
        }])
        try:
            predicao = modelo.predict(amostra)
            assert predicao[0] in [0, 1]
        except Exception as e:
            pytest.fail(f"O modelo não deveria falhar com NaN: {e}")


# ── Testes de desempenho (métricas) ──────────────────────────────────────────

class TestDesempenhoModelo:
    """
    Verifica se o modelo atende aos requisitos mínimos de desempenho.
    Estes testes são os guardiões da qualidade do modelo em produção.
    """

    def test_acuracia_minima(self, predicoes):
        """Acurácia deve ser >= {MIN_ACCURACY:.0%}."""
        y_test, y_pred, _ = predicoes
        acc = accuracy_score(y_test, y_pred)
        assert acc >= MIN_ACCURACY, (
            f"Acurácia insuficiente: {acc:.4f} "
            f"(mínimo exigido: {MIN_ACCURACY:.4f})\n"
            f"O modelo NÃO deve ser implantado."
        )

    def test_f1_score_minimo(self, predicoes):
        """F1-Score deve ser >= {MIN_F1_SCORE:.0%}."""
        y_test, y_pred, _ = predicoes
        f1 = f1_score(y_test, y_pred, zero_division=0)
        assert f1 >= MIN_F1_SCORE, (
            f"F1-Score insuficiente: {f1:.4f} "
            f"(mínimo exigido: {MIN_F1_SCORE:.4f})\n"
            f"O modelo NÃO deve ser implantado."
        )

    def test_roc_auc_minimo(self, predicoes):
        """ROC-AUC deve ser >= {MIN_ROC_AUC:.0%}."""
        y_test, y_pred, y_prob = predicoes
        roc = roc_auc_score(y_test, y_prob)
        assert roc >= MIN_ROC_AUC, (
            f"ROC-AUC insuficiente: {roc:.4f} "
            f"(mínimo exigido: {MIN_ROC_AUC:.4f})\n"
            f"O modelo NÃO deve ser implantado."
        )

    def test_precisao_minima(self, predicoes):
        """Precisão deve ser >= {MIN_PRECISION:.0%} (controla falsos positivos)."""
        y_test, y_pred, _ = predicoes
        prec = precision_score(y_test, y_pred, zero_division=0)
        assert prec >= MIN_PRECISION, (
            f"Precisão insuficiente: {prec:.4f} "
            f"(mínimo exigido: {MIN_PRECISION:.4f})\n"
            f"Muitos falsos positivos — água não-potável classificada como potável."
        )

    def test_recall_minimo(self, predicoes):
        """Recall deve ser >= {MIN_RECALL:.0%} (controla falsos negativos)."""
        y_test, y_pred, _ = predicoes
        rec = recall_score(y_test, y_pred, zero_division=0)
        assert rec >= MIN_RECALL, (
            f"Recall insuficiente: {rec:.4f} "
            f"(mínimo exigido: {MIN_RECALL:.4f})\n"
            f"Muitos falsos negativos — água potável não identificada como tal."
        )

    def test_numero_correto_de_predicoes(self, predicoes, dados_teste):
        """O modelo deve retornar uma predição para cada amostra do conjunto de teste."""
        X_test, _ = dados_teste
        y_test, y_pred, _ = predicoes
        assert len(y_pred) == len(X_test), (
            f"Número de predições ({len(y_pred)}) diferente do "
            f"número de amostras ({len(X_test)})"
        )

    def test_modelo_nao_e_trivial(self, predicoes):
        """O modelo não deve predizer sempre a mesma classe (classificador trivial)."""
        _, y_pred, _ = predicoes
        classes_unicas = np.unique(y_pred)
        assert len(classes_unicas) > 1, (
            f"O modelo está predizendo somente a classe {classes_unicas[0]}. "
            "Isso indica um classificador trivial — verifique o treinamento."
        )

    def test_relatorio_completo(self, predicoes):
        """Gera e exibe um relatório de métricas completo (informativo, não falha)."""
        y_test, y_pred, y_prob = predicoes
        print("\n\n📋 RELATÓRIO COMPLETO DE MÉTRICAS DO MODELO")
        print("=" * 55)
        print(f"  Acurácia     : {accuracy_score(y_test, y_pred):.4f}  (mínimo: {MIN_ACCURACY})")
        print(f"  F1-Score     : {f1_score(y_test, y_pred, zero_division=0):.4f}  (mínimo: {MIN_F1_SCORE})")
        print(f"  ROC-AUC      : {roc_auc_score(y_test, y_prob):.4f}  (mínimo: {MIN_ROC_AUC})")
        print(f"  Precisão     : {precision_score(y_test, y_pred, zero_division=0):.4f}  (mínimo: {MIN_PRECISION})")
        print(f"  Recall       : {recall_score(y_test, y_pred, zero_division=0):.4f}  (mínimo: {MIN_RECALL})")
        print("=" * 55)
        assert True  # Este teste sempre passa — apenas informa


# ── Testes de robustez ────────────────────────────────────────────────────────

class TestRobustez:
    """Testa o comportamento do modelo em situações extremas."""

    def test_amostra_com_todos_nan(self, modelo):
        """O pipeline deve lidar com todos os valores ausentes (imputará pela mediana)."""
        amostra = pd.DataFrame([{f: np.nan for f in EXPECTED_FEATURES}])
        try:
            predicao = modelo.predict(amostra)
            assert predicao[0] in [0, 1]
        except Exception as e:
            pytest.fail(f"Falhou com todos NaN: {e}")

    def test_multiplas_amostras(self, modelo):
        """O modelo deve processar múltiplas amostras em lote."""
        amostras = pd.DataFrame([
            {'ph': 7.0, 'Hardness': 200.0, 'Solids': 20000.0, 'Chloramines': 7.0,
             'Sulfate': 350.0, 'Conductivity': 500.0, 'Organic_carbon': 10.0,
             'Trihalomethanes': 80.0, 'Turbidity': 3.0},
            {'ph': 6.5, 'Hardness': 150.0, 'Solids': 15000.0, 'Chloramines': 4.0,
             'Sulfate': 200.0, 'Conductivity': 300.0, 'Organic_carbon': 8.0,
             'Trihalomethanes': 50.0, 'Turbidity': 2.0},
            {'ph': 9.0, 'Hardness': 280.0, 'Solids': 45000.0, 'Chloramines': 12.0,
             'Sulfate': 450.0, 'Conductivity': 700.0, 'Organic_carbon': 25.0,
             'Trihalomethanes': 110.0, 'Turbidity': 6.0},
        ])
        predicoes = modelo.predict(amostras)
        assert len(predicoes) == 3, "Deve retornar 3 predições para 3 amostras"
        assert all(p in [0, 1] for p in predicoes), "Todas as predições devem ser 0 ou 1"
