"""
Backend Flask — Sistema de Predição de Potabilidade da Água
Engenharia de Software para Sistemas Inteligentes

Responsabilidade: Carregar o modelo de ML treinado e expor uma API REST
para que o frontend possa enviar dados e receber predições.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Permite requisições do frontend (Cross-Origin Resource Sharing)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_potabilidade.pkl')

FEATURE_NAMES = [
    'ph', 'Hardness', 'Solids', 'Chloramines',
    'Sulfate', 'Conductivity', 'Organic_carbon',
    'Trihalomethanes', 'Turbidity'
]

FEATURE_INFO = {
    'ph'              : {'label': 'pH',                        'unit': '0–14',    'min': 0,      'max': 14,       'step': 0.01},
    'Hardness'        : {'label': 'Dureza',                    'unit': 'mg/L',    'min': 47,     'max': 324,      'step': 0.01},
    'Solids'          : {'label': 'Sólidos Totais Dissolvidos','unit': 'ppm',     'min': 320,    'max': 61227,    'step': 0.01},
    'Chloramines'     : {'label': 'Cloraminas',                'unit': 'ppm',     'min': 0.35,   'max': 13.13,    'step': 0.01},
    'Sulfate'         : {'label': 'Sulfato',                   'unit': 'mg/L',    'min': 129,    'max': 481,      'step': 0.01},
    'Conductivity'    : {'label': 'Condutividade',             'unit': 'μS/cm',   'min': 181,    'max': 754,      'step': 0.01},
    'Organic_carbon'  : {'label': 'Carbono Orgânico',          'unit': 'ppm',     'min': 2.2,    'max': 28.3,     'step': 0.01},
    'Trihalomethanes' : {'label': 'Trihalometanos',            'unit': 'μg/L',    'min': 0.74,   'max': 124,      'step': 0.01},
    'Turbidity'       : {'label': 'Turbidez',                  'unit': 'NTU',     'min': 1.45,   'max': 6.74,     'step': 0.001},
}

CLASS_LABELS = {
    0: 'Não Potável',
    1: 'Potável'
}

modelo = None

def load_model():
    """Carrega o modelo de ML ao iniciar a aplicação."""
    global modelo
    if not os.path.exists(MODEL_PATH):
        logger.error(f'Arquivo de modelo não encontrado: {MODEL_PATH}')
        raise FileNotFoundError(
            f'Modelo não encontrado em: {MODEL_PATH}\n'
            'Execute o notebook para gerar o arquivo "modelo_potabilidade.pkl".'
        )
    modelo = joblib.load(MODEL_PATH)
    logger.info(f'Modelo carregado com sucesso: {MODEL_PATH}')
    logger.info(f'Tipo do modelo: {type(modelo).__name__}')

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde da API."""
    return jsonify({
        'status'      : 'ok',
        'model_loaded': modelo is not None,
        'model_path'  : MODEL_PATH
    })


@app.route('/features', methods=['GET'])
def get_features():
    """Retorna as features esperadas pelo modelo com metadados para o frontend."""
    return jsonify({
        'features'     : FEATURE_NAMES,
        'feature_info' : FEATURE_INFO,
        'class_labels' : CLASS_LABELS
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe os dados de entrada via JSON, realiza a predição e retorna o resultado.

    Body esperado (JSON):
    {
        "ph": 7.08,
        "Hardness": 204.89,
        "Solids": 20791.32,
        "Chloramines": 7.30,
        "Sulfate": 368.52,
        "Conductivity": 564.31,
        "Organic_carbon": 10.38,
        "Trihalomethanes": 86.99,
        "Turbidity": 2.96
    }

    Retorno:
    {
        "prediction": 0,
        "label": "Não Potável",
        "probability": {
            "nao_potavel": 0.7234,
            "potavel": 0.2766
        }
    }
    """
    if modelo is None:
        return jsonify({'error': 'Modelo não carregado. Verifique os logs do servidor.'}), 503

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({'error': 'Requisição inválida. Envie um JSON no corpo da requisição.'}), 400

    # Validação: verificar se todos os campos estão presentes
    campos_ausentes = [f for f in FEATURE_NAMES if f not in data]
    if campos_ausentes:
        return jsonify({
            'error'           : 'Campos obrigatórios ausentes.',
            'campos_ausentes' : campos_ausentes,
            'campos_esperados': FEATURE_NAMES
        }), 400

    try:
        # Montar DataFrame com os dados de entrada (o pipeline trata NaN automaticamente)
        input_data = {}
        for feature in FEATURE_NAMES:
            valor = data[feature]
            # Aceitar string vazia ou None como NaN (o pipeline irá imputar)
            if valor == '' or valor is None:
                input_data[feature] = np.nan
            else:
                input_data[feature] = float(valor)

        df_input = pd.DataFrame([input_data])
        logger.info(f'Predição solicitada para: {input_data}')

        # Predição
        predicao    = int(modelo.predict(df_input)[0])
        probabilidades = modelo.predict_proba(df_input)[0]

        resultado = {
            'prediction': predicao,
            'label'     : CLASS_LABELS[predicao],
            'probability': {
                'nao_potavel': round(float(probabilidades[0]), 4),
                'potavel'    : round(float(probabilidades[1]), 4)
            }
        }

        logger.info(f'Resultado: {resultado}')
        return jsonify(resultado)

    except ValueError as e:
        return jsonify({'error': f'Valor inválido: {str(e)}'}), 400
    except Exception as e:
        logger.error(f'Erro na predição: {str(e)}')
        return jsonify({'error': f'Erro interno do servidor: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
