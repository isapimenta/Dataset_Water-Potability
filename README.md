# 💧 Sistema de Predição de Potabilidade da Água

## 📋 Visão Geral

Sistema completo de Machine Learning para classificação de potabilidade da água, integrando:

- **Notebook Colab** com pipeline completo de ML
- **Backend Flask** com modelo embarcado
- **Frontend HTML** para entrada de dados e visualização de resultados
- **Testes PyTest** com thresholds de desempenho
- **Reflexões de Segurança** aplicadas ao problema

**Dataset:** [Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) — 3.276 amostras, 9 features, classificação binária

---

## 🗂️ Estrutura do Projeto

```
projeto-potabilidade/
│
├── 📓 notebook_potabilidade_agua.ipynb    ← Google Colab (itens 1, 2 e 3)
│
├── 🔧 backend/
│   ├── app.py                             ← API Flask (item 4)
│   ├── requirements.txt                   ← Dependências Python
│   └── modelo_potabilidade.pkl            ← Gerado pelo notebook
│
├── 🌐 frontend/
│   └── index.html                         ← Interface web (item 4)
│
├── 🧪 tests/
│   └── test_model.py                      ← Testes PyTest (item 5)
│
└── 🔒 REFLEXOES_SEGURANCA.md              ← Segurança (item 6)
```

---

## 🚀 Como Executar

### Pré-requisito: Gerar o modelo via Notebook

1. Abra `notebook_potabilidade_agua.ipynb` no [Google Colab](https://colab.research.google.com)
2. Execute todas as células (`Runtime > Run all`)
3. Faça o download do arquivo `modelo_potabilidade.pkl` gerado
4. Coloque o arquivo em `backend/modelo_potabilidade.pkl`

---

### Backend (Flask)

```bash
cd backend

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Instalar dependências
pip install -r requirements.txt

# Iniciar o servidor
python app.py
```

O servidor estará disponível em `http://localhost:5000`

**Endpoints disponíveis:**
| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Verificação de saúde |
| GET | `/features` | Lista as features esperadas |
| POST | `/predict` | Realiza predição |

---

### Frontend

Abra diretamente no navegador:
```bash
# Opção 1: Abrir o arquivo HTML diretamente
open frontend/index.html

# Opção 2: Servidor simples com Python
cd frontend
python -m http.server 8080
# Acesse: http://localhost:8080
```

---

### Testes (PyTest)

```bash
# Instalar pytest
pip install pytest requests

# Executar todos os testes
pytest tests/test_model.py -v

# Executar com relatório detalhado
pytest tests/test_model.py -v -s

# Executar apenas testes de desempenho
pytest tests/test_model.py::TestDesempenhoModelo -v
```

---

## 📊 Algoritmos Avaliados

| Algoritmo | Pré-processamento | Hiperparâmetros Otimizados |
|---|---|---|
| **KNN** | StandardScaler | n_neighbors, weights, metric |
| **Árvore de Decisão** | StandardScaler | criterion, max_depth, min_samples |
| **Naive Bayes** | MinMaxScaler | var_smoothing |
| **SVM** | StandardScaler | C, kernel, gamma |

---

## 🧪 Thresholds de Teste

| Métrica | Threshold Mínimo |
|---|---|
| Acurácia | ≥ 60% |
| F1-Score | ≥ 55% |
| ROC-AUC | ≥ 62% |
| Precisão | ≥ 55% |
| Recall | ≥ 50% |

---

## 🔒 Segurança

Veja `REFLEXOES_SEGURANCA.md` para detalhes sobre:
- Anonimização e pseudonimização de dados
- Segurança da API (input validation, rate limiting, HTTPS)
- ML Security (data poisoning, adversarial attacks, model stealing)
- Conformidade com LGPD
- Autenticação JWT e gestão de segredos
- Pipeline DevSecOps
