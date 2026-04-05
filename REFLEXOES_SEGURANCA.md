# 🔒 Reflexões sobre Desenvolvimento de Software Seguro
## Projeto: Sistema de Classificação de Potabilidade da Água

---

## Visão Geral

Este documento reflete sobre como as boas práticas de **Desenvolvimento de Software Seguro** podem — e devem — ser aplicadas ao sistema de predição de potabilidade da água, abrangendo desde a coleta dos dados até a exposição do modelo via API REST.

Embora dados de qualidade de água não sejam dados pessoais no sentido da LGPD, este projeto serve como exercício para pensar segurança em sistemas de IA de ponta a ponta.

---

## 1. Anonimização e Pseudonimização de Dados

### Contexto
Em variantes do problema com dados pessoais — como sistemas que coletam amostras de água de residências vinculadas a moradores — os dados poderiam conter:
- Localização geográfica da coleta (bairro, CEP, coordenadas GPS)
- Identificador do imóvel ou da família
- Data e hora da coleta (pode inferir hábitos)

### Técnicas Aplicáveis

| Técnica | Aplicação no Projeto |
|---|---|
| **Pseudonimização** | Substituir o ID do imóvel ou morador por um token aleatório antes de armazenar ou treinar o modelo |
| **Generalização geográfica** | Em vez de GPS exato, usar zona/bairro (k-anonimato geográfico) |
| **Supressão de atributos** | Remover colunas que não contribuem para a predição mas identificam o indivíduo |
| **Data Masking** | Truncar datas de coleta para mês/ano em vez de dia/hora exatos |
| **Differential Privacy** | Adicionar ruído calibrado às features durante o treinamento para que o modelo não "memorize" amostras individuais |

### Implementação Prática (exemplo em Python)
```python
import hashlib

def pseudonimizar_id(id_original: str, salt: str) -> str:
    """Substitui o identificador real por um hash irreversível."""
    return hashlib.sha256(f"{salt}{id_original}".encode()).hexdigest()[:16]

# Antes de treinar ou armazenar:
df['id_amostra'] = df['id_amostra'].apply(
    lambda x: pseudonimizar_id(x, salt="segredo-da-aplicacao")
)
df.drop(columns=['endereco', 'cpf_responsavel'], inplace=True)
```

---

## 2. Segurança da API (Backend Flask)

### Vulnerabilidades Identificadas e Mitigações

#### 2.1 Injeção de Dados (Input Validation)
O endpoint `/predict` recebe dados do usuário via JSON. Sem validação adequada, um atacante poderia enviar valores maliciosos.

**Vulnerabilidade:**
```python
# INSEGURO — aceita qualquer valor
input_data[feature] = float(data[feature])
```

**Mitigação:**
```python
# SEGURO — valida tipo e faixa
from pydantic import BaseModel, confloat, validator

class AguaInput(BaseModel):
    ph              : confloat(ge=0, le=14)
    Hardness        : confloat(ge=0)
    Solids          : confloat(ge=0)
    Chloramines     : confloat(ge=0)
    Sulfate         : confloat(ge=0)
    Conductivity    : confloat(ge=0)
    Organic_carbon  : confloat(ge=0)
    Trihalomethanes : confloat(ge=0)
    Turbidity       : confloat(ge=0)
```

#### 2.2 Controle de Taxa (Rate Limiting)
Sem rate limiting, a API fica vulnerável a ataques de força bruta ou DDoS.

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address, app=app, default_limits=["100 per hour"])

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    ...
```

#### 2.3 Exposição de Informações Sensíveis
Mensagens de erro detalhadas podem revelar estrutura interna do sistema.

**Vulnerabilidade:**
```python
return jsonify({'error': str(e)}), 500  # Expõe stack trace!
```

**Mitigação:**
```python
logger.error(f"Erro interno: {str(e)}", exc_info=True)  # Log interno apenas
return jsonify({'error': 'Erro interno. Contate o administrador.'}), 500
```

#### 2.4 HTTPS e Headers de Segurança
```python
from flask_talisman import Talisman

Talisman(app,
    content_security_policy={
        'default-src': "'self'",
        'script-src' : "'self'"
    },
    force_https=True
)
```

---

## 3. Segurança do Modelo de Machine Learning (ML Security)

### 3.1 Envenenamento de Dados (Data Poisoning)
Um atacante com acesso ao pipeline de dados poderia inserir amostras manipuladas para degradar o modelo.

**Mitigação:**
- Validar a proveniência dos dados (checksums, assinaturas digitais)
- Monitorar a distribuição estatística dos dados de treino antes de cada re-treinamento
- Usar testes automatizados (PyTest) como portão de qualidade antes de substituir o modelo

```python
# Verificação de integridade do dataset
import hashlib

HASH_ESPERADO = "abc123..."  # Hash calculado sobre o dataset oficial

with open('dataset.csv', 'rb') as f:
    hash_atual = hashlib.sha256(f.read()).hexdigest()

assert hash_atual == HASH_ESPERADO, "Dataset pode ter sido alterado!"
```

### 3.2 Ataques Adversariais (Adversarial Examples)
Pequenas perturbações nos dados de entrada podem enganar o modelo.

**Exemplo:** alterar o pH de 7.0 para 7.001 pode, em alguns modelos mal calibrados, mudar a predição de "Não Potável" para "Potável".

**Mitigação:**
- Validar faixas fisicamente plausíveis dos parâmetros na entrada
- Usar ensemble de modelos (se um discordar, escalar para análise humana)
- Monitorar as distribuições de entrada em produção (data drift)

### 3.3 Extração do Modelo (Model Stealing)
Chamadas repetitivas à API podem ser usadas para reconstituir o modelo.

**Mitigação:**
- Rate limiting rigoroso por IP e por usuário autenticado
- Não expor as probabilidades com alta precisão decimal
- Monitorar padrões suspeitos de chamadas

### 3.4 Integridade do Arquivo do Modelo
```python
import hashlib, joblib

def verificar_integridade_modelo(caminho_modelo: str, hash_esperado: str) -> bool:
    """Verifica se o arquivo .pkl não foi adulterado."""
    with open(caminho_modelo, 'rb') as f:
        hash_atual = hashlib.sha256(f.read()).hexdigest()
    return hash_atual == hash_esperado

# Ao iniciar o servidor:
assert verificar_integridade_modelo(MODEL_PATH, HASH_MODELO_APROVADO), \
    "ALERTA: Arquivo do modelo pode ter sido adulterado!"
```

---

## 4. Conformidade com a LGPD (Lei Geral de Proteção de Dados)

Mesmo que dados de qualidade de água não sejam dados pessoais diretos, em aplicações que coletam dados de cidadãos, a LGPD impõe obrigações importantes:

| Princípio LGPD | Aplicação no Projeto |
|---|---|
| **Finalidade** | O modelo deve ser usado apenas para classificação de potabilidade, não para outros fins |
| **Necessidade** | Coletar apenas os 9 parâmetros necessários, não dados adicionais dos coletores |
| **Transparência** | Informar os usuários sobre como os dados são processados |
| **Segurança** | Implementar controles técnicos (HTTPS, autenticação, logs) |
| **Não-discriminação** | O modelo não deve ser usado para discriminar geograficamente populações |

---

## 5. Autenticação e Autorização

Em um ambiente de produção real, a API deve ser protegida:

```python
from functools import wraps
from flask import request, jsonify
import jwt

SECRET_KEY = os.environ.get('JWT_SECRET')  # Nunca hardcode!

def requer_autenticacao(f):
    @wraps(f)
    def decorador(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Token inválido'}), 401
        return f(*args, **kwargs)
    return decorador

@app.route('/predict', methods=['POST'])
@requer_autenticacao
def predict():
    ...
```

---

## 6. Variáveis de Ambiente e Gestão de Segredos

**Nunca hardcode credenciais ou chaves no código-fonte:**

```python
# INSEGURO — NUNCA faça isso!
SECRET_KEY = "minha-chave-secreta-123"

# SEGURO — use variáveis de ambiente
import os
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    raise ValueError("SECRET_KEY não configurada no ambiente!")
```

Ferramentas recomendadas: **python-dotenv** para desenvolvimento local, **AWS Secrets Manager / HashiCorp Vault** para produção.

---

## 7. Registro e Monitoramento (Logging)

```python
import logging
from datetime import datetime

# Log estruturado para análise de segurança
logger.info({
    'timestamp'    : datetime.utcnow().isoformat(),
    'ip_origem'    : request.remote_addr,
    'endpoint'     : request.path,
    'metodo'       : request.method,
    'user_agent'   : request.headers.get('User-Agent'),
    'resultado'    : predicao,
    'confianca'    : round(max(probabilidades), 4)
})
```

**Importante:** Não logar os dados de entrada completos se eles puderem ser rastreados a indivíduos. Logar apenas metadados da requisição.

---

## 8. Pipeline de DevSecOps

Para um projeto de produção, recomenda-se:

```
Commit → [SAST: Bandit] → [Dependências: Safety] → [Testes: PyTest] → [Build] → Deploy
```

- **Bandit:** análise estática de segurança em código Python
- **Safety:** verifica vulnerabilidades conhecidas nas dependências
- **PyTest:** nossos testes garantem qualidade do modelo antes de qualquer deploy

```bash
# Verificação de segurança das dependências
pip install safety bandit
safety check -r requirements.txt
bandit -r backend/ -ll
pytest tests/ -v
```

---

## Conclusão

A segurança em sistemas de IA não é apenas sobre proteger a API — envolve toda a cadeia: desde a coleta e armazenamento dos dados até o monitoramento contínuo do modelo em produção. As técnicas apresentadas aqui, quando combinadas, constroem um sistema mais robusto, confiável e em conformidade com as regulamentações vigentes.
