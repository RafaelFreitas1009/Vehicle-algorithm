# Coupon Analysis — Random Forest & Explainable ML

Este projeto é uma **aplicação web interativa de Machine Learning** desenvolvida com foco didático e de negócio.
O sistema utiliza algoritmos de *Ensemble Learning* para prever se um motorista tende ou não a **aceitar um cupom de desconto**, considerando o contexto da viagem e características do usuário.

Além da modelagem preditiva, o projeto se destaca por oferecer:

- Visualizações claras e dinâmicas
- Comparação entre modelos
- Otimização automática
- **Explicações inteligentes em linguagem de negócio**, eliminando a necessidade de interpretar métricas técnicas complexas

---

## 🧠 Visão Geral do Sistema

O fluxo da aplicação segue o modelo abaixo:

1. **Pré-processamento dos dados**
2. **Treinamento e avaliação de modelos**
3. **Geração automática de gráficos**
4. **Interpretação inteligente dos resultados**
5. **Exibição via dashboard web**

Tudo isso é executado sob demanda, a partir das escolhas feitas pelo usuário na interface.

---

## 🚀 Tecnologias Utilizadas

- **Python 3**
- **Flask** (backend web)
- **Scikit-learn**
- **Pandas / NumPy**
- **Matplotlib / Seaborn**
- **HTML + CSS + JavaScript**
- **Arquitetura modular (MVC-like)**

---

## 📦 Estrutura do Projeto


```
PYTHON-RANDOMFOREST/
│
├── assets/
│   └── in-vehicle-coupon-recommendation.csv
│
├── data/
│   └── processor.py
│
├── models/
│   ├── pycache/
│   ├── coupon_model.py        # Treinamento, avaliação e otimização dos modelos
│   ├── explainer.py           # Geração de explicações em linguagem de negócio
│   ├── metrics.py             # Métricas de avaliação (accuracy, precision, recall, etc.)
│   └── plots.py               # Geração dos gráficos salvos em arquivo
│
├── web/
│   ├── static/
│   │   ├── css/               # Estilos da interface
│   │   ├── js/                # JavaScript (interações e experiência do usuário)
│   │   └── plots/             # Gráficos gerados dinamicamente pelo backend
│   │
│   └── templates/
│       ├── base.html          # Template base da aplicação
│       └── index.html         # Página principal (dashboard)
│
├── app.py                     # Backend Flask (rotas e orquestração)
├── main.py                    # Arquivo auxiliar de execução (opcional)
├── requirements.txt           # Dependências do projeto
├── LICENSE
└── README.md
```

## ▶️ Executando o Projeto

```
python main.py
```

Em seguida, acesse no navegador:

```
http://localhost:5000
```


## 🎛️ Funcionalidades Principais

### 🔹 Interface Web Interativa

* Filtros dinâmicos (CoffeeHouse, Destination)
* Seleção de eixos X, Y e Z
* Execução sob demanda sem reiniciar o sistema

### 🔹 Modelos de Machine Learning

* **Random Forest Classifier**
* **Extra Trees Classifier**
* Comparação direta entre modelos
* Opção de **Otimização de Hiperparâmetros (GridSearch)**

### 🔹 Visualizações Geradas Automaticamente

* Fronteira de decisão (2D)
* Matriz de confusão
* Importância das features
* Distribuição de probabilidades
* Distribuição 3D (opcional)
* Comparação de desempenho entre modelos

Os gráficos são gerados como arquivos e servidos diretamente pelo frontend.

---

## 🧠 Interpretação Inteligente dos Resultados

Um dos diferenciais do projeto é o  **módulo de explicação automática** , localizado em:

```
models/explainer.py
```

Esse módulo converte métricas técnicas em  **texto compreensível para tomada de decisão** , explicando:

* O que foi analisado
* Como o modelo chegou à conclusão
* Quais fatores mais influenciaram
* O nível de confiabilidade
* Impacto prático no negócio

### 🔁 Comportamento Adaptativo

Quando o usuário ativa a opção  **“Otimizar hiperparâmetros”** , o sistema:

* Detecta automaticamente a otimização
* Ajusta o texto explicativo
* Informa se houve ganho real de performance ou estabilidade
* Traduz o impacto técnico em linguagem estratégica

---

## 📊 Documentação Técnica do Processo

### 🔹 1. Pré-processamento — Label Encoding

Variáveis categóricas são convertidas para valores numéricos usando `LabelEncoder`.

**Por quê?**

* Modelos baseados em árvores exigem dados numéricos
* Permite splits eficientes durante o treinamento

---

### 🌳 2. Random Forest Classifier

Modelo principal do sistema.

**Características**

* Ensemble de múltiplas árvores
* Votação majoritária
* Alta robustez a ruídos
* Boa generalização

---

### 🌲 3. Extra Trees Classifier

Modelo alternativo para comparação.

**Diferencial**

* Maior aleatoriedade nos splits
* Útil para avaliar estabilidade e variância

---

### 🔁 4. Validação Cruzada

Utiliza  **K-Fold Cross-Validation (k=5)** .

**Benefícios**

* Reduz viés
* Mede estabilidade real do modelo

---

### ⚙️ 5. GridSearchCV — Otimização

Busca automática pelos melhores hiperparâmetros.

**Impacto**

* Pode melhorar performance
* Ou confirmar que o modelo base já está bem ajustado
* Sempre explicado em linguagem de negócio no frontend

---

## 🎯 Objetivo Didático e Prático

Este projeto foi construído para:

* Demonstrar Machine Learning de forma visual e compreensível
* Conectar modelos estatísticos à tomada de decisão real
* Eliminar a dependência de interpretação técnica por parte do usuário final

---

## 👤 Autor

* **Rafael Freitas**
  * GitHub: [@rafaelfreitas1009](https://github.com/rafaelfreitas1009)

---

## 📄 Licença

MIT License

[https://choosealicense.com/licenses/mit/](https://choosealicense.com/licenses/mit/)

---

## 📚 Referências

* Dataset: In-Vehicle Coupon Recommendation

  [https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation](https://archive.ics.uci.edu/dataset/603/in+vehicle+coupon+recommendation)
* Scikit-learn Documentation

  [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
