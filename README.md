# 🚢 Previsão de Sobrevivência no Titanic com XGBoost

Projeto de Machine Learning desenvolvido com o objetivo de prever a sobrevivência de passageiros do Titanic a partir de variáveis demográficas, sociais e estruturais do conjunto de dados disponibilizado na competição clássica do Kaggle.

O trabalho foi além da construção de um modelo único. Foram desenvolvidas, comparadas e refinadas diferentes versões de modelos preditivos com XGBoost, combinando análise exploratória, engenharia de atributos, seleção de variáveis, validação cruzada, interpretação de importância das features e ajuste de hiperparâmetros.

## 📌 Objetivo

Construir e otimizar modelos de classificação para prever a variável **Survived**, comparando diferentes estratégias de preparação de dados e seleção de atributos, e avaliar o desempenho final por meio de submissões no Kaggle.

## 🧠 Abordagem do projeto

O projeto foi estruturado em etapas:

- análise exploratória e limpeza dos dados
- tratamento de valores ausentes
- engenharia de atributos
- codificação de variáveis categóricas
- construção de modelos com XGBoost
- comparação entre diferentes conjuntos de variáveis
- balanceamento de classes com `scale_pos_weight` e `SMOTE`
- análise de importância com `feature_importances_` e `SHAP`
- seleção de atributos com `Sequential Feature Selector`
- otimização de hiperparâmetros com `RandomizedSearchCV` e `GridSearchCV`
- geração de arquivos de submissão para o Kaggle

## 📂 Base de dados

Foram utilizadas as bases públicas da competição Titanic do Kaggle:

- `train.csv`
- `test.csv`
- `gender_submission.csv`

A variável alvo está presente apenas na base de treino:

- **Survived**: indica se o passageiro sobreviveu (`1`) ou não (`0`)

## 🔍 Principais etapas do pré-processamento

Durante a preparação dos dados, foram realizadas transformações importantes para melhorar a capacidade preditiva do modelo.

### Tratamento e limpeza
- remoção de `PassengerId` por não agregar valor preditivo
- tratamento de valores ausentes em variáveis como `Embarked`
- verificação de consistência de tipos e categorias

### Engenharia de atributos
Foram criadas ou transformadas variáveis para capturar melhor padrões relevantes:

- extração e agrupamento de títulos da variável `Name`
  - Civil
  - Militar
  - Religioso
  - Nobreza
- criação da variável **Sozinho**
- criação da variável **Group**
- criação da variável **Has_cabin**
- criação da variável **Qtd_cabins**
- extração da **Ala** da cabine
- codificação de variáveis categóricas com `OneHotEncoder`

### Análise de correlação
Foi construída uma matriz de correlação para investigar relações lineares entre as features e orientar testes com diferentes subconjuntos de variáveis.

## 🤖 Modelos avaliados

Ao longo do projeto, foram comparadas diferentes versões do XGBoost:

### 1. Modelo completo
Modelo com o conjunto mais amplo de variáveis após o pré-processamento.

### 2. Modelo `unique`
Versão reduzida com remoção de variáveis redundantes ou menos úteis, como atributos derivados de `Name`, `Embarked`, `Group`, `Has_cabin`, `Qtd_cabins` e `Ala_U`.

### 3. Modelo com `scale_pos_weight`
Versão ajustada para lidar com o desbalanceamento entre sobreviventes e não sobreviventes.

### 4. Modelo com `SMOTE`
Aplicação de oversampling na base de treino para balanceamento das classes.

### 5. Modelo `red`
Versão reduzida a partir da remoção de variáveis com importância nula no modelo.

### 6. Modelo `imp`
Modelo construído com subconjunto de variáveis selecionadas com base na importância preditiva.

### 7. Modelo `sfs`
Modelo construído a partir do `Sequential Feature Selector`, buscando o melhor subconjunto de atributos com foco em `f1_macro`.

## 📈 Interpretação das variáveis

A análise de importância indicou que a capacidade preditiva do modelo ficou concentrada principalmente em poucas variáveis, com destaque para:

- `Sex_female`
- `Pclass`
- `Fare`
- `Age`

Além da importância nativa do XGBoost, também foi utilizada análise com **SHAP**, permitindo uma interpretação mais robusta da contribuição das variáveis para as previsões.

As duas abordagens convergiram na identificação de atributos sem contribuição relevante, especialmente algumas variáveis ligadas às alas da cabine, o que orientou a construção de versões reduzidas do modelo.

## ⚙️ Otimização

Após a comparação inicial entre os modelos, os melhores candidatos passaram por ajuste de hiperparâmetros em duas etapas:

- `RandomizedSearchCV`
- `GridSearchCV`

A métrica principal utilizada no processo de busca foi **f1_macro**, para considerar de forma mais equilibrada o desempenho entre as classes.

## 📊 Resultados

### Modelos não otimizados
Os modelos iniciais apresentaram desempenhos relativamente próximos:

- **74,40%** de acerto: modelo completo, `unique` e `red`
- **74,16%** de acerto: modelo por importância e modelo com `SMOTE`
- **73,21%** de acerto: modelo com variáveis padronizadas
- **72,97%** de acerto: modelo com `Sequential Feature Selector`

### Modelos otimizados
Após o ajuste de hiperparâmetros, os melhores resultados foram:

- **76,31%** de acerto: `modelo_red`
- **76,08%** de acerto: `modelo_unique`
- **75,12%** de acerto: `modelo_sfs`
- **74,40%** de acerto: `modelo_imp`

## 🏆 Conclusão

Os resultados mostraram que, neste projeto, estratégias mais simples e bem orientadas de seleção de variáveis foram mais eficazes do que abordagens mais custosas.

A combinação entre:

- análise de correlação
- remoção de variáveis redundantes
- exclusão de atributos com importância nula

foi mais eficiente do que métodos mais pesados, como o `Sequential Feature Selector`, tanto em desempenho quanto em custo computacional.

O melhor resultado geral foi obtido pelo **modelo reduzido (`modelo_red`)**, sugerindo que a simplificação do conjunto de atributos ajudou a reduzir ruído e melhorar a generalização.

## 🛠️ Tecnologias e bibliotecas utilizadas

- Python
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
