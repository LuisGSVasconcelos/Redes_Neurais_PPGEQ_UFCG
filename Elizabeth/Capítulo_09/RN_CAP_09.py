# ============================================
# Importação de pacotes necessários
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree, datasets
from sklearn.ensemble import RandomForestRegressor, BaggingClassifier
from sklearn.model_selection import train_test_split

import xgboost

# ============================================
# PARTE 1: Dados simulados e Árvore de Decisão
# ============================================

# Geração de dados
x = np.linspace(-1, 1, 50).reshape(-1, 1)  # 50 pontos de -1 a 1
y = x**2 + 0.25 + np.random.normal(0, 0.15, (50, 1))  # função quadrática com ruído

# Modelo de Árvore de Decisão com profundidade máxima de 3
modelo_arvore = tree.DecisionTreeRegressor(max_depth=3).fit(x, y)
y_pred_arvore = modelo_arvore.predict(x)

# Visualização da árvore
plt.figure(figsize=(12, 6))
tree.plot_tree(modelo_arvore, feature_names=['x'], filled=True, rounded=True)
plt.title("Árvore de Decisão")
plt.show()

# ============================================
# PARTE 2: Regressão com Floresta Aleatória
# ============================================

# Treinamento do modelo com 20 árvores
modelo_rf = RandomForestRegressor(n_estimators=20, random_state=42).fit(x, y)
y_pred_rf = modelo_rf.predict(x)

# Predições de duas árvores específicas da floresta
y_pred_arvore5 = modelo_rf.estimators_[5].predict(x)
y_pred_arvore15 = modelo_rf.estimators_[15].predict(x)

# ============================================
# PARTE 3: Análise de resistência do cimento
# ============================================

# Leitura do arquivo de dados
dados_cimento = np.loadtxt('cement_strength.txt', delimiter=',', skiprows=1)
X_cimento = dados_cimento[:, :-1]
y_cimento = dados_cimento[:, -1]

# Divisão em dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_cimento, y_cimento, test_size=0.33, random_state=100)

# Treinamento do modelo RF com validação interna (OOB)
modelo_cimento = RandomForestRegressor(
    n_estimators=200,
    max_features=3,
    oob_score=True,
    random_state=1
).fit(X_train, y_train)

# Impressão da pontuação OOB (out-of-bag)
print('Pontuação OOB:', modelo_cimento.oob_score_)

# Importância das variáveis
nomes_variaveis = ['cimento', 'escória', 'cinza volante', 'água', 'superplastificante',
                   'agregado grosso', 'agregado fino', 'idade']

importancias = modelo_cimento.feature_importances_

# Visualização
plt.figure(figsize=(10, 6))
plt.barh(nomes_variaveis, importancias)
plt.xlabel('Importância das variáveis')
plt.title('Importância das variáveis na predição da resistência do cimento')
plt.show()

# ============================================
# PARTE 4: Classificador Bagging com dados simulados
# ============================================

# Geração de dados "moons" (lua crescente) com ruído
X_bag, y_bag = datasets.make_moons(n_samples=200, noise=0.3, random_state=10)

# Modelo Bagging com 500 classificadores (amostras de 50)
modelo_bagging = BaggingClassifier(
    n_estimators=500,
    max_samples=50,
    random_state=100
).fit(X_bag, y_bag)

# ============================================
# PARTE 5: Tratamento de dados da água
# ============================================

# Leitura dos dados, tratando valores ausentes
dados_agua = pd.read_csv('water-treatment.data', header=None, na_values="?")
X_raw = dados_agua.iloc[:, 1:23]
y_raw = dados_agua.iloc[:, 29]
dados = pd.concat([X_raw, y_raw], axis=1)

# Verificação de valores ausentes
print(dados.info())

# Remoção de linhas com dados ausentes
dados.dropna(inplace=True)
print('Número de amostras restantes:', dados.shape[0])

# Separação das variáveis de entrada e saída
X_agua = dados.iloc[:, :-1]
y_agua = dados.iloc[:, -1]

# Divisão dos dados: treino (fit), validação e teste
X_train, X_test, y_train, y_test = train_test_split(X_agua, y_agua, test_size=0.2, random_state=100)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=100)

# ============================================
# PARTE 6: Modelo XGBoost para regressão
# ============================================

modelo_xgb = xgboost.XGBRegressor(
    max_depth=3,
    learning_rate=0.1,
    random_state=100
)

# Treinamento com parada antecipada (early stopping)
modelo_xgb.fit(X_fit, y_fit)

#OBS: Não consegui gerar a parada antecipada porque não consegui atualizar o xgboost. 