import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler

# Carregar os dados
VSdata = np.loadtxt('VSdata.csv', delimiter=',')

# Separar variável alvo (y) e variáveis preditoras (X)
y = VSdata[:, 0]
X = VSdata[:, 1:]

# Normalizar os dados para melhor análise estatística
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Seleção das 10 melhores variáveis com f_regression
k = 10  # Número de variáveis a selecionar
VSmodel = SelectKBest(f_regression, k=k).fit(X_scaled, y)

# Obter escores de importância das variáveis
input_scores = VSmodel.scores_

# Encontrar os 10 melhores preditores
top_k_indices = np.argsort(input_scores)[::-1][:k]
top_k_inputs = top_k_indices + 1  # Ajustando os índices para corresponder às colunas originais

# Reduzir X para apenas as variáveis selecionadas
X_relevant = VSmodel.transform(X_scaled)

# Plotando a importância das variáveis
plt.figure(figsize=(10, 6))
plt.bar(range(k), input_scores[top_k_indices], tick_label=top_k_inputs, color='royalblue')
plt.xlabel("Variáveis")
plt.ylabel("Escores de Relevância")
plt.title("Importância das 10 Principais Variáveis")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


