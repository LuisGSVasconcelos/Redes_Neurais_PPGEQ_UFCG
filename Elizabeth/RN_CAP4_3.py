import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Ridge
from sklearn.feature_selection import SequentialFeatureSelector

# Carregar os dados
VSdata = np.loadtxt('VSdata.csv', delimiter=',')

# Separar variável alvo (y) e variáveis preditoras (X)
y = VSdata[:, 0]
X = VSdata[:, 1:]

# Normalizar os dados
xscaler = StandardScaler()
X_scaled = xscaler.fit_transform(X)

yscaler = StandardScaler()
print("Antes da normalização:", y[:5])  # Exibe os primeiros valores
y_scaled = yscaler.fit_transform(y[:, None]).ravel()  # ravel() transforma de (n,1) para (n,)
print("Depois da normalização:", y_scaled[:5])  # Exibe os valores transformados

# -------------------------------------------------
# Parte 1: Seleção de Variáveis com LassoCV

# Ajusta o modelo LassoCV com validação cruzada (5 folds)
lasso_model = LassoCV(cv=5).fit(X_scaled, y_scaled)

# Obter os coeficientes absolutos das variáveis e identificar as mais relevantes
top_k_inputs_lasso = np.argsort(np.abs(lasso_model.coef_))[-10:][::-1]  # Top 10 coeficientes mais importantes
selected_features_lasso = top_k_inputs_lasso + 1  # Ajuste para refletir os números originais das colunas

# Exibir as variáveis selecionadas pelo LassoCV
print("Variáveis Selecionadas pelo LassoCV:", selected_features_lasso)

# -------------------------------------------------
# Parte 2: Seleção de Variáveis com Sequential Feature Selector (SFS) - Backward

# Número de variáveis a selecionar
k = 10

# Usando Ridge Regression como modelo para a seleção de variáveis
model_ridge = Ridge(alpha=1.0)

# Realizar a seleção de variáveis com SFS (Backward)
BSFS = SequentialFeatureSelector(model_ridge, n_features_to_select=k, direction='backward', cv=5)
BSFS.fit(X_scaled, y_scaled)

# Obter índices das variáveis selecionadas pelo SFS
selected_indices_sfs = BSFS.get_support(indices=True)
selected_features_sfs = selected_indices_sfs + 1  # Ajuste para refletir os números originais das colunas

# Exibir as variáveis selecionadas pelo SFS
print("Variáveis Selecionadas pelo SFS (Backward):", selected_features_sfs)

# -------------------------------------------------
# Visualização das variáveis selecionadas pelo LassoCV
plt.figure(figsize=(10, 6))
plt.bar(range(k), np.arange(k)[::-1] + 1, tick_label=selected_features_lasso, color='royalblue')
plt.xlabel("Variáveis Selecionadas - LassoCV")
plt.ylabel("Ranking de Importância")
plt.title("Top 10 Variáveis Selecionadas pelo LassoCV")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Visualização das variáveis selecionadas pelo SFS
plt.figure(figsize=(10, 6))
plt.bar(range(k), np.arange(k)[::-1] + 1, tick_label=selected_features_sfs, color='darkorange')
plt.xlabel("Variáveis Selecionadas - SFS")
plt.ylabel("Ranking de Importância")
plt.title("Top 10 Variáveis Selecionadas pelo SFS (Backward)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

