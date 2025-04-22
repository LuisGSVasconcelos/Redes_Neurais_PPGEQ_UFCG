import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge

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

# Seleção de variáveis com Backward SFS usando Ridge Regression
k = 10  # Número de variáveis a selecionar
model = Ridge(alpha=1.0)  # Regularização para evitar overfitting

BSFS = SequentialFeatureSelector(model, 
                                 n_features_to_select=k, 
                                 direction='backward',
                                 cv=5).fit(X_scaled, y_scaled)

# Obter índices das variáveis selecionadas
selected_indices = BSFS.get_support(indices=True)
selected_features = selected_indices + 1  # Ajuste para refletir os números originais das colunas

# Reduzir X para apenas as variáveis escolhidas
X_relevant = BSFS.transform(X)

# Visualizar as variáveis selecionadas
plt.figure(figsize=(10, 6))
plt.bar(range(k), np.arange(k)[::-1] + 1, tick_label=selected_features, color='royalblue')
plt.xlabel("Variáveis Selecionadas")
plt.ylabel("Ranking de Importância")
plt.title("Top 10 Variáveis Selecionadas - Backward SFS")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print("Rodando o código...")  # Mensagem para saber se a execução começou
# Exibir as variáveis escolhidas no terminal
print(f"Variáveis Selecionadas: {selected_features}")

