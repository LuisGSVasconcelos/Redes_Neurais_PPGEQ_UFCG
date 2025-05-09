# Importação de bibliotecas
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Dados originais
X = np.array([
    [1000, 0.01, 300],
    [1200, 0.06, 350],
    [1500, 0.10, 320]
])

# ============================

# Padronização (Standardization)
standard_scaler = StandardScaler() # calcula média e desvio padrão por coluna
X_standardized = standard_scaler.fit_transform(X)  # aplica a transformação

# Formatar a saída com 4 casas decimais
np.set_printoptions(precision=4, suppress=True)

# Exibir os dados
print("Original Data:\n", X)
print("\nStandardized Data (mean=0, variance=1):\n", X_standardized)

# ============================

# Normalização (Normalization)
minmax_scaler = MinMaxScaler() # cria o objeto
X_minmax = minmax_scaler.fit_transform(X) # aplica a transformação

# Formatar a saída com 2 casas decimais
np.set_printoptions(precision=2, suppress=True)

# Exibir os dados
print("\nMin-Max Scaled Data (range 0 to 1):\n", X_minmax)
