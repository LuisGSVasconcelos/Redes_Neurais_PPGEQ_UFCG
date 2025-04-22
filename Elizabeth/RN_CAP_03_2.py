# Importação das bibliotecas
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Definição dos dados
X = np.array([
    [1000, 0.01, 300],
    [1200, 0.06, 350],
    [1500, 0.1, 320]])

# Instanciação do MinMaxScaler e aplicação do fit_transform
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)  # Normaliza os dados no intervalo [0,1]

# Exibição dos resultados
print("Dados normalizados:\n", X_scaled)
