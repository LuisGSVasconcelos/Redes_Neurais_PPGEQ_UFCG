# Importação das bibliotecas
import numpy as np
from sklearn.preprocessing import StandardScaler

# Definição dos dados
X = np.array([
    [1000, 0.01, 300],
    [1200, 0.06, 350],
    [1500, 0.1, 320]])

# Instanciação do StandardScaler e aplicação do fit_transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Calcula a média, desvio padrão e transforma os dados

# Exibição dos resultados
print("Dados normalizados:\n", X_scaled)

