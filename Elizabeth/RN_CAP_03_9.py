# Importação das bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Definição dos dados
x = np.array([[1], [2], [3], [4], [5]])  # Exemplo de dados para ajuste
y = np.array([2.1, 2.9, 3.8, 5.1, 5.9])  # Valores reais

# Divisão dos dados em treino (70%), validação (30% do treino) e teste (20% do total)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Divisão do conjunto de treino em fitting (70%) e validação (30%)
x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)

# Exibição dos tamanhos dos conjuntos
print(f"Number of samples in fitting set: {x_fit.shape[0]}")
print(f"Number of samples in validation set: {x_val.shape[0]}")
print(f"Number of samples in test set: {x_test.shape[0]}")
