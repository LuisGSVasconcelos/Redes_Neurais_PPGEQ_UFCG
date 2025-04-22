# Importação das bibliotecas
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np

# Definição dos dados
x = np.array([[1], [2], [3], [4], [5]])  # Exemplo de dados para ajuste
y = np.array([2.1, 2.9, 3.8, 5.1, 5.9])  # Exemplo de valores alvo

# Criação do pipeline para ajuste quadrático via modelo linear
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Geração de características polinomiais
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('model', LinearRegression())  # Modelo de regressão linear
])

# Ajuste do pipeline aos dados
pipe.fit(x, y)

# Previsões
y_predicted = pipe.predict(x)

# Exibição dos resultados
print("Coeficientes do modelo:", pipe.named_steps['model'].coef_)
print("Intercepto:", pipe.named_steps['model'].intercept_)
print("Previsões:", y_predicted)
