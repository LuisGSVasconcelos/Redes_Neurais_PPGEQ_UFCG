# Importação das bibliotecas
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Definição dos dados
x = np.array([[1], [2], [3], [4], [5]])  # Exemplo de dados para ajuste
y = np.array([2.1, 2.9, 3.8, 5.1, 5.9])  # Valores reais

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

# Cálculo e exibição da métrica de ajuste (R²)
r2 = r2_score(y, y_predicted)

# Exibição dos resultados
print("Coeficientes do modelo:", pipe.named_steps['model'].coef_)
print("Intercepto:", pipe.named_steps['model'].intercept_)
print("Previsões:", y_predicted)
print(f"Fitting metric (R²) = {r2:.4f}")
