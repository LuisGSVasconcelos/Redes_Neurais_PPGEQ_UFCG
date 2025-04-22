# Importação das bibliotecas
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

# Leitura dos dados brutos
data = np.loadtxt('quadratic_raw_data.csv', delimiter=',')
x, y = data[:, 0:1], data[:, 1:2]  # Mantém as dimensões corretas para compatibilidade com scikit-learn

# Geração de características quadráticas
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x)  # 1ª coluna: x, 2ª coluna: x²

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Treinamento do modelo linear
model = LinearRegression()
model.fit(X_scaled, y)

# Previsões
y_predicted = model.predict(X_scaled)

# Exibição de resultados (opcional)
print("Coeficientes do modelo:", model.coef_)
print("Intercepto:", model.intercept_)
print("Previsões:", y_predicted[:5])  # Mostra as primeiras 5 previsões
