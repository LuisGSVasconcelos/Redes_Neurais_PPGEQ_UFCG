import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Gerar dados sintéticos para regressão
X, y = make_regression(n_samples=300, n_features=1, noise=10, random_state=42)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Treinar o modelo
gbr.fit(X_train, y_train)

# Previsões
y_pred = gbr.predict(X_test)

# Avaliação
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("R²:", r2)

# Criar o modelo Linear para a linha de regressão
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Gerar pontos para a reta
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = linear_model.predict(x_line)

# Plot das previsões
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='navy', label='Raw data', alpha=0.6)
plt.scatter(X_test, y_pred, color='red', label='Predictions', alpha=0.6)

# Plot da linha de regressão (linha entre menor e maior valor de X_test)
# x_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
# y_line = gbr.predict(x_line)
# plt.plot(x_line, y_line, color='green', linewidth=2, label='Regression Line')

# Linha de regressão linear (reta)
plt.plot(x_line, y_line, color='green', linewidth=2, label='Regression Line')

# Mostrar R² no gráfico
plt.text(0.6, 0.5, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Boosting Regression')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
