# Importação das bibliotecas
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Definição dos dados
x = np.array([[1], [2], [3], [4], [5]])  # Exemplo de dados para ajuste
y = np.array([2.1, 2.9, 3.8, 5.1, 5.9])  # Valores reais

# Criação do pipeline para ajuste quadrático via modelo linear
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),  # Geração de características polinomiais
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('model', LinearRegression())  # Modelo de regressão linear
])

# Separação dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Exibição dos tamanhos dos conjuntos de dados
print(f"Samples in training set: {x_train.shape[0]}")
print(f"Samples in test set: {x_test.shape[0]}")

# Ajuste do pipeline com os dados de treino
pipe.fit(x_train, y_train)

# Previsões para treino e teste
y_pred_train = pipe.predict(x_train)
y_pred_test = pipe.predict(x_test)

# Cálculo das métricas
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_train, y_pred_train)

# Exibição dos resultados
print(f"Coeficientes do modelo: {pipe.named_steps['model'].coef_}")
print(f"Intercepto: {pipe.named_steps['model'].intercept_}")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Fitting metric (R²) = {r2:.4f}")

