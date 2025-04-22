# Importação das bibliotecas necessárias
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error as mse

# Definição dos dados (exemplo)
x = np.array([[1], [2], [3], [4], [5]])  # Exemplo de entrada
y = np.array([2.1, 2.9, 3.8, 5.1, 5.9])  # Valores reais

# Separação dos dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Configuração da validação cruzada K-Fold (k=3)
kfold = KFold(n_splits=3, shuffle=True, random_state=1)

# Inicialização das listas de erro para diferentes graus polinomiais
overall_fit_MSEs = []
overall_val_MSEs = []

max_polyDegree = 5  # Testaremos graus de 1 a 5

for poly_degree in range(1, max_polyDegree + 1):
    # Criar novo pipeline para evitar sobrescrever o objeto original
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    split_fit_MSEs = []
    split_val_MSEs = []

    # Loop sobre as divisões do KFold
    for fit_indices, val_indices in kfold.split(x_train):
        x_fit, y_fit = x_train[fit_indices], y_train[fit_indices]
        x_val, y_val = x_train[val_indices], y_train[val_indices]

        # Treinamento e predição
        pipe.fit(x_fit, y_fit)
        y_pred_fit = pipe.predict(x_fit)
        y_pred_val = pipe.predict(x_val)

        # Cálculo dos erros e armazenamento
        split_fit_MSEs.append(mse(y_fit, y_pred_fit))
        split_val_MSEs.append(mse(y_val, y_pred_val))

    # Armazenamento dos erros médios para esse grau polinomial
    overall_fit_MSEs.append(np.mean(split_fit_MSEs))
    overall_val_MSEs.append(np.mean(split_val_MSEs))

# Exibição dos resultados
print("\nErro médio quadrático (MSE) por grau polinomial:")
for degree, (fit_mse, val_mse) in enumerate(zip(overall_fit_MSEs, overall_val_MSEs), start=1):
    print(f"Grau {degree}: Train MSE = {fit_mse:.4f}, Val MSE = {val_mse:.4f}")

# Obtém os coeficientes do modelo treinado
coef = pipe.named_steps['model'].coef_
intercept = pipe.named_steps['model'].intercept_

# Exibição formatada dos coeficientes
print("\nCoeficientes do modelo (sem intercepto):", coef)
print("Intercepto:", intercept)

# Separação dos dados em treino e teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Busca do melhor hiperparâmetro via GridSearchCV
from sklearn.model_selection import GridSearchCV

# Definição do espaço de busca para o grau do polinômio
param_grid = {'poly__degree': np.arange(1, 6)}

# Configuração do GridSearchCV com validação cruzada (cv=3)
grid_search = GridSearchCV(pipe, param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)

# Treinamento do modelo com GridSearch
grid_search.fit(x_train, y_train)

# Exibição do melhor hiperparâmetro encontrado
print(f"Melhor hiperparâmetro encontrado: {grid_search.best_params_}")
