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

# Separação dos dados em treino e validação
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=1)

# Exibição dos tamanhos dos conjuntos de dados
print(f"Samples in training set: {x_train.shape[0]}")
print(f"Samples in validation set: {x_val.shape[0]}")

# Avaliação de diferentes graus polinomiais
train_MSEs, val_MSEs = [], []
degrees = range(1, 6)

for degree in degrees:
    # Criar pipeline para cada grau polinomial
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),  
        ('scaler', StandardScaler()),  
        ('model', LinearRegression())  
    ])
    
    # Ajuste e predição
    pipe.fit(x_train, y_train)
    y_train_pred = pipe.predict(x_train)
    y_val_pred = pipe.predict(x_val)
    
    # Cálculo dos erros
    train_MSEs.append(mean_squared_error(y_train, y_train_pred))
    val_MSEs.append(mean_squared_error(y_val, y_val_pred))

# Criar e exibir gráfico da curva de validação
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_MSEs, 'bo-', label='Training MSE')
plt.plot(degrees, val_MSEs, 'go-', label='Validation MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Validation Curve')
plt.legend()
plt.grid(True)
plt.show()
