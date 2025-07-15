#########################
# Importar as bibliotecas 
#########################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor # Implementa o modelo de regressão com RF
# Modelo mais robusto e preciso, evita overfitting típico da DT
from sklearn.metrics import mean_squared_error as mse # Calcula o erro quadrático médio
# Quanto menor MSE, melhor o ajuste


########################
# Gerar dados sintéticos 
########################
np.random.seed()
print(np.random.rand(10))
x = np.linspace(-1, 1, 50).reshape(-1, 1)
#print(x)
y = x**2 + 0.25 + np.random.normal(0, 0.15, size=(50, 1)) 
y = y.ravel()
#print(y)

##########################
# Gráfico dos dados brutos
##########################
plt.figure()
plt.scatter(x, y, edgecolor='black', facecolor='darkorange')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Raw Data")
plt.tight_layout()
plt.show()

######################
# Treinar RandomForest
######################
# Cria um modelo de regressão usando RF:
    # Define que a floresta será composta por 20 DT individuais
rf_model = RandomForestRegressor(n_estimators=20).fit(x,y)

#############################
# Predição do modelo completo
#############################
y_pred_rf = rf_model.predict(x) # Usa o modelo de RF treinado para gerar predições
# Cada uma das 20 árvores da floresta faz sua própria predição;
# A previsão final é a média das saídas das árvores para cada ponto de entrada.

#######################################
# Predições de duas árvores individuais
#######################################
# Abaixo os estimadores são uma lista de árvores individuais que compõem a floresta:
    # A contagem inicia do 0, logo são acessadas a 6ª e a 16ª DT;
        # Usa essa árvore específica para gerar previsões individuais com base nas entradas "x".
tree1_pred = rf_model.estimators_[5].predict(x)
tree2_pred = rf_model.estimators_[15].predict(x)
# Serve para analisar o comportamento isolado de um árvore da floresta.

##########################
# Gerar dados de validação
##########################
np.random.seed()
print(np.random.rand(10))
x_val = np.linspace(-1, 1, 50).reshape(-1, 1)
#print(x_val)
y_val = x_val**2 + 0.25 + np.random.normal(0, 0.15, size=(50, 1))
y_val = y_val.ravel()
#print(y_val)

#########################################
# Gráfico dos dados brutos para validação
#########################################
plt.figure()
plt.scatter(x_val, y_val, edgecolor="black", c="darkorange")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Raw Validation Data")
plt.tight_layout()
plt.show()

#############################################
# Avaliar erro com número variável de árvores
#############################################
val_errors = [] # Cria lista vazia para armazenar os resultados MSE obtidos para cada modelo.
trees_range = np.arange(2,250,5) # Gera uma sequência de números de 2 até 249 com incremento de 5.
for n in trees_range: # Inicia um loop para iterar sobre cada valor "n" na sequência de cima.
    temp_model = RandomForestRegressor(n_estimators=n) # Nova instância de RF com "n" árvores.
    temp_model.fit(x, y)
    y_val_pred = temp_model.predict(x_val) # Usa o modelo treinado para prever os valores do conjunto "x_val"
    val_errors.append(mse(y_val, y_val_pred)) # Calcula o MSE entre os valores previstos e os reais.

#####################################
# Plotar os três gráficos lado a lado
#####################################
fig, axs = plt.subplots(1, 3, figsize=(16, 4))

#########################
# (1) Random forest final
#########################
axs[0].scatter(x, y, edgecolor='black', facecolor='darkorange', label='raw data')
axs[0].plot(x, y_pred_rf, color='yellowgreen', label='predictions')
axs[0].set_title("Random Forest regression predictions")
axs[0].legend()

##############################
# (2) Duas árvores do ensemble
##############################
axs[1].scatter(x, y, edgecolor='black', facecolor='darkorange', label='raw data')
axs[1].plot(x, tree1_pred, color='red', alpha=0.5, label='DT6 predictions')
axs[1].plot(x, tree2_pred, color='blue', alpha=0.5, label='DT16 predictions')
axs[1].set_title("Predictions from as couple of constituent Decision Trees")
axs[1].legend()

####################################
# (3) Curva de erro vs nº de árvores
####################################
axs[2].plot(trees_range, val_errors, color='mediumorchid')
axs[2].set_title("Impact of number trees in RF model on validation error")
axs[2].set_xlabel("# of Trees")
axs[2].set_ylabel("Validation MSE")

plt.tight_layout()
plt.show()
