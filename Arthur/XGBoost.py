#########################
# Importar as bibliotecas 
#########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb # Métodos eficientes de boosting para classificação e regressão.
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

###############################
# Ler dados e separar variáveis
###############################
data_raw = pd.read_csv(r'C:\Users\arthu\PY\Arquivos Python\Redes Neurais\Cap 9\water-treatment.data', header=None, na_values="?")
X_raw = data_raw.iloc[:,1:23] # Extrai as colunas 1 a 22 como variáveis de entrada - final exclusivo em Python.
y_raw = data_raw.iloc[:,29] # Extrai a coluna 29 como a variável alvo - 30ª, pois começa em 0.
# "iloc" é usado quando as colunas não têm nomes ou são selecionadas pela posição.
data = pd.concat([X_raw, y_raw], axis=1) # Junta as variáveis de entrada e a variável alvo em um único DataFrame.
# DataFrame é uma estrutura de dados tabulares, geralmente bidimensional.
print(data.info()) # Exibe informações resumidas sobre o DataFrame

#####################################
# Remover linhas com valores ausentes
#####################################
data.dropna(axis=0, how='any', inplace=True) # Remove todas as linhas com pelo menos um valor ausente.
# Garante que o conjunto de dados não contenha valores faltanres.
print('Number of samples remaining:', data.shape[0])

##########################
# Separar entradas e saída
##########################
X = data.iloc[:,:-1] # Seleciona todas as linhas e todas as colunas menos a última.
y = data.iloc[:,-1] # Seleciona todas as linhas e apenas a última coluna.

######################################
# Dividir em treino, validação e teste
######################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=100)

###########################################
# Converter para DMatrix (interface nativa)
###########################################
# DMatrix é uma estrutura de dados otimizada internamente pelo xgb:
    # Mais eficiente que arrays NumPy ou DataFrames;
    # Suporta operações específicas:
        # Pesos, índices de grupo (ranking) etc.
dtrain = xgb.DMatrix(X_fit, label=y_fit)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

#######################
# Configurar parâmetros
#######################
params = {
    'max_depth': 3, # Limite de profundidade das árvores.
    'eta': 0.1, # Taxa de aprendizado - menor = mais conservador.
    'objective': 'reg:squarederror', # Modelo de regressão MSE
    'eval_metric': 'rmse' # Métrica de avaliação = raiz do erro quadrático médio RMSE
} 

############################
# Treinar com early stopping
############################
# Técnica de regularização utilizada em ML para evitar overfitting:
    # Interrompe o treinamento de um modelo antes que ele alcance a sua convergência final;
    # Monitora o seu desempenho em um conjunto de validação.
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(
    params, # Define parâmetros do modelo.
    dtrain, # Dados de treinamento (DMatrix).
    num_boost_round=1000, # Máximo de 1000 iterações (boosting rounds).
    evals=evals, # Lista de conjuntos monitorados (ex.: treino e validação).
    early_stopping_rounds=2, # Para se a validação não melhorar após 2 rounds.
    verbose_eval=False # Desativa a impressão de mensagens durante o treinamento.
)

##########
# Predição
##########
y_train_pred = model.predict(dtrain)
y_test_pred = model.predict(dtest)

###########
# Avaliação
###########
r2_train = r2_score(y_fit, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f'Accuracy over training data = {r2_train:.3f}')
print(f'Accuracy over test data: = {r2_test:.3f}')

#######################################
# Plot das variáveis de entrada e saída
#######################################
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.plot(X.iloc[:, 0], color='red', linestyle=':', marker='.', linewidth=0.5, markeredgecolor='k', alpha=0.7)
plt.xlabel('Sample #')
plt.ylabel('Input flow to plant')

plt.subplot(1,3,2)
plt.plot(X.iloc[:, 4], color='red', linestyle=':', marker='.', linewidth=0.5, markeredgecolor='k', alpha=0.7)
plt.xlabel('Sample #')
plt.ylabel('Input conductivity to plant')

plt.subplot(1,3,3)
plt.plot(y, color='navy', linestyle=':', marker='.', linewidth=0.5, markeredgecolor='k', alpha=0.7)
plt.xlabel('Sample #')
plt.ylabel('Output Conductivity')

plt.tight_layout()
plt.show()

########################################
# Plot Previsão vs Real (Treino e Teste)
########################################

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.scatter(y_fit, y_train_pred, c='steelblue', edgecolors='k')
plt.plot([y_fit.min(), y_fit.max()], [y_fit.min(), y_fit.max()], 'r-', lw=2)
plt.xlabel('raw training data')
plt.ylabel('prediction')
plt.title(f'Training R² = {r2_train:.3f}')

plt.subplot(1,2,2)
plt.scatter(y_test, y_test_pred, c='steelblue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r-', lw=2)
plt.xlabel('raw test data')
plt.ylabel('prediction')
plt.title(f'Test R² = {r2_test:.3f}')

plt.tight_layout()
plt.show()