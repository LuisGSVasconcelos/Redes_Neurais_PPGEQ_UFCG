#########################
# Importar as bibliotecas 
#########################
import pandas as pd # Usado para trabalhar com estruturas de dados tabulares
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression # Utiliza modelo de regressão PLS
# Transforma as variáveis independentes em um conjunto menor de variáveis latentes (não diretamente observável)
from sklearn.model_selection import train_test_split # Divide conjunto de dados em treino e teste.
# Separa os dados que possa treinar o modelo em uma parte e avalia o seu desempenho na outra.
from sklearn.metrics import r2_score # Função que calcula o coeficiente de determinação R².
# R² = 1 (predição perfeita), 
# R² = 0 (modelo não explica nada além da média);
# R² < 0 (predições piores que simplesmente chutar a média).

####################
# Leitura do arquivo
####################
file_path = r"C:\Users\arthu\PY\Arquivos Python\Redes Neurais\Cap 9\cement_strength.txt" # Caminho do arquivo.
df = pd.read_csv(file_path) # Leitura do arquivo.

#######################
# Divisão das variáveis
#######################
X = df.drop(columns=["csMPa"]).values # Cria um array com os valores das variáveis preditoras.
y = df["csMPa"].values # Cria um array com os valores da variável-alvo (resistência do cimento).
plt.figure()
plt.plot(y, color='navy', linestyle = ':', marker='.', linewidth=0.5, markeredgecolor = 'k')
plt.xlabel('Sample #')
plt.ylabel('Concrete strength (MPa)')
plt.show()

######################
# Divisão treino/teste
######################
# Dividir os dados em 67% para treino e 33% para teste;
# O "random_state" garante que a divisão será sempre a mesma (reprodutibilidade)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100) 


###############
# Random Forest
###############
# RF com 200 árvores e 3 variáveis por divisão em cada árvore;
# "oob_score" ativa o cálculo de acurácia usando os dados fora da mostra (out-of-bag)
rf = RandomForestRegressor(n_estimators=200, max_features=3, oob_score=True, random_state=1)
rf.fit(X_train, y_train)
y_train_rf = rf.predict(X_train)
y_test_rf = rf.predict(X_test)

############
# Out-of-Bag
############
print('OOB score: ', rf.oob_score_)

#######################
# Partial Least Squares
#######################
pls = PLSRegression(n_components=8) # Reduz as variáveis de entrada a 8 componentes
pls.fit(X_train, y_train)
y_train_pls = pls.predict(X_train).ravel()
y_test_pls = pls.predict(X_test).ravel()

#######################
# Gráficos Comparativos
#######################
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

###########
# RF Treino
###########
axs[0, 0].scatter(y_train, y_train_rf, c='steelblue', edgecolors='k', alpha=0.6)
axs[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', linewidth=1)
# axs[0, 0].set_title(f'RF: $R^2$ = {r2_score(y_train, y_train_rf):.3f}')
axs[0, 0].set_xlabel("raw training data")
axs[0, 0].set_ylabel("prediction")
axs[0, 0].text(0.05, 0.95, 
               f'RF: $R^2$ = {r2_score(y_train, y_train_rf):.3f}',
               transform=axs[0, 0].transAxes,  # Usa sistema de coordenadas relativo ao gráfico
               color='steelblue',
               fontsize=12,
               fontweight='bold',
               verticalalignment='top')

##########
# RF Teste
##########
axs[0, 1].scatter(y_test, y_test_rf, c='steelblue', edgecolors='k', alpha=0.6)
axs[0, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', linewidth=1)
#axs[0, 1].set_title(f'RF: $R^2$ = {r2_score(y_test, y_test_rf):.3f}')
axs[0, 1].set_xlabel("raw test data")
axs[0, 1].set_ylabel("prediction")
axs[0, 1].text(0.05, 0.95, 
               f'RF: $R^2$ = {r2_score(y_test, y_test_rf):.3f}',
               transform=axs[0, 1].transAxes,
               color='steelblue',
               fontsize=12, 
               fontweight='bold',
               verticalalignment='top')


############
# PLS Treino
############
axs[1, 0].scatter(y_train, y_train_pls, c='steelblue', edgecolors='k', alpha=0.6)
axs[1, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', linewidth=1)
#axs[1, 0].set_title(f'PLS: $R^2$ = {r2_score(y_train, y_train_pls):.3f}', color='magenta')
axs[1, 0].set_xlabel("raw training data")
axs[1, 0].set_ylabel("prediction")
axs[1, 0].text(0.05, 0.95, 
               f'PLS: $R^2$ = {r2_score(y_train, y_train_pls):.3f}',      
               transform=axs[1, 0].transAxes,
               color='magenta',
               fontsize=12, 
               fontweight='bold',
               verticalalignment='top')

###########
# PLS Teste
###########
axs[1, 1].scatter(y_test, y_test_pls, c='steelblue', edgecolors='k', alpha=0.6)
axs[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r-', linewidth=1)
#axs[1, 1].set_title(f'PLS: $R^2$ = {r2_score(y_test, y_test_pls):.3f}', color='magenta')
axs[1, 1].set_xlabel("raw test data")
axs[1, 1].set_ylabel("prediction")
axs[1, 1].text(0.05, 0.95, 
               f'PLS: $R^2$ = {r2_score(y_test, y_test_pls):.3f}',      
               transform=axs[1, 1].transAxes,
               color='magenta',
               fontsize=12, 
               fontweight='bold',
               verticalalignment='top')

plt.tight_layout()
plt.show()

###########################################
# Gráfico de Importância das Variáveis (RF)
###########################################
feature_names = ['cement','slag','flyash','water','superplasticizer','coarseaggregate','fineaggregate','age']
importances = rf.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances, color='steelblue')
plt.xlabel("Feature importances")
plt.tight_layout()
plt.show()