#########################
# Importar as bibliotecas 
#########################
import numpy as np # Cálculos numéricos com arrays e matrizes.
import matplotlib.pyplot as plt # Submódulo 'pyplot' responsável.
# Acessar o módulo 'tree' do pacote 'scikit-learn', que é ferramenta de Machine Learning:
    # 'DecisionTreeRegressor' é uma classe usada para criar e treinar modelos de DT;
    # 'plot_tree' permite visualizar graficamente a estrutura da árvore treinada.
from sklearn.tree import DecisionTreeRegressor, plot_tree

########################
# Gerar dados sintéticos 
########################
np.random.seed() # Gera números aleatórios de um ponto de partida fixo.
print(np.random.rand(10)) # Exibe os 10 primeiros números gerados
# Criar um array com 50 valores igualmente espaçados entre -1 e 1:
    # 'reshape(-1, 1)' reorganiza o array de 1D (50) para 2D (50, 1).
x = np.linspace(-1, 1, 50).reshape(-1, 1)
#print(x)
# Função quadrática com deslocamento de 0,25 no eixo y;
    # Segunda parte gera um array (50,1) com guassian noise (média 0, desvio 0.15):
        # Ruído Guassiano é um ruído com distribuição normal de Gauss (bell curve).
y = x**2 + 0.25 + np.random.normal(0, 0.15, size=(50, 1)) 
y = y.ravel() # Converte 'y' de matriz coluna para vetor unidimensional
#print(y)

#################################
# Gráfico apenas com dados brutos
#################################
plt.figure(figsize=(6, 4)) # Cria uma figura 6'x4'.
plt.scatter(x, y, edgecolor='black', facecolor='orange', label='Raw Data') # Cria um gráfico de dispersão.
plt.xlabel('x') # Rotula o eixo 'x'.
plt.ylabel('y') # Rotula o eixo 'y'.
plt.title('Raw Data') # Define o título do gráfico.
plt.grid(True) # Ativa grade de fundo no gráfico.
plt.tight_layout() # Ajusta automaticamente o espaçamento dos elementos da figura.
plt.show() # Exibe o gráfico na tela.

###################################################
# Modelos de árvore regularizado e não regularizado
###################################################
model_reg = DecisionTreeRegressor(max_depth=3).fit(x,y) # DT de regressão com 3 níveis de profundidade.
model_noreg = DecisionTreeRegressor(max_depth=None).fit(x,y) # DT de regressão sem limite de profundidade.

###########
# Predições
###########
x_test = np.linspace(-1, 1, 500).reshape(-1, 1) # Gera 500 valores igualmente espaçados entre -1 e 1
#print(x_test) # Imprime a matriz 500x1.
y_pred_reg = model_reg.predict(x_test) # Usa o modelo regularizado
y_pred_noreg = model_noreg.predict(x_test) # Usa o modelo não regularizado
#print(y_pred_reg)
#print(y_pred_noreg)

####################################
# Gráfico comparando os dois modelos
####################################
# Criar uma figura 1 linha x 2 colunas de tamanho 12x4:
    # 'sharex=True' faz com que os dois subgáficos compartilhem o mesmo eixo x (pode ser 'shared').
fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True) # Objeto de figura inteira utilizando arrays

#########################
# Modelo não regularizado
#########################
axs[0].scatter(x, y, edgecolor='black', facecolor='orange', label='raw data')
axs[0].plot(x_test, y_pred_noreg, color='yellowgreen', label='predictions')
axs[0].set_title('Decision Tree (unregularized model)')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()

#####################
# Modelo regularizado
#####################
axs[1].scatter(x, y, edgecolor='black', facecolor='orange', label='raw data')
axs[1].plot(x_test, y_pred_reg, color='yellowgreen', label='predictions')
axs[1].set_title('Decision Tree (regularized model)')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()

plt.tight_layout()
plt.show()

##################################
# Visualizar a estrutura da árvore
##################################
plt.figure(figsize=(20, 8))
plot_tree(model_reg, feature_names=['x'], filled=True, rounded=True)
plt.title('Decision Tree Structure')
plt.show()