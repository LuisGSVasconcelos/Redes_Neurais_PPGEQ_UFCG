#########################
# Importar as bibliotecas 
#########################
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets # Fornece funções para gerar dados sintéticos - facilita testes.
from sklearn.ensemble import BaggingClassifier # Implementa o método Bagging
# Cria vários classificadores (ex: várias árvores) usando amostras aleatórias com reposição;
# Faz a predição final por votação (majority vote);
# Reduz a variância de modelos instáveis como árvores de decisão, melhorando generalização.
from sklearn.tree import DecisionTreeClassifier # Cria modelos base para Bagging.
from matplotlib.colors import ListedColormap # Permite criar mapa de cores personalizado.

#################
# Gerar o dataset
#################
# Gera um conjunto de dados sintéticos para classificação binária:
    # Possui formato de duas luas entrelaçadas (moons);
    # Muito usado para testar algoritmos de classificação em problemas não-lineares.
X, y = datasets.make_moons(n_samples=200, noise=0.3, random_state=10) 
# 200 amostra (100 para cada moon);
# 0,3 de ruído gaussiano ao posicionamento dos pontos;
# Semente geradora de números aleatórios garantindo reprodutibilidade.
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['#FF0000', '#0000FF']))
plt.title(" Make Moons Dataset")
plt.show()

####################################
# Modelo base: uma árvore de decisão
####################################
tree_model = DecisionTreeClassifier(random_state=0) # Implementa uma DT para classificação.
tree_model.fit(X, y)

#############################################
# Modelo ensemble: Bagging com Decision Trees
#############################################
bagging_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(), # Define a DT como modelo base (fraco).
    n_estimators=500, # Cria 500 modelos base - maior = mais estável (alto custo de pc).
    max_samples=50, # Cada modelo base será treinado com 50 amostras com bootstrap.
    random_state=100 # Define uma semente para garantir a reprodutibilidade.
)
bagging_model.fit(X, y)

###########################################
# Função para plotar a fronteira de decisão
###########################################
def plot_decision_boundary(model, X, y, ax, title):
    cmap_light = ListedColormap(['#FFBBBB', '#BBBBFF'])
    cmap_bold = ListedColormap(['#660000', '#000066'])
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # Limites do gráfico com margem 1.
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid( # Cria uma grade de pontos (malha 2D) sobre o espaço de entrada.
        np.arange(x_min, x_max, 0.02), # Espaçamento 0,02: mais fino = mais detalhado.
        np.arange(y_min, y_max, 0.02)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # Para cada ponto da grade, faz a predição.
    Z = Z.reshape(xx.shape) # Reorganiza as predições para o formato 2D.

    ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8) # Preenche as regiões conforme classe prevista.
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k')
    ax.set_xlabel(r'$x_1$', fontsize=14)
    ax.set_ylabel(r'$x_2$', fontsize=14)
    ax.set_title(title, fontsize=14)

###################
# Criar os subplots
###################
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

####################
# Plotar modelo base
####################
plot_decision_boundary(tree_model, X, y, axes[0], 'Base Model') # Fronteira de Decisão

#########################
# Plotar ensemble Bagging
#########################
plot_decision_boundary(bagging_model, X, y, axes[1], 'Bagging Ensemble')

#################
# Ajustar legenda
#################
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label='Class A',
               markerfacecolor='#660000', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Class B',
               markerfacecolor='#000066', markersize=10)
]
fig.legend(handles=handles, loc='upper left', ncol=2, fontsize=12)

plt.tight_layout()
plt.show()