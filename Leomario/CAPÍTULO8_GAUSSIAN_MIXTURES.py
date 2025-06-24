##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                  Algoritmo GMM para o exemplo ilustrativo
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% Importação das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

#%% Gerando os dados
n_amostras = 1500  # Número de amostras
X, _ = make_blobs(n_samples=n_amostras, random_state=100)  # Criando clusters artificiais

# Exibindo os dados iniciais
plt.figure()
plt.scatter(X[:,0], X[:,1])

# Aplicando uma transformação rotacional nos dados
matriz_rotacao = [[0.60, -0.70], [-0.5, 0.7]]
X_transformado = np.dot(X, matriz_rotacao)

# Exibindo os dados transformados
plt.figure()
plt.scatter(X_transformado[:,0], X_transformado[:,1])

#%% Ajustando o modelo GMM (Gaussian Mixture Model)
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=100)  # Definindo 3 clusters
rotulos_cluster = gmm.fit_predict(X_transformado)  # Ajustando e classificando os dados

# Exibindo os clusters identificados
plt.figure()
plt.scatter(X_transformado[:, 0], X_transformado[:, 1], c=rotulos_cluster, s=20, cmap='viridis')
plt.xlabel('Variável 1')
plt.ylabel('Variável 2')

# Obtendo os centros dos clusters
centros_clusters = gmm.means_  # Centros dos clusters
rotulos_plot_clusters = ['Cluster ' + str(i+1) for i in range(gmm.n_components)]
for i in range(gmm.n_components):
    plt.scatter(centros_clusters[i, 0], centros_clusters[i, 1], c='red', s=20, marker='*', alpha=0.5)
    plt.annotate(rotulos_plot_clusters[i], (centros_clusters[i,0], centros_clusters[i,1]))

#%% Cálculo das probabilidades de associação a cada cluster
probs = gmm.predict_proba(X_transformado[1069, np.newaxis])  # Probabilidades para um ponto específico
print('Probabilidades posteriores para os clusters 1, 2 e 3: ', probs[-1, :])

#%% Cálculo manual da probabilidade posterior
x = X_transformado[1069, np.newaxis]  # Selecionando um ponto

import scipy.stats
# Calculando as densidades de probabilidade dos componentes gaussianos
g1 = scipy.stats.multivariate_normal(gmm.means_[0,:], gmm.covariances_[0,:]).pdf(x)
g2 = scipy.stats.multivariate_normal(gmm.means_[1,:], gmm.covariances_[1,:]).pdf(x)
g3 = scipy.stats.multivariate_normal(gmm.means_[2,:], gmm.covariances_[2,:]).pdf(x)
print('Densidades dos componentes locais: ', g1, g2, g3)

# Cálculo da probabilidade posterior
normalizador = gmm.weights_[0]*g1 + gmm.weights_[1]*g2 + gmm.weights_[2]*g3
prob_posterior_cluster1 = gmm.weights_[0]*g1 / normalizador
prob_posterior_cluster2 = gmm.weights_[1]*g2 / normalizador
prob_posterior_cluster3 = gmm.weights_[2]*g3 / normalizador
print('Probabilidades posteriores: ', prob_posterior_cluster1, prob_posterior_cluster2, prob_posterior_cluster3)

#%% Encontrando o número ideal de componentes usando o critério BIC
BICs = []
menor_BIC = np.inf
for n_cluster in range(1, 10):
    gmm = GaussianMixture(n_components=n_cluster, random_state=100)
    gmm.fit(X_transformado)
    BIC = gmm.bic(X_transformado)
    BICs.append(BIC)
    
    if BIC < menor_BIC:
        num_cluster_otimo = n_cluster 
        menor_BIC = BIC

# Exibindo a curva do BIC
plt.figure()
plt.plot(range(1, 10), BICs, marker='o')
plt.scatter(num_cluster_otimo, menor_BIC, c='red', marker='*', s=1000)
plt.xlabel('Número de clusters')
plt.ylabel('BIC')
plt.show()

#%% Encontrando o número de componentes via algoritmo FJ
from gmm_mml import GmmMml  # Importando um módulo externo

gmmFJ = GmmMml(plots=False)
gmmFJ.fit(X_transformado)
rotulos_cluster = gmmFJ.predict(X_transformado)

# Exibindo os clusters encontrados pelo método FJ
plt.figure()
plt.scatter(X_transformado[:, 0], X_transformado[:, 1], c=rotulos_cluster, s=20, cmap='viridis')
plt.xlabel('Variável 1')
plt.ylabel('Variável 2')

clusters_unicos = np.unique(rotulos_cluster)
print(clusters_unicos)