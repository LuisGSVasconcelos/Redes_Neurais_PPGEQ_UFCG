import pandas as pd

# Carregar os dados do arquivo CSV
file_path = "Metal_etch_2DPCA_trainingData.csv"
df = pd.read_csv(file_path)

# Renomear colunas para garantir que estejam corretas
df.columns = ["PC1", "PC2"]

# Converter para um array NumPy
score_train = df.values
# Importar bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Aplicar DBSCAN
db = DBSCAN(eps=5, min_samples=3).fit(score_train)
cluster_label = db.labels_

# Criar gráfico de dispersão dos clusters
plt.figure(figsize=(8, 6))
scatter = plt.scatter(score_train[:, 0], score_train[:, 1], c=cluster_label, s=20, cmap='viridis', edgecolors='k')
plt.xlabel('PC1 scores')
plt.ylabel('PC2 scores')
plt.title('DBSCAN Clustering')

# Adicionar barra de cores para melhor visualização dos clusters
plt.colorbar(scatter, label="Cluster Label")
plt.show()

# Exibir os rótulos dos clusters encontrados
print('Cluster labels:', np.unique(cluster_label))