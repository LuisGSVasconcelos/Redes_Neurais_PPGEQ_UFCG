import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance
from scipy import stats

# 1. Carregar os dados
data_2Doutlier = np.loadtxt('simple2D_outlier.csv', delimiter=',')

# 2. Calcular as distâncias de Mahalanobis e transformá-las para distribuição Gaussiana usando raiz cúbica
emp_cov = EmpiricalCovariance().fit(data_2Doutlier)
mahalanobis_distances = emp_cov.mahalanobis(data_2Doutlier)
mahalanobis_cube_root = np.power(mahalanobis_distances, 1/3)  # Raiz cúbica

# 3. Calcular os limites de Hampel (upper e lower bounds)
median = np.median(mahalanobis_cube_root)
mad = stats.median_abs_deviation(mahalanobis_cube_root)
upper_bound = np.power(median + 3 * mad, 3)
lower_bound = np.power(median - 3 * mad, 3)

# 4. Plotar as distâncias de Mahalanobis com os limites (outliers destacados em vermelho)
plt.figure(figsize=(10, 6))

# Plot dos dados não-outliers (excluindo os últimos 5)
plt.plot(mahalanobis_distances[:-5], '.', markeredgecolor='k', markeredgewidth=0.5, ms=9, label='Non-outliers')

# Plot dos últimos 5 outliers em vermelho
plt.plot(np.arange(len(mahalanobis_distances)-5, len(mahalanobis_distances)), mahalanobis_distances[-5:], '.r', markeredgecolor='k', markeredgewidth=0.5, ms=11, label='Outliers')

# Adicionar os limites superiores e inferiores
plt.hlines(upper_bound, 0, len(mahalanobis_distances), colors='r', linestyles='dashdot', label='Upper Bound')
plt.hlines(lower_bound, 0, len(mahalanobis_distances), colors='r', linestyles='dashed', label='Lower Bound')

# Adicionar título e rótulos
plt.title('Mahalanobis Distances with Hampel Bounds')
plt.xlabel('Samples')
plt.ylabel('Mahalanobis Distance')

# Exibir legenda
plt.legend()

# Exibir o gráfico
plt.show()

