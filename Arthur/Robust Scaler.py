# Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import median_abs_deviation

# Entrada dos dados simples
X = np.array([
    [1000, 0.01, 300],
    [1200, 0.06, 350],
    [1500, 0.10, 320]
])

# Standard Scaling
scaler_std = StandardScaler()
X_standardized = scaler_std.fit_transform(X)
np.set_printoptions(precision=4, suppress=True) #  Saída com 4 casas decimais
print("Standard Scaled Data:\n", X_standardized)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
np.set_printoptions(precision=2, suppress=True) #  Saída com 2 casas decimais
print("Min-Max Scaled Data:\n", X_minmax)

# Entrada de dados outliers
np.random.seed(0)
X_outlier = np.random.normal(40, 1, (1500, 1))
X_outlier[200:300] += 8
X_outlier[1000:1150] += 8

# Standard scaling
X_std = StandardScaler().fit_transform(X_outlier)

# MAD scaling
median = np.median(X_outlier)
mad = median_abs_deviation(X_outlier, scale='normal')
X_mad = (X_outlier - median) / mad

# Comparativo
fig, axs = plt.subplots(1, 3, figsize=(18, 4))

plt.style.use('default') # Padrão do matplotlib

# Original (sem escalonamento)
axs[0].plot(X_outlier, '.-', color='blue')
axs[0].set_title('(a) No scaling')
axs[0].set_xlabel('sample #')
axs[0].set_ylabel('variable measurement')

# Standard scaling
axs[1].plot(X_std, '.-', color='blue')
axs[1].axhline(0, color='red')
axs[1].set_title('(b) Standard scaling')
axs[1].set_xlabel('sample #')
axs[1].set_ylabel('scaled variable measurement')

# MAD scaling
axs[2].plot(X_mad, '.-', color='blue')
axs[2].axhline(0, color='red')
axs[2].set_title('(c) MAD scaling')
axs[2].set_xlabel('sample #')
axs[2].set_ylabel('scaled variable measurement')

plt.tight_layout()
plt.show()