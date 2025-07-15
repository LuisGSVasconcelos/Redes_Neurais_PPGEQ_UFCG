import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from matplotlib.colors import ListedColormap

# 1. Gerar dados sintéticos
X, y = make_classification(n_samples=300, 
                           n_features=2, 
                           n_informative=2, 
                           n_redundant=0,
                           n_clusters_per_class=1, 
                           random_state=42)

# 2. Criar o modelo AdaBoost
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base_estimator, 
                           n_estimators=50, 
                           learning_rate=1, 
                           random_state=42)

# 3. Treinar o modelo
model.fit(X, y)

# 4. Plotar a fronteira de decisão
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 5. Configurar cores
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']

plt.figure(figsize=(8, 6))

# 6. Fronteira
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

# 7. Dados de entrada
for idx, color in enumerate(cmap_bold):
    plt.scatter(X[y == idx, 0], X[y == idx, 1], 
                c=color, label=f"Classe {idx}", 
                edgecolor='k', s=50)

# 8. Configurações
plt.title("Ensemble AdaBoost - Fronteira de Decisão")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
