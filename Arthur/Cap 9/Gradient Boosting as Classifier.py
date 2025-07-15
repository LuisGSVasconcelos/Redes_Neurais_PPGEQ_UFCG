import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Gerar dados sintéticos para classificação
X, y = make_classification(n_samples=300, n_features=4, n_informative=2, 
                           n_redundant=0, random_state=42)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar o modelo Gradient Boosting Classifier
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Treinar o modelo
gbc.fit(X_train, y_train)

# Previsões
y_pred = gbc.predict(X_test)

# Avaliação
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Importância das features
plt.figure(figsize=(6,4))
plt.bar(range(X.shape[1]), gbc.feature_importances_)
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.title('Gradient Boosting Classifier')
plt.show()
