# Importação das bibliotecas
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# Definição dos dados categóricos
x = np.array([['type A'], ['type C'], ['type B'], ['type C']])

# Instanciação do OneHotEncoder e transformação dos dados
ohe = OneHotEncoder(sparse_output=False)  # Retorna um array denso
X_encoded = ohe.fit_transform(x)  

# Exibição dos resultados
print("Categorias:", ohe.categories_)
print("Dados codificados:\n", X_encoded)
