import numpy as np
from sklearn.impute import KNNImputer

def imputar_dados_knn(dados, n_neighbors=2):
    """
    Função para realizar imputação de dados ausentes usando o algoritmo KNN (K-Nearest Neighbors).
    :param dados: Dados de entrada com valores ausentes (np.nan).
    :param n_neighbors: Número de vizinhos a serem considerados para a imputação.
    :return: Dados com valores ausentes imputados.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    dados_imputados = imputer.fit_transform(dados)
    return dados_imputados

# Dados de exemplo com valores ausentes (np.nan)
dados_amostra = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]

# Imputar os dados usando KNN
dados_imputados_knn = imputar_dados_knn(dados_amostra, n_neighbors=2)

# Exibir os dados antes e depois da imputação
print("Dados originais:")
print(np.array(dados_amostra))
print("\nDados após imputação KNN:")
print(dados_imputados_knn)


