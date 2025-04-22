import numpy as np
from sklearn.covariance import MinCovDet

def load_data(file_path):
    """
    Função para carregar dados de um arquivo CSV.
    :param file_path: Caminho para o arquivo CSV.
    :return: Dados carregados como um array NumPy.
    """
    try:
        data = np.loadtxt(file_path, delimiter=',')
        print(f"Dados carregados com sucesso do arquivo {file_path}")
        return data
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return None

def compute_mahalanobis_distances(data):
    """
    Função para calcular as distâncias de Mahalanobis usando a estimativa MCD (Minimum Covariance Determinant).
    :param data: Dados de entrada.
    :return: Distâncias de Mahalanobis baseadas no MCD.
    """
    MCD_cov = MinCovDet().fit(data)
    MD_MCD = MCD_cov.mahalanobis(data)
    return MD_MCD

# Carregar dados do arquivo
data_2Doutlier = load_data('complex2D_outlier.csv')

# Verificar se os dados foram carregados corretamente antes de continuar
if data_2Doutlier is not None:
    # Calcular as distâncias de Mahalanobis usando MCD
    MD_MCD = compute_mahalanobis_distances(data_2Doutlier)
    print("Distâncias de Mahalanobis calculadas com sucesso.")
else:
    print("Falha ao carregar os dados. O código não prosseguiu.")

