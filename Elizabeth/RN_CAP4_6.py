import numpy as np
from sklearn.impute import SimpleImputer

def imputar_dados_medianos(dados):
    """
    Função para realizar imputação de dados ausentes usando a média.
    :param dados: Dados de entrada com valores ausentes (np.nan).
    :return: Dados com valores ausentes substituídos pela média.
    """
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    dados_imputados = imputer.fit_transform(dados)
    return dados_imputados

# Dados de exemplo com valores ausentes (np.nan)
dados_amostra = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]

# Imputar os dados usando a média
dados_imputados = imputar_dados_medianos(dados_amostra)

# Exibir os dados antes e depois da imputação
print("Dados originais:")
print(np.array(dados_amostra))
print("\nDados após imputação de média:")
print(dados_imputados)

