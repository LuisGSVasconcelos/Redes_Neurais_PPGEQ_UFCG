# Importando as bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import scipy.stats
import scipy.linalg

# Carregando os dados (substitua 'proc1a.xls' pelo caminho correto do seu arquivo)
dados = pd.read_excel('proc1a.xls', skiprows=1, usecols='C:AI')

# Separando os dados de treino (as 69 primeiras amostras)
dados_treino = dados.iloc[:69, :]

# Normalizando os dados
normalizador = StandardScaler()
dados_treino_normalizado = normalizador.fit_transform(dados_treino)

# Aplicando PCA
pca = PCA()
componentes_treino = pca.fit_transform(dados_treino_normalizado)

# Matriz de correlação entre os componentes principais
coef_corr = np.corrcoef(componentes_treino, rowvar=False)
print('Matriz de correlação (3 primeiros componentes):\n', coef_corr[:3, :3])

# Variância explicada por cada componente principal
variancia_explicada = 100 * pca.explained_variance_ratio_
variancia_acumulada = np.cumsum(variancia_explicada)

# Plotando a variância explicada
plt.figure()
plt.plot(variancia_acumulada, 'r+', label='Variância acumulada (%)')
plt.plot(variancia_explicada, 'b+', label='Variância explicada por cada componente')
plt.xlabel('Componente Principal')
plt.ylabel('Variância Explicada (%)')
plt.legend()
plt.show()

# Número de componentes que explicam pelo menos 90% da variância
n_componentes = np.argmax(variancia_acumulada >= 90) + 1
print('Número de componentes necessários:', n_componentes)

# Redução dos dados para esses componentes
componentes_reduzidos = componentes_treino[:, :n_componentes]
matriz_V = pca.components_.T
matriz_P = matriz_V[:, :n_componentes]

# Reconstrução dos dados
dados_reconstruidos = np.dot(componentes_reduzidos, matriz_P.T)

# Calculando a perda de informação
pontuacao_r2 = r2_score(dados_treino_normalizado, dados_reconstruidos)
print('Informação perdida (%):', 100 * (1 - pontuacao_r2))


# Índice T²
valores_autovetores = np.diag(pca.explained_variance_[:n_componentes])
valores_inv = np.linalg.inv(valores_autovetores)

T2_treino = np.array([
    np.dot(np.dot(comp, valores_inv), comp.T) for comp in componentes_reduzidos
])

# Erro de reconstrução (Q)
erro_reconstrucao = dados_treino_normalizado - dados_reconstruidos
Q_treino = np.sum(erro_reconstrucao ** 2, axis=1)

# Limite de controle para T²
N = dados_treino_normalizado.shape[0]
k = n_componentes
alpha = 0.01  # 99% de confiança
T2_limite = k * (N**2 - 1) * scipy.stats.f.ppf(1 - alpha, k, N - k) / (N * (N - k))

# Limite de controle para Q (SPE)
valores_autovalores = pca.explained_variance_
m = dados_treino_normalizado.shape[1]
theta1 = np.sum(valores_autovalores[k:])
theta2 = np.sum(valores_autovalores[k:]**2)
theta3 = np.sum(valores_autovalores[k:]**3)
h0 = 1 - 2 * theta1 * theta3 / (3 * theta2**2)
z_alpha = scipy.stats.norm.ppf(1 - alpha)
Q_limite = theta1 * (z_alpha * np.sqrt(2 * theta2 * h0**2) / theta1 + 1 + theta2 * h0 * (1 - h0) / theta1**2)**2

# Plotando os índices
plt.figure()
plt.plot(Q_treino, label='Q (SPE)')
plt.axhline(Q_limite, color='red', label='Limite de Controle')
plt.xlabel('Amostra')
plt.ylabel('Q')
plt.legend()
plt.show()

plt.figure()
plt.plot(T2_treino, label='T²')
plt.axhline(T2_limite, color='red', label='Limite de Controle')
plt.xlabel('Amostra')
plt.ylabel('T²')
plt.legend()
plt.show()

# Dados de teste (resto das amostras)
dados_teste = dados.iloc[69:, :]
dados_teste_normalizado = normalizador.transform(dados_teste)

# Aplicando transformação
componentes_teste = pca.transform(dados_teste_normalizado)
componentes_teste_reduzidos = componentes_teste[:, :n_componentes]
dados_teste_reconstruido = np.dot(componentes_teste_reduzidos, matriz_P.T)

# Calculando T² e Q para os dados de teste
T2_teste = np.array([
    np.dot(np.dot(comp, valores_inv), comp.T) for comp in componentes_teste_reduzidos
])

erro_teste = dados_teste_normalizado - dados_teste_reconstruido
Q_teste = np.sum(erro_teste ** 2, axis=1)

# Plotando T² e Q para os dados de teste
plt.figure()
plt.plot(Q_teste, label='Q Teste')
plt.axhline(Q_limite, color='red', label='Limite de Controle')
plt.legend()
plt.show()

plt.figure()
plt.plot(T2_teste, label='T² Teste')
plt.axhline(T2_limite, color='red', label='Limite de Controle')
plt.legend()
plt.show()

# Escolhendo uma amostra de teste específica (exemplo: amostra 85)
indice_amostra = 85 - 69  # ajuste para índice de dados_teste
amostra = dados_teste_normalizado[indice_amostra]
D = np.dot(np.dot(matriz_P, valores_inv), matriz_P.T)
contrib_T2 = np.dot(scipy.linalg.sqrtm(D), amostra.T)**2

plt.figure()
plt.plot(contrib_T2)
plt.ylabel('Contribuição T²')
plt.title('Contribuição por variável')
plt.show()

# Contribuição do erro (SPE)
erro_amostra = erro_teste[indice_amostra]
contrib_SPE = erro_amostra**2

plt.figure()
plt.plot(contrib_SPE)
plt.ylabel('Contribuição SPE')
plt.title('Contribuição do Erro')
plt.show()
