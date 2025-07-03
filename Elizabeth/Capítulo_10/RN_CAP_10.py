import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Carregando arquivo de dados
df = pd.read_csv('d00.txt', delimiter='\s+', header=None)  
ICA_statistics_train = df.values  # converte para NumPy array 


def encontrar_largura_banda_otima(valores_metricas):
    """
    Encontra a largura de banda ótima para KDE usando GridSearchCV com validação cruzada leave-one-out.
    
    Parâmetros:
    -----------
    valores_metricas : np.array de formato (n_amostras,)
    
    Retorna:
    --------
    largura_banda_otima : float
    """
    N = len(valores_metricas)
    largura_empirica = max(1e-3, 1.06 * np.std(valores_metricas) * N**(-1/5)) 
    grade_largura = np.linspace(0, 5, 50) * largura_empirica

    grid_cv = GridSearchCV(KernelDensity(), {'bandwidth': grade_largura}, cv=N)
    grid_cv.fit(valores_metricas[:, None])
    largura_banda_otima = grid_cv.best_params_['bandwidth']
    return largura_banda_otima, grid_cv.best_estimator_

def calcular_limite_controle_via_KDE(valores_metricas, percentual, nome_metrica):
    """
    Calcula o limite de controle para uma estatística de monitoramento via KDE.
    
    Parâmetros:
    -----------
    valores_metricas : np.array (n_amostras,)
    percentual : float (percentil para o limite de controle, ex: 99)
    nome_metrica : str (para uso em gráficos ou logs)
    
    Retorna:
    --------
    limite_controle : float
    """
    # Encontrar KDE ótimo
    largura_otima, kde_otimo = encontrar_largura_banda_otima(valores_metricas)

    # Geração de grade para avaliação da densidade
    grade_metricas = np.linspace(0, np.max(valores_metricas), 100)[:, None]
    densidade_pdf = np.exp(kde_otimo.score_samples(grade_metricas))

    # Cálculo da função densidade acumulada (CDF) via integração numérica
    cdf_metricas = [np.trapz(densidade_pdf[:i], grade_metricas[:i, 0]) for i in range(1, 101)]

    # Encontrar o valor do limite de controle para o percentil dado
    indice_limite = np.argmax(np.array(cdf_metricas) >= percentual / 100)
    limite_controle = grade_metricas[indice_limite, 0]

    print(f"{nome_metrica}: largura de banda ótima = {largura_otima:.4f}, limite controle = {limite_controle:.4f}")

    return limite_controle

def plotar_densidade_histograma(valores_metricas, kde, titulo_eixo_x):
    """
    Plota histograma e curva KDE para os valores da métrica.
    
    Parâmetros:
    -----------
    valores_metricas : np.array (n_amostras,)
    kde : objeto KernelDensity treinado
    titulo_eixo_x : str
    """
    grade_metricas = np.linspace(np.min(valores_metricas), np.max(valores_metricas), 100)[:, None]
    densidade = np.exp(kde.score_samples(grade_metricas))

    plt.hist(valores_metricas, bins=50, color='grey', alpha=0.8, density=True, rwidth=0.7, label='Densidade Histograma')
    plt.plot(grade_metricas, densidade, label='Densidade KDE')
    plt.xlabel(titulo_eixo_x)
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.show()

# ICA_statistics_train é sua matriz de dados de treino, formato (n_amostras, 3)

# Calcular limites de controle para três métricas
I2_CL = calcular_limite_controle_via_KDE(ICA_statistics_train[:, 0], 99, 'I2')
Ie2_CL = calcular_limite_controle_via_KDE(ICA_statistics_train[:, 1], 99, 'Ie2')
SPE_CL = calcular_limite_controle_via_KDE(ICA_statistics_train[:, 2], 99, 'SPE')

# Plotar densidade e histograma para a métrica Ie2 como exemplo
_, kde_Ie2 = encontrar_largura_banda_otima(ICA_statistics_train[:, 1])
plotar_densidade_histograma(ICA_statistics_train[:, 1], kde_Ie2, 'Valores Ie2')

# Pipeline para escalonamento e PCA
pipe = Pipeline([
    ('escalonador', StandardScaler()),
    ('pca', PCA(n_components=3))
])

# Ajusta modelo PCA nos dados de treino
df_raw = pd.read_csv('d00.txt', delimiter=',') 
unfolded_dataMatrix = df_raw.values
score_train = pipe.fit_transform(unfolded_dataMatrix)

# k-vizinhos mais próximos para dados de treino
nbrs = NearestNeighbors(n_neighbors=6).fit(score_train)  # inclui o próprio ponto

distancias_nbrs, indices_nbrs = nbrs.kneighbors(score_train)
distancias_quadradas = distancias_nbrs ** 2

# Estatística D2 (soma dos quadrados das distâncias)
D2 = np.sum(distancias_quadradas, axis=1)
D2_log = np.log(D2)

# Calcular limite controle para D2_log
D2_log_CL = calcular_limite_controle_via_KDE(D2_log, 95, 'D2_log')

# Aplicar escalonamento e PCA nos dados de teste
unfolded_dataMatrix = ICA_statistics_train
score_test = pipe.transform(unfolded_dataMatrix)

# Calcular D2_log para dados de teste
distancias_nbrs_test, indices_test = nbrs.kneighbors(score_test)
distancias_nbrs_test = distancias_nbrs_test[:, :5]  # 5 vizinhos mais próximos
distancias_quadradas_test = distancias_nbrs_test ** 2
D2_test = np.sum(distancias_quadradas_test, axis=1)
D2_log_test = np.log(D2_test)


