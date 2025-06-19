import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import scipy.stats

# 1. Carregar dados normais e com falha
dados_normais = np.loadtxt('d00.txt').T
dados_falha10 = np.loadtxt('d10.txt')

# 2. Visualizar temperatura da coluna Stripper
plt.figure()
plt.plot(dados_normais[:, 17])
plt.xlabel('Amostra')
plt.ylabel('Temperatura do Stripper')
plt.title('Operação normal')

plt.figure()
plt.plot(dados_falha10[:, 17])
plt.xlabel('Amostra')
plt.ylabel('Temperatura do Stripper')
plt.title('Operação com falha 10')

# 3. Selecionar variáveis de processo e variáveis manipuladas
xmeas = dados_normais[:, 0:22]
xmv = dados_normais[:, 41:52]
dados_treinamento = np.hstack((xmeas, xmv))

# 4. Escalar dados
escalador = StandardScaler()
dados_treinamento_normalizado = escalador.fit_transform(dados_treinamento)

# 5. Aplicar ICA
ica = FastICA(max_iter=1000, tol=0.005)
ica.fit(dados_treinamento_normalizado)
W = ica.components_

# 6. Ordenar Componentes Independentes (ICs)
normas_L2 = np.linalg.norm(W, axis=1)
ordem = np.flip(np.argsort(normas_L2))
W_ordenado = W[ordem, :]
normas_pct = 100 * normas_L2[ordem] / np.sum(normas_L2)

# 7. Gráficos das normas
plt.figure()
plt.plot(normas_L2)
plt.xlabel('IC não ordenado')
plt.ylabel('Norma L2')

plt.figure()
plt.plot(normas_pct, 'b+')
plt.xlabel('IC ordenado')
plt.ylabel('% da norma L2')

# 8. Usar PCA para definir quantos ICs manter (90% variância)
pca = PCA().fit(dados_treinamento_normalizado)
var_explicada = 100 * pca.explained_variance_ratio_
var_acumulada = np.cumsum(var_explicada)
n_comp = np.argmax(var_acumulada >= 90) + 1
print('Número de Componentes que explicam pelo menos 90% da variância:', n_comp)

Wd = W_ordenado[0:n_comp, :]
Sd = np.dot(Wd, dados_treinamento_normalizado.T)

# 9. Função para calcular estatísticas de monitoramento ICA
def calcula_metricas_ICA(modelo_ica, n_comp, dados):
    n = dados.shape[0]
    W = modelo_ica.components_
    ordem = np.flip(np.argsort(np.linalg.norm(W, axis=1)))
    W_ordenado = W[ordem, :]
    Wd = W_ordenado[0:n_comp, :]
    Sd = np.dot(Wd, dados.T)
    I2 = np.array([np.dot(Sd[:, i], Sd[:, i]) for i in range(n)])
    We = W_ordenado[n_comp:, :]
    Se = np.dot(We, dados.T)
    Ie2 = np.array([np.dot(Se[:, i], Se[:, i]) for i in range(n)])
    Q = modelo_ica.whitening_
    Q_inv = np.linalg.inv(Q)
    A = modelo_ica.mixing_
    B = np.dot(Q, A)
    B_ordenado = B[:, ordem]
    Bd = B_ordenado[:, :n_comp]
    reconstruido = np.dot(np.dot(np.dot(Q_inv, Bd), Wd), dados.T)
    erro = dados.T - reconstruido
    SPE = np.array([np.dot(erro[:, i], erro[:, i]) for i in range(n)])
    return np.column_stack((I2, Ie2, SPE))

# 10. Funções de visualização
def grafico_monitoramento(valores, limite, rotulo):
    plt.figure()
    plt.plot(valores)
    plt.axhline(limite, color="red", linestyle="--")
    plt.xlabel("Amostra")
    plt.ylabel(rotulo)

def graficos_ICA(estatisticas, limites, tipo):
    grafico_monitoramento(estatisticas[:,0], limites[0], f"I2 - {tipo}")
    grafico_monitoramento(estatisticas[:,1], limites[1], f"Ie2 - {tipo}")
    grafico_monitoramento(estatisticas[:,2], limites[2], f"SPE - {tipo}")

# 11. Avaliar dados normais (treinamento)
estat_train = calcula_metricas_ICA(ica, n_comp, dados_treinamento_normalizado)
limites = [np.percentile(estat_train[:,i], 99) for i in range(3)]
graficos_ICA(estat_train, limites, 'treinamento')

# 12. Função para taxa de alarme
def calcula_taxa_alarme(estatisticas, limites):
    alarme = estatisticas > limites
    alarme_total = np.any(alarme, axis=1)
    return 100 * np.sum(alarme_total) / len(alarme_total)

# 13. Avaliar dados com falha
dados_falha10_test = np.loadtxt('d10_te.txt')
xmeas = dados_falha10_test[:, 0:22]
xmv = dados_falha10_test[:, 41:52]
dados_teste = np.hstack((xmeas, xmv))
dados_teste_normalizado = escalador.transform(dados_teste)

estat_teste = calcula_metricas_ICA(ica, n_comp, dados_teste_normalizado)
graficos_ICA(estat_teste, limites, 'teste')
taxa_alarme = calcula_taxa_alarme(estat_teste[160:], limites)
print("Taxa de alarme após a falha:", taxa_alarme, "%")

# 14. Preparar dados de treino para LDA (falhas 5, 10 e 19)
d5 = np.loadtxt('d05.txt')
d10 = np.loadtxt('d10.txt')
d19 = np.loadtxt('d19.txt')
dados_falhas = np.vstack((d5, d10, d19))
xmeas = dados_falhas[:, 0:22]
xmv = dados_falhas[:, 41:52]
dados_falhas_completo = np.hstack((xmeas, xmv))

n_amostras = d5.shape[0]
y = np.concatenate((5*np.ones(n_amostras), 10*np.ones(n_amostras), 19*np.ones(n_amostras)))
dados_falhas_escalado = escalador.fit_transform(dados_falhas_completo)

# 15. Aplicar LDA
lda = LinearDiscriminantAnalysis()
scores_lda = lda.fit_transform(dados_falhas_escalado, y)

# 16. Visualizar LDA
plt.figure()
plt.plot(scores_lda[0:n_amostras,0], scores_lda[0:n_amostras,1], 'b.', label='Falha 5')
plt.plot(scores_lda[n_amostras:2*n_amostras,0], scores_lda[n_amostras:2*n_amostras,1], 'r.', label='Falha 10')
plt.plot(scores_lda[2*n_amostras:3*n_amostras,0], scores_lda[2*n_amostras:3*n_amostras,1], 'm.', label='Falha 19')
plt.legend()

# 17. Calcular limite T² para Falha 5
alpha = 0.01
k = 2
Nj = n_amostras
T2_CL = k*(Nj**2-1)*scipy.stats.f.ppf(1-alpha,k,Nj-k)/(Nj*(Nj-k))
media_f5 = np.mean(scores_lda[0:n_amostras], axis=0)
cov_f5 = np.cov(scores_lda[0:n_amostras].T)

# 18. Testar dados da falha 5
teste_f5 = np.loadtxt('d05_te.txt')[160:, :]
xmeas = teste_f5[:, 0:22]
xmv = teste_f5[:, 41:52]
teste_f5_completo = np.hstack((xmeas, xmv))
teste_f5_normalizado = escalador.transform(teste_f5_completo)
scores_teste_f5 = lda.transform(teste_f5_normalizado)

# 19. Calcular estatística T²
T2 = []
for i in range(scores_teste_f5.shape[0]):
    delta = scores_teste_f5[i] - media_f5
    T2.append(np.dot(np.dot(delta[np.newaxis,:], np.linalg.inv(cov_f5)), delta[np.newaxis,:].T)[0][0])

T2 = np.array(T2)
fora = T2 > T2_CL
dentro = T2 <= T2_CL

# 20. Visualizar T²
plt.figure()
plt.plot(scores_teste_f5[dentro,0], scores_teste_f5[dentro,1], 'k.', label='dentro do limite')
plt.plot(scores_teste_f5[fora,0], scores_teste_f5[fora,1], 'b.', label='fora do limite')
plt.xlabel('FD1 (teste)')
plt.ylabel('FD2 (teste)')
plt.legend()
