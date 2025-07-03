import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import scipy.io
import scipy.stats

# -----------------------------
# 1. Carregar dados do arquivo MATLAB
# -----------------------------
dados_matlab = scipy.io.loadmat('MACHINE_Data.mat', struct_as_record=False)
etch = dados_matlab['LAMDATA'][0, 0]
calibration_dataAll = etch.calibration
variable_names = etch.variables

# -----------------------------
# 2. Visualizar sinal de uma variável ao longo do tempo
# -----------------------------
plt.figure()
for expt in range(calibration_dataAll.size):
    plt.plot(calibration_dataAll[expt, 0][:, 6])
plt.xlabel('Tempo (s)')
plt.ylabel(variable_names[6][0])
plt.title('Evolução da variável 6 em calibrações')
plt.show()

# -----------------------------
# 3. Construir matriz "unfolded" dos dados
# -----------------------------
n_vars = variable_names.size - 2
n_samples = 85
rows = []

for expt in range(calibration_dataAll.size):
    dados = calibration_dataAll[expt, 0][5:90, 2:]
    if dados.shape[0] == n_samples:
        rows.append(np.ravel(dados, order='F'))
matriz_unfolded = np.array(rows)

# -----------------------------
# 4. Normalizar os dados
# -----------------------------
scaler = StandardScaler()
dados_norm = scaler.fit_transform(matriz_unfolded)

# -----------------------------
# 5. Aplicar PCA (3 componentes principais)
# -----------------------------
pca = PCA(n_components=3)
scores_train = pca.fit_transform(dados_norm)

# -----------------------------
# 6. K‑Means para agrupamento
# -----------------------------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=100)
labels_km = kmeans.fit_predict(scores_train)

plt.figure()
plt.scatter(scores_train[:, 0], scores_train[:, 1], c=labels_km, cmap='viridis', s=30)
for i, centro in enumerate(kmeans.cluster_centers_):
    plt.scatter(*centro[:2], color='red', marker='*', s=100)
    plt.annotate(f'Cluster {i+1}', centro[:2])
plt.title('K‑Means após PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# -----------------------------
# 7. Método do cotovelo (SSE)
# -----------------------------
SSEs = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, random_state=100).fit(scores_train)
    SSEs.append(km.inertia_)
plt.figure()
plt.plot(range(1, 10), SSEs, marker='o')
plt.title('Método do Cotovelo (SSE)')
plt.xlabel('Número de clusters')
plt.ylabel('SSE')
plt.show()

# -----------------------------
# 8. Índice de Silhueta
# -----------------------------
silh_avg = silhouette_score(scores_train, labels_km)
print(f'Índice médio de silhueta: {silh_avg:.4f}')

plt.figure()
silh_vals = silhouette_samples(scores_train, labels_km)
y_lower = 0
for i in range(n_clusters):
    vals = np.sort(silh_vals[labels_km == i])
    y_upper = y_lower + len(vals)
    plt.barh(range(y_lower, y_upper), vals, height=1.0,
             color=plt.cm.nipy_spectral(i / n_clusters))
    y_lower = y_upper
plt.axvline(silh_avg, color='red', linestyle='--')
plt.title('Gráfico de Silhueta')
plt.xlabel('Coeficiente de Silhueta')
plt.ylabel('Clusters')
plt.show()

# -----------------------------
# 9. DBSCAN nos dados PCA
# -----------------------------
db = DBSCAN(eps=5, min_samples=3).fit(scores_train)
labels_db = db.labels_
print('Clusters DBSCAN:', np.unique(labels_db))

plt.figure()
plt.scatter(scores_train[:, 0], scores_train[:, 1], c=labels_db, cmap='viridis', s=30)
plt.title('DBSCAN sobre dados PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# -----------------------------
# 10. Dados elipsoidais e GMM
# -----------------------------
X, _ = make_blobs(n_samples=1500, random_state=100)
rot = np.array([[0.6, -0.7], [-0.5, 0.7]])
X_elong = X @ rot

plt.figure()
plt.scatter(X_elong[:, 0], X_elong[:, 1], s=10)
plt.title('Dados elipsoidais gerados')
plt.show()

# GMM com 3 componentes
gmm = GaussianMixture(n_components=3, random_state=100)
labels_gmm = gmm.fit_predict(X_elong)

plt.figure()
plt.scatter(X_elong[:, 0], X_elong[:, 1], c=labels_gmm, cmap='viridis', s=10)
for i, m in enumerate(gmm.means_):
    plt.scatter(m[0], m[1], color='red', marker='*', s=100)
    plt.annotate(f'Cluster {i+1}', m[:2])
plt.title('GMM em dados elipsoidais')
plt.show()

# Probabilidades de pertencimento para um ponto
pt = X_elong[1069:1070]
probs = gmm.predict_proba(pt)
print('Probabilidades do ponto:', probs[0])

# -----------------------------
# 11. Número ótimo de componentes via BIC
# -----------------------------
BICs = []
for k in range(1, 10):
    bm = GaussianMixture(n_components=k, random_state=100).fit(X_elong)
    BICs.append(bm.bic(X_elong))
bic_min = min(BICs)
opt_k = BICs.index(bic_min) + 1
plt.figure()
plt.plot(range(1, 10), BICs, marker='o')
plt.scatter(opt_k, bic_min, color='red', s=100)
plt.title('Seleção de modelo GMM via BIC')
plt.xlabel('Componentes')
plt.ylabel('BIC')
plt.show()
print(f'Número ótimo de componentes segundo BIC: {opt_k}')

# -----------------------------
# 12. Distância global de Mahalanobis
# -----------------------------
# Ajustar GMM com componentes ótimos nos dados PCA
gmm = GaussianMixture(n_components=opt_k, random_state=100).fit(scores_train)
probs_train = gmm.predict_proba(scores_train)
Dglob_train = np.zeros(scores_train.shape[0])

for i, x in enumerate(scores_train):
    for c in range(opt_k):
        diff = x - gmm.means_[c]
        invcov = np.linalg.inv(gmm.covariances_[c])
        Dglob_train[i] += probs_train[i, c] * (diff @ invcov @ diff)

N, r = scores_train.shape
alpha = 0.05
Dglob_CL = r * (N**2 - 1) * scipy.stats.f.ppf(1 - alpha, r, N - r) / (N * (N - r))

plt.figure()
plt.plot(Dglob_train, label='D_global')
plt.axhline(Dglob_CL, color='red', linestyle='--', label='Limite 95%')
plt.title('Controle via Distância de Mahalanobis')
plt.xlabel('Amostra')
plt.ylabel('D_global')
plt.legend()
plt.show()

# -----------------------------
# 13. Aplicar em dados de teste
# -----------------------------
test_all = etch.test
rows_t = []
for expt in range(test_all.size):
    d = test_all[expt, 0][5:90, 2:]
    if d.shape[0] == n_samples:
        rows_t.append(np.ravel(d, order='F'))
test_unfold = np.array(rows_t)

scores_test = pca.transform(scaler.transform(test_unfold))
probs_test = gmm.predict_proba(scores_test)
Dglob_test = np.sum(probs_test * 
                   np.array([np.sum((scores_test - m) @ np.linalg.inv(cov) * (scores_test - m), axis=1)
                             for m, cov in zip(gmm.means_, gmm.covariances_)]).T, axis=1)

n_falhas = np.sum(Dglob_test > Dglob_CL)
print(f'Falhas identificadas: {n_falhas} de {len(Dglob_test)} testes')
