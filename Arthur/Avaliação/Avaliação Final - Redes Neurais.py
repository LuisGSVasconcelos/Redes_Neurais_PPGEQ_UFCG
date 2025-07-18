# O Script inclui: 
# organização dos resultados;
# curvas de aprendizado;
# parity plots conjuntos;
# ranking automático;
# logs e exportação estruturada.

import warnings
warnings.filterwarnings("ignore")

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error as mse, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#  1. SEEDS GLOBAIS 
np.random.seed(42)
random.seed(42)

# As linhas abaixo foram postas em comentário, mas caso queira estabelecer comunicação com o Aspen será esse procedimento:

# 2. INTERFACE ASPEN PYTHON
# file_path = r'C:\Users\arthu\OneDrive\Documentos\AspenTech\Aspen Plus V14.0\XXXXX.apwz' # Caminho do diretório do seu arquivo Aspen
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f'O arquivo {file_path} não foi encontrado.')

# total_start = time.perf_counter()
# try:
#     aspen = win32.Dispatch('Apwn.Document')
#     aspen.InitFromFile(file_path)
#     aspen.Visible = True
#     time.sleep(2)

#     # Define parâmetros
#     params = {
#         r'\Data\Streams\FEED\Input\TEMP\MIXED': 298.15,
#         r'\Data\Streams\FEED\Input\PRES\MIXED': 1.5,
#         r'\Data\Streams\EO\Input\TEMP\MIXED': 298.15,
#         r'\Data\Streams\EO\Input\PRES\MIXED': 1.5,
#         r'\Data\Blocks\C1\Input\NSTAGE': 27,
#         r'\Data\Blocks\C1\Input\BASIS_RR': 0.821,
#         r'\Data\Blocks\C1\Input\BASIS_D': 65.981,
#         r'\Data\Blocks\C2\Input\NSTAGE': 28,
#         r'\Data\Blocks\C2\Input\BASIS_D': 32.992,
#         r'\Data\Blocks\C3\Input\NSTAGE': 25,
#         r'\Data\Blocks\C3\Input\BASIS_D': 32.999,
#         r'\Data\Blocks\COOLER\Input\TEMP': 300,
#         r'\Data\Blocks\COOLER\Input\PRES': 1.0
#     }
#     for path, value in params.items():
#         node = aspen.Tree.FindNode(path)
#         node.Value = value

#     aspen.Engine.Run2()
#     while aspen.Engine.IsRunning:
#         time.sleep(5)

# except Exception as e:
#     print(f"Erro: {e}")
# finally:
#     total_end = time.perf_counter()
#     print(f"Tempo total: {total_end - total_start:.2f} segundos.")

#  3. DIRETÓRIOS DE SAÍDA 
output_dir = "ML_results"
plots_dir = os.path.join(output_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

#  4. ARQUIVOS DE ENTRADA #ATENÇÃO: Aqui você deve colocar o caminho do diretório que se encontra os arquivos .xlsx, eles serão base para o código
file_paths = {
    "C1": {
        "a": r'C:\XXXXX\C1 - Liquid.xlsx',
        "b": r'C:\XXXXX\C1 - TPFQ.xlsx',
        "target": ["EG"]
    },
    "C2": {
        "a": r'C:\XXXXX\C2 - Liquid.xlsx',
        "b": r'C:\XXXXX\C2 - TPFQ.xlsx',
        "target": ["THF"]
    },
    "C3": {
        "a": r'C:\XXXXX\C3 - Liquid.xlsx',
        "b": r'C:\XXXXX\C3 - TPFQ.xlsx',
        "target": ["ETOH", "DMSO"]
    }
}

#  5. FUNÇÃO: GRÁFICOS DE COLUNAS (COMPOSIÇÃO E TEMPERATURA) 
def plot_colunas(df_c1_a, df_c1_b, df_c2_a, df_c2_b, df_c3_a, df_c3_b):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    # C1
    ax = axs[0, 0]
    ax.plot(df_c1_a['Stage'], df_c1_a['THF'], 'k-s', label='THF')
    ax.plot(df_c1_a['Stage'], df_c1_a['ETOH'], 'r-o', label='ethanol')
    ax.plot(df_c1_a['Stage'], df_c1_a['WATER'], 'm-^', label='water')
    ax.plot(df_c1_a['Stage'], df_c1_a['EO'], 'b-v', label='EO')
    ax.plot(df_c1_a['Stage'], df_c1_a['EG'], color='purple', marker='D', label='EG')
    ax.set_xlim(0, 30)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xlabel('Stage of column C1')
    ax.set_ylabel('Liquid composition')
    ax.legend(loc='best')
    ax = axs[0, 1]
    ax.plot(df_c1_b['Stage'], df_c1_b['Temperature'], 'k-s', label='Temperature')
    ax.set_xlim(0, 30)
    ax.set_ylim(320, 480)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(320, 481, 40))
    ax.set_xlabel('Stage of column C1')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='best')
    # C2
    ax = axs[1, 0]
    ax.plot(df_c2_a['Stage'], df_c2_a['THF'], 'k-s', label='THF')
    ax.plot(df_c2_a['Stage'], df_c2_a['ETOH'], 'r-o', label='ethanol')
    ax.plot(df_c2_a['Stage'], df_c2_a['WATER'], 'b-^', label='water')
    ax.plot(df_c2_a['Stage'], df_c2_a['EO'], 'm-v', label='EO')
    ax.plot(df_c2_a['Stage'], df_c2_a['EG'], color='green', marker='D', label='EG')
    ax.plot(df_c2_a['Stage'], df_c2_a['DMSO'], color='navy', marker='<', label='DMSO')
    ax.set_xlim(0, 30)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xlabel('Stage of column C2')
    ax.set_ylabel('Liquid composition')
    ax.legend(loc='best')
    ax = axs[1, 1]
    ax.plot(df_c2_b['Stage'], df_c2_b['Temperature'], 'k-s', label='Temperature')
    ax.set_xlim(0, 30)
    ax.set_ylim(320, 400)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks(np.arange(320, 401, 20))
    ax.set_xlabel('Stage of column C2')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='best')
    # C3
    ax = axs[2, 0]
    ax.plot(df_c3_a['Stage'], df_c3_a['THF'], 'k-s', label='THF')
    ax.plot(df_c3_a['Stage'], df_c3_a['ETOH'], 'r-o', label='ethanol')
    ax.plot(df_c3_a['Stage'], df_c3_a['WATER'], 'b-^', label='water')
    ax.plot(df_c3_a['Stage'], df_c3_a['EG'], 'm-v', label='EG')
    ax.plot(df_c3_a['Stage'], df_c3_a['DMSO'], color='green', marker='D', label='DMSO')
    ax.set_xlim(0, 25)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.set_xlabel('Stage of column C3')
    ax.set_ylabel('Liquid composition')
    ax.legend(loc='best')
    ax = axs[2, 1]
    ax.plot(df_c3_b['Stage'], df_c3_b['Temperature'], 'k-s', label='Temperature')
    ax.set_xlim(0, 25)
    ax.set_ylim(340, 480)
    ax.set_xticks(np.arange(0, 26, 5))
    ax.set_yticks(np.arange(340, 481, 20))
    ax.set_xlabel('Stage of column C3')
    ax.set_ylabel('Temperature (K)')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'colunas_composicao_temperatura.png'), dpi=300, bbox_inches='tight')
    plt.show()

#  6. FUNÇÃO: CURVA DE APRENDIZADO 
def plot_learning_curves_subplot(file_paths, model_dict, plots_dir):
    for grupo, arquivos in file_paths.items():
        n_vars = len(arquivos["target"])
        fig, axs = plt.subplots(1, n_vars, figsize=(7*n_vars, 5))
        if n_vars == 1:
            axs = [axs]
        for idx, alvo in enumerate(arquivos["target"]):
            df_b = pd.read_excel(arquivos["b"])
            if alvo not in df_b.columns:
                print(f"Coluna '{alvo}' não encontrada em {arquivos['b']}. Colunas disponíveis: {df_b.columns.tolist()}")
                continue
            X = df_b[['Temperature', 'Pressure', 'Heat duty']]
            y = df_b[alvo]
            for name, model in model_dict.items():
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X, y, cv=5, scoring="neg_mean_squared_error",
                    train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1)
                train_scores_mean = -np.mean(train_scores, axis=1)
                test_scores_mean = -np.mean(test_scores, axis=1)
                axs[idx].plot(train_sizes, train_scores_mean, 'o-', label=f"Treino - {name}")
                axs[idx].plot(train_sizes, test_scores_mean, 'o--', label=f"Validação - {name}")
            axs[idx].set_xlabel("Nº of Training Samples")
            axs[idx].set_ylabel("RMSE")
            axs[idx].set_title(f"Learning Curve - {grupo} - {alvo}")
            axs[idx].legend(loc="best", fontsize=8)
            axs[idx].grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{grupo}_learning_curves_subplot.png"), dpi=300, bbox_inches='tight')
        plt.show()

#  7. FUNÇÃO: PARITY PLOTS LADO A LADO PARA TODOS OS MODELOS 
def plot_parity_all(y_tests, y_preds, model_names, dataset_label, save_path=None):
    n = len(y_tests)
    fig, axs = plt.subplots(1, n, figsize=(6*n, 6))
    if n == 1:
        axs = [axs]
    for i, (y_true, y_pred, name) in enumerate(zip(y_tests, y_preds, model_names)):
        axs[i].scatter(y_true, y_pred, alpha=0.75, edgecolor='k', s=60)
        axs[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r', lw=2)
        axs[i].set_xlabel("Measured", fontsize=12)
        axs[i].set_ylabel("Predicted", fontsize=12)
        axs[i].set_title(f"{name}\n({dataset_label})", fontsize=14)
        axs[i].grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

#  8. FUNÇÃO: AVALIAÇÃO DE MODELOS + LOGS + PARITY 
def train_and_evaluate(X, y, models, dataset_label, save_graphs=True):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    results, y_tests, y_preds, model_names = [], [], [], []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - start
        rmse = mse(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)
        results.append({
            "Model": name,
            "RMSE": rmse,
            "R2": r2,
            "Tempo Treino (s)": elapsed
        })
        y_tests.append(y_test)
        y_preds.append(preds)
        model_names.append(name)
        try:
            if hasattr(model, 'named_steps'):
                m = model.named_steps['model']
            else:
                m = model
            if hasattr(m, 'feature_importances_'):
                print(f"\n{name} - Feature Importances ({dataset_label}):")
                for feat, imp in zip(X.columns, m.feature_importances_):
                    print(f"  {feat}: {imp:.4f}")
        except Exception:
            pass
    parity_path = os.path.join(plots_dir, f"{dataset_label}_parity.png")
    plot_parity_all(y_tests, y_preds, model_names, dataset_label, save_path=parity_path if save_graphs else None)
    return results

#  9. DICIONÁRIO DE MODELOS COM PIPELINE 
model_dict = {
    "Random Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=42))
    ]),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(random_state=42, verbosity=0))
    ]),
    "AdaBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', AdaBoostRegressor(random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42))
    ])
}

#  10. LEITURA DOS DATAFRAMES E GRÁFICOS DE COLUNAS 
df_c1_a = pd.read_excel(file_paths["C1"]["a"])
df_c1_b = pd.read_excel(file_paths["C1"]["b"])
df_c2_a = pd.read_excel(file_paths["C2"]["a"])
df_c2_b = pd.read_excel(file_paths["C2"]["b"])
df_c3_a = pd.read_excel(file_paths["C3"]["a"])
df_c3_b = pd.read_excel(file_paths["C3"]["b"])

plot_colunas(df_c1_a, df_c1_b, df_c2_a, df_c2_b, df_c3_a, df_c3_b)

#  11. TREINAMENTO, AVALIAÇÃO, LOG E RANKING 
resultados_gerais = []
for conjunto, arquivos in file_paths.items():
    df_a = pd.read_excel(arquivos["a"])
    df_b = pd.read_excel(arquivos["b"])
    for alvo in arquivos["target"]:
        if alvo not in df_a.columns:
            print(f"[ERRO] Coluna '{alvo}' não encontrada em {arquivos['a']}. Colunas disponíveis: {df_a.columns.tolist()}")
            continue
        # Realize o merge pelo Stage para juntar processo e alvo
        df = pd.merge(df_b[['Stage', 'Temperature', 'Pressure', 'Heat duty']],
                      df_a[['Stage', alvo]],
                      on='Stage')
        X = df[['Temperature', 'Pressure', 'Heat duty']]
        y = df[alvo]
        label = f"{conjunto} - {alvo}"
        print(f"\nTreinando para: {label}")
        resultados = train_and_evaluate(X, y, model_dict, label)
        for res in resultados:
            res.update({"Dataset": label, "Group": conjunto, "Variable": alvo})
            resultados_gerais.append(res)

df_resultados = pd.DataFrame(resultados_gerais)
print("\n RESULTADOS POR CONJUNTO E MODELO ")
for grupo in df_resultados["Group"].unique():
    df_group = df_resultados[df_resultados["Group"] == grupo]
    print(f"\nGroup: {grupo}")
    for dataset in df_group["Dataset"].unique():
        df_dataset = df_group[df_group["Dataset"] == dataset]
        print(f"\n{dataset}")
        for _, row in df_dataset.iterrows():
            print(f"{row['Model']}: RMSE = {row['RMSE']:.6f}, R2 = {row['R2']:.6f}")
            
#  12. RANKING DOS MELHORES MODELOS POR VARIÁVEL 
ranking = []
for grupo in df_resultados["Group"].unique():
    df_group = df_resultados[df_resultados["Group"] == grupo]
    for var in df_group["Variable"].unique():
        df_var = df_group[df_group["Variable"] == var]
        best_r2 = df_var.loc[df_var["R2"].idxmax()]
        best_rmse = df_var.loc[df_var["RMSE"].idxmin()]
        ranking.append({
            "Group": grupo,
            "Variable": var,
            "Best R2": best_r2["Model"],
            "R2": best_r2["R2"],
            "Best RMSE": best_rmse["Model"],
            "RMSE": best_rmse["RMSE"]
        })
df_ranking = pd.DataFrame(ranking)

#  13. SALVAMENTO DOS RESULTADOS, RANKING E LOG 
xlsx_path = os.path.join(output_dir, 'resultados_modelos.xlsx')
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    for grupo in df_resultados["Group"].unique():
        df_group = df_resultados[df_resultados["Group"] == grupo]
        df_group.to_excel(writer, sheet_name=grupo, index=False)
    df_ranking.to_excel(writer, sheet_name='Ranking', index=False)
print(f"\nArquivo Excel salvo em: {xlsx_path}")

log_path = os.path.join(output_dir, "resumo_modelagem.txt")
with open(log_path, "w") as f:
    f.write(" RESULTADOS POR CONJUNTO E MODELO \n")
    for grupo in df_resultados["Group"].unique():
        df_group = df_resultados[df_resultados["Group"] == grupo]
        f.write(f"\nGroup: {grupo}\n")
        for dataset in df_group["Dataset"].unique():
            df_dataset = df_group[df_group["Dataset"] == dataset]
            f.write(f"\n{dataset}\n")
            for _, row in df_dataset.iterrows():
                f.write(f"{row['Model']}: RMSE = {row['RMSE']:.6f}, R2 = {row['R2']:.6f}, Tempo Treino = {row['Tempo Treino (s)']:.3f}s\n")
    f.write("\n RANKING MELHORES MODELOS POR VARIÁVEL \n")
    f.write(df_ranking.to_string(index=False))
print(f"Log resumo salvo em: {log_path}")

#  14. GRÁFICO FINAL COMPARATIVO (R2 x RMSE) 
sns.set(style="whitegrid", context="talk", font_scale=1.1)
for grupo in df_resultados["Group"].unique():
    df_plot = df_resultados[df_resultados["Group"] == grupo]
    plt.figure(figsize=(10, 7))
    ax = sns.scatterplot(
        data=df_plot,
        x="RMSE",
        y="R2",
        hue="Model",
        style="Variable",
        s=180,
        palette="Set2",
        legend="full"
    )
    altura_base = 0.05
    incremento = 0.15
    for idx, alvo in enumerate(df_plot["Variable"].unique()):
        sub_df = df_plot[df_plot["Variable"] == alvo]
        maior_r2 = sub_df.loc[sub_df["R2"].idxmax()]
        menor_rmse = sub_df.loc[sub_df["RMSE"].idxmin()]
        ax.annotate(
            f'{alvo} - Best R²:\n{maior_r2["Model"]} ({maior_r2["R2"]:.4f})',
            xy=(1.02, altura_base + idx * incremento),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightgreen'),
            fontsize=10,
            horizontalalignment='left'
        )
        ax.annotate(
            f'{alvo} - Best RMSE:\n{menor_rmse["Model"]} ({menor_rmse["RMSE"]:.4f})',
            xy=(1.02, altura_base + 0.07 + idx * incremento),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'),
            fontsize=10,
            horizontalalignment='left'
        )
    plt.title(f'Model Performance: Column {grupo}')
    plt.xlabel("RMSE")
    plt.ylabel("R²")
    plt.xlim(0, df_resultados["RMSE"].max() * 1.1)
    plt.ylim(df_resultados["R2"].min() - 0.05, 1.05)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{grupo}_desempenho.png"), dpi=300, bbox_inches='tight')
    plt.show()