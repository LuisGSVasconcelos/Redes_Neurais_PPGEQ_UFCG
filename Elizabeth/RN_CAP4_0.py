import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Carregar sinal ruidoso
file_path = "noisy_flow_signal.csv"  # Certifique-se de que o arquivo está no mesmo diretório do script
noisy_signal = np.loadtxt(file_path, delimiter=',')

# Aplicar média móvel (SMA)
window_size = 15
smoothed_signal_MA = pd.DataFrame(noisy_signal).rolling(window_size).mean().values

# Aplicar filtro de Savitzky-Golay
smoothed_signal_SG = savgol_filter(noisy_signal, window_length=15, polyorder=2)

# Plotar os sinais
plt.figure(figsize=(10, 5))
plt.plot(noisy_signal, label='Sinal Ruidoso', alpha=0.5)
plt.plot(smoothed_signal_MA, label=f'Sinal Suavizado (SMA, janela={window_size})', linewidth=2)
plt.plot(smoothed_signal_SG, label='Sinal Suavizado (Savitzky-Golay)', linewidth=2, linestyle='dashed')
plt.legend()
plt.xlabel('Tempo')
plt.ylabel('Intensidade do Sinal')
plt.title('Filtragem com Média Móvel e Savitzky-Golay')
plt.show()



