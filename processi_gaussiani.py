import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io
import sys
import os
import pandas as pd
import argparse
import re
from scipy.interpolate import interp1d
# Nuovi import per i Processi Gaussiani
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# =============================================================================
# MODELLO FISICO
# =============================================================================
def black_body(x, A, T):
    """ Modello teorico: Legge di Planck (Corpo Nero) """
    lam = x * 1e-9 
    h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
    
    exponent = (h * c_vel) / (lam * kB * T)
    exponent = np.clip(exponent, None, 7000) 
    planck = (2 * h * c_vel**2) / (lam**5 * (np.exp(exponent) - 1))
    return A * planck

def loss_temperature_only(params, batch_x, batch_y, T_predicted, y_avg2):
    """
    Funzione di costo ottimizzata per trovare SOLO T e A.
    Il filtro viene estratto dinamicamente come media dei residui.
    """
    num_file = len(batch_x)
    T_array = params[:num_file]           
    A = params[num_file]                  
    
    # Calcoliamo il filtro stimato empirico per questo set di T e A
    filtri_stimati = []
    for i in range(num_file):
        bb = black_body(batch_x[i], A, T_array[i])
        # Evitiamo divisioni per zero
        bb = np.where(bb == 0, 1e-10, bb)
        filtri_stimati.append(batch_y[i] / bb)
    
    # Il filtro comune è la media dei filtri stimati dai vari file
    filtro_comune_medio = np.mean(filtri_stimati, axis=0)
    
    # Calcolo del costo
    cost = 0.0
    for i in range(num_file):
        y_pred = black_body(batch_x[i], A, T_array[i]) * filtro_comune_medio
        cost += np.sum((batch_y[i] - y_pred) ** 2) / y_avg2[i]
        cost += 50.0 * ((T_predicted[i] - T_array[i]) / T_predicted[i]) ** 2
        
    return cost

def interpolate_temperature(tabella_calibrazione, r_val):
    maschera = tabella_calibrazione[:, 0] >= 300.0
    tabella_filtrata = tabella_calibrazione[maschera]
    temp_col = tabella_filtrata[:, 0]
    res_col = tabella_filtrata[:, 1]
    indici_ordinati = np.argsort(res_col)
    f_interp = interp1d(res_col[indici_ordinati], temp_col[indici_ordinati], kind='linear', bounds_error=False, fill_value="extrapolate")
    return float(f_interp(r_val))

def estrai_voltaggio_da_nome(nome_file):
    match = re.search(r"([0-9.]+)\s*V", nome_file)
    if match: return float(match.group(1))
    raise ValueError(f"Impossibile trovare il valore di tensione nel nome del file: {nome_file}")

def trova_resistenza_da_voltaggio(tabella_vr, v_val):
    f_vr = interp1d(tabella_vr[:, 0], tabella_vr[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    return float(f_vr(v_val))

# =============================================================================
# GESTIONE ARGOMENTI TERMINALE
# =============================================================================
parser = argparse.ArgumentParser(description="Batch fit con Processi Gaussiani per la risposta del filtro.")
parser.add_argument('-s', '--spectra', nargs='+', required=True, help="Lista dei file CSV degli spettri")
parser.add_argument('-t', '--table', type=str, required=True, help="File .dat tabulato con T e R")
parser.add_argument('-vr', '--voltaggio_resistenza', type=str, required=True, help="File .dat tabulato con V e R")
parser.add_argument('-r', '--resistance', type=float, required=True, help="Resistenza reale a 300K")
args = parser.parse_args()

# [1. Caricamento e Riscalamento Tabelle - Identico al tuo precedente]
if not os.path.exists(args.table) or not os.path.exists(args.voltaggio_resistenza):
    print("Errore: File di calibrazione mancanti.")
    sys.exit(1)

df_tabella = pd.read_csv(args.table, sep=r"\s+", header=None, comment='#')
tabella_TR = df_tabella.to_numpy()
R_room_reale = args.resistance
idx_300K = np.where(tabella_TR[:, 0] == 300.0)[0]
R_tabella_300K = tabella_TR[idx_300K[0], 1] if len(idx_300K) > 0 else tabella_TR[np.abs(tabella_TR[:, 0] - 300.0).argmin(), 1]
tabella_TR[:, 1] = (tabella_TR[:, 1] / R_tabella_300K) * R_room_reale

df_vr = pd.read_csv(args.voltaggio_resistenza, sep=r"\s+", header=None, comment='#')
tabella_VR = df_vr.to_numpy()

x_lista, y_lista, file_validi, v_lista, r_lista, T_lista = [], [], [], [], [], []

for percorso in args.spectra:
    if not os.path.exists(percorso): continue
    nome_file = os.path.basename(percorso)
    try:
        v_estratto = estrai_voltaggio_da_nome(nome_file)
        r_reale_calcolata = trova_resistenza_da_voltaggio(tabella_VR, v_estratto)
        df = pd.read_csv(percorso, sep=r"\s+")
        if "Wavelength (nm)" not in df.columns: continue
        
        x_lista.append(df["Wavelength (nm)"].values)
        y_lista.append(df["Intensity"].values)
        v_lista.append(v_estratto)
        r_lista.append(r_reale_calcolata)
        file_validi.append(nome_file)
        T_lista.append(interpolate_temperature(tabella_TR, r_reale_calcolata))
    except Exception as e:
        print(f"Errore lettura {nome_file}: {e}")

num_file = len(x_lista)
y_average2 = np.power([np.average(y) for y in y_lista], 2)

# =============================================================================
# OPTIMIZATION PHASE 1: Trovare Temperature e Ampiezza A
# =============================================================================
print("\n[CONFIG] Fase 1: Ottimizzazione delle temperature dei corpi neri...")
stima_iniziale = T_lista + [4.76e-7]

# Utilizziamo Powell per convergere sulle temperature reali liberi da vincoli parametrici del filtro
result = minimize(
    loss_temperature_only, 
    stima_iniziale, 
    args=(x_lista, y_lista, T_lista, y_average2), 
    method='Powell'
)

T_fit = result.x[:num_file]
A_fit = result.x[num_file]

# =============================================================================
# OPTIMIZATION PHASE 2: Addestramento del Processo Gaussiano sul Filtro
# =============================================================================
print("\n[CONFIG] Fase 2: Addestramento del Processo Gaussiano sul filtro comune...")

# Calcoliamo i vettori del filtro empirico sui punti dati per estrarre il dataset del GP
X_gp = x_lista[0].reshape(-1, 1) # Assumiamo la stessa griglia x per i file, o usiamo il primo come riferimento
filtri_sperimentali = []
for i in range(num_file):
    bb_ottimizzato = black_body(x_lista[i], A_fit, T_fit[i])
    filtri_sperimentali.append(y_lista[i] / np.where(bb_ottimizzato == 0, 1e-10, bb_ottimizzato))

Y_gp = np.mean(filtri_sperimentali, axis=0)

# Definiamo il Kernel del GP: RBF (smoothness) + WhiteKernel (assorbimento rumore ad alta frequenza)
# Lunghezza di scala iniziale a 5.0 nm (aggiustabile a piacimento)
kernel = RBF(length_scale=5.0, length_scale_bounds=(0.5, 50.0)) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=42)
gp.fit(X_gp, Y_gp)

print(f"[GP INFO] Kernel ottimizzato finale: {gp.kernel_}")

# =============================================================================
# ESPORTAZIONE ED OUTPUT
# =============================================================================
wl_min, wl_max = min(np.min(x) for x in x_lista), max(np.max(x) for x in x_lista)
wl_min_round, wl_max_round = np.floor(wl_min * 10) / 10, np.ceil(wl_max * 10) / 10
x_uniforme = np.linspace(wl_min_round, wl_max_round, int(round((wl_max_round - wl_min_round) / 0.1)) + 1)

# Predizione del filtro tramite GP (con deviazione standard)
y_filtro_esportazione, y_std = gp.predict(x_uniforme.reshape(-1, 1), return_std=True)

df_filtro = pd.DataFrame({"Wavelength (nm)": x_uniforme, "Intensity": y_filtro_esportazione, "Uncertainty_Std": y_std})
df_filtro.to_csv("filtro_stimato_gp_passo_0.1nm.csv", sep=" ", index=False)
print(f"[SUCCESS] Filtro GP salvato con successo!")

# =============================================================================
# PLOTTING
# =============================================================================
# Grafico 1: Fit complessivo contro Dati
plt.figure(figsize=(12, 5))
for i in range(num_file):
    plt.scatter(x_lista[i], y_lista[i], alpha=0.3, s=1, label=f'Dati {file_validi[i]}')
    y_pred_gp = black_body(x_lista[i], A_fit, T_fit[i]) * gp.predict(x_lista[i].reshape(-1, 1))
    plt.plot(x_lista[i], y_pred_gp, color='black', lw=1.5)
plt.title('Fit Globale combinato con Processo Gaussiano')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--')

# Grafico 2: Risposta del Filtro con Banda di Incertezza (95% CI)
plt.figure(figsize=(12, 5))
plt.plot(X_gp, Y_gp, 'o', alpha=0.3, label='Filtro Empirico Medio', markersize=2)
plt.plot(x_uniforme, y_filtro_esportazione, color='crimson', lw=2, label='Filtro Ottimizzato GP')
plt.fill_between(x_uniforme, y_filtro_esportazione - 1.96 * y_std, y_filtro_esportazione + 1.96 * y_std, color='crimson', alpha=0.2, label='Incertezza GP (95%)')
plt.title('Risposta del Filtro Estratta da Processo Gaussiano')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Response')
plt.legend()
plt.grid(True, linestyle='--')

plt.show()