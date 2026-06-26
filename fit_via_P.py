import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io
import sys
import os
import math
import numba
import pandas as pd
import argparse
import re
from scipy.interpolate import interp1d

# =============================================================================
# LOSS JIT ULTRA-VELOCE CON MATRICE DI MASCHERAMENTO
# =============================================================================
@numba.njit(fastmath=True)
def loss_function_masked_jit(params, P_array, Y_matrix, Mask_matrix, y_average2, planck_factor1, planck_factor2, x_shared, n_sig):
    num_file = Y_matrix.shape[0]
    N_wl = Y_matrix.shape[1]
    
    T_array = params[:num_file]
    A = params[num_file]
    K_rad = params[num_file+1] * 1e-13
    K_cond = params[num_file+2] * 1e-4
    funzioni_filtro = params[num_file+3:]
    
    if K_rad < 1e-25 or K_cond < 0.0 or A < 0.0: 
        return 1e20

    # Calcolo del filtro UNA SOLA VOLTA sulla griglia globale
    filtro = np.zeros(N_wl)
    for i in range(0, n_sig * 3, 3):
        amp = funzioni_filtro[i]
        lam_cut = funzioni_filtro[i+1]
        sigma = funzioni_filtro[i+2]
        s = sigma if sigma > 0 else 2.0
        for j in range(N_wl):
            filtro[j] += amp * (1.0 + math.erf((x_shared[j] - lam_cut) / s))

    # Calcolo dei residui pesati solo sulle zone valide
    cost = 0.0
    for i in range(num_file):
        T = T_array[i]
        P_target = P_array[i]
        
        residual_sum = 0.0
        for j in range(N_wl):
            if Mask_matrix[i, j] == 1:
                exponent = planck_factor2[j] / T
                if exponent > 7000.0: 
                    exponent = 7000.0
                
                planck = planck_factor1[j] / (math.exp(exponent) - 1.0)
                y_pred = A * planck * filtro[j]
                residual_sum += ((Y_matrix[i, j] - y_pred)) ** 2
            
        cost += residual_sum/ y_average2[i]
        
        P_modello = K_rad * (T**4) + K_cond * (T - 300.0)
        cost += 300 * ((P_modello - P_target) / P_target)**2

    return cost

def somma_sigmoidi(x, *parametri_sigmoidi):
    filtro = np.zeros_like(x, dtype=float)
    for i in range(0, len(parametri_sigmoidi), 3):
        a = parametri_sigmoidi[i]
        lam_cut = parametri_sigmoidi[i+1]
        sigma = parametri_sigmoidi[i+2]
        s = np.where(sigma <= 0, 2, sigma)
        from scipy.special import erf
        filtro += a * (1.0 + erf((x - lam_cut) / s))
    return filtro

def black_body(x, A, T):
    lam = x * 1e-9 
    h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
    exponent = (h * c_vel) / (lam * kB * T)
    exponent = np.clip(exponent, None, 7000) 
    planck = (2 * h * c_vel**2) / (lam**5 * (np.exp(exponent) - 1))
    return A * planck

def teorical_model(x, T, A, n_sig, *rest_params):
    return black_body(x, A, T) * somma_sigmoidi(x, *rest_params)

def interpolate_temperature(tabella_calibrazione, r_val):
    maschera = tabella_calibrazione[:, 0] >= 300.0
    tabella_filtrata = tabella_calibrazione[maschera]
    f_interp = interp1d(tabella_filtrata[:, 1], tabella_filtrata[:, 0], kind='linear', bounds_error=False, fill_value="extrapolate")
    return float(f_interp(r_val))

def estrai_voltaggio_da_nome(nome_file):
    match = re.search(r"([0-9.]+)\s*V", nome_file)
    if match: return float(match.group(1))
    raise ValueError(f"Tensione non trovata in: {nome_file}")

def trova_resistenza_da_voltaggio(tabella_vr, v_val):
    f_vr = interp1d(tabella_vr[:, 0], tabella_vr[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    return float(f_vr(v_val))

# =============================================================================
# PARSING ARGOMENTI E CARICAMENTO DATASET
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--spectra', nargs='+', required=True)
parser.add_argument('-p', '--powers', type=str, required=True)
parser.add_argument('-t', '--table', type=str, required=True)
parser.add_argument('-vr', '--voltaggio_resistenza', type=str, required=True)
parser.add_argument('-r', '--resistance', type=float, required=True)
args = parser.parse_args()

df_p = pd.read_csv(args.powers, sep=r"\s+", header=None, comment='#')
diz_p = dict(zip(df_p[0].astype(str), df_p[1].astype(float)))

tabella_TR = pd.read_csv(args.table, sep=r"\s+", header=None, comment='#').to_numpy()
R_room_reale = args.resistance
idx_300K = np.where(tabella_TR[:, 0] == 300.0)[0]
R_tab_300 = tabella_TR[idx_300K[0], 1] if len(idx_300K) > 0 else tabella_TR[np.abs(tabella_TR[:, 0] - 300.0).argmin(), 1]
tabella_TR[:, 1] = (tabella_TR[:, 1] / R_tab_300) * R_room_reale

tabella_VR = pd.read_csv(args.voltaggio_resistenza, sep=r"\s+", header=None, comment='#').to_numpy()

x_raw_list, y_raw_list = [], []
file_validi, T_lista, P_lista = [], [], []

for percorso in args.spectra:
    if not os.path.exists(percorso): continue
    nome_file = os.path.basename(percorso)
    if nome_file not in diz_p: continue
    try:
        v_estratto = estrai_voltaggio_da_nome(nome_file)
        r_calc = trova_resistenza_da_voltaggio(tabella_VR, v_estratto)
        df = pd.read_csv(percorso, sep=r"\s+")
        if "Wavelength (nm)" not in df.columns: continue
        
        x_raw_list.append(df["Wavelength (nm)"].values)
        y_raw_list.append(df["Intensity"].values)
        P_lista.append(diz_p[nome_file])
        file_validi.append(nome_file)
        #T_lista.append(interpolate_temperature(tabella_TR, r_calc)) WRONG
        T_lista.append((r_calc/12.8 - 1.0)/0.0045 + 300.0) # Formula empirica per stimare T da R
    except Exception as e:
        print(f"Salto file {nome_file}: {e}")

num_file = len(x_raw_list)
if num_file == 0: sys.exit("Nessun file valido.")

# =============================================================================
# COSTRUZIONE DELLA GRIGLIA GLOBALE ALLARGATA UNIFICATA
# =============================================================================
wl_assoluto_min = min(np.min(x) for x in x_raw_list)
wl_assoluto_max = max(np.max(x) for x in x_raw_list)

passo_nominale = 0.1
num_punti_totali = int(np.floor((wl_assoluto_max - wl_assoluto_min) / passo_nominale)) + 1
x_shared = np.linspace(wl_assoluto_min, wl_assoluto_max, num_punti_totali)

Y_matrix = np.zeros((num_file, num_punti_totali), dtype=np.float64)
Mask_matrix = np.zeros((num_file, num_punti_totali), dtype=np.int32)

for i in range(num_file):
    f_interp = interp1d(x_raw_list[i], y_raw_list[i], kind='linear', bounds_error=False, fill_value=0.0)
    Y_matrix[i] = f_interp(x_shared)
    
    X_min_locale = np.min(x_raw_list[i])
    X_max_locale = np.max(x_raw_list[i])
    Mask_matrix[i] = ((x_shared >= X_min_locale) & (x_shared <= X_max_locale)).astype(np.int32)

P_array = np.array(P_lista)
y_average2 = np.array([np.power(np.average(y), 2) for y in y_raw_list])

print(f"\n[PIPELINE VETTORIZZATA] Unione canali Raman completata.")
print(f"[PIPELINE VETTORIZZATA] Range totale: {wl_assoluto_min:.2f} nm -> {wl_assoluto_max:.2f} nm ({num_punti_totali} punti).\n")

h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
lam_meters = x_shared * 1e-9
planck_factor1 = (2.0 * h * c_vel**2) / (lam_meters**5)
planck_factor2 = (h * c_vel) / (lam_meters * kB)

# =============================================================================
# COSTRUZIONE GEOMETRICA DEL FILTRO ADATTIVO E OTTIMIZZAZIONE
# =============================================================================
N_SIGMOIDI = 12
centri_dinamici = np.linspace(wl_assoluto_min + 3, wl_assoluto_max - 3, N_SIGMOIDI)

p0_sigmoide = []
for c in centri_dinamici:
    p0_sigmoide += [1.0, c, 2.5]

stima_iniziale = T_lista + [1e-7] + [3.0] + [1.0] + p0_sigmoide

print("Avvio ottimizzazione vettorizzata ad alta velocità...")
result = minimize(
    loss_function_masked_jit, 
    stima_iniziale, 
    args=(P_array, Y_matrix, Mask_matrix, y_average2, planck_factor1, planck_factor2, x_shared, N_SIGMOIDI), 
    method='Powell',
    options={'maxfev': 150000, 'maxiter': 150000, 'xtol': 1e-17, 'ftol': 1e-17}
)

# =============================================================================
# ESTRAZIONE PARAMETRI, RIEPILOGO OUTPUT E GRAFICA
# =============================================================================
T_fit = result.x[:num_file]
A_fit = result.x[num_file]
K_rad_fit = result.x[num_file+1] * 1e-13
K_cond_fit = result.x[num_file+2] * 1e-4
sigmoidi_ottimizzate = result.x[num_file+3:]

# -------------- CONSOLE REPORT --------------
print("\n" + "="*75)
print(" RIEPILOGO OTTIMIZZAZIONE GLOBALE VETTORIZZATA ".center(75))
print("="*75)
status_str = "SUCCESSO" if result.success else "FALLITO"
print(f" Status Ottimizzazione  : {status_str} ({result.message})")
print(f" Iterazioni / Fev       : {result.nit} / {result.nfev}")
print(f" Loss Function Finale   : {result.fun:.4e}")
print("-" * 75)
print(" PARAMETRI DI SISTEMA CONDIVISI ".center(75))
print("-" * 75)
print(f" Ampiezza Globale (A)   : {A_fit:.4e}")
print(f" Cost. Irraggiamento    : {K_rad_fit:.4e} W/K^4")
print(f" Cost. Conduzione       : {K_cond_fit:.4e} W/K")
print("-" * 75)
print(" RISULTATI TEMPERATURE PER FILE ".center(75))
print("-" * 75)
print(f"{'Nome File':<25} | {'P (mW)':<8} | {'T Guess':<9} | {'T Fit':<9} | {'Delta T':<8}")
print("-" * 75)

for i in range(num_file):
    t_g = T_lista[i]
    t_f = T_fit[i]
    delta = t_f - t_g
    power = P_lista[i]
    print(f"{file_validi[i]:<25} | {power:>6.1f}   | {t_g:>7.1f} K | {t_f:>7.1f} K | {delta:>+6.1f} K")
print("="*75 + "\n")
# --------------------------------------------

y_filtro_globale = somma_sigmoidi(x_shared, *sigmoidi_ottimizzate)
pd.DataFrame({"Wavelength (nm)": x_shared, "Filter_Response": y_filtro_globale}).to_csv("filtro_Raman_esteso_veloce.csv", sep=" ", index=False)

plt.figure(figsize=(12, 5))
for i in range(num_file):
    idx_validi = (Mask_matrix[i] == 1)
    plt.scatter(x_shared[idx_validi], Y_matrix[i, idx_validi], alpha=0.35, s=1)
    plt.plot(x_shared[idx_validi], teorical_model(x_shared[idx_validi], T_fit[i], A_fit, N_SIGMOIDI, *sigmoidi_ottimizzate), color='black', lw=1.2, alpha=0.8)
plt.title('Batch Fit Ottimizzato e Veloce (900 cm⁻¹ + 1200 cm⁻¹)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--')

plt.figure(figsize=(12, 4))
plt.plot(x_shared, y_filtro_globale, color='crimson', lw=2.5)
plt.title('Filtro di Risposta di Sistema Unificato (Calcolo Veloce)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Response')
plt.grid(True, linestyle='--')
plt.show()