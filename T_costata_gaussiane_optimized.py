import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io
import sys
import os
from scipy.special import erf
import pandas as pd
import argparse
import re
from scipy.interpolate import interp1d

def somma_gaussiane(x, *parametri_gaussiane):
    """
    Calcola la somma di N gaussiane standard.
    """
    filtro = np.zeros_like(x, dtype=float)
    for i in range(0, len(parametri_gaussiane), 3):
        a = parametri_gaussiane[i]
        lam_centro = parametri_gaussiane[i+1]
        sigma = parametri_gaussiane[i+2]
        s = np.where(sigma <= 0, 2e-1, sigma)
        filtro += a * np.exp(-((x - lam_centro) / s)**2)
    return filtro

def somma_sigmoidi(x, *parametri_sigmoidi):
    """
    Calcola la somma di N sigmoidi (basate sulla funzione di errore 'erf').
    """
    filtro = np.zeros_like(x, dtype=float)
    for i in range(0, len(parametri_sigmoidi), 3):
        a = parametri_sigmoidi[i]
        lam_cut = parametri_sigmoidi[i+1]
        sigma = parametri_sigmoidi[i+2]
        s = np.where(sigma <= 0, 2, sigma)
        filtro += a * (1.0 + erf((x - lam_cut) / s))
    return filtro

def black_body(x, A, T):
    """
    Modello teorico: Legge di Planck (Corpo Nero)
    """
    lam = x * 1e-9 
    h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
    
    exponent = (h * c_vel) / (lam * kB * T)
    exponent = np.clip(exponent, None, 7000) 
    planck = (2 * h * c_vel**2) / (lam**5 * (np.exp(exponent) - 1))
    return A * planck

def teorical_model(x, T, A, n_sig, n_gauss, *rest_params):
    """
    Modello teorico: Corpo Nero * (Somma Sigmoidi + Somma Gaussiane)
    """    
    cut_sig = n_sig * 3
    params_sig = rest_params[:cut_sig]
    params_gauss = rest_params[cut_sig : cut_sig + (n_gauss * 3)]
    
    filtro = np.zeros_like(x, dtype=float)
    
    if n_sig > 0:
        filtro += somma_sigmoidi(x, *params_sig)
    if n_gauss > 0:
        filtro += somma_gaussiane(x, *params_gauss)
        
    return black_body(x, A, T) * filtro

def loss_function_1_file(file_params, x, y, n_sig, n_gauss):
    """
    Somma dei residui quadratici per un singolo file.
    """
    T = file_params[0]
    A = file_params[1]
    rest_params = file_params[2:]
    
    y_pred = teorical_model(x, T, A, n_sig, n_gauss, *rest_params)
    return np.sum((y - y_pred) ** 2)

def loss_function(params, T_predicted, batch_x, batch_y, y_avg2, n_sig, n_gauss):
    """
    Funzione di costo globale adattiva.
    """
    cost = 0.0
    num_file = len(batch_x)
    
    T_array = params[:num_file]           
    A = params[num_file]                  
    funzioni_filtro = params[num_file+1:] 
    
    for i in range(num_file):
        curr_params = [T_array[i], A] + list(funzioni_filtro)
        
        x_data = batch_x[i]
        y_data = batch_y[i]
        cost += loss_function_1_file(curr_params, x_data, y_data, n_sig, n_gauss) / y_avg2[i]
        cost += 50.0 * ((T_predicted[i] - T_array[i])/T_predicted[i])**2

    return cost

def interpolate_temperature(tabella_calibrazione, r_val):
    maschera = tabella_calibrazione[:, 0] >= 300.0
    tabella_filtrata = tabella_calibrazione[maschera]
    
    temp_col = tabella_filtrata[:, 0]
    res_col = tabella_filtrata[:, 1]
    
    indici_ordinati = np.argsort(res_col)
    res_ordinata = res_col[indici_ordinati]
    temp_ordinata = temp_col[indici_ordinati]
    
    f_interp = interp1d(res_ordinata, temp_ordinata, kind='linear', bounds_error=False, fill_value="extrapolate")
    return float(f_interp(r_val))

def estrai_voltaggio_da_nome(nome_file):
    match = re.search(r"([0-9.]+)\s*V", nome_file)
    if match:
        return float(match.group(1))
    raise ValueError(f"Impossibile trovare il valore di tensione (es. '12V') nel nome del file: {nome_file}")

def trova_resistenza_da_voltaggio(tabella_vr, v_val):
    v_col = tabella_vr[:, 0]
    r_col = tabella_vr[:, 1]
    f_vr = interp1d(v_col, r_col, kind='linear', bounds_error=False, fill_value="extrapolate")
    return float(f_vr(v_val))


# =============================================================================
# GESTIONE ARGOMENTI TERMINALE CON ARGPARSE
# =============================================================================
parser = argparse.ArgumentParser(description="Batch fit con temperature come guess da calibrazione V-R ed R-T.")
parser.add_argument('-s', '--spectra', nargs='+', required=True, help="Lista dei file CSV degli spettri")
parser.add_argument('-t', '--table', type=str, required=True, help="File .dat tabulato con Temperature (sinistra) e Resistenze da tabella (destra)")
parser.add_argument('-vr', '--voltaggio_resistenza', type=str, required=True, help="File .dat tabulato sperimentale con Volt (sinistra) e Resistenze reali (destra)")
parser.add_argument('-r', '--resistance', type=float, required=True, help="Resistenza REALE della tua lampadina a temperatura ambiente (300K) in Ohm")
args = parser.parse_args()


# =============================================================================
# 1. CARICAMENTO E RISCALAMENTO TABELLA R-T (LETTERATURA)
# =============================================================================
if not os.path.exists(args.table):
    print(f"Errore: Il file di calibrazione {args.table} non esiste.")
    sys.exit(1)

df_tabella = pd.read_csv(args.table, sep=r"\s+", header=None, comment='#')
tabella_TR = df_tabella.to_numpy()

R_room_reale = args.resistance
print(f"\n[INFO] R_reale_300K impostata da terminale = {R_room_reale} Ohm.")

idx_300K = np.where(tabella_TR[:, 0] == 300.0)[0]
if len(idx_300K) == 0:
    idx_vicino = np.abs(tabella_TR[:, 0] - 300.0).argmin()
    R_tabella_300K = tabella_TR[idx_vicino, 1]
else:
    R_tabella_300K = tabella_TR[idx_300K[0], 1]

tabella_TR[:, 1] = (tabella_TR[:, 1] / R_tabella_300K) * R_room_reale
print(f"[INFO] Tabella R-T riscalata sulle specifiche della tua lampadina.")


# =============================================================================
# 2. CARICAMENTO TABELLA V-R E CICLO FILE
# =============================================================================
if not os.path.exists(args.voltaggio_resistenza):
    print(f"Errore: Il file voltaggio-resistenza {args.voltaggio_resistenza} non esiste.")
    sys.exit(1)

df_vr = pd.read_csv(args.voltaggio_resistenza, sep=r"\s+", header=None, comment='#')
tabella_VR = df_vr.to_numpy()
print(f"[INFO] Tabella V-R sperimentale caricata correttamente ({len(tabella_VR)} punti).\n")

x_lista = []
y_lista = []
file_validi = []
v_lista = []
r_lista = []
T_lista = [] 

for percorso in args.spectra:
    if not os.path.exists(percorso):
        print(f"Salto il file: '{percorso}' (Non esiste)")
        continue
    nome_file = os.path.basename(percorso)
    try:
        v_estratto = estrai_voltaggio_da_nome(nome_file)
        r_reale_calcolata = trova_resistenza_da_voltaggio(tabella_VR, v_estratto)
        df = pd.read_csv(percorso, sep=r"\s+")
        if "Wavelength (nm)" not in df.columns or "Intensity" not in df.columns:
            print(f"File ignorato (colonne errate): {nome_file}")
            continue
        
        x_lista.append(df["Wavelength (nm)"].values)
        y_lista.append(df["Intensity"].values)
        v_lista.append(v_estratto)
        r_lista.append(r_reale_calcolata)
        file_validi.append(nome_file)
        
        t_interpolata = interpolate_temperature(tabella_TR, r_reale_calcolata)
        T_lista.append(t_interpolata)
        print(f"Caricamento file: {nome_file}")
    except Exception as e:
        print(f"Errore durante la lettura di {percorso}: {e}")

if not x_lista:
    print("Errore: Nessun file valido è stato caricato.")
    sys.exit(1)

print(f"\nCaricati con successo {len(x_lista)} file.")
num_file = len(x_lista)
y_average = [np.average(y_col) for y_col in y_lista]
y_average2 = np.power(y_average, 2)


# =============================================================================
# 3. COSTRUZIONE VETTORE DI STIMA INIZIALE DINAMICO
# =============================================================================
p0_T = T_lista
p0_A = [4.76e-7]       

# Inserisci qui le tue stime iniziali liberamente
p0_sigmoide = [
    # a,     lam_cut,  sigma
    1.0,     640.0,    1.0,
    1.0,     660.0,    1.0,
    1.0,     672.0,    1.0,
    1.0,     690.0,    1.0,
    1.0,     700.0,    1.0
] * 2

#p0_sigmoide = (
#[1.94912179e+00] +
#[7.71875830e+02] +
#[3.97825610e+02] +
#
#[1.14407559e+01] +
#[6.56869886e+02] +
#[4.03037013e+00] +
#
#[3.43613173e+00] +
#[6.59149419e+02] +
#[2.16741102e+00] +
#
#[8.32546978e+00] +
#[6.71201576e+02] +
#[2.17583250e+00] +
#
#[2.39684922e+00] +
#[6.48484764e+02] +
#[7.63656868e+00] +
#
#[-1.37677777e+04] +
#[7.66717335e+02] +
#[1.88224033e+01] +
#
#[-6.13206859e-02] +
#[6.79808714e+02] +
#[3.67033950e-04] +
#
#[2.94884095e+00] +
#[6.53274030e+02] +
#[5.45864293e+00] +
#
#[-7.46418087e+08] +
#[7.72152363e+02] +
#[-1.95269905e+02] +
#
#[6.25475724e+00] +
#[6.73737992e+02] +
#[1.31930552e+00] +
#
#[1.40030870e+00] +
#[6.74915724e+02] +
#[4.95358883e-01] +
#
#[-1.35116096e+01] +
#[7.72974421e+02] +
#[8.40687895e+01]
#)

p0_gaussiana = [
    # a,         lam_centro,  sigma
    0.05,        679.00,      5,   -0.05,       680.54,      5,
    0.05,        682.08,      5,   -0.05,       683.62,      5,
    0.05,        685.15,      5,   -0.05,       686.69,      5,
    0.05,        688.23,      5,   -0.05,       689.77,      5,
    0.05,        691.31,      5,   -0.05,       692.85,      5,
    0.05,        694.38,      5,   -0.05,       695.92,      5,
    0.05,        697.46,      5,   -0.05,       699.00,      5
]

N_SIGMOIDI = len(p0_sigmoide) // 3
N_GAUSSIANE = len(p0_gaussiana) // 3
stima_iniziale = p0_T + p0_A + p0_sigmoide + p0_gaussiana

# -----------------------------------------------------------------------------
# GENERAZIONE DINAMICA DEI BOUNDS (CONFINI)
# -----------------------------------------------------------------------------
#bounds = []
#
## 1. Bounds per le Temperature: lasciamole variare intorno al guess (es. 800K - 2500K)
#for T_guess in T_lista:
#    bounds.append((800.0, 2500.0))
#
## 2. Bounds per l'Ampiezza globale A (strettamente positiva)
#bounds.append((1e-9, 1e-4))
#
## 3. Bounds per le Sigmoidi (a, lam_cut, sigma)
#for _ in range(N_SIGMOIDI):
#    bounds.append((-10000.0, 10000.0))    # Ampiezza della sigmoide
#    bounds.append((300.0, 900.0))     # Taglio lam_cut (dentro lo spettro)
#    bounds.append((-5000.0, 5000.0))        # Sigma sigmoide
#
## 4. Bounds per le Gaussiane (a, lam_centro, sigma)
#for _ in range(N_GAUSSIANE):
#    bounds.append((-10, 10))      # Ampiezza picco
#    bounds.append((660.0, 720.0))     # Centro lam_centro (confinato nella zona oscillazioni)
#    bounds.append((1.0, 10.0))        # <--- ECCOLO! Sigma vincolato: minimo 1.2, massimo 10

# -----------------------------------------------------------------------------

print(f"[CONFIG] Avvio fit con {N_SIGMOIDI} sigmoidi e {N_GAUSSIANE} gaussiane.")
print("[CONFIG] Limiti rigidi applicati (Soglia minima Sigma Gaussiane = 1.2).")
print("\nAvvio dell'ottimizzazione globale. Attendere...")

# Passiamo a L-BFGS-B che digerisce i bounds nativi ed è una scheggia
result = minimize(
    loss_function, 
    stima_iniziale, 
    args=(T_lista, x_lista, y_lista, y_average2, N_SIGMOIDI, N_GAUSSIANE), 
    method='Powell',
    #bounds=bounds
)

# =============================================================================
# 4. ESTRAZIONE E STAMPA DEI PARAMETRI FINALI FITTATI
# =============================================================================
print("\n" + "="*50)
print(f"{'RISULTATI DELL OTTIMIZZAZIONE':^50}")
print("="*50)
print(f"Successo del fit: {result.success}")
print(f"Messaggio:        {result.message}")
print("-"*50)

T_fit = result.x[:num_file]
A_fit = result.x[num_file]

# Separazione corretta ed esplicita dei blocchi indici
inizio_filtri = num_file + 1
fine_sigmoidi = inizio_filtri + (N_SIGMOIDI * 3)
fine_gaussiane = fine_sigmoidi + (N_GAUSSIANE * 3)

s_fit = result.x[inizio_filtri : fine_sigmoidi]
g_fit = result.x[fine_sigmoidi : fine_gaussiane]
tutti_i_filtri_fit = result.x[inizio_filtri:] 

print(f"--- PARAMETRI FISICI CONDIVISI ---")
print(f"Ampiezza (A):       {A_fit:.8e}")
print(f"--- TEMPERATURE (Guess iniziale vs Ottimizzate) ---")
for i in range(num_file):
    print(f"File {file_validi[i]}: R = {r_lista[i]:.2f} Ohm | Guess T = {T_lista[i]:.1f} K -> Fit T = {T_fit[i]:.1f} K")

if N_SIGMOIDI > 0:
    print(f"\n--- PARAMETRI SIGMOIDI ({N_SIGMOIDI} componenti) ---")
    print(f"{'Sigmoide':<10} | {'Peso (a)':<12} | {'Taglio (lam)':<12} | {'Larghezza (σ)':<12}")
    print("-"*60)
    for idx, i in enumerate(range(0, len(s_fit), 3)):
        print(f"N. {idx+1:<7} | {s_fit[i]:<12.4e} | {s_fit[i+1]:<12.2f} | {s_fit[i+2]:<12.4f}")

if N_GAUSSIANE > 0:
    print(f"\n--- PARAMETRI GAUSSIANE ({N_GAUSSIANE} componenti) ---")
    print(f"{'Gaussiana':<10} | {'Peso (a)':<12} | {'Centro (lam)':<12} | {'Larghezza (σ)':<12}")
    print("-"*60)
    for idx, i in enumerate(range(0, len(g_fit), 3)):
        print(f"N. {idx+1:<7} | {g_fit[i]:<12.4e} | {g_fit[i+1]:<12.2f} | {g_fit[i+2]:<12.4f}")

print("-"*50)

print("--- COPIA E INCOLLA DA QUI PER I NUOVI GUESS ---")
print("p0_sigmoide = [")
for i in range(0, len(s_fit), 3):
    print(f"    {s_fit[i]:.8e}, {s_fit[i+1]:.4f}, {s_fit[i+2]:.4f},")
print("]")
print("p0_gaussiana = [")
for i in range(0, len(g_fit), 3):
    print(f"    {g_fit[i]:.8e}, {g_fit[i+1]:.4f}, {g_fit[i+2]:.4f},")
print("]")
print("="*50 + "\n")


# =============================================================================
# 5. ESPORTAZIONE FILTRO AD ALTA RISOLUZIONE
# =============================================================================
print("="*50)
print(f"{'ESPORTAZIONE FILTRO AD ALTA RISOLUZIONE':^50}")
print("="*50)

wl_min = min(np.min(x) for x in x_lista)
wl_max = max(np.max(x) for x in x_lista)
wl_min_round = np.floor(wl_min * 10) / 10
wl_max_round = np.ceil(wl_max * 10) / 10

num_punti = int(round((wl_max_round - wl_min_round) / 0.1)) + 1
x_uniforme = np.linspace(wl_min_round, wl_max_round, num_punti)

y_filtro_esportazione = np.zeros_like(x_uniforme, dtype=float)
if N_SIGMOIDI > 0:
    y_filtro_esportazione += somma_sigmoidi(x_uniforme, *s_fit)
if N_GAUSSIANE > 0:
    y_filtro_esportazione += somma_gaussiane(x_uniforme, *g_fit)

df_filtro = pd.DataFrame({"Wavelength (nm)": x_uniforme, "Intensity": y_filtro_esportazione})
nome_file_output = "filtro_stimato_passo_0.1nm.csv"
df_filtro.to_csv(nome_file_output, sep=" ", index=False)

print(f"Intervallo coperto: da {wl_min_round:.1f} nm a {wl_max_round:.1f} nm")
print(f"Numero totale di punti esportati: {len(x_uniforme)}")
print(f"[SUCCESS] File salvato in:\n--> {os.path.abspath(nome_file_output)}\n")


# =============================================================================
# 6. STAMPA COLONNA TEMPERATURE E TENSIONI FITTATE
# =============================================================================
print("--- TEMPERATURE FITTATE vs TENSIONE ---")
for i, t in enumerate(T_fit):
    print(f"{t:.1f} K --> {v_lista[i]:.1f} V")
print("-" * 39 + "\n")


# =============================================================================
# 7. GRAPHIC PLOTS
# =============================================================================
# FIGURA 1: Dati e Fit Globale
plt.figure(figsize=(12, 7))
for i in range(num_file):
    x_data = x_lista[i]
    y_data = y_lista[i]
    params_file = [T_fit[i], A_fit] + list(tutti_i_filtri_fit)
    
    plt.scatter(x_data, y_data, alpha=0.4, label=f'Dati {file_validi[i]}', s=1)
    plt.plot(x_data, teorical_model(x_data, T_fit[i], A_fit, N_SIGMOIDI, N_GAUSSIANE, *tutti_i_filtri_fit), lw=2, label=f'Fit T={T_fit[i]:.1f}K', color='black')

plt.title('Batch Fit simultaneo (Corpo Nero * Filtro Combinato)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# FIGURA 2: Corpo Nero Puro
plt.figure(figsize=(12, 7))
for i in range(num_file):
    x_data = x_lista[i]
    y_black_body = black_body(x_data, A_fit, T_fit[i])
    plt.plot(x_data, y_black_body, lw=2.5, label=f'Corpo Nero Puro T={T_fit[i]:.1f}K')

plt.title('Spettro Teorico del Corpo Nero Puro')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Theoretical Intensity')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# FIGURA 3: Andamento Risposta del Filtro con Dettaglio componenti
plt.figure(figsize=(12, 7))
x_plot_filtro = np.linspace(wl_min_round, wl_max_round, 1000)
y_tot = np.zeros_like(x_plot_filtro)

if N_SIGMOIDI > 0:
    y_sig = somma_sigmoidi(x_plot_filtro, *s_fit)
    y_tot += y_sig
    plt.plot(x_plot_filtro, y_sig, '--', label='Contributo Totale Sigmoidi', alpha=0.7)
if N_GAUSSIANE > 0:
    y_gau = somma_gaussiane(x_plot_filtro, *g_fit)
    y_tot += y_gau
    plt.plot(x_plot_filtro, y_gau, ':', label='Contributo Totale Gaussiane', alpha=0.7)

plt.plot(x_plot_filtro, y_tot, lw=3, color='crimson', label='Filtro Complessivo (Sig + Gauss)')
plt.title('Risposta del Filtro Ottimizzato')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Response')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()