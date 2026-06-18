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

def somma_sigmoidi(x, *parametri_sigmoidi):
    """
    Calcola la somma di N sigmoidi (basate sulla funzione di errore 'erf').
    """
    filtro = np.zeros_like(x, dtype=float)
    for i in range(0, len(parametri_sigmoidi), 3):
        a = parametri_sigmoidi[i]
        lam_cut = parametri_sigmoidi[i+1]
        sigma = parametri_sigmoidi[i+2]
        s = np.where(sigma <= 0, 8e-1, sigma)
        filtro += a * (1.0 + erf((x - lam_cut) / s))
    return filtro

def teorical_model(x, T4, A, *s):
    """
    Modello teorico: Legge di Planck (Corpo Nero) * Filtro a Sigmoidi
    """
    lam = x * 1e-9 
    h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
    
    exponent = (h * c_vel) / (lam * kB * T4**0.25)
    exponent = np.clip(exponent, None, 7000) 
    planck = (2 * h * c_vel**2) / (lam**5 * (np.exp(exponent) - 1))
    
    filtro = somma_sigmoidi(x, *s)
    return A * planck * filtro

def loss_function_1_file(file_params, x, y):
    """
    Somma dei residui quadratici per un singolo file.
    """
    y_pred = teorical_model(x, *file_params)
    return np.sum((y - y_pred) ** 2)

def loss_function(params, batch_x, batch_y):
    """
    Funzione di costo globale per il fit simultaneo (T libere ottimizzate).
    """
    cost = 0.0
    num_file = len(batch_x)
    
    # Spacchettamento coerente con il vettore stima_iniziale
    T_array = params[:num_file]           # Le prime 'num_file' posizioni sono le temperature
    A = params[num_file]                  # Poi c'è l'Ampiezza A comune
    sigmoidi = params[num_file+1:]        # Tutto il resto sono le sigmoidi comuni
    
    for i in range(num_file):
        # Ricostruiamo i parametri per il file i-esimo: [T_i, A, s1, s2...]
        curr_params = [T_array[i], A] + list(sigmoidi)
        
        x_data = batch_x[i]
        y_data = batch_y[i]
        cost += loss_function_1_file(curr_params, x_data, y_data)
    return cost

def interpolate_temperature(tabella_calibrazione, r_val):
    """
    Prende la tabella numpy 2xN (temperatura a sinistra, resistenza a destra).
    Filtra e ordina i dati per garantire la perfetta monotonicità richiesta da SciPy.
    """
    # Teniamo solo la zona da 300K in su per evitare inversioni criogeniche
    maschera = tabella_calibrazione[:, 0] >= 300.0
    tabella_filtrata = tabella_calibrazione[maschera]
    
    temp_col = tabella_filtrata[:, 0]
    res_col = tabella_filtrata[:, 1]
    
    # Ordiniamo i vettori in base alla resistenza crescente (asse X)
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
    print(f"[ATTENZIONE] Valore 300.0 K non trovato esattamente. Uso il valore a T={tabella_TR[idx_vicino, 0]}K: {R_tabella_300K}")
else:
    R_tabella_300K = tabella_TR[idx_300K[0], 1]

# Convertiamo la tabella di letteratura in Ohm REALI della tua lampadina
tabella_TR[:, 1] = (tabella_TR[:, 1] / R_tabella_300K) * R_room_reale
print(f"[INFO] Tabella R-T riscalata sulle specifiche della tua lampadina (valore a 300K imposto a {R_room_reale} Ohm).")


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
T_lista = [] # Diventerà p0_T (il nostro guess iniziale)

for percorso in args.spectra:
    if not os.path.exists(percorso):
        print(f"Salto il file: '{percorso}' (Non esiste)")
        continue
        
    print(f"Caricamento file: {percorso}")
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
        
        # Interpolazione basata su Ohm reali per estrarre la T iniziale di guess
        t_interpolata = interpolate_temperature(tabella_TR, r_reale_calcolata)
        T_lista.append(t_interpolata)
        
    except Exception as e:
        print(f"Errore durante la lettura di {percorso}: {e}")

if not x_lista:
    print("Errore: Nessun file valido è stato caricato.")
    sys.exit(1)

print(f"\nCaricati con successo {len(x_lista)} file.")
num_file = len(x_lista)


# =============================================================================
# 3. COSTRUZIONE VETTORE DI STIMA INIZIALE E OTTIMIZZAZIONE
# =============================================================================
p0_T = np.power(T_lista, 4).tolist()
p0_A = [1e-9]       # Guess ampiezza comune
p0_sigmoide = (
[1.0]+[640.0]+[1.0]+
[1.0]+[660.0]+[1.0]+
[1.0]+[670.0]+[1.0]+
[1.0]+[690.0]+[1.0]
) * 3

# Mega-vettore: [T1, T2, ..., Tn, A, s1, s2, ...]
stima_iniziale = p0_T + p0_A + p0_sigmoide

print("\nAvvio dell'ottimizzazione globale. Attendere...")
result = minimize(
    loss_function, 
    stima_iniziale, 
    args=(x_lista, y_lista), 
    method='Powell'
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

# Estrazione corretta usando gli indici del mega-vettore fittato
T_fit = result.x[:num_file]
A_fit = result.x[num_file]
s_fit = result.x[num_file+1:]

print(f"--- PARAMETRI FISICI CONDIVISI ---")
print(f"Ampiezza (A):       {A_fit:.8e}")
print(f"--- TEMPERATURE (Guess iniziale vs Ottimizzate) ---")
for i in range(num_file):
    print(f"File {file_validi[i]}: R = {r_lista[i]:.2f} Ohm | Guess T = {T_lista[i]:.1f} K -> Fit T = {T_fit[i]**0.25:.1f} K")

print(f"\n--- PARAMETRI SIGMOIDI ({len(s_fit) // 3} componenti) ---")
print(f"{'Sigmoide':<10} | {'Peso (a)':<12} | {'Taglio (lam)':<12} | {'Larghezza (σ)':<12}")
print("-"*50)

for idx, i in enumerate(range(0, len(s_fit), 3)):
    print(f"N. {idx+1:<7} | {s_fit[i]:<12.4e} | {s_fit[i+1]:<12.2f} | {s_fit[i+2]:<12.4f}")

print("-"*50)

stringhe_sigmoidi = []
for i in range(0, len(s_fit), 3):
    blocco = f"[{s_fit[i]:.8e}] +\n[{s_fit[i+1]:.8e}] +\n[{s_fit[i+2]:.8e}]"
    stringhe_sigmoidi.append(blocco)

output_copiaincolla = " +\n".join(stringhe_sigmoidi)

print("--- COPIA E INCOLLA DA QUI PER p0_sigmoide ---")
print("p0_sigmoide = (")
print(output_copiaincolla)
print(")")
print("="*50 + "\n")


# =============================================================================
# 5. GRAPHIC PLOT
# =============================================================================
plt.figure(figsize=(12, 7))

for i in range(num_file):
    x_data = x_lista[i]
    y_data = y_lista[i]
    
    # Costruiamo il vettore parametri specifico per l'i-esimo file usando la T fittata
    params_file = [T_fit[i], A_fit] + list(s_fit)
    
    plt.scatter(x_data, y_data, alpha=0.4, label=f'Dati {file_validi[i]}')
    plt.plot(x_data, teorical_model(x_data, *params_file), lw=2, label=f'Fit T={T_fit[i]**0.25:.1f}K', color='black')

plt.title('Batch Fit simultaneo (Temperature Libere da Guess R-T)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()