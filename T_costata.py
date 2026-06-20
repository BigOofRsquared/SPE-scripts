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

#def somma_gaussiane(x, *parametri_sigmoidi):
#    """
#    Calcola la somma di N sigmoidi (basate sulla funzione di errore 'erf').
#    """
#    filtro = np.zeros_like(x, dtype=float)
#    for i in range(0, len(parametri_sigmoidi), 3):
#        a = parametri_sigmoidi[i]
#        lam_centro = parametri_sigmoidi[i+1]
#        sigma = parametri_sigmoidi[i+2]
#        s = np.where(sigma <= 0, 2e-1, sigma)
#        filtro += a * np.exp((x-lam_centro)**2 / )
#    return filtro

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

def teorical_model(x, T, A, *s):
    """
    Modello teorico: Legge di Planck (Corpo Nero) * Filtro a Sigmoidi
    """    
    filtro = somma_sigmoidi(x, *s)
    return black_body(x, A, T) * filtro

def loss_function_1_file(file_params, x, y):
    """
    Somma dei residui quadratici per un singolo file.
    """
    y_pred = teorical_model(x, *file_params)
    return np.sum((y - y_pred) ** 2)

def loss_function(params, T_predicted, batch_x, batch_y, y_avg2):
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
        cost += loss_function_1_file(curr_params, x_data, y_data) / y_avg2[i]
        cost += 50.0 * ((T_predicted[i] - T_array[i])/T_predicted[i])**2

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

y_average = []
for i, y_col in enumerate(y_lista):
    y_average.append(np.average(y_col))

y_average2 = np.power(y_average, 2)

# =============================================================================
# 3. COSTRUZIONE VETTORE DI STIMA INIZIALE E OTTIMIZZAZIONE
# =============================================================================
p0_T = T_lista
p0_A = [3e-7]       # Guess ampiezza comune
p0_sigmoide = (
#[0.2]+[630.0]+[5.0]+
#[0.2]+[635.0]+[5.0]+
#[0.2]+[645.0]+[5.0]+
#[0.2]+[650.0]+[5.0]
#) + (
[1.0]+[640.0]+[1.0]+
[1.0]+[660.0]+[1.0]+
[1.0]+[672.0]+[1.0]+
[1.0]+[690.0]+[1.0]
) * 3 # + (
#[ 0.2]+[679.0]+[2.0]+
#[-0.2]+[681.0]+[2.0]+
#[ 0.2]+[683.0]+[2.0]+
#[-0.2]+[685.0]+[2.0]+
#[ 0.2]+[688.0]+[2.0]+
#[-0.2]+[690.0]+[2.0]+
#[ 0.2]+[791.0]+[1.0]+
#[-0.2]+[793.0]+[1.0]+
#[ 0.2]+[695.0]+[1.0]+
#[-0.2]+[700.0]+[1.0]+
#[ 0.2]+[705.0]+[1.0]+
#[-0.2]+[710.0]+[1.0]
#)


# Mega-vettore: [T1, T2, ..., Tn, A, s1, s2, ...]
stima_iniziale = p0_T + p0_A + p0_sigmoide

print("\nAvvio dell'ottimizzazione globale. Attendere...")
result = minimize(
    loss_function, 
    stima_iniziale, 
    args=(T_lista, x_lista, y_lista, y_average2), 
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
    print(f"File {file_validi[i]}: R = {r_lista[i]:.2f} Ohm | Guess T = {T_lista[i]:.1f} K -> Fit T = {T_fit[i]:.1f} K")

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

print("\n" + "="*50)
print(f"{'ESPORTAZIONE FILTRO AD ALTA RISOLUZIONE':^50}")
print("="*50)

# Troviamo gli estremi assoluti di lunghezza d'onda combinando tutti i file caricati
wl_min = min(np.min(x) for x in x_lista)
wl_max = max(np.max(x) for x in x_lista)

# Arrotondiamo al decimo di nm per avere estremi puliti
wl_min_round = np.floor(wl_min * 10) / 10
wl_max_round = np.ceil(wl_max * 10) / 10

# Generiamo la nuova griglia lineare con passo 0.1 nm
# (max - min) / 0.1 + 1 calcola esattamente quanti punti servono
num_punti = int(round((wl_max_round - wl_min_round) / 0.1)) + 1
x_uniforme = np.linspace(wl_min_round, wl_max_round, num_punti)

# Calcoliamo la risposta del filtro fittato sulla nuova griglia
y_filtro_esportazione = somma_sigmoidi(x_uniforme, *s_fit)

# Creiamo il DataFrame coerente con le colonne standard
df_filtro = pd.DataFrame({
    "Wavelength (nm)": x_uniforme,
    "Intensity": y_filtro_esportazione
})

nome_file_output = "filtro_stimato_passo_0.1nm.csv"

# Salvataggio in formato CSV (valori separati da spazio)
df_filtro.to_csv(nome_file_output, sep=" ", index=False)

print(f"Intervallo coperto: da {wl_min_round:.1f} nm a {wl_max_round:.1f} nm")
print(f"Passo di campionamento: 0.1 nm")
print(f"Numero totale di punti esportati: {len(x_uniforme)}")
print(f"[SUCCESS] File salvato in:")
print(f"--> {os.path.abspath(nome_file_output)}")
print("="*50 + "\n")

# =============================================================================
# 6. STAMPA COLONNA TEMPERATURE E TENSIONI FITTATE
# =============================================================================
print("--- TEMPERATURE FITTATE vs TENSIONE ---")
for i, t in enumerate(T_fit):
    print(f"{t:.1f} K --> {v_lista[i]:.1f} V")
print("-" * 39 + "\n")

# =============================================================================
# 5. GRAPHIC PLOT (LOG-LOG SCALE)
# =============================================================================
plt.figure(figsize=(12, 7))

for i in range(num_file):
    x_data = x_lista[i]
    y_data = y_lista[i]
    
    # Costruiamo il vettore parametri specifico per l'i-esimo file usando la T fittata
    params_file = [T_fit[i], A_fit] + list(s_fit)
    
    plt.scatter(x_data, y_data, alpha=0.4, label=f'Dati {file_validi[i]}', s=1)
    plt.plot(x_data, teorical_model(x_data, *params_file), lw=2, label=f'Fit T={T_fit[i]:.1f}K', color='black')

# Abilitazione della scala log-log
#plt.xscale('log')
#plt.yscale('log')

plt.title('Batch Fit simultaneo (Scala Log-Log)')
plt.xlabel('Wavelength (nm) [Log Scale]')
plt.ylabel('Intensity [Log Scale]')
#plt.legend(loc='upper right', fontsize='small', ncol=2)
plt.grid(True, which="both", linestyle='--', alpha=0.5) # "both" mostra la griglia anche per i minor ticks del logaritmo
plt.tight_layout()

plt.figure(figsize=(12, 7)) # Apre una NUOVA finestra/figura separata

for i in range(num_file):
    x_data = x_lista[i]
    
    # Calcolo del corpo nero teorico puro per la temperatura di questo file
    # Uso T_fit[i] o T_fit[i]**0.25 a seconda di come hai scalato la T nel fit
    y_black_body = black_body(x_data, A_fit, T_fit[i])
    area = np.trapz(y_black_body)
    #y_black_body /= area
    
    plt.plot(x_data, y_black_body, lw=2.5, label=f'Corpo Nero Puro T={T_fit[i]:.1f}K')

#plt.xscale('log')
#plt.yscale('log')
plt.title('Spettro Teorico del Corpo Nero Puro (Senza Risposta Strumentale)')
plt.xlabel('Wavelength (nm) [Log Scale]')
plt.ylabel('Theoretical Intensity [Log Scale]')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()

plt.figure(figsize=(12, 7)) # Apre una NUOVA finestra/figura separata

for i in range(num_file):
    x_data = x_lista[i]
    
    # Calcolo del corpo nero teorico puro per la temperatura di questo file
    # Uso T_fit[i] o T_fit[i]**0.25 a seconda di come hai scalato la T nel fit
    y_filter = somma_sigmoidi(x_data, *s_fit)
    
    plt.plot(x_data, y_filter, lw=2.5, label=f'Filtro')

#plt.xscale('log')
#plt.yscale('log')
plt.title('Filtro stimato')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Response')
plt.legend(loc='upper right', fontsize='small')
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.tight_layout()

plt.show()

