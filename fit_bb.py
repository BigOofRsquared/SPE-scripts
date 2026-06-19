import sys
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import erf
import scipy.constants as const

# --- Modello Fisico con DOPPIA SIGMOIDE (15 Parametri) ---
def teorica_model(lam_nm, A, T, c0, c1, c2, c3, c4, c5, a, lam_cut1, sigma1, d, lam_cut2, sigma2):
    lam = lam_nm * 1e-9 
    h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
    
    # 1. Componente Corpo Nero (Planck)
    exponent = (h * c_vel) / (lam * kB * T)
    exponent = np.clip(exponent, None, 7000) 
    planck = (2 * h * c_vel**2) / (lam**5 * (np.exp(exponent) - 1))
    
    # 2. Polinomio di correzione della risposta
    polinomio = c0 + c1*lam_nm + c2*(lam_nm**2) + c3*(lam_nm**3) + c4*(lam_nm**4) + c5*(lam_nm**5)
    
    # 3. Doppia Sigmoide (I due gradini di salita)
    sigma1 = max(sigma1, 1e-5)
    sigma2 = max(sigma2, 1e-5)
    
    filtro = a * erf((lam_nm - lam_cut1) / sigma1) + d * erf((lam_nm - lam_cut2) / sigma2)
    
    return A * planck * polinomio * filtro

# --- Risposta Strumentale Pura (Polinomio * Doppio Filtro) ---
def calcola_risposta_strumentale(lam_nm, c0, c1, c2, c3, c4, c5, a, lam_cut1, sigma1, d, lam_cut2, sigma2):
    polinomio = c0 + c1*lam_nm + c2*(lam_nm**2) + c3*(lam_nm**3) + c4*(lam_nm**4) + c5*(lam_nm**5)
    sigma1 = max(sigma1, 1e-5)
    sigma2 = max(sigma2, 1e-5)
    filtro = a * erf((lam_nm - lam_cut1) / sigma1) + d * erf((lam_nm - lam_cut2) / sigma2)
    return polinomio * filtro

# --- Funzione Costo Globale ---
def global_cost_function(params, datasets):
    # Spacchettamento rigido dei primi 15 parametri globali
    A, c0, c1, c2, c3, c4, c5, a, lam_cut1, sigma1, d, lam_cut2, sigma2 = params[:14]
    temperatures = params[14:]
    
    total_squared_error = 0.0
    for idx, data in enumerate(datasets):
        x_data = data['x']
        y_data = data['y']
        T_file = temperatures[idx]
        
        y_pred = teorica_model(x_data, A, T_file, c0, c1, c2, c3, c4, c5, a, b, lam_cut1, sigma1, d, lam_cut2, sigma2)
        total_squared_error += np.sum((y_data - y_pred) ** 2)
        
    return total_squared_error

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script:")
        print("  python fit_bb.py *.csv")
        sys.exit(1)

    argomenti = sys.argv[1:]
    files_to_process = []
    for pattern in argomenti:
        matched = glob.glob(pattern)
        for f in matched:
            if f.lower().endswith('.csv'):
                files_to_process.append(f)
    files_to_process = list(dict.fromkeys(files_to_process))

    if not files_to_process:
        print("[ERRORE] Nessun file CSV valido trovato.")
        sys.exit(1)

    datasets = []
    file_names = []
    x_column = 'Wavelength (nm)'
    y_column = 'Intensity'

    print("\nCaricamento dei file:")
    for f_path in files_to_process:
        try:
            df = pd.read_csv(f_path, sep=' ')
            df.columns = df.columns.str.replace('"', '').str.strip()
            if x_column not in df.columns or y_column not in df.columns:
                continue
            df_clean = df[[x_column, y_column]].dropna().sort_values(by=x_column)
            datasets.append({'x': df_clean[x_column].values, 'y': df_clean[y_column].values})
            file_names.append(os.path.basename(f_path))
            print(f"  -> Caricato {os.path.basename(f_path)} ({len(df_clean)} punti)")
        except Exception as e:
            print(f"  [ERRORE] Lettura fallita per {os.path.basename(f_path)}: {e}")

    num_files = len(datasets)
    if num_files == 0:
        print("[ERRORE] Nessun dato valido caricato."); sys.exit(1)

    # --- Configurazione dei 2 Gradini ---
    guess_cut1 = 600.0  # Primo gradino (modificabile se sai dove si trova)
    guess_cut2 = 950.0  # Secondo gradino

    # Vettore p0 di partenza: ORA SONO ESATTAMENTE 15 PARAMETRI
    # [1,  2,   3,   4,   5,   6,   7,   8,   9,   10,         11,     12,  13,         14,     15]
    optical_p0 = [
        1e-12,                        # A
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, # c0, c1, c2, c3, c4, c5
        0.3, guess_cut1, 5.0,         # b, lam_cut1, sigma1  (Salita 1)
        0.3, guess_cut2, 5.0          # d, lam_cut2, sigma2  (Salita 2)
    ]
    
    t_guesses = [3000.0] * num_files
    initial_params = optical_p0 + t_guesses

    # Bounds corrispondenti (15 tuple per l'ottica)
    optical_bounds = [
        (1e-25, 1e-1), 
        (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), 
        (-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf), # c0..c5
        (-5.0, 5.0),                                            # a
        (-5.0, 5.0), (200.0, 900.0), (0.1, 100.0) ,             # b, cut1, sigma1
        (-5.0, 5.0), (200.0, 900.0), (0.1, 100.0)               # d, cut2, sigma2
    ]
    t_bounds = [(1000.0, 6000.0)] * num_files
    total_bounds = optical_bounds + t_bounds

    print(f"\nAvvio ottimizzazione globale (Modello a 2 Salite)...")
    
    res = minimize(
        global_cost_function, 
        initial_params, 
        args=(datasets,), 
        bounds=total_bounds, 
        method='L-BFGS-B', 
        options={'maxiter': 3000, 'disp': True}
    )

    # Estrazione dei parametri corretti (taglio a 15)
    opt_optical = res.x[:15]
    opt_temperatures = res.x[15:]

    # Scompattamento per l'esportazione
    _, c0, c1, c2, c3, c4, c5, a, b, lam_cut1, sigma1, d, lam_cut2, sigma2 = opt_optical

    print("\n" + "="*60)
    print(" PARAMETRI DI RISPOSTA OTTICA FINALI")
    print("="*60)
    nomi_ottica = ['A', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'a', 'b', 'cut1 (nm)', 'sigma1', 'd', 'cut2 (nm)', 'sigma2']
    for name, val in zip(nomi_ottica, opt_optical):
        print(f"  {name:<18}: {val:12.5e}")
    print("-"*60)
    print(" TEMPERATURE OTTENUTE:")
    for name, t_val in zip(file_names, opt_temperatures):
        print(f"  {name:<30}: {t_val:.2f} K")
    print("="*60)

    # --- ESPORTAZIONE AD ALTA RISOLUZIONE ---
    x_export = np.arange(350.0, 900.5, 0.5)
    risposta_ottica = calcola_risposta_strumentale(x_export, c0, c1, c2, c3, c4, c5, a, b, lam_cut1, sigma1, d, lam_cut2, sigma2)
    
    # Evitiamo asintoti strani nell'inversa
    risposta_ottica_safe = np.where(risposta_ottica <= 1e-6, 1e-6, risposta_ottica)
    inversa_risposta = 1.0 / risposta_ottica_safe

    pd.DataFrame({'Wavelength (nm)': x_export, 'Instrument_Response': risposta_ottica}).to_csv('risposta_filtro.csv', sep=' ', index=False)
    pd.DataFrame({'Wavelength (nm)': x_export, 'Correction_Factor': inversa_risposta}).to_csv('correzione_strumentale.csv', sep=' ', index=False)
    print("\n[OK] Esportati: 'risposta_filtro.csv' e 'correzione_strumentale.csv'")

    # --- PLOT FINALE ---
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    cmap = plt.get_cmap('tab10')
    for idx, data in enumerate(datasets):
        color = cmap(idx % 10)
        plt.plot(data['x'], data['y'], '.', color=color, alpha=0.2)
        x_grid = np.linspace(data['x'].min(), data['x'].max(), 300)
        plt.plot(x_grid, teorica_model(x_grid, *opt_optical, opt_temperatures[idx]), '-', color=color, label=f"{file_names[idx]} ({opt_temperatures[idx]:.0f}K)")
    plt.title("Dati Sperimentali vs Fit Globale")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    p1, = ax1.plot(x_export, risposta_ottica, 'g-', label='Risposta Strumento (2 Salite)', linewidth=2)
    p2, = ax2.plot(x_export, inversa_risposta, 'r--', label='Fattore Correzione (1/Filtro)', linewidth=2)
    ax1.set_title("Curve di Taratura Esportate")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Risposta Strumento", color='g')
    ax2.set_ylabel("Fattore di Correzione (1/R)", color='r')
    ax1.grid(True, alpha=0.3)
    ax1.legend(handles=[p1, p2], loc='best')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()