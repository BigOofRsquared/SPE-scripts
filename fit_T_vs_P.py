#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# 1. MODELLO FISICO REALE (Radiazione T^4 fissa + Conducibilità Variabile)
# =============================================================================
def modello_lampadina_reale(T, A, B1, B2, T0=300.0):
    """
    P = A*(T^4 - T0^4) + B1*(T - T0) - B2*(T - T0)^2
    - Radiazione netta verso la stanza bloccata a T^4.
    - Conduzione/Convezione con correzione al secondo ordine (k decrescente con T).
    """
    P_rad = A * (T**4 - T0**4)
    P_cond = B1 * (T - T0) - B2 * (T - T0)**2
    return P_rad + P_cond

# =============================================================================
# 2. FUNZIONE COSTO CON PENALIZZAZIONE RIGIDA (BARRIERA PER POWELL)
# =============================================================================
def loss_function_powell(params, T_dati, P_dati, T0):
    A, B1, B2 = params
    
    # Vincoli fisici: A >= 0, B1 >= 0, B2 >= 0
    # Se l'algoritmo esce dai bordi, restituiamo un valore enorme (infinito)
    if A < 0.0 or B1 < 0.0 or B2 < 0.0:
        return np.inf
        
    P_pred = modello_lampadina_reale(T_dati, A, B1, B2, T0=T0)
    
    # Pesiamo sui residui relativi percentuali per non schiacciare la bassa temperatura
    residui = (P_pred - P_dati) / np.maximum(P_dati, 1e-3)
    return np.sum(residui**2)

# =============================================================================
# 3. GESTIONE ARGOMENTI
# =============================================================================
def gestisci_argomenti():
    parser = argparse.ArgumentParser(
        description="Fit Lampadina tramite metodo Powell con iterazioni massime."
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True, 
        help="File dei dati (Colonna 1: T [K], Colonna 2: P [W])"
    )
    parser.add_argument(
        '--t0', type=float, default=300.0, 
        help="Temperatura ambiente T0 in Kelvin (default: 300 K)"
    )
    return parser.parse_args()

# =============================================================================
# 4. MAIN SCRIPT
# =============================================================================
def main():
    args = gestisci_argomenti()
    
    if not os.path.exists(args.input):
        print(f"Errore: Il file '{args.input}' non esiste.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(args.input, sep=r"\s+", header=None, comment='#')
        data = df.to_numpy()
        T_dati = data[:, 0]  
        P_dati = data[:, 1]  
    except Exception as e:
        print(f"Errore lettura file: {e}")
        sys.exit(1)
        
    print(f"[INFO] Caricati {len(T_dati)} punti sperimentali.")

    # =============================================================================
    # 5. OTTIMIZZAZIONE CON METODO POWELL E ITERAZIONI MASSIME
    # =============================================================================
    # Guess iniziali stabili
    p0 = [1e-12, 1e-3, 1e-7]
    
    print(f"[INFO] Avvio ottimizzazione tramite metodo POWELL (maxiter={sys.maxsize})...")
    
    res = minimize(
        loss_function_powell, 
        p0, 
        args=(T_dati, P_dati, args.t0), 
        method='Powell', 
        options={
            'maxiter': sys.maxsize,  # Massimo numero di iterazioni possibili sul sistema
            'maxfev': sys.maxsize,    # Massimo numero di valutazioni della funzione costo
            'xtol': 1e-15,            # Tolleranza di convergenza sui parametri
            'ftol': 1e-15             # Tolleranza di convergenza sulla loss
        }
    )
    
    if not res.success:
        print(f"[WARNING] L'ottimizzatore ha terminato con messaggio: {res.message}")
    else:
        print(f"[INFO] Ottimizzazione completata con successo in {res.nit} iterazioni.")
    
    A_fit, B1_fit, B2_fit = res.x

    # Calcolo incertezze dall'Hessiana numerica (Powell non la calcola nativamente)
    # Usiamo una matrice di zeri se non disponibile per non rompere il plot
    perr = np.zeros(3)
    if hasattr(res, 'hess_inv') and res.hess_inv is not None:
        try:
            cov = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else res.hess_inv.todense()
            varianza_residui = res.fun / (len(T_dati) - 3)
            perr = np.sqrt(np.diag(cov) * varianza_residui)
        except Exception:
            perr = [np.nan, np.nan, np.nan]

    print(f"\nRisultati del Fit (Powell, T^4 fisso):")
    print(f"  A (Coeff. Radiativo): {A_fit:.6e}")
    print(f"  B1 (Cond. Lineare):   {B1_fit:.6e}")
    print(f"  B2 (Rettifica k):     {B2_fit:.6e}")

    # =============================================================================
    # 6. PLOTTING
    # =============================================================================
    plt.figure(figsize=(8, 6))
    
    plt.scatter(T_dati, P_dati, color='blue', alpha=0.7, s=50, label='Dati Sperimentali', zorder=3)
    
    T_teorico = np.linspace(min(T_dati) - 20, max(T_dati) + 20, 500)
    P_teorico = modello_lampadina_reale(T_teorico, A_fit, B1_fit, B2_fit, T0=args.t0)
    
    plt.plot(T_teorico, P_teorico, color='grey', linestyle='-.', lw=1.5, label='Fit Metodo Powell', zorder=2)
    
    # Componenti separate
    P_rad = A_fit * (T_teorico**4 - args.t0**4)
    P_cond = B1_fit * (T_teorico - args.t0) - B2_fit * (T_teorico - args.t0)**2
    
    plt.plot(T_teorico, P_rad, '--', color='red', alpha=0.5, label='Radiazione Netta ($T^4$)')
    plt.plot(T_teorico, P_cond, '--', color='green', alpha=0.5, label='Conduzione/Convezione $k(T)$')

    testo_box = (
        rf"$A$ = {A_fit:.2e}" + "\n" +
        rf"$B_1$ = {B1_fit:.2e}" + "\n" +
        rf"$B_2$ = {B2_fit:.2e}"
    )
    plt.text(
        min(T_dati) + (max(T_dati) - min(T_dati)) * 0.05,
        max(P_dati) * 0.75,
        testo_box,
        fontsize=14,
        color="darkblue",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        zorder=5
    )

    plt.xlabel("Filament Temperature $T$ (K)", fontsize=20)
    plt.ylabel("Dissipated Power $P$ (W)", fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    main()