#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================================================================
# 1. DEFINIZIONE DEL MODELLO FISICO
# =============================================================================
def modello_potenza(T, A, B, T0=300.0):
    """
    Modello di potenza dissipata: P = A*(T^4) - B*(T - T0)
    T0 è fissato di default a temperatura ambiente (300 K), 
    ma può essere lasciato come parametro libero se necessario.
    """
    return A * (T**4) - B * (T - T0)

# =============================================================================
# 2. GESTIONE ARGOMENTI DA TERMINALE
# =============================================================================
def gestisci_argomenti():
    parser = argparse.ArgumentParser(
        description="Fit della potenza in funzione della temperatura: P = A*(T^4) - B*(T - T0)"
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True, 
        help="File di testo/dat contenente le Temperature (colonna 1) e Potenze (colonna 2)"
    )
    parser.add_argument(
        '--t0', type=float, default=300.0, 
        help="Temperatura ambiente T0 in Kelvin (default: 300 K)"
    )
    return parser.parse_args()

# =============================================================================
# 3. SCRIPT PRINCIPALE
# =============================================================================
def main():
    args = gestisci_argomenti()
    
    if not os.path.exists(args.input):
        print(f"Errore: Il file di input '{args.input}' non esiste.")
        sys.exit(1)
        
    print(f"[INFO] Caricamento dati da: {args.input}")
    
    # Caricamento del file (gestisce spazi multipli o tabulazioni, ignora i commenti '#')
    try:
        df = pd.read_csv(args.input, sep=r"\s+", header=None, comment='#')
        data = df.to_numpy()
        
        T_dati = data[:, 0]  # Prima colonna: Temperatura (K)
        P_dati = data[:, 1]  # Seconda colonna: Potenza (W)
    except Exception as e:
        print(f"Errore durante la lettura del file: {e}")
        sys.exit(1)
        
    print(f"[INFO] Caricati con successo {len(T_dati)} punti sperimentali.")

    # =============================================================================
    # 4. FITTING DEI DATI
    # =============================================================================
    # Stime iniziali (p0) per i parametri [A, B]
    # A è legato a Stefan-Boltzmann (ordine tipico molto piccolo, es. 1e-12 o 1e-10)
    # B è legato alle perdite lineari (ordine tipico 1e-3 o 1e-2)
    p0 = [1e-11, 1e-3]
    
    # Definiamo i limiti fisici: A e B devono essere rigorosamente positivi
    bounds = ((0, 0), (np.inf, np.inf))
    
    print("[INFO] Avvio del fit non-lineare dei minimi quadrati...")
    try:
        # Fissiamo T0 nel fit usando una lambda function lambda T, A, B: modello(T, A, B, T0)
        popt, pcov = curve_fit(
            lambda T, A, B: modello_potenza(T, A, B, T0=args.t0), 
            T_dati, 
            P_dati, 
            p0=p0,
            bounds=bounds
        )
        A_fit, B_fit = popt
        perr = np.sqrt(np.diag(pcov)) # Errori standard sui parametri fittati
    except Exception as e:
        print(f"Errore durante l'ottimizzazione del fit: {e}")
        sys.exit(1)

    # Stampa dei risultati sul terminale
    print("\n" + "="*50)
    print(f"{'RISULTATI DEL FIT DELLA POTENZA':^50}")
    print("="*50)
    print(f"Modello fittato: P = A*(T^4) - B*(T - {args.t0:.1f})")
    print("-"*50)
    print(f"Parametro A (Irradiamento): {A_fit:.6e} ± {perr[0]:.6e}")
    print(f"Parametro B (Conduzione):   {B_fit:.6e} ± {perr[1]:.6e}")
    print("="*50 + "\n")

    # =============================================================================
    # 5. GRAFICO DEI RISULTATI
    # =============================================================================
    plt.figure(figsize=(10, 6))
    
    # Plotta i punti sperimentali
    plt.scatter(T_dati, P_dati, color='red', alpha=0.7, edgecolors='k', label='Dati Sperimentali', zorder=3)
    
    # Genera una curva fitta e continua per il modello teorico
    T_teorico = np.linspace(np.min(T_dati)*0.9, np.max(T_dati)*1.1, 500)
    P_teorico = modello_potenza(T_teorico, A_fit, B_fit, T0=args.t0)
    
    plt.plot(T_teorico, P_teorico, color='black', lw=2.5, label='Modello Fittato')
    
    # Separazione grafica dei singoli contributi (opzionale, utile per l'analisi fisica)
    P_radiazione = A_fit * (T_teorico**4)
    P_conduzione = - B_fit * (T_teorico - args.t0)
    plt.plot(T_teorico, P_radiazione, '--', color='blue', alpha=0.5, label='Componente Radiativa ($A \cdot T^4$)')
    plt.plot(T_teorico, P_conduzione, '--', color='green', alpha=0.5, label='Componente Lineare ($-B \cdot (T-T_0)$)')

    # Labelling e abbellimenti assi
    plt.title('Fit della Potenza Dissipata vs Temperatura del Filamento')
    plt.xlabel('Temperatura $T$ (K)')
    plt.ylabel('Potenza $P$ (W)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left', fontsize='medium')
    plt.tight_layout()
    
    # Salvataggio del grafico come immagine
    #nome_grafico = "fit_potenza_temperatura.png"
    #plt.savefig(nome_grafico, dpi=300)
    #print(f"[INFO] Grafico salvato in: {os.path.abspath(nome_grafico)}")
    
    plt.show()

if __name__ == '__main__':
    main()