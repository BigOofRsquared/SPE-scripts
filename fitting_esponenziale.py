#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# 1. DEFINIZIONE DEL MODELLO FISICO (CORRETTO CON IL SEGNO MENO PER IL DECADIMENTO)
def modello_esponenziale(x, a, b, c):
    # b rappresenta proprio tau. Il segno meno garantisce che la curva decada nel tempo.
    return a * np.exp(-x / b) + c


def main():
    # 2. Controllo e lettura del file passato come parametro
    if len(sys.argv) < 2:
        print("Errore: Devi passare il nome del file dei dati Y.")
        print("Uso: python fit_esponenziale.py <nome_file.txt>")
        sys.exit(1)

    nome_file = sys.argv[1]

    try:
        y_completo = np.loadtxt(nome_file)
    except Exception as e:
        print(f"Errore nella lettura del file {nome_file}: {e}")
        sys.exit(1)

    # 3. Generazione dell'asse X e applicazione maschera di salto indici
    n_punti = len(y_completo)
    x_completo = np.arange(n_punti) * 2.0

    # Creiamo la maschera booleana coerente con i valori moltiplicati per 2
    maschera = (x_completo != 72) & (x_completo != 0)

    # Applichiamo la maschera ai dati per il fit
    x = x_completo[maschera]
    y = y_completo[maschera]
    
    # Isoliamo i punti scartati per il plotting dedicato
    x_scartati = x_completo[~maschera]
    y_scartati = y_completo[~maschera]

    # 4. Fit Esponenziale con scipy.optimize.curve_fit
    # a (I0) ~ 7000, b (tau) ~ 60 secondi (positivo!), c (offset) ~ 200
    stima_iniziale = [7000.0, 60.0, 200.0]

    try:
        popt, pcov = curve_fit(modello_esponenziale, x, y, p0=stima_iniziale)
        a, b, c = popt
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Errore durante l'ottimizzazione del fit: {e}")
        print("Controlla che i dati siano effettivamente un decadimento.")
        sys.exit(1)

    print(f"Risultati del Fit non lineare (y = I0 * exp(-x / tau) + c):")
    print(f"  Parametro a (I0):  {a:.6e} ± {perr[0]:.2e}")
    print(f"  Parametro b (tau): {b:.6e} ± {perr[1]:.2e}")
    print(f"  Parametro c (c):   {c:.6e} ± {perr[2]:.2e}")

    # 5. Generazione range denso per la linea continua del fit
    x_fit = np.linspace(min(x_completo) - 5, max(x_completo) + 5, 500)
    y_fit = modello_esponenziale(x_fit, a, b, c)

    # 6. Plotting con l'estetica richiesta
    plt.figure(figsize=(8, 6))

    # Dati sperimentali inclusi
    plt.scatter(
        x,
        y,
        s=30,
        color="blue",
        alpha=0.7,
        label="Dati inclusi",
        zorder=3,
    )

    # Punti esclusi disegnati in grigio leggero
    if len(x_scartati) > 0:
        plt.scatter(
            x_scartati,
            y_scartati,
            color="gray",
            marker="X",
            s=40,
            alpha=0.6,
            label="Dati esclusi",
            zorder=4,
        )

    # Line plot del fit (Grigio, tratto punto-linea -.)
    plt.plot(
        x_fit,
        y_fit,
        color="red",
        linestyle="-.",
        linewidth=2,
        label="Fit Esponenziale",
        zorder=7
    )

    # Testo del box (Corretto il segno meno anche nella stringa LaTeX)
    testo_box = rf"$I_0$ = {a:.2f} ± {perr[0]:.2f}" + "\n" + rf"$\tau$ = {b:.4f} ± {perr[1]:.4f}" + "\n" + f"c = {c:.2f} ± {perr[2]:.2f}"

    # Coordinate posizionate al 55% di X e 75% di Y per non sparire fuori scala
    plt.text(
        max(x_completo) * 0.55,
        max(y_completo) * 0.75,
        testo_box,
        fontsize=15,
        color="darkblue",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        zorder=5
    )

    # Personalizzazione assi
    plt.xlabel("Exposure time (s)", fontsize=20)
    plt.ylabel(r"Counts (avg)", fontsize=20)
    plt.tick_params(axis='both', labelsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()