import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize  # Usiamo direttamente minimize per avere Powell
from scipy.special import erf


# 1. Modello con due Error Function con CENTRO IN COMUNE (x0)
def fit_erf_comune_func(x, A1, sigma1, A2, sigma2, x0, y0):
    erf1 = A1 * erf((x - x0) / sigma1)
    erf2 = A2 * erf((x - x0) / sigma2)
    return erf1 + erf2 + y0


# 2. Funzione di costo (Somma dei Quadrati dei Residui) da minimizzare
def funzione_costo(parametri, x, y):
    A1, sigma1, A2, sigma2, x0, y0 = parametri
    y_modello = fit_erf_comune_func(x, A1, sigma1, A2, sigma2, x0, y0)
    return np.sum((y - y_modello) ** 2)


def main():
    if len(sys.argv) < 2:
        print("\nUso dello script:")
        print("  python fit_erf_powell.py <nome_file.txt>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.isfile(file_path):
        print(f"[ERRORE] Il file '{file_path}' non esiste.")
        sys.exit(1)

    # 3. Lettura dei dati
    y_data = []
    try:
        with open(file_path, "r") as f:
            for riga in f:
                riga_pulita = riga.strip()
                if not riga_pulita:
                    continue
                try:
                    y_data.append(float(riga_pulita))
                except ValueError:
                    continue
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file: {e}")
        sys.exit(1)

    if not y_data:
        print("[ERRORE] Nessun dato numerico trovato.")
        sys.exit(1)

    y_data = np.array(y_data)
    n_punti = len(y_data)
    x_data = np.arange(0, n_punti) * 0.2

    # 4. Guess dei parametri (p0)
    y_min, y_max = np.min(y_data), np.max(y_data)
    ampiezza_totale = y_max - y_min

    est_A1 = -ampiezza_totale * 0.7
    est_sigma1 = 1.5
    est_A2 = ampiezza_totale * 0.4
    est_sigma2 = 0.6
    est_x0 = 4.8
    est_y0 = y_max - (ampiezza_totale * 0.2)

    p0 = [est_A1, est_sigma1, est_A2, est_sigma2, est_x0, est_y0]

    # 5. Ottimizzazione con metodo POWELL e iterazioni aumentate
    # maxiter e maxfev controllano i limiti di calcolo
    options_powell = {"maxiter": 20000, "maxfev": 20000}

    print("\nAvvio minimizzazione con metodo Powell...")
    risultato = minimize(
        funzione_costo,
        x0=p0,
        args=(x_data, y_data),
        method="Powell",
        options=options_powell,
    )

    if not risultato.success:
        print(
            f"[ERRORE] L'ottimizzazione Powell ha fallito. Motivo: {risultato.message}"
        )
        sys.exit(1)

    # Estraiamo i parametri ottimali calcolati da Powell
    A1, sigma1, A2, sigma2, x0, y0 = risultato.x

    # Nota: Il metodo Powell non calcola direttamente la matrice di covarianza (pcov).
    # Se serve l'errore formale sui parametri, si usa di solito Levenberg-Marquardt.

    # 6. Stampa dei risultati sul terminale
    print("\n" + "=" * 50)
    print(" RISULTATI MINIMIZZAZIONE METODO POWELL")
    print("=" * 50)
    print(f" CENTRO UNICO (x0):    {x0:.4f}")
    print(f" BASELINE (y0):        {y0:.4f}")
    print("-" * 50)
    print(f" ERF 1:")
    print(f"   Ampiezza (A1):      {A1:.4f}")
    print(f"   Larghezza (σ1):     {sigma1:.4f}")
    print(f" ERF 2:")
    print(f"   Ampiezza (A2):      {A2:.4f}")
    print(f"   Larghezza (σ2):     {sigma2:.4f}")
    print("=" * 50)

    # 7. Generazione curve fluide per il grafico
    x_fit = np.linspace(x_data.min(), x_data.max(), 500)
    y_fit_totale = fit_erf_comune_func(x_fit, A1, sigma1, A2, sigma2, x0, y0)

    y_solo_erf1 = A1 * erf((x_fit - x0) / sigma1) + y0
    y_solo_erf2 = A2 * erf((x_fit - x0) / sigma2) + y0

    # 8. Plotting
    plt.figure(figsize=(10, 6))

    # Dati sperimentali
    plt.scatter(
        x_data,
        y_data,
        color="black",
        alpha=0.5,
        label="Dati Sperimentali",
        zorder=3,
    )

    # Fit complessivo Powell
    plt.plot(
        x_fit,
        y_fit_totale,
        color="teal",
        linewidth=3,
        label=f"Fit Powell (x₀={x0:.2f})",
        zorder=4,
    )

    # Componenti singole scisse
    plt.plot(
        x_fit,
        y_solo_erf1,
        color="royalblue",
        linestyle="--",
        alpha=0.7,
        label=f"Erf 1 (σ₁={sigma1:.2f})",
    )
    plt.plot(
        x_fit,
        y_solo_erf2,
        color="orangered",
        linestyle="--",
        alpha=0.7,
        label=f"Erf 2 (σ₂={sigma2:.2f})",
    )

    # Box parametri sul grafico
    testo_parametri = (
        f"Powell Centro x₀ = {x0:.2f}\n"
        f"A₁ = {A1:.1f} (σ₁ = {sigma1:.2f})\n"
        f"A₂ = {A2:.1f} (σ₂ = {sigma2:.2f})\n"
        f"y₀ = {y0:.1f}"
    )
    plt.text(
        0.05,
        0.70,
        testo_parametri,
        transform=plt.gca().transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.title(
        f"Fit Doppia Erf (Powell Method): {os.path.basename(file_path)}",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Asse X (step 0.2)")
    plt.ylabel("Asse Y")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()