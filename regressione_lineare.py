import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

x = np.array([10.5, 20.5, 30.5, 40.5, 60.5])
y = np.array([1.38619, 2.9969, 4.19336, 5.92063, 9.05287])
y_err = np.array(
    [1.687140e-01, 9.381876e-01, 6.189007e-01, 8.459864e-01, 8.259582e-01] / np.sqrt(20)
)


# 2. Definiamo la funzione lineare per curve_fit
def retta(x, m, q):
    return m * x + q


# 3. Calcolo della regressione lineare pesata con curve_fit
# - sigma=y_err indica le incertezze sulle Y
# - absolute_sigma=True forza l'algoritmo a considerare i valori di y_err come errori assoluti sperimentali
popt, pcov = curve_fit(retta, x, y, sigma=y_err, absolute_sigma=True)
m, q = popt

# Calcolo delle deviazioni standard (incertezze) dei parametri fittati
m_err, q_err = np.sqrt(np.diag(pcov))

print(f"  Counts/sec (m): {m:.6e} ± {m_err:.2e}")
print(f"  Offset (q):     {q:.6e} ± {q_err:.2e}")

# 4. Creazione della stringa con l'equazione della retta
equazione_testo = f"y = ({m:.4f}±{m_err:.4f})x + ({q:.4f}±{q_err:.4f})"

# 5. Calcolo dei punti della retta di regressione (usa linspace per farla fluida)
x_fit = np.linspace(min(x) - 5, max(x) + 5, 100)
y_pred = retta(x_fit, m, q)

# 6. Plotting dei dati con barre d'errore e della retta
plt.figure(figsize=(8, 6))

# Usiamo plt.errorbar invece di plt.scatter per mostrare l'incertezza dei dati
plt.errorbar(
    x,
    y,
    yerr=y_err,
    fmt=".",
    markersize = 11,
    color="blue",
    ecolor="blue",
    capsize=4,
    alpha=0.7,
    label="Dati sperimentali",
    zorder=3,
    linewidth=2
)

# Line plot della retta di regressione
plt.plot(
    x_fit,
    y_pred,
    color="grey",
    linestyle="-.",
    linewidth=1,
    label="Regressione Pesata",
    zorder=2,
)

# Inseriamo i risultati del fit come testo direttamente nel grafico
plt.text(
    12,
    max(y) * 0.85,
    f"m = {m:.4f} ± {m_err:.4f}\nq = {q:.4f} ± {q_err:.4f}",
    fontsize=15,
    color="darkblue",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
)

# Personalizzazione del grafico
#plt.title(
#    "Regressione Lineare Pesata (Minimi Quadrati)",
#    fontsize=14,
#    fontweight="bold",
#)
plt.xlabel("Exposure time (s)", fontsize=20)
plt.ylabel(r"$\Delta$Counts (thermal)", fontsize=20)
plt.tick_params(axis='both', labelsize=15)
plt.grid(True, linestyle="--", alpha=0.6)
#plt.legend(fontsize=11, loc="upper left")

# Mostra il grafico
plt.show()