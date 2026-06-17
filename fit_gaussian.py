import io
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
import pandas as pd

# 1. Controllo dell'argomento da terminale
if len(sys.argv) < 2:
    print("Errore: Devi passare il percorso del file da terminale!")
    print("Uso: python fit_gaussian.py <percorso_del_file>")
    sys.exit(1)

percorso = sys.argv[1]

if not os.path.exists(percorso):
    print(f"Errore: Il file '{percorso}' non esiste.")
    sys.exit(1)

print(f"Caricamento file: {percorso}")

# 2. Lettura del file con Pandas
df = pd.read_csv(percorso, sep=r"\s+")
x_data = df["Wavelength (nm)"].values
y_data = df["Intensity"].values

# Modello fisico: Corpo nero di Planck * Risposta strumentale * Filtri sigmoidei
# PARAMETRI (13 totali):
# x, A, T, c0, c1, c2, c3, a, lam_cut1, sigma1, d, lam_cut2, sigma2
def polinomio_risposta(x, x0, *coefficienti):
    """
    Calcola un polinomio di grado arbitrario centrato in x0.
    I coefficienti vengono passati come una lista/tupla grazie a *coefficienti.
    """
    x_minus_x0 = x - x0
    risultato = 0
    
    # Il ciclo for calcola automaticamente: c0 + c1*dx + c2*dx^2 + ... + cn*dx^n
    for grado, c in enumerate(coefficienti):
        risultato += c * (x_minus_x0 ** grado)
        
    return risultato

def somma_sigmoidi(x, *parametri_sigmoidi):
    """
    Calcola la somma di N sigmoidi (basate sulla funzione di errore 'erf').
    I parametri devono essere passati in gruppi di 3 per ogni sigmoide:
    [peso_1, cut_1, sigma_1, peso_2, cut_2, sigma_2, ...]
    """
    # Inizializziamo il filtro a zero
    filtro = np.zeros_like(x, dtype=float)
    
    # Cicliamo di 3 in 3 lungo la lista dei parametri
    for i in range(0, len(parametri_sigmoidi), 3):
        # Estraiamo la tripletta per la sigmoide corrente
        a = parametri_sigmoidi[i]
        lam_cut = parametri_sigmoidi[i+1]
        sigma = parametri_sigmoidi[i+2]
        
        # Protezione contro sigma <= 0 (come nel tuo codice originale)
        s = np.where(sigma <= 0, 1e-5, sigma)
        
        # Correzione del bug delle parentesi: la divisione per 's' deve stare DENTRO l'erf
        filtro += a * (1.0 + erf((x - lam_cut) / s))
        
    return filtro

def teorical_model(x, A, T, a, lam_cut1, sigma1, d, lam_cut2, sigma2, *c):
    lam = x * 1e-9 
    h, c_vel, kB = 6.62607015e-34, 299792458, 1.380649e-23
    
    # 1. Componente Corpo Nero (Planck)
    exponent = (h * c_vel) / (lam * kB * T)
    exponent = np.clip(exponent, None, 7000) 
    planck = (2 * h * c_vel**2) / (lam**5 * (np.exp(exponent) - 1))
    # 2. Polinomio di correzione della risposta
    polinomio = polinomio_risposta(x, 650, *c)
    
    # 3. Doppia Sigmoide (I due gradini di salita)
    # Protezione contro divisioni per zero o numeri negativi
    s1 = np.where(sigma1 <= 0, 1e-5, sigma1)
    s2 = np.where(sigma2 <= 0, 1e-5, sigma2)
    
    filtro = a * (1.0 + erf((x - lam_cut1) / s1)) + d * (1.0 + erf((x - lam_cut2)/ s2))
    return A * planck * polinomio * filtro

# 3. Definiamo la stima iniziale (CORRETTA A 13 PARAMETRI)
# Associazioni:  [  A,     T,    c0,  c1,  c2,  c3,   a,  lam_cut1, sigma1,  d,  lam_cut2, sigma2]
# NOTA: Visto che la Planck sputa numeri nell'ordine di 10^11 o 10^12 in questa regione, 
# la costante di ampiezza 'A' deve essere piccolissima (es. 1e-9 o meno) per ridimensionarla su intensità umane (~250).
p0_fisici = [4e-5, 1500]

# 19 coefficienti del polinomio (c0=1, tutti gli altri a 0)
#p0_polinomio = ( [7.22445624e-03] + [2.22332308e-06] + [3.52642101e-04] \
#                     + [ 1.14381026e-04] + [-8.00964479e-06] + [-1.09976084e-06] \
#                     + [1.24355703e-07] + [-3.64660156e-09] + [-5.16073404e-12] + [ 1.18186437e-12] \
#                     + [1.38289741e-14] + [-2.79311248e-16] + [-1.01137092e-17] + [-6.18714586e-20] \
#                     + [3.28495398e-21] + [ 8.70851129e-23] + [-1.96293714e-25] + [-5.05919894e-26] \
#                     + [5.01323391e-28] + [0.0] * 5)
p0_polinomio = [1.0]
p0_sigmoide = [0.28, 646.0, 10.0, 0.2, 680, 10.0]

#stima_iniziale = [4.23867098e-05]+[2.14134393e+03]+[2.82578220e-01]+[6.58622002e+02]+[2.38970827e+01]+[2.82578136e-01]
stima_iniziale = p0_fisici + p0_sigmoide + p0_polinomio

# Generiamo i punti X per le curve fluide
x_plot = np.linspace(min(x_data), max(x_data), 500)

# Calcoliamo i valori Y del GUESS INIZIALE prima di fare il fit
y_guess = teorical_model(x_plot, *stima_iniziale)

# 4. Esecuzione del fit vero e proprio
# 4. Esecuzione del fit vero e proprio
try:
    popt, pcov = curve_fit(teorical_model, x_data, y_data, p0=stima_iniziale, maxfev=10000)
    fit_success = True
    print("\n[INFO] Fit completato con successo!")
except Exception as e:
    print(f"\n[ERRORE] Il fit ha fallito. Mostro comunque il Guess Iniziale. Dettaglio: {e}")
    fit_success = False

# 5. Grafico combinato
plt.figure(figsize=(10, 6))

# I tuoi punti sperimentali
plt.scatter(x_data, y_data, color='black', alpha=0.7, label='Dati Sperimentali (Spettro)', zorder=2)

# Grafico del GUESS INIZIALE (Tratteggiato blu)
plt.plot(x_plot, y_guess, 'b--', lw=1.5, label='Guess Iniziale (p0)')

# Grafico del FIT (Linea continua rossa, eseguito SOLO se fit_success è True)
if fit_success:
    plt.plot(x_plot, teorical_model(x_plot, *popt), 'r-', lw=2.5, label='Fit Ottimizzato')
    
    # --- CORREZIONE QUI ---
    # Estraiamo i coefficienti (saltando A e T)
    coefficienti_ottimizzati = popt[2:]
    # Usiamo l'asterisco * per spacchettare la lista dei coefficienti!
    y_filter = polinomio_risposta(x_plot, 650, *coefficienti_ottimizzati)
    
    # Se vuoi plottare ANCHE solo la curva del polinomio per vederne la forma, scommenta la riga sotto:
    #plt.plot(x_plot, y_filter, 'g-.', label='Solo Filtro Polinomiale (Risposta)')

    # Mostriamo i risultati a schermo con gli indici corretti
    print("\n--- RISULTATI DEL FIT ---")
    print(f"Costante di scala (A): {popt[0]:.2e}")
    print(f"Temperatura stimata (T): {popt[1]:.1f} K")
else:
    plt.title('CONFRONTO: Dati reali vs Il tuo Guess Iniziale (Il fit è fallito)')

plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
print(popt)
plt.show()

