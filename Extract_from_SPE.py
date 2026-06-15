import os
import sys
import numpy as np
import pandas as pd
import spe2py as spe

# ==============================================================================
# CONFIGURAZIONE LASER: Imposta qui la lunghezza d'onda del tuo laser in nm
LASER_WAVELENGTH = 785.0
# ==============================================================================

# 1. CONTROLLO PARAMETRO DI CHIMATA
if len(sys.argv) < 2:
    print("\n[ERRORE] Manca il nome del file di output!")
    print("Uso corretto:")
    print("  python Extract_from_SPE.py <nome_file_output.csv>")
    sys.exit(1)

output_file = sys.argv[1]
if not output_file.lower().endswith('.csv'):
    output_file += '.csv'

# Separiamo il nome dall'estensione (es. "misura_test" e ".csv")
file_base, file_ext = os.path.splitext(output_file)


# 2. SELEZIONE INTERATTIVA DEL FILE .SPE
print(f"\n--> Laser impostato: {LASER_WAVELENGTH} nm")
print("--> Si sta aprendo la finestra di selezione. Scegli il file .spe...")

spe_tool = spe.load()

if not spe_tool or not hasattr(spe_tool, 'file') or spe_tool.file is None:
    print("[ANNULLATO] Selezione annullata. Uscita.")
    sys.exit(0)

spe_file = spe_tool.file

try:
    # Conteggio dei frame disponibili nel file
    num_frames = len(spe_file.data)
    print(f"File SPE caricato con successo. Frame totali rilevati: {num_frames}")

    # 3. Recupera la calibrazione (comune a tutti i frame)
    wavelengths = None
    relative_wavenumbers = None
    has_calibration = False

    if hasattr(spe_file, 'wavelength') and spe_file.wavelength is not None:
        wavelengths = spe_file.wavelength.flatten()
        
        # Se la calibrazione è valida, calcoliamo il Raman Shift una volta sola
        if len(wavelengths) > 0:
            wavelengths_nm = np.array(wavelengths, dtype=float)
            relative_wavenumbers = (1.0 / LASER_WAVELENGTH - 1.0 / wavelengths_nm) * 1e7
            relative_wavenumbers = np.round(relative_wavenumbers, 2)
            has_calibration = True

    # 4. LOOP DI ESPORTAZIONE PER OGNI FRAME
    for f_idx in range(num_frames):
        # Estraiamo le intensità del frame corrente [frame_index][roi_index]
        intensities = spe_file.data[f_idx][0].flatten()
        bins = list(range(1, len(intensities) + 1))

        # Gestione dei vettori calibrazione in caso di assenza dati
        if not has_calibration:
            wavelengths = ["Non presente"] * len(intensities)
            relative_wavenumbers = ["N/A"] * len(intensities)

        # Creazione del DataFrame per il frame corrente
        df = pd.DataFrame({
            'Bin': bins,
            'Wavelength (nm)': wavelengths,
            'Relative Wavenumber (cm-1)': relative_wavenumbers,
            'Intensity': intensities
        })

        # DETERMINAZIONE DEL NOME FILE
        # Se c'è più di un frame, aggiungiamo il suffisso numerico _001, _002, ecc.
        if num_frames > 1:
            current_output_name = f"{file_base}_{f_idx + 1:03d}{file_ext}"
        else:
            current_output_name = output_file

        # Salvataggio del CSV
        df.to_csv(current_output_name, index=False)
        print(f" -> Esportato: {current_output_name}")

    print(f"\n[SUCCESSO] Esportazione completata! Generati {num_frames} file.\n")

except Exception as e:
    print(f"\n[ERRORE] Si è verificato un problema durante l'estrazione: {e}\n")