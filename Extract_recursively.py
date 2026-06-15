import os
import sys
import warnings
import numpy as np
import pandas as pd

# Intercettiamo e ammutoliamo i warning prima di importare la libreria
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SpeTools.*")

import spe2py as spe

# ==============================================================================
# CONFIGURAZIONE LASER: Imposta qui la lunghezza d'onda del tuo laser in nm
LASER_WAVELENGTH = 632.8
# ==============================================================================

def batch_convert_spe():
    print(f"=== AVVIO CONVERSIONE AUTOMATICA (Laser: {LASER_WAVELENGTH} nm) ===")
    
    # 1. Trova tutti i file .spe ricorsivamente
    spe_files = []
    for root, _, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.spe'):
                spe_files.append(os.path.abspath(os.path.join(root, file)))
                
    total_files = len(spe_files)
    if total_files == 0:
        print("[AVVISO] Nessun file .spe trovato.")
        return
        
    print(f"Trovati {total_files} file .spe da elaborare.\n" + "-"*50)
    
    # 2. HACK: Forza spe2py a prendere la lista dei file senza aprire la GUI
    spe.fdialog.askopenfilenames = lambda **kwargs: spe_files
    
    print("Caricamento dei file...")
    spe_tool = spe.load()

    # Gestione del ritorno di spe.load()
    if isinstance(spe_tool, list):
        loaded_files = spe_tool
    elif hasattr(spe_tool, 'files') and spe_tool.files:
        loaded_files = spe_tool.files
    else:
        loaded_files = [spe_tool]

    # 3. Ciclo di esportazione
    for idx, obj in enumerate(loaded_files):
        full_path = spe_files[idx]
        dir_name = os.path.dirname(full_path)
        file_name = os.path.basename(full_path)
        base_name, _ = os.path.splitext(file_name)
        
        print(f"In lavorazione: {file_name}")
        
        # Uso della sintassi aggiornata
        if hasattr(obj, 'file') and obj.file is not None:
            spe_file = obj.file
        else:
            spe_file = obj

        # Esportazione frame per frame
        num_frames = len(spe_file.data)
        for f_idx in range(num_frames):
            # L'intensità comanda la lunghezza del frame attuale
            intensities = spe_file.data[f_idx][0].flatten()
            num_points = len(intensities)
            bins = list(range(1, num_points + 1))

            # Gestione della calibrazione con controllo ferreo della lunghezza
            wavelengths_col = ["Non presente"] * num_points
            wavenumbers_col = ["N/A"] * num_points

            if hasattr(spe_file, 'wavelength') and spe_file.wavelength is not None:
                wavelengths_raw = spe_file.wavelength.flatten()
                
                if len(wavelengths_raw) > 0:
                    # Se la lunghezza combacia perfettamente, usiamo i dati pronti
                    if len(wavelengths_raw) == num_points:
                        wavelengths_nm = np.array(wavelengths_raw, dtype=float)
                    else:
                        # Se differisce (es. asse mappato male), interpoliamo sui punti reali dell'intensità
                        wavelengths_nm = np.interp(
                            np.linspace(0, 1, num_points), 
                            np.linspace(0, 1, len(wavelengths_raw)), 
                            wavelengths_raw
                        )
                    
                    # Calcolo dei cm-1 sui dati validati e pareggiati
                    relative_wavenumbers = (1.0 / LASER_WAVELENGTH - 1.0 / wavelengths_nm) * 1e7
                    
                    # Popoliamo le colonne finali convertendole in liste con lunghezza garantita
                    wavelengths_col = np.round(wavelengths_nm, 4).tolist()
                    wavenumbers_col = np.round(relative_wavenumbers, 2).tolist()

            # Qui non può più fallire: tutte le liste hanno lunghezza 'num_points'
            df = pd.DataFrame({
                'Bin': bins,
                'Wavelength (nm)': wavelengths_col,
                'Relative Wavenumber (cm-1)': wavenumbers_col,
                'Intensity': intensities
            })

            output_name = f"EXPORT_{base_name}_{f_idx + 1:03d}.csv"
            df.to_csv(os.path.join(dir_name, output_name), sep=' ', index=False)
            print(f"   -> Creato: {output_name}")
            
        print("-" * 50)

    print("\n=== SCRIPT COMPLETATO ===")

if __name__ == "__main__":
    batch_convert_spe()