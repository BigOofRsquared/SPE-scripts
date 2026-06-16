import os
import sys
import warnings
import numpy as np
import pandas as pd

# Intercettiamo e ammutoliamo i warning prima di importare la libreria
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*SpeTools.*")

import spe2py as spe

# ==============================================================================
# CONFIGURAZIONE LASER: Imposta qui la lunghezza d'onda del tuo laser in nm
LASER_WAVELENGTH = 632.8
# ==============================================================================

def batch_convert_spe():
    argomenti = sys.argv[1:]
    
    # 1. GESTIONE FLAG: --raw e -v (verbosity)
    exclude_raw = '--raw' in argomenti
    verbose = '-v' in argomenti

    print(f"=== AVVIO CONVERSIONE AUTOMATICA (Laser: {LASER_WAVELENGTH} nm) ===")
    print(f"Filtro globale --raw: {'ATTIVO (cerca *-raw*.spe)' if exclude_raw else 'DISATTIVO'}")
    print(f"Modalità Verbose (-v): {'ATTIVA' if verbose else 'DISATTIVA (Solo errori)'}")
    print("-" * 60)
    
    # 2. Trova tutti i file .spe ricorsivamente con filtro selettivo
    spe_files = []
    file_saltati_count = 0

    for root, _, files in os.walk('.'):
        for file in files:
            if file.lower().endswith('.spe'):
                # Applica il filtro chirurgico richiesto solo se la flag è attiva
                if exclude_raw and "-raw" in file.lower():
                    if verbose:
                        print(f"   [SALTATO] {file}: Escluso tramite flag --raw")
                    file_saltati_count += 1
                    continue
                spe_files.append(os.path.abspath(os.path.join(root, file)))
                
    total_files = len(spe_files)
    if total_files == 0:
        print(f"\n[AVVISO] Nessun file .spe da elaborare (Saltati tramite --raw: {file_saltati_count}).")
        return
        
    print(f"File totali da elaborare: {total_files} (Saltati tramite --raw: {file_saltati_count})")
    print("-" * 60)
    
    # 3. HACK: Forza spe2py a prendere la lista dei file senza aprire la GUI
    if verbose:
        print("Caricamento dei file in corso...")
    
    spe.fdialog.askopenfilenames = lambda **kwargs: spe_files
    
    try:
        spe_tool = spe.load()
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Impossibile caricare i file tramite spe2py: {e}")
        sys.exit(1)

    # Gestione del ritorno di spe.load()
    if isinstance(spe_tool, list):
        loaded_files = spe_tool
    elif hasattr(spe_tool, 'files') and spe_tool.files:
        loaded_files = spe_tool.files
    else:
        loaded_files = [spe_tool]

    # 4. Ciclo di esportazione
    for idx, obj in enumerate(loaded_files):
        full_path = spe_files[idx]
        dir_name = os.path.dirname(full_path)
        file_name = os.path.basename(full_path)
        base_name, _ = os.path.splitext(file_name)
        
        if verbose:
            print(f"In lavorazione: {file_name}")
        
        try:
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

                # DataFrame strutturato
                df = pd.DataFrame({
                    'Bin': bins,
                    'Wavelength (nm)': wavelengths_col,
                    'Relative Wavenumber (cm-1)': wavenumbers_col,
                    'Intensity': intensities
                })

                output_name = f"EXPORT_{base_name}_{f_idx + 1:03d}.csv"
                df.to_csv(os.path.join(dir_name, output_name), sep=' ', index=False)
                
                if verbose:
                    print(f"   -> Creato: {output_name}")
                    
            if verbose:
                print("-" * 60)

        except Exception as e:
            # Questo viene stampato SEMPRE, anche senza la flag -v attiva
            print(f"\n[ERRORE] Fallita l'elaborazione del file: {file_name}")
            print(f"          Dettaglio errore: {e}")
            print("-" * 60)

    print("\n=== SCRIPT COMPLETATO ===")

if __name__ == "__main__":
    batch_convert_spe()