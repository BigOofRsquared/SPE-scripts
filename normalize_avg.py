import sys
import glob
import os
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script di normalizzazione:")
        print("  Normalizza tutto:   python normalize_pattern.py *.csv")
        print("  Escludere i RAW:    python normalize_pattern.py -r *.csv\n")
        sys.exit(1)

    argomenti = sys.argv[1:]

    # 1. GESTIONE FLAG: Stessa logica blindata del plotter
    exclude_raw = '-r' in argomenti

    # 2. PULIZIA ARGOMENTI: Rimuoviamo i flag per tenere solo i file
    file_args = [arg for arg in argomenti if arg not in ['-r']]

    # 3. RACCOLTA FILE
    files_to_process = []
    for item in file_args:
        item_pulito = item.replace('"', '').strip()
        
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
            
        for f in espanse:
            if f.lower().endswith('.csv'):
                # Salta i file RAW se c'è il flag -r
                if exclude_raw and "-raw" in os.path.basename(f).lower():
                    continue
                # Evita di rinormalizzare file già normalizzati in precedenza
                if os.path.basename(f).startswith("NORM_"):
                    continue
                files_to_process.append(os.path.abspath(f))

    # Rimuove i duplicati mantenendo l'ordine
    files_to_process = list(dict.fromkeys(files_to_process))

    if not files_to_process:
        print("[ERRORE] Nessun file CSV valido trovato da normalizzare.")
        sys.exit(1)

    print("\n" + "="*60)
    print(f" AVVIO NORMALIZZAZIONE (Media ad 1.0)")
    print(f" File totali da elaborare: {len(files_to_process)}")
    print(f" Esclusione file RAW: {exclude_raw}")
    print("="*60 + "\n")

    # 4. CICLO DI NORMALIZZAZIONE
    for file_path in files_to_process:
        dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        try:
            # Carica il dataframe
            df = pd.read_csv(file_path, sep=' ')
            
            # Pulisce i nomi delle colonne da spazi o virgolette residue
            df.columns = df.columns.str.replace('"', '').str.strip()
            
            if 'Intensity' not in df.columns:
                print(f"   [SALTATO] {file_name} non contiene la colonna 'Intensity'")
                continue
                
            # Calcola la media dell'intensità attuale
            mean_intensity = df['Intensity'].mean()
            
            if mean_intensity == 0:
                print(f"   [ERRORE] {file_name} ha media intensità pari a 0! Impossibile dividere.")
                continue
            
            # Applica la normalizzazione matematica (Media = 1.0)
            df['Intensity'] = df['Intensity'] / mean_intensity
            
            # Genera il nuovo nome file aggiungendo NORM_ all'inizio
            output_name = f"NORM_{file_name}"
            output_path = os.path.join(dir_name, output_name)
            
            # Salva il nuovo CSV mantenendo la formattazione originale a spazi
            df.to_csv(output_path, sep=' ', index=False)
            print(f"   -> Creato: {output_name} (Media originale: {mean_intensity:.2f})")
            
        except Exception as e:
            print(f"   [ERRORE] Impossibile elaborare {file_name}: {e}")

    print("\n=== ELABORAZIONE COMPLETATA ===")

if __name__ == "__main__":
    main()