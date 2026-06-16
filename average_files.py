import sys
import glob
import os
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script di mediazione:")
        print("  python average_pattern.py -o <nome_output.csv> [flag] <pattern_file>")
        print("\nFlag disponibili:")
        print("  -r                 Esclude i file RAW (*-raw*.csv) dalla media")
        print("\nEsempi:")
        print("  python average_pattern.py -o media_10V.csv -r 10V/EXPORT*.csv")
        print("  python average_pattern.py -r -o media_all.csv *.csv\n")
        sys.exit(1)

    # Prendiamo la lista grezza degli argomenti
    argomenti = sys.argv[1:]

    # 1. GESTIONE FLAG -r: Controlliamo se è presente
    exclude_raw = '-r' in argomenti

    # 2. PARSING DEL NOME FILE DI OUTPUT (-o <filename>)
    output_filename = None
    if '-o' in argomenti:
        idx_o = argomenti.index('-o')
        if idx_o + 1 < len(argomenti):
            output_filename = argomenti[idx_o + 1].replace('"', '').strip()
            # Rimuoviamo il valore dell'output dagli argomenti
            del argomenti[idx_o + 1]
        else:
            print("[ERRORE] Specificare il nome del file dopo il flag -o")
            sys.exit(1)

    # 3. PULIZIA ARGOMENTI: Rimuoviamo i letterali dei flag per tenere solo i file
    file_args = [arg for arg in argomenti if arg not in ['-o', '-r', output_filename]]

    if not output_filename:
        print("[ERRORE] Parametro obbligatorio di output mancante. Usa la flag: -o <nome_file.csv>")
        sys.exit(1)

    # Assicuriamoci che l'output finisca con .csv
    if not output_filename.lower().endswith('.csv'):
        output_filename += '.csv'

    # 4. RACCOLTA E FILTRAGGIO DEI FILE DA MEDIARE
    files_to_average = []
    for item in file_args:
        item_pulito = item.replace('"', '').strip()
        
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
            
        for f in espanse:
            if f.lower().endswith('.csv'):
                # Salta i file RAW se c'è la flag -r
                if exclude_raw and "-raw" in os.path.basename(f).lower():
                    continue
                # Salta l'output stesso se per caso ricade nel pattern globale *
                if os.path.abspath(f) == os.path.abspath(output_filename):
                    continue
                files_to_average.append(os.path.abspath(f))

    # Rimuove duplicati mantenendo l'ordine
    files_to_average = list(dict.fromkeys(files_to_average))

    if len(files_to_average) < 2:
        print(f"[ERRORE] Trovati {len(files_to_average)} file validi. Per fare una media servono almeno 2 file.")
        sys.exit(1)

    print("\n" + "="*60)
    print(f" AVVIO MEDIAZIONE SPETTRI")
    print(f" File totali da mediare: {len(files_to_average)}")
    print(f" Esclusione file RAW (-r): {exclude_raw}")
    print(f" File di output impostato: {output_filename}")
    print("="*60 + "\n")

    # 5. CONTROLLO DI COINCIDENZA E ACCUMULO DATI
    struttura_riferimento = None
    intensita_totale = []

    for idx, file_path in enumerate(files_to_average):
        file_name = os.path.basename(file_path)
        
        try:
            df = pd.read_csv(file_path, sep=' ')
            df.columns = df.columns.str.replace('"', '').str.strip()
            
            # Verifichiamo le colonne
            colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
            if not all(col in df.columns for col in colonne_obbligatorie):
                print(f"[ABORT] Il file {file_name} non ha tutte le colonne Raman standard.")
                sys.exit(1)
            
            struttura_attuale = df[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']]
            
            if idx == 0:
                struttura_riferimento = struttura_attuale.copy()
                intensita_totale.append(df['Intensity'].values)
                print(f"   [OK] Riferimento iniziale: {file_name}")
            else:
                if len(struttura_attuale) != len(struttura_riferimento):
                    print(f"[ABORT] {file_name} ha una lunghezza di punti ({len(struttura_attuale)}) diversa dal primo file ({len(struttura_riferimento)}).")
                    sys.exit(1)
                
                check_bins = np.array_equal(struttura_attuale['Bin'].values, struttura_riferimento['Bin'].values)
                check_nm = np.allclose(struttura_attuale['Wavelength (nm)'].values, struttura_riferimento['Wavelength (nm)'].values, atol=1e-4)
                check_cm = np.allclose(struttura_attuale['Relative Wavenumber (cm-1)'].values, struttura_riferimento['Relative Wavenumber (cm-1)'].values, atol=1e-2)
                
                if not (check_bins and check_nm and check_cm):
                    print(f"[ABORT] Gli assi X (Bin, Wavelength o cm-1) di {file_name} NON coincidono con il riferimento.")
                    sys.exit(1)
                
                intensita_totale.append(df['Intensity'].values)
                print(f"   [OK] Verificato e aggiunto: {file_name}")

        except Exception as e:
            print(f"[ABORT] Errore critico su {file_name}: {e}")
            sys.exit(1)

    # 6. CALCOLO DELLA MEDIA E SALVATAGGIO FINALIZZATO
    print("\nCalcolo della media aritmetica in corso...")
    matrice_intensita = np.array(intensita_totale)
    media_intensita = np.mean(matrice_intensita, axis=0)

    df_output = struttura_riferimento.copy()
    df_output['Intensity'] = media_intensita

    df_output.to_csv(output_filename, sep=' ', index=False)
    
    print("="*60)
    print(f"[SUCCESSO] File mediato creato senza i RAW: {output_filename}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()