import sys
import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script Batch Operations:")
        print("  python batch_operations.py -s <pattern_samples> [operazione] <singolo_file>")
        print("\nOperazioni disponibili (sceglierne una):")
        print("  -sub <file_back>       Sottrazione -> OUT: sample_MINUS_back.csv")
        print("  -m   <file_molt>       Moltiplicazione pura -> OUT: sample_TIMES_molt.csv")
        print("  -f   <file_filtro>     Filtro con interpolazione -> OUT: sample_FILTERED_BY_filtro.csv")
        print("\nEsempi:")
        print("  python batch_operations.py -s 10V/EXPORT*.csv -sub background.csv")
        print("  python batch_operations.py -s *.csv -f curva_risposta.csv\n")
        sys.exit(1)

    argomenti = sys.argv[1:]

    # 1. PARSING DEGLI ARGOMENTI
    mode = None  # Sarà 'sub', 'm' o 'f'
    second_file_path = None
    sample_patterns = []

    # Identifichiamo l'operazione richiesta e il secondo file
    if '-sub' in argomenti:
        mode = 'sub'
        idx = argomenti.index('-sub')
        if idx + 1 < len(argomenti): second_file_path = argomenti[idx + 1].replace('"', '').strip()
    elif '-m' in argomenti:
        mode = 'm'
        idx = argomenti.index('-m')
        if idx + 1 < len(argomenti): second_file_path = argomenti[idx + 1].replace('"', '').strip()
    elif '-f' in argomenti:
        mode = 'f'
        idx = argomenti.index('-f')
        if idx + 1 < len(argomenti): second_file_path = argomenti[idx + 1].replace('"', '').strip()

    if not mode or not second_file_path:
        print("[ERRORE] Devi specificare un'operazione valida (-sub, -m, -f) seguita dal relativo file.")
        sys.exit(1)

    if not os.path.isfile(second_file_path):
        print(f"[ERRORE] Il file operatore specificato non esiste: {second_file_path}")
        sys.exit(1)

    # Estraiamo tutto ciò che viene dopo la flag -s (fino alla flag dell'operazione)
    if '-s' in argomenti:
        idx_s = argomenti.index('-s')
        # Prendiamo tutti gli argomenti successivi a -s che non siano le flag operative o il secondo file
        for arg in argomenti[idx_s + 1:]:
            if arg in ['-sub', '-m', '-f', second_file_path]:
                break
            sample_patterns.append(arg)
    else:
        print("[ERRORE] Manca la flag obbligatoria -s per i file di campioni.")
        sys.exit(1)

    # 2. RACCOLTA DEI FILE DI SAMPLE (GESTIONE PATTERN *)
    files_sample = []
    for item in sample_patterns:
        item_pulito = item.replace('"', '').strip()
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
        for f in espanse:
            if f.lower().endswith('.csv'):
                files_sample.append(os.path.abspath(f))

    # Rimuove duplicati mantenendo l'ordine
    files_sample = list(dict.fromkeys(files_sample))

    # Rimuoviamo il secondo file dalla lista dei sample se per caso è caduto nel pattern *
    abs_second_file = os.path.abspath(second_file_path)
    if abs_second_file in files_sample:
        files_sample.remove(abs_second_file)

    if not files_sample:
        print("[ERRORE] Nessun file di sample valido trovato con il pattern fornito.")
        sys.exit(1)

    # 3. CARICAMENTO DEL SECONDO FILE (OPERATORE)
    try:
        df_2 = pd.read_csv(second_file_path, sep=' ')
        df_2.columns = df_2.columns.str.replace('"', '').str.strip()
        colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
        if not all(col in df_2.columns for col in colonne_obbligatorie):
            print(f"[ABORT] Il file operatore {os.path.basename(second_file_path)} non ha le colonne standard.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file operatore: {e}")
        sys.exit(1)

    second_name_no_ext = os.path.splitext(os.path.basename(second_file_path))[0]

    print("\n" + "="*60)
    print(f" AVVIO ELABORAZIONE BATCH")
    print(f" Modalità:             {mode.upper()}")
    print(f" File Operatore:       {os.path.basename(second_file_path)}")
    print(f" File Sample Trovati:  {len(files_sample)}")
    print("="*60 + "\n")

    # PRE-CALCOLO SE SIAMO IN MODALITÀ FILTRO
    if mode == 'f':
        f_min_nm = df_2['Wavelength (nm)'].min()
        f_max_nm = df_2['Wavelength (nm)'].max()
        funzione_filtro = interp1d(
            df_2['Wavelength (nm)'].values, 
            df_2['Intensity'].values, 
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

    # 4. CICLO DI ELABORAZIONE BATCH
    for file_path in files_sample:
        dir_name = os.path.dirname(file_path)
        sample_full_name = os.path.basename(file_path)
        sample_name_no_ext = os.path.splitext(sample_full_name)[0]
        
        try:
            df_s = pd.read_csv(file_path, sep=' ')
            df_s.columns = df_s.columns.str.replace('"', '').str.strip()
            
            if not all(col in df_s.columns for col in colonne_obbligatorie):
                print(f"   [SALTATO] {sample_full_name} non ha colonne standard.")
                continue

            # LOGICA A BIVIO IN BASE AL MODE
            if mode in ['sub', 'm']:
                # Controllo compatibilità rigido per Sottrazione e Moltiplicazione pura
                if len(df_s) != len(df_2):
                    print(f"   [SALTATO] {sample_full_name} ha lunghezza diversa ({len(df_s)} vs {len(df_2)})")
                    continue
                
                check_bins = np.array_equal(df_s['Bin'].values, df_2['Bin'].values)
                check_nm = np.allclose(df_s['Wavelength (nm)'].values, df_2['Wavelength (nm)'].values, atol=1e-4)
                check_cm = np.allclose(df_s['Relative Wavenumber (cm-1)'].values, df_2['Relative Wavenumber (cm-1)'].values, atol=1e-2)
                
                if not (check_bins and check_nm and check_cm):
                    print(f"   [SALTATO] {sample_full_name}: gli assi X non coincidono con l'operatore.")
                    continue
                
                # Calcolo effettivo
                df_output = df_s[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
                if mode == 'sub':
                    df_output['Intensity'] = df_s['Intensity'].values - df_2['Intensity'].values
                    suffix = f"_MINUS_{second_name_no_ext}.csv"
                else:
                    df_output['Intensity'] = df_s['Intensity'].values * df_2['Intensity'].values
                    suffix = f"_TIMES_{second_name_no_ext}.csv"

            elif mode == 'f':
                # Logica Filtro con ritaglio dinamico e interpolazione
                df_s_clipped = df_s[
                    (df_s['Wavelength (nm)'] >= f_min_nm - 1e-5) & 
                    (df_s['Wavelength (nm)'] <= f_max_nm + 1e-5)
                ].copy()
                
                if df_s_clipped.empty:
                    print(f"   [SALTATO] {sample_full_name} non ha nessuna regione in comune con il filtro.")
                    continue
                
                nm_sample = df_s_clipped['Wavelength (nm)'].values
                intensita_filtro_interpolata = funzione_filtro(nm_sample)
                
                df_output = df_s_clipped[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
                df_output['Intensity'] = df_s_clipped['Intensity'].values * intensita_filtro_interpolata
                suffix = f"_FILTERED_BY_{second_name_no_ext}.csv"

            # 5. SALVATAGGIO AUTOMATICO
            output_filename = f"{sample_name_no_ext}{suffix}"
            output_path = os.path.join(dir_name, output_filename)
            df_output.to_csv(output_path, sep=' ', index=False)
            print(f"   -> Creato: {output_filename}")

        except Exception as e:
            print(f"   [ERRORE] Impossibile elaborare {sample_full_name}: {e}")

    print("\n=== ELABORAZIONE BATCH COMPLETATA ===")

if __name__ == "__main__":
    main()