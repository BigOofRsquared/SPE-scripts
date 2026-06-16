import sys
import os
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script Batch Operations (4-in-1):")
        print("  python batch_operations.py -s <pattern_samples> [operazione] <singolo_file> [opzioni_finestra]")
        print("\nOperazioni disponibili (sceglierne una):")
        print("  -sub <file_back>       Sottrazione -> OUT: sample_MINUS_back.csv")
        print("  -m   <file_molt>       Moltiplicazione pura -> OUT: sample_TIMES_molt.csv")
        print("  -f   <file_filtro>     Filtro con interpolazione -> OUT: sample_FILTERED_BY_filtro.csv")
        print("  -res <file_rif>        Riscalamento locale -> OUT: sample_RESCALED_TO_rif.csv")
        print("\nOpzioni Finestra (Obbligatorie SOLO per il riscalamento -res, sceglierne una):")
        print("  -w <min> <max>         Intervallo in Wavelength (nm)")
        print("  -rwn <min> <max>       Intervallo in Relative Wavenumber (cm-1)")
        print("  -b <min> <max>         Intervallo in Bin (pixel)")
        print("\nEsempi:")
        print("  python batch_operations.py -s 10V/EXPORT*.csv -sub background.csv")
        print("  python batch_operations.py -s fluo*.csv -res sample.csv -w 640 650\n")
        sys.exit(1)

    argomenti = sys.argv[1:]

    # 1. PARSING DELL'OPERAZIONE E DEL SECONDO FILE
    mode = None  # 'sub', 'm', 'f', o 'res'
    second_file_path = None

    for flag, m_type in [('-sub', 'sub'), ('-m', 'm'), ('-f', 'f'), ('-res', 'res')]:
        if flag in argomenti:
            if mode is not None:
                print("[ERRORE] Puoi specificare una sola operazione alla volta tra -sub, -m, -f, -res.")
                sys.exit(1)
            mode = m_type
            idx = argomenti.index(flag)
            if idx + 1 < len(argomenti):
                second_file_path = argomenti[idx + 1].replace('"', '').strip()

    if not mode or not second_file_path:
        print("[ERRORE] Devi specificare un'operazione valida (-sub, -m, -f, -res) seguita dal relativo file.")
        sys.exit(1)

    if not os.path.isfile(second_file_path):
        print(f"[ERRORE] Il file operatore specificato non esiste: {second_file_path}")
        sys.exit(1)

    # 2. PARSING DELLA FINESTRA DI INTEGRAZIONE (SOLO SE IN MODALITÀ -res)
    window_type = None
    w_min, w_max = None, None

    for flag in ['-w', '-rwn', '-b']:
        if flag in argomenti:
            if window_type is not None:
                print("[ERRORE] Puoi specificare una sola finestra alla volta (-w, -rwn o -b).")
                sys.exit(1)
            window_type = flag.replace('-', '')
            idx = argomenti.index(flag)
            try:
                w_min = float(argomenti[idx + 1])
                w_max = float(argomenti[idx + 2])
            except (IndexError, ValueError):
                print(f"[ERRORE] La flag {flag} richiede due valori numerici (min e max).")
                sys.exit(1)

    if mode == 'res' and not window_type:
        print("[ERRORE] La modalità riscalamento (-res) richiede obbligatoriamente un intervallo (-w, -rwn o -b).")
        sys.exit(1)

    # 3. PARSING DEI FILE DI SAMPLE (-s)
    sample_patterns = []
    if '-s' in argomenti:
        idx_s = argomenti.index('-s')
        # Raccogliamo i sample fermandoci se becchiamo altre flag operative o di finestra
        for arg in argomenti[idx_s + 1:]:
            if arg in ['-sub', '-m', '-f', '-res', '-w', '-rwn', '-b', second_file_path]:
                break
            sample_patterns.append(arg)
    else:
        print("[ERRORE] Manca la flag obbligatoria -s per i file di campioni.")
        sys.exit(1)

    # Raccolta reale ed espansione dei file con glob
    files_sample = []
    for item in sample_patterns:
        item_pulito = item.replace('"', '').strip()
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
        for f in espanse:
            if f.lower().endswith('.csv'):
                files_sample.append(os.path.abspath(f))

    # Rimozione duplicati mantenendo l'ordine
    files_sample = list(dict.fromkeys(files_sample))

    # Escludiamo l'operatore/riferimento dalla lista dei sample se cade nel pattern *
    abs_second_file = os.path.abspath(second_file_path)
    if abs_second_file in files_sample:
        files_sample.remove(abs_second_file)

    if not files_sample:
        print("[ERRORE] Nessun file di sample valido trovato con il pattern fornito.")
        sys.exit(1)

    # 4. CARICAMENTO DEL FILE OPERATORE / RIFERIMENTO
    colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
    try:
        df_2 = pd.read_csv(second_file_path, sep=' ')
        df_2.columns = df_2.columns.str.replace('"', '').str.strip()
        if not all(col in df_2.columns for col in colonne_obbligatorie):
            print(f"[ABORT] Il file {os.path.basename(second_file_path)} non ha le colonne standard.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file operatore/riferimento: {e}")
        sys.exit(1)

    second_name_no_ext = os.path.splitext(os.path.basename(second_file_path))[0]

    print("\n" + "="*60)
    print(f" AVVIO ELABORAZIONE BATCH (4-in-1)")
    print(f" Modalità Operativa:   {mode.upper()}")
    print(f" File Operatore/Rif:   {os.path.basename(second_file_path)}")
    if mode == 'res':
        print(f" Finestra Scelta:      -{window_type} [{w_min} : {w_max}]")
    print(f" File Sample Trovati:  {len(files_sample)}")
    print("="*60 + "\n")

    # PRE-CALCOLI SPECIFICI PER LE MODALITÀ
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
    elif mode == 'res':
        # Prepariamo la maschera booleana sul file di riferimento per il calcolo della somma locale
        if window_type == 'w':
            mask_ref = (df_2['Wavelength (nm)'] >= w_min) & (df_2['Wavelength (nm)'] <= w_max)
        elif window_type == 'rwn':
            mask_ref = (df_2['Relative Wavenumber (cm-1)'] >= w_min) & (df_2['Relative Wavenumber (cm-1)'] <= w_max)
        else:
            mask_ref = (df_2['Bin'] >= w_min) & (df_2['Bin'] <= w_max)
        
        somma_ref = df_2[mask_ref]['Intensity'].sum()
        if somma_ref == 0:
            print("[ABORT] La somma dell'intensità nell'intervallo sul file di riferimento è ZERO. Calcolo impossibile.")
            sys.exit(1)

    # 5. CICLO DI ELABORAZIONE BATCH SUI SAMPLES
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

            # LOGICHE DI COMPATIBILITÀ ASSI E CALCOLO A QUADRIVIO
            if mode in ['sub', 'm', 'res']:
                # Controllo compatibilità geometrica assi X rigidissimo
                if len(df_s) != len(df_2):
                    print(f"   [SALTATO] {sample_full_name} ha lunghezza diversa rispetto alla reference ({len(df_s)} vs {len(df_2)})")
                    continue
                
                check_bins = np.array_equal(df_s['Bin'].values, df_2['Bin'].values)
                check_nm = np.allclose(df_s['Wavelength (nm)'].values, df_2['Wavelength (nm)'].values, atol=1e-4)
                check_cm = np.allclose(df_s['Relative Wavenumber (cm-1)'].values, df_2['Relative Wavenumber (cm-1)'].values, atol=1e-2)
                
                if not (check_bins and check_nm and check_cm):
                    print(f"   [SALTATO] {sample_full_name}: gli assi X non coincidono col riferimento.")
                    continue

                df_output = df_s[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
                
                if mode == 'sub':
                    df_output['Intensity'] = df_s['Intensity'].values - df_2['Intensity'].values
                    suffix = f"_MINUS_{second_name_no_ext}.csv"
                elif mode == 'm':
                    df_output['Intensity'] = df_s['Intensity'].values * df_2['Intensity'].values
                    suffix = f"_TIMES_{second_name_no_ext}.csv"
                elif mode == 'res':
                    # Logica di riscalamento locale
                    somma_s = df_s[mask_ref]['Intensity'].sum()
                    if somma_s == 0:
                        print(f"   [SALTATO] {sample_full_name} ha somma dell'intensità pari a ZERO nella finestra.")
                        continue
                    k_factor = somma_ref / somma_s
                    df_output['Intensity'] = df_s['Intensity'].values * k_factor
                    suffix = f"_RESCALED_TO_{second_name_no_ext}.csv"

            elif mode == 'f':
                # Logica Filtro con ritaglio dinamico e interpolazione SciPy
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

            # 6. SALVATAGGIO AUTOMATICO NELLA STESSA CARTELLA DEL FILE SORGENTE
            output_filename = f"{sample_name_no_ext}{suffix}"
            output_path = os.path.join(dir_name, output_filename)
            df_output.to_csv(output_path, sep=' ', index=False)
            
            if mode == 'res':
                print(f"   -> Creato: {output_filename} (Fattore K: {k_factor:.6f})")
            else:
                print(f"   -> Creato: {output_filename}")

        except Exception as e:
            print(f"   [ERRORE] Impossibile elaborare {sample_full_name}: {e}")

    print("\n=== ELABORAZIONE BATCH COMPLETATA ===")

if __name__ == "__main__":
    main()