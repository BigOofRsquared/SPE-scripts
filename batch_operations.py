import sys
import os
import glob
import csv
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def find_common_prefix(strings):
    """Trova la massima stringa comune iniziale o parziale tra i nomi dei file."""
    if not strings:
        return ""
    s1 = strings[0]
    common = ""
    for i in range(1, len(s1) + 1):
        candidate = s1[:i]
        if all(s.startswith(candidate) for s in strings):
            common = candidate
        else:
            break
    common = common.rstrip('_- ')
    if len(common) >= 3:
        return common
    return "BUNCH_OF_FILES"

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script Batch Operations (8-in-1):")
        print("  python batch_operations.py -s <pattern_samples> [operazione] <parametro/file> [opzioni]")
        print("\nOperazioni disponibili (sceglierne una):")
        print("  -base <file_back>      Riscalamento locale + Sottrazione automatica su finestra -> OUT: sample_BASE_CORRECTED_BY_back.csv")
        print("  -sub  <file_back>      Sottrazione pura -> OUT: sample_MINUS_back.csv")
        print("  -m    <file_molt>      Moltiplicazione pura -> OUT: sample_TIMES_molt.csv")
        print("  -f    <file_filtro>    Filtro con interpolazione -> OUT: sample_FILTERED_BY_filtro.csv")
        print("  -res  <file_rif>       Riscalamento locale puro -> OUT: sample_RESCALED_TO_rif.csv")
        print("  -sm   <valore_scalare> Moltiplicazione per uno scalare -> OUT: sample_TIMES_SCALAR_valore.csv")
        print("  -avg                   Mediazione spettri -> OUT: nomecomune_AVG.csv")
        print("  -norm                  Normalizzazione intensità (Media ad 1.0) -> OUT: NORM_sample.csv")
        print("\nOpzioni Finestra (Obbligatorie per -res e -base, sceglierne una):")
        print("  -w <min> <max>         Intervallo in Wavelength (nm)")
        print("  -rwn <min> <max>       Intervallo in Relative Wavenumber (cm-1)")
        print("  -b <min> <max>         Intervallo in Bin (pixel)")
        print("\nOpzioni Globali (Opzionali):")
        print("  -o <cartella>          Specifica una cartella di destinazione personalizzata")
        print("  --raw                  Esclude i file *-raw*.csv da QUALSIASI operazione")
        print("  --relax                Interpola l'operatore sull'asse del campione (senza estrapolare) se disallineati")
        sys.exit(1)

    argomenti = sys.argv[1:]

    file_saltati = {}
    file_elaborati_count = 0

    # 1. GESTIONE FLAG GLOBALI
    exclude_raw = '--raw' in argomenti
    relax_axis = '--relax' in argomenti

    # 2. PARSING DELL'OPERAZIONE E DEL PARAMETRO
    mode = None  
    second_param = None

    operazioni_lista = [
        ('-base', 'base'), ('-sub', 'sub'), ('-m', 'm'), 
        ('-f', 'f'), ('-res', 'res'), ('-sm', 'sm'), 
        ('-avg', 'avg'), ('-norm', 'norm')
    ]
    for flag, m_type in operazioni_lista:
        if flag in argomenti:
            if mode is not None:
                print("[ERRORE] Puoi specificare una sola operazione alla volta tra quelle disponibili.")
                sys.exit(1)
            mode = m_type
            idx = argomenti.index(flag)
            if mode not in ['avg', 'norm'] and idx + 1 < len(argomenti):
                second_param = argomenti[idx + 1].replace('"', '').strip()

    if not mode:
        print("[ERRORE] Devi specificare un'operazione valida.")
        sys.exit(1)

    if mode not in ['avg', 'norm'] and not second_param:
        print(f"[ERRORE] L'operazione {mode} richiede un file o un parametro successivo.")
        sys.exit(1)

    scalar_value = None
    if mode == 'sm':
        try:
            scalar_value = float(second_param)
        except ValueError:
            print(f"[ERRORE] La modalita -sm richiede un valore numerico valido. Ricevuto: '{second_param}'")
            sys.exit(1)
    elif mode != 'avg' and mode != 'norm':
        if not os.path.isfile(second_param):
            print(f"[ERRORE] Il file operatore specificato non esiste: {second_param}")
            sys.exit(1)

    # 3. PARSING DELLA CARTELLA DI OUTPUT (-o)
    custom_output_dir = None
    if '-o' in argomenti:
        idx_o = argomenti.index('-o')
        if idx_o + 1 < len(argomenti):
            custom_output_dir = argomenti[idx_o + 1].replace('"', '').strip()
            if custom_output_dir:
                os.makedirs(custom_output_dir, exist_ok=True)

    # 4. PARSING DELLA FINESTRA DI INTEGRAZIONE (OBBLIGATORIA PER -res E -base)
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

    if mode in ['res', 'base'] and not window_type:
        print(f"[ERRORE] La modalita {flag} richiede obbligatoriamente una finestra di ancoraggio (-w, -rwn o -b).")
        sys.exit(1)

    # 5. PARSING DEI FILE DI SAMPLE (-s)
    sample_patterns = []
    if '-s' in argomenti:
        idx_s = argomenti.index('-s')
        for arg in argomenti[idx_s + 1:]:
            stop_conditions = ['-base', '-sub', '-m', '-f', '-res', '-sm', '-avg', '-norm', '-w', '-rwn', '-b', '-o', '--raw', '--relax', custom_output_dir]
            if second_param:
                stop_conditions.append(second_param)
            if arg in stop_conditions:
                break
            sample_patterns.append(arg)
    else:
        print("[ERRORE] Manca la flag obbligatoria -s per i file di campioni.")
        sys.exit(1)

    # Raccolta ed espansione dei file con glob
    files_sample = []
    for item in sample_patterns:
        item_pulito = item.replace('"', '').strip()
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
        for f in espanse:
            if f.lower().endswith('.csv'):
                base_name = os.path.basename(f)
                
                if exclude_raw and "-raw" in base_name.lower():
                    motivo = "Escluso tramite flag --raw"
                    file_saltati[base_name] = motivo
                    print(f"   [SALTATO] {base_name}: {motivo}")
                    continue
                if mode == 'norm' and base_name.startswith("NORM_"):
                    motivo = "Escluso automaticamente (Già normalizzato)"
                    file_saltati[base_name] = motivo
                    print(f"   [SALTATO] {base_name}: {motivo}")
                    continue
                files_sample.append(os.path.abspath(f))

    files_sample = list(dict.fromkeys(files_sample))

    if mode not in ['sm', 'avg', 'norm']:
        abs_second_file = os.path.abspath(second_param)
        if abs_second_file in files_sample:
            files_sample.remove(abs_second_file)

    if not files_sample and not file_saltati:
        print("[ERRORE] Nessun file di sample valido trovato con il pattern fornito.")
        sys.exit(1)

    # 6. CARICAMENTO FILE OPERATORE / RIFERIMENTO
    colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
    df_2 = None
    
    if mode not in ['sm', 'avg', 'norm']:
        try:
            df_2 = pd.read_csv(second_param, sep=' ', skipinitialspace=True)
            df_2.columns = df_2.columns.str.replace('"', '').str.strip()
            if not all(col in df_2.columns for col in colonne_obbligatorie):
                print(f"[ABORT] Il file {os.path.basename(second_param)} non ha le colonne standard.")
                sys.exit(1)
        except Exception as e:
            print(f"[ERRORE] Impossibile leggere il file operatore/riferimento: {e}")
            sys.exit(1)
        second_name_no_ext = os.path.splitext(os.path.basename(second_param))[0]

    print("\n" + "="*60)
    print(f" AVVIO ELABORAZIONE BATCH (8-in-1)")
    print(f" Modalita Operativa:   {mode.upper()}")
    print(f" Filtro globale --raw: {'ATTIVO (cerca *-raw*.csv)' if exclude_raw else 'DISATTIVO'}")
    print(f" Gestione Assi X:      {'RELAX (Interpolazione senza estrapolazione)' if relax_axis else 'RIGIDA (Coincidenza esatta)'}")
    if mode == 'sm':
        print(f" Scalare impostato:    {scalar_value}")
    elif mode in ['avg', 'norm']:
        print(f" File totali trovati:  {len(files_sample) + len(file_saltati)}")
    else:
        print(f" File Baseline/Rif:    {os.path.basename(second_param)}")
        
    if mode in ['res', 'base']:
        print(f" Finestra Ancoraggio:  -{window_type} [{w_min} : {w_max}]")
    print(f" Destinazione Output:  {custom_output_dir if custom_output_dir else 'Stessa cartella dei sorgenti'}")
    print("="*60 + "\n")

    # PRE-CALCOLI PER INTERPOLAZIONE
    if mode == 'f':
        funzione_filtro = interp1d(df_2['Wavelength (nm)'].values, df_2['Intensity'].values, kind='linear', bounds_error=False, fill_value=np.nan)
    elif mode in ['sub', 'm', 'res', 'base'] and relax_axis:
        funzione_operatore = interp1d(df_2['Wavelength (nm)'].values, df_2['Intensity'].values, kind='linear', bounds_error=False, fill_value=np.nan)

    # ----------------------------------------------------
    # RAMO DI LOGICA SPECIFICO PER LA MEDIAZIONE (-avg)
    # ----------------------------------------------------
    if mode == 'avg':
        struttura_riferimento = None
        intensita_totale = []
        nomi_file_pure = []

        if len(files_sample) >= 2:
            for idx, file_path in enumerate(files_sample):
                file_name = os.path.basename(file_path)
                nomi_file_pure.append(os.path.splitext(file_name)[0])
                try:
                    df = pd.read_csv(file_path, sep=' ', skipinitialspace=True)
                    df.columns = df.columns.str.replace('"', '').str.strip()
                    
                    if not all(col in df.columns for col in colonne_obbligatorie):
                        motivo = "Mancano colonne standard"
                        file_saltati[file_name] = motivo
                        print(f"   [SALTATO] {file_name}: {motivo}")
                        continue
                    
                    struttura_attuale = df[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']]
                    
                    if struttura_riferimento is None:
                        struttura_riferimento = struttura_attuale.copy()
                        intensita_totale.append(df['Intensity'].values)
                        file_elaborati_count += 1
                        print(f"   [OK] Riferimento iniziale: {file_name}")
                    else:
                        if len(struttura_attuale) != len(struttura_riferimento):
                            motivo = f"Lunghezza diversa ({len(struttura_attuale)} vs {len(struttura_riferimento)})"
                            file_saltati[file_name] = motivo
                            print(f"   [SALTATO] {file_name}: {motivo}")
                            continue
                        
                        check_bins = np.array_equal(struttura_attuale['Bin'].values, struttura_riferimento['Bin'].values)
                        check_nm = np.allclose(struttura_attuale['Wavelength (nm)'].values, struttura_riferimento['Wavelength (nm)'].values, atol=1e-4)
                        check_cm = np.allclose(struttura_attuale['Relative Wavenumber (cm-1)'].values, struttura_riferimento['Relative Wavenumber (cm-1)'].values, atol=1e-2)
                        
                        if not (check_bins and check_nm and check_cm):
                            motivo = "Assi X non coincidenti con il riferimento"
                            file_saltati[file_name] = motivo
                            print(f"   [SALTATO] {file_name}: {motivo}")
                            continue
                        
                        intensita_totale.append(df['Intensity'].values)
                        file_elaborati_count += 1
                        print(f"   [OK] Aggiunto alla media: {file_name}")
                except Exception as e:
                    motivo = f"Errore di lettura: {e}"
                    file_saltati[file_name] = motivo
                    print(f"   [ERRORE] {file_name}: {motivo}")

        if file_elaborati_count >= 2:
            print("\nCalcolo della media aritmetica...")
            media_intensita = np.mean(np.array(intensita_totale), axis=0)
            df_output = struttura_riferimento.copy()
            df_output['Intensity'] = media_intensita

            prefisso_comune = find_common_prefix(nomi_file_pure)
            output_filename = f"{prefisso_comune}_AVG.csv"
            dir_dest = custom_output_dir if custom_output_dir else os.path.dirname(files_sample[0])
            output_path = os.path.join(dir_dest, output_filename)
            
            df_output.to_csv(output_path, sep=' ', index=False, quoting=csv.QUOTE_NONNUMERIC)
            print(f"   -> Creato file mediato: {os.path.basename(output_path)}")
        else:
            print("\n[ERRORE] Impossibile calcolare la media: rimasti meno di 2 file validi.")

        print("\n" + "="*40)
        print(f" RECONCILIATION RECAP (-AVG)")
        print(f" File elaborati con successo: {file_elaborati_count}")
        print(f" File saltati:               {len(file_saltati)}")
        if file_saltati:
            print("-"*40)
            for f, motivo in file_saltati.items():
                print(f"  - {f} -> RAGIONE: {motivo}")
        print("="*40 + "\n")
        sys.exit(0)

    # ----------------------------------------------------
    # CICLO DI ELABORAZIONE 1-by-1 FOR ALL OTHER MODES
    # ----------------------------------------------------
    for file_path in files_sample:
        dir_dest = custom_output_dir if custom_output_dir else os.path.dirname(file_path)
        sample_full_name = os.path.basename(file_path)
        sample_name_no_ext = os.path.splitext(sample_full_name)[0]
        
        try:
            df_s = pd.read_csv(file_path, sep=' ', skipinitialspace=True)
            df_s.columns = df_s.columns.str.replace('"', '').str.strip()
            
            if 'Intensity' not in df_s.columns:
                motivo = "Manca la colonna 'Intensity'"
                file_saltati[sample_full_name] = motivo
                print(f"   [SALTATO] {sample_full_name}: {motivo}")
                continue
                
            if mode != 'norm' and not all(col in df_s.columns for col in colonne_obbligatorie):
                motivo = "Mancano colonne Raman standard"
                file_saltati[sample_full_name] = motivo
                print(f"   [SALTATO] {sample_full_name}: {motivo}")
                continue

            if mode == 'norm':
                mean_intensity = df_s['Intensity'].mean()
                if mean_intensity == 0:
                    motivo = "Media intensità pari a ZERO! Divisione impossibile."
                    file_saltati[sample_full_name] = motivo
                    print(f"   [SALTATO] {sample_full_name}: {motivo}")
                    continue
                df_output = df_s.copy()
                df_output['Intensity'] = df_s['Intensity'].values / mean_intensity
                output_filename = f"NORM_{sample_full_name}"

            elif mode == 'sm':
                df_output = df_s.copy()
                df_output['Intensity'] = df_s['Intensity'].values * scalar_value
                suffix = f"_TIMES_SCALAR_{second_param}.csv"
                output_filename = f"{sample_name_no_ext}{suffix}"
                
            elif mode in ['sub', 'm', 'res', 'base']:
                check_lunghezza = len(df_s) == len(df_2)
                check_bins = np.array_equal(df_s['Bin'].values, df_2['Bin'].values) if check_lunghezza else False
                check_nm = np.allclose(df_s['Wavelength (nm)'].values, df_2['Wavelength (nm)'].values, atol=1e-4) if check_lunghezza else False
                check_cm = np.allclose(df_s['Relative Wavenumber (cm-1)'].values, df_2['Relative Wavenumber (cm-1)'].values, atol=1e-2) if check_lunghezza else False
                
                assi_coincidenti = check_bins and check_nm and check_cm

                if not assi_coincidenti and not relax_axis:
                    motivo = "Assi X non coincidenti col riferimento (Usa --relax se vuoi forzare l'allineamento)"
                    file_saltati[sample_full_name] = motivo
                    print(f"   [SALTATO] {sample_full_name}: {motivo}")
                    continue

                df_output = df_s[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
                
                if not assi_coincidenti and relax_axis:
                    intensita_operatore_interp = funzione_operatore(df_s['Wavelength (nm)'].values)
                else:
                    intensita_operatore_interp = df_2['Intensity'].values

                if mode in ['res', 'base']:
                    if window_type == 'w':
                        mask_s = (df_s['Wavelength (nm)'] >= w_min) & (df_s['Wavelength (nm)'] <= w_max)
                        if not assi_coincidenti and relax_axis:
                            mask_ref = mask_s
                            somma_ref = np.nansum(intensita_operatore_interp[mask_ref])
                        else:
                            mask_ref = (df_2['Wavelength (nm)'] >= w_min) & (df_2['Wavelength (nm)'] <= w_max)
                            somma_ref = df_2[mask_ref]['Intensity'].sum()
                    elif window_type == 'rwn':
                        mask_s = (df_s['Relative Wavenumber (cm-1)'] >= w_min) & (df_s['Relative Wavenumber (cm-1)'] <= w_max)
                        if not assi_coincidenti and relax_axis:
                            mask_ref = mask_s
                            somma_ref = np.nansum(intensita_operatore_interp[mask_ref])
                        else:
                            mask_ref = (df_2['Relative Wavenumber (cm-1)'] >= w_min) & (df_2['Relative Wavenumber (cm-1)'] <= w_max)
                            somma_ref = df_2[mask_ref]['Intensity'].sum()
                    else: # Bin
                        mask_s = (df_s['Bin'] >= w_min) & (df_s['Bin'] <= w_max)
                        if not assi_coincidenti and relax_axis:
                            mask_ref = mask_s
                            somma_ref = np.nansum(intensita_operatore_interp[mask_ref])
                        else:
                            mask_ref = (df_2['Bin'] >= w_min) & (df_2['Bin'] <= w_max)
                            somma_ref = df_2[mask_ref]['Intensity'].sum()
                    
                    somma_s = df_s[mask_s]['Intensity'].sum()
                    
                    if somma_s == 0 or somma_ref == 0:
                        motivo = "Somma intensita pari a ZERO nella finestra scelta"
                        file_saltati[sample_full_name] = motivo
                        print(f"   [SALTATO] {sample_full_name}: {motivo}")
                        continue
                        
                    k_factor = somma_ref / somma_s
                
                if mode == 'sub':
                    df_output['Intensity'] = df_s['Intensity'].values - intensita_operatore_interp
                    suffix = f"_MINUS_{second_name_no_ext}.csv"
                elif mode == 'm':
                    df_output['Intensity'] = df_s['Intensity'].values * intensita_operatore_interp
                    suffix = f"_TIMES_{second_name_no_ext}.csv"
                elif mode == 'res':
                    df_output['Intensity'] = df_s['Intensity'].values * k_factor
                    suffix = f"_RESCALED_TO_{second_name_no_ext}.csv"
                elif mode == 'base':
                    baseline_riscalata = intensita_operatore_interp / k_factor
                    df_output['Intensity'] = df_s['Intensity'].values - baseline_riscalata
                    # AGGIORNATO: cita espressamente il file operatore nel nome output
                    suffix = f"_BASE_CORRECTED_BY_{second_name_no_ext}.csv"

                if not assi_coincidenti and relax_axis:
                    df_output = df_output.dropna(subset=['Intensity'])
                    if df_output.empty:
                        motivo = "Spettro azzerato dopo il troncamento: nessuna regione X in comune con l'operatore"
                        file_saltati[sample_full_name] = motivo
                        print(f"   [SALTATO] {sample_full_name}: {motivo}")
                        continue
                
                output_filename = f"{sample_name_no_ext}{suffix}"

            elif mode == 'f':
                intensita_filtro_interpolata = function_filtro(df_s['Wavelength (nm)'].values) if 'function_filtro' in locals() else funzione_filtro(df_s['Wavelength (nm)'].values)
                df_output = df_s[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
                df_output['Intensity'] = df_s['Intensity'].values * intensita_filtro_interpolata
                df_output = df_output.dropna(subset=['Intensity'])
                
                if df_output.empty:
                    motivo = "Spettro azzerato: nessuna regione in comune con il filtro"
                    file_saltati[sample_full_name] = motivo
                    print(f"   [SALTATO] {sample_full_name}: {motivo}")
                    continue
                    
                suffix = f"_FILTERED_BY_{second_name_no_ext}.csv"
                output_filename = f"{sample_name_no_ext}{suffix}"

            # Salvataggio
            output_path = os.path.join(dir_dest, output_filename)
            df_output.to_csv(output_path, sep=' ', index=False, quoting=csv.QUOTE_NONNUMERIC)
            
            file_elaborati_count += 1
            info_relax = " [RELAX ASSI COINVOLTO]" if (mode in ['sub', 'm', 'res', 'base'] and not assi_coincidenti) else ""
            if mode == 'norm':
                print(f"   [OK] Creato: {output_filename} (Media orig: {mean_intensity:.2f})")
            elif mode == 'res':
                print(f"   [OK] Creato: {output_filename} (K-factor: {k_factor:.4f}){info_relax}")
            elif mode == 'base':
                print(f"   [OK] Creato: {output_filename} (Adattato su finestra, K-factor: {k_factor:.4f}){info_relax}")
            else:
                print(f"   [OK] Creato: {output_filename}{info_relax}")

        except Exception as e:
            motivo = f"Errore critico imprevisto: {e}"
            file_saltati[sample_full_name] = motivo
            print(f"   [ERRORE] {sample_full_name}: {motivo}")

    # ----------------------------------------------------
    # RECAP FINALE GLOBALE PER TUTTE LE OPERAZIONI 1-by-1
    # ----------------------------------------------------
    print("\n" + "="*40)
    print(f" RECONCILIATION RECAP ({mode.upper()})")
    print(f" File elaborati con successo: {file_elaborati_count}")
    print(f" File saltati:               {len(file_saltati)}")
    if file_saltati:
        print("-"*40)
        for f, motivo in file_saltati.items():
            print(f"  - {f} -> RAGIONE: {motivo}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()