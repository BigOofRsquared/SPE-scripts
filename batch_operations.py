import csv
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    common = common.rstrip("_- ")
    if len(common) >= 3:
        return common
    return "BUNCH_OF_FILES"


def main():
    if len(sys.argv) < 2:
        print("\nUso dello script Batch Operations (11-in-1):")
        print(
            "  python batch_operations.py -s <pattern_samples> [operazione] <parametro/file> [opzioni]"
        )
        print("\nOperazioni disponibili (sceglierne una):")
        print(
            "  -base <file_back>      Riscalamento locale + Sottrazione automatica su finestra -> OUT: sample_BASE_CORRECTED_BY_back.csv"
        )
        print(
            "  -sub  <file_back>      Sottrazione pura (Asse X completo coordinato) -> OUT: sample_MINUS_back.csv"
        )
        print(
            "  -subb <file_back>      Sottrazione pura BIN-A-BIN (No Interp, ottimo per rumore CCD) -> OUT: sample_MINUSBIN_back.csv"
        )
        print(
            "  -back <file_o_valore>  Sottrazione rumore termico/buio BIN-A-BIN (Accetta un file CSV o un valore costante) -> OUT: sample_THERMAL_SUB_..."
        )
        print(
            "  -m    <file_molt>      Moltiplicazione pura -> OUT: sample_TIMES_molt.csv"
        )
        print(
            "  -f    <file_filtro>    Rimuove filtro con interpolazione -> OUT: sample_FILTERED_BY_filtro.csv"
        )
        print(
            "  -res  <file_rif>       Riscalamento locale puro -> OUT: sample_RESCALED_TO_rif.csv"
        )
        print(
            "  -sm   <valore_scalare> Moltiplicazione per uno scalare -> OUT: sample_TIMES_SCALAR_valore.csv"
        )
        print("  -avg                   Mediazione spettri -> OUT: nomecomune_AVG.csv")
        print(
            "  -norm                  Normalizzazione intensità (Media ad 1.0) -> OUT: NORM_sample.csv"
        )
        print(
            "  -int                   Integrazione dell'intensità su finestra -> Stampa a video Valore Totale e Bin sommati"
        )
        print(
            "\nOpzioni Finestra (Obbligatorie per -res, -base e -int, sceglierne una):"
        )
        print("  -w <min> <max>         Intervallo in Wavelength (nm)")
        print(
            "  -rwn <min> <max>       Intervallo in Relative Wavenumber (cm-1)"
        )
        print("  -b <min> <max>         Intervallo in Bin (pixel)")
        print("\nOpzioni specifiche per -back:")
        print(
            "  -rate <valore>         Counts/sec aggiuntivi per rumore termico (default: 0.0)"
        )
        print("  -t <secondi>           Tempo di esposizione in secondi (default: 1.0)")
        print("\nOpzioni Globali (Opzionali):")
        print(
            "  -xc <nome_colonna>     Specifica il nome della colonna dell'asse X da usare per l'allineamento/fit (default: 'Wavelength (nm)')"
        )
        print(
            "  -o <cartella>          Specifica una cartella di destinazione personalizzata"
        )
        print(
            "  --raw                  Esclude i file *-raw*.csv da QUALSIASI operazione"
        )
        print(
            "  --relax                Interpola l'operatore sull'asse del campione (senza estrapolare) se disallineati [Ignorato in -subb e -back]"
        )
        sys.exit(1)

    argomenti = sys.argv[1:]

    file_saltati = {}
    file_elaborati_count = 0
    medie_totali = []  # <--- Per memorizzare le medie di ogni file in modalita -int

    # 1. GESTIONE FLAG GLOBALI
    exclude_raw = "--raw" in argomenti
    relax_axis = "--relax" in argomenti

    # Parsing della flag -xc (X-Column)
    x_col_name = "Wavelength (nm)"
    if "-xc" in argomenti:
        idx_xc = argomenti.index("-xc")
        if idx_xc + 1 < len(argomenti):
            x_col_name = argomenti[idx_xc + 1].replace('"', "").strip()

    # Parsing dei parametri specifici per -back
    thermal_rate = 0.0
    exposure_time = 1.0
    if "-rate" in argomenti:
        idx_rate = argomenti.index("-rate")
        if idx_rate + 1 < len(argomenti):
            try:
                thermal_rate = float(argomenti[idx_rate + 1])
            except ValueError:
                print(
                    f"[ERRORE] La flag -rate richiede un valore numerico. Ricevuto: '{argomenti[idx_rate + 1]}'"
                )
                sys.exit(1)
    if "-t" in argomenti:
        idx_t = argomenti.index("-t")
        if idx_t + 1 < len(argomenti):
            try:
                exposure_time = float(argomenti[idx_t + 1])
            except ValueError:
                print(
                    f"[ERRORE] La flag -t richiede un valore numerico. Ricevuto: '{argomenti[idx_t + 1]}'"
                )
                sys.exit(1)

    # 2. PARSING DELL'OPERAZIONE E DEL PARAMETRO
    mode = None
    second_param = None

    operazioni_lista = [
        ("-base", "base"),
        ("-sub", "sub"),
        ("-subb", "subb"),
        ("-back", "back"),
        ("-m", "m"),
        ("-f", "f"),
        ("-res", "res"),
        ("-sm", "sm"),
        ("-avg", "avg"),
        ("-norm", "norm"),
        ("-int", "int"),
    ]
    for flag, m_type in operazioni_lista:
        if flag in argomenti:
            if mode is not None:
                print(
                    "[ERRORE] Puoi specificare una sola operazione alla volta tra quelle disponibili."
                )
                sys.exit(1)
            mode = m_type
            idx = argomenti.index(flag)
            if mode not in ["avg", "norm", "int"] and idx + 1 < len(argomenti):
                second_param = argomenti[idx + 1].replace('"', "").strip()

    if not mode:
        print("[ERRORE] Devi specificare un'operazione valida.")
        sys.exit(1)

    if mode not in ["avg", "norm", "int"] and not second_param:
        print(
            f"[ERRORE] L'operazione {mode} richiede un file o un parametro successivo."
        )
        sys.exit(1)

    scalar_value = None
    back_constant_value = None
    is_back_constant = False

    if mode == "sm":
        try:
            scalar_value = float(second_param)
        except ValueError:
            print(
                f"[ERRORE] La modalita -sm richiede un valore numerico valido. Ricevuto: '{second_param}'"
            )
            sys.exit(1)
    elif mode == "back":
        try:
            back_constant_value = float(second_param)
            is_back_constant = True
        except ValueError:
            is_back_constant = False
            if not os.path.isfile(second_param):
                print(
                    f"[ERRORE] Per -back devi specificare un file esistente o un valore numerico costante. Ricevuto: '{second_param}'"
                )
                sys.exit(1)
    elif mode not in ["avg", "norm", "int"]:
        if not os.path.isfile(second_param):
            print(
                f"[ERRORE] Il file operatore specificato non esiste: {second_param}"
            )
            sys.exit(1)

    # 3. PARSING DELLA CARTELLA DI OUTPUT (-o)
    custom_output_dir = None
    if "-o" in argomenti:
        idx_o = argomenti.index("-o")
        if idx_o + 1 < len(argomenti):
            custom_output_dir = argomenti[idx_o + 1].replace('"', "").strip()
            if custom_output_dir:
                os.makedirs(custom_output_dir, exist_ok=True)

    # 4. PARSING DELLA FINESTRA DI INTEGRAZIONE (OBBLIGATORIA PER -res, -base e -int)
    window_type = None
    w_min, w_max = None, None

    for flag in ["-w", "-rwn", "-b"]:
        if flag in argomenti:
            if window_type is not None:
                print(
                    "[ERRORE] Puoi specificare una sola finestra alla volta (-w, -rwn o -b)."
                )
                sys.exit(1)
            window_type = flag.replace("-", "")
            idx = argomenti.index(flag)
            try:
                w_min = float(argomenti[idx + 1])
                w_max = float(argomenti[idx + 2])
            except (IndexError, ValueError):
                print(
                    f"[ERRORE] La flag {flag} richiede due valori numerici (min e max)."
                )
                sys.exit(1)

    if mode in ["res", "base", "int"] and not window_type:
        print(
            f"[ERRORE] La modalita {mode} richiede obbligatoriamente una finestra di ancoraggio (-w, -rwn o -b)."
        )
        sys.exit(1)

    # 5. PARSING DEI FILE DI SAMPLE (-s)
    sample_patterns = []
    if "-s" in argomenti:
        idx_s = argomenti.index("-s")
        for arg in argomenti[idx_s + 1:]:
            stop_conditions = [
                "-base",
                "-sub",
                "-subb",
                "-back",
                "-m",
                "-f",
                "-res",
                "-sm",
                "-avg",
                "-norm",
                "-int",
                "-w",
                "-rwn",
                "-b",
                "-o",
                "-xc",
                "--raw",
                "--relax",
                "-rate",
                "-t",
                custom_output_dir,
            ]
            if arg in stop_conditions:
                break
            sample_patterns.append(arg)
    else:
        print("[ERRORE] Manca la flag obbligatoria -s per i file di campioni.")
        sys.exit(1)

    # Raccolta ed espansione dei file con glob
    files_sample = []
    for item in sample_patterns:
        item_pulito = item.replace('"', "").strip()
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
        for f in espanse:
            if f.lower().endswith(".csv"):
                base_name = os.path.basename(f)

                if exclude_raw and "-raw" in base_name.lower():
                    motivo = "Escluso tramite flag --raw"
                    file_saltati[base_name] = motivo
                    print(f"   [SALTATO] {base_name}: {motivo}")
                    continue
                if mode == "norm" and base_name.startswith("NORM_"):
                    motivo = "Escluso automaticamente (Già normalizzato)"
                    file_saltati[base_name] = motivo
                    print(f"   [SALTATO] {base_name}: {motivo}")
                    continue
                files_sample.append(os.path.abspath(f))

    files_sample = list(dict.fromkeys(files_sample))

    if mode not in ["sm", "avg", "norm", "int"] and not is_back_constant:
        abs_second_file = os.path.abspath(second_param)
        if abs_second_file in files_sample:
            files_sample.remove(abs_second_file)

    if not files_sample and not file_saltati:
        print(
            "[ERRORE] Nessun file di sample valido trovato con il pattern fornito."
        )
        sys.exit(1)

    # 6. CARICAMENTO FILE OPERATORE / RIFERIMENTO
    df_2 = None

    if mode not in ["sm", "avg", "norm", "int"] and not is_back_constant:
        try:
            df_2 = pd.read_csv(second_param, sep=" ", skipinitialspace=True)
            df_2.columns = df_2.columns.str.replace('"', "").str.strip()

            control_col = "Bin" if (mode == "back" and "Bin" in df_2.columns) else x_col_name
            if (mode != "back" and x_col_name not in df_2.columns) or "Intensity" not in df_2.columns:
                print(
                    f"[ABORT] Il file {os.path.basename(second_param)} non contiene le colonne richieste ('{x_col_name}' o 'Bin' e 'Intensity')."
                )
                sys.exit(1)
        except Exception as e:
            print(
                f"[ERRORE] Impossibile leggere il file operatore/riferimento: {e}"
            )
            sys.exit(1)
        second_name_no_ext = os.path.splitext(os.path.basename(second_param))[0]
    elif is_back_constant:
        second_name_no_ext = f"CONST_{second_param}"

    print("\n" + "=" * 60)
    print(" AVVIO ELABORAZIONE BATCH (11-in-1)")
    print(f" Modalita Operativa:   {mode.upper()}")
    print(f" Colonna Asse X scelta: {x_col_name}")
    print(
        f" Filtro globale --raw: {'ATTIVO (cerca *-raw*.csv)' if exclude_raw else 'DISATTIVO'}"
    )
    print(
        f" Gestione Assi X:      {'PURAMENTE GEOMETRICA (Bin-to-Bin)' if mode in ['subb', 'back'] else ('RELAX (Interpolazione)' if relax_axis else 'RIGIDA (Coincidenza esatta)')}"
    )
    if mode == "sm":
        print(f" Scalare impostato:    {scalar_value}")
    elif mode == "back":
        if is_back_constant:
            print(f" Baseline Background:     COSTANTE ({back_constant_value} counts)")
        else:
            print(f" File Background Termico: {os.path.basename(second_param)}")
        print(f" Rate (counts/sec):       {thermal_rate}")
        print(f" Tempo esposizione (t):   {exposure_time} s")
    elif mode in ["avg", "norm", "int"]:
        print(
            f" File totali trovati:  {len(files_sample) + len(file_saltati)}"
        )
    else:
        print(f" File Baseline/Rif:    {os.path.basename(second_param)}")

    if mode in ["res", "base", "int"]:
        print(f" Finestra Ancoraggio:  -{window_type} [{w_min} : {w_max}]")
    if mode != "int":
        print(
            f" Destinazione Output:  {custom_output_dir if custom_output_dir else 'Stessa cartella dei sorgenti'}"
        )
    print("=" * 60 + "\n")

    # PRE-CALCOLI PER INTERPOLAZIONE
    if mode == "f" and not is_back_constant:
        funzione_filtro = interp1d(
            df_2[x_col_name].values,
            df_2["Intensity"].values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
    elif mode in ["sub", "m", 'res', 'base'] and relax_axis and not is_back_constant:
        funzione_operatore = interp1d(
            df_2[x_col_name].values,
            df_2["Intensity"].values,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

    # ----------------------------------------------------
    # RAMO DI LOGICA SPECIFICO PER LA MEDIAZIONE (-avg)
    # ----------------------------------------------------
    if mode == "avg":
        struttura_riferimento = None
        intensita_totale = []
        nomi_file_pure = []

        if len(files_sample) >= 2:
            for idx, file_path in enumerate(files_sample):
                file_name = os.path.basename(file_path)
                nomi_file_pure.append(os.path.splitext(file_name)[0])
                try:
                    df = pd.read_csv(file_path, sep=" ", skipinitialspace=True)
                    df.columns = df.columns.str.replace('"', "").str.strip()

                    if (
                        x_col_name not in df.columns
                        or "Intensity" not in df.columns
                    ):
                        motivo = f"Mancano le colonne '{x_col_name}' o 'Intensity'"
                        file_saltati[file_name] = motivo
                        print(f"   [SALTATO] {file_name}: {motivo}")
                        continue

                    struttura_attuale = df.drop(columns=["Intensity"])

                    if struttura_riferimento is None:
                        struttura_riferimento = struttura_attuale.copy()
                        intensita_totale.append(df["Intensity"].values)
                        file_elaborati_count += 1
                        print(f"   [OK] Riferimento iniziale: {file_name}")
                    else:
                        if len(struttura_attuale) != len(
                            struttura_riferimento
                        ):
                            motivo = f"Lunghezza diversa ({len(struttura_attuale)} vs {len(struttura_riferimento)})"
                            file_saltati[file_name] = motivo
                            print(f"   [SALTATO] {file_name}: {motivo}")
                            continue

                        check_x = np.allclose(
                            struttura_attuale[x_col_name].values,
                            struttura_riferimento[x_col_name].values,
                            atol=1e-4,
                        )

                        if not check_x:
                            motivo = (
                                "Asse X non coincidente con il riferimento"
                            )
                            file_saltati[file_name] = motivo
                            print(f"   [SALTATO] {file_name}: {motivo}")
                            continue

                        intensita_totale.append(df["Intensity"].values)
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
            df_output["Intensity"] = media_intensita

            prefisso_comune = find_common_prefix(nomi_file_pure)
            output_filename = f"{prefisso_comune}_AVG.csv"
            dir_dest = (
                custom_output_dir
                if custom_output_dir
                else os.path.dirname(files_sample[0])
            )
            output_path = os.path.join(dir_dest, output_filename)

            df_output.to_csv(
                output_path,
                sep=" ",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )
            print(
                f"   -> Creato file mediato: {os.path.basename(output_path)}"
            )
        else:
            print(
                "\n[ERRORE] Impossibile calcolare la media: rimasti meno di 2 file validi."
            )

        print("\n" + "=" * 40)
        print(" RECONCILIATION RECAP (-AVG)")
        print(f" File elaborati con successo: {file_elaborati_count}")
        print(f" File saltati:               {len(file_saltati)}")
        if file_saltati:
            print("-" * 40)
            for f, motivo in file_saltati.items():
                print(f"  - {f} -> RAGIONE: {motivo}")
        print("=" * 40 + "\n")
        sys.exit(0)

    # ----------------------------------------------------
    # CICLO DI ELABORAZIONE 1-by-1 FOR ALL OTHER MODES
    # ----------------------------------------------------
    for file_path in files_sample:
        dir_dest = (
            custom_output_dir
            if custom_output_dir
            else os.path.dirname(file_path)
        )
        sample_full_name = os.path.basename(file_path)
        sample_name_no_ext = os.path.splitext(sample_full_name)[0]

        try:
            df_s = pd.read_csv(file_path, sep=" ", skipinitialspace=True)
            df_s.columns = df_s.columns.str.replace('"', "").str.strip()

            if "Intensity" not in df_s.columns or (
                mode != "norm" and x_col_name not in df_s.columns
            ):
                motivo = f"Mancano le colonne necessarie ('{x_col_name}' o 'Intensity')"
                file_saltati[sample_full_name] = motivo
                print(f"   [SALTATO] {sample_full_name}: {motivo}")
                continue

            # --- MODALITÀ: INTEGRAZIONE (-int) ---
            if mode == "int":
                if window_type == "b" and "Bin" in df_s.columns:
                    mask_col_s = df_s["Bin"]
                elif (
                    window_type == "rwn"
                    and "Relative Wavenumber (cm-1)" in df_s.columns
                ):
                    mask_col_s = df_s["Relative Wavenumber (cm-1)"]
                else:
                    mask_col_s = df_s[x_col_name]

                mask_s = (mask_col_s >= w_min) & (mask_col_s <= w_max)

                somma_integrale = df_s[mask_s]["Intensity"].sum()
                conteggio_bin = int(mask_s.sum())

                if conteggio_bin > 0:
                    media_file = somma_integrale / conteggio_bin
                    medie_totali.append(media_file)  # <--- Salva la media per la statistica finale
                else:
                    media_file = 0.0

                file_elaborati_count += 1
                print(
                    f"   [INTEGRALE] {sample_full_name} -> Valore: {somma_integrale:.4e} | Bin sommati: {conteggio_bin} | Media sui bin: {media_file:.6f}"
                )
                continue

            if mode == "norm":
                mean_intensity = df_s["Intensity"].mean()
                if mean_intensity == 0:
                    motivo = "Media intensità pari a ZERO! Divisione impossibile."
                    file_saltati[sample_full_name] = motivo
                    print(f"   [SALTATO] {sample_full_name}: {motivo}")
                    continue
                df_output = df_s.copy()
                df_output["Intensity"] = (
                    df_s["Intensity"].values / mean_intensity
                )
                output_filename = f"NORM_{sample_full_name}"

            elif mode == "sm":
                df_output = df_s.copy()
                df_output["Intensity"] = (
                    df_s["Intensity"].values * scalar_value
                )
                suffix = f"_TIMES_SCALAR_{second_param}.csv"
                output_filename = f"{sample_name_no_ext}{suffix}"

            elif mode in ["sub", "subb", "back", "m", "res", "base"]:
                
                if is_back_constant:
                    check_lunghezza = True
                    assi_coincidenti = True
                    intensita_operatore_interp = np.full(len(df_s), back_constant_value)
                else:
                    check_lunghezza = len(df_s) == len(df_2)
                    intensita_operatore_interp = df_2["Intensity"].values

                if mode in ["subb", "back"] and not is_back_constant:
                    if not check_lunghezza:
                        motivo = f"Incompatibile: numero di pixel CCD differente ({len(df_s)} vs {len(df_2)} del background)"
                        file_saltati[sample_full_name] = motivo
                        print(f"   [SALTATO] {sample_full_name}: {motivo}")
                        continue
                    assi_coincidenti = True
                elif not is_back_constant:
                    check_x = (
                        np.allclose(
                            df_s[x_col_name].values,
                            df_2[x_col_name].values,
                            atol=1e-4,
                        )
                        if check_lunghezza
                        else False
                    )

                    if (
                        "Bin" in df_s.columns
                        and "Bin" in df_2.columns
                        and check_lunghezza
                    ):
                        check_bins = np.array_equal(
                            df_s["Bin"].values, df_2["Bin"].values
                        )
                    else:
                        check_bins = True

                    assi_coincidenti = check_x and check_bins

                    if not assi_coincidenti and not relax_axis:
                        motivo = "Assi X non coincidenti col riferimento (Usa --relax se vuoi forzare l'allineamento)"
                        file_saltati[sample_full_name] = motivo
                        print(f"   [SALTATO] {sample_full_name}: {motivo}")
                        continue

                df_output = df_s.copy()

                if not assi_coincidenti and relax_axis and not is_back_constant:
                    intensita_operatore_interp = funzione_operatore(
                        df_s[x_col_name].values
                    )

                if mode in ["res", "base"]:
                    if (
                        window_type == "b"
                        and "Bin" in df_s.columns
                        and "Bin" in df_2.columns
                    ):
                        mask_col_s, mask_col_ref = df_s["Bin"], df_2["Bin"]
                    else:
                        mask_col_s, mask_col_ref = (
                            df_s[x_col_name],
                            df_2[x_col_name],
                        )

                    mask_s = (mask_col_s >= w_min) & (mask_col_s <= w_max)
                    if not assi_coincidenti and relax_axis:
                        mask_ref = mask_s
                        somma_ref = np.nansum(
                            intensita_operatore_interp[mask_ref]
                        )
                    else:
                        mask_ref = (mask_col_ref >= w_min) & (
                            mask_col_ref <= w_max
                        )
                        somma_ref = df_2[mask_ref]["Intensity"].sum()

                    somma_s = df_s[mask_s]["Intensity"].sum()

                    if somma_s == 0 or somma_ref == 0:
                        motivo = (
                            "Somma intensita pari a ZERO nella finestra scelta"
                        )
                        file_saltati[sample_full_name] = motivo
                        print(f"   [SALTATO] {sample_full_name}: {motivo}")
                        continue

                    k_factor = somma_ref / somma_s

                if mode == "sub":
                    df_output["Intensity"] = (
                        df_s["Intensity"].values - intensita_operatore_interp
                    )
                    suffix = f"_MINUS_{second_name_no_ext}.csv"
                elif mode == "subb":
                    df_output["Intensity"] = (
                        df_s["Intensity"].values - intensita_operatore_interp
                    )
                    suffix = f"_MINUSBIN_{second_name_no_ext}.csv"
                elif mode == "back":
                    profilo_termico = (
                        intensita_operatore_interp
                        + (thermal_rate * exposure_time)
                    )
                    df_output["Intensity"] = (
                        df_s["Intensity"].values - profilo_termico
                    )
                    suffix = f"_THERMAL_SUB_{second_name_no_ext}.csv"
                elif mode == "m":
                    df_output["Intensity"] = (
                        df_s["Intensity"].values * intensita_operatore_interp
                    )
                    suffix = f"_TIMES_{second_name_no_ext}.csv"
                elif mode == "res":
                    df_output["Intensity"] = (
                        df_s["Intensity"].values * k_factor
                    )
                    suffix = f"_RESCALED_TO_{second_name_no_ext}.csv"
                elif mode == "base":
                    baseline_riscalata = intensita_operatore_interp / k_factor
                    df_output["Intensity"] = (
                        df_s["Intensity"].values - baseline_riscalata
                    )
                    suffix = f"_BASE_CORRECTED_BY_{second_name_no_ext}.csv"

                if (
                    not assi_coincidenti
                    and relax_axis
                    and mode not in ["subb", "back"]
                ):
                    df_output = df_output.dropna(subset=["Intensity"])
                    if df_output.empty:
                        motivo = "Spettro azzerato dopo il troncamento: nessuna regione X in comune con l'operatore"
                        file_saltati[sample_full_name] = motivo
                        print(f"   [SALTATO] {sample_full_name}: {motivo}")
                        continue

                output_filename = f"{sample_name_no_ext}{suffix}"

            elif mode == "f":
                intensita_filtro_interpolata = funzione_filtro(
                    df_s[x_col_name].values
                )
                df_output = df_s.copy()
                df_output["Intensity"] = (
                    df_s["Intensity"].values / intensita_filtro_interpolata
                )
                df_output = df_output.dropna(subset=["Intensity"])

                if df_output.empty:
                    motivo = (
                        "Spettro azzerato: nessuna regione in comune con il filtro"
                    )
                    file_saltati[sample_full_name] = motivo
                    print(f"   [SALTATO] {sample_full_name}: {motivo}")
                    continue

                suffix = f"_FILTERED_BY_{second_name_no_ext}.csv"
                output_filename = f"{sample_name_no_ext}{suffix}"

            # Salvataggio
            output_path = os.path.join(dir_dest, output_filename)
            df_output.to_csv(
                output_path,
                sep=" ",
                index=False,
                quoting=csv.QUOTE_NONNUMERIC,
            )

            file_elaborati_count += 1
            info_relax = (
                " [RELAX ASSI COINVOLTO]"
                if (
                    mode in ["sub", "m", "res", "base"]
                    and not assi_coincidenti
                )
                else ""
            )
            if mode == "norm":
                print(
                    f"   [OK] Creato: {output_filename} (Media orig: {mean_intensity:.2f})"
                )
            elif mode == "res":
                print(
                    f"   [OK] Creato: {output_filename} (K-factor: {k_factor:.4f}){info_relax}"
                )
            elif mode == "base":
                print(
                    f"   [OK] Creato: {output_filename} (Adattato su finestra, K-factor: {k_factor:.4f}){info_relax}"
                )
            elif mode == "back":
                tipo_back = f"costante={back_constant_value}" if is_back_constant else "profilo file"
                print(
                    f"   [OK] Creato: {output_filename} (Termico {tipo_back}: rate={thermal_rate}, t={exposure_time}s)"
                )
            else:
                print(f"   [OK] Creato: {output_filename}{info_relax}")

        except Exception as e:
            motivo = f"Errore critico imprevisto: {e}"
            file_saltati[sample_full_name] = motivo
            print(f"   [ERRORE] {sample_full_name}: {motivo}")

    # ----------------------------------------------------
    # RECAP FINALE GLOBALE PER TUTTE LE OPERAZIONI 1-by-1
    # ----------------------------------------------------
    print("\n" + "=" * 40)
    print(f" RECONCILIATION RECAP ({mode.upper()})")
    print(f" File elaborati con successo: {file_elaborati_count}")
    print(f" File saltati:               {len(file_saltati)}")
    
    # Se siamo in modalità integrazione ed abbiamo calcolato almeno una media, stampiamo la statistica sulle medie
    if mode == "int" and medie_totali:
        array_medie = np.array(medie_totali)
        media_delle_medie = np.mean(array_medie)
        std_delle_medie = np.std(array_medie, ddof=1) if len(array_medie) > 1 else 0.0
        
        print("-" * 40)
        print(" ANALISI STATISTICA SULLE MEDIE:")
        print(f"  Media delle Medie:         {media_delle_medie:.6e}")
        print(f"  Deviazione Standard (std): {std_delle_medie:.6e}")

    if file_saltati:
        print("-" * 40)
        for f, motivo in file_saltati.items():
            print(f"  - {f} -> RAGIONE: {motivo}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()