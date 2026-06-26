import csv
import os
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("\nUso dello script Heatmap Batch:")
        print(
            "  python heatmap_batch.py -s <file1.csv file2.csv ...> [finestra] <min> <max> [opzioni]"
        )
        print("\nOpzioni Finestra (Sceglierne una, -rwn è il default implicito):")
        print("  -rwn <min> <max>       Intervallo in Relative Wavenumber (cm-1)")
        print("  -w <min> <max>         Intervallo in Wavelength (nm)")
        print("  -b <min> <max>         Intervallo in Bin (pixel)")
        print("\nOpzioni Globali (Opzionali):")
        print(
            "  --raw                  Esclude i file *-raw*.csv dalla generazione della mappa"
        )
        sys.exit(1)

    argomenti = sys.argv[1:]

    file_saltati = {}
    data_points = []

    # 1. GESTIONE FLAG GLOBALI
    exclude_raw = "--raw" in argomenti

    # 2. PARSING DELLA FINESTRA DI SELEZIONE (-rwn, -w, -b)
    window_type = "rwn"  # Default implicito
    w_min, w_max = None, None

    # Cerchiamo se l'utente ha esplicitato una flag di finestra diversa
    for flag in ["-rwn", "-w", "-b"]:
        if flag in argomenti:
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
            break

    # Se non è stata trovata nessuna flag esplicita, cerchiamo i due valori numerici finali assumendo -rwn
    if w_min is None and w_max is None:
        numeri_finali = []
        for arg in reversed(argomenti):
            if arg == "--raw":
                continue
            try:
                numeri_finali.append(float(arg))
                if len(numeri_finali) == 2:
                    break
            except ValueError:
                if numeri_finali:
                    numeri_finali = []
                continue

        if len(numeri_finali) == 2:
            w_max = numeri_finali[0]
            w_min = numeri_finali[1]
            window_type = "rwn"
        else:
            print(
                "[ERRORE] Devi specificare i valori di min e max per la finestra."
            )
            sys.exit(1)

    # Mappiamo il tipo di finestra sul nome effettivo della colonna nel file ripulito
    mappa_colonne = {
        "rwn": "Relative Wavenumber (cm-1)",
        "w": "Wavelength (nm)",
        "b": "Bin",
    }
    target_x_column = mappa_colonne[window_type]

    # 3. PARSING DEI FILE DI SAMPLE (-s) ESPANSI DAL TERMINALE
    files_sample = []
    if "-s" in argomenti:
        idx_s = argomenti.index("-s")
        for arg in argomenti[idx_s + 1:]:
            if arg.startswith("-") or arg.replace(".", "", 1).isdigit():
                # Si ferma se incontra un'altra flag o i numeri della finestra
                break

            if arg.lower().endswith(".csv") and os.path.isfile(arg):
                base_name = os.path.basename(arg)

                if exclude_raw and "-raw" in base_name.lower():
                    file_saltati[base_name] = "Escluso tramite flag --raw"
                    continue

                files_sample.append(os.path.abspath(arg))
    else:
        print("[ERRORE] Manca la flag obbligatoria -s per i file di input.")
        sys.exit(1)

    files_sample = list(dict.fromkeys(files_sample))

    if not files_sample:
        print(
            "[ERRORE] Nessun file valido trovato. Controlla i percorsi o la flag --raw."
        )
        sys.exit(1)

    # Regex per estrarre le prime due coordinate (x,y)
    pattern_coordinate = re.compile(r"\((-?\d+),(-?\d+)")

    print("\n" + "=" * 60)
    print(" AVVIO GENERAZIONE HEATMAP 2D")
    print(f" File totali da analizzare: {len(files_sample)}")
    print(f" Controllo su asse X:       {target_x_column}")
    print(f" Finestra di media:         [{w_min} : {w_max}]")
    print(
        f" Filtro globale --raw:     {'ATTIVO' if exclude_raw else 'DISATTIVO'}"
    )
    print("=" * 60 + "\n")

    # 4. CICLO DI LETTURA ED ESTRAZIONE DATI
    for file_path in files_sample:
        sample_full_name = os.path.basename(file_path)

        match = pattern_coordinate.search(sample_full_name)
        if not match:
            motivo = "Formato nome non valido (Mancano coordinate '(x,y)')"
            file_saltati[sample_full_name] = motivo
            print(f"   [SALTATO] {sample_full_name}: {motivo}")
            continue

        x_coord = int(match.group(1))
        y_coord = int(match.group(2))

        try:
            # Spazia solo fuori dalle virgolette per preservare "Wavelength (nm)" e "Relative Wavenumber (cm-1)"
            df = pd.read_csv(
                file_path,
                sep=r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)',
                engine="python",
            )
            # Pulisce i nomi delle colonne rimuovendo le virgolette e gli spazi esterni
            df.columns = df.columns.str.replace('"', "").str.strip()

            if target_x_column not in df.columns or "Intensity" not in df.columns:
                motivo = f"Mancano le colonne '{target_x_column}' o 'Intensity'"
                file_saltati[sample_full_name] = motivo
                print(f"   [SALTATO] {sample_full_name}: {motivo}")
                continue

            # Applica il filtro numerico sulla colonna scelta
            mask = (df[target_x_column] >= w_min) & (
                df[target_x_column] <= w_max
            )
            avg_intensity = df.loc[mask, "Intensity"].mean()

            if np.isnan(avg_intensity):
                motivo = f"Nessun dato presente nella finestra [{w_min}:{w_max}]"
                file_saltati[sample_full_name] = motivo
                print(f"   [SALTATO] {sample_full_name}: {motivo}")
                continue

            data_points.append(
                {"x": x_coord, "y": y_coord, "val": avg_intensity}
            )
            print(
                f"   [OK] {sample_full_name} -> Spostato in ({x_coord}, {y_coord}) | Valore: {avg_intensity:.4f}"
            )

        except Exception as e:
            motivo = f"Errore di lettura: {e}"
            file_saltati[sample_full_name] = motivo
            print(f"   [ERRORE] {sample_full_name}: {motivo}")

    # 5. RECAP SUI FILE TRATTATI
    print("\n" + "=" * 40)
    print(" RECONCILIATION RECAP")
    print(f" File mappati con successo:  {len(data_points)}")
    print(f" File scartati o saltati:   {len(file_saltati)}")
    if file_saltati:
        print("-" * 40)
        for f, motivo in file_saltati.items():
            print(f"  - {f} -> RAGIONE: {motivo}")
    print("=" * 40 + "\n")

    if not data_points:
        print("[ABORT] Nessun punto dati valido per generare l'heatmap.")
        sys.exit(1)

    # 6. CREAZIONE MATRICE E PLOTTING
    res_df = pd.DataFrame(data_points)
    pivot = res_df.pivot_table(
        index="y", columns="x", values="val", aggfunc="mean"
    )

    plt.figure(figsize=(10, 8))
    im = plt.imshow(
        pivot,
        origin="lower",
        extent=[
            res_df.x.min(),
            res_df.x.max(),
            res_df.y.min(),
            res_df.y.max(),
        ],
        cmap="inferno",
        aspect="auto",
    )

    plt.colorbar(im, label=f"Avg Intensity ({w_min} - {w_max} {window_type})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Heatmap Spaziale 2D ({len(data_points)} punti mappati)")
    plt.grid(False)

    plt.show()


if __name__ == "__main__":
    main()