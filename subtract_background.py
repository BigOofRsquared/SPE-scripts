import sys
import os
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script di sottrazione:")
        print("  python subtract_pattern.py -s <file_segnale.csv> -b <file_background.csv> -o <file_output.csv>")
        print("\nEsempio:")
        print("  python subtract_pattern.py -s 10V/EXPORT_001.csv -b background_buio.csv -o 10V_sottratto.csv\n")
        sys.exit(1)

    argomenti = sys.argv[1:]

    # 1. PARSING DEGLI ARGOMENTI (Estraiamo i tre file obbligatori)
    file_segnale = None
    file_back = None
    file_output = None

    if '-s' in argomenti:
        idx = argomenti.index('-s')
        if idx + 1 < len(argomenti):
            file_segnale = argomenti[idx + 1].replace('"', '').strip()
    
    if '-b' in argomenti:
        idx = argomenti.index('-b')
        if idx + 1 < len(argomenti):
            file_back = argomenti[idx + 1].replace('"', '').strip()

    if '-o' in argomenti:
        idx = argomenti.index('-o')
        if idx + 1 < len(argomenti):
            file_output = argomenti[idx + 1].replace('"', '').strip()

    # Controlli di esistenza dei parametri
    if not file_segnale or not file_back or not file_output:
        print("[ERRORE] Mancano dei parametri obbligatori!")
        print("Assicurati di usare: -s (segnale) -b (background) -o (output)")
        sys.exit(1)

    if not os.path.isfile(file_segnale):
        print(f"[ERRORE] Il file del segnale non esiste: {file_segnale}")
        sys.exit(1)

    if not os.path.isfile(file_back):
        print(f"[ERRORE] Il file del background non esiste: {file_back}")
        sys.exit(1)

    # Assicuriamoci che l'output finisca con .csv
    if not file_output.lower().endswith('.csv'):
        file_output += '.csv'

    print("\n" + "="*60)
    print(f" AVVIO SOTTRAZIONE SPETTRI")
    print(f" Segnale (Minuendo):   {os.path.basename(file_segnale)}")
    print(f" Background (Sottraendo): {os.path.basename(file_back)}")
    print(f" Output generato:      {file_output}")
    print("="*60 + "\n")

    try:
        # 2. CARICAMENTO DATI
        df_sig = pd.read_csv(file_segnale, sep=' ')
        df_sig.columns = df_sig.columns.str.replace('"', '').str.strip()

        df_bg = pd.read_csv(file_back, sep=' ')
        df_bg.columns = df_bg.columns.str.replace('"', '').str.strip()

        # Verifica colonne standard
        colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
        for col in colonne_obbligatorie:
            if col not in df_sig.columns or col not in df_bg.columns:
                print(f"[ABORT] Uno dei file non contiene la colonna standard: '{col}'")
                sys.exit(1)

        # 3. VERIFICA COINCIDENZA ASSI X
        if len(df_sig) != len(df_bg):
            print(f"[ABORT] I due file hanno lunghezze diverse ({len(df_sig)} punti vs {len(df_bg)} punti). Impossible sottrarre.")
            sys.exit(1)

        check_bins = np.array_equal(df_sig['Bin'].values, df_bg['Bin'].values)
        check_nm = np.allclose(df_sig['Wavelength (nm)'].values, df_bg['Wavelength (nm)'].values, atol=1e-4)
        check_cm = np.allclose(df_sig['Relative Wavenumber (cm-1)'].values, df_bg['Relative Wavenumber (cm-1)'].values, atol=1e-2)

        if not (check_bins and check_nm and check_cm):
            print("[ABORT] Gli assi X (Bin, Wavelength o cm-1) dei due file NON coincidono. Sottrazione bloccata per evitare dati falsati.")
            sys.exit(1)

        # 4. SOTTRAZIONE MATEMATICA E SALVATAGGIO
        # Creiamo il dataframe di output copiando la struttura degli assi del segnale
        df_output = df_sig[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
        
        # Sottrazione pura senza arrotondamenti forzati
        df_output['Intensity'] = df_sig['Intensity'].values - df_bg['Intensity'].values

        # Salva il file risultante
        df_output.to_csv(file_output, sep=' ', index=False)

        print("="*60)
        print(f"[SUCCESSO] Sottrazione completata. Creato: {file_output}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[ERRORE CRITICO] Errore durante l'elaborazione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()