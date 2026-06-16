import sys
import os
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script di moltiplicazione:")
        print("  python multiply_pattern.py -f1 <file_1.csv> -f2 <file_2.csv> -o <file_output.csv>")
        print("\nEsempio:")
        print("  python multiply_pattern.py -f1 spettro.csv -f2 curva_efficienza.csv -o spettro_calibrato.csv\n")
        sys.exit(1)

    argomenti = sys.argv[1:]

    # 1. PARSING DEGLI ARGOMENTI
    file_1 = None
    file_2 = None
    file_output = None

    if '-f1' in argomenti:
        idx = argomenti.index('-f1')
        if idx + 1 < len(argomenti):
            file_1 = argomenti[idx + 1].replace('"', '').strip()
    
    if '-f2' in argomenti:
        idx = argomenti.index('-f2')
        if idx + 1 < len(argomenti):
            file_2 = argomenti[idx + 1].replace('"', '').strip()

    if '-o' in argomenti:
        idx = argomenti.index('-o')
        if idx + 1 < len(argomenti):
            file_output = argomenti[idx + 1].replace('"', '').strip()

    # Controlli di esistenza dei parametri
    if not file_1 or not file_2 or not file_output:
        print("[ERRORE] Mancano dei parametri obbligatori!")
        print("Assicurati di usare: -f1 (primo file) -f2 (secondo file) -o (output)")
        sys.exit(1)

    if not os.path.isfile(file_1):
        print(f"[ERRORE] Il primo file non esiste: {file_1}")
        sys.exit(1)

    if not os.path.isfile(file_2):
        print(f"[ERRORE] Il secondo file non esiste: {file_2}")
        sys.exit(1)

    # Assicuriamoci che l'output finisca con .csv
    if not file_output.lower().endswith('.csv'):
        file_output += '.csv'

    print("\n" + "="*60)
    print(f" AVVIO MOLTIPLICAZIONE COMPONENT-WISE")
    print(f" File 1:          {os.path.basename(file_1)}")
    print(f" File 2:          {os.path.basename(file_2)}")
    print(f" Output generato: {file_output}")
    print("="*60 + "\n")

    try:
        # 2. CARICAMENTO DATI
        df_1 = pd.read_csv(file_1, sep=' ')
        df_1.columns = df_1.columns.str.replace('"', '').str.strip()

        df_2 = pd.read_csv(file_2, sep=' ')
        df_2.columns = df_2.columns.str.replace('"', '').str.strip()

        # Verifica colonne standard
        colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
        for col in colonne_obbligatorie:
            if col not in df_1.columns or col not in df_2.columns:
                print(f"[ABORT] Uno dei file non contiene la colonna standard: '{col}'")
                sys.exit(1)

        # 3. CONTROLLO COMPATIBILITÀ ASSI X (Blindato)
        if len(df_1) != len(df_2):
            print(f"[ABORT] I file hanno lunghezze diverse ({len(df_1)} punti vs {len(df_2)} punti).")
            sys.exit(1)

        check_bins = np.array_equal(df_1['Bin'].values, df_2['Bin'].values)
        check_nm = np.allclose(df_1['Wavelength (nm)'].values, df_2['Wavelength (nm)'].values, atol=1e-4)
        check_cm = np.allclose(df_1['Relative Wavenumber (cm-1)'].values, df_2['Relative Wavenumber (cm-1)'].values, atol=1e-2)

        if not (check_bins and check_nm and check_cm):
            print("[ABORT] Gli assi X non coincidono. Moltiplicazione bloccata per evitare dati sballati.")
            sys.exit(1)

        # 4. MOLTIPLICAZIONE ELEMENTO PER ELEMENTO
        df_output = df_1[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
        
        # Moltiplicazione pura array NumPy
        df_output['Intensity'] = df_1['Intensity'].values * df_2['Intensity'].values

        # Salvataggio
        df_output.to_csv(file_output, sep=' ', index=False)

        print("="*60)
        print(f"[SUCCESSO] Moltiplicazione completata. Creato: {file_output}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[ERRORE CRITICO] Errore durante l'elaborazione: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()