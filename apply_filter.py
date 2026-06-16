import sys
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script di filtraggio con interpolazione:")
        print("  python apply_filter.py -s <file_sample.csv> -f <file_filter.csv> -o <file_output.csv>")
        print("\nEsempio:")
        print("  python apply_filter.py -s campione.csv -f curva_filtro.csv -o calibrato.csv\n")
        sys.exit(1)

    argomenti = sys.argv[1:]

    # 1. PARSING DEI PARAMETRI
    file_sample = None
    file_filter = None
    file_output = None

    if '-s' in argomenti:
        idx = argomenti.index('-s')
        if idx + 1 < len(argomenti):
            file_sample = argomenti[idx + 1].replace('"', '').strip()
    
    if '-f' in argomenti:
        idx = argomenti.index('-f')
        if idx + 1 < len(argomenti):
            file_filter = argomenti[idx + 1].replace('"', '').strip()

    if '-o' in argomenti:
        idx = argomenti.index('-o')
        if idx + 1 < len(argomenti):
            file_output = argomenti[idx + 1].replace('"', '').strip()

    # Controlli iniziali
    if not file_sample or not file_filter or not file_output:
        print("[ERRORE] Mancano dei parametri obbligatori! Usa: -s (sample) -f (filter) -o (output)")
        sys.exit(1)

    if not os.path.isfile(file_sample) or not os.path.isfile(file_filter):
        print("[ERRORE] Uno dei file di input non esiste. Controlla i percorsi.")
        sys.exit(1)

    if not file_output.lower().endswith('.csv'):
        file_output += '.csv'

    try:
        # 2. CARICAMENTO DATI
        df_s = pd.read_csv(file_sample, sep=' ')
        df_s.columns = df_s.columns.str.replace('"', '').str.strip()

        df_f = pd.read_csv(file_filter, sep=' ')
        df_f.columns = df_f.columns.str.replace('"', '').str.strip()

        # Verifica colonne standard
        colonne_obbligatorie = ['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)', 'Intensity']
        for col in colonne_obbligatorie:
            if col not in df_s.columns or col not in df_f.columns:
                print(f"[ABORT] Colonne non standard. Entrambi i file devono avere: {colonne_obbligatorie}")
                sys.exit(1)

        print("\n" + "="*60)
        print(f" APVIO APPLICAZIONE FILTRO (INTERPOLATO)")
        print(f" Sample: {os.path.basename(file_sample)} ({len(df_s)} punti)")
        print(f" Filter: {os.path.basename(file_filter)} ({len(df_f)} punti)")

        # 3. DETERMINAZIONE DEL RANGE DI INTERSEZIONE (Sui Nanometri)
        # Troviamo i limiti del filtro
        f_min_nm = df_f['Wavelength (nm)'].min()
        f_max_nm = df_f['Wavelength (nm)'].max()

        # Filtriamo il file sample tenendo SOLO i punti coperti dal filtro
        # Tolleranza minima per evitare problemi di arrotondamento float alle estremità
        df_s_clipped = df_s[
            (df_s['Wavelength (nm)'] >= f_min_nm - 1e-5) & 
            (df_s['Wavelength (nm)'] <= f_max_nm + 1e-5)
        ].copy()

        punti_tagliati = len(df_s) - len(df_s_clipped)
        if punti_tagliati > 0:
            print(f"   [INFO] Tagliati {punti_tagliati} punti dal sample non coperti dal filtro.")

        if df_s_clipped.empty:
            print("[ABORT] Il filtro e il sample non hanno nessuna regione spettrale in comune!")
            sys.exit(1)

        # 4. INTERPOLAZIONE DEL FILTRO SUI PUNTI DEL SAMPLE RIMASTI
        # Creiamo la funzione interpolatrice basandoci sui dati del filtro
        # 'linear' va benissimo, ma se la curva è molto morbida si può usare anche 'cubic'
        funzione_filtro = interp1d(
            df_f['Wavelength (nm)'].values, 
            df_f['Intensity'].values, 
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate" # Sicurezza per micro-arrotondamenti sui bordi
        )

        # Calcoliamo i coefficienti del filtro esattamente sui nanometri del sample tagliato
        nm_sample = df_s_clipped['Wavelength (nm)'].values
        intensita_filtro_interpolata = funzione_filtro(nm_sample)

        # 5. MOLTIPLICAZIONE COMPONENT-WISE E COSTRUZIONE OUTPUT
        # L'output mantiene le colonne (Bin, Wavelength, cm-1) ereditate dal sample tagliato
        df_output = df_s_clipped[['Bin', 'Wavelength (nm)', 'Relative Wavenumber (cm-1)']].copy()
        
        # Moltiplicazione elemento per elemento
        df_output['Intensity'] = df_s_clipped['Intensity'].values * intensita_filtro_interpolata

        # Salvataggio file finale
        df_output.to_csv(file_output, sep=' ', index=False)

        print(f" Output: {file_output} ({len(df_output)} punti finali)")
        print("="*60 + "\n")

    except Exception as e:
        print(f"[ERRORE CRITICO] Errore durante l'elaborazione del filtro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()