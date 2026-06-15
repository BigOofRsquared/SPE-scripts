import pandas as pd
import matplotlib.pyplot as plt
import argparse
import glob
import re
import os
import sys

def main():
    # 1. Impostiamo gli argomenti da terminale
    parser = argparse.ArgumentParser(description="Somma le intensità in un range di lunghezze d'onda e plotta i risultati.")
    parser.add_argument("prefix", type=str, help="Il prefisso dei file da cercare (es. 'Foo')")
    parser.add_argument("wmin", type=float, help="Wavelength minima")
    parser.add_argument("wmax", type=float, help="Wavelength massima")
    
    args = parser.parse_args()

    # 2. Troviamo tutti i file con quel prefisso nella cartella corrente
    # Aggiungiamo un * per prendere tutto ciò che segue il prefisso e finisce con .csv
    search_pattern = f"{args.prefix}*.csv"
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"Nessun file trovato con il prefisso '{args.prefix}'.")
        sys.exit(1)

    print(f"Trovati {len(files)} file. Inizio l'elaborazione...")

    # Lista dove salveremo le tuple (numero_file, somma_intensità)
    data_points = []

    # Regex per estrarre il primo numero dentro le parentesi. 
    # Cerca una "(", poi cattura tutto fino alla prima virgola.
    regex_pattern = r'\(([^,]+)'

    # 3. Iteriamo sui file
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # Estraiamo il numero dal nome del file
        match = re.search(regex_pattern, filename)
        if match:
            try:
                # Convertiamo la stringa catturata in float
                file_num = float(match.group(1).strip())
            except ValueError:
                print(f"Salto {filename}: impossibile convertire in numero il valore estratto.")
                continue
        else:
            print(f"Salto {filename}: nessuna parentesi con virgole trovata.")
            continue

        # 4. Apriamo il file ed estrapoliamo i dati
        # sep=r'\s+' gestisce sia spazi che tabulazioni come separatori
        try:
            df = pd.read_csv(filepath, sep=r'\s+')
            
            # Filtriamo il dataframe tra wmin e wmax
            mask = (df['Wavelength'] >= args.wmin) & (df['Wavelength'] <= args.wmax)
            
            # Sommiamo l'intensità per le righe filtrate
            intensity_sum = df.loc[mask, 'Intensity'].sum()
            
            # Aggiungiamo i dati alla nostra lista
            data_points.append((file_num, intensity_sum))
            
        except Exception as e:
            print(f"Errore nella lettura di {filename}: {e}")

    if not data_points:
        print("Nessun dato valido estratto. Termino il programma.")
        sys.exit(1)

    # 5. Ordiniamo i punti in base al "numero file" (x) per avere un plot pulito
    data_points.sort(key=lambda x: x[0])

    # Separiamo le X e le Y
    x = [p[0] for p in data_points]
    y = [p[1] for p in data_points]

    # 6. Plottiamo
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label=f'Wavelength [{args.wmin} - {args.wmax}]')
    
    plt.title(f"Somma Intensity vs File Number (Prefisso: {args.prefix})")
    plt.xlabel("File Number (Estratto dal nome file)")
    plt.ylabel("Somma Intensity")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()