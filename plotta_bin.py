import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Servono: colonna_bin, val_min, val_max + almeno 1 prefisso
    if len(sys.argv) < 5:
        print("Uso: python script.py <colonna_binning> <val_min> <val_max> <prefisso1> [<prefisso2> ...]")
        sys.exit(1)

    # Parametri per il binning
    bin_column = sys.argv[1]
    val_min = float(sys.argv[2])
    val_max = float(sys.argv[3])
    
    # Tutti gli argomenti successivi sono trattati come prefissi
    prefixes = sys.argv[4:]

    print(f"Filtro binning su colonna '{bin_column}' tra {val_min} e {val_max}...")
    print(f"Calcolo media ed errore su colonna 'Intensity'...")
    print(f"Prefissi impostati: {', '.join(prefixes)}")

    # Dizionario per dividere i dati in base al prefisso
    data_by_prefix = {p: [] for p in prefixes}

    # Ciclo sui file della cartella corrente
    for filename in os.listdir('.'):
        if not filename.endswith('.csv'):
            continue
            
        # Verifica se il file appartiene a uno dei prefissi indicati
        for p in prefixes:
            if p in filename:
                
                # Nuova RegEx più robusta:
                # ^(\d+) -> Cattura il numero iniziale (es. 11, 1, 21, 31)
                # .*?    -> Salta il testo in mezzo (inclusa la 's' e il prefisso)
                # (\d+)\.csv$ -> Cattura il numero finale prima di .csv (es. 20 o 19)
                match = re.match(r"^(\d+).*?(\d+)\.csv$", filename)
                
                if match:
                    file_number = int(match.group(1))
                    n_from_file = int(match.group(2))
                    
                    try:
                        # Lettura del file CSV con rilevamento automatico del separatore
                        df = pd.read_csv(filename, sep=None, engine='python')
                        df.columns = df.columns.str.strip()
                        
                        # Controllo presenza colonne necessarie
                        if bin_column not in df.columns or 'Intensity' not in df.columns:
                            print(f"Avviso: Colonne richieste non trovate in {filename}")
                            continue

                        # APPLICAZIONE BINNING: Filtro sulle righe usando la colonna scelta da terminale
                        mask = (df[bin_column] >= val_min) & (df[bin_column] <= val_max)
                        sub_df = df.loc[mask]
                        
                        n_bins = len(sub_df)
                        
                        # CALCOLO SULL'INTENSITÀ: Media dei punti che sono caduti nel bin
                        avg_intensity = sub_df['Intensity'].mean()
                        
                        if not np.isnan(avg_intensity) and n_bins > 0:
                            # Calcolo dell'errore basato sull'intensità media binnata
                            err = np.sqrt(avg_intensity) / np.sqrt(n_bins * n_from_file)
                            
                            data_by_prefix[p].append({
                                'num': file_number, 
                                'val': avg_intensity,
                                'err': err
                            })
                            
                    except Exception as e:
                        print(f"Errore durante la lettura di {filename}: {e}")
                
                # Esci dal ciclo dei prefissi per passare al file successivo
                break 

    # Generazione del grafico cartesiano
    plt.figure(figsize=(10, 6))
    any_data_plotted = False

    for p in prefixes:
        points = data_by_prefix[p]
        if not points:
            print(f"Nessun file valido trovato per il prefisso: '{p}'")
            continue
            
        # Conversione in DataFrame e ordinamento per l'asse X (i secondi: 1, 11, 21, 31...)
        df_p = pd.DataFrame(points).sort_values(by='num')
        
        # Tracciamento della linea con barre di errore
        plt.errorbar(
            df_p['num'], 
            df_p['val'], 
            yerr=df_p['err'], 
            marker='o', 
            linestyle='-', 
            capsize=4,        
            label=f"Gruppo: {p}"
        )
        any_data_plotted = True

    if not any_data_plotted:
        print("Nessun dato estratto. Controlla che i file contengano le colonne corrette.")
        return

    plt.xlabel('Tempo d\'esposizione / Identificativo File (s)')
    plt.ylabel(f'Intensità Media binnata su {bin_column} ({val_min}-{val_max})')
    plt.title(f'Andamento Intensità con Barre d\'Errore Statistico')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    print("Grafico generato con successo.")
    plt.show()

if __name__ == "__main__":
    main()