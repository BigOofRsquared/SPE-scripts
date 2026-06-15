import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print("\nUso dello script:")
        print("  Di default (cm-1):  python plot_pattern.py *.csv")
        print("  Per i Nanometri:    python plot_pattern.py -w *.csv")
        print("  Per i Pixel (Bin):  python plot_pattern.py -b *.csv")
        print("  Mostrare la legenda:python plot_pattern.py -l *.csv")
        print("  Escludere i RAW:    python plot_pattern.py -r *.csv\n")
        sys.exit(1)

    # Prendiamo la lista grezza degli argomenti
    argomenti = sys.argv[1:]

    # 1. GESTIONE FLAG: Ora show_legend diventa True SOLO se passi -l
    exclude_raw = '-r' in argomenti
    show_legend = '-l' in argomenti  # <-- CORRETTO: Attiva SOLO se presente!

    if '-w' in argomenti:
        x_column = 'Wavelength (nm)'
    elif '-b' in argomenti:
        x_column = 'Bin'
    else:
        x_column = 'Relative Wavenumber (cm-1)'

    # 2. PULIZIA ARGOMENTI: Rimuoviamo i flag letterali
    file_args = [arg for arg in argomenti if arg not in ['-w', '-b', '-l', '-r']]

    # 3. RACCOLTA FILE
    files_to_plot = []
    for item in file_args:
        item_pulito = item.replace('"', '').strip()
        
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
            
        for f in espanse:
            if f.lower().endswith('.csv'):
                if exclude_raw and "-raw" in os.path.basename(f).lower():
                    continue
                files_to_plot.append(os.path.abspath(f))

    # Rimuove i duplicati mantenendo l'ordine
    files_to_plot = list(dict.fromkeys(files_to_plot))

    if not files_to_plot:
        print("[ERRORE] Nessun file CSV valido trovato. Controlla il path o i flag.")
        sys.exit(1)

    # Banner di log iniziale
    print("\n" + "="*60)
    print(f" AVVIO PLOT: '{x_column}' (Asse X) vs 'Intensity' (Asse Y)")
    print(f" File totali da elaborare: {len(files_to_plot)}")
    print(f" Esclusione file RAW: {exclude_raw}")
    print(f" Legenda visibile: {show_legend}")
    print("="*60 + "\n")

    cmap = plt.get_cmap('rainbow')
    plt.figure(figsize=(12, 7))

    for idx, file_path in enumerate(files_to_plot):
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, sep=' ')
            df.columns = df.columns.str.replace('"', '').str.strip()
            
            plt.plot(df[x_column], df['Intensity'], label=file_name, color=cmap((idx / len(files_to_plot)), alpha=0.85))
        except KeyError:
            print(f"   [SALTATO] {file_name} non ha la colonna '{x_column}'")
        except Exception as e:
            print(f"   [ERRORE] Errore su {file_name}: {e}")

    # Estetica del grafico
    label_grafico_x = "Raman Shift (cm$^{-1}$)" if x_column == 'Relative Wavenumber (cm-1)' else x_column
    plt.title(f"Spettri Raman ({x_column} vs Intensity)", fontsize=14, fontweight='bold')
    plt.xlabel(label_grafico_x, fontsize=12)
    plt.ylabel("Intensity (counts)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mostra la legenda solo se hai esplicitamente digitato -l
    if show_legend:
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()