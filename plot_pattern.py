import sys
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 1. CONTROLLO PARAMETRI DI CHIAMATA
    if len(sys.argv) < 2:
        print("\n[ERRORE] Devi passare almeno un file o un pattern da plottare!")
        print("Uso corretto:")
        print("  python plot_spectra.py <file1.csv> <file2.csv>")
        print("  python plot_spectra.py *.csv")
        print("  python plot_spectra.py EXPORT_Quarzo_*.csv\n")
        sys.exit(1)

    # 2. ESPANSIONE DEI CARATTERI JOLLY (*)
    # Prendiamo tutti gli argomenti passati da terminale e li espandiamo con glob
    input_arguments = sys.argv[1:]
    files_to_plot = []
    
    for arg in input_arguments:
        # glob.glob risolve i vari asterischi cercando i file reali sul disco
        matched_files = glob.glob(arg)
        for f in matched_files:
            if os.path.isfile(f) and f.lower().endswith('.csv'):
                files_to_plot.append(f)

    # Rimuoviamo eventuali duplicati mantenendo l'ordine
    files_to_plot = list(dict.fromkeys(files_to_plot))

    if not files_to_plot:
        print("\n[ERRORE] Nessun file CSV valido trovato con i parametri passati.")
        sys.exit(1)

    print(f"Trovati {len(files_to_plot)} file da plottare. Caricamento in corso...")

    # 3. CREAZIONE DEL PLOT
    plt.figure(figsize=(10, 6))

    for file_path in files_to_plot:
        file_name = os.path.basename(file_path)
        try:
            # Leggiamo il file impostando lo SPAZIO come separatore (sep=' ')
            df = pd.read_csv(file_path, sep=' ')
            
            # Verifichiamo che le colonne necessarie esistano
            if 'Relative Wavenumber (cm-1)' in df.columns and 'Intensity' in df.columns:
                x = df['Relative Wavenumber (cm-1)']
                y = df['Intensity']
                
                # Plottiamo lo spettro usando il nome del file come etichetta in legenda
                plt.plot(x, y, label=file_name, alpha=0.8)
            else:
                print(f"   [SALTATO] {file_name} non ha le colonne corrette.")
        except Exception as e:
            print(f"   [ERRORE] Impossibile leggere {file_name}: {e}")

    # 4. CONFIGURAZIONE ESTETICA DEL GRAFICO
    plt.title("Spettri Raman", fontsize=14, fontweight='bold')
    plt.xlabel("Raman Shift (cm$^{-1}$)", fontsize=12)
    plt.ylabel("Intensity (counts)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Mostra la legenda solo se non ci sono troppi file (altrimenti copre tutto)
    if len(files_to_plot) <= 20:
        plt.legend(loc='best', fontsize=9)
    else:
        print("[INFO] Troppi file da visualizzare in legenda (>20), visualizzazione legenda disattivata.")

    plt.tight_layout()
    
    print("Visualizzazione grafico... Chiudi la finestra del grafico per terminare lo script.")
    plt.show()

if __name__ == "__main__":
    main()