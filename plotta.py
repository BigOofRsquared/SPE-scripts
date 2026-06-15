import sys
import os
import matplotlib.pyplot as plt

def plotta_spettri(prefixes):
    # Cerca i file .csv nella cartella corrente che iniziano con ALMENO UNO dei prefissi forniti
    files = []
    for f in os.listdir('.'):
        if f.endswith('.csv') and any(f.startswith(p) for p in prefixes):
            files.append(f)
    
    files = sorted(list(set(files))) # Ordina ed evita duplicati se i prefissi si sovrappongono
    
    if not files:
        print(f"Errore: Nessun file trovato con i prefissi: {', '.join(prefixes)}")
        return

    plt.figure(figsize=(10, 6))

    # Variabile d'appoggio per memorizzare i valori X per il xlim finale
    all_x_vals = []

    for f_name in files:
        x_vals, y_vals = [], []
        with open(f_name, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()[1:] # Salta l'header
            for line in lines:
                # Split sul TAB
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                
                # Usiamo i valori così come sono nel file
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
        
        if x_vals:
            all_x_vals.extend(x_vals)

        # Plotta ogni file. Se è il file _avg, lo mettiamo più in evidenza
        if "_avg" in f_name:
            plt.plot(x_vals, y_vals, label=f_name, linewidth=0.5, zorder=5) # Aumentato un po' il linewidth per l'average
        else:
            plt.plot(x_vals, y_vals, label=f_name, alpha=0.5, linewidth=1)

    plt.xlabel('Wavenumber ($cm^{-1}$)')
    plt.ylabel('Intensity (a.u.)')
    
    # Titolo dinamico in base a quanti prefissi hai inserito
    titolo_prefissi = ", ".join(prefixes) if len(prefixes) <= 3 else f"{len(prefixes)} prefissi"
    plt.title(f'Plot Spettroscopico: {titolo_prefissi}')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Se vuoi essere SICURO che l'asse X vada dal minimo al massimo globale di tutti i file:
    if all_x_vals:
        plt.xlim(left=min(all_x_vals), right=max(all_x_vals)) 

    # Mostra la legenda solo se non ci sono troppi file, altrimenti copre il grafico
    if len(files) < 15:
        plt.legend()
    
    plt.tight_layout()
    print(f"Grafico generato elaborando {len(files)} file. Chiudi la finestra per uscire.")
    plt.show()

if __name__ == "__main__":
    # sys.argv[1:] prende tutti gli argomenti passati dopo il nome del file
    if len(sys.argv) > 1:
        plotta_spettri(sys.argv[1:])
    else:
        # Se non passi argomenti, ti spiega come fare
        print('Uso: python plotta.py "prefisso1" ["prefisso2" "prefisso3" ...]')