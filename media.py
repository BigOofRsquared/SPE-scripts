import sys
import os

def calcola_media(prefix):
    # Trova i file che iniziano col prefisso
    files = sorted([f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.csv')])
    
    if not files:
        print(f"Errore: Nessun file trovato con prefisso '{prefix}'")
        return

    n_files = len(files)
    
    # Strutture dati per accumulare in modo sicuro
    intensities_sum = {}  # Chiave: lunghezza_d'onda (stringa), Valore: somma intensità
    coords_map = {}       # Chiave: lunghezza_d'onda, Valore: (Column, Row)
    wavelength_order = [] # Per mantenere l'ordine originale delle righe nello spettro

    for i, file_path in enumerate(files):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            if not lines: 
                continue
            
            # Saltiamo l'header
            for line in lines[1:]:
                line_clean = line.strip()
                if not line_clean: 
                    continue # Salta righe completamente vuote a fine file
                
                # Split rigoroso sul TAB
                parts = line_clean.split('\t')
                if len(parts) < 2: 
                    continue # Salta righe corrotte o parziali
                
                wl = parts[0]
                try:
                    val = float(parts[1])
                except ValueError:
                    continue # Salta se il valore non è un numero convertibile
                
                if i == 0:
                    # Al primo file inizializziamo i dati e teniamo traccia dell'ordine delle righe
                    wavelength_order.append(wl)
                    intensities_sum[wl] = val
                    
                    c = parts[2] if len(parts) > 2 else "0"
                    r = parts[3] if len(parts) > 3 else "0"
                    coords_map[wl] = (c, r)
                else:
                    # Dai file successivi, sommiamo solo se la lunghezza d'onda esiste nel primo file
                    if wl in intensities_sum:
                        intensities_sum[wl] += val

    # Scrittura file finale identico all'originale
    output_name = f"{prefix}_avg_{n_files}.csv"
    with open(output_name, 'w', encoding='utf-8') as f:
        # Header originale con TAB
        f.write("Wavelength\tIntensity\tColumn\tRow\n")
        
        # Scriviamo i dati seguendo l'ordine esatto del primo file
        for wl in wavelength_order:
            avg_val = intensities_sum[wl] / n_files
            c, r = coords_map[wl]
            f.write(f"{wl}\t{avg_val}\t{c}\t{r}\n")

    print(f"Finito. Media di {n_files} file salvata (formato TAB) in: {output_name}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python media.py <prefisso>")
    else:
        calcola_media(sys.argv[1])