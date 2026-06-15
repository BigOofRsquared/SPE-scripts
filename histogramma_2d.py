import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) < 4:
        print("Usage: python script.py <prefix> <lambda_min> <lambda_max>")
        sys.exit(1)

    prefix = sys.argv[1]
    lambda_min = float(sys.argv[2])
    lambda_max = float(sys.argv[3])

    data_points = []
    # Regex per estrarre x e y dal formato prefix(x,y,z,b).csv
    pattern = re.compile(rf"^{re.escape(prefix)}\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)\.csv$")

    print(f"Analisi file con prefisso '{prefix}' tra {lambda_min} e {lambda_max} nm...")

    for filename in os.listdir('.'):
        match = pattern.match(filename)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            
            try:
                # sep=None con engine='python' rileva automaticamente se il separatore è virgola, punto e virgola o tab
                df = pd.read_csv(filename, sep=None, engine='python')
                
                # Pulizia nomi colonne: rimuove spazi bianchi invisibili (es. 'Wavelength ' -> 'Wavelength')
                df.columns = df.columns.str.strip()
                
                if 'Wavelength' not in df.columns or 'Intensity' not in df.columns:
                    print(f"Attenzione: Colonne non trovate in {filename}. Colonne presenti: {list(df.columns)}")
                    continue

                # Filtro e media
                mask = (df['Wavelength'] >= lambda_min) & (df['Wavelength'] <= lambda_max)
                avg_intensity = df.loc[mask, 'Intensity'].mean()
                
                if not np.isnan(avg_intensity):
                    data_points.append({'x': x, 'y': y, 'val': avg_intensity})
            
            except Exception as e:
                print(f"Errore critico su {filename}: {e}")

    if not data_points:
        print("Nessun dato valido trovato. Controlla i nomi delle colonne o il separatore del CSV.")
        return

    # Trasformazione in DataFrame per il plotting
    res_df = pd.DataFrame(data_points)
    
    # Creazione della matrice per il plot (Pivot table)
    # Usiamo un pivot per mappare le coordinate X e Y in una griglia 2D
    pivot = res_df.pivot(index='y', columns='x', values='val')

    plt.figure(figsize=(10, 8))
    # origin='lower' per avere la coordinata (0,0) in basso a sinistra
    im = plt.imshow(pivot, origin='lower', extent=[res_df.x.min(), res_df.x.max(), res_df.y.min(), res_df.y.max()],
                    cmap='viridis', aspect='auto')
    
    plt.colorbar(im, label=f'Avg Intensity ({lambda_min}-{lambda_max} nm)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Heatmap 2D: {prefix}')
    
    print(f"Plot generato con {len(data_points)} punti.")
    plt.show()

if __name__ == "__main__":
    main()