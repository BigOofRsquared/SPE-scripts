import sys
import os
import re

if len(sys.argv) < 2:
    print("[ERRORE] Passa il file .spe come argomento.")
    sys.exit(1)

file_path = sys.argv[1]

if not os.path.isfile(file_path):
    print(f"[ERRORE] Il file non esiste: {file_path}")
    sys.exit(1)

try:
    # Leggiamo il file cercando la stringa XML in modalità binaria/tolerant
    with open(file_path, 'rb') as f:
        content = f.read()

    # Cerchiamo il tag <ExposureTime> nell'XML di LightField
    # Il formato tipico è <ExposureTime Units="MilliSeconds">valore</ExposureTime>
    # o <ExposureTime ...>valore</ExposureTime>
    match = re.search(r'<ExposureTime[^>]*>([\d.]+)</ExposureTime>', content.decode('utf-8', errors='ignore'), re.IGNORECASE)

    print("-" * 40)
    print(f"File: {os.path.basename(file_path)}")
    
    if match:
        exposure_time = float(match.group(1))
        
        # Di solito LightField salva l'esposizione in millisecondi nell'XML.
        # Controlliamo se nel tag si parla di millisecondi per fare l'eventuale conversione.
        tag_completo = match.group(0)
        if "millisecond" in tag_completo.lower():
            print(f"Exposure Time: {exposure_time / 1000.0} secondi (convertito da {exposure_time} ms)")
        else:
            print(f"Exposure Time: {exposure_time} secondi")
    else:
        print("[ERRORE] Non ho trovato il tag <ExposureTime> nell'XML di LightField.")
    print("-" * 40)

except Exception as e:
    print(f"[ERRORE] Impossibile analizzare il file: {e}")