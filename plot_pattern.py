import argparse
import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # 1. CONFIGURAZIONE DI ARGPARSE
    parser = argparse.ArgumentParser(
        description="Script avanzato per il plotting di spettri Raman con personalizzazione grafica estrema."
    )

    # File passati tramite flag -s (Obbligatoria)
    parser.add_argument(
        "-s",
        "--files",
        nargs="+",
        required=True,
        help="File o pattern CSV da plottare (es: -s *.csv)",
    )

    # Selezione Asse X
    parser.add_argument(
        "-w",
        "--wavelength",
        action="store_true",
        help="Usa 'Wavelength (nm)' come asse X",
    )
    parser.add_argument(
        "-b", "--bin", action="store_true", help="Usa 'Bin' come asse X"
    )

    # Filtri e Legenda
    parser.add_argument(
        "-r",
        "--exclude-raw",
        action="store_true",
        help="Esclude i file che contengono '-raw' nel nome",
    )
    parser.add_argument(
        "-l", "--legend", action="store_true", help="Mostra la legenda"
    )
    # CORRETTO: default impostato a None per evitare falsi positivi
    parser.add_argument(
        "-ll",
        "--legend-loc",
        type=str,
        default=None,
        help="Posizione della legenda (es: 'upper right', 'top right', 'best'). Default se attivo: 'upper left'",
    )

    # --- FLAGS MODIFICA DATI GRAFICI ---
    parser.add_argument(
        "--tilt",
        type=float,
        nargs="+",
        default=[0.0],
        help="Inclinazione lineare geometrica applicata lungo l'asse X. Accetta un valore per file.",
    )
    parser.add_argument(
        "--offset",
        type=float,
        nargs="+",
        default=[0.0],
        help="Spostamento geometrico della baseline dell'asse Y (traslativo puro). Accetta un valore per file.",
    )

    # --- FLAGS GRAFICHE ---
    parser.add_argument(
        "--title", type=str, default=None, help="Titolo personalizzato del grafico"
    )
    parser.add_argument(
        "--xlabel",
        type=str,
        default=None,
        help="Etichetta personalizzata asse X",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default=None,
        help="Etichetta personalizzata asse Y (singola e nera a sinistra)",
    )

    parser.add_argument(
        "--no-title", action="store_true", help="Disattiva il titolo del grafico"
    )
    parser.add_argument(
        "--no-xlabel",
        action="store_true",
        help="Disattiva l'etichetta dell'asse X",
    )
    parser.add_argument(
        "--no-ylabel",
        action="store_true",
        help="Disattiva l'etichetta dell'asse Y",
    )

    # Dimensioni Font
    parser.add_argument(
        "--fs-title", type=int, default=14, help="Dimensione font del titolo"
    )
    parser.add_argument(
        "--fs-labels", type=int, default=12, help="Dimensione font delle etichette"
    )
    parser.add_argument(
        "--fs-ticks", type=int, default=10, help="Dimensione font dei tick"
    )
    parser.add_argument(
        "--fs-legend", type=int, default=8, help="Dimensione font della legenda"
    )

    # Stile delle Linee e Colori
    parser.add_argument(
        "--cmap", type=str, default="rainbow", help="Colormap Matplotlib"
    )
    parser.add_argument(
        "--color", 
        type=str, 
        nargs="+", 
        default=None, 
        help="Lista di colori personalizzati per ogni file (es: --color red blue)"
    )
    parser.add_argument(
        "--lw", "--linewidth", type=float, default=1.5, help="Spessore linea"
    )
    parser.add_argument(
        "--ls",
        "--linestyle",
        type=str,
        default="-",
        choices=["-", "--", "-.", ":", "None"],
        help="Stile linea",
    )
    parser.add_argument(
        "--marker", type=str, default="None", help="Marker per i punti"
    )

    # Scale, Limiti e Griglia
    parser.add_argument(
        "--scale",
        type=str,
        default="linear",
        choices=["linear", "logx", "logy", "loglog"],
        help="Scala degli assi grafici",
    )
    parser.add_argument(
        "--ticks-x", type=int, default=None, help="Numero di tick asse X"
    )
    parser.add_argument(
        "--ticks-y", type=int, default=None, help="Numero di tick asse Y"
    )
    parser.add_argument(
        "--tight-x", action="store_true", help="Azzera i margini vuoti sull'asse X"
    )
    parser.add_argument(
        "--alignplts",
        action="store_true",
        help="Attiva la modalità multi-asse secondaria accoppiata geometricamente.",
    )
    parser.add_argument(
        "--no-grid", action="store_true", help="Disattiva la griglia di sfondo"
    )

    # Salvataggio
    parser.add_argument(
        "-o", "--output", type=str, default=None, help="Salva su file"
    )

    args = parser.parse_args()

    # 2. SELEZIONE ASSE X
    if args.wavelength:
        x_column = "Wavelength (nm)"
    elif args.bin:
        x_column = "Bin"
    else:
        x_column = "Relative Wavenumber (cm-1)"

    # 3. RACCOLTA FILE
    files_to_plot = []
    for item in args.files:
        item_pulito = item.replace('"', "").strip()
        espanse = glob.glob(item_pulito)
        if not espanse and os.path.isfile(item_pulito):
            espanse = [item_pulito]
        for f in espanse:
            if f.lower().endswith(".csv"):
                if args.exclude_raw and "-raw" in os.path.basename(f).lower():
                    continue
                files_to_plot.append(os.path.abspath(f))

    files_to_plot = list(dict.fromkeys(files_to_plot))

    if not files_to_plot:
        print("[ERRORE] Nessun file CSV valido trovato.")
        sys.exit(1)

    # 4. PREPARAZIONE CANVAS
    fig, base_ax = plt.subplots(figsize=(13, 7))
    axes = []

    try:
        cmap = plt.get_cmap(args.cmap)
    except ValueError:
        cmap = plt.get_cmap("rainbow")

    lines_labels = []

    # 5. CICLO DI PLOTTING DEI DATI ORIGINALI
    for idx, file_path in enumerate(files_to_plot):
        file_name = os.path.basename(file_path)
        label_clean = os.path.splitext(file_name)[0]

        if idx == 0:
            current_ax = base_ax
        elif args.alignplts:
            current_ax = base_ax.twinx()
        else:
            current_ax = base_ax

        axes.append(current_ax)

        try:
            df = pd.read_csv(
                file_path,
                sep=r'\s+(?=(?:[^"]*"[^"]*")*[^"]*$)',
                engine="python",
            )
            df.columns = df.columns.str.replace('"', "").str.strip()

            x_data = df[x_column].to_numpy()
            y_data = df["Intensity"].to_numpy()
            
            if args.color:
                line_color = args.color[min(idx, len(args.color) - 1)]
            else:
                line_color = cmap(idx / max(1, len(files_to_plot) - 1), alpha=0.85)

            ln = current_ax.plot(
                x_data,
                y_data,
                label=label_clean,
                color=line_color,
                linewidth=args.lw,
                linestyle=args.ls,
                marker=args.marker,
            )
            lines_labels.append(ln[0])

            if args.alignplts:
                current_ax.tick_params(axis="y", labelcolor=line_color, colors=line_color)

        except KeyError:
            print(f"   [SALTATO] {file_name} manca di colonne valide.")
        except Exception as e:
            print(f"   [ERRORE] {file_name}: {e}")

    # --- APPLICAZIONE TRASLAZIONI GEOMETRICHE ---
    if args.alignplts:
        for idx, ax_unique in enumerate(axes):
            x_data_plot = lines_labels[idx].get_xdata()
            y_data_plot = lines_labels[idx].get_ydata()
            
            current_offset = args.offset[min(idx, len(args.offset) - 1)]
            current_tilt = args.tilt[min(idx, len(args.tilt) - 1)]
            
            linear_correction = current_offset + (current_tilt * x_data_plot)
            
            y_visual = y_data_plot + linear_correction
            ymin, ymax = y_visual.min(), y_visual.max()
            delta = ymax - ymin if ymax != ymin else 1.0
            
            ax_unique.set_ylim(bottom=ymin - (delta * 0.05), top=ymax + (delta * 0.05))
            ax_unique.plot(x_data_plot, linear_correction, color=lines_labels[idx].get_color(), linestyle=":", alpha=0.3)

    # 6. APPLICAZIONE OPZIONI GRAFICHE E FORMATTAZIONE
    titolo_finale = args.title if args.title else f"Spettri Raman ({x_column} vs Intensity)"
    label_x_finale = args.xlabel if args.xlabel else ("Raman Shift (cm$^{-1}$)" if x_column == "Relative Wavenumber (cm-1)" else x_column)
    label_y_finale = args.ylabel if args.ylabel else "Intensity (counts)"

    if not args.no_title:
        base_ax.set_title(titolo_finale, fontsize=args.fs_title, fontweight="bold")

    if not args.no_xlabel:
        base_ax.set_xlabel(label_x_finale, fontsize=args.fs_labels)

    if not args.no_ylabel:
        base_ax.set_ylabel(label_y_finale, fontsize=args.fs_labels, color="black")

    for idx, ax_unique in enumerate(axes):
        ax_unique.tick_params(axis="both", labelsize=args.fs_ticks)

        if args.scale == "logx":
            ax_unique.set_xscale("log")
        elif args.scale == "logy":
            ax_unique.set_yscale("log")
        elif args.scale == "loglog":
            ax_unique.set_xscale("log")
            ax_unique.set_yscale("log")

        if args.scale in ["linear", "logy"] and args.ticks_x:
            ax_unique.xaxis.set_major_locator(plt.MaxNLocator(args.ticks_x))
        if args.scale in ["linear", "logx"] and args.ticks_y:
            ax_unique.yaxis.set_major_locator(plt.MaxNLocator(args.ticks_y))

        if args.tight_x:
            ax_unique.set_xmargin(0)

        if not args.no_grid and idx == 0:
            ax_unique.grid(True, linestyle="--", alpha=0.5)

    # --- LOGICA CORRETTA DELLA LEGENDA ---
    if args.legend or args.legend_loc is not None:
        # Se l'utente ha usato solo -l, andiamo di fallback su 'upper left'
        loc_input = args.legend_loc.lower().strip() if args.legend_loc else "upper left"
        
        loc_mapping = {
            "top right": "upper right",
            "top left": "upper left",
            "bottom right": "lower right",
            "bottom left": "lower left",
            "top center": "upper center",
            "bottom center": "lower center",
        }
        
        final_loc = loc_mapping.get(loc_input, loc_input)
        labs = [l.get_label() for l in lines_labels]
        
        try:
            base_ax.legend(lines_labels, labs, loc=final_loc, fontsize=args.fs_legend)
        except ValueError:
            base_ax.legend(lines_labels, labs, loc="best", fontsize=args.fs_legend)

    # --- LOGICA DI PADDING MUTUO DEGLI ASSI SECONDARI ---
    if args.alignplts and len(axes) > 1:
        fig.tight_layout()
        plt.draw()
        
        spostamento_cumulativo = 0
        
        for idx in range(1, len(axes)):
            asse_precedente = axes[idx - 1]
            asse_corrente = axes[idx]
            
            if idx == 1:
                spostamento_cumulativo = 0
            else:
                bbox = asse_precedente.yaxis.get_tightbbox(fig.canvas.get_renderer())
                larghezza_punti = bbox.width
                spostamento_cumulativo += larghezza_punti + 45
                
                asse_corrente.spines["right"].set_position(("outward", spostamento_cumulativo))
                asse_corrente.spines["right"].set_visible(True)

        fig.tight_layout()
    else:
        fig.tight_layout()

    # 7. EXPORT
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(args.output, dpi=300)
        print(f"[OK] Grafico salvato correttamente in: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()