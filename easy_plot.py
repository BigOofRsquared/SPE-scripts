import matplotlib.pyplot as plt
import pandas as pd

# Metti qui il nome del tuo file
file = "T_vs_R_rel.dat" 

df = pd.read_csv(file, sep="\s+")
plt.plot(df.iloc[:, 0], df.iloc[:, 1], '.-') # '.-' mette punti e linee
plt.grid(True)
plt.show()