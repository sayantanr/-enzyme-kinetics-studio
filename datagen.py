import numpy as np
import pandas as pd

def mm(S, Vmax, Km): return Vmax * S / (Km + S)
def comp(S, I, Vmax, Km, Ki): return Vmax * S / (Km * (1 + I/Ki) + S)

S = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
rows = []
Vmax, Km = 90, 1.2

for n in range(1, 11): # 10 DMSO
    for s in S:
        v = mm(s, Vmax, Km) * np.random.normal(1, 0.03)
        rows.append([f'DMSO_{n:02d}', 'None', 0, s, v])

inhibitors = [('Acetazolamide', 1, 8), ('Acetazolamide', 10, 8),
              ('Brinzolamide', 5, 8), ('Dorzolamide', 2, 8)]

for name, conc, Ki in inhibitors:
    for n in range(1, 11): # 10 each = 40
        for s in S:
            v = comp(s, conc, Vmax, Km, Ki) * np.random.normal(1, 0.03)
            rows.append([f'{name[:3]}_{conc}uM_{n:02d}', name, conc, s, v])

df = pd.DataFrame(rows, columns=['curve_id','inhibitor','inhibitor_conc_uM','substrate_conc_mM','v0_uM_per_min'])
df.to_csv('ca2_50curves.csv', index=False)
print("Created 50 curves, 350 rows")
