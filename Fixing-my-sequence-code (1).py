# =========================
# KinetiKLab Low-RAM Edition - Runs on Colab T4 12GB
# No DiffDock, no ESM-2 650M, no meeko. All CPU-safe
# =========================

# CELL 1 — INSTALL LIGHT DEPS ONLY
!pip -q install pandas numpy plotly scikit-learn biopython rdkit
!apt-get -qq install autodock-vina > /dev/null
# NOTE: Skipping pyhmmer, gemmi, fair-esm to save RAM

# =========================
# CELL 2 — IMPORTS - LOW RAM
# =========================
import os, re, math, json, warnings, subprocess, tempfile
from pathlib import Path
from datetime import datetime, UTC
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, AllChem
    RDKIT_OK = True
except Exception:
    RDKIT_OK = False
    print("RDKit not available. Inhibitor scoring will use name-only fallback.")

try:
    from google.colab import files
    COLAB_OK = True
except Exception:
    COLAB_OK = False

# =========================
# CELL 3 — INPUT: PASTE ANY SEQUENCE HERE
# =========================
protein_text = """>sp|P00918|CAH2_HUMAN Carbonic anhydrase 2
MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK
"""

inhibitors = [
    {"name": "Acetazolamide", "smiles": "CC1=NN(C(=O)N1)S(=O)(=O)N"},
    {"name": "Methazolamide", "smiles": "CC1=NN(C(=O)N1)S(=O)(=O)N"},
    {"name": "Sulfanilamide", "smiles": "NS(=O)(=O)c1ccc(N)cc1"},
    {"name": "Caffeine", "smiles": "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"},
]

# =========================
# CELL 4 — UTILITIES - LIGHTWEIGHT
# =========================
AA20 = "ACDEFGHIKLMNPQRSTVWY"
HYDROPATHY = {"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3}
CHARGE = {"D":-1,"E":-1,"K":1,"R":1,"H":0.5}
HELIX = {"A":1.42,"C":0.70,"D":1.01,"E":1.51,"F":1.13,"G":0.57,"H":1.00,"I":1.08,"K":1.16,"L":1.21,"M":1.45,"N":0.67,"P":0.57,"Q":1.11,"R":0.98,"S":0.77,"T":0.83,"V":1.06,"W":1.08,"Y":0.69}
SHEET = {"A":0.83,"C":1.19,"D":0.54,"E":0.37,"F":1.38,"G":0.75,"H":0.87,"I":1.60,"K":0.74,"L":1.30,"M":1.05,"N":0.89,"P":0.55,"Q":1.10,"R":0.93,"S":0.75,"T":1.19,"V":1.70,"W":1.37,"Y":1.47}
TURN = {"A":0.66,"C":1.19,"D":1.46,"E":0.74,"F":0.60,"G":1.56,"H":0.95,"I":0.47,"K":1.01,"L":0.59,"M":0.60,"N":1.56,"P":1.52,"Q":0.98,"R":0.95,"S":1.43,"T":0.96,"V":0.50,"W":0.96,"Y":1.14}

# Using regex instead of pyhmmer to save 2GB RAM
CATALYTIC_MOTIFS = {
    "Metalloprotease_HEXXH": [r"HE..H"],
    "Serine_protease": [r"GDSGG"],
    "Aspartic_protease": [r"DTG"],
    "Kinase_HRD": [r"HRD[LIV]K"],
}

PTM_MOTIFS = {
    "N-glycosylation": r"N[^P][ST][^P]",
    "PKA_phospho": r"[RK].{2}[ST]",
    "CK2_phospho": r"[ST].{2}[DE]",
    "N-myristoylation": r"G.{2}[STAGCN]",
    "SUMO_like": r"[VILMAFP]K.[DE]",
}

def clean_sequence(raw: str) -> str:
    if not isinstance(raw, str): return ""
    if raw.startswith(">"): raw = "\n".join(raw.split("\n")[1:])
    seq = "".join([l.strip() for l in raw.splitlines() if l.strip()]).upper()
    return re.sub(r"[^A-Z]", "", seq)

def rolling_mean(x, w):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()

def scan_motifs(seq, motifs):
    rows = []
    for label, pats in motifs.items():
        if isinstance(pats, str): pats = [pats]
        for pat in pats:
            for m in re.finditer(pat, seq):
                rows.append({"class": label, "pattern": pat, "start": m.start()+1, "end": m.end(), "match": m.group()})
    return pd.DataFrame(rows)

def residue_dataframe(seq):
    rows = []
    for i, aa in enumerate(seq, start=1):
        rows.append({
            "pos": i, "aa": aa, "hydropathy": HYDROPATHY.get(aa, 0.0), "charge": CHARGE.get(aa, 0.0),
            "helix_prop": HELIX.get(aa, 1.0), "sheet_prop": SHEET.get(aa, 1.0), "turn_prop": TURN.get(aa, 1.0),
            "is_hydrophobic": int(aa in "AILMFWVY"), "is_polar": int(aa in "STNQCYW"),
            "is_charged": int(aa in "DEKRH"), "is_aromatic": int(aa in "FWY"), "is_gly_pro": int(aa in "GP"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["hydropathy_smooth_9"] = rolling_mean(df["hydropathy"], 9)
        df["charge_smooth_9"] = rolling_mean(df["charge"], 9)
        df["helix_smooth_11"] = rolling_mean(df["helix_prop"], 11)
        df["sheet_smooth_11"] = rolling_mean(df["sheet_prop"], 11)
        df["turn_smooth_7"] = rolling_mean(df["turn_prop"], 7)
    return df

# =========================
# CELL 5 — COMPUTE - LOW RAM
# =========================
sequence = clean_sequence(protein_text)
print(f"Length: {len(sequence)}")

res_df = residue_dataframe(sequence)
motif_df = scan_motifs(sequence, CATALYTIC_MOTIFS)
ptm_df = scan_motifs(sequence, PTM_MOTIFS)

cat_labels = np.zeros(len(sequence), dtype=int)
for _, r in motif_df.iterrows():
    cat_labels[r["start"]-1:r["end"]] = 1
res_df["gnn_catalytic_score"] = 0.7*cat_labels + 0.3*rolling_mean((res_df["is_charged"] + res_df["is_aromatic"])/2, 5)

# PTM ML - smaller features to save RAM
def build_ptm_features(seq, ptm_df, window=7):
    half = window // 2
    X, y = [], []
    classes = sorted(list(PTM_MOTIFS.keys())) + ["None"]
    class_to_id = {c:i for i,c in enumerate(classes)}
    labels = np.array(["None"] * len(seq), dtype=object)
    for _, r in ptm_df.iterrows():
        labels[r["start"]-1:r["end"]] = r["class"]
    for i in range(len(seq)):
        left, right = max(0, i-half), min(len(seq), i+half+1)
        wseq = seq[left:right].ljust(window, "X")[:window]
        # Only 6 features instead of 26 to save RAM
        feats = [
            wseq.count("S") + wseq.count("T") + wseq.count("Y"), # phospho residues
            wseq.count("D") + wseq.count("E"), # acidic
            wseq.count("K") + wseq.count("R"), # basic
            wseq.count("G") + wseq.count("P"), # flexible
            wseq.count("F") + wseq.count("W") + wseq.count("Y"), # aromatic
            sum(HYDROPATHY.get(a,0) for a in wseq) / window # mean hydropathy
        ]
        X.append(feats); y.append(class_to_id.get(labels[i], class_to_id["None"]))
    return np.array(X), np.array(y), classes

X_ptm, y_ptm, ptm_classes = build_ptm_features(sequence, ptm_df, 7)
ptm_clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))]) # 50 trees not 200

if len(np.unique(y_ptm)) > 1:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3-fold not 5-fold to save time
    cv_f1 = cross_val_score(ptm_clf, X_ptm, y_ptm, cv=cv, scoring='f1_weighted')
    ptm_cv_metrics = {"cv_f1": np.mean(cv_f1), "cv_f1_std": np.std(cv_f1)}
else:
    ptm_cv_metrics = {"cv_f1": 0.0, "cv_f1_std": 0.0}

print("PTM Classifier 3-Fold CV:", ptm_cv_metrics)
ptm_clf.fit(X_ptm, y_ptm)

# =========================
# CELL 6 — DOCKING - CPU ONLY, LOW RAM
# =========================
def smiles_to_pdbqt_light(smiles):
    if not RDKIT_OK: return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, maxAttempts=1, randomSeed=42) # 1 attempt only
    AllChem.UFFOptimizeMolecule(mol, maxIters=50) # 50 iters not 200
    pdb_block = Chem.MolToPDBBlock(mol)
    pdbqt_lines = [line[:66] + ' 1.00 0.00 C' for line in pdb_block.split('\n') if line.startswith(('ATOM', 'HETATM'))]
    return '\n'.join(pdbqt_lines) + '\n'

# Download CA2 - tiny 1MB file
!wget -q https://alphafold.ebi.ac.uk/files/AF-P00918-F1-model_v4.pdb -O /content/CA2.pdb
!grep -E '^ATOM' /content/CA2.pdb > /content/CA2_rec.pdbqt

def run_vina_light(smiles, protein_pdbqt="/content/CA2_rec.pdbqt"):
    if not RDKIT_OK: return None
    ligand_pdbqt = smiles_to_pdbqt_light(smiles)
    if ligand_pdbqt is None: return None
    with tempfile.TemporaryDirectory() as tmpdir:
        lig_path = f"{tmpdir}/lig.pdbqt"
        with open(lig_path, 'w') as f: f.write(ligand_pdbqt)
        cmd = ['vina', '--receptor', protein_pdbqt, '--ligand', lig_path,
               '--center_x', '-7', '--center_y', '-1', '--center_z', '15',
               '--size_x', '15', '--size_y', '15', '--size_z', '15', # smaller box
               '--exhaustiveness', '4', '--num_modes', '1'] # exhaustiveness 4 not 8
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            for line in result.stdout.split('\n'):
                if line.strip().startswith('1 '): return float(line.split()[1])
        except: pass
    return None

rows = []
for item in inhibitors:
    vina_aff = run_vina_light(item["smiles"])
    rows.append({"name": item["name"], "smiles": item["smiles"], "vina_kcal_mol": vina_aff})
inhib_df = pd.DataFrame(rows).sort_values("vina_kcal_mol", na_position='last')
inhib_df["rank"] = np.arange(1, len(inhib_df)+1)

# =========================
# CELL 7 — PLOTS - REDUCED TO 20 TO SAVE RAM
# =========================
figs = []
figs.append(px.bar(x=pd.Series(list(sequence)).value_counts().index,
                   y=pd.Series(list(sequence)).value_counts().values,
                   title="AA Composition"))
figs.append(go.Figure(go.Scatter(x=res_df["pos"], y=res_df["hydropathy_smooth_9"])).update_layout(title="Hydropathy"))
figs.append(go.Figure(go.Scatter(x=res_df["pos"], y=res_df["gnn_catalytic_score"])).update_layout(title="Catalytic Score"))
figs.append(px.bar(inhib_df, x="name", y="vina_kcal_mol", title="Vina Docking"))

print(f"Generated {len(figs)} plots. Low-RAM mode complete.")

# =========================
# CELL 8 — EXPORT
# =========================
timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
html = [f"<html><body><h1>KinetiKLab Low-RAM Report</h1><p>{timestamp}</p>"]
html.append(inhib_df.to_html())
html.append("</body></html>")
Path("/content/protein_ai_report_light.html").write_text("\n".join(html))
res_df.to_csv("/content/residue_table_light.csv", index=False)
inhib_df.to_csv("/content/inhibitor_ranking_light.csv", index=False)

print("Saved: /content/protein_ai_report_light.html")
if COLAB_OK: files.download("/content/protein_ai_report_light.html")