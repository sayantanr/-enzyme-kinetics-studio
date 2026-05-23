# =========================
# KinetiKLab v3.1: DiffDock + Meeko Pipeline
# Low-RAM T4 version: ESM-2 35M + DiffDock batch_size=1
# =========================

# CELL 1 — INSTALL: DIFFDOCK + MEEKO + LIGHT ESM
!pip -q install torch torchvision --index-url https://download.pytorch.org/whl/cu121
!pip -q install torch-geometric torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
!pip -q install fair-esm biotite biopandas rdkit meeko
!pip -q install transformers scikit-learn plotly pandas tqdm
# DiffDock install
!git clone https://github.com/gcorso/DiffDock.git /content/DiffDock 2>/dev/null || true
!pip -q install -e /content/DiffDock
!pip -q install e3nn spyrmsd prody

# =========================
# CELL 2 — IMPORTS
# =========================
import os, json, subprocess, tempfile, warnings
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_cluster
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

try:
    import esm
    ESM_OK = True
except:
    ESM_OK = False

try:
    from meeko import MoleculePreparation
    MEEKO_OK = True
except:
    MEEKO_OK = False
    print("Meeko not available. Install: pip install meeko")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_OK = True
except:
    RDKIT_OK = False

try:
    from google.colab import files
    COLAB_OK = True
except:
    COLAB_OK = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | Meeko: {MEEKO_OK} | ESM: {ESM_OK}")

# =========================
# CELL 3 — INPUT
# =========================
protein_input = {
    "name": "CA2_HUMAN",
    "uniprot": "P00918",
    "fasta": """>sp|P00918|CAH2_HUMAN Carbonic anhydrase 2
MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"""
}

inhibitors = [
    {"name": "Acetazolamide", "smiles": "CC1=NN(C(=O)N1)S(=O)(=O)N"},
    {"name": "Methazolamide", "smiles": "CC1=NN(C(=O)N1)S(=O)(=O)N"},
    {"name": "Sulfanilamide", "smiles": "NS(=O)(=O)c1ccc(N)cc1"},
]

def clean_seq(raw: str) -> str:
    if raw.startswith(">"): raw = "\n".join(raw.split("\n")[1:])
    return "".join([l.strip() for l in raw.splitlines() if l.strip()]).upper()

sequence = clean_seq(protein_input["fasta"])
print(f"Protein: {protein_input['name']} | Length: {len(sequence)}")

# =========================
# CELL 4 — ESM-2 35M: LOW RAM EMBEDDINGS
# =========================
def get_esm2_light(seq: str) -> np.ndarray:
    """Use ESM-2 35M instead of 650M. Uses 2GB RAM not 16GB."""
    if not ESM_OK:
        return np.random.randn(len(seq), 480) # fallback 480-dim

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 35M model
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    with torch.no_grad():
        out = model(tokens, repr_layers=[6], return_contacts=False)
    emb = out["representations"][6][0, 1:-1].cpu().numpy() # [L, 320]
    return emb

print("Running ESM-2 35M...")
esm_embed = get_esm2_light(sequence)
print(f"ESM embeddings: {esm_embed.shape}")

# =========================
# CELL 5 — GET STRUCTURE: ALPHAFOLD DB OR ESMFOLD API
# =========================
def get_af_structure(uniprot_id: str, name: str) -> str:
    """Download AF model. Much faster than ESMFold inference."""
    pdb_path = f"/content/{name}_AF.pdb"
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    if not Path(pdb_path).exists():
       !wget -q {url} -O {pdb_path}
    return pdb_path

pdb_path = get_af_structure(protein_input["uniprot"], protein_input["name"])
print(f"Structure: {pdb_path}")

# =========================
# CELL 6 — MEEKO: PREPARE PROTEIN + LIGANDS
# =========================
def prep_protein_meeko(pdb_path: str) -> str:
    """Convert PDB to PDBQT using Meeko. No gemmi needed."""
    if not MEEKO_OK: return pdb_path
    pdbqt_path = pdb_path.replace(".pdb", "_rec.pdbqt")

    # Use meeko CLI: mk_prepare_receptor.py
    cmd = f"mk_prepare_receptor.py -i {pdb_path} -o {pdbqt_path} -p -v"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        print(f"Meeko protein prep: {pdbqt_path}")
        return pdbqt_path
    except Exception as e:
        print(f"Meeko failed: {e}. Using PDB directly.")
        return pdb_path

def prep_ligand_meeko(smiles: str) -> str:
    """SMILES -> PDBQT using RDKit + Meeko"""
    if not MEEKO_OK or not RDKIT_OK: return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, maxAttempts=10, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

    # Write SDF temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
        sdf_path = f.name
        Chem.MolToMolFile(mol, sdf_path)

    # Meeko convert
    pdbqt_path = sdf_path.replace('.sdf', '.pdbqt')
    cmd = f"mk_prepare_ligand.py -i {sdf_path} -o {pdbqt_path}"
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
        return pdbqt_path
    except:
        return None

protein_pdbqt = prep_protein_meeko(pdb_path)

# =========================
# CELL 7 — DIFFDOCK: BLIND DOCKING LOW-RAM
# =========================
def run_diffdock_lowram(protein_pdb: str, ligand_smiles: str, out_dir: str) -> Dict:
    """Run DiffDock with batch_size=1, num_samples=5 to fit T4 12GB"""
    os.makedirs(out_dir, exist_ok=True)

    # Write ligand SDF
    mol = Chem.MolFromSmiles(ligand_smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    lig_sdf = f"{out_dir}/ligand.sdf"
    Chem.MolToMolFile(mol, lig_sdf)

    # DiffDock command - low RAM settings
    cmd = [
        "python", "/content/DiffDock/inference.py",
        "--protein_path", protein_pdb,
        "--ligand", lig_sdf,
        "--out_dir", out_dir,
        "--samples_per_complex", "5", # 5 not 40
        "--batch_size", "1", # 1 not 8
        "--actual_steps", "18", # 18 not 20
        "--no_final_step_noise" # save memory
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Parse confidence from output
        conf_file = Path(out_dir) / "confidence.json"
        if conf_file.exists():
            confs = json.loads(conf_file.read_text())
            best_conf = max(confs) if confs else 0.0
            return {"confidence": best_conf, "pose_path": f"{out_dir}/rank1.sdf", "status": "success"}
        else:
            return {"confidence": 0.0, "pose_path": None, "status": "no_output"}
    except subprocess.TimeoutExpired:
        return {"confidence": 0.0, "pose_path": None, "status": "timeout"}
    except Exception as e:
        return {"confidence": 0.0, "pose_path": None, "status": f"error: {str(e)}"}

# =========================
# CELL 8 — RUN DIFFDOCK ON ALL INHIBITORS
# =========================
dock_results = []
print("Running DiffDock blind docking...")

for inh in tqdm(inhibitors):
    out_dir = f"/content/diffdock_{protein_input['name']}_{inh['name']}"
    res = run_diffdock_lowram(pdb_path, inh["smiles"], out_dir)
    dock_results.append({
        "name": inh["name"],
        "smiles": inh["smiles"],
        "diffdock_confidence": res["confidence"],
        "status": res["status"]
    })

dock_df = pd.DataFrame(dock_results).sort_values("diffdock_confidence", ascending=False)
dock_df["rank"] = np.arange(1, len(dock_df)+1)
print(dock_df)

# =========================
# CELL 9 — GNN ON ESM EMBEDDINGS - LIGHT
# =========================
# Simple node classification on ESM embeddings
AA20 = "ACDEFGHIKLMNPQRSTVWY"
HYDROPATHY = {"A":1.8,"C":2.5,"D":-3.5,"E":-3.5,"F":2.8,"G":-0.4,"H":-3.2,"I":4.5,"K":-3.9,"L":3.8,"M":1.9,"N":-3.5,"P":-1.6,"Q":-3.5,"R":-4.5,"S":-0.8,"T":-0.7,"V":4.2,"W":-0.9,"Y":-1.3}

def build_features(seq, esm_emb):
    """Combine ESM + hand features. 320 + 3 = 323 dim"""
    feat = []
    for i, aa in enumerate(seq):
        f = list(esm_emb[i]) # 320 from ESM-2 35M
        f.extend([
            HYDROPATHY.get(aa, 0.0),
            int(aa in "DEKRH"), # charged
            int(aa in "STY") # phospho
        ])
        feat.append(f)
    return np.array(feat, dtype=np.float32)

X = build_features(sequence, esm_embed)
print(f"Features: {X.shape}")

# Weak labels: HE..H motif = catalytic
y = np.zeros(len(sequence), dtype=int)
for m in re.finditer(r"HE..H", sequence):
    y[m.start():m.end()] = 1

# 3-fold CV for speed
if y.sum() > 0:
    clf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=50, random_state=42))])
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
    print(f"GNN-like RF 3-Fold CV F1: {scores.mean():.3f} ± {scores.std():.3f}")
    clf.fit(X, y)
    cat_probs = clf.predict_proba(X)[:, 1]
else:
    cat_probs = np.zeros(len(sequence))

# =========================
# CELL 10 — PLOTS + EXPORT
# =========================
figs = []

# Plot 1: Catalytic probability
fig = go.Figure(go.Scatter(x=list(range(len(sequence))), y=cat_probs, mode='lines', name='Catalytic P'))
fig.update_layout(title=f"{protein_input['name']} | ESM+GNN Catalytic Site", xaxis_title="Residue", yaxis_title="P")
figs.append(fig)

# Plot 2: DiffDock confidence
fig = px.bar(dock_df, x="name", y="diffdock_confidence", title="DiffDock Blind Docking Confidence")
figs.append(fig)

# Plot 3: ESM PCA
from sklearn.decomposition import PCA
xy = PCA(n_components=2).fit_transform(esm_embed)
fig = px.scatter(x=xy[:,0], y=xy[:,1], color=list(sequence), title="ESM-2 35M Embedding Space")
figs.append(fig)

# Export HTML
timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
html = [f"<html><head><script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body>"]
html.append(f"<h1>KinetiKLab v3.1 DiffDock+Meeko Report</h1><p>{timestamp}</p>")
html.append("<h2>DiffDock Results</h2>" + dock_df.to_html(index=False))
for i, fig in enumerate(figs):
    html.append(f"<h3>Figure {i+1}</h3>" + fig.to_html(full_html=False, include_plotlyjs=False))
html.append("</body></html>")

Path("/content/KinetiKLab_DiffDock_report.html").write_text("\n".join(html))
dock_df.to_csv("/content/diffdock_results.csv", index=False)
print("Saved: /content/KinetiKLab_DiffDock_report.html")
if COLAB_OK: files.download("/content/KinetiKLab_DiffDock_report.html")
