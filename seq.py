import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Sequence Biochemistry Lab", page_icon="🧬", layout="wide")
st.title("🧬 Sequence Biochemistry Lab")
st.caption("Paste a protein sequence to scan Pfam/HMMER domains, secondary-structure propensity, catalytic motifs, and PTM signals.")


# -----------------------------
# Sequence utilities
# -----------------------------
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def parse_fasta(text: str) -> str:
    if not text:
        return ""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    return clean_sequence("".join(lines))


def clean_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\r", "")
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO]", "", seq)


# -----------------------------
# Motif libraries
# -----------------------------
CATALYTIC_MOTIFS = {
    "Metalloprotease": ["HEXXH", "HExxH"],
    "Serine protease": ["GDSGG", "GxSxG", "GXSXG"],
    "Aspartic protease": ["DTG", "DSG"],
    "Kinase": ["HRDLK", "HRD[LIV]K"],
    "Cysteine protease": ["CGSCWAFS", "CGSC"],
}

PTM_MOTIFS = {
    "N-glycosylation": r"N[^P][ST][^P]",
    "Protein kinase A": r"[RK].{2}[ST]",
    "Casein kinase II": r"[ST].{2}[DE]",
    "N-myristoylation": r"G.{2}[STAGCN]",
    "SUMOylation core": r"[VILMAFP]K.[DE]",
    "Tyrosine phosphorylation": r"[RK].{2}Y",
}


def motif_scan(sequence: str, motif_map: dict) -> pd.DataFrame:
    seq = clean_sequence(sequence)
    rows = []
    for label, patterns in motif_map.items():
        if isinstance(patterns, str):
            patterns = [patterns]
        for motif in patterns:
            regex = motif.replace("X", ".").replace("x", ".")
            for m in re.finditer(regex, seq):
                rows.append(
                    {
                        "category": label,
                        "pattern": motif,
                        "start": m.start() + 1,
                        "end": m.end(),
                        "match": m.group(),
                    }
                )
    return pd.DataFrame(rows)


# -----------------------------
# Secondary structure propensity
# Sequence-only heuristic, useful when you do not have a structure file.
# -----------------------------
HELIX_PROP = {
    "A": 1.42, "C": 0.70, "D": 1.01, "E": 1.51, "F": 1.13,
    "G": 0.57, "H": 1.00, "I": 1.08, "K": 1.16, "L": 1.21,
    "M": 1.45, "N": 0.67, "P": 0.57, "Q": 1.11, "R": 0.98,
    "S": 0.77, "T": 0.83, "V": 1.06, "W": 1.08, "Y": 0.69,
}
SHEET_PROP = {
    "A": 0.83, "C": 1.19, "D": 0.54, "E": 0.37, "F": 1.38,
    "G": 0.75, "H": 0.87, "I": 1.60, "K": 0.74, "L": 1.30,
    "M": 1.05, "N": 0.89, "P": 0.55, "Q": 1.10, "R": 0.93,
    "S": 0.75, "T": 1.19, "V": 1.70, "W": 1.37, "Y": 1.47,
}
TURN_PROP = {
    "A": 0.66, "C": 1.19, "D": 1.46, "E": 0.74, "F": 0.60,
    "G": 1.56, "H": 0.95, "I": 0.47, "K": 1.01, "L": 0.59,
    "M": 0.60, "N": 1.56, "P": 1.52, "Q": 0.98, "R": 0.95,
    "S": 1.43, "T": 0.96, "V": 0.50, "W": 0.96, "Y": 1.14,
}


def prop_score(seq: str, table: dict, window: int = 9) -> np.ndarray:
    seq = clean_sequence(seq)
    vals = np.array([table.get(aa, 1.0) for aa in seq], dtype=float)
    if len(vals) == 0:
        return vals
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(vals, kernel, mode="same")


def predict_secondary_structure(sequence: str) -> pd.DataFrame:
    seq = clean_sequence(sequence)
    if not seq:
        return pd.DataFrame(columns=["position", "aa", "helix", "sheet", "turn", "prediction"])

    helix = prop_score(seq, HELIX_PROP, window=11)
    sheet = prop_score(seq, SHEET_PROP, window=11)
    turn = prop_score(seq, TURN_PROP, window=7)

    pred = []
    for i, aa in enumerate(seq):
        scores = {"Helix": helix[i], "Sheet": sheet[i], "Turn": turn[i]}
        pred.append(max(scores, key=scores.get))

    return pd.DataFrame(
        {
            "position": np.arange(1, len(seq) + 1),
            "aa": list(seq),
            "helix": helix,
            "sheet": sheet,
            "turn": turn,
            "prediction": pred,
        }
    )


# -----------------------------
# HMMER / Pfam integration
# -----------------------------
@st.cache_data(show_spinner=False)
def run_hmmscan(sequence: str, hmmscan_bin: str, pfam_hmm_path: str) -> pd.DataFrame:
    seq = clean_sequence(sequence)
    if not seq:
        return pd.DataFrame()

    hmmscan_bin = hmmscan_bin or "hmmscan"
    pfam_hmm_path = pfam_hmm_path.strip()
    if not shutil.which(hmmscan_bin):
        raise FileNotFoundError("hmmscan executable not found in PATH.")
    if not pfam_hmm_path or not Path(pfam_hmm_path).exists():
        raise FileNotFoundError("Pfam HMM database file not found.")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fasta = td / "query.fasta"
        domtbl = td / "hmmscan.domtblout"
        fasta.write_text(">query\n" + seq + "\n")

        cmd = [
            hmmscan_bin,
            "--domtblout", str(domtbl),
            "--noali",
            pfam_hmm_path,
            str(fasta),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode not in (0, 1):
            raise RuntimeError(proc.stderr.strip() or "hmmscan failed")

        if not domtbl.exists():
            return pd.DataFrame()

        rows = []
        for line in domtbl.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            rows.append(
                {
                    "target_name": parts[0],
                    "target_accession": parts[1],
                    "query_name": parts[3],
                    "query_accession": parts[4],
                    "full_evalue": float(parts[6]),
                    "full_score": float(parts[7]),
                    "full_bias": float(parts[8]),
                    "domain_num": int(parts[9]),
                    "domain_of": int(parts[10]),
                    "c_evalue": float(parts[11]),
                    "i_evalue": float(parts[12]),
                    "hmm_from": int(parts[15]),
                    "hmm_to": int(parts[16]),
                    "ali_from": int(parts[17]),
                    "ali_to": int(parts[18]),
                    "env_from": int(parts[19]),
                    "env_to": int(parts[20]),
                    "acc": float(parts[21]),
                    "description": " ".join(parts[22:]),
                }
            )

        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["full_evalue", "full_score"], ascending=[True, False]).reset_index(drop=True)
        return out


# -----------------------------
# Visual helpers
# -----------------------------

def structure_plot(df: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["position"], y=df["helix"], mode="lines", name="Helix propensity"))
    fig.add_trace(go.Scatter(x=df["position"], y=df["sheet"], mode="lines", name="Sheet propensity"))
    fig.add_trace(go.Scatter(x=df["position"], y=df["turn"], mode="lines", name="Turn propensity"))
    fig.update_layout(height=400, xaxis_title="Residue position", yaxis_title="Propensity score", legend_title_text="Prediction")
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Input")
    sequence_text = st.text_area(
        "Paste FASTA or raw protein sequence",
        height=220,
        placeholder=">sp|P00918|CAH2_HUMAN\nMSHHWGYGKHNGPEHWHKDFPIAKGER...",
    )
    uploaded = st.file_uploader("Or upload a FASTA/TXT file", type=["fasta", "fa", "faa", "txt"])

    st.header("HMMER / Pfam")
    hmmscan_bin = st.text_input("hmmscan executable", value="hmmscan")
    pfam_hmm_path = st.text_input("Pfam-A HMM database path", value="")

    run = st.button("Analyze sequence", type="primary")


# -----------------------------
# Main input handling
# -----------------------------
if uploaded is not None and not sequence_text.strip():
    sequence_text = uploaded.getvalue().decode("utf-8", errors="ignore")

sequence = parse_fasta(sequence_text)

if not sequence:
    st.info("Paste a protein sequence or upload a FASTA file to begin.")
    st.stop()

st.success(f"Loaded sequence with {len(sequence)} amino acids.")

col1, col2 = st.columns([2, 1])
with col1:
    st.text_area("Cleaned sequence", value=sequence, height=180)
with col2:
    st.metric("Length", len(sequence))
    st.metric("Valid amino acids", sum(aa in AA20 for aa in sequence))
    st.metric("Hydrophobic fraction", round(sum(aa in set("AILMFWVY") for aa in sequence) / max(len(sequence), 1), 3))

if run:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Pfam / HMMER",
        "Secondary structure",
        "Catalytic residues",
        "PTM prediction",
        "Summary",
    ])

    with tab1:
        st.subheader("Pfam / HMMER domain prediction")
        st.caption("This uses a real local hmmscan run when hmmscan and Pfam-A.hmm are installed.")
        try:
            hmmer_df = run_hmmscan(sequence, hmmscan_bin, pfam_hmm_path)
            if hmmer_df.empty:
                st.warning("No Pfam hits found, or the database returned no significant matches.")
            else:
                st.dataframe(hmmer_df, use_container_width=True)
                st.download_button(
                    "Download HMMER hits as CSV",
                    hmmer_df.to_csv(index=False).encode("utf-8"),
                    file_name="hmmer_hits.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"HMMER could not run: {e}")
            st.info("Install HMMER, download Pfam-A.hmm, and point the app to the database path.")

    with tab2:
        st.subheader("Secondary structure estimation")
        sec_df = predict_secondary_structure(sequence)
        st.dataframe(sec_df.head(50), use_container_width=True)
        structure_plot(sec_df)
        counts = sec_df["prediction"].value_counts().reindex(["Helix", "Sheet", "Turn"]).fillna(0)
        st.bar_chart(counts)

    with tab3:
        st.subheader("Catalytic residue and motif scan")
        cat_df = motif_scan(sequence, CATALYTIC_MOTIFS)
        if cat_df.empty:
            st.warning("No strong catalytic motifs detected from the current motif library.")
        else:
            st.dataframe(cat_df, use_container_width=True)
        st.caption("This is a motif-based screen. It is best used as a candidate-site filter, not a final annotation.")

    with tab4:
        st.subheader("PTM prediction")
        ptm_df = motif_scan(sequence, PTM_MOTIFS)
        if ptm_df.empty:
            st.warning("No common PTM motifs detected.")
        else:
            st.dataframe(ptm_df, use_container_width=True)
        st.caption("Detected motifs include common phosphorylation, glycosylation, myristoylation, and SUMO-like signals.")

    with tab5:
        st.subheader("Sequence summary")
        summary = {
            "length": len(sequence),
            "molecular_class_hint": "protein",
            "helix_propensity_mean": float(np.mean(predict_secondary_structure(sequence)["helix"])),
            "sheet_propensity_mean": float(np.mean(predict_secondary_structure(sequence)["sheet"])),
            "turn_propensity_mean": float(np.mean(predict_secondary_structure(sequence)["turn"])),
            "catalytic_hits": int(len(motif_scan(sequence, CATALYTIC_MOTIFS))),
            "ptm_hits": int(len(motif_scan(sequence, PTM_MOTIFS))),
        }
        st.json(summary)
        st.caption("Use the tabs above for detailed domain, structure, catalytic-site, and PTM views.")
else:
    st.info("Press **Analyze sequence** to run the full pipeline.")
    st.write("Supported inputs: raw amino-acid sequence or FASTA format.")
    st.write("For real Pfam annotation, install HMMER and provide a Pfam-A HMM database path.")
