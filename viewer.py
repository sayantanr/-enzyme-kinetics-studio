import re
from pathlib import Path

import requests
import streamlit as st
import streamlit.components.v1 as components
import py3Dmol

st.set_page_config(page_title="Protein 3D Viewer", page_icon="🧬", layout="wide")

st.title("🧬 UniProt / AlphaFold 3D Structure Viewer")
st.caption("Paste a UniProt ID to load an AlphaFold model and optionally highlight catalytic motifs.")


CATALYTIC_MOTIFS = {
    "HEXXH": "Metalloprotease / zinc-binding",
    "GDSGG": "Serine protease",
    "DTG": "Aspartic protease",
    "HRDLK": "Kinase active site",
    "CGSCWAFS": "Cysteine protease",
}


def clean_sequence(raw: str) -> str:
    """Remove FASTA headers, whitespace, and non-amino-acid characters."""
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    seq = "".join(lines).upper()
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZJUO]", "", seq)


def find_motifs(sequence: str):
    """Find known catalytic motifs in a pasted protein sequence."""
    seq = clean_sequence(sequence)
    hits = []
    if not seq:
        return hits, seq

    for motif, label in CATALYTIC_MOTIFS.items():
        pattern = motif.replace("X", ".")
        for match in re.finditer(pattern, seq):
            hits.append(
                {
                    "motif": label,
                    "pattern": motif,
                    "start": match.start() + 1,  # 1-indexed
                    "end": match.end(),         # inclusive position
                    "sequence": match.group(),
                }
            )
    return hits, seq


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def fetch_alphafold_pdb(uniprot_id: str) -> str:
    """
    Download AlphaFold PDB text for a UniProt ID.
    Tries a few known model versions from newest to older.
    """
    uid = uniprot_id.strip().upper()
    if not uid:
        raise ValueError("UniProt ID is empty.")

    versions = ["v6", "v5", "v4", "v3", "v2", "v1"]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; StreamlitProteinViewer/1.0)"
    }

    last_error = None
    for version in versions:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_{version}.pdb"
        try:
            resp = requests.get(url, timeout=20, headers=headers)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
            last_error = f"{url} returned HTTP {resp.status_code}"
        except Exception as exc:
            last_error = f"{url} failed: {exc}"

    raise RuntimeError(
        f"AlphaFold model not found for {uid}. Last error: {last_error}"
    )


def render_structure(pdb_text: str, hits=None, height: int = 540):
    """Render a PDB structure in a Streamlit app."""
    view = py3Dmol.view(width=900, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.setBackgroundColor("#111827")

    if hits:
        for hit in hits:
            sel = {"resi": f"{hit['start']}-{hit['end']}"}
            view.addStyle(sel, {"stick": {"radius": 0.35, "colorscheme": "redCarbon"}})
            try:
                view.addLabel(
                    hit["motif"],
                    {
                        "position": {"resi": int(hit["start"])},
                        "backgroundColor": "#fde68a",
                        "fontColor": "black",
                        "fontSize": 12,
                        "inFront": True,
                    },
                )
            except Exception:
                # Labeling should never stop the structure from rendering
                pass

    view.zoomTo()
    components.html(view._make_html(), height=height + 35, scrolling=False)


with st.sidebar:
    st.header("Input")
    uniprot_id = st.text_input("UniProt ID", value="P00918", placeholder="P00918").strip().upper()

    seq_input = st.text_area(
        "Optional: Paste sequence to find motifs",
        placeholder=">sp|P00918|CAH2_HUMAN\nMSHHWGYGKHNGPEHW...",
        height=200,
    )

    show_motifs = st.checkbox("Highlight catalytic motifs", value=True)

    st.markdown("---")
    st.caption("Tip: AlphaFold covers most proteins. If a model is missing, the protein may be too short, disordered, or absent from AlphaFold DB.")


if not uniprot_id and not seq_input:
    st.info("Enter a UniProt ID or paste a protein sequence in the sidebar to begin.")
    st.stop()


hits = []
clean_seq = ""

if seq_input:
    hits, clean_seq = find_motifs(seq_input)
    st.subheader("Motif scan")
    if hits:
        st.success(f"Found {len(hits)} motif hit(s).")
        st.dataframe(hits, use_container_width=True)
    else:
        st.warning("No known catalytic motifs detected in the pasted sequence.")

if uniprot_id:
    st.subheader(f"Structure: {uniprot_id}")
    try:
        with st.spinner(f"Loading AlphaFold model for {uniprot_id}..."):
            pdb_text = fetch_alphafold_pdb(uniprot_id)

        render_structure(pdb_text, hits if show_motifs else None)
        st.success(f"Loaded AlphaFold model for {uniprot_id}")

        with st.expander("Direct AlphaFold link"):
            st.write(f"https://alphafold.ebi.ac.uk/entry/{uniprot_id}")

    except Exception as e:
        st.error(f"Failed to load structure: {e}")
        st.info("Check that the UniProt ID is correct and that your machine has internet access.")
        st.markdown(f"[Open {uniprot_id} on AlphaFold](https://alphafold.ebi.ac.uk/entry/{uniprot_id})")


with st.expander("Example IDs"):
    st.code(
        """P00918  - Carbonic anhydrase 2
P07477  - Trypsin
P00698  - Lysozyme
P02768  - Albumin
P68871  - Hemoglobin beta"""
    )
