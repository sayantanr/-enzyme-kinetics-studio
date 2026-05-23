import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional dependencies
try:
    from lmfit import Parameters, minimize
    HAS_LMFIT = True
except Exception:
    HAS_LMFIT = False

# ----------------------------
# PAGE
# ----------------------------
st.set_page_config(page_title="Enzyme Kinetics Lab", page_icon="⚗", layout="wide")
st.title("⚗ Enzyme Kinetics Simulator + Inhibitor Screening")
st.caption("50-curve dashboard | 50-row table | model comparison | no 3D dependencies")

# ----------------------------
# MODELS
# ----------------------------
def michaelis_menten(S, Vmax, Km):
    S = np.asarray(S, dtype=float)
    return Vmax * S / (Km + S + 1e-12)

def hill(S, Vmax, Khalf, n):
    S = np.asarray(S, dtype=float)
    return Vmax * S**n / (Khalf**n + S**n + 1e-12)

def substrate_inhibition(S, Vmax, Km, Ki_sub):
    S = np.asarray(S, dtype=float)
    return Vmax * S / (Km + S + (S**2) / (Ki_sub + 1e-12))

def competitive(S, I, Vmax, Km, Ki):
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    Km_app = Km * (1 + I / (Ki + 1e-12))
    return Vmax * S / (Km_app + S + 1e-12)

def noncompetitive(S, I, Vmax, Km, Ki):
    S = np.asarray(S, dtype=float)
    I = np.asarray(I, dtype=float)
    Vmax_app = Vmax / (1 + I / (Ki + 1e-12))
    return Vmax_app * S / (Km + S + 1e-12)

MODELS = {
    "Michaelis-Menten": michaelis_menten,
    "Hill": hill,
    "Substrate Inhibition": substrate_inhibition,
    "Competitive": competitive,
    "Non-competitive": noncompetitive,
}

# ----------------------------
# SEQUENCE / MOTIFS
# ----------------------------
CATALYTIC_MOTIFS = {
    "Serine protease": "GDSGG",
    "Cysteine protease": "CGSCWAFS",
    "Aspartic protease": "DTG",
    "Metalloprotease": "HEXXH",
    "Kinase": "HRD[LIV]K",
}

def clean_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "")
    seq = "".join(ch for ch in seq if ch.isalpha())
    return seq

def predict_active_site(seq: str):
    import re
    seq = clean_sequence(seq)
    hits = []
    for motif_name, motif in CATALYTIC_MOTIFS.items():
        pattern = motif.replace("X", ".")
        for match in re.finditer(pattern, seq):
            hits.append({
                "motif": motif_name,
                "pattern": motif,
                "start": match.start() + 1, # 1-indexed
                "end": match.end(),
                "sequence": match.group(),
            })
    return hits, seq

# ----------------------------
# VALIDATION / HELPERS
# ----------------------------
REQUIRED_COLUMNS = {"curve_id", "inhibitor", "inhibitor_conc_uM", "substrate_conc_mM", "v0_uM_per_min"}

def validate_input_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out["curve_id"] = out["curve_id"].astype(str)
    out["inhibitor"] = out["inhibitor"].astype(str)
    for col in ["inhibitor_conc_uM", "substrate_conc_mM", "v0_uM_per_min"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["substrate_conc_mM", "v0_uM_per_min"]).reset_index(drop=True)
    out["inhibitor_conc_uM"] = out["inhibitor_conc_uM"].fillna(0.0)
    out["substrate_conc_mM"] = out["substrate_conc_mM"].clip(lower=0)
    out["v0_uM_per_min"] = out["v0_uM_per_min"].clip(lower=0)
    out["inhibitor_conc_uM"] = out["inhibitor_conc_uM"].clip(lower=0)
    return out

def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce")
    return float(np.nanmean(s)) if np.isfinite(np.nanmean(s)) else np.nan

def curve_summary_table(df: pd.DataFrame, pred_col: str | None = None) -> pd.DataFrame:
    grp = []
    for curve_id, d in df.groupby("curve_id", sort=False):
        row = {
            "curve_id": curve_id,
            "inhibitor": d["inhibitor"].iloc[0] if "inhibitor" in d.columns else "",
            "inhibitor_conc_uM": float(d["inhibitor_conc_uM"].iloc[0]) if "inhibitor_conc_uM" in d.columns else 0.0,
            "n_points": int(len(d)),
            "substrate_min": float(np.nanmin(d["substrate_conc_mM"])),
            "substrate_max": float(np.nanmax(d["substrate_conc_mM"])),
            "v0_mean": safe_mean(d["v0_uM_per_min"]),
            "v0_max": float(np.nanmax(d["v0_uM_per_min"])),
        }
        if pred_col and pred_col in d.columns:
            resid = d["v0_uM_per_min"] - d[pred_col]
            row["rmse"] = float(np.sqrt(np.nanmean(np.square(resid))))
        grp.append(row)
    return pd.DataFrame(grp)

# ----------------------------
# FITTING
# ----------------------------
def residual_global(params, data, model_name):
    Vmax = params["Vmax"].value
    Km = params["Km"].value
    residuals = []

    for _, row in data.iterrows():
        S = float(row["substrate_conc_mM"])
        v_obs = float(row["v0_uM_per_min"])
        I = float(row.get("inhibitor_conc_uM", 0.0))

        if model_name == "Michaelis-Menten":
            v_pred = michaelis_menten(S, Vmax, Km)
        elif model_name == "Hill":
            v_pred = hill(S, Vmax, params["Khalf"].value, params["n"].value)
        elif model_name == "Substrate Inhibition":
            v_pred = substrate_inhibition(S, Vmax, Km, params["Ki_sub"].value)
        elif model_name == "Competitive":
            v_pred = competitive(S, I, Vmax, Km, params["Ki"].value)
        elif model_name == "Non-competitive":
            v_pred = noncompetitive(S, I, Vmax, Km, params["Ki"].value)
        else:
            v_pred = michaelis_menten(S, Vmax, Km)

        residuals.append(v_obs - v_pred)

    return np.asarray(residuals, dtype=float)

def fit_all_models(data: pd.DataFrame):
    if not HAS_LMFIT:
        return None, {}

    results = {}
    S_data = data["substrate_conc_mM"].values
    v_data = data["v0_uM_per_min"].values

    for name in MODELS.keys():
        params = Parameters()
        params.add("Vmax", value=float(np.nanmax(v_data) * 1.2 if np.isfinite(np.nanmax(v_data)) else 1.0), min=0)
        params.add("Km", value=float(np.nanmedian(S_data) if np.isfinite(np.nanmedian(S_data)) else 1.0), min=1e-9)

        if name == "Hill":
            params.add("n", value=2.0, min=0.1, max=10.0)
            params.add("Khalf", value=float(np.nanmedian(S_data) if np.isfinite(np.nanmedian(S_data)) else 1.0), min=1e-9)
        elif name == "Substrate Inhibition":
            params.add("Ki_sub", value=float(np.nanmax(S_data) * 2 if np.isfinite(np.nanmax(S_data)) else 1.0), min=1e-9)
        elif name in ["Competitive", "Non-competitive"]:
            params.add("Ki", value=50.0, min=1e-9)

        try:
            result = minimize(residual_global, params, args=(data, name), method="leastsq")
            result.model_name = name
            results[name] = result
        except Exception:
            continue

    if not results:
        return None, {}

    best = min(results.values(), key=lambda x: x.aic)
    return best, results

def fallback_prediction(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vmax = float(np.nanmax(out["v0_uM_per_min"]) if len(out) else 1.0)
    km = float(np.nanmedian(out["substrate_conc_mM"]) if len(out) else 1.0)
    out["v_pred"] = michaelis_menten(out["substrate_conc_mM"], vmax, km)
    out["model_name"] = "Michaelis-Menten (fallback)"
    return out

def build_predictions(df: pd.DataFrame, best_fit):
    out = df.copy()

    if best_fit is None or not HAS_LMFIT:
        return fallback_prediction(out), None, pd.DataFrame()

    Vmax = best_fit.params["Vmax"].value
    Km = best_fit.params["Km"].value

    model_name = best_fit.model_name
    if model_name == "Competitive":
        out["v_pred"] = competitive(out["substrate_conc_mM"], out["inhibitor_conc_uM"], Vmax, Km, best_fit.params["Ki"].value)
    elif model_name == "Non-competitive":
        out["v_pred"] = noncompetitive(out["substrate_conc_mM"], out["inhibitor_conc_uM"], Vmax, Km, best_fit.params["Ki"].value)
    elif model_name == "Hill":
        out["v_pred"] = hill(out["substrate_conc_mM"], Vmax, best_fit.params["Khalf"].value, best_fit.params["n"].value)
    elif model_name == "Substrate Inhibition":
        out["v_pred"] = substrate_inhibition(out["substrate_conc_mM"], Vmax, Km, best_fit.params["Ki_sub"].value)
    else:
        out["v_pred"] = michaelis_menten(out["substrate_conc_mM"], Vmax, Km)

    out["model_name"] = model_name
    out["residual"] = out["v0_uM_per_min"] - out["v_pred"]

    summary = curve_summary_table(out, "v_pred")
    return out, best_fit, summary

# ----------------------------
# PLOTS
# ----------------------------
def plot_small_multiples(df_plot: pd.DataFrame, best_fit, max_curves_plot: int):
    curves = list(df_plot["curve_id"].dropna().unique())[:max_curves_plot]
    if not curves:
        st.info("No curve_id values found.")
        return

    n_cols = 5
    n_rows = int(np.ceil(len(curves) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[str(c) for c in curves],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    for idx, curve in enumerate(curves):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        d = df_plot[df_plot["curve_id"] == curve].sort_values("substrate_conc_mM")

        fig.add_trace(
            go.Scatter(
                x=d["substrate_conc_mM"],
                y=d["v0_uM_per_min"],
                mode="markers",
                marker=dict(size=6),
                name=f"{curve} data",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        smax = float(np.nanmax(d["substrate_conc_mM"])) if len(d) else 1.0
        smax = max(smax, 1e-6)
        S_smooth = np.linspace(0, smax * 1.1, 120)
        I = float(d["inhibitor_conc_uM"].iloc[0]) if "inhibitor_conc_uM" in d.columns and len(d) else 0.0

        if best_fit is None or not HAS_LMFIT:
            v_smooth = michaelis_menten(S_smooth, float(np.nanmax(df_plot["v0_uM_per_min"])), float(np.nanmedian(df_plot["substrate_conc_mM"])))
        else:
            Vmax = best_fit.params["Vmax"].value
            Km = best_fit.params["Km"].value
            if best_fit.model_name == "Competitive":
                v_smooth = competitive(S_smooth, I, Vmax, Km, best_fit.params["Ki"].value)
            elif best_fit.model_name == "Non-competitive":
                v_smooth = noncompetitive(S_smooth, I, Vmax, Km, best_fit.params["Ki"].value)
            elif best_fit.model_name == "Hill":
                v_smooth = hill(S_smooth, Vmax, best_fit.params["Khalf"].value, best_fit.params["n"].value)
            elif best_fit.model_name == "Substrate Inhibition":
                v_smooth = substrate_inhibition(S_smooth, Vmax, Km, best_fit.params["Ki_sub"].value)
            else:
                v_smooth = michaelis_menten(S_smooth, Vmax, Km)

        fig.add_trace(
            go.Scatter(
                x=S_smooth,
                y=v_smooth,
                mode="lines",
                name=f"{curve} fit",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_xaxes(title_text="[S] (mM)")
    fig.update_yaxes(title_text="v0 (uM/min)")
    fig.update_layout(height=max(600, 220 * n_rows), title_text=f"Top {len(curves)} Curves - M-M Plots")
    st.plotly_chart(fig, use_container_width=True)

def plot_lineweaver_burk(df_plot: pd.DataFrame):
    st.subheader("📉 Lineweaver-Burk Plot")
    fig = go.Figure()
    for inhibitor in df_plot["inhibitor"].fillna("Unknown").unique():
        d = df_plot[df_plot["inhibitor"].fillna("Unknown") == inhibitor].copy()
        d = d[(d["substrate_conc_mM"] > 0) & (d["v0_uM_per_min"] > 0)]
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=1 / d["substrate_conc_mM"],
                y=1 / d["v0_uM_per_min"],
                mode="markers+lines",
                name=str(inhibitor),
            )
        )
    fig.update_layout(xaxis_title="1/[S] (1/mM)", yaxis_title="1/v0 (min/uM)", height=500)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("1. Enzyme Input")
enzyme_seq = st.sidebar.text_area(
    "Paste enzyme sequence",
    height=160,
    placeholder=">sp|P00918|CAH2_HUMAN\nMSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDT..."
)

st.sidebar.header("2. Upload Kinetics Data")
uploaded = st.sidebar.file_uploader(
    "CSV columns: curve_id,inhibitor,inhibitor_conc_uM,substrate_conc_mM,v0_uM_per_min",
    type=["csv"],
)

st.sidebar.header("3. Dashboard Options")
max_curves_plot = st.sidebar.slider("Max curves to plot", 5, 50, 50)
show_raw_50 = st.sidebar.checkbox("Show first 50 raw rows", value=True)
show_summary_50 = st.sidebar.checkbox("Show 50-row curve summary", value=True)
run_fit = st.sidebar.button("🚀 Fit Models + Build Dashboard", type="primary")

# ----------------------------
# MAIN
# ----------------------------
if enzyme_seq:
    st.subheader("🧬 Active Site Prediction")
    hits, clean_seq = predict_active_site(enzyme_seq)

    if hits:
        st.success(f"Found {len(hits)} potential catalytic motif hit(s)")
        st.dataframe(pd.DataFrame(hits), use_container_width=True)
    else:
        st.warning("No known catalytic motifs detected in this sequence.")

if uploaded is not None:
    try:
        df = validate_input_df(pd.read_csv(uploaded))
    except Exception as e:
        st.error(f"CSV error: {e}")
        st.stop()

    st.subheader("📊 Raw Data Preview")
    if show_raw_50:
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    st.info(f"Loaded {len(df)} rows across {df['curve_id'].nunique()} curve(s).")

    if run_fit:
        if HAS_LMFIT:
            with st.spinner("Fitting models across all curves..."):
                best_fit, all_fits = fit_all_models(df)
            df_plot, best_fit, summary_df = build_predictions(df, best_fit)
        else:
            best_fit = None
            all_fits = {}
            df_plot = fallback_prediction(df)
            summary_df = curve_summary_table(df_plot, "v_pred")

        st.subheader("📐 50-Row Curve Summary Table")
        if show_summary_50:
            if summary_df.empty:
                summary_df = curve_summary_table(df_plot, "v_pred")
            st.dataframe(summary_df.head(50), use_container_width=True)
        else:
            st.dataframe(summary_df, use_container_width=True)

        if best_fit is not None and HAS_LMFIT:
            st.success(
                f"Best model: **{best_fit.model_name}** | "
                f"AIC: {best_fit.aic:.2f} | BIC: {best_fit.bic:.2f} | "
                f"Reduced χ²: {best_fit.redchi:.4g}"
            )

            st.subheader("📐 Fitted Parameters")
            param_df = pd.DataFrame(
                {k: [v.value, v.stderr] for k, v in best_fit.params.items()},
                index=["value", "stderr"]
            ).T
            st.dataframe(param_df, use_container_width=True)

            st.subheader("🔬 Model Comparison")
            aic_df = pd.DataFrame(
                {m: [r.aic, r.bic, r.redchi] for m, r in all_fits.items()},
                index=["AIC", "BIC", "Reduced χ²"]
            ).T
            st.dataframe(aic_df.style.highlight_min(axis=0), use_container_width=True)
        else:
            st.warning("lmfit not available, so the app is using a stable fallback prediction.")

        st.subheader(f"📈 50 Graph Dashboard: Top {max_curves_plot} Curves")
        plot_small_multiples(df_plot, best_fit if HAS_LMFIT else None, max_curves_plot=max_curves_plot)

        plot_lineweaver_burk(df_plot)

        if best_fit is not None and HAS_LMFIT and hasattr(best_fit, "params") and "Ki" in best_fit.params:
            st.subheader("🏆 Inhibitor Ranking by Ki")
            ki_val = float(best_fit.params["Ki"].value)
            ki_df = (
                df_plot.groupby("inhibitor", as_index=False)
               .agg(
                    inhibitor_conc_uM=("inhibitor_conc_uM", "mean"),
                    curve_count=("curve_id", "nunique"),
                )
               .copy()
            )
            ki_df["Ki_uM"] = ki_val
            ki_df["pKi"] = -np.log10(max(ki_val, 1e-12) * 1e-6)
            ki_df = ki_df.sort_values("Ki_uM")
            st.dataframe(ki_df, use_container_width=True)
            st.plotly_chart(px.bar(ki_df, x="inhibitor", y="pKi", title="Inhibitor Potency"), use_container_width=True)

    else:
        st.info("Press **Fit Models + Build Dashboard** to generate the 50-graph and 50-row views.")
        st.subheader("📈 Preview Plot")
        if len(df) > 0:
            preview_df = df.copy()
            preview_df["v_pred"] = michaelis_menten(
                preview_df["substrate_conc_mM"],
                float(np.nanmax(preview_df["v0_uM_per_min"])),
                float(np.nanmedian(preview_df["substrate_conc_mM"]))
            )
            plot_small_multiples(preview_df, None, max_curves_plot=min(max_curves_plot, 10))
        st.subheader("📐 Preview Table")
        st.dataframe(df.head(50), use_container_width=True)

else:
    st.info("👈 Paste an enzyme sequence and upload a CSV to start.")
    with st.expander("Example CSV format"):
        example = pd.DataFrame({
            "curve_id": ["Ctrl_1", "Ctrl_1", "InhA_5uM", "InhA_5uM"],
            "inhibitor": ["None", "None", "Compound_A", "Compound_A"],
            "inhibitor_conc_uM": [0, 0, 5, 5],
            "substrate_conc_mM": [0.1, 0.5, 0.1, 0.5],
            "v0_uM_per_min": [10, 35, 4, 18],
        })
        st.dataframe(example, use_container_width=True)