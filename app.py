import io
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# PNS Linguistic Tables (from PNMEREC paper)
# Each scale is independent. Strict lookup only.
# -----------------------------
PNS_TABLES: Dict[int, Dict[int, Tuple[float, float, float]]] = {
    5: {
        1: (0.10, 0.85, 0.90),
        2: (0.30, 0.65, 0.70),
        3: (0.50, 0.45, 0.45),
        4: (0.70, 0.25, 0.20),
        5: (0.90, 0.10, 0.05),
    },
    7: {
        1: (0.10, 0.80, 0.90),
        2: (0.20, 0.70, 0.80),
        3: (0.35, 0.60, 0.60),
        4: (0.50, 0.40, 0.45),
        5: (0.65, 0.30, 0.25),
        6: (0.80, 0.20, 0.15),
        7: (0.90, 0.10, 0.10),
    },
    9: {
        1: (0.05, 0.90, 0.95),
        2: (0.10, 0.85, 0.90),
        3: (0.20, 0.80, 0.75),
        4: (0.35, 0.65, 0.60),
        5: (0.50, 0.50, 0.45),
        6: (0.65, 0.35, 0.30),
        7: (0.80, 0.25, 0.20),
        8: (0.90, 0.15, 0.10),
        9: (0.95, 0.05, 0.05),
    },
    11: {
        1: (0.05, 0.90, 0.95),
        2: (0.10, 0.80, 0.85),
        3: (0.20, 0.70, 0.75),
        4: (0.30, 0.60, 0.65),
        5: (0.40, 0.50, 0.55),
        6: (0.50, 0.45, 0.45),
        7: (0.60, 0.40, 0.35),
        8: (0.70, 0.30, 0.25),
        9: (0.80, 0.20, 0.15),
        10: (0.90, 0.15, 0.10),
        11: (0.95, 0.05, 0.05),
    },
}


# -----------------------------
# Helpers
# -----------------------------
def is_bc_row(values: List[str]) -> bool:
    """Return True if all non-empty values are B/C (case-insensitive)."""
    if len(values) == 0:
        return False
    cleaned = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s == "":
            continue
        cleaned.append(s.upper())
    if len(cleaned) == 0:
        return False
    return all(x in {"B", "C"} for x in cleaned)


def coerce_int_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce all cells to int; raise ValueError if impossible."""
    out = df.copy()
    for c in out.columns:
        try:
            out[c] = out[c].apply(lambda x: int(str(x).strip()))
        except Exception as e:
            raise ValueError(f"Column '{c}' contains non-integer values. ({e})")
    return out


def validate_score_range(df_int: pd.DataFrame, scale: int) -> None:
    lo, hi = 1, scale
    bad = (df_int < lo) | (df_int > hi)
    if bad.values.any():
        idx = np.argwhere(bad.values)[0]
        r, c = idx[0], idx[1]
        bad_val = df_int.iloc[r, c]
        raise ValueError(
            f"Invalid crisp score: {bad_val}. Allowed range for {scale}-point scale is {lo}..{hi}."
        )


def map_crisp_to_pns(df_int: pd.DataFrame, scale: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (tau, xi, eta) arrays of shape (m, n)."""
    table = PNS_TABLES[scale]
    m, n = df_int.shape
    tau = np.zeros((m, n), dtype=float)
    xi = np.zeros((m, n), dtype=float)
    eta = np.zeros((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            s = int(df_int.iat[i, j])
            t, x, e = table[s]
            tau[i, j] = t
            xi[i, j] = x
            eta[i, j] = e
    return tau, xi, eta


def normalize_pns(
    tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, crit_types: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Component-wise normalization.
    Benefit: component / max(component)
    Cost:    min(component) / component
    """
    m, n = tau.shape
    tau_n = np.zeros_like(tau)
    xi_n = np.zeros_like(xi)
    eta_n = np.zeros_like(eta)

    for j in range(n):
        ctype = crit_types[j].upper()
        if ctype not in {"B", "C"}:
            raise ValueError("Criterion types must be 'B' or 'C' for every criterion.")

        if ctype == "B":
            tmax = float(np.max(tau[:, j]))
            xmax = float(np.max(xi[:, j]))
            emax = float(np.max(eta[:, j]))
            if tmax == 0 or xmax == 0 or emax == 0:
                raise ValueError(
                    f"Normalization failed: max component is 0 for criterion {j+1}. "
                    "Please check your table / data."
                )
            tau_n[:, j] = tau[:, j] / tmax
            xi_n[:, j] = xi[:, j] / xmax
            eta_n[:, j] = eta[:, j] / emax

        else:  # Cost
            tmin = float(np.min(tau[:, j]))
            xmin = float(np.min(xi[:, j]))
            emin = float(np.min(eta[:, j]))
            if np.any(tau[:, j] == 0) or np.any(xi[:, j] == 0) or np.any(eta[:, j] == 0):
                raise ValueError(
                    f"Normalization failed: a component is 0 in criterion {j+1}, "
                    "cannot divide by zero in cost normalization."
                )
            tau_n[:, j] = tmin / tau[:, j]
            xi_n[:, j] = xmin / xi[:, j]
            eta_n[:, j] = emin / eta[:, j]

    return tau_n, xi_n, eta_n


def apply_weights(
    tau_n: np.ndarray, xi_n: np.ndarray, eta_n: np.ndarray, w: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multiply each criterion column by its weight."""
    tau_w = tau_n * w.reshape(1, -1)
    xi_w = xi_n * w.reshape(1, -1)
    eta_w = eta_n * w.reshape(1, -1)
    return tau_w, xi_w, eta_w


def compute_ideals(
    tau_w: np.ndarray, xi_w: np.ndarray, eta_w: np.ndarray, crit_types: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    PIS/NIS exactly as per your paper:

    For Benefit (K1):
      V+ = (max tau, min xi, min eta)
      V- = (min tau, max xi, max eta)

    For Cost (K2):
      V+ = (min tau, max xi, max eta)
      V- = (max tau, min xi, min eta)
    """
    n = tau_w.shape[1]
    tau_p = np.zeros(n, dtype=float)
    xi_p = np.zeros(n, dtype=float)
    eta_p = np.zeros(n, dtype=float)

    tau_n = np.zeros(n, dtype=float)
    xi_n = np.zeros(n, dtype=float)
    eta_n = np.zeros(n, dtype=float)

    for j in range(n):
        ctype = crit_types[j].upper()
        if ctype == "B":
            tau_p[j] = float(np.max(tau_w[:, j]))
            xi_p[j] = float(np.min(xi_w[:, j]))
            eta_p[j] = float(np.min(eta_w[:, j]))

            tau_n[j] = float(np.min(tau_w[:, j]))
            xi_n[j] = float(np.max(xi_w[:, j]))
            eta_n[j] = float(np.max(eta_w[:, j]))
        else:  # Cost
            tau_p[j] = float(np.min(tau_w[:, j]))
            xi_p[j] = float(np.max(xi_w[:, j]))
            eta_p[j] = float(np.max(eta_w[:, j]))

            tau_n[j] = float(np.max(tau_w[:, j]))
            xi_n[j] = float(np.min(xi_w[:, j]))
            eta_n[j] = float(np.min(eta_w[:, j]))

    return tau_p, xi_p, eta_p, tau_n, xi_n, eta_n


def pn_euclidean_distance_over_criteria(
    tau_row: np.ndarray,
    xi_row: np.ndarray,
    eta_row: np.ndarray,
    tau_ideal: np.ndarray,
    xi_ideal: np.ndarray,
    eta_ideal: np.ndarray,
) -> float:
    """
    Normalized PN-Euclidean distance over criteria j:
      d = sqrt( (1/(3n)) * sum_j [ (tau-tau*)^2 + (xi-xi*)^2 + (eta-eta*)^2 ] )
    """
    n = tau_row.shape[0]
    diff2 = (tau_row - tau_ideal) ** 2 + (xi_row - xi_ideal) ** 2 + (eta_row - eta_ideal) ** 2
    return float(math.sqrt((1.0 / (3.0 * n)) * float(np.sum(diff2))))


def format_triplets(tau: np.ndarray, xi: np.ndarray, eta: np.ndarray, decimals: int = 2) -> pd.DataFrame:
    """Create a DataFrame of '(tau, xi, eta)' strings."""
    m, n = tau.shape
    out = np.empty((m, n), dtype=object)
    for i in range(m):
        for j in range(n):
            out[i, j] = f"({tau[i,j]:.{decimals}f}, {xi[i,j]:.{decimals}f}, {eta[i,j]:.{decimals}f})"
    return pd.DataFrame(out)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=True)
    buf.seek(0)
    return buf.read()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="PNTOPSIS (PNS-TOPSIS) Ranking", layout="wide")
st.title("PNTOPSIS Ranking (Pythagorean Neutrosophic TOPSIS)")

with st.expander("Method implemented (fixed)", expanded=False):
    st.markdown(
        """
- Crisp scores map to **PNS triplets** using a selected **5/7/9/11** linguistic table (strict lookup).
- Criterion types are **Benefit (B)** or **Cost (C)**. If missing in file, you set them in the UI.
- Weights: **Equal** or **Manual** (no auto-normalization; warning if sum != 1, but computation proceeds).
- PIS/NIS:
  - Benefit: V+ = (max τ, min ξ, min η), V- = (min τ, max ξ, max η)
  - Cost:    V+ = (min τ, max ξ, max η), V- = (max τ, min ξ, min η)
- Separation uses **normalized PN–Euclidean distance** with factor 1/(3n), summed over criteria.
- Rank by relative closeness **Pᵢ = Sᵢ⁻ / (Sᵢ⁺ + Sᵢ⁻)**.
"""
    )

# Sidebar
st.sidebar.header("Settings")
scale = st.sidebar.selectbox("Select linguistic scale", options=[5, 7, 9, 11], index=2)
decimals = st.sidebar.slider("Triplet display decimals", 2, 6, 2)

st.sidebar.subheader("Upload")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

st.sidebar.subheader("Weights")
weight_mode = st.sidebar.radio("Criteria weights", ["Equal Weights", "Manual Weights"], index=0)

# Linguistic table reference
table_df = pd.DataFrame(
    [{"Score": k, "τ": v[0], "ξ": v[1], "η": v[2]} for k, v in PNS_TABLES[scale].items()]
).sort_values("Score")


# -----------------------------
# Input
# -----------------------------
st.subheader("1) Data input")


def read_uploaded_file(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


raw_df: Optional[pd.DataFrame] = None

if uploaded is not None:
    try:
        raw_df = read_uploaded_file(uploaded)
        st.success(f"Loaded file: {uploaded.name}")
        st.dataframe(raw_df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.info("No file uploaded. Use the manual input grid below (optional).")
    m = st.number_input("Number of alternatives (m)", min_value=2, max_value=500, value=5, step=1)
    n = st.number_input("Number of criteria (n)", min_value=2, max_value=200, value=4, step=1)
    default_grid = pd.DataFrame(np.ones((m, n), dtype=int), columns=[f"C{j+1}" for j in range(n)])
    raw_df = st.data_editor(default_grid, use_container_width=True, key="manual_grid")

if raw_df is None or raw_df.shape[0] == 0:
    st.error("Empty input.")
    st.stop()

df = raw_df.copy()

# Decide if first column is alternative names
first_col_values = df.iloc[:, 0].tolist()
first_col_is_alt = False
try:
    _ = [int(str(v).strip()) for v in first_col_values[: min(10, len(first_col_values))]]
    first_col_is_alt = False
except Exception:
    first_col_is_alt = True

if first_col_is_alt:
    alt_names = df.iloc[:, 0].astype(str).tolist()
    df_mat = df.iloc[:, 1:].copy()
    crit_names = [str(c) for c in df_mat.columns]
else:
    alt_names = [f"A{i+1}" for i in range(df.shape[0])]
    df_mat = df.copy()
    crit_names = [str(c) for c in df_mat.columns]

# Detect B/C row in first row of matrix-part
first_row = df_mat.iloc[0, :].tolist()
has_bc = is_bc_row(first_row)

if has_bc:
    detected_types = [str(x).strip().upper() for x in first_row]
    df_scores = df_mat.iloc[1:, :].copy()
    alt_names = alt_names[1:]
    crit_types = detected_types
    st.info("Detected criterion types row (top row) in the uploaded file.")
else:
    df_scores = df_mat.copy()
    crit_types = ["B"] * len(crit_names)  # default, user can change
    st.warning("No criterion types row found. Defaulting all criteria to Benefit (B). Please adjust below before computing.")

# Criterion types editor
st.subheader("2) Criterion types (Benefit/Cost)")
type_df = pd.DataFrame([crit_types], columns=crit_names, index=["Type (B/C)"])
edited_type_df = st.data_editor(type_df, use_container_width=True, key="crit_types_editor")
crit_types = [str(edited_type_df.iloc[0, j]).strip().upper() for j in range(len(crit_names))]

if any(t not in {"B", "C"} for t in crit_types):
    st.error("Criterion types must be only 'B' or 'C' for every criterion. Fix them above, then continue.")
    st.stop()

# Crisp matrix
try:
    crisp_df = df_scores.copy()
    crisp_df = crisp_df.loc[:, ~crisp_df.columns.astype(str).str.contains("^Unnamed")]
    crisp_df.columns = crit_names[: crisp_df.shape[1]]
    crisp_df = crisp_df.reset_index(drop=True)

    crisp_int = coerce_int_matrix(crisp_df)
    validate_score_range(crisp_int, scale)
except Exception as e:
    st.error(f"Input validation error: {e}")
    st.stop()

st.subheader("3) Crisp decision matrix (validated)")
crisp_show = crisp_int.copy()
crisp_show.index = alt_names
st.dataframe(crisp_show, use_container_width=True)

# Weights
st.subheader("4) Criteria weights")
n_criteria = len(crit_names)

if weight_mode == "Equal Weights":
    w = np.array([1.0 / n_criteria] * n_criteria, dtype=float)
    st.info(f"Using equal weights: each wⱼ = 1/{n_criteria} = {1.0/n_criteria:.6f}")
    w_df = pd.DataFrame([w], columns=crit_names, index=["w"])
    st.dataframe(w_df, use_container_width=True)
else:
    w_default = pd.DataFrame([[round(1.0 / n_criteria, 6)] * n_criteria], columns=crit_names, index=["w"])
    w_edit = st.data_editor(w_default, use_container_width=True, key="weights_editor")
    try:
        w = np.array([float(w_edit.iloc[0, j]) for j in range(n_criteria)], dtype=float)
    except Exception as e:
        st.error(f"Manual weights must be numeric. ({e})")
        st.stop()

    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum):
        st.error("Manual weights contain invalid numbers.")
        st.stop()

    if abs(w_sum - 1.0) > 1e-3:
        st.warning(
            f"Sum of weights = {w_sum:.6f} (should be 1.000000). "
            "Computation will proceed anyway (no auto-normalization)."
        )

# Linguistic table reference
st.subheader("Reference: Selected PNS linguistic table")
st.dataframe(table_df, use_container_width=True)

# Compute
st.subheader("5) Compute PNTOPSIS ranking")
run = st.button("Run PNTOPSIS", type="primary")
if not run:
    st.stop()

# Step A: Mapping
tau, xi, eta = map_crisp_to_pns(crisp_int, scale)

# Step B: Normalization
try:
    tau_n, xi_n, eta_n = normalize_pns(tau, xi, eta, crit_types)
except Exception as e:
    st.error(f"Normalization error: {e}")
    st.stop()

# Step C: Weighting
tau_w, xi_w, eta_w = apply_weights(tau_n, xi_n, eta_n, w)

# Step D: Ideals
tau_p, xi_p, eta_p, tau_neg, xi_neg, eta_neg = compute_ideals(tau_w, xi_w, eta_w, crit_types)

# Step E: Distances + closeness
m_alt = tau_w.shape[0]
S_plus = np.zeros(m_alt, dtype=float)
S_minus = np.zeros(m_alt, dtype=float)

for i in range(m_alt):
    S_plus[i] = pn_euclidean_distance_over_criteria(
        tau_w[i, :], xi_w[i, :], eta_w[i, :], tau_p, xi_p, eta_p
    )
    S_minus[i] = pn_euclidean_distance_over_criteria(
        tau_w[i, :], xi_w[i, :], eta_w[i, :], tau_neg, xi_neg, eta_neg
    )

Pi = S_minus / (S_plus + S_minus)

result = pd.DataFrame(
    {"S_i_plus": S_plus, "S_i_minus": S_minus, "P_i": Pi},
    index=alt_names,
)
result["Rank"] = (-result["P_i"]).rank(method="dense").astype(int)
result = result.sort_values(["Rank", "P_i"], ascending=[True, False])

# Outputs
st.subheader("Outputs")

colA, colB = st.columns(2)
with colA:
    st.markdown("### Converted PNS matrix (numeric triplets)")
    pns_df = format_triplets(tau, xi, eta, decimals=decimals)
    pns_df.columns = crit_names
    pns_df.index = alt_names
    st.dataframe(pns_df, use_container_width=True)

with colB:
    st.markdown("### Normalized PNS matrix (numeric triplets)")
    norm_df = format_triplets(tau_n, xi_n, eta_n, decimals=decimals)
    norm_df.columns = crit_names
    norm_df.index = alt_names
    st.dataframe(norm_df, use_container_width=True)

st.markdown("### Weighted normalized PNS matrix (numeric triplets)")
w_df2 = format_triplets(tau_w, xi_w, eta_w, decimals=decimals)
w_df2.columns = crit_names
w_df2.index = alt_names
st.dataframe(w_df2, use_container_width=True)

st.markdown("### PIS (V⁺) and NIS (V⁻) per criterion")
pis = pd.DataFrame({"τ+": tau_p, "ξ+": xi_p, "η+": eta_p}, index=crit_names)
nis = pd.DataFrame({"τ-": tau_neg, "ξ-": xi_neg, "η-": eta_neg}, index=crit_names)
pisnis = pd.concat([pis, nis], axis=1)
st.dataframe(pisnis, use_container_width=True)

st.markdown("### Distances, closeness, and ranking")
st.dataframe(result, use_container_width=True)

st.markdown("### Closeness chart (Pᵢ)")
st.bar_chart(result["P_i"])

# Export
st.subheader("Export")
meta = pd.DataFrame({"Criterion": crit_names, "Type (B/C)": crit_types, "Weight": w})

sheets = {
    "Crisp_Matrix": crisp_show,
    "PNS_Matrix": pns_df,
    "Normalized": norm_df,
    "Weighted_Normalized": w_df2,
    "PIS_NIS": pisnis,
    "Results": result,
    "Meta": meta.set_index("Criterion"),
    "Linguistic_Table": table_df.set_index("Score"),
}

xlsx_bytes = to_excel_bytes(sheets)
st.download_button(
    "Download results (Excel)",
    data=xlsx_bytes,
    file_name="pntopsis_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
