import json
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from tensorflow.keras.models import load_model


# ---------- Helpers ----------
def build_window_ending_at(X_scaled: np.ndarray, end_idx: int, W: int) -> np.ndarray:
    """Return a (W, D) window ending at `end_idx` (inclusive)."""
    return X_scaled[end_idx - W + 1: end_idx + 1, :]


def mse_for_window(model, window_scaled: np.ndarray) -> float:
    """Compute MSE for a window using the trained autoencoder."""
    W, D = window_scaled.shape
    x = window_scaled.reshape(1, W * D)
    recon = model.predict(x, verbose=0).reshape(W, D)
    return float(((recon - window_scaled) ** 2).mean())


def feature_contrib_for_window(model, window_scaled: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Returns per-feature absolute error averaged across time in the window."""
    W, D = window_scaled.shape
    x = window_scaled.reshape(1, W * D)
    recon = model.predict(x, verbose=0).reshape(W, D)
    per_feat_abs = np.abs(recon - window_scaled).mean(axis=0)
    total = per_feat_abs.sum() + 1e-12
    contrib = per_feat_abs / total
    return pd.DataFrame({"feature": feature_names, "contribution": contrib}).sort_values("contribution", ascending=False)


def fit_percentile_mapping(model, X_scaled: np.ndarray, W: int, max_windows: int = 3000) -> Tuple[np.ndarray, float, float]:
    """Compute baseline error distribution for percentile mapping."""
    n = len(X_scaled)
    if n < W:
        return np.array([1e-6]), 1e-6, 1e-6
    end_indices = list(range(W - 1, n))[:max_windows]
    errors = []
    for idx in end_indices:
        w = build_window_ending_at(X_scaled, idx, W)
        errors.append(mse_for_window(model, w))
    errors = np.array(errors)
    errors += np.random.uniform(1e-8, 1e-6, size=errors.shape)
    return errors, float(errors.min()), float(errors.max())


def percentile_score(mse_value: float, baseline_errors: np.ndarray) -> float:
    """Convert MSE to percentile score."""
    return float((baseline_errors <= mse_value).mean() * 100.0)


# ---------- App ----------
st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("📊 Anomaly Detection Dashboard (Windowed MLP Autoencoder)")

# Load metadata, scaler, model, and output CSV
meta_path = "models/metadata.json"
model_path = "models/autoencoder_model.h5"
scaler_path = "models/scaler.pkl"
csv_path = "data/output_with_anomalies.csv"

if not (os.path.exists(meta_path) and os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(csv_path)):
    st.error("Required files not found. Please run the pipeline with --save_model first.")
    st.stop()

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

W = int(meta["window"])
feature_names = meta["feature_names"]

df = pd.read_csv(csv_path)
scaler = joblib.load(scaler_path)
model = load_model(model_path)

# Prepare numeric matrix
X = df[feature_names].copy()
X = X.ffill().bfill().to_numpy()
X_scaled = scaler.transform(X)

# Sidebar controls
st.sidebar.header("Controls")
show_last_n = st.sidebar.slider("Show last N rows in trend", min_value=100, max_value=len(df), value=min(len(df), 2000), step=50)
risk_bands = st.sidebar.checkbox("Show risk bands", value=True)

# ---------- 1) Anomaly Trend ----------
st.subheader("1) Live Anomaly Trend")
trend = df[["anomaly_score_0_100"]].copy()
trend.index = pd.RangeIndex(len(trend))
trend_tail = trend.tail(show_last_n)

fig = px.line(trend_tail, y="anomaly_score_0_100", title="Anomaly Score Over Time")
if risk_bands:
    fig.add_hrect(y0=0, y1=40, fillcolor="green", opacity=0.08, line_width=0)
    fig.add_hrect(y0=40, y1=70, fillcolor="yellow", opacity=0.08, line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.08, line_width=0)
st.plotly_chart(fig, use_container_width=True)

# ---------- 2) Feature Importance ----------
st.subheader("2) Feature Importance (per selected time)")
max_end = len(df) - 1
end_idx = st.slider("Select time index (end of window)", min_value=W - 1, max_value=max_end, value=max_end, step=1)

window_scaled = build_window_ending_at(X_scaled, end_idx, W)
contrib_df = feature_contrib_for_window(model, window_scaled, feature_names).head(15)
fig_bar = px.bar(contrib_df, x="feature", y="contribution", title=f"Top Feature Contributions at index {end_idx}")
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---------- 3) What-If Analysis ----------
st.subheader("3) What-If Analysis (tweak the last row in the selected window)")
st.write("Adjust the features for the **last row** in the selected window to recompute anomaly score & contributions.")

baseline_errors, _, _ = fit_percentile_mapping(model, X_scaled[: max(3*W, end_idx+1)], W)
last_row_original = X[end_idx, :].copy()

custom_values = []
for i, feat in enumerate(feature_names):
    vmin = float(np.nanmin(X[:, i]))
    vmax = float(np.nanmax(X[:, i]))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0
    custom_values.append(st.slider(feat, vmin, vmax, float(last_row_original[i])))

custom_values = np.array(custom_values, dtype=float)

# Apply What-If change
window_scaled_whatif = window_scaled.copy()
last_row_scaled = scaler.transform(custom_values.reshape(1, -1)).reshape(-1)
window_scaled_whatif[-1, :] = last_row_scaled

# Compute scores
mse_base = mse_for_window(model, window_scaled)
mse_wi = mse_for_window(model, window_scaled_whatif)
score_base = percentile_score(mse_base, baseline_errors)
score_wi = percentile_score(mse_wi, baseline_errors)

# Show scores
colA, colB = st.columns(2)
with colA:
    st.metric("Original Score", f"{score_base:.2f}")
with colB:
    st.metric("What-If Score", f"{score_wi:.2f}", delta=f"{(score_wi - score_base):+.2f}")

# Risk Status (Dynamic)
st.subheader("Risk Assessment")
if "score_wi" in locals():  # What-If mode active
    current_score = score_wi
    st.write("**Current Risk (What-If):**")
else:
    latest_idx = int(trend.dropna().index[-1]) if trend.dropna().shape[0] else None
    current_score = float(trend.loc[latest_idx, "anomaly_score_0_100"]) if latest_idx is not None else 0
    st.write("**Current Risk (Dataset):**")

if current_score < 30:
    st.success(f"LOW ({current_score:.2f})")
elif current_score < 70:
    st.warning(f"MEDIUM ({current_score:.2f})")
else:
    st.error(f"CRITICAL ({current_score:.2f})")

# Feature contributions after What-If
contrib_wi = feature_contrib_for_window(model, window_scaled_whatif, feature_names).head(15)
fig_bar_wi = px.bar(contrib_wi, x="feature", y="contribution", title=f"What-If Top Contributions at index {end_idx}")
st.plotly_chart(fig_bar_wi, use_container_width=True)
