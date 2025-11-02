# streamlit_app/pages/3_STL_and_Spectrogram.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from scipy.signal import spectrogram
from pathlib import Path

st.set_page_config(page_title="STL & Spectrogram", layout="wide")

# ---------- Data load ----------
APP_ROOT = Path(__file__).resolve().parents[1]   # .../streamlit_app
ELHUB_CSV = APP_ROOT / "elhub_prod_snapshot.csv"

@st.cache_data(show_spinner=False)
def load_elhub(path: Path = ELHUB_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["startTime"])
    df = df.sort_values("startTime")
    return df[["priceArea", "productionGroup", "startTime", "quantityKwh"]]

elhub = load_elhub()

# Defaults from area selector on Page 2 (if present)
default_area = st.session_state.get("selected_area", "NO1")

tab_stl, tab_spec = st.tabs(["STL decomposition", "Spectrogram"])

# ---------- Tab 1: STL ----------
with tab_stl:
    st.subheader("Seasonal-Trend decomposition using LOESS (STL)")

    area = st.selectbox(
        "Price area",
        sorted(elhub["priceArea"].unique()),
        index=sorted(elhub["priceArea"].unique()).index(default_area),
    )
    groups = sorted(elhub.loc[elhub["priceArea"] == area, "productionGroup"].unique())
    group = st.selectbox("Production group", groups)

    col1, col2, col3, col4 = st.columns(4)
    period   = col1.number_input("Period (hours)", 24, 24*60, 24*7, step=24)
    seasonal = col2.slider("Seasonal smoother", 7, 61, 13, step=2)
    trend    = col3.slider("Trend smoother", 51, 601, 301, step=10)
    robust   = col4.checkbox("Robust", value=True)

    s = (
        elhub[(elhub.priceArea == area) & (elhub.productionGroup == group)]
        .set_index("startTime")["quantityKwh"]
        .asfreq("H")
        .interpolate()
    )

    if s.empty:
        st.info("No data for the selected combination.")
    else:
        res = STL(s, period=int(period), seasonal=int(seasonal), trend=int(trend), robust=robust).fit()
        fig = res.plot()

        # Replace verbose auto titles and remove y-axis labels to avoid duplicate headings
        tidy_titles = ["Observed", "Trend", "Seasonal", "Resid"]
        for ax, ttl in zip(fig.axes, tidy_titles):
            ax.set_title(ttl, fontsize=12, pad=6)
            ax.set_ylabel("")  # remove left-side headings
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.tick_params(axis="x", labelsize=9)
            ax.tick_params(axis="y", labelsize=9)

        # Limit and tidy the bottom (residual) x-axis tick labels
        ax_bottom = fig.axes[-1]
        loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
        fmt = mdates.ConciseDateFormatter(loc)
        ax_bottom.xaxis.set_major_locator(loc)
        ax_bottom.xaxis.set_major_formatter(fmt)
        ax_bottom.tick_params(axis="x", labelsize=9, rotation=0, pad=2)

        # Space so UI/expander don’t collide
        fig.suptitle(f"STL — area={area}, group={group}, period={period}", y=1.02, fontsize=14)
        fig.subplots_adjust(top=0.90, bottom=0.18, hspace=0.40)

        st.pyplot(fig, use_container_width=True, clear_figure=True)

# ---------- Tab 2: Spectrogram ----------
with tab_spec:
    st.subheader("Spectrogram (hourly production)")

    area2 = st.selectbox(
        "Price area (spectrogram)",
        sorted(elhub["priceArea"].unique()),
        index=sorted(elhub["priceArea"].unique()).index(default_area),
        key="sg_area",
    )
    groups2 = sorted(elhub.loc[elhub["priceArea"] == area2, "productionGroup"].unique())
    group2 = st.selectbox("Production group (spectrogram)", groups2, key="sg_group")

    colA, colB = st.columns(2)
    window_len = colA.number_input("Window length (hours)", 24, 24*60, 24*14, step=24)
    overlap    = colB.slider("Window overlap", 0.0, 0.9, 0.5, 0.05)

    s2 = (
        elhub[(elhub.priceArea == area2) & (elhub.productionGroup == group2)]
        .set_index("startTime")["quantityKwh"]
        .asfreq("H")
        .interpolate()
    )

    if s2.empty:
        st.info("No data for the selected combination.")
    else:
        nperseg = int(window_len)
        noverlap = int(nperseg * float(overlap))
        f, tt, Sxx = spectrogram(s2.values, fs=1.0, nperseg=nperseg, noverlap=noverlap, scaling="density")

        # Use only constrained layout (avoid tight_layout to prevent engine clash)
        fig2, ax2 = plt.subplots(figsize=(10, 4), constrained_layout=True)
        im = ax2.pcolormesh(tt, f, 10*np.log10(Sxx + 1e-12), shading="auto")
        ax2.set_title(f"Spectrogram — area={area2}, group={group2}", pad=6)
        ax2.set_xlabel("Window index")
        ax2.set_ylabel("Frequency (cycles/hour)")
        fig2.colorbar(im, ax=ax2, label="Power (dB)")
        st.pyplot(fig2, use_container_width=True, clear_figure=True)

st.expander("Data source").write(
    "Production: Elhub snapshot (hourly). Parameters exposed per assignment: "
    "price area, production group, period length, seasonal smoother, trend smoother, robust flag; "
    "and for spectrogram: window length, overlap."
)
