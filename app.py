import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress, kendalltau

st.set_page_config(page_title="MonsoonLens | Earth Trend Detective", page_icon="🌏", layout="wide")

LOCATIONS = {
    "Kochi, Kerala": (9.9312, 76.2673),
    "Thiruvananthapuram, Kerala": (8.5241, 76.9366),
    "Kozhikode, Kerala": (11.2588, 75.7804),
    "Bengaluru, Karnataka": (12.9716, 77.5946),
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "New Delhi": (28.6139, 77.2090),
}

st.markdown("""
<style>
.stApp {background: #07111d; color: #e8f1fa;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
.block-container {padding-top: 1.8rem; max-width: 1450px;}
h1,h2,h3 {color:#edf6ff;}
.eyebrow {color:#56d6c7; letter-spacing:.18em; font-size:.75rem; font-weight:700;}
.subtle {color:#91a6ba;}
.metric-card {background:linear-gradient(135deg,#102337,#0b1a29);border:1px solid #20394e;border-radius:14px;padding:18px 20px;min-height:112px;}
.metric-label {color:#8fa9bd;font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;}
.metric-value {color:#f3f8fc;font-size:1.8rem;font-weight:700;margin-top:7px;}
.metric-note {color:#83a0b5;font-size:.78rem;margin-top:3px;}
section[data-testid="stSidebar"] {background:#0a1826;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">EARTH SYSTEM TREND DETECTIVE · NASA POWER</div>', unsafe_allow_html=True)
st.title("🌏 MonsoonLens")
st.markdown('<p class="subtle">Investigating long-term temperature patterns across India using NASA Earth observation and model data.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Investigation")
    place = st.selectbox("Study location", list(LOCATIONS.keys()))
    start_year = st.selectbox("Start year", [2001, 2005, 2010], index=0)
    end_year = st.selectbox("End year", [2023, 2024, 2025], index=2)
    refresh = st.button("↻ Refresh NASA data", use_container_width=True)
    st.divider()
    st.caption("DATA SOURCE")
    st.markdown("[NASA POWER Monthly API](https://power.larc.nasa.gov/docs/services/api/temporal/monthly/)")
    st.caption("Parameter: T2M · Community: RE")

lat, lon = LOCATIONS[place]
if start_year >= end_year:
    st.error("End year must be later than start year.")
    st.stop()

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(latitude, longitude, first_year, last_year):
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"
    params = {
        "parameters": "T2M",
        "community": "RE",
        "longitude": longitude,
        "latitude": latitude,
        "start": first_year,
        "end": last_year,
        "format": "JSON",
    }
    response = requests.get(url, params=params, timeout=45)
    response.raise_for_status()
    payload = response.json()
    values = payload["properties"]["parameter"]["T2M"]
    rows = []
    for date_key, value in values.items():
        if value is None or float(value) <= -900:
            continue
        # NASA POWER date keys may include a trailing day/metadata digit.
        # Monthly records are identified by the first six YYYYMM digits.
        date_token = str(date_key).strip()
        if len(date_token) < 6 or not date_token[:6].isdigit():
            continue
        month_date = pd.to_datetime(date_token[:6], format="%Y%m", errors="coerce")
        if pd.isna(month_date):
            continue
        rows.append({"date": month_date, "temperature": float(value)})
    df = pd.DataFrame(rows).sort_values("date")
    if df.empty:
        raise ValueError("NASA POWER returned no valid observations for this request.")
    return df

if refresh:
    fetch_data.clear()

with st.spinner("Retrieving monthly observations from NASA POWER…"):
    try:
        monthly = fetch_data(lat, lon, start_year, end_year)
    except Exception as exc:
        st.error(f"Could not retrieve NASA POWER data: {exc}")
        st.info("Check your internet connection and try refreshing. NASA POWER may occasionally be unavailable.")
        st.stop()

monthly["year"] = monthly["date"].dt.year
annual = monthly.groupby("year", as_index=False).agg(
    annual_mean=("temperature", "mean"),
    months_available=("temperature", "count")
)
annual = annual[annual["months_available"] >= 10].copy()
if len(annual) < 5:
    st.error("Not enough complete annual observations to estimate a reliable trend.")
    st.stop()

x = annual["year"].to_numpy(dtype=float)
y = annual["annual_mean"].to_numpy(dtype=float)
fit = linregress(x, y)
tau = kendalltau(x, y, nan_policy="omit")
slope_decade = fit.slope * 10
change = y[-1] - y[0]
latest_year = int(annual.iloc[-1]["year"])
latest_temp = float(annual.iloc[-1]["annual_mean"])
pvalue = float(tau.pvalue) if np.isfinite(tau.pvalue) else np.nan
significant = bool(pvalue < 0.05) if np.isfinite(pvalue) else False

def metric(label, value, note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1: metric("Latest annual mean", f"{latest_temp:.2f} °C", f"Calendar year {latest_year}")
with m2: metric("Observed change", f"{change:+.2f} °C", f"{int(x[0])}–{latest_year}; first vs last year")
with m3: metric("Linear trend rate", f"{slope_decade:+.3f} °C", "per decade · least-squares slope")
with m4: metric("Kendall trend p-value", f"{pvalue:.4f}" if np.isfinite(pvalue) else "N/A", "Significant at 0.05" if significant else "Not significant at 0.05")

st.markdown("")
left, right = st.columns([2.2, 1], gap="large")
with left:
    st.subheader("Annual temperature trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=annual["year"], y=annual["annual_mean"], mode="lines+markers",
        name="Annual mean", line=dict(color="#56d6c7", width=3),
        marker=dict(size=6), hovertemplate="%{x}: %{y:.2f} °C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=annual["year"], y=fit.intercept + fit.slope*x, mode="lines",
        name="Linear trend", line=dict(color="#f6b85c", width=2, dash="dash"),
        hovertemplate="Trend: %{y:.2f} °C<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark", height=390, margin=dict(l=10,r=10,t=20,b=10),
        paper_bgcolor="#07111d", plot_bgcolor="#0b1a29",
        xaxis_title="Year", yaxis_title="Annual mean 2 m air temperature (°C)",
        legend=dict(orientation="h", y=1.12), hovermode="x unified"
    )
    fig.update_xaxes(gridcolor="#1b3042")
    fig.update_yaxes(gridcolor="#1b3042")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Trend assessment")
    if significant:
        st.success("Statistically significant monotonic trend detected (p < 0.05).")
    else:
        st.warning("No statistically significant monotonic trend detected at p < 0.05.")
    direction = "upward" if slope_decade > 0 else "downward" if slope_decade < 0 else "approximately flat"
    st.markdown(f"**Direction:** {direction.title()}")
    st.markdown(f"**Kendall's τ:** {tau.statistic:.3f}")
    st.markdown(f"**Regression R²:** {fit.rvalue**2:.3f}")
    st.markdown(f"**Observations:** {len(annual)} annual means")
    st.markdown(f"**Coverage:** {int(x[0])}–{latest_year}")
    st.caption("The p-value tests evidence for a monotonic association over time. It does not establish causation.")

st.subheader("What does the data suggest?")
if significant:
    significance_text = "Kendall's rank test indicates a statistically significant monotonic association with time in this series."
else:
    significance_text = "Kendall's rank test does not find sufficient evidence of a monotonic association with time at the 5% significance level."
if slope_decade > 0:
    direction_text = f"The fitted linear trend is upward at approximately {slope_decade:.3f} °C per decade."
elif slope_decade < 0:
    direction_text = f"The fitted linear trend is downward at approximately {abs(slope_decade):.3f} °C per decade."
else:
    direction_text = "The fitted linear trend is approximately flat."
st.info(f"For {place}, the annual mean 2 m air temperature changed by {change:+.2f} °C between the first and last available years. {direction_text} {significance_text} This is a point-based analysis and should not be interpreted as a complete regional climate assessment.")

with st.expander("Methodology, limitations & data provenance"):
    st.markdown("""
    **Data:** NASA POWER Monthly Point API, parameter `T2M` (temperature at 2 metres), community `RE`. The meteorological data in NASA POWER are derived from the MERRA-2 assimilation system.

    **Processing:** Monthly means are averaged into calendar-year means. Years with fewer than 10 valid monthly observations are excluded.

    **Linear trend:** Ordinary least-squares regression of annual mean temperature against calendar year. The slope is multiplied by 10 to express °C per decade.

    **Monotonic trend check:** Kendall's rank correlation between year and annual mean temperature is used as a non-parametric association test. The p-value threshold is 0.05.

    **Limitations:** This is a single coordinate, not a city-wide measurement. Annual averaging can hide seasonal extremes. Autocorrelation and changes in observing/model systems can affect significance. Statistical association does not establish a cause of warming or cooling. NASA POWER gridded/model-derived values are not a substitute for a local weather station.
    """)
st.caption("Source: NASA POWER · Data accessed live by this application · Built for the Earth System Trend Detective challenge")
st.download_button(
    "Download annual results (CSV)",
    data=annual.to_csv(index=False).encode("utf-8"),
    file_name="monsoonlens_annual_temperature.csv",
    mime="text/csv"
)
