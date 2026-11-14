import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import kendalltau

st.set_page_config(
    page_title="MonsoonLens | Earth Trend Detective",
    page_icon="🌏",
    layout="wide",
)

LOCATIONS = {
    "Kochi, Kerala": (9.9312, 76.2673),
    "Thiruvananthapuram, Kerala": (8.5241, 76.9366),
    "Kozhikode, Kerala": (11.2588, 75.7804),
    "Bengaluru, Karnataka": (12.9716, 77.5946),
    "Mumbai, Maharashtra": (19.0760, 72.8777),
    "New Delhi": (28.6139, 77.2090),
}

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="eyebrow">EARTH SYSTEM TREND DETECTIVE · NASA POWER</div>',
    unsafe_allow_html=True,
)
st.title("🌏 MonsoonLens")
st.markdown(
    '<p class="subtle">Investigating long-term temperature patterns '
    "across India using NASA Earth observation and model data.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Investigation")

    place = st.selectbox(
        "Study location",
        list(LOCATIONS.keys()),
    )

    compare_places = st.multiselect(
        "Compare locations (up to 3)",
        options=list(LOCATIONS.keys()),
        default=[place],
        max_selections=3,
        help="Choose up to three locations for a side-by-side comparison.",
    )

    start_year = st.selectbox(
        "Start year",
        [2001, 2005, 2010],
        index=0,
    )

    end_year = st.selectbox(
        "End year",
        [2023, 2024, 2025],
        index=2,
    )

    refresh = st.button(
        "↻ Refresh NASA data",
        use_container_width=True,
    )

    st.divider()
    st.caption("DATA SOURCE")
    st.markdown(
        "[NASA POWER Monthly API]"
        "(https://power.larc.nasa.gov/docs/services/api/temporal/monthly/)"
    )
    st.caption("Parameters: T2M, PRECTOTCORR, RH2M · Community: RE")

lat, lon = LOCATIONS[place]

if start_year >= end_year:
    st.error("End year must be later than start year.")
    st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(
    latitude: float,
    longitude: float,
    first_year: int,
    last_year: int,
) -> pd.DataFrame:
    """Retrieve and validate monthly temperature observations from NASA POWER."""
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

    response = requests.get(
        url,
        params=params,
        timeout=45,
    )
    response.raise_for_status()

    payload = response.json()

    try:
        values = payload["properties"]["parameter"]["T2M"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "NASA POWER returned an unexpected response structure."
        ) from exc

    if not isinstance(values, dict):
        raise TypeError("NASA POWER returned an invalid temperature dataset.")

    rows = []

    for date_key, value in values.items():
        if value is None:
            continue

        try:
            temperature = float(value)
        except (ValueError, TypeError):
            continue

        if not np.isfinite(temperature) or temperature <= -900:
            continue

        # Monthly records are identified by their first six YYYYMM digits.
        date_token = str(date_key).strip()

        if len(date_token) < 6 or not date_token[:6].isdigit():
            continue

        month_date = pd.to_datetime(
            date_token[:6],
            format="%Y%m",
            errors="coerce",
        )

        if pd.isna(month_date):
            continue

        rows.append(
            {
                "date": month_date,
                "temperature": temperature,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["date", "temperature"],
    ).sort_values("date")

    if df.empty:
        raise ValueError("NASA POWER returned no valid observations for this request.")

    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_climate_data(
    latitude: float,
    longitude: float,
    first_year: int,
    last_year: int,
) -> pd.DataFrame:
    """Retrieve monthly precipitation and relative humidity from NASA POWER."""
    url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

    params = {
        "parameters": "PRECTOTCORR,RH2M",
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

    try:
        parameters = payload["properties"]["parameter"]
        precipitation_values = parameters["PRECTOTCORR"]
        humidity_values = parameters["RH2M"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "NASA POWER returned an unexpected climate-data response."
        ) from exc

    if not isinstance(precipitation_values, dict) or not isinstance(
        humidity_values, dict
    ):
        raise TypeError("NASA POWER returned invalid climate datasets.")

    rows = []

    for date_key, precipitation_value in precipitation_values.items():
        date_token = str(date_key).strip()

        if len(date_token) < 6 or not date_token[:6].isdigit():
            continue

        month_date = pd.to_datetime(
            date_token[:6],
            format="%Y%m",
            errors="coerce",
        )

        if pd.isna(month_date):
            continue

        try:
            precipitation = float(precipitation_value)
        except (ValueError, TypeError):
            precipitation = np.nan

        humidity_value = humidity_values.get(date_key)
        try:
            humidity = float(humidity_value) if humidity_value is not None else np.nan
        except (ValueError, TypeError):
            humidity = np.nan

        if not np.isfinite(precipitation) or precipitation < 0:
            precipitation = np.nan

        if not np.isfinite(humidity) or not 0 <= humidity <= 100:
            humidity = np.nan

        rows.append(
            {
                "date": month_date,
                "precipitation_rate": precipitation,
                "humidity": humidity,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["date", "precipitation_rate", "humidity"],
    ).sort_values("date")

    if df.empty:
        raise ValueError("NASA POWER returned no valid climate observations.")

    return df


def calculate_annual(monthly_data: pd.DataFrame) -> pd.DataFrame:
    data: pd.DataFrame = monthly_data.copy()
    data["year"] = data["date"].dt.year

    annual_data: pd.DataFrame = data.groupby("year", as_index=False).agg(
        annual_mean=("temperature", "mean"),
        months_available=("temperature", "count"),
    )

    result: pd.DataFrame = annual_data.loc[annual_data["months_available"] >= 10].copy()

    return result


def calculate_climate_annual(monthly_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate annual rainfall, monsoon rainfall and day-weighted humidity."""
    data = monthly_data.copy()

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["days_in_month"] = data["date"].dt.days_in_month

    # Convert monthly mean daily precipitation rate into monthly rainfall.
    data["rainfall_mm"] = data["precipitation_rate"] * data["days_in_month"]

    # Calculate day-weighted humidity.
    data["weighted_humidity"] = data["humidity"] * data["days_in_month"]

    data["humidity_days"] = data["days_in_month"].where(
        data["humidity"].notna(),
        0,
    )

    annual = data.groupby("year").agg(
        annual_rainfall=("rainfall_mm", "sum"),
        rainfall_months=("rainfall_mm", "count"),
        humidity_weighted_sum=("weighted_humidity", "sum"),
        humidity_days=("humidity_days", "sum"),
        humidity_months=("humidity", "count"),
    )

    annual["annual_humidity"] = (
        annual["humidity_weighted_sum"]
        .div(annual["humidity_days"])
        .where(annual["humidity_days"] > 0)
    )

    # Require all 12 months for annual rainfall.
    annual.loc[
        annual["rainfall_months"] != 12,
        "annual_rainfall",
    ] = np.nan

    # Require at least 10 valid months for annual humidity.
    annual.loc[
        annual["humidity_months"] < 10,
        "annual_humidity",
    ] = np.nan

    # June–September monsoon rainfall.
    monsoon_data = data.loc[data["month"].between(6, 9)]

    monsoon = monsoon_data.groupby("year").agg(
        monsoon_rainfall=("rainfall_mm", "sum"),
        monsoon_months=("rainfall_mm", "count"),
    )

    annual = annual.join(monsoon, how="left")

    # Require all four monsoon months.
    annual.loc[
        annual["monsoon_months"] != 4,
        "monsoon_rainfall",
    ] = np.nan

    return annual.reset_index()


def calculate_trend(
    years: np.ndarray,
    temperatures: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Calculate linear trend, R-squared and Kendall rank statistics.

    Returns:
        slope, intercept, r_squared, kendall_tau, p_value
    """
    if len(years) < 5 or len(temperatures) < 5:
        raise ValueError("At least five annual observations are required.")

    # NumPy ordinary least-squares regression.
    coefficients = np.polyfit(
        years,
        temperatures,
        1,
    )

    slope_value = float(coefficients[0])
    intercept_value = float(coefficients[1])

    # Calculate R-squared using Pearson correlation.
    correlation_matrix = np.corrcoef(
        years,
        temperatures,
    )

    correlation = float(correlation_matrix[0, 1])
    r_squared = correlation**2

    # Kendall rank correlation and its p-value.
    tau_result, pvalue_result = kendalltau(
        years,
        temperatures,
        nan_policy="omit",
    )

    tau_value = float(tau_result)
    pvalue = float(pvalue_result) if np.isfinite(pvalue_result) else np.nan

    return (
        slope_value,
        intercept_value,
        r_squared,
        tau_value,
        pvalue,
    )


def metric(
    label: str,
    value: str,
    note: str,
) -> None:
    """Render a dashboard metric card."""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# MAIN LOCATION DATA
# ---------------------------------------------------------

if refresh:
    fetch_data.clear()

with st.spinner("Retrieving monthly observations from NASA POWER…"):
    try:
        monthly = fetch_data(
            lat,
            lon,
            start_year,
            end_year,
        )
    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:
        st.error(f"Could not retrieve NASA POWER data: {exc}")
        st.info(
            "Check your internet connection and try refreshing. "
            "NASA POWER may occasionally be unavailable."
        )
        st.stop()

annual = calculate_annual(monthly)

if len(annual) < 5:
    st.error("Not enough complete annual observations to estimate a reliable trend.")
    st.stop()

x = np.asarray(
    annual["year"],
    dtype=np.float64,
)

y = np.asarray(
    annual["annual_mean"],
    dtype=np.float64,
)

(
    slope,
    intercept,
    r_squared,
    tau_statistic,
    pvalue,
) = calculate_trend(x, y)

slope_decade = slope * 10.0
change = float(y[-1] - y[0])

latest_year = int(x[-1])
latest_temp = float(y[-1])

significant = bool(np.isfinite(pvalue) and pvalue < 0.05)

# ---------------------------------------------------------
# MAIN DASHBOARD METRICS
# ---------------------------------------------------------

m1, m2, m3, m4 = st.columns(4)

with m1:
    metric(
        "Latest annual mean",
        f"{latest_temp:.2f} °C",
        f"Calendar year {latest_year}",
    )

with m2:
    metric(
        "Observed change",
        f"{change:+.2f} °C",
        f"{int(x[0])}–{latest_year}; first vs last year",
    )

with m3:
    metric(
        "Linear trend rate",
        f"{slope_decade:+.3f} °C",
        "per decade · least-squares slope",
    )

with m4:
    metric(
        "Kendall trend p-value",
        f"{pvalue:.4f}" if np.isfinite(pvalue) else "N/A",
        "Significant at 0.05" if significant else "Not significant at 0.05",
    )

st.markdown("")

left, right = st.columns(
    [2.2, 1],
    gap="large",
)

with left:
    st.subheader("Annual temperature trend")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=annual["year"],
            y=annual["annual_mean"],
            mode="lines+markers",
            name="Annual mean",
            line={"color": "#56d6c7", "width": 3},
            marker={"size": 6},
            hovertemplate="%{x}: %{y:.2f} °C<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=annual["year"],
            y=intercept + slope * x,
            mode="lines",
            name="Linear trend",
            line={
                "color": "#f6b85c",
                "width": 2,
                "dash": "dash",
            },
            hovertemplate="Trend: %{y:.2f} °C<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=390,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="#07111d",
        plot_bgcolor="#0b1a29",
        xaxis_title="Year",
        yaxis_title="Annual mean 2 m air temperature (°C)",
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
    )

    fig.update_xaxes(
        gridcolor="#1b3042",
        range=[start_year, end_year],
    )

    fig.update_yaxes(
        gridcolor="#1b3042",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with right:
    st.subheader("Trend assessment")

    if significant:
        st.success("Statistically significant monotonic trend detected (p < 0.05).")
    else:
        st.warning("No statistically significant monotonic trend detected at p < 0.05.")

    direction = (
        "upward"
        if slope_decade > 0
        else "downward"
        if slope_decade < 0
        else "approximately flat"
    )

    st.markdown(f"**Direction:** {direction.title()}")
    st.markdown(f"**Kendall's τ:** {tau_statistic:.3f}")
    st.markdown(f"**Regression R²:** {r_squared:.3f}")
    st.markdown(f"**Observations:** {len(annual)} annual means")
    st.markdown(f"**Coverage:** {int(x[0])}–{latest_year}")

    st.caption(
        "The p-value tests evidence for a monotonic association over time. "
        "It does not establish causation."
    )


# ---------------------------------------------------------
# AUTOMATIC INTERPRETATION
# ---------------------------------------------------------

st.subheader("What does the data suggest?")

if significant:
    significance_text = (
        "Kendall's rank test indicates a statistically significant "
        "monotonic association with time in this series."
    )
else:
    significance_text = (
        "Kendall's rank test does not find sufficient evidence of a "
        "monotonic association with time at the 5% significance level."
    )

if slope_decade > 0:
    direction_text = (
        f"The fitted linear trend is upward at approximately "
        f"{slope_decade:.3f} °C per decade."
    )
elif slope_decade < 0:
    direction_text = (
        f"The fitted linear trend is downward at approximately "
        f"{abs(slope_decade):.3f} °C per decade."
    )
else:
    direction_text = "The fitted linear trend is approximately flat."

st.info(
    f"For {place}, the annual mean 2 m air temperature changed by "
    f"{change:+.2f} °C between the first and last available years. "
    f"{direction_text} {significance_text} "
    "This is a point-based analysis and should not be interpreted "
    "as a complete regional climate assessment."
)

# ---------------------------------------------------------
# RAINFALL AND HUMIDITY ANALYSIS
# ---------------------------------------------------------

st.divider()
st.header("Rainfall and humidity patterns")

st.caption(
    "Investigate annual precipitation, monsoon rainfall "
    "and relative humidity using NASA POWER data."
)

if refresh:
    fetch_climate_data.clear()

with st.spinner("Retrieving rainfall and humidity data from NASA POWER…"):
    try:
        climate_monthly = fetch_climate_data(
            lat,
            lon,
            start_year,
            end_year,
        )
        climate_annual = calculate_climate_annual(climate_monthly)

    except (
        requests.RequestException,
        ValueError,
        TypeError,
    ) as exc:
        climate_annual = pd.DataFrame()
        st.warning(f"Could not retrieve rainfall and humidity data: {exc}")

if not climate_annual.empty:
    rainfall_data = climate_annual.dropna(subset=["annual_rainfall"])
    monsoon_data = climate_annual.dropna(subset=["monsoon_rainfall"])
    humidity_data = climate_annual.dropna(subset=["annual_humidity"])

    latest_rainfall = rainfall_data.iloc[-1] if not rainfall_data.empty else None
    latest_humidity = humidity_data.iloc[-1] if not humidity_data.empty else None

    r1, r2 = st.columns(2)

    with r1:
        if latest_rainfall is not None:
            metric(
                "Latest annual rainfall",
                f"{latest_rainfall['annual_rainfall']:,.1f} mm",
                f"Year {int(latest_rainfall['year'])}",
            )
        else:
            st.info("No complete annual rainfall total is available.")

    with r2:
        if latest_humidity is not None:
            metric(
                "Latest annual mean humidity",
                f"{latest_humidity['annual_humidity']:.1f}%",
                f"Year {int(latest_humidity['year'])}",
            )
        else:
            st.info("No annual humidity estimate is available.")

    st.subheader("Annual and monsoon rainfall")

    rainfall_fig = go.Figure()

    if not rainfall_data.empty:
        rainfall_fig.add_trace(
            go.Scatter(
                x=rainfall_data["year"],
                y=rainfall_data["annual_rainfall"],
                mode="lines+markers",
                name="Annual rainfall",
                line={"color": "#56d6c7", "width": 3},
                hovertemplate="%{x}: %{y:,.1f} mm<extra></extra>",
            )
        )

    if not monsoon_data.empty:
        rainfall_fig.add_trace(
            go.Scatter(
                x=monsoon_data["year"],
                y=monsoon_data["monsoon_rainfall"],
                mode="lines+markers",
                name="June–September rainfall",
                line={"color": "#f6b85c", "width": 2},
                hovertemplate="%{x}: %{y:,.1f} mm<extra></extra>",
            )
        )

    rainfall_fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#07111d",
        plot_bgcolor="#0b1a29",
        xaxis_title="Year",
        yaxis_title="Accumulated precipitation (mm)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )

    rainfall_fig.update_xaxes(
        gridcolor="#1b3042",
        range=[start_year, end_year],
    )
    rainfall_fig.update_yaxes(gridcolor="#1b3042")

    st.plotly_chart(rainfall_fig, use_container_width=True)

    st.subheader("Annual relative humidity")

    if not humidity_data.empty:
        humidity_fig = go.Figure(
            go.Scatter(
                x=humidity_data["year"],
                y=humidity_data["annual_humidity"],
                mode="lines+markers",
                name="Annual mean humidity",
                line={"color": "#8fa9ff", "width": 3},
                hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
            )
        )

        humidity_fig.update_layout(
            template="plotly_dark",
            height=350,
            paper_bgcolor="#07111d",
            plot_bgcolor="#0b1a29",
            xaxis_title="Year",
            yaxis_title="Relative humidity (%)",
        )

        humidity_fig.update_xaxes(
            gridcolor="#1b3042",
            range=[start_year, end_year],
        )
        humidity_fig.update_yaxes(gridcolor="#1b3042")

        st.plotly_chart(humidity_fig, use_container_width=True)
    else:
        st.info("Not enough valid humidity observations to display a trend.")

    with st.expander("View annual rainfall and humidity data"):
        st.dataframe(
            climate_annual.style.format(
                {
                    "annual_rainfall": "{:,.1f}",
                    "annual_humidity": "{:.1f}",
                    "monsoon_rainfall": "{:,.1f}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Annual rainfall requires 12 valid monthly observations. "
        "Monsoon rainfall requires all four June–September observations. "
        "Humidity requires at least 10 valid months."
    )

# ---------------------------------------------------------
# INTERACTIVE INDIA MAP
# ---------------------------------------------------------

st.divider()
st.subheader("Explore temperature trends across India")

st.caption(
    "Explore the study locations geographically. "
    "Hover over a marker to inspect its latest annual mean "
    "and estimated temperature trend."
)

map_names: list[str] = []
map_lats: list[float] = []
map_lons: list[float] = []
map_customdata: list[list[str | int | float]] = []

with st.spinner("Preparing location data for the India map…"):
    for map_place, (map_lat, map_lon) in LOCATIONS.items():
        try:
            map_monthly = fetch_data(
                map_lat,
                map_lon,
                start_year,
                end_year,
            )

            map_annual = calculate_annual(map_monthly)

            if len(map_annual) < 5:
                continue

            map_years = np.asarray(
                map_annual["year"],
                dtype=np.float64,
            )

            map_temperatures = np.asarray(
                map_annual["annual_mean"],
                dtype=np.float64,
            )

            (
                map_slope,
                _,
                _,
                _,
                _,
            ) = calculate_trend(
                map_years,
                map_temperatures,
            )

            map_names.append(map_place)
            map_lats.append(map_lat)
            map_lons.append(map_lon)

            map_customdata.append(
                [
                    int(map_years[-1]),
                    float(map_temperatures[-1]),
                    float(map_slope * 10.0),
                ]
            )

        except (
            requests.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            st.warning(f"Could not load map data for {map_place}: {exc}")

if map_names:
    marker_colors = [
        "#f6b85c"
        if name == place
        else "#56d6c7"
        if name in compare_places
        else "#7890a4"
        for name in map_names
    ]

    marker_sizes = [
        19 if name == place else 15 if name in compare_places else 12
        for name in map_names
    ]

    india_map = go.Figure(
        go.Scattergeo(
            lon=map_lons,
            lat=map_lats,
            text=map_names,
            customdata=map_customdata,
            mode="markers+text",
            textposition="top center",
            textfont={
                "color": "#e8f1fa",
                "size": 11,
            },
            marker={
                "size": marker_sizes,
                "color": marker_colors,
                "opacity": 0.95,
                "line": {
                    "color": "#edf6ff",
                    "width": 1,
                },
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Latest year: %{customdata[0]}<br>"
                "Annual mean: %{customdata[1]:.2f} °C<br>"
                "Trend: %{customdata[2]:+.3f} °C/decade"
                "<extra></extra>"
            ),
        )
    )

    india_map.update_layout(
        template="plotly_dark",
        height=620,
        margin={
            "l": 5,
            "r": 5,
            "t": 10,
            "b": 10,
        },
        paper_bgcolor="#07111d",
        geo={
            "scope": "asia",
            "projection": {
                "type": "mercator",
                "scale": 3.0,
            },
            "center": {
                "lat": 20.5,
                "lon": 79.0,
            },
            "lataxis": {
                "range": [5, 36],
            },
            "lonaxis": {
                "range": [65, 98],
            },
            "showland": True,
            "landcolor": "#102337",
            "showocean": True,
            "oceancolor": "#07111d",
            "showcountries": True,
            "countrycolor": "#476176",
            "showcoastlines": True,
            "coastlinecolor": "#476176",
            "bgcolor": "#07111d",
        },
        showlegend=False,
    )

    st.plotly_chart(
        india_map,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displayModeBar": True,
        },
    )

    st.markdown(
        """
        <span style="color:#f6b85c">●</span> Selected study location
        &nbsp;&nbsp;
        <span style="color:#56d6c7">●</span> Comparison locations
        &nbsp;&nbsp;
        <span style="color:#7890a4">●</span> Other locations
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Marker colours identify selection status, not temperature. "
        "Use the sidebar to change the selected study and comparison locations."
    )

else:
    st.warning(
        "The map could not be populated because no locations "
        "had enough valid annual observations."
    )


# ---------------------------------------------------------
# METHODOLOGY
# ---------------------------------------------------------

with st.expander("Methodology, limitations & data provenance"):
    st.markdown(
        """
**Data:** NASA POWER Monthly Point API, parameter `T2M`
(temperature at 2 metres), community `RE`. The meteorological data
in NASA POWER are derived from the MERRA-2 assimilation system.

**Processing:** Monthly means are averaged into calendar-year means.
Years with fewer than 10 valid monthly observations are excluded.

**Linear trend:** Ordinary least-squares regression of annual mean
temperature against calendar year. The slope is multiplied by 10
to express °C per decade.

**Monotonic trend check:** Kendall's rank correlation between year
and annual mean temperature is used as a non-parametric association
test. The p-value threshold is 0.05.

**Limitations:** This is a single coordinate, not a city-wide
measurement. Annual averaging can hide seasonal extremes.
Autocorrelation and changes in observing/model systems can affect
significance. Statistical association does not establish a cause
of warming or cooling. NASA POWER gridded/model-derived values are
not a substitute for a local weather station.

**Rainfall:** Monthly mean daily precipitation rate (`PRECTOTCORR`)
is multiplied by the number of calendar days in each month to estimate
monthly accumulated rainfall. Annual totals are reported only when
all 12 months are valid.

**Monsoon rainfall:** June–September precipitation is accumulated
for each year. Years missing any of the four monsoon months are excluded.

**Relative humidity:** Monthly mean relative humidity at 2 metres
(`RH2M`) is averaged into annual means. Years with fewer than 10
valid months are excluded.
"""
    )

st.caption(
    "Source: NASA POWER · Data accessed live by this application · "
    "Built for the Earth System Trend Detective challenge"
)

st.download_button(
    "Download annual results (CSV)",
    data=annual.to_csv(index=False).encode("utf-8"),
    file_name="monsoonlens_annual_temperature.csv",
    mime="text/csv",
)


# ---------------------------------------------------------
# MULTI-LOCATION COMPARISON
# ---------------------------------------------------------

st.divider()
st.subheader("Compare locations")

st.caption("Compare annual temperature patterns over the same period.")

if not compare_places:
    st.info("Select at least one location in the sidebar.")

else:
    comparison_series = []
    comparison_rows = []

    for comparison_place in compare_places:
        comparison_lat, comparison_lon = LOCATIONS[comparison_place]

        try:
            comparison_monthly = fetch_data(
                comparison_lat,
                comparison_lon,
                start_year,
                end_year,
            )

            yearly = calculate_annual(comparison_monthly)

            if len(yearly) < 5:
                st.warning(f"Not enough annual observations for {comparison_place}.")
                continue

            x_values = np.asarray(
                yearly["year"],
                dtype=np.float64,
            )

            y_values = np.asarray(
                yearly["annual_mean"],
                dtype=np.float64,
            )

            (
                comparison_slope,
                _,
                _,
                _,
                comparison_pvalue,
            ) = calculate_trend(
                x_values,
                y_values,
            )

            yearly["location"] = comparison_place
            comparison_series.append(yearly)

            comparison_rows.append(
                {
                    "Location": comparison_place,
                    "Trend (°C/decade)": comparison_slope * 10.0,
                    "Observed change (°C)": float(y_values[-1] - y_values[0]),
                    "Latest mean (°C)": float(y_values[-1]),
                    "Kendall p-value": comparison_pvalue,
                    "Coverage": (f"{int(x_values[0])}–{int(x_values[-1])}"),
                }
            )

        except (
            requests.RequestException,
            ValueError,
            TypeError,
        ) as exc:
            st.warning(f"Could not analyse {comparison_place}: {exc}")

    if comparison_series:
        comparison_data = pd.concat(
            comparison_series,
            ignore_index=True,
        )

        comparison_fig = go.Figure()

        for comparison_place in compare_places:
            location_data = comparison_data[
                comparison_data["location"] == comparison_place
            ]

            if location_data.empty:
                continue

            comparison_fig.add_trace(
                go.Scatter(
                    x=location_data["year"],
                    y=location_data["annual_mean"],
                    mode="lines+markers",
                    name=comparison_place,
                    hovertemplate=(
                        comparison_place
                        + "<br>Year: %{x}"
                        + "<br>Temperature: %{y:.2f} °C"
                        + "<extra></extra>"
                    ),
                )
            )

        comparison_fig.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="#07111d",
            plot_bgcolor="#0b1a29",
            xaxis_title="Year",
            yaxis_title="Annual mean temperature (°C)",
            hovermode="x unified",
        )

        comparison_fig.update_xaxes(
            gridcolor="#1b3042",
            range=[start_year, end_year],
        )

        comparison_fig.update_yaxes(
            gridcolor="#1b3042",
        )

        st.plotly_chart(
            comparison_fig,
            use_container_width=True,
        )

        st.subheader("Location statistics")

        comparison_table = pd.DataFrame(comparison_rows)

        st.dataframe(
            comparison_table.style.format(
                {
                    "Trend (°C/decade)": "{:+.3f}",
                    "Observed change (°C)": "{:+.2f}",
                    "Latest mean (°C)": "{:.2f}",
                    "Kendall p-value": "{:.4f}",
                },
                na_rep="N/A",
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Each location is analysed independently. The Kendall "
            "p-value measures rank association with time; it does "
            "not establish causation. Compare trends as well as "
            "absolute temperatures."
        )
    else:
        st.warning(
            "No locations had enough valid annual observations to display a comparison."
        )
