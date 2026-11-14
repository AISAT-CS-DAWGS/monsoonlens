# MonsoonLens

A small NASA POWER temperature-trend investigation dashboard for the Earth System Trend Detective challenge.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app fetches monthly T2M values from the NASA POWER Monthly Point API when opened. It computes annual means, an ordinary least-squares trend in °C/decade, and a Kendall rank trend significance check.

## Data source
https://power.larc.nasa.gov/docs/services/api/temporal/monthly/

## Notes
- Default location: Kochi, Kerala (9.9312, 76.2673)
- This is a point-based model/reanalysis product, not a city-wide station observation.
- Statistical significance is not causation.
