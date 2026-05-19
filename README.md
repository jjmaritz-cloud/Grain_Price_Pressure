# El Niño + Grain Price Pressure Tracker

This Streamlit app plots:

- Observed Niño3.4 values
- An automatic 6-month ENSO trend projection
- BOM-style El Niño and La Niña thresholds
- A Higgins-style very strong El Niño warning zone
- A grain price pressure index on a second y-axis

## Files

- `streamlit_app.py` - main app
- `requirements.txt` - Python packages required
- `run_el_nino_app.bat` - Windows batch file to install packages and run/restart the app

## How to run on Windows

1. Unzip this folder.
2. Double-click `run_el_nino_app.bat`.
3. The app should open in your browser.

## Batch file behaviour

When you stop Streamlit with CTRL + C, the window stays open.

After editing `streamlit_app.py`, return to the batch window and press any key to restart the app.

## Important notes

- The observed ENSO line attempts to fetch NOAA/CPC weekly Niño3.4 data.
- The forecast line is a recent-trend projection, not an official BOM ACCESS-S2 forecast.
- The grain price pressure line currently uses built-in sample data so the dashboard works immediately.
- Next step: replace the sample grain data with automatic ABARES, local delivered grain prices, or company buying data.
