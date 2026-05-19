# grain_live_forecast_openmeteo_v14_auto_refresh_90.py
# Standalone Streamlit app for farmers:
# NSW & Victoria grain buying weather outlook using Open-Meteo Seasonal Forecast API.
# Green = BUY, Amber = WATCH, Red = CAUTION.

from __future__ import annotations

import math
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import streamlit.components.v1 as components


# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="Grain Weather Buying Outlook",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    html { zoom: 90%; }
    .block-container { padding-top: 0.8rem; padding-bottom: 1.5rem; }

    /* Framed blocks make tables, source details and code-style areas easier to read. */
    div[data-testid="stExpander"] {
        border: 1px solid #dbe4ef !important;
        border-radius: 14px !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(15,23,42,0.035) !important;
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"], div[data-testid="stCodeBlock"] {
        border: 1px solid #dbe4ef !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(15,23,42,0.035) !important;
        background: #ffffff !important;
    }
    pre, code {
        border: 1px solid #dbe4ef;
        border-radius: 10px;
        background: #f8fafc;
        padding: 2px 5px;
    }
    .card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 16px 18px;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
        min-height: 124px;
    }
    .card-title { color:#64748b; font-size:0.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; }
    .card-value { color:#0f172a; font-size:1.35rem; font-weight:900; margin-top:2px; line-height:1.15; }
    .card-sub { color:#64748b; font-size:0.86rem; margin-top:6px; line-height:1.35; }
    .plain-note { color:#475569; font-size:0.92rem; line-height:1.45; }
    .source-note { color:#64748b; font-size:0.78rem; line-height:1.35; }

    .watch-section { margin-left: 45px; margin-right: 170px; margin-top: 0; }
    .watch-section-title { font-size: 1.28rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; }
    .watch-grid { display: grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 10px; align-items: stretch; width: 100%; }
    .watch-card { border: 1px solid #e2e8f0; border-radius: 14px; padding: 13px 11px; background: #ffffff; min-height: 205px; box-shadow: 0 2px 8px rgba(15,23,42,0.04); overflow: hidden; box-sizing: border-box; }
    .watch-month { font-size: 0.68rem; color: #64748b; font-weight: 900; letter-spacing: 0.11em; text-transform: uppercase; white-space: nowrap; }
    .watch-action { font-size: 0.86rem; font-weight: 900; line-height: 1.12; margin-top: 7px; min-height: 42px; }
    .watch-body { color: #475569; font-size: 0.74rem; line-height: 1.28; margin-top: 7px; }

    .ai-card { border: 1px solid #e2e8f0; border-radius: 18px; padding: 15px 16px; background: #ffffff; min-height: 142px; box-shadow: 0 2px 8px rgba(15,23,42,0.04); }
    .ai-card-title { font-size: 0.75rem; color: #64748b; font-weight: 900; text-transform: uppercase; letter-spacing: 0.08em; }
    .ai-card-value { font-size: 1.25rem; font-weight: 900; margin-top: 8px; line-height: 1.15; }
    .ai-card-body { color: #475569; font-size: 0.86rem; line-height: 1.35; margin-top: 8px; }

    @media (max-width: 1250px) {
        .watch-section { margin-left: 10px; margin-right: 10px; }
        .watch-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) { .watch-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# COLOURS / SETTINGS
# -----------------------------
BUY = "#16a34a"
WATCH = "#f59e0b"
CAUTION = "#dc2626"
BLUE = "#2563eb"
ORANGE = "#ea580c"
SLATE = "#334155"
GRID = "#dbe4ef"

START_DATE = "2026-01-01"
END_DATE = "2027-07-31"
EL_NINO_THRESHOLD = 0.8
LA_NINA_THRESHOLD = -0.8
OPEN_METEO_URL = "https://seasonal-api.open-meteo.com/v1/seasonal"

# Representative grain-growing areas. The app averages them into one plain-English NSW/VIC view.
DEFAULT_REGIONS = [
    {"name": "Wagga Wagga / Riverina", "lat": -35.11, "lon": 147.37},
    {"name": "Griffith / Murrumbidgee", "lat": -34.29, "lon": 146.05},
    {"name": "Dubbo / Central West NSW", "lat": -32.25, "lon": 148.60},
    {"name": "Horsham / Wimmera", "lat": -36.71, "lon": 142.20},
    {"name": "Bendigo / Central Victoria", "lat": -36.76, "lon": 144.28},
]


# -----------------------------
# BASE DATA
# -----------------------------
def build_default_monthly_data() -> pd.DataFrame:
    """Fallback planning values so the app still works if the internet/API is unavailable."""
    months = pd.date_range(START_DATE, END_DATE, freq="MS")
    enso = [
        -0.25, -0.10, 0.05, 0.25, 0.52, 0.65, 0.78, 0.90, 0.95, 1.00, 0.92, 0.75,
        0.62, 0.55, 0.48, 0.42, 0.35, 0.25, 0.15,
    ]
    grain_pressure = [
        38, 42, 47, 50, 51, 58, 65, 68, 66, 64, 62, 55,
        52, 54, 60, 64, 68, 70, 69,
    ]
    df = pd.DataFrame({
        "month": months,
        "enso_index": enso[: len(months)],
        "base_pressure": grain_pressure[: len(months)],
    })
    df["rain_message"] = "No live rain data"
    df["temp_message"] = "No live heat data"
    df["rain_adjustment"] = 0.0
    df["temp_adjustment"] = 0.0
    df["weather_adjustment"] = 0.0
    df["rain_mm_vs_normal"] = np.nan
    df["temp_c_vs_normal"] = np.nan
    df["live_weather"] = False
    return df


# -----------------------------
# LIVE OPEN-METEO WEATHER
# -----------------------------
def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def make_weather_error_plain(error_text: str) -> str:
    """Turn computer-style errors into a farmer-friendly message."""
    t = str(error_text)
    tl = t.lower()
    if "name or service not known" in tl or "nameresolution" in tl or "dns" in tl or "failed to resolve" in tl:
        return "Your computer could not find the weather service address. This is usually an internet, DNS, firewall or proxy issue."
    if "timed out" in tl or "timeout" in tl:
        return "The weather service took too long to answer. Try again later, or check the internet connection."
    if "429" in tl:
        return "The free weather service is busy or has had too many requests. Try again later."
    if "403" in tl or "401" in tl:
        return "The weather service blocked the request. This can happen on some work networks or if a paid key is needed later."
    if "invalid string value" in tl or "forecastvariablemonthly" in tl:
        return "The weather service rejected one of the weather fields. This version asks for the correct seasonal rain field: precipitation_mean, not precipitation_sum."
    if "no monthly data" in tl:
        return "The weather service answered, but did not send monthly rain and heat figures for this region."
    return "The weather service could not be read. The app is safely using built-in planning values instead."


def open_meteo_param_options(region: dict[str, Any]) -> list[list[tuple[str, Any]]]:
    """Build Open-Meteo request options.

    Important fix: the Seasonal API does not use the same monthly rain name
    as the normal daily forecast API.

    Daily forecast uses precipitation_sum.
    Seasonal monthly outlook uses precipitation_mean and precipitation_anomaly.

    We still send monthly fields as repeated fields because Open-Meteo accepts that
    format reliably:
        monthly=temperature_2m_mean&monthly=precipitation_mean
    """
    monthly_main = [
        "temperature_2m_mean",
        "temperature_2m_anomaly",
        "precipitation_mean",
        "precipitation_anomaly",
    ]
    monthly_alt = [
        "temperature_2m_mean",
        "temperature_2m_mean_anomaly",
        "precipitation_mean",
        "precipitation_anomaly",
    ]

    def build(model: str | None, monthly_vars: list[str]) -> list[tuple[str, Any]]:
        params: list[tuple[str, Any]] = [
            ("latitude", region["lat"]),
            ("longitude", region["lon"]),
            ("forecast_days", 214),  # about 7 months
            ("timezone", "Australia/Sydney"),
        ]
        if model:
            params.append(("models", model))
        for var in monthly_vars:
            params.append(("monthly", var))
        return params

    return [
        build("ecmwf_seasonal_seamless_mean", monthly_main),
        build("ecmwf_seas5_mean", monthly_main),
        build("ecmwf_seasonal_seamless", monthly_main),
        build("ecmwf_seas5", monthly_main),
        build(None, monthly_main),
        build(None, monthly_alt),
    ]


def pick_monthly_value(monthly: dict[str, Any], key_options: list[str], i: int, fallback_len: int) -> float | None:
    for key in key_options:
        values = monthly.get(key)
        if isinstance(values, list) and i < len(values):
            v = safe_float(values[i])
            if v is not None:
                return v
    return None


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_open_meteo_for_region(region: dict[str, Any]) -> pd.DataFrame:
    last_error = None

    for params in open_meteo_param_options(region):
        try:
            response = requests.get(OPEN_METEO_URL, params=params, timeout=20)
            payload = response.json()

            # Open-Meteo often returns useful error messages inside JSON.
            if isinstance(payload, dict) and payload.get("error"):
                raise ValueError(str(payload.get("reason", "The weather service rejected the request.")))

            response.raise_for_status()
            monthly = payload.get("monthly", {}) if isinstance(payload, dict) else {}
            times = monthly.get("time", [])
            if not times:
                raise ValueError("No monthly data returned")

            rows = []
            for i, t in enumerate(times):
                rows.append({
                    "region": region["name"],
                    "month": pd.to_datetime(t, errors="coerce").to_period("M").to_timestamp(),
                    # Seasonal monthly rain is a monthly average signal from the model,
                    # not an exact farm rainfall total. We use the anomaly to decide
                    # drier/wetter than normal.
                    "rain_mm": pick_monthly_value(monthly, ["precipitation_mean"], i, len(times)),
                    "rain_mm_vs_normal": pick_monthly_value(monthly, ["precipitation_anomaly"], i, len(times)),
                    "temp_c": pick_monthly_value(monthly, ["temperature_2m_mean"], i, len(times)),
                    "temp_c_vs_normal": pick_monthly_value(monthly, ["temperature_2m_anomaly", "temperature_2m_mean_anomaly"], i, len(times)),
                })

            return pd.DataFrame(rows).dropna(subset=["month"])

        except Exception as e:
            last_error = e
            continue

    raise ValueError(str(last_error) if last_error else "The live weather service could not be read.")


def classify_rain(mm_vs_normal: float | None) -> tuple[str, float, str]:
    """Return plain message, pressure adjustment, and colour."""
    if mm_vs_normal is None or pd.isna(mm_vs_normal):
        return "Rain outlook unavailable", 0.0, SLATE
    if mm_vs_normal <= -30:
        return "Much drier than normal", 10.0, CAUTION
    if mm_vs_normal <= -15:
        return "Drier than normal", 7.0, CAUTION
    if mm_vs_normal <= -5:
        return "A bit drier than normal", 3.0, WATCH
    if mm_vs_normal < 10:
        return "Close to normal rain", 0.0, SLATE
    if mm_vs_normal < 25:
        return "A bit wetter than normal", -3.0, BUY
    return "Wetter than normal", -6.0, BUY


def classify_temperature(c_vs_normal: float | None) -> tuple[str, float, str]:
    if c_vs_normal is None or pd.isna(c_vs_normal):
        return "Heat outlook unavailable", 0.0, SLATE
    if c_vs_normal >= 2.0:
        return "Much hotter than normal", 8.0, CAUTION
    if c_vs_normal >= 1.0:
        return "Hotter than normal", 5.0, CAUTION
    if c_vs_normal >= 0.4:
        return "A bit warmer than normal", 2.0, WATCH
    if c_vs_normal > -0.5:
        return "Close to normal heat", 0.0, SLATE
    return "Cooler than normal", -2.0, BUY


def fetch_regional_weather(regions: list[dict[str, Any]]) -> tuple[pd.DataFrame | None, str]:
    frames = []
    errors = []
    for region in regions:
        try:
            frames.append(fetch_open_meteo_for_region(region))
        except Exception as e:
            errors.append(f"{region['name']}: {e}")

    if not frames:
        first_error = errors[0] if errors else "No detail available"
        plain_reason = make_weather_error_plain(first_error)
        detail_lines = "\n".join([f"- {make_weather_error_plain(e)}" for e in errors[:3]])
        raw_lines = "\n".join([f"- {e}" for e in errors[:3]])
        return None, (
            "Could not reach the live weather service. Using built-in planning values.\n\n"
            f"Most likely reason: {plain_reason}\n\n"
            "What the app tried to read: Open-Meteo Seasonal Forecast for monthly rain and heat.\n\n"
            "Quick checks:\n"
            "- Make sure the PC running Streamlit has internet access.\n"
            "- Try opening seasonal-api.open-meteo.com in the same browser.\n"
            "- If you are on a work network, firewall/DNS/proxy rules may be blocking it.\n\n"
            "Region checks:\n"
            f"{detail_lines}\n\n"
            "Computer detail, if needed:\n"
            f"{raw_lines}"
        )

    raw = pd.concat(frames, ignore_index=True)
    # Average the main grain regions into one simple NSW/VIC view.
    regional = raw.groupby("month", as_index=False).agg(
        rain_mm=("rain_mm", "mean"),
        rain_mm_vs_normal=("rain_mm_vs_normal", "mean"),
        temp_c=("temp_c", "mean"),
        temp_c_vs_normal=("temp_c_vs_normal", "mean"),
        regions_used=("region", "nunique"),
    )
    regional["rain_message"] = regional["rain_mm_vs_normal"].apply(lambda x: classify_rain(x)[0])
    regional["rain_adjustment"] = regional["rain_mm_vs_normal"].apply(lambda x: classify_rain(x)[1])
    regional["rain_colour"] = regional["rain_mm_vs_normal"].apply(lambda x: classify_rain(x)[2])
    regional["temp_message"] = regional["temp_c_vs_normal"].apply(lambda x: classify_temperature(x)[0])
    regional["temp_adjustment"] = regional["temp_c_vs_normal"].apply(lambda x: classify_temperature(x)[1])
    regional["temp_colour"] = regional["temp_c_vs_normal"].apply(lambda x: classify_temperature(x)[2])
    regional["weather_adjustment"] = regional["rain_adjustment"] + regional["temp_adjustment"]
    regional["live_weather"] = True

    status = f"Live weather loaded for {int(regional['regions_used'].max())} grain regions."
    if errors:
        status += f" Some regions were skipped ({len(errors)})."
    return regional, status


def apply_live_weather_to_base(base_df: pd.DataFrame, weather_df: pd.DataFrame | None) -> pd.DataFrame:
    df = base_df.copy()
    df["grain_pressure"] = df["base_pressure"]
    if weather_df is None or weather_df.empty:
        return df

    weather_cols = [
        "month", "rain_mm", "rain_mm_vs_normal", "rain_message", "rain_adjustment", "rain_colour",
        "temp_c", "temp_c_vs_normal", "temp_message", "temp_adjustment", "temp_colour",
        "weather_adjustment", "live_weather",
    ]
    df = df.drop(columns=[c for c in weather_cols if c in df.columns and c != "month"], errors="ignore")
    df = df.merge(weather_df[weather_cols], on="month", how="left")

    for col, default in [
        ("rain_message", "No live rain data"), ("temp_message", "No live heat data"),
        ("rain_adjustment", 0.0), ("temp_adjustment", 0.0), ("weather_adjustment", 0.0),
        ("live_weather", False), ("rain_colour", SLATE), ("temp_colour", SLATE),
    ]:
        df[col] = df[col].fillna(default)

    # Live weather nudges the pressure score up or down. It does not replace market judgement.
    df["grain_pressure"] = (df["base_pressure"] + df["weather_adjustment"]).clip(20, 90)
    return df


# -----------------------------
# SEASONAL WINDOWS
# -----------------------------
SEASON_WINDOWS = [
    dict(start="2026-01-01", end="2026-02-28", label="Summer moisture / fallow", risk="Low-Med", colour="rgba(148,163,184,0.22)"),
    dict(start="2026-03-01", end="2026-05-31", label="Autumn break / sowing", risk="Medium", colour="rgba(245,158,11,0.18)"),
    dict(start="2026-06-01", end="2026-08-31", label="Establishment / biomass", risk="Medium-High", colour="rgba(234,179,8,0.22)"),
    dict(start="2026-09-01", end="2026-10-31", label="Flowering / frost risk", risk="High", colour="rgba(239,68,68,0.18)"),
    dict(start="2026-10-01", end="2026-11-30", label="Grain fill / spring finish", risk="High", colour="rgba(220,38,38,0.16)"),
    dict(start="2026-11-15", end="2027-01-31", label="Harvest / quality known", risk="Medium", colour="rgba(100,116,139,0.18)"),
    dict(start="2027-02-01", end="2027-05-31", label="2027 autumn break", risk="Medium", colour="rgba(14,165,233,0.13)"),
    dict(start="2027-06-01", end="2027-07-31", label="2027 early crop growth", risk="Medium", colour="rgba(34,197,94,0.13)"),
]

CROP_NOTES = [
    dict(date="2026-05-15", text="Autumn break sets up\nsowing confidence", y=62),
    dict(date="2026-07-15", text="Follow-up rain matters\nfor crop establishment", y=78),
    dict(date="2026-09-15", text="Flowering and frost\nrisk window", y=81),
    dict(date="2026-10-25", text="Spring finish can shift\nyield and price pressure", y=76),
    dict(date="2026-12-10", text="Harvest shows real\nquality and supply", y=66),
]


def seasonal_boost(month: pd.Timestamp) -> int:
    m = int(month.month)
    if m in (9, 10):
        return 10
    if m == 11:
        return 7
    if m in (6, 7, 8):
        return 5
    if m in (3, 4, 5):
        return 3
    return 0


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("month").reset_index(drop=True)
    if "grain_pressure" not in out.columns:
        out["grain_pressure"] = out.get("base_pressure", 50)
    out["pressure_trend"] = out["grain_pressure"].diff().fillna(0)
    out["seasonal_boost"] = out["month"].apply(seasonal_boost)
    out["caution_score"] = out["grain_pressure"] + (out["pressure_trend"] * 1.5) + out["seasonal_boost"]

    def classify(score: float) -> str:
        if score < 52:
            return "BUY"
        if score < 67:
            return "WATCH"
        return "CAUTION"

    out["signal"] = out["caution_score"].apply(classify)
    out["signal_colour"] = out["signal"].map({"BUY": BUY, "WATCH": WATCH, "CAUTION": CAUTION})
    out["signal_reason"] = np.select(
        [out["signal"].eq("BUY"), out["signal"].eq("WATCH"), out["signal"].eq("CAUTION")],
        ["Better buying window", "Keep watching rain, heat and crop reports", "Be careful: weather/crop risk is higher"],
        default="",
    )
    out["period_type"] = np.where(
        out["month"] <= pd.Timestamp.today().replace(day=1),
        "Current / past",
        "Forward view",
    )
    return out


# -----------------------------
# BUY / WAIT RECOMMENDATIONS
# -----------------------------
def crop_window_for_month(month: pd.Timestamp) -> tuple[str, str]:
    month_mid = pd.Timestamp(month).to_period("M").to_timestamp() + pd.Timedelta(days=14)
    for w in SEASON_WINDOWS:
        if pd.Timestamp(w["start"]) <= month_mid <= pd.Timestamp(w["end"]):
            return w["label"], w["risk"]
    return "Outside main crop window", "Low"


def enso_assessment(enso_index: float) -> tuple[str, str, int, str]:
    if enso_index >= 1.2:
        return "Strong dry-weather warning", CAUTION, 10, "The ocean signal is well into El Niño territory, so dry risk needs extra care."
    if enso_index >= EL_NINO_THRESHOLD:
        return "El Niño line reached", CAUTION, 7, "The ocean signal is at or above the dry-risk line."
    if enso_index >= 0.55:
        return "Close to dry-risk line", WATCH, 4, "Not over the line yet, but close enough to watch."
    if enso_index <= LA_NINA_THRESHOLD:
        return "Wet-weather signal", BUY, -4, "This can reduce dry-weather pressure, but local rainfall still matters."
    return "Neutral", SLATE, 0, "No strong El Niño or La Niña signal."


def crop_risk_assessment(risk: str, window_label: str) -> tuple[str, str, int, str]:
    risk_clean = risk.lower()
    if "high" in risk_clean and "medium" not in risk_clean:
        return "High crop-risk time", CAUTION, 9, f"{window_label} is when the crop can move the market quickly."
    if "high" in risk_clean and "medium" in risk_clean:
        return "Crop needs watching", WATCH, 6, f"{window_label} can change confidence quickly if rain does not follow up."
    if "medium" in risk_clean:
        return "Normal crop watch", WATCH, 3, f"{window_label} matters, but it is not the peak danger time."
    return "Lower crop risk", BUY, 0, f"{window_label} is usually a lower-risk buying period."


def price_pressure_assessment(grain_pressure: float, pressure_trend: float) -> tuple[str, str, int, str]:
    if grain_pressure >= 67 and pressure_trend > 0:
        return "High and rising", CAUTION, 10, "Pressure is already high and still moving up."
    if grain_pressure >= 67:
        return "High", CAUTION, 8, "Pressure is high, so avoid panic buying unless cover is short."
    if grain_pressure >= 52 and pressure_trend > 0:
        return "Rising", WATCH, 5, "Not critical yet, but moving against buyers."
    if grain_pressure >= 52:
        return "Middle ground", WATCH, 3, "This is a watch period."
    if pressure_trend > 2:
        return "Low but rising", WATCH, 2, "Still favourable, but starting to lift."
    return "Low / easier", BUY, -2, "This supports buying or adding cover."


def ai_buying_recommendation(row: pd.Series) -> dict[str, Any]:
    month = row["month"]
    enso_index = float(row["enso_index"])
    grain_pressure = float(row["grain_pressure"])
    pressure_trend = float(row["pressure_trend"])
    caution_score = float(row["caution_score"])
    weather_adj = float(row.get("weather_adjustment", 0.0) or 0.0)

    window_label, crop_risk = crop_window_for_month(month)
    enso_label, enso_colour, enso_points, enso_reason = enso_assessment(enso_index)
    crop_label, crop_colour, crop_points, crop_reason = crop_risk_assessment(crop_risk, window_label)
    price_label, price_colour, price_points, price_reason = price_pressure_assessment(grain_pressure, pressure_trend)

    ai_score = caution_score + enso_points + crop_points + price_points + (weather_adj * 0.6)

    if ai_score < 58:
        action = "BUY / ADD COVER"
        action_colour = BUY
        confidence = "Favourable"
        summary = "Good buying window. Consider adding some cover instead of waiting for the perfect low."
        next_step = "Look for a sensible buying parcel or layer cover gradually."
    elif ai_score < 74:
        action = "WATCH / BUY SELECTIVELY"
        action_colour = WATCH
        confidence = "Mixed"
        summary = "Mixed signals. Buy only where cover is short or pricing is attractive."
        next_step = "Watch rain follow-up, crop reports and weekly price movement."
    else:
        action = "CAUTION / ESSENTIAL ONLY"
        action_colour = CAUTION
        confidence = "Caution"
        summary = "Risk is high. Avoid chasing the market unless cover is exposed."
        next_step = "Only cover essential tonnes; wait for clearer crop or price signals."

    return {
        "month": month,
        "month_label": month.strftime("%b %Y"),
        "action": action,
        "action_colour": action_colour,
        "confidence": confidence,
        "ai_score": round(ai_score, 1),
        "summary": summary,
        "next_step": next_step,
        "window_label": window_label,
        "crop_risk": crop_risk,
        "enso_label": enso_label,
        "enso_colour": enso_colour,
        "enso_reason": enso_reason,
        "crop_label": crop_label,
        "crop_colour": crop_colour,
        "crop_reason": crop_reason,
        "price_label": price_label,
        "price_colour": price_colour,
        "price_reason": price_reason,
        "grain_pressure": grain_pressure,
        "pressure_trend": pressure_trend,
        "enso_index": enso_index,
        "rain_message": row.get("rain_message", "No live rain data"),
        "temp_message": row.get("temp_message", "No live heat data"),
        "rain_mm_vs_normal": row.get("rain_mm_vs_normal", np.nan),
        "temp_c_vs_normal": row.get("temp_c_vs_normal", np.nan),
        "chart_signal": row["signal"],
    }


# -----------------------------
# CHART
# -----------------------------
def month_end(x: pd.Timestamp) -> pd.Timestamp:
    return x + pd.offsets.MonthEnd(1)


def add_pressure_zones(fig: go.Figure) -> None:
    zones = [
        (20, 52, "rgba(22,163,74,0.12)", "BUY ZONE", BUY),
        (52, 67, "rgba(245,158,11,0.13)", "WATCH", WATCH),
        (67, 90, "rgba(220,38,38,0.12)", "CAUTION", CAUTION),
    ]
    for y0, y1, colour, label, font_colour in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=colour, line_width=0, row=1, col=1, secondary_y=False)
        fig.add_annotation(
            x=pd.Timestamp(START_DATE) + pd.Timedelta(days=8),
            y=(y0 + y1) / 2,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=12, color=font_colour),
            bgcolor="rgba(255,255,255,0.65)",
            borderpad=4,
            xanchor="left",
            row=1,
            col=1,
            secondary_y=False,
        )


def build_chart(df: pd.DataFrame, show_callouts: bool = True) -> go.Figure:
    df = calculate_signals(df)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.68, 0.17, 0.15],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
        subplot_titles=("", "Crop timing", "Buying signal"),
    )

    for row in [2, 3]:
        fig.add_trace(go.Scatter(x=[pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)], y=[0, 0], mode="lines", line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"), row=row, col=1)

    add_pressure_zones(fig)

    for w in SEASON_WINDOWS:
        fig.add_vrect(x0=w["start"], x1=w["end"], fillcolor=w["colour"], line_width=0, layer="below", row=1, col=1)

    observed = df[df["period_type"].str.contains("Current", case=False, na=False)]
    forward = df[~df.index.isin(observed.index)]

    if not observed.empty:
        fig.add_trace(
            go.Scatter(
                x=observed["month"], y=observed["grain_pressure"], mode="lines+markers", name="Price pressure - current/past",
                line=dict(color=ORANGE, width=5), marker=dict(size=13, color=observed["signal_colour"], line=dict(color="white", width=2)),
                customdata=np.stack([observed["signal"], observed["rain_message"], observed["temp_message"]], axis=-1),
                hovertemplate="<b>%{x|%b %Y}</b><br>Price pressure: %{y:.0f}/100<br>Action: <b>%{customdata[0]}</b><br>Rain: %{customdata[1]}<br>Heat: %{customdata[2]}<extra></extra>",
            ), row=1, col=1, secondary_y=False,
        )

    if not forward.empty:
        forward_plot = pd.concat([observed.tail(1), forward], ignore_index=True) if not observed.empty else forward
        fig.add_trace(
            go.Scatter(
                x=forward_plot["month"], y=forward_plot["grain_pressure"], mode="lines+markers", name="Price pressure - forecast view",
                line=dict(color="#fb923c", width=5, dash="dash"), marker=dict(size=13, color=forward_plot["signal_colour"], line=dict(color="white", width=2)),
                customdata=np.stack([forward_plot["signal"], forward_plot["rain_message"], forward_plot["temp_message"]], axis=-1),
                hovertemplate="<b>%{x|%b %Y}</b><br>Price pressure: %{y:.0f}/100<br>Action: <b>%{customdata[0]}</b><br>Rain: %{customdata[1]}<br>Heat: %{customdata[2]}<extra></extra>",
            ), row=1, col=1, secondary_y=False,
        )

    fig.add_trace(
        go.Scatter(x=df["month"], y=df["enso_index"], mode="lines+markers", name="ENSO ocean signal", line=dict(color=BLUE, width=3, dash="dot"), marker=dict(size=7, color=BLUE), hovertemplate="<b>%{x|%b %Y}</b><br>ENSO: %{y:.2f}°C<extra></extra>"),
        row=1, col=1, secondary_y=True,
    )
    fig.add_hline(y=EL_NINO_THRESHOLD, line_dash="dash", line_color="#ef4444", line_width=1.5, row=1, col=1, secondary_y=True)
    fig.add_hline(y=LA_NINA_THRESHOLD, line_dash="dash", line_color="#3b82f6", line_width=1.5, row=1, col=1, secondary_y=True)

    for w in SEASON_WINDOWS:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        mid = start + (end - start) / 2
        fill = w["colour"].replace("0.13", "0.65").replace("0.16", "0.65").replace("0.18", "0.65").replace("0.22", "0.65")
        fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1, fillcolor=fill, line=dict(color="rgba(148,163,184,0.45)", width=1), row=2, col=1)
        fig.add_annotation(x=mid, y=0.62, text=f"<b>{w['label']}</b><br><span style='font-size:10px'>Risk: {w['risk']}</span>", showarrow=False, font=dict(size=12, color="#0f172a"), row=2, col=1)

    for _, r in df.iterrows():
        start = r["month"]
        end = month_end(start)
        fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1, fillcolor=r["signal_colour"], opacity=0.92, line=dict(color="white", width=1), row=3, col=1)
        fig.add_annotation(x=start + pd.Timedelta(days=14), y=0.52, text=f"<b>{r['signal']}</b>", showarrow=False, font=dict(size=11, color="white"), row=3, col=1)

    if show_callouts:
        for note in CROP_NOTES:
            fig.add_annotation(x=pd.Timestamp(note["date"]), y=note["y"], text=note["text"], showarrow=True, arrowhead=2, arrowsize=0.8, arrowwidth=1.2, arrowcolor="#475569", bgcolor="rgba(255,255,255,0.94)", bordercolor="#cbd5e1", borderwidth=1, borderpad=5, font=dict(size=11, color="#0f172a"), row=1, col=1, secondary_y=False)

    fig.update_layout(
        height=860,
        margin=dict(l=45, r=55, t=90, b=35),
        title=dict(text="NSW & Victoria Grain Buying Weather Outlook", x=0.5, xanchor="center", font=dict(size=24, color="#0f172a")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, bgcolor="rgba(255,255,255,0.85)"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )
    for row in [1, 2, 3]:
        fig.update_xaxes(tickformat="%b\n%Y", dtick="M1", range=[pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)], showgrid=True, gridcolor=GRID, zeroline=False, row=row, col=1)
    fig.update_yaxes(title_text="Price pressure score", range=[20, 90], tickvals=[30, 40, 50, 60, 70, 80, 90], showgrid=True, gridcolor=GRID, zeroline=False, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ENSO °C", range=[-1.2, 1.4], showgrid=False, zeroline=False, row=1, col=1, secondary_y=True)
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)
    return fig


# -----------------------------
# RENDER HELPERS
# -----------------------------
def fmt_value(value: Any, suffix: str = "") -> str:
    try:
        if pd.isna(value):
            return "Not available"
        return f"{float(value):+.1f}{suffix}"
    except Exception:
        return "Not available"


def render_eight_month_buying_watch(df: pd.DataFrame, active_row: pd.Series) -> None:
    next_eight = df[df["month"] > active_row["month"]].head(8)
    if next_eight.empty:
        return
    cards_html = []
    for _, row in next_eight.iterrows():
        r = ai_buying_recommendation(row)
        cards_html.append(
            f"""
            <div class="watch-card">
              <div class="watch-month">{r['month_label']}</div>
              <div class="watch-action" style="color:{r['action_colour']};">{r['action']}</div>
              <div class="watch-body">
                Score: <b>{r['ai_score']}</b><br>
                {r['rain_message']}<br>
                {r['temp_message']}<br>
                <b>Next:</b> {r['next_step']}
              </div>
            </div>
            """
        )
    html = f"""
    <!DOCTYPE html><html><head><style>
    body {{ margin:0; padding:0; font-family:"Source Sans Pro", Arial, sans-serif; background:transparent; }}
    .watch-section {{ margin-left:45px; margin-right:170px; margin-top:0; }}
    .watch-section-title {{ font-size:1.28rem; font-weight:800; color:#0f172a; margin-bottom:10px; }}
    .watch-grid {{ display:grid; grid-template-columns:repeat(8,minmax(0,1fr)); gap:10px; align-items:stretch; width:100%; }}
    .watch-card {{ border:1px solid #e2e8f0; border-radius:14px; padding:13px 11px; background:#fff; min-height:205px; box-shadow:0 2px 8px rgba(15,23,42,0.04); overflow:hidden; box-sizing:border-box; }}
    .watch-month {{ font-size:0.68rem; color:#64748b; font-weight:900; letter-spacing:0.11em; text-transform:uppercase; white-space:nowrap; }}
    .watch-action {{ font-size:0.86rem; font-weight:900; line-height:1.12; margin-top:7px; min-height:42px; }}
    .watch-body {{ color:#475569; font-size:0.74rem; line-height:1.28; margin-top:7px; }}
    @media (max-width:1250px) {{ .watch-section {{ margin-left:10px; margin-right:10px; }} .watch-grid {{ grid-template-columns:repeat(4,minmax(0,1fr)); }} }}
    @media (max-width:760px) {{ .watch-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    </style></head><body><div class="watch-section"><div class="watch-section-title">Next 8-month buying watch</div><div class="watch-grid">{''.join(cards_html)}</div></div></body></html>
    """
    components.html(html, height=310, scrolling=False)


def recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    recs = [ai_buying_recommendation(row) for _, row in df.iterrows()]
    return pd.DataFrame([
        {
            "Month": r["month_label"],
            "What to do": r["action"],
            "Score": r["ai_score"],
            "Rain outlook": r["rain_message"],
            "Rain vs normal": fmt_value(r["rain_mm_vs_normal"], " mm"),
            "Heat outlook": r["temp_message"],
            "Heat vs normal": fmt_value(r["temp_c_vs_normal"], "°C"),
            "Crop timing": r["window_label"],
            "Next step": r["next_step"],
        }
        for r in recs
    ])



# -----------------------------
# LIVE RADAR MAP
# -----------------------------
RADAR_LOCATIONS = [
    {"name": "NSW/VIC grain belt overview", "lat": -35.4, "lon": 145.6, "zoom": 6},
    {"name": "Wagga Wagga / Riverina", "lat": -35.11, "lon": 147.37, "zoom": 7},
    {"name": "Griffith / Murrumbidgee", "lat": -34.29, "lon": 146.05, "zoom": 7},
    {"name": "Dubbo / Central West NSW", "lat": -32.25, "lon": 148.60, "zoom": 7},
    {"name": "Horsham / Wimmera", "lat": -36.71, "lon": 142.20, "zoom": 7},
    {"name": "Bendigo / Central Victoria", "lat": -36.76, "lon": 144.28, "zoom": 7},
]


def render_radar_weather_map() -> None:
    """Show a simple live radar map page for current rain.

    This is deliberately kept separate from the 3-6 month outlook.
    The radar is useful for today's rain and short-term movement, while the
    main buying page is for seasonal pressure and buying decisions.
    """
    st.title("🌧️ Live Rain Radar Map")
    st.caption("Use this to see current rain around the grain regions. It is for today/now, not a long-term price forecast.")

    c1, c2 = st.columns([1.2, 2.8])
    with c1:
        place_name = st.selectbox(
            "Area to view",
            [r["name"] for r in RADAR_LOCATIONS],
            index=0,
            help="Pick a broad grain area or one of the key regions.",
        )
        selected = next(r for r in RADAR_LOCATIONS if r["name"] == place_name)
        st.markdown(
            """
            <div class='card'>
              <div class='card-title'>Plain read</div>
              <div class='card-sub'>
                This map shows rain that is around now and where it has been moving recently.
                It helps with short-term checking, but the main buying page is still the place
                for 3–6 month rain, heat and price-pressure guidance.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info("Tip: use the + / - buttons on the map to zoom in around farms or supplier areas.")

    with c2:
        # RainViewer public map embed. It provides a simple animated radar layer without keys.
        loc = f"{selected['lat']},{selected['lon']},{selected['zoom']}"
        radar_url = (
            "https://www.rainviewer.com/map.html?"
            f"loc={loc}"
            "&oFa=1&oC=1&oU=0&oCS=1&oF=0&oAP=1"
            "&c=3&o=83&lm=1&layer=radar&sm=1&sn=1&hu=1"
        )
        html = f"""
        <iframe
            src="{radar_url}"
            width="100%"
            height="720"
            frameborder="0"
            style="border:1px solid #e2e8f0; border-radius:18px; box-shadow:0 2px 8px rgba(15,23,42,0.06);"
            allowfullscreen>
        </iframe>
        """
        components.html(html, height=740, scrolling=False)

    st.markdown(
        """
        <p class='source-note'>
        Radar map: RainViewer public radar map. It is useful for current rain movement and quick visual checks.
        It should not be used as the seasonal rain outlook that drives the buying pressure score.
        </p>
        """,
        unsafe_allow_html=True,
    )




# -----------------------------
# GRAIN OFFERS PAGE
# -----------------------------
OFFER_TEMPLATE_COLUMNS = [
    "Offer Date", "State", "Region", "Commodity", "Grade", "Delivery Month",
    "Delivery Type", "Location", "Buyer", "Price $/t", "Change $/t", "Tonnes",
    "Status", "Notes",
]


def build_sample_offers() -> pd.DataFrame:
    """Small example list so the page is useful before a live pricing feed is connected."""
    today = pd.Timestamp.today().normalize()
    rows = [
        [today, "NSW", "Riverina", "Wheat", "APW1", today + pd.DateOffset(months=0), "Delivered", "Wagga Wagga", "Example buyer", 365, 4, 500, "Indicative", "Buyer interest has lifted."],
        [today, "NSW", "Central West", "Barley", "F1", today + pd.DateOffset(months=1), "Site", "Dubbo", "Example buyer", 302, -2, 400, "Indicative", "Softer tone today."],
        [today, "NSW", "Murrumbidgee", "Sorghum", "SOR1", today + pd.DateOffset(months=0), "Delivered", "Griffith", "Example buyer", 352, 8, 300, "Indicative", "Feed demand looks firmer."],
        [today, "VIC", "Wimmera", "Wheat", "APW1", today + pd.DateOffset(months=2), "Delivered", "Horsham", "Example buyer", 356, 1, 350, "Indicative", "Fair value, no panic."],
        [today, "VIC", "Central Victoria", "Canola", "CAN1", today + pd.DateOffset(months=1), "Site", "Bendigo", "Example buyer", 615, 6, 150, "Indicative", "Oilseed tone is firmer."],
        [today, "QLD", "Darling Downs", "Sorghum", "SOR1", today + pd.DateOffset(months=0), "Delivered", "Toowoomba", "Example buyer", 360, 7, 450, "Indicative", "Nearby demand looks solid."],
        [today, "QLD", "Darling Downs", "Wheat", "APW1", today + pd.DateOffset(months=3), "Delivered", "Dalby", "Example buyer", 372, 0, 250, "Indicative", "Flat today."],
    ]
    return pd.DataFrame(rows, columns=OFFER_TEMPLATE_COLUMNS)


def _normalise_col_name(col: str) -> str:
    return str(col).strip().lower().replace("_", " ").replace("/", " ")


def standardise_offers_file(uploaded_file) -> pd.DataFrame | None:
    """Accept a simple CSV/XLSX and turn common column names into the app's friendly names."""
    if uploaded_file is None:
        return None

    try:
        name = uploaded_file.name.lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            raw = pd.read_excel(uploaded_file)
        else:
            raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.warning(f"I could not read that file. Try saving it as CSV. Detail: {e}")
        return None

    aliases = {
        "Offer Date": ["offer date", "date", "pricing date", "quote date", "updated", "last updated"],
        "State": ["state", "province"],
        "Region": ["region", "area", "zone"],
        "Commodity": ["commodity", "grain", "crop", "product"],
        "Grade": ["grade", "quality", "spec"],
        "Delivery Month": ["delivery month", "month", "delivery", "period", "delivery period", "delivery date"],
        "Delivery Type": ["delivery type", "type", "basis", "delivered site", "site delivered", "location type"],
        "Location": ["location", "site", "port", "town", "delivery point", "destination"],
        "Buyer": ["buyer", "counterparty", "merchant"],
        "Price $/t": ["price $/t", "price", "bid", "offer", "buyer bid", "grower offer", "$/t", "aud t", "aud/t"],
        "Change $/t": ["change $/t", "change", "move", "daily change", "change aud", "change aud/t"],
        "Tonnes": ["tonnes", "tons", "volume", "mt"],
        "Status": ["status", "firm indicative", "firmness"],
        "Notes": ["notes", "comment", "comments", "what it means"],
    }

    lookup = {_normalise_col_name(c): c for c in raw.columns}
    out = pd.DataFrame()
    for friendly, names in aliases.items():
        source = None
        for n in names:
            if n in lookup:
                source = lookup[n]
                break
        if source is not None:
            out[friendly] = raw[source]
        else:
            out[friendly] = ""

    if out["Price $/t"].eq("").all():
        st.warning("I found the file, but I could not find a price column. Use a column called Price, Bid, Offer, or Price $/t.")
        return None

    out["Offer Date"] = pd.to_datetime(out["Offer Date"], errors="coerce").fillna(pd.Timestamp.today().normalize())
    out["Delivery Month"] = pd.to_datetime(out["Delivery Month"], errors="coerce").fillna(pd.Timestamp.today().normalize())
    out["Price $/t"] = pd.to_numeric(out["Price $/t"], errors="coerce")
    out["Change $/t"] = pd.to_numeric(out["Change $/t"], errors="coerce").fillna(0)
    out["Tonnes"] = pd.to_numeric(out["Tonnes"], errors="coerce")
    out = out.dropna(subset=["Price $/t"])

    for col in ["State", "Region", "Commodity", "Grade", "Delivery Type", "Location", "Buyer", "Status", "Notes"]:
        out[col] = out[col].astype(str).replace({"nan": ""}).str.strip()

    out["State"] = out["State"].str.upper()
    out["Commodity"] = out["Commodity"].str.title()
    out["Status"] = out["Status"].replace("", "Uploaded")
    return out[OFFER_TEMPLATE_COLUMNS]


def explain_offer(change: float, delivery_month: pd.Timestamp) -> tuple[str, str]:
    """Turn market movement into plain English."""
    months_ahead = (delivery_month.to_period("M") - pd.Timestamp.today().to_period("M")).n
    if change >= 5:
        tone = "Firming"
        read = "Buyers are lifting bids. Do not assume this price will still be here later."
    elif change >= 1:
        tone = "Slightly firmer"
        read = "The market is edging up. Worth watching closely if cover is short."
    elif change <= -5:
        tone = "Softening"
        read = "Prices have pulled back. Avoid panic buying unless you need cover."
    elif change <= -1:
        tone = "Slightly softer"
        read = "A small pullback. Watch for a better buying window."
    else:
        tone = "Flat"
        read = "No big move today. Decision should be based on your cover position."

    if months_ahead >= 2:
        read += " This is a forward month, so check contract terms and delivery timing."
    return tone, read


def render_grain_offers_page() -> None:
    st.title("🌾 Grain Offers & Bids")
    st.caption("Shows the bids/offers the app can update automatically. No uploads and no manual typing.")

    if st.button("Refresh offers / bids now", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        """
        <div class='plain-note'>
        <b>Plain answer:</b> there is no connected local supplier feed yet, so this page now shows the automatic market
        bid/offer information we can read today: ASX grain futures bid/ask/last where available, weekly public benchmarks,
        and clear source links. Local delivered bids from Cargill, GrainCorp, Clear Grain Exchange or DailyGrain still need a proper feed/login/API.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Checking live bid/offer sources..."):
        try:
            asx_df, asx_status = fetch_asx_grain_futures()
        except Exception as e:
            asx_df, asx_status = pd.DataFrame(), f"ASX bids/offers could not update. Detail: {e}"

        try:
            abares_df, abares_status = fetch_abares_weekly_prices()
        except Exception as e:
            abares_df, abares_status = pd.DataFrame(), f"ABARES weekly benchmark could not update. Detail: {e}"

    # Turn the automatic ASX read into a farmer-friendly bid/offer board.
    bid_board = pd.DataFrame()
    if asx_df is not None and not asx_df.empty:
        bid_board = asx_df.copy()
        for col in ["Bid $/t", "Ask $/t", "Last $/t", "Settle $/t", "Change $/t", "Volume"]:
            if col not in bid_board.columns:
                bid_board[col] = np.nan
        if "Product" not in bid_board.columns:
            bid_board["Product"] = bid_board.get("ASX code", "ASX grain")
        if "Contract month" not in bid_board.columns:
            bid_board["Contract month"] = ""
        if "Contract" not in bid_board.columns:
            bid_board["Contract"] = ""
        if "Source" not in bid_board.columns:
            bid_board["Source"] = "ASX / public futures"

        # Prefer rows that have at least one useful price.
        price_cols = ["Bid $/t", "Ask $/t", "Last $/t", "Settle $/t"]
        bid_board = bid_board.dropna(subset=price_cols, how="all")
        if not bid_board.empty:
            bid_board["Plain read"] = bid_board.get("Tone", "Updated").astype(str).replace({"nan": "Updated"})
            bid_board["What it means"] = np.where(
                bid_board["Plain read"].str.contains("firm", case=False, na=False),
                "Outside market is firmer. Waiting may carry more price risk.",
                np.where(
                    bid_board["Plain read"].str.contains("soft", case=False, na=False),
                    "Outside market is softer. Do not rush unless cover is short.",
                    "Outside market direction only. Confirm local delivered prices before buying.",
                ),
            )

    local_feed_connected = False
    asx_loaded = bid_board is not None and not bid_board.empty
    abares_loaded = abares_df is not None and not abares_df.empty

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Local supplier feed</div>
          <div class='card-value' style='color:{WATCH};'>Not connected</div>
          <div class='card-sub'>No live local delivered bids yet. Needs provider access/API.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>ASX bids / asks</div>
          <div class='card-value' style='color:{BUY if asx_loaded else WATCH};'>{'Updated' if asx_loaded else 'Not available'}</div>
          <div class='card-sub'>{asx_status}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Weekly benchmark</div>
          <div class='card-value' style='color:{BUY if abares_loaded else WATCH};'>{'Updated' if abares_loaded else 'Not available'}</div>
          <div class='card-sub'>{abares_status}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Farmer view</div>
          <div class='card-value'>Auto refresh</div>
          <div class='card-sub'>Open app, press refresh if needed. No upload box and no manual entry.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Live / updated bids and offers")
    if asx_loaded:
        display = bid_board[[
            "Product", "ASX code", "Contract", "Contract month", "Bid $/t", "Ask $/t", "Last $/t", "Settle $/t", "Volume", "Plain read", "What it means", "Source"
        ]].copy()
        for col in ["Bid $/t", "Ask $/t", "Last $/t", "Settle $/t"]:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"${x:,.1f}")
        display["Volume"] = pd.to_numeric(display["Volume"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("ASX futures are not local delivered cash offers. They are a direction check. Local basis, freight, grade and delivery month still matter.")
    else:
        st.warning("No automatic bid/offer rows were visible to the app just now. The app is still working; the public source did not expose a readable price table at this moment.")

    st.markdown("### Local supplier bids/offers")
    local_status = pd.DataFrame([
        {"Source": "Clear Grain Exchange", "Status": "Needs credentials / API access", "What farmers would see": "Current bids, offers, trades and contracts where access allows"},
        {"Source": "Cargill Pricing Hub", "Status": "Needs login/feed access", "What farmers would see": "Live buyer bids by grain, grade, location and delivery period"},
        {"Source": "GrainCorp / DailyGrain / other", "Status": "Needs provider feed", "What farmers would see": "Local delivered/site prices and movement"},
    ])
    st.dataframe(local_status, use_container_width=True, hide_index=True)

    if abares_loaded:
        st.markdown("### Weekly public benchmark")
        st.caption("This is a weekly benchmark, not a local firm bid. It helps explain whether the broader market is firmer or softer.")
        st.dataframe(abares_df.head(20), use_container_width=True, hide_index=True)

    st.markdown("### Official/source buttons")
    l1, l2, l3, l4 = st.columns(4)
    with l1:
        st.link_button("Open ASX Grains", MARKET_SOURCE_LINKS["ASX Grains live screen"], use_container_width=True)
    with l2:
        st.link_button("Open ASX grain prices", MARKET_SOURCE_LINKS["ASX grain contract prices"], use_container_width=True)
    with l3:
        st.link_button("Open ABARES weekly update", MARKET_SOURCE_LINKS["ABARES weekly update"], use_container_width=True)
    with l4:
        st.link_button("Open Cargill grain prices", MARKET_SOURCE_LINKS.get("Cargill grain prices", MARKET_SOURCE_LINKS["Cargill live pricing hub"]), use_container_width=True)

    st.markdown(
        """
        <div class='plain-note'>
        <b>Important:</b> the app will not invent local offers. If a live supplier feed is not connected, it will say so.
        ASX and ABARES help with market direction, but firm buying still needs confirmed local bids/offers.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# MARKET WATCH PAGE
# -----------------------------
MARKET_SOURCE_LINKS = {
    "ABARES weekly update": "https://www.agriculture.gov.au/abares/data/weekly-commodity-price-update",
    "ASX grain futures": "https://www.asxgrains.com.au/",
    "Barchart ASX wheat futures": "https://www.barchart.com/futures/quotes/X9%2A0/futures-prices",
    "Barchart ASX barley futures": "https://www.barchart.com/futures/quotes/KI%2A0/futures-prices",
    "NSW DPI weekly commodity report": "https://www.dpi.nsw.gov.au/agriculture/commodity-report",
    "Cargill live pricing hub": "https://www.cargill.com.au/en/grain-prices",
    "Cargill grain prices": "https://www.cargill.com.au/en/grain-prices",
    "ASX Grains live screen": "https://www.asxgrains.com.au/",
    "ASX grain contract prices": "https://www.asx.com.au/markets/trade-our-derivatives-market/derivatives-market-prices/grain-derivatives",
    "ASX grain derivatives overview": "https://www.asx.com.au/markets/trade-our-derivatives-market/overview/grain-derivatives",
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flat_cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            flat_cols.append(" ".join(str(x) for x in c if str(x) != "nan").strip())
        else:
            flat_cols.append(str(c).strip())
    out.columns = flat_cols
    return out




class _BasicTableParser(HTMLParser):
    """Small no-extra-package HTML table reader.

    pandas.read_html normally wants lxml/html5lib. Some farm/work PCs do not have
    those packages installed, so this parser gives the app a safe fallback.
    It will not be perfect on every website, but it avoids crashing the app.
    """

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._in_table = False
        self._in_row = False
        self._in_cell = False
        self._current_table: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._current_table = []
        elif self._in_table and tag == "tr":
            self._in_row = True
            self._current_row = []
        elif self._in_table and self._in_row and tag in ("td", "th"):
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data):
        if self._in_cell:
            cleaned = " ".join(data.replace("\xa0", " ").split())
            if cleaned:
                self._current_cell.append(cleaned)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._in_table and self._in_row and self._in_cell and tag in ("td", "th"):
            self._current_row.append(" ".join(self._current_cell).strip())
            self._current_cell = []
            self._in_cell = False
        elif self._in_table and self._in_row and tag == "tr":
            if any(c for c in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = []
            self._in_row = False
        elif self._in_table and tag == "table":
            if self._current_table:
                self.tables.append(self._current_table)
            self._current_table = []
            self._in_table = False


def _table_rows_to_dataframe(rows: list[list[str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    # Treat first row as a header if it looks different from the data rows.
    header = padded[0]
    data = padded[1:] if len(padded) > 1 else []
    if data and len(set(header)) == len(header) and any(h for h in header):
        df = pd.DataFrame(data, columns=header)
    else:
        df = pd.DataFrame(padded)
    return df.dropna(axis=1, how="all").dropna(axis=0, how="all")


def _read_html_tables_without_lxml(html: str) -> list[pd.DataFrame]:
    parser = _BasicTableParser()
    parser.feed(html)
    return [_table_rows_to_dataframe(rows) for rows in parser.tables]


def _safe_read_html_tables(html: str) -> tuple[list[pd.DataFrame], str | None]:
    """Read simple web tables without extra packages.

    Some farm/work PCs do not have lxml installed. This app now uses only the
    small built-in reader first, so users do not see confusing lxml messages.
    If a website builds its table with JavaScript, there may be nothing readable
    in the page source; in that case we show a plain-English message.
    """
    fallback_tables = _read_html_tables_without_lxml(html)
    fallback_tables = [t for t in fallback_tables if not t.empty]
    if fallback_tables:
        return fallback_tables, "Read with the built-in table reader."
    return [], "The page opened, but there was no simple table for the app to read. The site may load prices after the page opens."


def _clean_table_for_display(df: pd.DataFrame, max_cols: int = 8) -> pd.DataFrame:
    out = _flatten_columns(df).copy()
    out = out.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for c in out.columns:
        out[c] = out[c].astype(str).str.replace("\u00a0", " ", regex=False).str.strip()
    # Drop very wide / empty-looking tables so farmers do not get a messy screen.
    keep_cols = [c for c in out.columns if not out[c].eq("").all()]
    out = out[keep_cols[:max_cols]]
    return out.head(20)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_abares_weekly_prices() -> tuple[pd.DataFrame, str]:
    """Read ABARES weekly commodity price update when the PC has internet.

    This is not paddock-level cash pricing. It is a weekly benchmark that helps show
    whether export grain values are firmer or softer.
    """
    url = MARKET_SOURCE_LINKS["ABARES weekly update"]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GrainWatch/1.0)"}
    try:
        html = requests.get(url, timeout=15, headers=headers).text
        tables, table_note = _safe_read_html_tables(html)
        if not tables:
            return pd.DataFrame(), f"Could not read ABARES weekly update. The app will still work. Detail: {table_note}"
    except Exception as e:
        return pd.DataFrame(), f"Could not read ABARES weekly update. The app will still work. Detail: {e}"

    keywords = ["wheat", "barley", "sorghum", "canola", "grain", "hay"]
    picked = []
    for tbl in tables:
        t = _clean_table_for_display(tbl, max_cols=10)
        if t.empty:
            continue
        row_text = t.astype(str).agg(" ".join, axis=1).str.lower()
        mask = row_text.apply(lambda s: any(k in s for k in keywords))
        if mask.any():
            picked.append(t.loc[mask].copy())

    if not picked:
        return pd.DataFrame(), "ABARES page opened, but no grain rows were found in a table."

    out = pd.concat(picked, ignore_index=True).drop_duplicates()
    return out.head(30), "ABARES weekly benchmark loaded."



BARCHART_ASX_SOURCES = [
    {
        "Product": "Eastern Australia Wheat",
        "ASX code": "WM",
        "Barchart root": "X9",
        "URL": "https://www.barchart.com/futures/quotes/X9%2A0/futures-prices",
    },
    {
        "Product": "Eastern Australia Feed Barley",
        "ASX code": "UB",
        "Barchart root": "KI",
        "URL": "https://www.barchart.com/futures/quotes/KI%2A0/futures-prices",
    },
]

ASX_CONTRACT_MONTHS = {
    "F": "Jan", "G": "Feb", "H": "Mar", "J": "Apr", "K": "May", "M": "Jun",
    "N": "Jul", "Q": "Aug", "U": "Sep", "V": "Oct", "X": "Nov", "Z": "Dec",
}


def _asx_product_from_code(contract: str) -> str:
    code = str(contract).upper()
    if code.startswith("WM"):
        return "Eastern Australia Wheat"
    if code.startswith("UB"):
        return "Eastern Australia Feed Barley"
    if code.startswith("US"):
        return "Australian Sorghum"
    return "Other ASX grain"


def _asx_month_from_contract(contract: str) -> str:
    code = str(contract).upper().strip()
    m = re.match(r"^[A-Z]{2}([FGHJKMNQUVXZ])(\d{4})$", code)
    if not m:
        return ""
    return f"{ASX_CONTRACT_MONTHS.get(m.group(1), m.group(1))} {m.group(2)}"


def _to_price(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace("$", "").replace(",", "").strip()
    if text in ("", "-", "None", "nan"):
        return None
    try:
        return float(text)
    except Exception:
        return None


def _asx_history_path() -> Path:
    try:
        return Path(__file__).with_name("asx_grains_history.csv")
    except Exception:
        return Path("asx_grains_history.csv")


def _load_asx_history() -> pd.DataFrame:
    path = _asx_history_path()
    if not path.exists():
        return pd.DataFrame()
    try:
        out = pd.read_csv(path)
        out["Read time"] = pd.to_datetime(out.get("Read time"), errors="coerce")
        return out.dropna(subset=["Read time"])
    except Exception:
        return pd.DataFrame()


def _save_asx_history(snapshot: pd.DataFrame) -> None:
    if snapshot is None or snapshot.empty:
        return
    path = _asx_history_path()
    cols = ["Read time", "Source", "Product", "ASX code", "Contract", "Contract month", "Bid $/t", "Ask $/t", "Last $/t", "Volume", "Settle $/t"]
    keep = snapshot.copy()
    for c in cols:
        if c not in keep.columns:
            keep[c] = np.nan
    keep = keep[cols]
    old = _load_asx_history()
    combined = pd.concat([old, keep], ignore_index=True) if not old.empty else keep
    combined = combined.drop_duplicates(subset=["Read time", "Contract"], keep="last")
    # Keep the local history small and fast.
    combined = combined.tail(3000)
    try:
        combined.to_csv(path, index=False)
    except Exception:
        pass


def _parse_asx_snapshot_from_text(text: str) -> pd.DataFrame:
    """Parse the public ASX Grains snapshot text if the page exposes it.

    The official ASX Grains screen is a JavaScript site. On some networks it still
    exposes enough snapshot text for simple reading; on others it only exposes the
    app shell. This parser is deliberately careful: if it cannot see contract rows,
    it returns an empty table instead of making numbers up.
    """
    cleaned = re.sub(r"<[^>]+>", " ", str(text))
    cleaned = cleaned.replace("&nbsp;", " ").replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    contract_pat = re.compile(r"\b(?:WM|UB|US)[FGHJKMNQUVXZ]\d{4}\b", re.I)
    matches = list(contract_pat.finditer(cleaned))
    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, match in enumerate(matches):
        contract = match.group(0).upper()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else min(len(cleaned), match.end() + 220)
        segment = cleaned[match.end():end]
        # Expected order on ASX Grains snapshot is roughly: Bid, Ask, Last, Volume, Open Interest, Settle.
        tokens = re.findall(r"-?\d+(?:\.\d+)?|-", segment)[:6]
        if len(tokens) < 3:
            continue
        while len(tokens) < 6:
            tokens.append("-")
        bid, ask, last, vol, oi, settle = tokens[:6]
        rows.append({
            "Read time": now,
            "Product": _asx_product_from_code(contract),
            "ASX code": contract[:2],
            "Contract": contract,
            "Contract month": _asx_month_from_contract(contract),
            "Bid $/t": _to_price(bid),
            "Ask $/t": _to_price(ask),
            "Last $/t": _to_price(last),
            "Volume": _to_price(vol),
            "Open Interest": _to_price(oi),
            "Settle $/t": _to_price(settle),
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).drop_duplicates(subset=["Contract"], keep="last")
    out = out[out["ASX code"].isin(["WM", "UB", "US"])]
    return out.reset_index(drop=True)


def _add_asx_tone_from_history(snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot is None or snapshot.empty:
        return pd.DataFrame()
    out = snapshot.copy()
    hist = _load_asx_history()
    out["Change $/t"] = np.nan
    out["Tone"] = "Updated"

    if not hist.empty and "Settle $/t" in hist.columns:
        hist["Read time"] = pd.to_datetime(hist["Read time"], errors="coerce")
        hist = hist.dropna(subset=["Read time"]).sort_values("Read time")
        for i, r in out.iterrows():
            contract = r.get("Contract")
            current = _to_price(r.get("Settle $/t"))
            if current is None:
                current = _to_price(r.get("Last $/t"))
            prev_rows = hist[hist["Contract"].astype(str).eq(str(contract))].copy()
            if prev_rows.empty or current is None:
                continue
            prev_price = None
            for _, pr in prev_rows.iloc[::-1].iterrows():
                prev_price = _to_price(pr.get("Settle $/t"))
                if prev_price is None:
                    prev_price = _to_price(pr.get("Last $/t"))
                if prev_price is not None:
                    break
            if prev_price is None:
                continue
            change = current - prev_price
            out.loc[i, "Change $/t"] = change
            out.loc[i, "Tone"] = _tone_from_change(change)

    # Save after tone is calculated so the next app open has a comparison point.
    _save_asx_history(out)
    return out



def _barchart_to_asx_contract(symbol: str, asx_code: str) -> str:
    """Turn Barchart symbols into the ASX codes farmers see on ASX Grains.

    Example: X9K26 becomes WMK2026, and KIH27 becomes UBH2027.
    """
    symbol = str(symbol).upper().strip()
    m = re.match(r"^(?:X9|KI)([FGHJKMNQUVXZ])(\d{2})$", symbol)
    if not m:
        return symbol
    year = int(m.group(2))
    full_year = 2000 + year if year < 80 else 1900 + year
    return f"{asx_code}{m.group(1)}{full_year}"


def _parse_barchart_price_text(text: str, source: dict[str, str]) -> pd.DataFrame:
    """Read ASX grain futures from Barchart text when ASX Grains hides values.

    Barchart is used here only as an automatic public/delayed direction check.
    We do not pretend it is a local cash grain bid.
    """
    html = str(text or "")
    cleaned = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
    cleaned = re.sub(r"<style[\s\S]*?</style>", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ").replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)

    root = re.escape(source["Barchart root"])
    # Looks for lines like: X9F27 374.00s +6.00  or KIH27 343.50s -1.50
    row_pat = re.compile(
        rf"\b({root}[FGHJKMNQUVXZ]\d{{2}})\b\s+([0-9]+(?:\.[0-9]+)?)[a-zA-Z]*\s+([+-]?[0-9]+(?:\.[0-9]+)?|unch|UNCH|Unch)\b"
    )

    rows: list[dict[str, Any]] = []
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    seen: set[str] = set()

    for m in row_pat.finditer(cleaned):
        symbol, price_text, change_text = m.groups()
        symbol = symbol.upper()
        if symbol in seen:
            continue
        seen.add(symbol)
        change = 0.0 if change_text.lower() == "unch" else _to_price(change_text)
        price = _to_price(price_text)
        contract = _barchart_to_asx_contract(symbol, source["ASX code"])
        rows.append({
            "Read time": now,
            "Source": "Barchart public delayed read",
            "Product": source["Product"],
            "ASX code": source["ASX code"],
            "Contract": contract,
            "Contract month": _asx_month_from_contract(contract),
            "Bid $/t": np.nan,
            "Ask $/t": np.nan,
            "Last $/t": price,
            "Volume": np.nan,
            "Open Interest": np.nan,
            "Settle $/t": price,
            "Change $/t": change,
            "Tone": _tone_from_change(change),
        })

    # Fallback: individual quote pages often show a headline like "ASX ... (KIH27) 343.50s -1.50".
    if not rows:
        head_pat = re.compile(
            rf"\b({root}[FGHJKMNQUVXZ]\d{{2}})\b[\s\S]{{0,120}}?([0-9]+(?:\.[0-9]+)?)[a-zA-Z]*\s+([+-][0-9]+(?:\.[0-9]+)?|unch|UNCH|Unch)\b"
        )
        m = head_pat.search(cleaned)
        if m:
            symbol, price_text, change_text = m.groups()
            change = 0.0 if change_text.lower() == "unch" else _to_price(change_text)
            price = _to_price(price_text)
            contract = _barchart_to_asx_contract(symbol, source["ASX code"])
            rows.append({
                "Read time": now,
                "Source": "Barchart public delayed read",
                "Product": source["Product"],
                "ASX code": source["ASX code"],
                "Contract": contract,
                "Contract month": _asx_month_from_contract(contract),
                "Bid $/t": np.nan,
                "Ask $/t": np.nan,
                "Last $/t": price,
                "Volume": np.nan,
                "Open Interest": np.nan,
                "Settle $/t": price,
                "Change $/t": change,
                "Tone": _tone_from_change(change),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset=["Contract"], keep="first").reset_index(drop=True)


@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_barchart_asx_grains() -> tuple[pd.DataFrame, str]:
    """Automatic backup source for ASX grain futures direction.

    It uses public Barchart ASX wheat/barley futures pages because the ASX Grains
    screen can hide its numbers inside a browser-only JavaScript app. This is an
    outside-market direction check, not a cash bid and not a guaranteed licensed
    live feed.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-AU,en;q=0.9",
    }
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for source in BARCHART_ASX_SOURCES:
        try:
            response = requests.get(source["URL"], timeout=25, headers=headers)
            response.raise_for_status()
            parsed = _parse_barchart_price_text(response.text, source)
            if parsed.empty:
                errors.append(f"{source['Product']}: page opened but no price row was visible to the app")
            else:
                frames.append(parsed)
        except Exception as e:
            errors.append(f"{source['Product']}: {e}")

    if not frames:
        detail = "; ".join(errors[:3]) if errors else "No futures rows found."
        return pd.DataFrame(), f"Could not update the public ASX backup read. Detail: {detail}"

    out = pd.concat(frames, ignore_index=True)
    # Barchart already provides a daily move; still save it so the app chart builds over time.
    _save_asx_history(out)
    return out, "ASX direction updated from public Barchart futures pages. Treat as delayed/indicative."


@st.cache_data(ttl=5 * 60, show_spinner=False)
def fetch_asx_grain_futures() -> tuple[pd.DataFrame, str]:
    """Try to read the live ASX Grains snapshot automatically.

    This is not a guaranteed tick-by-tick licensed feed. It is a public snapshot
    check from ASX Grains when the page exposes contract rows. If the page does
    not expose them, the app clearly says so and keeps working.
    """
    url = MARKET_SOURCE_LINKS["ASX grain futures"]
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        snapshot = _parse_asx_snapshot_from_text(response.text)
    except Exception:
        snapshot = pd.DataFrame()

    if snapshot is not None and not snapshot.empty:
        snapshot["Source"] = "ASX Grains public snapshot"
        snapshot = _add_asx_tone_from_history(snapshot)
        return snapshot, "Live ASX Grains snapshot updated."

    # ASX Grains often hides the values in a browser-only app. Try the public futures
    # pages as a no-import backup so farmers still see a live/delayed direction.
    backup, backup_status = fetch_barchart_asx_grains()
    if backup is not None and not backup.empty:
        return backup, backup_status

    return pd.DataFrame(), backup_status


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_nsw_dpi_weekly_report() -> tuple[pd.DataFrame, str]:
    """Try to read a simple weekly commodity report table if the site exposes one."""
    url = MARKET_SOURCE_LINKS["NSW DPI weekly commodity report"]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GrainWatch/1.0)"}
    try:
        html = requests.get(url, timeout=15, headers=headers).text
        tables, table_note = _safe_read_html_tables(html)
        if not tables:
            return pd.DataFrame(), f"Could not read NSW weekly commodity report. Detail: {table_note}"
    except Exception as e:
        return pd.DataFrame(), f"Could not read NSW weekly commodity report. Detail: {e}"

    keywords = ["wheat", "barley", "sorghum", "canola"]
    picked = []
    for tbl in tables:
        t = _clean_table_for_display(tbl, max_cols=8)
        if t.empty:
            continue
        row_text = t.astype(str).agg(" ".join, axis=1).str.lower()
        mask = row_text.apply(lambda s: any(k in s for k in keywords))
        if mask.any():
            picked.append(t.loc[mask].copy())

    if not picked:
        return pd.DataFrame(), "NSW weekly report page opened, but no simple grain table was found."

    out = pd.concat(picked, ignore_index=True).drop_duplicates()
    return out.head(20), "NSW weekly commodity report loaded."


def explain_market_direction(text: str) -> tuple[str, str, str]:
    """Turn market-source status text into very simple display wording."""
    s = str(text).lower()
    if "loaded" in s:
        return "Updated", BUY, "This source was read successfully. Use it as one market signal, not the only reason to buy."
    if "opened" in s:
        return "Partly read", WATCH, "The page opened, but the app could not find a clean table. Use the link or upload your price sheet."
    return "Not available", CAUTION, "The app could not read this source from this PC/network. The rest of the dashboard still works."


def summarise_offers_for_market_watch(offers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create a simple price-sheet summary for the Market Watch page."""
    if offers is None or offers.empty:
        return pd.DataFrame(), {
            "source": "No offer sheet uploaded",
            "count": 0,
            "avg": np.nan,
            "firming": 0,
            "softening": 0,
            "plain": "Upload a current offer sheet to bring local buyer bids into this page.",
        }

    view = offers.copy()
    view["Price $/t"] = pd.to_numeric(view["Price $/t"], errors="coerce")
    view["Change $/t"] = pd.to_numeric(view["Change $/t"], errors="coerce").fillna(0)
    view = view.dropna(subset=["Price $/t"])
    if view.empty:
        return pd.DataFrame(), {
            "source": "Offer sheet had no readable prices",
            "count": 0,
            "avg": np.nan,
            "firming": 0,
            "softening": 0,
            "plain": "The uploaded sheet was found, but no price column could be read.",
        }

    view["Market tone"] = np.where(view["Change $/t"] > 0, "Firming", np.where(view["Change $/t"] < 0, "Softening", "Flat"))
    view["Delivery Month"] = pd.to_datetime(view["Delivery Month"], errors="coerce").fillna(pd.Timestamp.today())
    view["Delivery Month Text"] = view["Delivery Month"].dt.strftime("%b %Y")
    best = view.sort_values("Price $/t", ascending=False).groupby(["State", "Commodity"], as_index=False).first()
    best = best[["State", "Commodity", "Grade", "Region", "Delivery Month Text", "Delivery Type", "Location", "Price $/t", "Change $/t", "Market tone"]].head(12)
    stats = {
        "source": "Uploaded / manual offers",
        "count": len(view),
        "avg": view["Price $/t"].mean(),
        "firming": int((view["Change $/t"] > 0).sum()),
        "softening": int((view["Change $/t"] < 0).sum()),
        "plain": "Local prices are included from your offer sheet. Confirm firm bids before acting.",
    }
    return best, stats


def market_watch_final_read(active_rec: dict, offer_stats: dict[str, Any], abares_status: str, asx_status: str) -> tuple[str, str, str]:
    """Combine the available signals into one farmer-friendly message."""
    action = active_rec.get("action", "WATCH")
    score = float(active_rec.get("ai_score", 60))

    firming = int(offer_stats.get("firming", 0) or 0)
    softening = int(offer_stats.get("softening", 0) or 0)
    extra_pressure = 0
    if firming > softening:
        extra_pressure += 4
    if "loaded" in abares_status.lower():
        extra_pressure += 1
    asx_lower = asx_status.lower()
    if "firmer" in asx_lower:
        extra_pressure += 4
    elif "softer" in asx_lower:
        extra_pressure -= 3
    elif "flat" in asx_lower:
        extra_pressure += 0
    elif "loaded" in asx_lower:
        extra_pressure += 1

    combined = score + extra_pressure
    if combined < 58:
        return "BUY / ADD COVER", BUY, "Signals are friendly enough to add cover in stages, especially where physical cover is below target."
    if combined < 74:
        return "WATCH / BUY SELECTIVELY", WATCH, "Do not panic, but stay close to offers. Buy only where cover is short or the price is attractive."
    return "CAUTION / ESSENTIAL ONLY", CAUTION, "Risk is high enough that waiting too long may hurt. Cover exposed tonnes first and avoid chasing big tonnes blindly."


def render_market_watch_page() -> None:
    st.title("📈 Market Watch")
    st.caption("Live/updated signals only: weather pressure, weekly benchmarks and ASX direction. No upload boxes.")

    if st.button("Refresh live market data now", use_container_width=False):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Loading market and weather signals..."):
        weather_df, live_status = fetch_regional_weather(DEFAULT_REGIONS)
        base = build_default_monthly_data()
        df_live = apply_live_weather_to_base(base, weather_df)
        df_live = calculate_signals(df_live)

        today_month = pd.Timestamp.today().replace(day=1)
        if today_month < df_live["month"].min() or today_month > df_live["month"].max():
            active_month = pd.Timestamp("2026-05-01")
        else:
            active_month = today_month
        active_row_local = df_live.iloc[(df_live["month"] - active_month).abs().idxmin()]
        active_rec_local = ai_buying_recommendation(active_row_local)

        abares_df, abares_status = fetch_abares_weekly_prices()
        asx_df, asx_status = get_asx_snapshot_for_market_watch()
        nsw_df, nsw_status = fetch_nsw_dpi_weekly_report()

    offers_for_summary = pd.DataFrame()
    best_offers, offer_stats = summarise_offers_for_market_watch(offers_for_summary)
    final_action, final_colour, final_message = market_watch_final_read(active_rec_local, offer_stats, abares_status, asx_status)

    st.markdown(
        """
        <div class='plain-note'>
        <b>Plain idea:</b> this page updates from available live/public sources when the app opens. It does not ask farmers to import files.
        It checks weather, crop timing, weekly public benchmarks and ASX direction. Firm supplier bids can be connected later when we have a proper feed.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Live source details", expanded=False):
        st.markdown(f"**Weather:** {live_status}")
        st.markdown(f"**ABARES weekly benchmark:** {abares_status}")
        st.markdown(f"**ASX / outside market direction:** {asx_status}")
        st.markdown(f"**NSW weekly report:** {nsw_status}")
        st.markdown("The app refreshes these sources automatically on open and uses short cache times so it does not hammer public websites.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Overall read</div>
          <div class='card-value' style='color:{final_colour};'>{final_action}</div>
          <div class='card-sub'>{final_message}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Rain outlook</div>
          <div class='card-value' style='color:{active_row_local.get('rain_colour', SLATE)};'>{active_row_local.get('rain_message', 'No live rain data')}</div>
          <div class='card-sub'>Compared with normal: <b>{fmt_value(active_row_local.get('rain_mm_vs_normal'), ' mm')}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>Heat outlook</div>
          <div class='card-value' style='color:{active_row_local.get('temp_colour', SLATE)};'>{active_row_local.get('temp_message', 'No live heat data')}</div>
          <div class='card-sub'>Compared with normal: <b>{fmt_value(active_row_local.get('temp_c_vs_normal'), '°C')}</b></div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        asx_label, asx_colour, asx_body = explain_market_direction(asx_status)
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>ASX direction</div>
          <div class='card-value' style='color:{asx_colour};'>{asx_label}</div>
          <div class='card-sub'>{asx_body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Simple signals")
    s1, s2, s3 = st.columns(3)
    for col, title, status in [
        (s1, "ABARES weekly benchmark", abares_status),
        (s2, "ASX / outside market", asx_status),
        (s3, "NSW weekly commodity report", nsw_status),
    ]:
        label, colour, body = explain_market_direction(status)
        with col:
            st.markdown(f"""
            <div class='ai-card'>
              <div class='ai-card-title'>{title}</div>
              <div class='ai-card-value' style='color:{colour};'>{label}</div>
              <div class='ai-card-body'>{body}</div>
            </div>
            """, unsafe_allow_html=True)

    if asx_df is not None and not asx_df.empty:
        st.markdown("### ASX read")
        st.dataframe(asx_df, use_container_width=True, hide_index=True)

    tabs = st.tabs(["ABARES weekly", "ASX futures", "NSW weekly report", "Source links"])
    with tabs[0]:
        st.caption("Weekly benchmark only. Not the same as a firm local cash bid.")
        if abares_df.empty:
            st.warning(abares_status)
        else:
            st.dataframe(abares_df, use_container_width=True, hide_index=True)
    with tabs[1]:
        st.caption("ASX futures show broader market direction. Local basis and freight still matter.")
        st.info(asx_status)
        if asx_df is not None and not asx_df.empty:
            st.dataframe(asx_df, use_container_width=True, hide_index=True)
        st.markdown("Use the **ASX Futures** page for the latest ASX read and the official source buttons.")
    with tabs[2]:
        st.caption("Useful weekly commentary if the page exposes a table the app can read.")
        if nsw_df.empty:
            st.warning(nsw_status)
        else:
            st.dataframe(nsw_df, use_container_width=True, hide_index=True)
    with tabs[3]:
        st.markdown("- ABARES weekly commodity price update")
        st.markdown("- ASX futures: automatic ASX Grains read first, then public futures backup where available")
        st.markdown("- NSW DPI weekly commodity report")
        st.markdown("- Firm local bids/offers still need a proper supplier or market-data feed connection later.")

    st.markdown(
        """
        <p class='source-note'>
        Market Watch keeps live public sources separate from firm supplier offers. Public benchmarks help explain direction; supplier bids and contract prices still need to be confirmed before buying.
        </p>
        """,
        unsafe_allow_html=True,
    )




# -----------------------------
# ASX FUTURES PAGE
# -----------------------------
ASX_CONTRACT_GUIDE = pd.DataFrame([
    {"Product": "Eastern Australia Wheat", "Simple use": "Broad wheat price direction for NSW/VIC/QLD", "ASX code": "WM", "Common months": "Jan, Mar, May, Jul, Sep, Nov"},
    {"Product": "Eastern Australia Feed Barley", "Simple use": "Broad feed barley direction", "ASX code": "UB", "Common months": "Jan, Mar, May, Jul, Sep, Nov"},
])


def _default_asx_snapshot() -> pd.DataFrame:
    return pd.DataFrame([
        {"Product": "Eastern Australia Wheat", "Contract": "WM", "Month": "", "Last $/t": np.nan, "Change $/t": np.nan, "Tone": "Not entered", "Note": ""},
        {"Product": "Eastern Australia Feed Barley", "Contract": "UB", "Month": "", "Last $/t": np.nan, "Change $/t": np.nan, "Tone": "Not entered", "Note": ""},
    ])


def _tone_from_change(change: Any) -> str:
    try:
        val = float(change)
    except Exception:
        return "Not entered"
    if val > 1:
        return "Firmer"
    if val < -1:
        return "Softer"
    return "Flat"


def get_asx_snapshot_for_market_watch() -> tuple[pd.DataFrame, str]:
    """Return ASX grain direction for Market Watch.

    First choice: automatic ASX Grains public snapshot.
    Backup: any screen-assisted snapshot the user entered earlier.
    """
    live_df, live_status = fetch_asx_grain_futures()
    if live_df is not None and not live_df.empty:
        tones = set(live_df.get("Tone", pd.Series(dtype=str)).dropna().astype(str))
        if "Firmer" in tones:
            return live_df, "Firmer live ASX snapshot. This adds buying pressure."
        if "Softer" in tones and "Firmer" not in tones:
            return live_df, "Softer live ASX snapshot. This may reduce buying pressure."
        if "Flat" in tones:
            return live_df, "Flat live ASX snapshot. No major change to buying pressure."
        return live_df, live_status

    snap = st.session_state.get("asx_snapshot")
    if snap is None or not isinstance(snap, pd.DataFrame) or snap.empty:
        return pd.DataFrame(), live_status

    snap = snap.copy()
    if "Change $/t" in snap.columns:
        snap["Tone"] = snap["Change $/t"].apply(_tone_from_change)
    tones = set(snap.get("Tone", pd.Series(dtype=str)).dropna().astype(str))
    if "Firmer" in tones:
        return snap, "Firmer ASX snapshot entered. This adds buying pressure."
    if "Softer" in tones and "Firmer" not in tones:
        return snap, "Softer ASX snapshot entered. This may reduce buying pressure."
    if "Flat" in tones:
        return snap, "Flat ASX snapshot entered. No major change to buying pressure."
    return snap, "ASX snapshot saved, but no direction was entered."


def build_asx_history_chart(snapshot: pd.DataFrame) -> go.Figure:
    """Create an ASX-style chart from the locally saved automatic reads."""
    hist = _load_asx_history()
    fig = go.Figure()

    if hist.empty:
        if snapshot is not None and not snapshot.empty:
            display = snapshot.copy()
            display["Chart price"] = display["Last $/t"].combine_first(display["Settle $/t"])
            fig.add_trace(go.Bar(x=display["Contract"], y=display["Chart price"], name="Latest price"))
            fig.update_layout(title="Latest ASX snapshot", height=420, margin=dict(l=20, r=20, t=55, b=35))
            fig.update_yaxes(title_text="$/t")
        return fig

    hist = hist.copy()
    hist["Read time"] = pd.to_datetime(hist["Read time"], errors="coerce")
    hist = hist.dropna(subset=["Read time"])
    hist["Chart price"] = pd.to_numeric(hist.get("Settle $/t"), errors="coerce")
    if "Last $/t" in hist.columns:
        hist["Chart price"] = hist["Chart price"].fillna(pd.to_numeric(hist["Last $/t"], errors="coerce"))
    hist = hist.dropna(subset=["Chart price"])

    # Show the most relevant contracts first: nearest wheat and barley rows that have data.
    contracts = []
    if snapshot is not None and not snapshot.empty:
        for code in ["WM", "UB", "US"]:
            subset = snapshot[snapshot["ASX code"].eq(code)]
            subset = subset[pd.to_numeric(subset["Settle $/t"].fillna(subset["Last $/t"]), errors="coerce").notna()]
            if not subset.empty:
                contracts.append(str(subset.iloc[0]["Contract"]))
    if not contracts:
        contracts = list(hist["Contract"].dropna().astype(str).unique()[:4])

    for contract in contracts[:4]:
        h = hist[hist["Contract"].astype(str).eq(contract)].sort_values("Read time")
        if h.empty:
            continue
        fig.add_trace(go.Scatter(x=h["Read time"], y=h["Chart price"], mode="lines+markers", name=contract))

    fig.update_layout(
        title="ASX grain futures trend saved by this app",
        height=460,
        margin=dict(l=20, r=20, t=55, b=35),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="$/t", gridcolor=GRID)
    fig.update_xaxes(gridcolor=GRID)
    return fig


def render_asx_futures_page() -> None:
    st.title("🌾 ASX Futures")
    st.caption("Automatic ASX grain futures direction. First tries ASX Grains, then uses public futures pages if ASX hides the numbers. Direction only, not a local delivered bid.")

    if st.button("Refresh ASX snapshot", use_container_width=False):
        fetch_asx_grain_futures.clear()
        st.rerun()

    with st.spinner("Checking ASX Grains..."):
        live_asx, live_status = fetch_asx_grain_futures()

    if live_asx is not None and not live_asx.empty:
        tones = set(live_asx.get("Tone", pd.Series(dtype=str)).dropna().astype(str))
        if "Firmer" in tones:
            top_label, top_colour = "Firmer", CAUTION
            top_message = "The latest ASX read is firmer than the last saved read. Waiting may carry more price risk."
        elif "Softer" in tones and "Firmer" not in tones:
            top_label, top_colour = "Softer", BUY
            top_message = "The latest ASX read is softer than the last saved read. Avoid rushing unless cover is short."
        elif "Flat" in tones:
            top_label, top_colour = "Flat", WATCH
            top_message = "The latest ASX read is broadly flat. Use local offers and weather as the main guide."
        else:
            top_label, top_colour = "Updated", BUY
            top_message = "ASX updated. The app has saved this read so it can compare next time."
    else:
        top_label, top_colour = "Not updated", WATCH
        top_message = live_status

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>ASX read</div>
          <div class='card-value' style='color:{top_colour};'>{top_label}</div>
          <div class='card-sub'>{top_message}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
          <div class='card-title'>Main contracts</div>
          <div class='card-value'>WM / UB</div>
          <div class='card-sub'>WM = Eastern wheat. UB = Eastern feed barley. Futures are not local cash offers.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='card'>
          <div class='card-title'>Plain use</div>
          <div class='card-value'>Direction only</div>
          <div class='card-sub'>Firmer, flat or softer. Local basis, freight, grade and delivery month still matter.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Live ASX snapshot")
    st.caption("The app checks for grain futures when it opens. If ASX Grains hides the numbers, it tries public futures pages so the screen can still populate automatically.")

    if live_asx is not None and not live_asx.empty:
        display = live_asx.copy()
        for col in ["Bid $/t", "Ask $/t", "Last $/t", "Settle $/t", "Change $/t"]:
            if col in display.columns:
                display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"${x:,.1f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.plotly_chart(build_asx_history_chart(live_asx), use_container_width=True)
    else:
        st.info(live_status)
        st.markdown(
            """
            <div class='plain-note'>
            This does not stop the rest of the system. Weather, radar, Market Watch and local offers still work.
            For a guaranteed official live ASX feed, the clean next step is a licensed market-data feed.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### ASX contract guide")
    st.dataframe(ASX_CONTRACT_GUIDE, use_container_width=True, hide_index=True)

    st.markdown("### Official / public market links")
    l1, l2, l3, l4, l5 = st.columns(5)
    with l1:
        st.link_button("Open ASX Grains screen", MARKET_SOURCE_LINKS["ASX Grains live screen"], use_container_width=True)
    with l2:
        st.link_button("Open ASX grain prices", MARKET_SOURCE_LINKS["ASX grain contract prices"], use_container_width=True)
    with l3:
        st.link_button("Open ASX overview", MARKET_SOURCE_LINKS["ASX grain derivatives overview"], use_container_width=True)
    with l4:
        st.link_button("Open wheat futures", MARKET_SOURCE_LINKS["Barchart ASX wheat futures"], use_container_width=True)
    with l5:
        st.link_button("Open barley futures", MARKET_SOURCE_LINKS["Barchart ASX barley futures"], use_container_width=True)

    with st.expander("If ASX cannot update", expanded=False):
        st.markdown("The app will keep trying the automatic ASX read. If ASX hides the numbers from apps, use the official buttons above. No manual entry is shown to farmers.")



# -----------------------------
# APP START / PAGE ROUTING
# -----------------------------
st.title("🌾 Grain Weather Buying Outlook")
st.caption("Simple buying guide for NSW/VIC grain. Green = BUY, amber = WATCH, red = CAUTION.")

with st.sidebar:
    st.header("Menu")
    page = st.radio(
        "Go to",
        ["Buying Outlook", "Market Watch", "ASX Futures", "Radar Map", "Grain Offers"],
        index=0,
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("Refresh live data now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("The app refreshes live sources when it opens. Use the button if you want to check again now.")
    st.divider()
    st.markdown("### Included grain areas")
    for r in DEFAULT_REGIONS:
        st.caption(f"• {r['name']}")
    st.divider()
    st.caption("This is a buying aid, not a guaranteed grain price forecast. Use it with local offers, cover position and crop reports.")

if page == "Radar Map":
    render_radar_weather_map()
    st.stop()

if page == "ASX Futures":
    render_asx_futures_page()
    st.stop()

if page == "Market Watch":
    render_market_watch_page()
    st.stop()

if page == "Grain Offers":
    render_grain_offers_page()
    st.stop()

# Buying Outlook page
show_callouts = True
base_df = build_default_monthly_data()

with st.spinner("Loading rain and heat outlook..."):
    weather_df, live_status = fetch_regional_weather(DEFAULT_REGIONS)

df = apply_live_weather_to_base(base_df, weather_df)
df = calculate_signals(df)

# Use the current month where possible. If today is outside the planning range, use May 2026.
today_month = pd.Timestamp.today().replace(day=1)
if today_month < df["month"].min() or today_month > df["month"].max():
    active_month = pd.Timestamp("2026-05-01")
else:
    active_month = today_month

active_row = df.iloc[(df["month"] - active_month).abs().idxmin()]
active_rec = ai_buying_recommendation(active_row)

next_caution = df[df["signal"].eq("CAUTION")]
first_caution_text = "None in range" if next_caution.empty else next_caution.iloc[0]["month"].strftime("%b %Y")
peak_row = df.iloc[df["caution_score"].idxmax()]

with st.expander("Live weather source details", expanded=False):
    st.markdown(f"**Status:** {live_status}")
    st.markdown(
        "The app reads a seasonal outlook for key grain areas and turns it into simple wording: "
        "drier/wetter than normal and hotter/cooler than normal. Those signals nudge the buying score up or down."
    )
    if weather_df is not None and not weather_df.empty:
        st.dataframe(
            weather_df[["month", "rain_message", "rain_mm_vs_normal", "temp_message", "temp_c_vs_normal", "regions_used"]]
            .assign(month=lambda x: x["month"].dt.strftime("%b %Y"))
            .rename(columns={
                "month": "Month",
                "rain_message": "Rain outlook",
                "rain_mm_vs_normal": "Rain vs normal (mm)",
                "temp_message": "Heat outlook",
                "temp_c_vs_normal": "Heat vs normal (°C)",
                "regions_used": "Regions used",
            }),
            use_container_width=True,
            hide_index=True,
        )

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class='card'>
      <div class='card-title'>What to do now</div>
      <div class='card-value' style='color:{active_rec['action_colour']};'>{active_rec['action']}</div>
      <div class='card-sub'>{active_rec['summary']}</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='card'>
      <div class='card-title'>Rain outlook</div>
      <div class='card-value' style='color:{active_row.get('rain_colour', SLATE)};'>{active_row.get('rain_message', 'No live rain data')}</div>
      <div class='card-sub'>Compared with normal: <b>{fmt_value(active_row.get('rain_mm_vs_normal'), ' mm')}</b></div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='card'>
      <div class='card-title'>Heat outlook</div>
      <div class='card-value' style='color:{active_row.get('temp_colour', SLATE)};'>{active_row.get('temp_message', 'No live heat data')}</div>
      <div class='card-sub'>Compared with normal: <b>{fmt_value(active_row.get('temp_c_vs_normal'), '°C')}</b></div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class='card'>
      <div class='card-title'>Highest risk month</div>
      <div class='card-value'>{peak_row['month'].strftime('%b %Y')}</div>
      <div class='card-sub'>First caution month: <b style='color:{CAUTION};'>{first_caution_text}</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class='plain-note' style='margin: 10px 45px 0 45px;'>
<b>Plain read:</b> {active_rec['next_step']} Rain and heat are used as early warning signs because dry or hot months can tighten crop confidence and push price pressure higher.
</div>
""", unsafe_allow_html=True)

fig = build_chart(df, show_callouts=show_callouts)
st.plotly_chart(fig, use_container_width=True)

render_eight_month_buying_watch(df, active_row)

st.markdown(
    """
    <div style="margin-left:45px; margin-right:55px; margin-top:18px;">
      <h3 style="margin-bottom:0.65rem;">How to read this</h3>
      <ul>
        <li><b>BUY</b> means the weather and crop signals are friendlier for buying or adding cover.</li>
        <li><b>WATCH</b> means do not rush. Keep an eye on rain, heat and crop reports.</li>
        <li><b>CAUTION</b> means the market could get jumpy. Only buy what you need unless your cover is exposed.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Monthly buying guide", expanded=False):
    st.dataframe(recommendation_table(df), use_container_width=True, hide_index=True)

with st.expander("Values used by the chart", expanded=False):
    show_cols = [
        "month", "enso_index", "base_pressure", "rain_message", "rain_mm_vs_normal", "temp_message", "temp_c_vs_normal",
        "weather_adjustment", "grain_pressure", "pressure_trend", "seasonal_boost", "caution_score", "signal",
    ]
    existing = [c for c in show_cols if c in df.columns]
    st.dataframe(df[existing].assign(month=lambda x: x["month"].dt.strftime("%b %Y")), use_container_width=True, hide_index=True)

st.markdown(
    """
    <p class='source-note'>
    Weather data: Open-Meteo Seasonal Forecast API, based on ECMWF seasonal forecasts. The app uses broad area guidance, not paddock-level certainty.
    Forecasts should be checked against local agronomist updates, supplier offers and your actual grain cover position.
    </p>
    """,
    unsafe_allow_html=True,
)
