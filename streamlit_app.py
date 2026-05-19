# grain_live_forecast_openmeteo_v3.py
# Standalone Streamlit app for farmers:
# NSW & Victoria grain buying weather outlook using Open-Meteo Seasonal Forecast API.
# Green = BUY, Amber = WATCH, Red = CAUTION.

from __future__ import annotations

import math
from datetime import datetime
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
    .block-container { padding-top: 0.8rem; padding-bottom: 1.5rem; }
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
# APP CONTENT
# -----------------------------
st.title("🌾 Grain Weather Buying Outlook")
st.caption("Simple buying guide for NSW/VIC grain. Green = BUY, amber = WATCH, red = CAUTION.")

with st.sidebar:
    st.header("Menu")
    page = st.radio(
        "Choose page",
        ["Buying Outlook", "Radar Map"],
        index=0,
    )
    st.divider()

    if page == "Radar Map":
        st.header("Radar")
        st.caption("Current rain map for quick checks.")
        data_mode = "Live weather outlook"
        show_callouts = True
    else:
        st.header("Settings")
        data_mode = st.radio(
            "Weather data",
            ["Live weather outlook", "Built-in planning values"],
            index=0,
            help="Live weather uses Open-Meteo's seasonal outlook for key NSW/VIC grain regions.",
        )
        show_callouts = st.checkbox("Show crop notes on chart", value=True)
        st.divider()
        st.markdown("### Regions included")
        for r in DEFAULT_REGIONS:
            st.caption(f"• {r['name']}")
        st.divider()
        st.caption("This is a buying aid, not a guaranteed price forecast. Use it with cover position, supplier offers and local crop reports.")

if page == "Radar Map":
    render_radar_weather_map()
    st.stop()

base_df = build_default_monthly_data()
live_status = "Using built-in planning values."
weather_df = None

if data_mode == "Live weather outlook":
    with st.spinner("Loading rain and heat outlook..."):
        weather_df, live_status = fetch_regional_weather(DEFAULT_REGIONS)

df = apply_live_weather_to_base(base_df, weather_df)
df = calculate_signals(df)

# Pick active month. For this planning example, use May 2026 if today's month is outside the demo timeline.
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

with st.expander("Live weather source", expanded=False):
    st.markdown(f"**Status:** {live_status}")
    st.markdown(
        "This app reads a seasonal outlook for key NSW/VIC grain areas and turns it into plain words: "
        "drier/wetter than normal and hotter/cooler than normal. Those signals nudge the price-pressure score up or down."
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
