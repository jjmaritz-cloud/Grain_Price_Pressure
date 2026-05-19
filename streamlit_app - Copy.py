# grain_seasonal_tracker_better_buy_wait.py
# Streamlit app: NSW & Victoria grain seasonal tracker with stronger seasonal context and BUY / WAIT indicators.
# Green = BUY. Red = WAIT.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(
    page_title="NSW & Victoria Grain Seasonal Tracker",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.0rem; padding-bottom: 1.5rem; }
    .card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 16px 18px;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(15,23,42,0.06);
    }
    .card-title { color:#64748b; font-size:0.78rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; }
    .card-value { color:#0f172a; font-size:1.35rem; font-weight:900; margin-top:2px; }
    .card-sub { color:#64748b; font-size:0.86rem; margin-top:5px; line-height:1.3; }
    .note { color:#64748b; font-size:0.86rem; }

    .watch-section {
        margin-left: 45px;
        margin-right: 170px;
        margin-top: 10px;
    }
    .watch-section-title {
        font-size: 1.28rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 10px;
    }
    .watch-grid {
        display: grid;
        grid-template-columns: repeat(8, minmax(0, 1fr));
        gap: 10px;
        align-items: stretch;
        width: 100%;
    }
    .watch-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 13px 11px;
        background: #ffffff;
        min-height: 205px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
        overflow: hidden;
    }
    .watch-month {
        font-size: 0.68rem;
        color: #64748b;
        font-weight: 900;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .watch-action {
        font-size: 0.86rem;
        font-weight: 900;
        line-height: 1.12;
        margin-top: 7px;
        min-height: 42px;
    }
    .watch-body {
        color: #475569;
        font-size: 0.74rem;
        line-height: 1.28;
        margin-top: 7px;
    }
    @media (max-width: 1250px) {
        .watch-section {
            margin-left: 10px;
            margin-right: 10px;
        }
        .watch-grid {
            grid-template-columns: repeat(4, minmax(0, 1fr));
        }
    }
    @media (max-width: 760px) {
        .watch-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    .ai-panel {
        border: 1px solid #e2e8f0;
        border-radius: 22px;
        padding: 18px 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        margin-top: 12px;
        margin-bottom: 18px;
    }
    .ai-title {
        font-size: 1.35rem;
        font-weight: 900;
        color: #0f172a;
        margin-bottom: 4px;
    }
    .ai-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 14px;
    }
    .ai-card {
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 15px 16px;
        background: #ffffff;
        min-height: 142px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }
    .ai-card-title {
        font-size: 0.75rem;
        color: #64748b;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .ai-card-value {
        font-size: 1.25rem;
        font-weight: 900;
        margin-top: 8px;
        line-height: 1.15;
    }
    .ai-card-body {
        color: #475569;
        font-size: 0.86rem;
        line-height: 1.35;
        margin-top: 8px;
    }
    .ai-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 9px;
        font-size: 0.72rem;
        font-weight: 900;
        color: white;
        margin-right: 6px;
        margin-top: 7px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# COLOURS
# -----------------------------
BUY = "#16a34a"       # green
MONITOR = "#f59e0b"   # amber
WAIT = "#dc2626"      # red
ORANGE = "#ea580c"
BLUE = "#2563eb"
SLATE = "#334155"
GRID = "#dbe4ef"

START_DATE = "2026-01-01"
END_DATE = "2027-07-31"
EL_NINO_THRESHOLD = 0.8
LA_NINA_THRESHOLD = -0.8


# -----------------------------
# BUILT-IN MONTHLY DATA
# Replace this later with your preferred live / CSV / API values.
# -----------------------------
def build_default_monthly_data() -> pd.DataFrame:
    months = pd.date_range(START_DATE, END_DATE, freq="MS")

    # Conceptual ENSO / Nino3.4 style values.
    # Visual only unless you replace with real observed/model data.
    enso = [
        -0.25, -0.10, 0.05, 0.25, 0.52, 0.65, 0.78, 0.90, 0.95, 1.00, 0.92, 0.75,
        0.62, 0.55, 0.48, 0.42, 0.35, 0.25, 0.15,
    ]

    # Conceptual grain price pressure score.
    # Higher = more supply/crop risk and more caution.
    grain_pressure = [
        38, 42, 47, 50, 51, 58, 65, 68, 66, 64, 62, 55,
        52, 54, 60, 64, 68, 70, 69,
    ]

    df = pd.DataFrame({
        "month": months,
        "enso_index": enso[: len(months)],
        "grain_pressure": grain_pressure[: len(months)],
    })

    # Use May 2026 as the current cutover for the planning example.
    df["period_type"] = np.where(
        df["month"] <= pd.Timestamp("2026-05-01"),
        "Observed / current",
        "Forward planning view",
    )
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
    dict(start="2026-11-15", end="2027-01-31", label="Harvest / quality realised", risk="Medium", colour="rgba(100,116,139,0.18)"),
    dict(start="2027-02-01", end="2027-05-31", label="2027 crop intent / autumn break", risk="Medium", colour="rgba(14,165,233,0.13)"),
    dict(start="2027-06-01", end="2027-07-31", label="2027 early establishment", risk="Medium", colour="rgba(34,197,94,0.13)"),
]

CROP_NOTES = [
    dict(date="2026-03-15", text="Wet March rebuilt\nmoisture in parts\nof Vic / NSW", y=56),
    dict(date="2026-05-15", text="ENSO neutral, but\ndrier east risk flagged", y=63),
    dict(date="2026-07-15", text="Follow-up rain drives\nestablishment confidence", y=78),
    dict(date="2026-09-15", text="Flowering + frost\nyield-risk window", y=81),
    dict(date="2026-10-25", text="Grain fill / spring finish\nis the critical price-risk window", y=76),
    dict(date="2026-12-10", text="Harvest quality, logistics\nand supply realised", y=66),
    dict(date="2027-04-01", text="Autumn break drives\n2027 sowing confidence", y=80),
]


# -----------------------------
# BUY / WAIT LOGIC
# -----------------------------
def seasonal_boost(month: pd.Timestamp) -> int:
    m = int(month.month)
    if m in (9, 10):
        return 10   # flowering / frost / spring finish risk
    if m == 11:
        return 7
    if m in (6, 7, 8):
        return 5    # establishment and biomass
    if m in (3, 4, 5):
        return 3    # autumn break / sowing
    return 0


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pressure_trend"] = out["grain_pressure"].diff().fillna(0)
    out["seasonal_boost"] = out["month"].apply(seasonal_boost)
    out["caution_score"] = out["grain_pressure"] + (out["pressure_trend"] * 1.5) + out["seasonal_boost"]

    def classify(score: float) -> str:
        # Green = buy, red = wait.
        if score < 52:
            return "BUY"
        if score < 67:
            return "MONITOR"
        return "WAIT"

    out["signal"] = out["caution_score"].apply(classify)
    out["signal_colour"] = out["signal"].map({"BUY": BUY, "MONITOR": MONITOR, "WAIT": WAIT})
    out["signal_reason"] = np.select(
        [
            out["signal"].eq("BUY"),
            out["signal"].eq("MONITOR"),
            out["signal"].eq("WAIT"),
        ],
        [
            "Lower/easing pressure window",
            "Watch weather and crop reports",
            "Critical crop/supply risk window",
        ],
        default="",
    )
    return out


# -----------------------------
# OPTIONAL CSV HANDLING
# -----------------------------
def standardise_uploaded_data(uploaded_file):
    if uploaded_file is None:
        return None

    raw = pd.read_csv(uploaded_file)
    cols = {c.lower().strip(): c for c in raw.columns}

    def find_col(options):
        for c in options:
            if c in cols:
                return cols[c]
        return None

    date_col = find_col(["month", "date", "period"])
    enso_col = find_col(["enso_index", "nino34", "nino_34", "bom_enso", "enso"])
    pressure_col = find_col(["grain_pressure", "price_pressure", "pressure", "grain_price_pressure"])

    if date_col is None or enso_col is None or pressure_col is None:
        st.warning("CSV ignored. It needs columns like: month/date, enso_index/nino34, grain_pressure/price_pressure.")
        return None

    df = pd.DataFrame({
        "month": pd.to_datetime(raw[date_col], errors="coerce"),
        "enso_index": pd.to_numeric(raw[enso_col], errors="coerce"),
        "grain_pressure": pd.to_numeric(raw[pressure_col], errors="coerce"),
    }).dropna()

    df["month"] = df["month"].dt.to_period("M").dt.to_timestamp()
    df = df.groupby("month", as_index=False).mean(numeric_only=True)
    df["period_type"] = np.where(
        df["month"] <= pd.Timestamp.today().replace(day=1),
        "Observed / current",
        "Forward planning view",
    )
    return df


# -----------------------------
# CHART HELPERS
# -----------------------------
def month_end(x: pd.Timestamp) -> pd.Timestamp:
    return x + pd.offsets.MonthEnd(1)


def add_pressure_zones(fig: go.Figure) -> None:
    # Horizontal criticality zones. This is what was missing visually.
    zones = [
        (20, 52, "rgba(22,163,74,0.12)", "BUY ZONE", BUY),
        (52, 67, "rgba(245,158,11,0.13)", "MONITOR", MONITOR),
        (67, 90, "rgba(220,38,38,0.12)", "WAIT / CRITICAL", WAIT),
    ]
    for y0, y1, colour, label, font_colour in zones:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=colour, line_width=0, row=1, col=1, secondary_y=False)
        fig.add_annotation(
            x=pd.Timestamp("2026-01-08"),
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


def add_season_bands_to_main(fig: go.Figure) -> None:
    for w in SEASON_WINDOWS:
        fig.add_vrect(
            x0=w["start"],
            x1=w["end"],
            fillcolor=w["colour"],
            line_width=0,
            layer="below",
            row=1,
            col=1,
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
        subplot_titles=("", "Seasonal crop risk windows", "Indicative grain buying signal"),
    )

    # Invisible date anchors so all subplot x-axes behave as date axes.
    for row in [2, 3]:
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)],
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

    add_pressure_zones(fig)
    add_season_bands_to_main(fig)

    observed = df[df["period_type"].str.contains("Observed", case=False, na=False)]
    forward = df[~df.index.isin(observed.index)]

    # Grain pressure line: keep it bold and close to the previous better look.
    if not observed.empty:
        fig.add_trace(
            go.Scatter(
                x=observed["month"],
                y=observed["grain_pressure"],
                mode="lines+markers",
                name="Grain pressure - observed/current",
                line=dict(color=ORANGE, width=5),
                marker=dict(size=13, color=observed["signal_colour"], line=dict(color="white", width=2)),
                customdata=np.stack([observed["signal"], observed["caution_score"], observed["signal_reason"]], axis=-1),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Grain pressure: %{y:.0f}/100<br>"
                    "Signal: <b>%{customdata[0]}</b><br>"
                    "Caution score: %{customdata[1]:.0f}<br>"
                    "%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    if not forward.empty:
        forward_plot = pd.concat([observed.tail(1), forward], ignore_index=True) if not observed.empty else forward
        fig.add_trace(
            go.Scatter(
                x=forward_plot["month"],
                y=forward_plot["grain_pressure"],
                mode="lines+markers",
                name="Grain pressure - forward planning view",
                line=dict(color="#fb923c", width=5, dash="dash"),
                marker=dict(size=13, color=forward_plot["signal_colour"], line=dict(color="white", width=2)),
                customdata=np.stack([forward_plot["signal"], forward_plot["caution_score"], forward_plot["signal_reason"]], axis=-1),
                hovertemplate=(
                    "<b>%{x|%b %Y}</b><br>"
                    "Grain pressure: %{y:.0f}/100<br>"
                    "Signal: <b>%{customdata[0]}</b><br>"
                    "Caution score: %{customdata[1]:.0f}<br>"
                    "%{customdata[2]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
            secondary_y=False,
        )

    # ENSO line on secondary y-axis.
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["enso_index"],
            mode="lines+markers",
            name="ENSO / Nino3.4 style index",
            line=dict(color=BLUE, width=3, dash="dot"),
            marker=dict(size=7, color=BLUE),
            hovertemplate="<b>%{x|%b %Y}</b><br>ENSO index: %{y:.2f} °C<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # ENSO threshold lines.
    fig.add_hline(
        y=EL_NINO_THRESHOLD,
        line_dash="dash",
        line_color="#ef4444",
        line_width=1.5,
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_hline(
        y=LA_NINA_THRESHOLD,
        line_dash="dash",
        line_color="#3b82f6",
        line_width=1.5,
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.add_annotation(
        x=pd.Timestamp("2027-07-15"),
        y=EL_NINO_THRESHOLD,
        text="El Niño +0.8°C",
        showarrow=False,
        font=dict(size=11, color="#b91c1c"),
        bgcolor="rgba(255,255,255,0.8)",
        row=1,
        col=1,
        secondary_y=True,
    )

    # Seasonal row.
    for w in SEASON_WINDOWS:
        start = pd.Timestamp(w["start"])
        end = pd.Timestamp(w["end"])
        mid = start + (end - start) / 2
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=w["colour"].replace("0.13", "0.65").replace("0.16", "0.65").replace("0.18", "0.65").replace("0.22", "0.65"),
            line=dict(color="rgba(148,163,184,0.45)", width=1),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=mid,
            y=0.62,
            text=f"<b>{w['label']}</b><br><span style='font-size:10px'>Risk: {w['risk']}</span>",
            showarrow=False,
            font=dict(size=12, color="#0f172a"),
            row=2,
            col=1,
        )

    # BUY/MONITOR/WAIT signal strip.
    for _, r in df.iterrows():
        start = r["month"]
        end = month_end(start)
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=r["signal_colour"],
            opacity=0.92,
            line=dict(color="white", width=1),
            row=3,
            col=1,
        )
        fig.add_annotation(
            x=start + pd.Timedelta(days=14),
            y=0.52,
            text=f"<b>{r['signal']}</b>",
            showarrow=False,
            font=dict(size=11, color="white"),
            row=3,
            col=1,
        )

    # Crop callouts.
    if show_callouts:
        for note in CROP_NOTES:
            x = pd.Timestamp(note["date"])
            fig.add_annotation(
                x=x,
                y=note["y"],
                text=note["text"],
                showarrow=True,
                arrowhead=2,
                arrowsize=0.8,
                arrowwidth=1.2,
                arrowcolor="#475569",
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                borderpad=5,
                font=dict(size=11, color="#0f172a"),
                row=1,
                col=1,
                secondary_y=False,
            )

    # Layout.
    fig.update_layout(
        height=860,
        margin=dict(l=45, r=55, t=90, b=35),
        title=dict(
            text="NSW & Victoria Grain Seasonal Tracker | ENSO, seasonal crop risk and BUY / WAIT signal",
            x=0.5,
            xanchor="center",
            font=dict(size=24, color="#0f172a"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
    )

    for row in [1, 2, 3]:
        fig.update_xaxes(
            tickformat="%b\n%Y",
            dtick="M1",
            range=[pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)],
            showgrid=True,
            gridcolor=GRID,
            zeroline=False,
            row=row,
            col=1,
        )

    fig.update_yaxes(
        title_text="Indicative grain price pressure score",
        range=[20, 90],
        tickvals=[30, 40, 50, 60, 70, 80, 90],
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        row=1,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="ENSO index °C",
        range=[-1.2, 1.4],
        showgrid=False,
        zeroline=False,
        row=1,
        col=1,
        secondary_y=True,
    )
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=2, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, showgrid=False, zeroline=False, row=3, col=1)

    return fig




# -----------------------------
# AI BUYING / WAITING RECOMMENDATIONS
# -----------------------------
def crop_window_for_month(month: pd.Timestamp) -> tuple[str, str]:
    """Return the seasonal crop window label and risk level for a month."""
    month_start = pd.Timestamp(month).to_period("M").to_timestamp()
    month_mid = month_start + pd.Timedelta(days=14)

    for w in SEASON_WINDOWS:
        if pd.Timestamp(w["start"]) <= month_mid <= pd.Timestamp(w["end"]):
            return w["label"], w["risk"]

    return "Outside defined crop window", "Low"


def enso_assessment(enso_index: float) -> tuple[str, str, int, str]:
    """Classify ENSO contribution to grain buying risk."""
    if enso_index >= 1.2:
        return "Strong El Niño risk", WAIT, 10, "ENSO is well above the El Niño threshold, lifting dry-season concern."
    if enso_index >= EL_NINO_THRESHOLD:
        return "El Niño threshold reached", WAIT, 7, "ENSO is at or above +0.8°C, so dry/warm risk deserves extra caution."
    if enso_index >= 0.55:
        return "El Niño watch zone", MONITOR, 4, "ENSO is below threshold but close enough to watch closely."
    if enso_index <= LA_NINA_THRESHOLD:
        return "La Niña threshold reached", BUY, -4, "Cooler ENSO signal may reduce dry-pressure risk, but local rainfall still matters."
    return "ENSO neutral", SLATE, 0, "ENSO is neutral, so crop risk is driven more by local rainfall timing and seasonal windows."


def crop_risk_assessment(risk: str, window_label: str) -> tuple[str, str, int, str]:
    risk_clean = risk.lower()

    if "high" in risk_clean and "medium" not in risk_clean:
        return "High crop-risk window", WAIT, 9, f"{window_label} is a high-risk production window."
    if "high" in risk_clean and "medium" in risk_clean:
        return "Medium-high crop-risk window", MONITOR, 6, f"{window_label} can quickly change crop confidence if follow-up rain fails."
    if "medium" in risk_clean:
        return "Medium crop-risk window", MONITOR, 3, f"{window_label} should be watched, but it is not the peak risk window."
    return "Lower crop-risk window", BUY, 0, f"{window_label} is normally a lower-risk planning period."


def price_pressure_assessment(grain_pressure: float, pressure_trend: float) -> tuple[str, str, int, str]:
    if grain_pressure >= 67 and pressure_trend > 0:
        return "High and rising price pressure", WAIT, 10, "Price pressure is already high and still rising."
    if grain_pressure >= 67:
        return "High price pressure", WAIT, 8, "Price pressure is high, so avoid chasing unless cover is exposed."
    if grain_pressure >= 52 and pressure_trend > 0:
        return "Medium pressure, rising", MONITOR, 5, "Pressure is not critical yet, but the trend is moving against buyers."
    if grain_pressure >= 52:
        return "Medium pressure", MONITOR, 3, "Pressure is in the monitor zone."
    if pressure_trend > 2:
        return "Low pressure but rising", MONITOR, 2, "The current level is favourable, but it is starting to move up."
    return "Low/easing price pressure", BUY, -2, "Pressure is low or easing, which supports a buying opportunity."


def ai_buying_recommendation(row: pd.Series) -> dict:
    month = row["month"]
    enso_index = float(row["enso_index"])
    grain_pressure = float(row["grain_pressure"])
    pressure_trend = float(row["pressure_trend"])
    caution_score = float(row["caution_score"])

    window_label, crop_risk = crop_window_for_month(month)

    enso_label, enso_colour, enso_points, enso_reason = enso_assessment(enso_index)
    crop_label, crop_colour, crop_points, crop_reason = crop_risk_assessment(crop_risk, window_label)
    price_label, price_colour, price_points, price_reason = price_pressure_assessment(grain_pressure, pressure_trend)

    ai_score = caution_score + enso_points + crop_points + price_points

    if ai_score < 58:
        action = "BUY / LAYER COVER"
        action_colour = BUY
        confidence = "Favourable"
        summary = "Conditions support buying or layering additional cover, especially if physical cover is below target."
        next_step = "Consider staged buying rather than waiting for a perfect low."
    elif ai_score < 74:
        action = "MONITOR / BUY SELECTIVELY"
        action_colour = MONITOR
        confidence = "Mixed"
        summary = "Signals are mixed. Buying should be selective and linked to cover gaps, pricing opportunities and rainfall updates."
        next_step = "Watch rainfall follow-up, crop reports and weekly price movement before committing larger volume."
    else:
        action = "WAIT / COVER ESSENTIAL ONLY"
        action_colour = WAIT
        confidence = "Caution"
        summary = "Risk is high enough that chasing price should be avoided unless cover is exposed."
        next_step = "Only cover essential tonnes or known exposure; wait for clearer crop or price signal before larger buys."

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
        "chart_signal": row["signal"],
    }


def build_ai_recommendation_table(df: pd.DataFrame) -> pd.DataFrame:
    recs = [ai_buying_recommendation(row) for _, row in df.iterrows()]

    out = pd.DataFrame(
        [
            {
                "Month": r["month_label"],
                "AI Action": r["action"],
                "AI Score": r["ai_score"],
                "Confidence": r["confidence"],
                "Chart Signal": r["chart_signal"],
                "ENSO": r["enso_label"],
                "Crop Window": r["window_label"],
                "Crop Risk": r["crop_risk"],
                "Grain Pressure": round(r["grain_pressure"], 0),
                "Pressure Trend": round(r["pressure_trend"], 1),
                "Next Step": r["next_step"],
            }
            for r in recs
        ]
    )

    return out


def render_ai_recommendations(df: pd.DataFrame) -> None:
    # Use active month already calculated above where possible.
    active_recommendation = ai_buying_recommendation(active_row)

    next_three = df[df["month"] > active_row["month"]].head(3)
    next_recommendations = [ai_buying_recommendation(row) for _, row in next_three.iterrows()]

    st.markdown(
        f"""
        <div class="ai-panel">
          <div class="ai-title">🤖 AI buying / waiting recommendation</div>
          <div class="ai-subtitle">
            Based on ENSO threshold position, crop-risk window, grain price pressure, pressure trend and seasonal caution score.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.15, 1, 1, 1])

    with c1:
        r = active_recommendation
        st.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-card-title">Current recommendation</div>
              <div class="ai-card-value" style="color:{r['action_colour']};">{r['action']}</div>
              <div class="ai-card-body">
                <b>{r['month_label']}</b> | AI score: <b>{r['ai_score']}</b><br>
                {r['summary']}<br>
                <span class="ai-pill" style="background:{r['action_colour']};">{r['confidence']}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        r = active_recommendation
        st.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-card-title">ENSO signal</div>
              <div class="ai-card-value" style="color:{r['enso_colour']};">{r['enso_label']}</div>
              <div class="ai-card-body">
                ENSO index: <b>{r['enso_index']:.2f}°C</b><br>
                {r['enso_reason']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        r = active_recommendation
        st.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-card-title">Crop-risk window</div>
              <div class="ai-card-value" style="color:{r['crop_colour']};">{r['crop_risk']}</div>
              <div class="ai-card-body">
                <b>{r['window_label']}</b><br>
                {r['crop_reason']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        r = active_recommendation
        st.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-card-title">Price pressure</div>
              <div class="ai-card-value" style="color:{r['price_colour']};">{r['price_label']}</div>
              <div class="ai-card-body">
                Pressure: <b>{r['grain_pressure']:.0f}/100</b><br>
                Monthly trend: <b>{r['pressure_trend']:+.1f}</b><br>
                {r['price_reason']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Next 3-month buying watch")
    cols = st.columns(3)

    for col, r in zip(cols, next_recommendations):
        col.markdown(
            f"""
            <div class="ai-card">
              <div class="ai-card-title">{r['month_label']}</div>
              <div class="ai-card-value" style="color:{r['action_colour']};">{r['action']}</div>
              <div class="ai-card-body">
                AI score: <b>{r['ai_score']}</b><br>
                {r['window_label']}<br>
                <b>Next step:</b> {r['next_step']}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("AI recommendation table", expanded=False):
        st.dataframe(
            build_ai_recommendation_table(df),
            use_container_width=True,
            hide_index=True,
        )




def render_eight_month_buying_watch(df: pd.DataFrame) -> None:
    """Compact 8-month buying watch row aligned to the chart edges."""
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
                AI score: <b>{r['ai_score']}</b><br>
                {r['window_label']}<br>
                <b>Next:</b> {r['next_step']}
              </div>
            </div>
            """
        )

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: "Source Sans Pro", Arial, sans-serif;
            background: transparent;
        }}
        .watch-section {{
            margin-left: 45px;
            margin-right: 170px;
            margin-top: 0;
        }}
        .watch-section-title {{
            font-size: 1.28rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 10px;
        }}
        .watch-grid {{
            display: grid;
            grid-template-columns: repeat(8, minmax(0, 1fr));
            gap: 10px;
            align-items: stretch;
            width: 100%;
        }}
        .watch-card {{
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 13px 11px;
            background: #ffffff;
            min-height: 205px;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
            overflow: hidden;
            box-sizing: border-box;
        }}
        .watch-month {{
            font-size: 0.68rem;
            color: #64748b;
            font-weight: 900;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            white-space: nowrap;
        }}
        .watch-action {{
            font-size: 0.86rem;
            font-weight: 900;
            line-height: 1.12;
            margin-top: 7px;
            min-height: 42px;
        }}
        .watch-body {{
            color: #475569;
            font-size: 0.74rem;
            line-height: 1.28;
            margin-top: 7px;
        }}
        @media (max-width: 1250px) {{
            .watch-section {{
                margin-left: 10px;
                margin-right: 10px;
            }}
            .watch-grid {{
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }}
        }}
        @media (max-width: 760px) {{
            .watch-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}
      </style>
    </head>
    <body>
        <div class="watch-section">
            <div class="watch-section-title">Next 8-month buying watch</div>
            <div class="watch-grid">
                {''.join(cards_html)}
            </div>
        </div>
    </body>
    </html>
    """

    components.html(html, height=310, scrolling=False)

# -----------------------------
# APP CONTENT
# -----------------------------
st.title("🌾 NSW & Victoria Grain Seasonal Tracker")
st.caption("A visual planning tool. Green = BUY, amber = MONITOR, red = WAIT. Seasonal windows explain why the risk changes through the crop year.")

with st.sidebar:
    st.header("Settings")
    st.markdown("### Signal colours")
    st.markdown("🟢 **BUY** = more favourable buying window")
    st.markdown("🟠 **MONITOR** = keep watching crop/weather signals")
    st.markdown("🔴 **WAIT / CRITICAL** = higher crop/supply risk window")
    st.divider()
    uploaded = st.file_uploader(
        "Optional: upload monthly CSV",
        type=["csv"],
        help="Needs columns like month/date, enso_index/nino34, grain_pressure/price_pressure.",
    )
    show_callouts = st.checkbox("Show crop-season callouts", value=True)
    st.divider()
    st.markdown("### Logic")
    st.caption("Signal = grain pressure + pressure trend + seasonal risk boost. You can tune the thresholds in calculate_signals().")

uploaded_df = standardise_uploaded_data(uploaded)
df = uploaded_df if uploaded_df is not None else build_default_monthly_data()
df = calculate_signals(df)

# Pick active month. For this planning example, use May 2026 if today's month is outside the timeline.
today_month = pd.Timestamp.today().replace(day=1)
if today_month < df["month"].min() or today_month > df["month"].max():
    active_month = pd.Timestamp("2026-05-01")
else:
    active_month = today_month
active_row = df.iloc[(df["month"] - active_month).abs().idxmin()]

next_critical = df[df["signal"].eq("WAIT")]
first_critical_text = "None in range" if next_critical.empty else next_critical.iloc[0]["month"].strftime("%b %Y")
peak_row = df.iloc[df["caution_score"].idxmax()]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class='card'>
          <div class='card-title'>Current signal</div>
          <div class='card-value' style='color:{active_row['signal_colour']};'>{active_row['signal']}</div>
          <div class='card-sub'>{active_row['month'].strftime('%b %Y')} | {active_row['signal_reason']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class='card'>
          <div class='card-title'>Grain pressure</div>
          <div class='card-value'>{active_row['grain_pressure']:.0f} / 100</div>
          <div class='card-sub'>Higher = more supply/crop pressure</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class='card'>
          <div class='card-title'>First critical window</div>
          <div class='card-value' style='color:{WAIT};'>{first_critical_text}</div>
          <div class='card-sub'>Red months are where the model says wait/caution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class='card'>
          <div class='card-title'>Peak caution</div>
          <div class='card-value'>{peak_row['month'].strftime('%b %Y')}</div>
          <div class='card-sub'>Caution score {peak_row['caution_score']:.0f} | {peak_row['signal']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

fig = build_chart(df, show_callouts=show_callouts)
st.plotly_chart(fig, use_container_width=True)

render_eight_month_buying_watch(df)

st.markdown(
    """
    <div style="margin-left:45px; margin-right:55px; margin-top:18px;">
      <h3 style="margin-bottom:0.65rem;">Practical read</h3>
      <ul>
        <li><b>Green months</b> are the better buying windows because pressure is lower or easing.</li>
        <li><b>Amber months</b> are watch periods where follow-up rain, establishment reports, frost risk and spring forecasts matter.</li>
        <li><b>Red months</b> are the higher-risk windows where crop size and grain quality can change quickly. That does not mean never buy; it means buying needs stronger justification and tighter risk control.</li>
      </ul>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Monthly values used by the chart", expanded=False):
    st.dataframe(
        df[["month", "enso_index", "grain_pressure", "pressure_trend", "seasonal_boost", "caution_score", "signal", "period_type"]]
        .assign(month=lambda x: x["month"].dt.strftime("%b %Y")),
        use_container_width=True,
        hide_index=True,
    )

st.markdown(
    """
    <p class='note'>
    Note: the built-in values are planning values so the app runs without manual imports. Replace them with live BOM/NOAA/market values later.
    The seasonal overlays are the main purpose here: they show why grain pressure becomes more critical around establishment, flowering, grain fill and harvest.
    </p>
    """,
    unsafe_allow_html=True,
)
