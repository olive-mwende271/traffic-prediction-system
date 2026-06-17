import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import math

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {font-size: 2rem; font-weight: 600; margin-bottom: 0.2rem;}
    .sub-header  {color: #888; font-size: 0.9rem; margin-bottom: 1.5rem;}
    .result-box  {padding: 1.5rem; border-radius: 14px; margin-top: 1rem; text-align: center;}
    .result-heavy    {background:#FAECE7; border-left: 6px solid #D85A30;}
    .result-moderate {background:#FAEEDA; border-left: 6px solid #EF9F27;}
    .result-light    {background:#EAF3DE; border-left: 6px solid #97C459;}
    .result-clear    {background:#E1F5EE; border-left: 6px solid #1D9E75;}
    .result-label{font-size:0.8rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
    .result-value{font-size:2.4rem;font-weight:700;margin-bottom:4px}
    .result-sub  {font-size:0.95rem}
    .factor-up   {color:#D85A30;font-size:0.85rem}
    .factor-down {color:#1D9E75;font-size:0.85rem}
    .stButton>button {width:100%;padding:0.75rem;font-size:1rem;font-weight:600;
                      border-radius:10px;background:#1D9E75;color:white;border:none;}
    .stButton>button:hover {background:#0F6E56;}
    div[data-testid="stMetricValue"] {font-size:1.3rem!important;}
</style>
""", unsafe_allow_html=True)


# ─── Hybrid Model Approximation ────────────────────────────────────────────────
# ─── Hybrid Model Approximation (EDA-aligned) ─────────────────────────────────
def hybrid_predict(hour, dow, month, temp_c, clouds_pct, rain_mm, snow_mm,
                   weather_type, is_holiday, is_weekend):
    """
    Rebuilt to match EDA charts more faithfully:
    - Strong hourly base from 'Avg Traffic by Hour'
    - Additive DOW + Month effects (instead of aggressive multipliers)
    - Strong categorical weather/holiday overrides
    - Moderate precip/temp penalties
    """

    # ── 1. Hourly Base (most important signal) ──
    HOUR_AVG = {
        0: 800, 1: 480, 2: 300, 3: 260, 4: 370,
        5: 2050, 6: 4100, 7: 4650, 8: 4600, 9: 4350,
        10: 4250, 11: 4300, 12: 4450, 13: 4550, 14: 4750,
        15: 5500, 16: 5550, 17: 5250, 18: 4200, 19: 3250,
        20: 2850, 21: 2700, 22: 2150, 23: 1450,
    }
    base = float(HOUR_AVG.get(int(hour), 3260))

    # ── 2. Day-of-Week adjustment (additive, from DOW chart) ──
    DOW_ADJ = [3300, 3500, 3560, 3590, 3600, 2790, 2380]  # Mon=0 ... Sun=6
    overall_mean = 3260
    dow_adj = DOW_ADJ[int(dow)] - overall_mean
    base += dow_adj * 0.65   # softened multiplier so hourly still dominates

    # ── 3. Month adjustment (additive, from Month chart) ──
    MONTH_ADJ = {
        1: 3050, 2: 3200, 3: 3280, 4: 3320, 5: 3370, 6: 3320,
        7: 3220, 8: 3300, 9: 3340, 10: 3380, 11: 3130, 12: 3060,
    }
    month_adj = MONTH_ADJ.get(int(month), 3260) - overall_mean
    base += month_adj * 0.55

    # ── 4. Holiday (very strong suppressor) ──
    if is_holiday:
        base = base * 0.28 + 865 * 0.72   # blend toward holiday average

    # ── 5. Weather Condition (strong categorical) ──
    WEATHER_AVG = {
        "Clouds": 3600, "Haze": 3530, "Rain": 3300, "Drizzle": 3270,
        "Smoke": 3250, "Clear": 3100, "Snow": 2950, "Thunderstorm": 2860,
        "Mist": 2890, "Fog": 2650, "Squall": 1580,
    }
    w_target = WEATHER_AVG.get(weather_type, overall_mean)
    base = base * 0.45 + w_target * 0.55   # strong pull toward weather avg

    # ── 6. Cloud Cover (non-linear, peak at 26-50%) ──
    if clouds_pct <= 25:
        cloud_target = 3000
    elif clouds_pct <= 50:
        cloud_target = 3650
    elif clouds_pct <= 75:
        cloud_target = 3500
    else:
        cloud_target = 3280
    base = base * 0.6 + cloud_target * 0.4

    # ── 7. Precipitation penalties (additive, modest) ──
    rain_penalty = math.log1p(rain_mm) * 95
    snow_penalty = math.log1p(snow_mm) * 240
    base -= rain_penalty + snow_penalty

    # ── 8. Temperature effect ──
    if temp_c < 0:
        base += (temp_c * 22)          # cold suppression
    elif temp_c > 32:
        base -= (temp_c - 32) * 35     # extreme heat

    # Weekend override (if forced)
    if is_weekend and not is_holiday:
        base = base * 0.78

    # Final bounds
    vol = max(150, min(7280, int(round(base))))
    return vol


def classify(vol):
    if vol >= 5000:
        return "heavy",    "🔴 Heavy traffic",   "Significant delays expected",    "#D85A30"
    elif vol >= 3000:
        return "moderate", "🟡 Moderate traffic", "Some congestion likely",          "#BA7517"
    elif vol >= 1200:
        return "light",    "🟢 Light traffic",    "Mostly free-flowing roads",       "#639922"
    else:
        return "clear",    "🟢 Very low traffic", "Roads are clear — smooth drive",  "#1D9E75"


def contributing_factors(hour, dow, month, temp_c, rain_mm, snow_mm,
                          weather_type, is_holiday, is_weekend):
    ups, downs = [], []
    is_rush  = (7 <= hour <= 9) or (16 <= hour <= 18)
    is_night = hour >= 23 or hour < 5

    if is_rush:                             ups.append("Rush hour")
    if not is_weekend and not is_holiday:   ups.append("Weekday")
    if weather_type == "Clear":             ups.append("Clear weather")
    if 10 <= temp_c <= 28:                  ups.append("Mild temperature")
    if 4 <= month <= 10:                    ups.append("Warmer season")

    if is_night:                                              downs.append("Nighttime hours")
    if is_weekend:                                            downs.append("Weekend")
    if is_holiday:                                            downs.append("Public holiday")
    if weather_type in ("Rain","Snow","Thunderstorm","Drizzle"): downs.append("Poor weather")
    if rain_mm > 0:                                           downs.append("Recent rainfall")
    if snow_mm > 0:                                           downs.append("Snow conditions")

    return ups, downs


# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-header">🚦 Traffic Volume Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid SARIMAX-LSTM model · Metro Interstate I-94</div>',
    unsafe_allow_html=True
)

# ── Section: Time & date ─────────────────────────────────────────────────────
with st.container():
    st.subheader("⏰ Time & date")
    c1, c2, c3 = st.columns(3)

    with c1:
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_name  = st.selectbox("Day of week", day_names, index=4)
        dow       = day_names.index(dow_name)

    with c2:
        hour = st.number_input(
            "Hour of day (0 – 23)", min_value=0, max_value=23, value=8, step=1
        )

    with c3:
        month_names = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]
        month_name = st.selectbox("Month", month_names, index=4)
        month      = month_names.index(month_name) + 1

    c4, c5 = st.columns(2)
    with c4:
        is_holiday = st.checkbox("🎉 Public holiday")
    with c5:
        weekend_override = st.checkbox("📅 Force weekend", value=(dow >= 5))

    is_weekend = (dow >= 5) or weekend_override

st.divider()

# ── Section: Weather ──────────────────────────────────────────────────────────
with st.container():
    st.subheader("🌤 Weather conditions")
    w1, w2 = st.columns(2)

    with w1:
        weather_type = st.selectbox(
            "Weather type",
            ["Clear","Clouds","Rain","Drizzle","Snow","Mist","Thunderstorm"],
            index=1
        )
        temp_c = st.number_input(
            "Temperature (°C)", min_value=-30, max_value=45, value=18, step=1
        )

    with w2:
        clouds_pct = st.slider("Cloud cover (%)", min_value=0, max_value=100, value=40, step=5)
        rain_mm    = st.number_input(
            "Rain last hour (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.5
        )
        snow_mm    = st.number_input(
            "Snow last hour (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5
        )

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Traffic Volume")

if predict_clicked:
    vol = hybrid_predict(
        hour=hour, dow=dow, month=month, temp_c=temp_c,
        clouds_pct=clouds_pct, rain_mm=rain_mm, snow_mm=snow_mm,
        weather_type=weather_type, is_holiday=is_holiday, is_weekend=is_weekend
    )

    tier, label, description, color = classify(vol)
    pct = round((vol / 7280) * 100)

    # ── Result box ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-box result-{tier}">
        <div class="result-label">{label}</div>
        <div class="result-value" style="color:{color}">{vol:,} veh/hr</div>
        <div class="result-sub">{description}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    delta_dir = "above" if pct > 50 else "below"
    m1.metric("Predicted volume",  f"{vol:,} veh/hr")
    m2.metric("Congestion level",  f"{pct}%", delta=f"{delta_dir} midpoint")
    m3.metric("Road status",       label.split()[-1].capitalize())

    # ── Progress bar ──────────────────────────────────────────────────────
    st.progress(pct / 100)
    st.caption("Volume as % of max observed on I-94 (7,280 veh/hr)")

    # ── Contributing factors ──────────────────────────────────────────────
    st.markdown("#### Key factors influencing this prediction")
    ups, downs = contributing_factors(
        hour, dow, month, temp_c, rain_mm, snow_mm,
        weather_type, is_holiday, is_weekend
    )
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown("**⬆ Increasing volume**")
        if ups:
            for f in ups:
                st.markdown(f"<span class='factor-up'>▲ {f}</span>", unsafe_allow_html=True)
        else:
            st.markdown("*None detected*")
    with fc2:
        st.markdown("**⬇ Reducing volume**")
        if downs:
            for f in downs:
                st.markdown(f"<span class='factor-down'>▼ {f}</span>", unsafe_allow_html=True)
        else:
            st.markdown("*None detected*")

    # ── Travel tip ────────────────────────────────────────────────────────
    tips = {
        "heavy":    "⚠️ Consider leaving earlier or taking an alternative route.",
        "moderate": "🕐 Allow extra travel time — some slowdowns are likely.",
        "light":    "✅ Good time to travel. Roads should flow smoothly.",
        "clear":    "✅ Excellent conditions — very little traffic expected."
    }
    st.info(tips[tier])


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Model:** Hybrid SARIMAX-LSTM  ·  "
    "**Dataset:** Metro Interstate Traffic Volume (UCI ML Repository)  ·  "
    "To plug in your trained Keras weights, replace the `hybrid_predict()` function with "
    "`model.predict([X_seq, X_sar, X_ctx])` after loading your `.keras` file."
)
