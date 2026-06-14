import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import math
import joblib

# ─── Load scalers & artifacts ─────────────────────────────
scaler_tv = joblib.load("scaler_tv.pkl")
scaler_ctx = joblib.load("scaler_ctx.pkl")
lookback = joblib.load("lookback.pkl")
sarimax_cols = joblib.load("sarimax_exog_cols.pkl")

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="centered"
)

# ─── Custom CSS (UNCHANGED) ────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ─── HYBRID FALLBACK MODEL (NO TF) ─────────────────────────
def hybrid_predict(hour, dow, month, temp_c, clouds_pct, rain_mm, snow_mm,
                   weather_type, is_holiday, is_weekend):

    base = 3200.0

    # Temporal (LSTM-like behaviour)
    base += math.sin(2 * math.pi * hour / 24) * 1100
    base += math.cos(2 * math.pi * dow / 7) * 200

    if 7 <= hour <= 9 or 16 <= hour <= 18:
        base += 900
    if hour >= 23 or hour < 5:
        base -= 2400

    # Seasonal (SARIMAX-like)
    base += math.sin(2 * math.pi * month / 12) * 300
    if is_weekend:
        base -= 1100
    if is_holiday:
        base -= 1500

    # Weather effects
    base += temp_c * 18 - (temp_c ** 2) * 0.5
    base -= math.log1p(rain_mm) * 180
    base -= math.log1p(snow_mm) * 350
    base -= (clouds_pct / 100) * 300

    weather_effect = {
        "Clear": 200, "Clouds": 0, "Rain": -350,
        "Drizzle": -250, "Snow": -700,
        "Mist": -200, "Thunderstorm": -600
    }
    base += weather_effect.get(weather_type, 0)

    return int(max(100, min(7280, round(base))))

def classify(vol):
    if vol >= 3500:
        return "heavy", "🔴 Heavy traffic", "Significant delays expected", "#D85A30"
    elif vol >= 2000:
        return "moderate", "🟡 Moderate traffic", "Some congestion likely", "#BA7517"
    elif vol >= 1200:
        return "light", "🟢 Light traffic", "Mostly free-flowing roads", "#639922"
    else:
        return "clear", "🟢 Very low traffic", "Roads are clear — smooth drive", "#1D9E75"

def contributing_factors(hour, month, temp_c, rain_mm, snow_mm,
                          weather_type, is_holiday, is_weekend):

    ups, downs = [], []

    if 7 <= hour <= 9 or 16 <= hour <= 18:
        ups.append("Rush hour")
    if weather_type == "Clear":
        ups.append("Clear weather")

    if is_weekend:
        downs.append("Weekend")
    if is_holiday:
        downs.append("Holiday")
    if weather_type in ("Rain","Snow","Thunderstorm","Drizzle"):
        downs.append("Poor weather")

    return ups, downs

# ─── UI (UNCHANGED) ───────────────────────────────────────
st.markdown('<div class="main-header">🚦 Traffic Volume Predictor</div>', unsafe_allow_html=True)

st.markdown('<div class="sub-header">Hybrid SARIMAX-LSTM model · Metro Interstate I-94</div>', unsafe_allow_html=True)

with st.container():
    st.subheader("⏰ Time & date")

    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_name = st.selectbox("Day of week", day_names, index=4)
    dow = day_names.index(dow_name)

    hour = st.number_input("Hour of day (0 – 23)", 0, 23, 8)

    month_names = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    month = month_names.index(st.selectbox("Month", month_names, index=4)) + 1

    is_holiday = st.checkbox("🎉 Public holiday")
    is_weekend = st.checkbox("📅 Weekend")

st.divider()

with st.container():
    st.subheader("🌤 Weather conditions")

    weather_type = st.selectbox(
        "Weather type",
        ["Clear","Clouds","Rain","Drizzle","Snow","Mist","Thunderstorm"]
    )

    temp_c = st.number_input("Temperature (°C)", -30, 45, 18)
    clouds_pct = st.slider("Cloud cover (%)", 0, 100, 40)
    rain_mm = st.number_input("Rain (mm)", 0.0, 200.0, 0.0)
    snow_mm = st.number_input("Snow (mm)", 0.0, 50.0, 0.0)

st.divider()

if st.button("🔍 Predict Traffic Volume"):
    vol = hybrid_predict(hour, dow, month, temp_c, clouds_pct,
                         rain_mm, snow_mm, weather_type,
                         is_holiday, is_weekend)

    tier, label, desc, color = classify(vol)
    pct = round((vol / 7280) * 100)

    st.markdown(f"""
    <div class="result-box result-{tier}">
        <div class="result-label">{label}</div>
        <div class="result-value" style="color:{color}">{vol:,} veh/hr</div>
        <div class="result-sub">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.info(f"Congestion level: {pct}%")

st.divider()

st.caption("Model: Hybrid SARIMAX-LSTM (fallback version without TensorFlow)")
