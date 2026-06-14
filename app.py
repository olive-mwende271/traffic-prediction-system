import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import joblib

# ─── Load trained model & scalers ─────────────────────────────
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("hybrid_model.keras")
    scaler_tv = joblib.load("scaler_tv.pkl")
    scaler_ctx = joblib.load("scaler_ctx.pkl")
    lookback = joblib.load("lookback.pkl")
    sarimax_cols = joblib.load("sarimax_exog_cols.pkl")
    return model, scaler_tv, scaler_ctx, lookback, sarimax_cols

model, scaler_tv, scaler_ctx, lookback, sarimax_cols = load_assets()

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


# ─── INPUT PREP ─────────────────────────────────────────────
def prepare_inputs(hour, dow, month, temp_c, clouds_pct, rain_mm, snow_mm,
                   weather_type, is_holiday, is_weekend):

    # sequence input
    X_seq = np.array([[hour, dow, month]]).reshape(1, 1, 3)

    # SARIMAX input
    X_sar = np.array([[dow, month, hour]]).reshape(1, 1, 3)

    # context input
    weather_map = {
        "Clear": 0, "Clouds": 1, "Rain": 2,
        "Drizzle": 3, "Snow": 4, "Mist": 5, "Thunderstorm": 6
    }

    X_ctx = np.array([[
        temp_c,
        clouds_pct,
        rain_mm,
        snow_mm,
        int(is_holiday),
        int(is_weekend),
        weather_map.get(weather_type, 0)
    ]])

    try:
        X_ctx = scaler_ctx.transform(X_ctx)
    except:
        pass

    return X_seq, X_sar, X_ctx


# ─── HYBRID PREDICT (FIXED) ─────────────────────────────
def hybrid_predict(hour, dow, month, temp_c, clouds_pct, rain_mm, snow_mm,
                   weather_type, is_holiday, is_weekend):

    X_seq, X_sar, X_ctx = prepare_inputs(
        hour, dow, month,
        temp_c, clouds_pct,
        rain_mm, snow_mm,
        weather_type,
        is_holiday, is_weekend
    )

    pred = model.predict([X_seq, X_sar, X_ctx], verbose=0)

    vol = scaler_tv.inverse_transform(pred.reshape(-1, 1))[0][0]

    return int(max(100, min(7280, vol)))


# ─── CLASSIFICATION ─────────────────────────────
def classify(vol):
    if vol >= 3500:
        return "heavy", "🔴 Heavy traffic", "Significant delays expected", "#D85A30"
    elif vol >= 2000:
        return "moderate", "🟡 Moderate traffic", "Some congestion likely", "#BA7517"
    elif vol >= 1200:
        return "light", "🟢 Light traffic", "Mostly free-flowing roads", "#639922"
    else:
        return "clear", "🟢 Very low traffic", "Roads are clear — smooth drive", "#1D9E75"


# ─── FACTORS ─────────────────────────────
def contributing_factors(hour, dow, month, temp_c, rain_mm, snow_mm,
                          weather_type, is_holiday, is_weekend):

    ups, downs = [], []
    is_rush = (7 <= hour <= 9) or (16 <= hour <= 18)
    is_night = hour >= 23 or hour < 5

    if is_rush: ups.append("Rush hour")
    if not is_weekend and not is_holiday: ups.append("Weekday")
    if weather_type == "Clear": ups.append("Clear weather")
    if 10 <= temp_c <= 28: ups.append("Mild temperature")
    if 4 <= month <= 10: ups.append("Warmer season")

    if is_night: downs.append("Nighttime hours")
    if is_weekend: downs.append("Weekend")
    if is_holiday: downs.append("Public holiday")
    if weather_type in ("Rain","Snow","Thunderstorm","Drizzle"):
        downs.append("Poor weather")
    if rain_mm > 0: downs.append("Recent rainfall")
    if snow_mm > 0: downs.append("Snow conditions")

    return ups, downs


# ─── UI (UNCHANGED) ─────────────────────────────
st.markdown('<div class="main-header">🚦 Traffic Volume Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid SARIMAX-LSTM model · Metro Interstate I-94</div>', unsafe_allow_html=True)

with st.container():
    st.subheader("⏰ Time & date")

    c1, c2, c3 = st.columns(3)

    with c1:
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_name = st.selectbox("Day of week", day_names, index=4)
        dow = day_names.index(dow_name)

    with c2:
        hour = st.number_input("Hour of day (0 – 23)", 0, 23, 8)

    with c3:
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        month_name = st.selectbox("Month", month_names, index=4)
        month = month_names.index(month_name) + 1

    c4, c5 = st.columns(2)
    with c4:
        is_holiday = st.checkbox("🎉 Public holiday")
    with c5:
        weekend_override = st.checkbox("📅 Force weekend", value=(dow >= 5))

    is_weekend = (dow >= 5) or weekend_override

st.divider()

with st.container():
    st.subheader("🌤 Weather conditions")

    w1, w2 = st.columns(2)

    with w1:
        weather_type = st.selectbox(
            "Weather type",
            ["Clear","Clouds","Rain","Drizzle","Snow","Mist","Thunderstorm"]
        )
        temp_c = st.number_input("Temperature (°C)", -30, 45, 18)

    with w2:
        clouds_pct = st.slider("Cloud cover (%)", 0, 100, 40)
        rain_mm = st.number_input("Rain last hour (mm)", 0.0, 200.0, 0.0)
        snow_mm = st.number_input("Snow last hour (mm)", 0.0, 50.0, 0.0)

st.divider()

if st.button("🔍 Predict Traffic Volume"):
    vol = hybrid_predict(
        hour, dow, month,
        temp_c, clouds_pct,
        rain_mm, snow_mm,
        weather_type, is_holiday, is_weekend
    )

    tier, label, description, color = classify(vol)
    pct = round((vol / 7280) * 100)

    st.markdown(f"""
    <div class="result-box result-{tier}">
        <div class="result-label">{label}</div>
        <div class="result-value" style="color:{color}">{vol:,} veh/hr</div>
        <div class="result-sub">{description}</div>
    </div>
    """, unsafe_allow_html=True)

    st.metric("Predicted volume", f"{vol:,}")
    st.metric("Congestion level", f"{pct}%")

    st.progress(pct / 100)
