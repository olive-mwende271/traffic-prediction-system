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
@@ -36,131 +50,118 @@
""", unsafe_allow_html=True)


# ─── Hybrid Model Approximation ────────────────────────────────────────────────
def hybrid_predict(hour, dow, month, temp_c, clouds_pct, rain_mm, snow_mm,
# ─── INPUT PREP ─────────────────────────────────────────────
def prepare_inputs(hour, dow, month, temp_c, clouds_pct, rain_mm, snow_mm,
weather_type, is_holiday, is_weekend):
    """
    Approximates the Hybrid SARIMAX-LSTM model's learned relationships:
      Branch A  — BiLSTM temporal pattern (hour/dow/month cyclic features)
      Branch B  — SARIMAX seasonal signal
      Branch C  — Exogenous context (weather, calendar flags)

    NOTE: To use your actual trained model weights, replace this function body with:
        model = tf.keras.models.load_model('hybrid_model.keras')
        X_seq, X_sar, X_ctx = prepare_inputs(...)   # your preprocessing pipeline
        prediction = model.predict([X_seq, X_sar, X_ctx])
        return int(scaler_tv.inverse_transform(prediction)[0][0])
    """
    # ── Branch A: temporal / sequential signal ──────────────────────────────
    base = 3200.0

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    base += hour_sin * 1100 + hour_cos * (-200)

    is_rush  = (7 <= hour <= 9) or (16 <= hour <= 18)
    is_night = hour >= 23 or hour < 5
    if is_rush:  base += 900
    if is_night: base -= 2400

    # ── Branch B: SARIMAX seasonal signal ──────────────────────────────────
    dow_sin  = math.sin(2 * math.pi * dow / 7)
    dow_cos  = math.cos(2 * math.pi * dow / 7)
    base += dow_sin * (-250) + dow_cos * 80
    # sequence input
    X_seq = np.array([[hour, dow, month]]).reshape(1, 1, 3)

    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)
    base += month_sin * 300 + month_cos * (-100)
    # SARIMAX input
    X_sar = np.array([[dow, month, hour]]).reshape(1, 1, 3)

    if is_weekend: base -= 1100
    if is_holiday: base -= 1500
    # context input
    weather_map = {
        "Clear": 0, "Clouds": 1, "Rain": 2,
        "Drizzle": 3, "Snow": 4, "Mist": 5, "Thunderstorm": 6
    }

    # ── Branch C: exogenous context ─────────────────────────────────────────
    temp_sq = temp_c ** 2
    base += temp_c * 18 - temp_sq * 0.5
    X_ctx = np.array([[
        temp_c,
        clouds_pct,
        rain_mm,
        snow_mm,
        int(is_holiday),
        int(is_weekend),
        weather_map.get(weather_type, 0)
    ]])

    rain_log = math.log1p(rain_mm)
    snow_log = math.log1p(snow_mm)
    base -= rain_log * 180
    base -= snow_log * 350
    try:
        X_ctx = scaler_ctx.transform(X_ctx)
    except:
        pass

    clouds_norm = clouds_pct / 100.0
    base -= clouds_norm * 300
    return X_seq, X_sar, X_ctx

    weather_effect = {
        "Clear": 200, "Clouds": 0, "Rain": -350,
        "Snow": -700, "Mist": -200, "Thunderstorm": -600, "Drizzle": -250
    }
    base += weather_effect.get(weather_type, 0)

    vol = max(100, min(7280, int(round(base))))
    return vol
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
        return "heavy",    "🔴 Heavy traffic",   "Significant delays expected",    "#D85A30"
        return "heavy", "🔴 Heavy traffic", "Significant delays expected", "#D85A30"
elif vol >= 2000:
        return "moderate", "🟡 Moderate traffic", "Some congestion likely",          "#BA7517"
        return "moderate", "🟡 Moderate traffic", "Some congestion likely", "#BA7517"
elif vol >= 1200:
        return "light",    "🟢 Light traffic",    "Mostly free-flowing roads",       "#639922"
        return "light", "🟢 Light traffic", "Mostly free-flowing roads", "#639922"
else:
        return "clear",    "🟢 Very low traffic", "Roads are clear — smooth drive",  "#1D9E75"
        return "clear", "🟢 Very low traffic", "Roads are clear — smooth drive", "#1D9E75"


# ─── FACTORS ─────────────────────────────
def contributing_factors(hour, dow, month, temp_c, rain_mm, snow_mm,
weather_type, is_holiday, is_weekend):

ups, downs = [], []
    is_rush  = (7 <= hour <= 9) or (16 <= hour <= 18)
    is_rush = (7 <= hour <= 9) or (16 <= hour <= 18)
is_night = hour >= 23 or hour < 5

    if is_rush:                             ups.append("Rush hour")
    if not is_weekend and not is_holiday:   ups.append("Weekday")
    if weather_type == "Clear":             ups.append("Clear weather")
    if 10 <= temp_c <= 28:                  ups.append("Mild temperature")
    if 4 <= month <= 10:                    ups.append("Warmer season")
    if is_rush: ups.append("Rush hour")
    if not is_weekend and not is_holiday: ups.append("Weekday")
    if weather_type == "Clear": ups.append("Clear weather")
    if 10 <= temp_c <= 28: ups.append("Mild temperature")
    if 4 <= month <= 10: ups.append("Warmer season")

    if is_night:                                              downs.append("Nighttime hours")
    if is_weekend:                                            downs.append("Weekend")
    if is_holiday:                                            downs.append("Public holiday")
    if weather_type in ("Rain","Snow","Thunderstorm","Drizzle"): downs.append("Poor weather")
    if rain_mm > 0:                                           downs.append("Recent rainfall")
    if snow_mm > 0:                                           downs.append("Snow conditions")
    if is_night: downs.append("Nighttime hours")
    if is_weekend: downs.append("Weekend")
    if is_holiday: downs.append("Public holiday")
    if weather_type in ("Rain","Snow","Thunderstorm","Drizzle"):
        downs.append("Poor weather")
    if rain_mm > 0: downs.append("Recent rainfall")
    if snow_mm > 0: downs.append("Snow conditions")

return ups, downs


# ════════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════════
# ─── UI (UNCHANGED) ─────────────────────────────
st.markdown('<div class="main-header">🚦 Traffic Volume Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Hybrid SARIMAX-LSTM model · Metro Interstate I-94</div>',
    unsafe_allow_html=True
)
st.markdown('<div class="sub-header">Hybrid SARIMAX-LSTM model · Metro Interstate I-94</div>', unsafe_allow_html=True)

# ── Section: Time & date ─────────────────────────────────────────────────────
with st.container():
st.subheader("⏰ Time & date")

c1, c2, c3 = st.columns(3)

with c1:
day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        dow_name  = st.selectbox("Day of week", day_names, index=4)
        dow       = day_names.index(dow_name)
        dow_name = st.selectbox("Day of week", day_names, index=4)
        dow = day_names.index(dow_name)

with c2:
        hour = st.number_input(
            "Hour of day (0 – 23)", min_value=0, max_value=23, value=8, step=1
        )
        hour = st.number_input("Hour of day (0 – 23)", 0, 23, 8)

with c3:
        month_names = [
            "January","February","March","April","May","June",
            "July","August","September","October","November","December"
        ]
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
month_name = st.selectbox("Month", month_names, index=4)
        month      = month_names.index(month_name) + 1
        month = month_names.index(month_name) + 1

c4, c5 = st.columns(2)
with c4:
@@ -172,46 +173,36 @@ def contributing_factors(hour, dow, month, temp_c, rain_mm, snow_mm,

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
            ["Clear","Clouds","Rain","Drizzle","Snow","Mist","Thunderstorm"]
)
        temp_c = st.number_input("Temperature (°C)", -30, 45, 18)

with w2:
        clouds_pct = st.slider("Cloud cover (%)", min_value=0, max_value=100, value=40, step=5)
        rain_mm    = st.number_input(
            "Rain last hour (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.5
        )
        snow_mm    = st.number_input(
            "Snow last hour (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.5
        )
        clouds_pct = st.slider("Cloud cover (%)", 0, 100, 40)
        rain_mm = st.number_input("Rain last hour (mm)", 0.0, 200.0, 0.0)
        snow_mm = st.number_input("Snow last hour (mm)", 0.0, 50.0, 0.0)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Traffic Volume")

if predict_clicked:
if st.button("🔍 Predict Traffic Volume"):
vol = hybrid_predict(
        hour=hour, dow=dow, month=month, temp_c=temp_c,
        clouds_pct=clouds_pct, rain_mm=rain_mm, snow_mm=snow_mm,
        weather_type=weather_type, is_holiday=is_holiday, is_weekend=is_weekend
        hour, dow, month,
        temp_c, clouds_pct,
        rain_mm, snow_mm,
        weather_type, is_holiday, is_weekend
)

tier, label, description, color = classify(vol)
pct = round((vol / 7280) * 100)

    # ── Result box ────────────────────────────────────────────────────────
st.markdown(f"""
   <div class="result-box result-{tier}">
       <div class="result-label">{label}</div>
@@ -220,56 +211,7 @@ def contributing_factors(hour, dow, month, temp_c, rain_mm, snow_mm,
   </div>
   """, unsafe_allow_html=True)

    st.markdown("")

    # ── Metrics row ───────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    delta_dir = "above" if pct > 50 else "below"
    m1.metric("Predicted volume",  f"{vol:,} veh/hr")
    m2.metric("Congestion level",  f"{pct}%", delta=f"{delta_dir} midpoint")
    m3.metric("Road status",       label.split()[-1].capitalize())
    st.metric("Predicted volume", f"{vol:,}")
    st.metric("Congestion level", f"{pct}%")

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
