"""
🚦 Traffic Volume Predictor — Hybrid SARIMA-LSTM
Fully integrated with the trained model from the research notebook.

HOW TO GENERATE THE MODEL FILES (run this in your Colab notebook AFTER training):
─────────────────────────────────────────────────────────────────────────────────
    import joblib, os

    # 1. Re-train the hybrid model on the FULL modelling dataset (not just last CV fold)
    model_final = build_hybrid_model(LOOKBACK, n_features, n_ctx)
    es  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)
    model_final.fit(
        [X_seq_h, X_sar_h, X_ctx_h], y_h,
        epochs=70, batch_size=64,
        validation_split=0.1,
        callbacks=[es, rlr],
        verbose=1
    )

    # 2. Save the Keras model
    model_final.save('hybrid_model.keras')

    # 3. Save the scalers and dataset metadata
    joblib.dump(scaler_mv,  'scaler_mv.pkl')   # MinMaxScaler for all 26 features
    joblib.dump(scaler_tv,  'scaler_tv.pkl')   # MinMaxScaler for traffic_volume only
    joblib.dump(scaler_ctx, 'scaler_ctx.pkl')  # MinMaxScaler for the 11 context features

    # 4. Save the last 24 hours of scaled data (needed as the sequence seed for inference)
    np.save('last_sequence.npy', mv_scaled[-24:])        # shape (24, n_features)
    np.save('sarimax_signal_last.npy', sarimax_signal_sc[-1:])  # last SARIMAX value
    np.save('ctx_last.npy', ctx_scaled[-1:])             # last context row

    # 5. Save the feature column names (so the app knows the exact order)
    joblib.dump(list(df_lstm_sub.columns), 'feature_columns.pkl')
    joblib.dump(sarimax_exog_cols,          'sarimax_exog_cols.pkl')

    # 6. Save the raw df_model so the app can display history charts
    df_model.to_csv('df_model.csv')

    print("✅ All model files saved. Download them and place in the same folder as app.py")
    # In Colab: files.download('hybrid_model.keras') etc.
─────────────────────────────────────────────────────────────────────────────────

DEPLOYMENT CHECKLIST (place all these in the same folder as app.py):
    hybrid_model.keras
    scaler_mv.pkl
    scaler_tv.pkl
    scaler_ctx.pkl
    feature_columns.pkl
    sarimax_exog_cols.pkl
    last_sequence.npy
    sarimax_signal_last.npy
    ctx_last.npy
    df_model.csv
    requirements.txt  (see bottom of this file)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import math
import numpy as np
import pandas as pd
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Volume Predictor — Hybrid SARIMA-LSTM",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .main-title { font-size: 2.1rem; font-weight: 700; color: #0D1B2A; }
    .sub-title   { color: #64748B; font-size: 0.95rem; margin-bottom: 0.5rem; }
    .model-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; font-weight: 600; letter-spacing: .05em;
        background: #0E8A7A; color: white; margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.6rem 2rem; border-radius: 14px;
        margin: 1rem 0; text-align: center;
    }
    .result-heavy    { background:#FAECE7; border-left: 7px solid #D85A30; }
    .result-moderate { background:#FAEEDA; border-left: 7px solid #EF9F27; }
    .result-light    { background:#EAF3DE; border-left: 7px solid #6DB33F; }
    .result-clear    { background:#E1F5EE; border-left: 7px solid #1D9E75; }
    .result-label { font-size:.78rem; font-weight:700; letter-spacing:.1em;
                    text-transform:uppercase; margin-bottom:4px; color:#475569; }
    .result-value { font-size:2.6rem; font-weight:800; margin-bottom:4px; }
    .result-sub   { font-size:1rem; color:#475569; }
    .stButton>button {
        width: 100%; padding: .8rem; font-size: 1.05rem; font-weight: 700;
        border-radius: 10px; background: #0D1B2A; color: white; border: none;
    }
    .stButton>button:hover { background: #0E8A7A; }
    .factor-up   { color: #D85A30; font-weight: 600; }
    .factor-down { color: #1D9E75; font-weight: 600; }
    .info-card {
        background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .warn-card {
        background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
MODEL_FILES = [
    'hybrid_model.keras', 'scaler_mv.pkl', 'scaler_tv.pkl',
    'scaler_ctx.pkl', 'feature_columns.pkl', 'sarimax_exog_cols.pkl',
    'last_sequence.npy', 'sarimax_signal_last.npy', 'ctx_last.npy'
]

@st.cache_resource(show_spinner="Loading Hybrid SARIMA-LSTM model…")
def load_model_artifacts():
    """Load the saved Keras model and all scalers/metadata."""
    import joblib
    import tensorflow as tf

    missing = [f for f in MODEL_FILES if not os.path.exists(f)]
    if missing:
        return None, None, None, None, None, None, None, None, None, missing

    model        = tf.keras.models.load_model('hybrid_model.keras')
    scaler_mv    = joblib.load('scaler_mv.pkl')
    scaler_tv    = joblib.load('scaler_tv.pkl')
    scaler_ctx   = joblib.load('scaler_ctx.pkl')
    feat_cols    = joblib.load('feature_columns.pkl')
    ctx_cols     = joblib.load('sarimax_exog_cols.pkl')
    last_seq     = np.load('last_sequence.npy')          # (24, n_features)
    sar_last     = np.load('sarimax_signal_last.npy')    # (1,)
    ctx_last     = np.load('ctx_last.npy')               # (1, 11)
    return model, scaler_mv, scaler_tv, scaler_ctx, feat_cols, ctx_cols, last_seq, sar_last, ctx_last, []


@st.cache_data(show_spinner="Loading historical data…")
def load_history():
    if os.path.exists('df_model.csv'):
        df = pd.read_csv('df_model.csv', index_col=0, parse_dates=True)
        return df
    return None


# ─── Feature Engineering (mirrors notebook exactly) ────────────────────────────
WEATHER_CATS = ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze', 'Mist',
                'Rain', 'Smoke', 'Snow', 'Squall', 'Thunderstorm']

SARIMAX_EXOG_COLS = [
    'temp_c', 'rain_log', 'snow_log', 'clouds_norm',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_holiday', 'is_rush_hour'
]


def build_feature_row(hour, dow, month, temp_c, clouds_pct,
                      rain_mm, snow_mm, weather_type,
                      is_holiday, is_weekend, feat_cols):
    """
    Build one feature row matching the exact column order used during training.
    This mirrors Cells 3-8 of the notebook (feature engineering pipeline).
    """
    row = {}

    # ── Target placeholder (will be scaled but not used for prediction)
    row['traffic_volume'] = 0.0

    # ── Weather transforms (Cell 4 logic)
    row['temp_c']      = float(temp_c)
    row['temp_sq']     = float(temp_c) ** 2
    row['rain_log']    = math.log1p(float(rain_mm))
    row['snow_log']    = math.log1p(float(snow_mm))
    row['clouds_norm'] = float(clouds_pct) / 100.0

    # ── Cyclic temporal encodings (Cell 5 logic)
    row['hour_sin']   = math.sin(2 * math.pi * hour / 24)
    row['hour_cos']   = math.cos(2 * math.pi * hour / 24)
    row['dow_sin']    = math.sin(2 * math.pi * dow  / 7)
    row['dow_cos']    = math.cos(2 * math.pi * dow  / 7)
    row['month_sin']  = math.sin(2 * math.pi * month / 12)
    row['month_cos']  = math.cos(2 * math.pi * month / 12)

    # ── Calendar flags (Cell 6 logic)
    row['is_weekend']   = int(is_weekend)
    row['is_holiday']   = int(is_holiday)
    row['is_rush_hour'] = int((7 <= hour <= 9) or (16 <= hour <= 18))

    # ── Weather one-hot dummies (Cell 7 logic — pd.get_dummies with prefix='weather_')
    for cat in WEATHER_CATS:
        row[f'weather_{cat}'] = 1 if weather_type == cat else 0

    # ── Build array in exact feature column order from training
    arr = np.array([row.get(c, 0.0) for c in feat_cols], dtype=np.float32)
    return arr, row


def build_context_row(row, ctx_cols):
    """Build the 11-feature context vector (Branch C) in the correct column order."""
    return np.array([row.get(c, 0.0) for c in ctx_cols], dtype=np.float32)


def make_prediction(model, scaler_mv, scaler_tv, scaler_ctx,
                    feat_cols, ctx_cols, last_seq,
                    sar_last, ctx_last,
                    hour, dow, month, temp_c, clouds_pct,
                    rain_mm, snow_mm, weather_type, is_holiday, is_weekend):
    """
    Full inference pipeline — mirrors notebook Cells 30-31:
      Branch A: slide last_seq window by 1, replace last row with new feature row
      Branch B: use the last SARIMAX in-sample scaled value (best proxy at inference)
      Branch C: scale the current-step context vector
    Returns predicted traffic volume in vehicles/hr.
    """
    # 1. Build raw feature row for the new timestep
    feat_row, row_dict = build_feature_row(
        hour, dow, month, temp_c, clouds_pct,
        rain_mm, snow_mm, weather_type, is_holiday, is_weekend, feat_cols
    )

    # 2. Scale the new row using the all-feature scaler
    feat_scaled = scaler_mv.transform(feat_row.reshape(1, -1))[0]  # (n_features,)

    # 3. Build Branch A input: drop oldest timestep, append new scaled row
    seq = np.vstack([last_seq[1:], feat_scaled])        # (24, n_features)
    X_seq = seq.reshape(1, 24, -1)                      # (1, 24, n_features)

    # 4. Branch B: SARIMAX signal — use last known value (best proxy for a new point)
    X_sar = sar_last.reshape(1, 1)                      # (1, 1)

    # 5. Branch C: scale the 11-feature context vector
    ctx_row = build_context_row(row_dict, ctx_cols)
    X_ctx = scaler_ctx.transform(ctx_row.reshape(1, -1))  # (1, 11)

    # 6. Model inference
    pred_scaled = model.predict([X_seq, X_sar, X_ctx], verbose=0).flatten()[0]

    # 7. Inverse-transform to vehicles/hr
    pred_vol = scaler_tv.inverse_transform([[pred_scaled]])[0][0]
    pred_vol = max(0, min(7280, int(round(pred_vol))))

    return pred_vol


# ─── Classification & prescriptive logic ──────────────────────────────────────
THRESHOLDS = {'p75': 4940, 'p90': 5800}   # approximate 75th/90th percentiles from EDA

def classify(vol):
    if vol >= THRESHOLDS['p90']:
        return "heavy",    "🔴 Heavy — Congestion Likely",   "#D85A30", "result-heavy"
    elif vol >= THRESHOLDS['p75']:
        return "moderate", "🟡 Elevated — Monitor",           "#BA7517", "result-moderate"
    elif vol >= 1200:
        return "light",    "🟢 Normal Flow",                  "#6DB33F", "result-light"
    else:
        return "clear",    "🟢 Very Low Traffic",             "#1D9E75", "result-clear"


def prescribe(tier, is_rush, is_holiday, weather_type, vol):
    """Threshold-based operational recommendation — prescriptive layer."""
    if tier == "heavy" and is_rush and not is_holiday:
        action = ("🚦 **Recommend Mitigation:** Extend green-signal phase on westbound approach. "
                  "Activate variable message signs. Consider deploying traffic management personnel.")
    elif tier == "heavy" and is_holiday:
        action = "⚠️ Unusually high volume for a holiday period. Investigate for incidents or special events."
    elif tier == "moderate" and is_rush:
        action = ("📡 **Monitor Closely:** Volume approaching congestion threshold. "
                  "Prepare signal retiming. Issue advisory if conditions worsen.")
    elif tier == "moderate" and weather_type in ("Snow", "Rain", "Thunderstorm"):
        action = ("🌧️ Elevated volume combined with adverse weather. "
                  "Recommend reduced speed advisories and increased following-distance alerts.")
    elif tier in ("light", "clear") and is_holiday:
        action = "✅ Holiday traffic pattern recognised. Standard operations — no action required."
    elif tier in ("light", "clear"):
        action = "✅ Normal flow. Standard signal timing appropriate — no action required."
    else:
        action = "ℹ️ Monitor using standard protocols."
    return action


def contributing_factors(hour, dow, temp_c, rain_mm, snow_mm,
                          weather_type, is_holiday, is_weekend):
    is_rush  = (7 <= hour <= 9) or (16 <= hour <= 18)
    is_night = (hour >= 23) or (hour < 5)
    ups, downs = [], []

    if is_rush and not is_weekend:      ups.append("Rush hour (weekday)")
    if not is_weekend and not is_holiday: ups.append("Working weekday")
    if weather_type == "Clear":         ups.append("Clear weather")
    if 10 <= temp_c <= 28:              ups.append("Mild temperature")

    if is_night:                                            downs.append("Nighttime hours")
    if is_weekend:                                          downs.append("Weekend")
    if is_holiday:                                          downs.append("Public holiday")
    if weather_type in ("Rain","Snow","Thunderstorm","Drizzle","Squall"):
        downs.append(f"Adverse weather ({weather_type})")
    if rain_mm > 0:                                         downs.append(f"Rain ({rain_mm:.1f} mm)")
    if snow_mm > 0:                                         downs.append(f"Snow ({snow_mm:.1f} mm)")
    if temp_c < -5:                                         downs.append("Very cold temperature")
    if temp_c > 35:                                         downs.append("Very high temperature")

    return ups, downs


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚦 Traffic Predictor")
    st.markdown(
        "<span class='model-badge'>Hybrid SARIMA-LSTM</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("""
**Model Architecture**
- Branch A: BiLSTM(128→64), 24-hr window
- Branch B: SARIMAX seasonal signal
- Branch C: 11-feature context vector

**Dataset:** Metro Interstate I-94  
**CV Results:** RMSE = 402.4 veh/hr  
**Training:** 5-Fold TimeSeriesSplit
""")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Predict", "📊 Historical Patterns", "🏗️ Model Architecture", "📋 Setup Guide"]
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════
artifacts = load_model_artifacts()
(model, scaler_mv, scaler_tv, scaler_ctx,
 feat_cols, ctx_cols, last_seq, sar_last, ctx_last, missing) = artifacts

model_ready = len(missing) == 0

if missing and page != "📋 Setup Guide":
    st.markdown("""
<div class='warn-card'>
⚠️ <b>Model files not found.</b> The trained Hybrid SARIMA-LSTM files are missing.<br>
Go to the <b>📋 Setup Guide</b> tab to see exactly how to generate and deploy them.<br>
<br>
Missing: <code>""" + "</code>, <code>".join(missing) + """</code>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Predict":
    st.markdown("<div class='main-title'>🚦 Traffic Volume Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>Hybrid SARIMA-LSTM · Metro Interstate I-94 · Real-time inference</div>",
        unsafe_allow_html=True
    )

    left, right = st.columns([1.1, 1], gap="large")

    # ── Inputs ────────────────────────────────────────────────────────────────
    with left:
        st.subheader("⏰ Time & Date")
        c1, c2, c3 = st.columns(3)
        day_names   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        month_names = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]

        with c1:
            dow_name = st.selectbox("Day of week", day_names, index=1)
            dow = day_names.index(dow_name)
        with c2:
            hour = st.number_input("Hour (0–23)", 0, 23, value=8, step=1)
        with c3:
            month_name = st.selectbox("Month", month_names, index=4)
            month = month_names.index(month_name) + 1

        c4, c5 = st.columns(2)
        with c4:
            is_holiday = st.checkbox("🎉 Public holiday")
        with c5:
            is_weekend = st.checkbox("📅 Weekend", value=(dow >= 5))

        st.markdown("---")
        st.subheader("🌤 Weather Conditions")
        w1, w2 = st.columns(2)
        with w1:
            weather_type = st.selectbox(
                "Weather type",
                ["Clear","Clouds","Rain","Drizzle","Snow","Mist","Fog",
                 "Haze","Thunderstorm","Squall","Smoke"],
                index=0
            )
            temp_c = st.number_input("Temperature (°C)", -30, 45, value=18, step=1)
        with w2:
            clouds_pct = st.slider("Cloud cover (%)", 0, 100, 20, step=5)
            rain_mm    = st.number_input("Rain last hour (mm)", 0.0, 200.0, 0.0, step=0.5)
            snow_mm    = st.number_input("Snow last hour (mm)", 0.0, 50.0,  0.0, step=0.5)

        st.markdown("")
        predict_btn = st.button("🔍  Predict Traffic Volume", disabled=not model_ready)

    # ── Results ───────────────────────────────────────────────────────────────
    with right:
        if predict_btn and model_ready:
            with st.spinner("Running Hybrid SARIMA-LSTM inference…"):
                vol = make_prediction(
                    model, scaler_mv, scaler_tv, scaler_ctx,
                    feat_cols, ctx_cols, last_seq, sar_last, ctx_last,
                    hour, dow, month, temp_c, clouds_pct,
                    rain_mm, snow_mm, weather_type, is_holiday, is_weekend
                )

            tier, label, color, css_class = classify(vol)
            pct = round((vol / 7280) * 100)
            is_rush = (7 <= hour <= 9) or (16 <= hour <= 18)

            # Result box
            st.markdown(f"""
            <div class="result-box {css_class}">
                <div class="result-label">{label}</div>
                <div class="result-value" style="color:{color}">{vol:,} veh/hr</div>
                <div class="result-sub">{pct}% of I-94 maximum capacity (7,280 veh/hr)</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Volume",  f"{vol:,} veh/hr")
            m2.metric("Capacity", f"{pct}%")
            m3.metric("Rush hour", "Yes" if is_rush else "No")
            st.progress(pct / 100)

            # Prescriptive recommendation
            rec = prescribe(tier, is_rush, is_holiday, weather_type, vol)
            st.markdown("#### 📋 Operational Recommendation")
            if tier == "heavy":
                st.error(rec)
            elif tier == "moderate":
                st.warning(rec)
            else:
                st.success(rec)

            # Contributing factors
            ups, downs = contributing_factors(
                hour, dow, temp_c, rain_mm, snow_mm,
                weather_type, is_holiday, is_weekend
            )
            st.markdown("#### 🔑 Key Factors")
            f1, f2 = st.columns(2)
            with f1:
                st.markdown("**⬆ Increasing volume**")
                for f in ups:
                    st.markdown(f"<span class='factor-up'>▲ {f}</span>", unsafe_allow_html=True)
                if not ups:
                    st.caption("None detected")
            with f2:
                st.markdown("**⬇ Reducing volume**")
                for f in downs:
                    st.markdown(f"<span class='factor-down'>▼ {f}</span>", unsafe_allow_html=True)
                if not downs:
                    st.caption("None detected")

            # Branch contribution summary
            st.markdown("#### 🏗️ What each model branch contributed")
            st.markdown(f"""
| Branch | Input | Role in this prediction |
|--------|-------|------------------------|
| A — BiLSTM | 24-hr multivariate sequence | Captured temporal dynamics (hour={hour}, recent pattern) |
| B — SARIMAX | Seasonal signal | Injected statistical daily/weekly seasonal baseline |
| C — Context | Weather + calendar flags | Applied {weather_type} weather, {'holiday' if is_holiday else 'rush-hour' if is_rush else 'off-peak'} context |
""")

        elif not model_ready:
            st.markdown("""
<div class='warn-card'>
Model files not loaded. See the <b>📋 Setup Guide</b> tab for instructions on saving and deploying your trained model.
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class='info-card'>
👈 Fill in the time and weather inputs on the left, then click <b>Predict Traffic Volume</b>.
<br><br>
The Hybrid SARIMA-LSTM model will run full inference using all three branches:
<ul>
  <li><b>Branch A</b>: the last 24-hour multivariate sequence</li>
  <li><b>Branch B</b>: the SARIMAX seasonal baseline signal</li>
  <li><b>Branch C</b>: your current weather & calendar context</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORICAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Historical Patterns":
    import matplotlib.pyplot as plt

    st.markdown("<div class='main-title'>📊 Historical Traffic Patterns</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>I-94 Metro Interstate · Oct 2016 – Sep 2018 (modelling window)</div>",
                unsafe_allow_html=True)

    df = load_history()

    if df is None:
        st.warning("Historical data file (df_model.csv) not found. Add it alongside app.py.")
        st.stop()

    st.success(f"✅ {len(df):,} hourly records loaded | {df.index[0].date()} → {df.index[-1].date()}")

    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🕐 Patterns by Hour/Day", "🌡️ Weather Influence"])

    with tab1:
        sample = df['traffic_volume'].resample('6h').mean().dropna()
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(sample.index, sample.values, color='steelblue', linewidth=0.8, alpha=0.9)
        ax.fill_between(sample.index, sample.values, alpha=0.15, color='steelblue')
        ax.set_title('Traffic Volume — 6-Hour Averages (2016–2018)', fontweight='bold')
        ax.set_ylabel('Avg Traffic Volume (veh/hr)')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

    with tab2:
        df_plot = df.copy()
        df_plot['hour']      = df_plot.index.hour
        df_plot['dayofweek'] = df_plot.index.dayofweek

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        hourly = df_plot.groupby('hour')['traffic_volume'].mean()
        axes[0].bar(hourly.index, hourly.values, color='steelblue', edgecolor='white')
        axes[0].set_title('Avg Traffic by Hour of Day', fontweight='bold')
        axes[0].set_xlabel('Hour')
        axes[0].set_ylabel('Avg Volume (veh/hr)')
        axes[0].grid(axis='y', alpha=0.4)

        daily = df_plot.groupby('dayofweek')['traffic_volume'].mean()
        days  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        colors = ['steelblue' if i < 5 else 'coral' for i in range(7)]
        axes[1].bar(days, daily.values, color=colors, edgecolor='white')
        axes[1].set_title('Avg Traffic by Day of Week', fontweight='bold')
        axes[1].set_xlabel('Day')
        axes[1].grid(axis='y', alpha=0.4)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab3:
        if 'temp_c' in df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            sample_s = df[['temp_c','traffic_volume']].dropna().sample(3000, random_state=42)
            axes[0].scatter(sample_s['temp_c'], sample_s['traffic_volume'],
                            alpha=0.2, s=6, color='mediumseagreen')
            axes[0].set_title('Temperature vs Traffic Volume', fontweight='bold')
            axes[0].set_xlabel('Temperature (°C)')
            axes[0].set_ylabel('Traffic Volume (veh/hr)')
            axes[0].grid(True, alpha=0.3)

            if 'is_holiday' in df.columns:
                hol = df.groupby('is_holiday')['traffic_volume'].mean()
                axes[1].bar(['Non-Holiday','Holiday'],
                            hol.reindex([0,1]).values,
                            color=['steelblue','tomato'], edgecolor='white')
                axes[1].set_title('Avg Traffic: Holiday vs Non-Holiday', fontweight='bold')
                axes[1].set_ylabel('Avg Volume (veh/hr)')
                axes[1].grid(axis='y', alpha=0.4)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Weather feature columns not found in df_model.csv.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏗️ Model Architecture":
    st.markdown("<div class='main-title'>🏗️ Hybrid SARIMA-LSTM Architecture</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Tri-branch neural architecture — trained with 5-fold TimeSeriesSplit CV</div>",
                unsafe_allow_html=True)

    st.markdown("""
### How the three branches work together

| Branch | Input Shape | Architecture | Role |
|--------|-------------|--------------|------|
| **A — Sequence** | `(24, 26)` | BiLSTM(128) → LayerNorm → Dropout(0.25) → BiLSTM(64) → Dropout(0.2) → Dense(128) | Captures non-linear temporal dynamics over the past 24 hours |
| **B — SARIMAX Signal** | `(1,)` | Dense(64) → Dense(32) | Injects the statistical seasonal baseline from SARIMAX(1,1,1)×(1,1,1,12) |
| **C — Context** | `(11,)` | Dense(64) → Dense(32) | Applies current weather + calendar context |
| **Merge Head** | Concatenated | Dense(256) → BatchNorm → Dropout(0.2) → Dense(128) → Dense(64) → Dense(32) → Dense(1) | Learns the optimal fusion of all three signals |

### Why a hybrid?

- **ARIMA** is interpretable but can't model non-linearity or exogenous inputs.
- **Pure BiLSTM** learns complex patterns but discards statistical seasonal structure.
- **The Hybrid** gets the best of both: Branch B injects the SARIMAX seasonal signal as a learned feature, so the network never has to re-discover the well-known daily/weekly cycles from scratch.

### Cross-validation results

| Model | MAE | RMSE ⭐ | MAPE |
|-------|-----|---------|------|
| ARIMA(2,1,2) | 1708.3 | 2007.8 | 182.7% |
| SARIMAX(1,1,1)×(1,1,1,8) | 619.8 | 869.7 | 34.0% |
| Multivariate BiLSTM | 311.6 | 422.9 | 14.0% |
| **Hybrid SARIMA-LSTM** | **298.1** | **402.4** | **14.6%** |

RMSE is the primary metric — lower is better. The Hybrid achieves an **80% reduction in RMSE vs ARIMA** and a **4.8% further reduction vs standalone BiLSTM**.

### Thresholds used by the prescriptive layer

| Band | Predicted Volume | Category | Default Action |
|------|-----------------|----------|---------------|
| Normal Flow | < {p75} veh/hr | 🟢 | No action required |
| Elevated — Monitor | {p75} – {p90} veh/hr | 🟡 | Monitor; prepare mitigation |
| Congestion Likely | > {p90} veh/hr (rush, non-holiday) | 🔴 | Extend green phase; activate advisory signs |
""".format(p75=THRESHOLDS['p75'], p90=THRESHOLDS['p90']))

    if model_ready:
        st.success(f"✅ Model loaded | Input shapes: sequence (24, {len(feat_cols)}), "
                   f"SARIMAX signal (1,), context ({len(ctx_cols)},)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Setup Guide":
    st.markdown("<div class='main-title'>📋 Setup Guide</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-title'>How to connect this app to your trained Hybrid SARIMA-LSTM model</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
## The Problem

Your model was trained inside the Colab notebook but **never saved to disk**.
Each CV fold trained a new model in memory and discarded it.
This app needs the trained weights + scalers saved as files.

---

## Step 1 — Add this cell to your notebook (run it after Cell 33)

Copy and paste this into a **new code cell** in your Colab notebook, then run it:

```python
# ════════════════════════════════════════════════════════
# SAVE ALL MODEL ARTIFACTS FOR STREAMLIT DEPLOYMENT
# Run this AFTER the hybrid model CV loop (Cell 33)
# ════════════════════════════════════════════════════════
import joblib

# Re-train on the FULL modelling dataset (not just one fold)
print("Re-training Hybrid model on full dataset for deployment...")
model_final = build_hybrid_model(LOOKBACK, n_features, n_ctx)
es_final  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rlr_final = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

model_final.fit(
    [X_seq_h, X_sar_h, X_ctx_h], y_h,
    epochs=70, batch_size=64,
    validation_split=0.1,
    callbacks=[es_final, rlr_final],
    verbose=1
)

# Save the Keras model
model_final.save('hybrid_model.keras')
print("✅ hybrid_model.keras saved")

# Save all three scalers
joblib.dump(scaler_mv,  'scaler_mv.pkl')
joblib.dump(scaler_tv,  'scaler_tv.pkl')
joblib.dump(scaler_ctx, 'scaler_ctx.pkl')
print("✅ scaler_mv.pkl, scaler_tv.pkl, scaler_ctx.pkl saved")

# Save feature column names (exact training order)
joblib.dump(list(df_lstm_sub.columns), 'feature_columns.pkl')
joblib.dump(sarimax_exog_cols,          'sarimax_exog_cols.pkl')
print("✅ feature_columns.pkl, sarimax_exog_cols.pkl saved")

# Save the last 24 rows as the sequence seed for inference
np.save('last_sequence.npy',      mv_scaled[-24:])           # shape (24, n_features)
np.save('sarimax_signal_last.npy', sarimax_signal_sc[-1:])   # shape (1,)
np.save('ctx_last.npy',           ctx_scaled[-1:])           # shape (1, n_ctx)
print("✅ last_sequence.npy, sarimax_signal_last.npy, ctx_last.npy saved")

# Save the modelling dataset for the historical patterns page
df_model.to_csv('df_model.csv')
print("✅ df_model.csv saved")

# Download everything
from google.colab import files
for fname in ['hybrid_model.keras', 'scaler_mv.pkl', 'scaler_tv.pkl',
              'scaler_ctx.pkl', 'feature_columns.pkl', 'sarimax_exog_cols.pkl',
              'last_sequence.npy', 'sarimax_signal_last.npy', 'ctx_last.npy',
              'df_model.csv']:
    files.download(fname)
    print(f"📥 Downloading {fname}...")

print("\\n✅ All files downloaded. Place them in the same folder as app.py")
```

---

## Step 2 — Place all files in the same folder as app.py

```
traffic-prediction-system/
├── app.py                    ← this file
├── hybrid_model.keras        ← trained Keras model weights
├── scaler_mv.pkl             ← MinMaxScaler for all 26 features
├── scaler_tv.pkl             ← MinMaxScaler for traffic_volume only
├── scaler_ctx.pkl            ← MinMaxScaler for the 11 context features
├── feature_columns.pkl       ← exact column order used during training
├── sarimax_exog_cols.pkl       ← the 11 SARIMAX exog column names
├── last_sequence.npy         ← last 24-hr scaled sequence (inference seed)
├── sarimax_signal_last.npy   ← last SARIMAX signal value (inference seed)
├── ctx_last.npy              ← last context vector (inference seed)
├── df_model.csv              ← historical data for the charts page
└── requirements.txt
```

---

## Step 3 — requirements.txt

```
streamlit
numpy
pandas
matplotlib
scikit-learn
tensorflow
joblib
statsmodels
```

---

## Step 4 — Deploy on Streamlit Cloud

1. Push all the above files to your GitHub repo: `olive-mwende271/traffic-prediction-system`
2. Go to **share.streamlit.io** → New app → select your repo → `app.py`
3. Done — Streamlit Cloud will install requirements and load the model automatically.

> ⚠️ **Note on model file size:** `hybrid_model.keras` may be ~20–80 MB.
> GitHub has a 100 MB file size limit. If the file is too large, use
> [Git LFS](https://git-lfs.github.com/) or host the `.keras` file on
> Google Drive and load it with `gdown` at startup.
""")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "**Model:** Hybrid SARIMA-LSTM  ·  "
    "**Dataset:** Metro Interstate Traffic Volume (UCI ML Repository, I-94 westbound)  ·  "
    "**CV Performance:** MAE=298.1, RMSE=402.4, MAPE=14.6%  ·  "
    "**Author:** Olive Mwende · BSc. Data Science · JKUAT · 2026"
)
