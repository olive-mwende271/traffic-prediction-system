import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATIONARITY & DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Stationarity & Decomposition":
    st.title("📈 Stationarity & Decomposition")

    if df_feat is None:
        st.warning("Please upload the dataset in the sidebar first.")
        st.stop()

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    base_features = [
        'traffic_volume', 'temp_c', 'temp_sq', 'rain_log', 'snow_log', 'clouds_norm',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'is_holiday', 'is_rush_hour'
    ]
    weather_cols = [c for c in df_feat.columns if c.startswith('weather_')]
    all_features = base_features + weather_cols
    all_features = [c for c in all_features if c in df_feat.columns]

    df_model = df_feat[all_features].copy()
    df_model = df_model.resample('h').mean()
    df_model = df_model.interpolate(method='time')
    df_model = df_model.last('730D')

    st.info(f"Modelling dataset: **{df_model.shape[0]:,} hourly records** | "
            f"{df_model.index[0].date()} → {df_model.index[-1].date()}")

    # ADF Test
    st.subheader("Augmented Dickey-Fuller (ADF) Stationarity Test")
    result = adfuller(df_model['traffic_volume'].dropna(), autolag='AIC')
    is_stat = result[1] < 0.05
    col1, col2, col3 = st.columns(3)
    col1.metric("ADF Statistic", f"{result[0]:.4f}")
    col2.metric("p-value", f"{result[1]:.6f}")
    col3.metric("Critical Value (5%)", f"{result[4]['5%']:.4f}")
    if is_stat:
        st.success("✅ Series is **stationary** (p < 0.05) — no differencing required.")
    else:
        st.warning("⚠️ Series is **non-stationary** — differencing may be needed.")

    # Seasonal Decomposition
    st.subheader("Seasonal Decomposition (Additive, period=24h)")
    try:
        decomp = seasonal_decompose(df_model['traffic_volume'].dropna(), model='additive', period=24)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        components = [
            ('Observed',       decomp.observed,  'steelblue'),
            ('Trend',          decomp.trend,     'darkorange'),
            ('Seasonal (24h)', decomp.seasonal,  'mediumseagreen'),
            ('Residual',       decomp.resid,     'tomato'),
        ]
        for ax, (title, data, color) in zip(axes, components):
            ax.plot(data.index[:2000], data.values[:2000], color=color, linewidth=0.8)
            ax.set_title(title, fontweight='bold'); ax.grid(True, alpha=0.3)
        plt.suptitle('Seasonal Decomposition (first 2000 hours)', fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Decomposition error: {e}")

    # ACF / PACF
    st.subheader("ACF & PACF")
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(df_model['traffic_volume'].dropna(),  lags=48, ax=axes2[0], title='ACF — Traffic Volume')
    plot_pacf(df_model['traffic_volume'].dropna(), lags=48, ax=axes2[1], title='PACF — Traffic Volume')
    plt.tight_layout()
    st.pyplot(fig2)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training & Evaluation":
    st.title("🤖 Model Training & Evaluation")

    if df_feat is None:
        st.warning("Please upload the dataset in the sidebar first.")
        st.stop()

    st.markdown("""
    > ⚠️ **Note:** Full model training (LSTM & Hybrid SARIMAX-LSTM) is computationally intensive
    > and is best run in Google Colab or a GPU environment. This page runs **ARIMA** and
    > **SARIMAX** directly; for deep-learning models, results from the notebook are displayed
    > as reference benchmarks.
    """)

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.model_selection import TimeSeriesSplit

    base_features = [
        'traffic_volume', 'temp_c', 'temp_sq', 'rain_log', 'snow_log', 'clouds_norm',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'is_weekend', 'is_holiday', 'is_rush_hour'
    ]
    weather_cols = [c for c in df_feat.columns if c.startswith('weather_')]
    all_features = base_features + [c for c in weather_cols if c in df_feat.columns]
    df_model = df_feat[[c for c in all_features if c in df_feat.columns]].copy()
    df_model = df_model.resample('h').mean().interpolate(method='time').last('730D')

    series = df_model['traffic_volume'].copy()
    exog_cols = [c for c in df_model.columns if c != 'traffic_volume']
    exog_df   = df_model[exog_cols].copy()

    N_SPLITS = 3
    tscv     = TimeSeriesSplit(n_splits=N_SPLITS)
    all_results = {}

    model_choice = st.selectbox(
        "Select model to train",
        ["ARIMA (fast, ~30s)", "SARIMAX (moderate, ~2-3 min)"]
    )

    run_btn = st.button("▶ Run Selected Model")

    if run_btn:
        if "ARIMA" in model_choice:
            ARIMA_ORDER  = (2, 1, 2)
            series_arima = series.iloc[::3].reset_index(drop=True)
            fold_metrics, all_act, all_pred = [], [], []

            with st.spinner(f"Running ARIMA{ARIMA_ORDER} — {N_SPLITS}-fold CV…"):
                progress = st.progress(0)
                for i, (tr_idx, te_idx) in enumerate(tscv.split(series_arima)):
                    test = series_arima.iloc[te_idx[:168]]
                    try:
                        fit   = ARIMA(series_arima.iloc[tr_idx], order=ARIMA_ORDER).fit()
                        preds = np.maximum(fit.forecast(steps=len(test)), 0)
                        m     = compute_metrics(test.values, preds)
                        fold_metrics.append(m)
                        all_act.extend(test.values)
                        all_pred.extend(preds)
                    except Exception as e:
                        st.warning(f"Fold {i+1} error: {e}")
                    progress.progress((i+1)/N_SPLITS)

            avg = {k: np.mean([m[k] for m in fold_metrics]) for k in ['MAE','RMSE','MAPE']}
            all_results['ARIMA'] = avg
            st.success(f"✅ ARIMA → MAE={avg['MAE']:.1f}  RMSE={avg['RMSE']:.1f}  MAPE={avg['MAPE']:.2f}%")

            # Plot
            n_plot = min(200, len(all_act))
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(all_act[:n_plot],  color='gray',      linewidth=1.2, label='Actual')
            ax.plot(all_pred[:n_plot], color='steelblue', linewidth=1.5, label='ARIMA Predicted')
            ax.fill_between(range(n_plot), all_act[:n_plot], all_pred[:n_plot], alpha=0.1, color='steelblue')
            ax.set_title('ARIMA — Actual vs Predicted', fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            res_df = pd.DataFrame([avg]).round(2)
            res_df.index = ['ARIMA']
            st.dataframe(res_df)

        elif "SARIMAX" in model_choice:
            series_s = series.iloc[::3].reset_index(drop=True)
            sarimax_exog_cols = [
                'temp_c', 'rain_log', 'snow_log', 'clouds_norm',
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'is_weekend', 'is_holiday', 'is_rush_hour'
            ]
            sarimax_exog_cols = [c for c in sarimax_exog_cols if c in exog_df.columns]
            exog_s = exog_df[sarimax_exog_cols].iloc[::3].reset_index(drop=True)

            fold_metrics, all_act, all_pred = [], [], []
            with st.spinner(f"Running SARIMAX — {N_SPLITS}-fold CV… (this may take a few minutes)"):
                progress = st.progress(0)
                for i, (tr_idx, te_idx) in enumerate(tscv.split(series_s)):
                    te_cut = te_idx[:56]
                    try:
                        fit = SARIMAX(
                            series_s.iloc[tr_idx], exog=exog_s.iloc[tr_idx],
                            order=(1,1,1), seasonal_order=(1,1,1,8),
                            enforce_stationarity=False, enforce_invertibility=False
                        ).fit(disp=False)
                        preds = np.maximum(fit.forecast(steps=len(te_cut), exog=exog_s.iloc[te_cut]), 0)
                        m = compute_metrics(series_s.iloc[te_cut].values, preds)
                        fold_metrics.append(m)
                        all_act.extend(series_s.iloc[te_cut].values)
                        all_pred.extend(preds)
                    except Exception as e:
                        st.warning(f"Fold {i+1} error: {e}")
                    progress.progress((i+1)/N_SPLITS)

            avg = {k: np.mean([m[k] for m in fold_metrics]) for k in ['MAE','RMSE','MAPE']}
            all_results['SARIMAX'] = avg
            st.success(f"✅ SARIMAX → MAE={avg['MAE']:.1f}  RMSE={avg['RMSE']:.1f}  MAPE={avg['MAPE']:.2f}%")

            n_plot = min(200, len(all_act))
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(all_act[:n_plot],  color='gray',         linewidth=1.2, label='Actual')
            ax.plot(all_pred[:n_plot], color='mediumseagreen', linewidth=1.5, label='SARIMAX Predicted')
            ax.fill_between(range(n_plot), all_act[:n_plot], all_pred[:n_plot], alpha=0.1, color='mediumseagreen')
            ax.set_title('SARIMAX — Actual vs Predicted', fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            res_df = pd.DataFrame([avg]).round(2)
            res_df.index = ['SARIMAX']
            st.dataframe(res_df)

    # Reference benchmarks from notebook
    st.markdown("---")
    st.subheader("📋 Reference Benchmarks (from full notebook, 5-fold CV)")
    ref = pd.DataFrame({
        'ARIMA':               {'MAE': 1012.1, 'RMSE': 1347.8, 'MAPE': 35.21},
        'SARIMAX':             {'MAE':  812.3, 'RMSE': 1102.5, 'MAPE': 27.44},
        'LSTM (Multivariate)': {'MAE':  623.7, 'RMSE':  874.2, 'MAPE': 19.83},
        'Hybrid SARIMAX-LSTM': {'MAE':  487.5, 'RMSE':  671.3, 'MAPE': 14.62},
    }).T.round(2)
    st.dataframe(ref.style.highlight_min(axis=0, color='lightgreen'))

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESULTS & DISCUSSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Results & Discussion":
    st.title("📊 Results & Discussion")

    all_results = {
        'ARIMA':               {'MAE': 1012.1, 'RMSE': 1347.8, 'MAPE': 35.21},
        'SARIMAX':             {'MAE':  812.3, 'RMSE': 1102.5, 'MAPE': 27.44},
        'LSTM (Multivariate)': {'MAE':  623.7, 'RMSE':  874.2, 'MAPE': 19.83},
        'Hybrid SARIMAX-LSTM': {'MAE':  487.5, 'RMSE':  671.3, 'MAPE': 14.62},
    }

    results_df = pd.DataFrame(all_results).T.round(2)
    results_df['Rank (RMSE)'] = results_df['RMSE'].rank().astype(int)
    results_df = results_df.sort_values('RMSE')

    st.subheader("Cross-Validated Model Performance (5-Fold CV)")
    st.dataframe(results_df.style.highlight_min(subset=['MAE','RMSE','MAPE'], axis=0, color='lightgreen'))
    st.success("🏆 Best Model: **Hybrid SARIMAX-LSTM** — lowest MAE, RMSE, and MAPE across all folds.")

    # Bar chart
    st.subheader("Metric Comparison — Bar Chart")
    models      = list(all_results.keys())
    bar_colors  = ['#4c72b0','#55a868','#c44e52','gold']
    metric_names = ['MAE','RMSE','MAPE']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, metric_names):
        vals = [all_results[m][metric] for m in models]
        bars = ax.bar(models, vals, color=bar_colors, edgecolor='white', linewidth=1.2, width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        best_idx = int(np.argmin(vals))
        bars[best_idx].set_edgecolor('black'); bars[best_idx].set_linewidth(3)
        ax.set_title(f'{metric} (lower = better)', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xticklabels(models, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.4)
        ax.set_ylim(0, max(vals)*1.15)

    plt.suptitle('Model Comparison — 5-Fold Time-Series CV', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    # Radar chart
    st.subheader("Radar Chart — Relative Performance")
    def norm_scores(res, metrics):
        sc = {}
        for metric in metrics:
            vals  = {m: res[m][metric] for m in res}
            max_v = max(vals.values())
            for model in vals:
                sc.setdefault(model, {})[metric] = 1 - (vals[model] / max_v)
        return sc

    radar_metrics = ['MAE','RMSE','MAPE']
    scores = norm_scores(all_results, radar_metrics)
    N_cat  = len(radar_metrics)
    angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist() + [0]
    r_colors = {'ARIMA':'#4c72b0','SARIMAX':'#55a868',
                'LSTM (Multivariate)':'#c44e52','Hybrid SARIMAX-LSTM':'gold'}

    fig2, ax2 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model, sc in scores.items():
        vals = [sc[c] for c in radar_metrics] + [sc[radar_metrics[0]]]
        lw   = 3 if 'Hybrid' in model else 1.8
        ax2.plot(angles, vals, 'o-', linewidth=lw,
                 label=model, color=r_colors.get(model, 'gray'))
        ax2.fill(angles, vals, alpha=0.07, color=r_colors.get(model, 'gray'))
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([f'{c}\n(↑ better)' for c in radar_metrics], fontsize=11)
    ax2.set_yticklabels([])
    ax2.set_title('Model Performance Radar', fontsize=13, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)

    # Discussion
    st.subheader("📝 Discussion")
    st.markdown("""
    ### Key Findings

    **ARIMA (Univariate Baseline):** Captures linear autoregressive structure but cannot model
    seasonality, non-linearity, or external context (weather, holidays). Produces the highest errors.

    **SARIMAX (Seasonal ARIMA + Exogenous):** Extending ARIMA with weather and calendar features provides
    a measurable gain, especially around holidays and precipitation events. However, it remains a linear model.

    **Multivariate LSTM:** The Bidirectional architecture captures non-linear temporal interactions
    across all features simultaneously. Outperforms both statistical models significantly.

    **Hybrid SARIMAX-LSTM ⭐ Best Model:** Combines three complementary information streams:
    - **Branch A** — BiLSTM over full 24-hour feature history
    - **Branch B** — Explicit SARIMAX forecast signal (structured seasonal knowledge)
    - **Branch C** — Current-timestep exogenous snapshot (weather + calendar)

    The merge layer learns to optimally weight these signals, making the hybrid model
    **strictly more expressive** than any individual component.

    ### Model Summary
    | Model | Key Strength | Key Limitation |
    |-------|-------------|----------------|
    | ARIMA | Interpretable, fast | No seasonality, no exogenous |
    | SARIMAX | Seasonality + exogenous | Linear only |
    | LSTM | Non-linear, multivariate | May underuse seasonal patterns |
    | **Hybrid** | **All combined** | Higher compute cost |
    """)
