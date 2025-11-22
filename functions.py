import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, mean_squared_error, \
    mean_absolute_error, mean_absolute_percentage_error, r2_score
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import xgboost as xgb
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

IMAGES_DIR = "images/"
DATASET_DIR = "data/"
TABLES_DIR = "tables/"
TRAIN_DATASET = "train_242325.csv"
FORECAST_DATASET = "forecast.csv"

# -------- PLOTTING STYLES
PLOT_STYLE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="CMU Serif, 'Times New Roman', serif", size=14, color="#222"),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#d0d0d0", borderwidth=1),
    margin=dict(t=60, r=30, b=50, l=70),
)

MATPLOTLIB_STYLE = dict(
    figure_facecolor="#ffffff",
    axes_facecolor="#ffffff",
    savefig_facecolor="#ffffff",
    grid_color="#D5D9DF",
    grid_linestyle="-",
    grid_linewidth=0.6,
    font_family="serif",
    font_serif=["Times New Roman", "CMU Serif", "DejaVu Serif"],
    font_size=12,
    axes_titlesize=13,
    axes_labelsize=12,
    legend_fontsize=11,
    xtick_labelsize=11,
    ytick_labelsize=11,
)

ENERGY_COLORS = {
    "grid": "#4472C4",
    "grid2": "darkgoldenrod",
    "solar": "#ED7D31",
    "wind": "#70AD47",
    "price": "#A64D79",
}

FORECAST_FEATURES = [
    "hour_sin", "hour_cos", "is_weekend", "cooling_degree", "heating_degree",
    "temperature", "pressure (hPa)", "cloud_cover (%)", "wind_speed_10m (km/h)",
    "shortwave_radiation (W/m²)", "direct_radiation (W/m²)",
    "diffuse_radiation (W/m²)", "direct_normal_irradiance (W/m²)", "price"
]

def academic_style():
    """Apply consistent academic-style settings to matplotlib."""
    mpl.rcParams.update({
        "figure.figsize": (10, 4),
        "axes.facecolor": MATPLOTLIB_STYLE["axes_facecolor"],
        "savefig.facecolor": MATPLOTLIB_STYLE["savefig_facecolor"],
        "axes.grid": True,
        "grid.color": MATPLOTLIB_STYLE["grid_color"],
        "grid.linestyle": MATPLOTLIB_STYLE["grid_linestyle"],
        "grid.linewidth": MATPLOTLIB_STYLE["grid_linewidth"],
        "font.family": MATPLOTLIB_STYLE["font_family"],
        "font.serif": MATPLOTLIB_STYLE["font_serif"],
        "font.size": MATPLOTLIB_STYLE["font_size"],
        "axes.titlesize": MATPLOTLIB_STYLE["axes_titlesize"],
        "axes.labelsize": MATPLOTLIB_STYLE["axes_labelsize"],
        "legend.fontsize": MATPLOTLIB_STYLE["legend_fontsize"],
        "xtick.labelsize": MATPLOTLIB_STYLE["xtick_labelsize"],
        "ytick.labelsize": MATPLOTLIB_STYLE["ytick_labelsize"],
    })

academic_style()

def load_data():
    df = pd.read_csv(DATASET_DIR + TRAIN_DATASET, parse_dates=['timestamp'], index_col='timestamp')
    df = df.rename(columns={'Demand': 'demand',
                            'Price': 'price',
                            'Temperature': 'temperature',
                            'Pressure (hPa)': 'pressure (hPa)',
                            'Cloud_cover (%)': 'cloud_cover (%)',
                            'Cloud_cover_low (%)': 'cloud_cover_low (%)',
                            'Cloud_cover_mid (%)': 'cloud_cover_mid (%)',
                            'Cloud_cover_high (%)': 'cloud_cover_high (%)',
                            'Wind_speed_10m (km/h)': 'wind_speed_10m (km/h)',
                            'Shortwave_radiation (W/m²)': 'shortwave_radiation (W/m²)'
                            })
    return df

def load_forecast_data():
    df = pd.read_csv(DATASET_DIR + FORECAST_DATASET, parse_dates=['timestamp'], index_col='timestamp')
    df = df.rename(columns={'Demand': 'demand',
                            'Price': 'price',
                            'Temperature': 'temperature',
                            'Pressure (hPa)': 'pressure (hPa)',
                            'Cloud_cover (%)': 'cloud_cover (%)',
                            'Cloud_cover_low (%)': 'cloud_cover_low (%)',
                            'Cloud_cover_mid (%)': 'cloud_cover_mid (%)',
                            'Cloud_cover_high (%)': 'cloud_cover_high (%)',
                            'Wind_speed_10m (km/h)': 'wind_speed_10m (km/h)',
                            'Shortwave_radiation (W/m²)': 'shortwave_radiation (W/m²)'
                            })
    return df

def save_fig(fig, filename: str):
    img_path = IMAGES_DIR + filename
    fig.savefig(img_path, bbox_inches='tight', facecolor='white', format='svg')
    return img_path

def save_fig_plotly(fig: go.Figure, filename: str, width: int = 1000, height: int = 600):
    """
    Save a Plotly figure to the images directory.
    """
    img_path = IMAGES_DIR + filename
    fig.write_image(img_path, format='svg', width=width, height=height)
    return img_path

def plot_and_save(series, filename, color, title, xlabel, ylabel):
    plt.figure(figsize=(10, 4))
    plt.plot(series.index, series.values, color=color, linewidth=1.5)
    plt.title(title, pad=10)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle=MATPLOTLIB_STYLE["grid_linestyle"], color=MATPLOTLIB_STYLE["grid_color"], linewidth=MATPLOTLIB_STYLE["grid_linewidth"])
    plt.tight_layout()
    img_path = IMAGES_DIR + filename
    plt.savefig(img_path, bbox_inches='tight', facecolor=MATPLOTLIB_STYLE["savefig_facecolor"], format='svg')
    plt.show()

def save_table(table, filename):
    table_path = TABLES_DIR + filename
    table.to_csv(table_path)
    return table

def evaluate(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred)
    precision = precision_score(y, y_pred)
    c_matrix = confusion_matrix(y, y_pred)

    return accuracy, recall, precision, c_matrix

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

# == TIME SERIES DIAGNOSTICS ==

def plot_acf_pacf(acf_df: pd.DataFrame, pacf_df: pd.DataFrame, title: str) -> go.Figure:
    """Plot ACF and PACF side by side."""
    if acf_df.empty or pacf_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="ACF/PACF unavailable",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#999"),
        )
        fig.update_layout(**PLOT_STYLE)
        return fig

    fig = make_subplots(rows=1, cols=2, subplot_titles=("ACF", "PACF"))

    # ACF
    fig.add_trace(
        go.Bar(
            x=acf_df["lag"],
            y=acf_df["value"],
            marker_color=ENERGY_COLORS["grid"],
            name="ACF",
        ),
        row=1, col=1,
    )

    # PACF
    fig.add_trace(
        go.Bar(
            x=pacf_df["lag"],
            y=pacf_df["value"],
            marker_color=ENERGY_COLORS["grid2"],
            name="PACF",
        ),
        row=1, col=2,
    )

    # Rakendame meie stiilid
    fig.update_layout(
        title=title,
        showlegend=False,
        **PLOT_STYLE,
        height=400,
        width=900,
    )

    # Telgede kujundus
    fig.update_xaxes(title_text="Lag", showline=True, linewidth=1, linecolor="#ccc")
    fig.update_yaxes(title_text="Autocorrelation", showline=True, linewidth=1, linecolor="#ccc")
    fig.update_yaxes(title_text="Partial autocorrelation", row=1, col=2)

    return fig


def stationarity_tests(series, lags=30):
    adf_result = adfuller(series.dropna(), autolag="AIC", maxlag=lags)
    kpss_result = kpss(series.dropna(), nlags=lags, regression="c")

    summary = pd.DataFrame([
        {
            "test": "ADF",
            "statistic": adf_result[0],
            "p_value": adf_result[1],
            "lags": adf_result[2],
            "critical_5%": adf_result[4]["5%"],
            "interpretation": "Stationary" if adf_result[1] < 0.05 else "Non-stationary"
        },
        {
            "test": "KPSS",
            "statistic": kpss_result[0],
            "p_value": kpss_result[1],
            "lags": kpss_result[2],
            "critical_5%": kpss_result[3]["5%"],
            "interpretation": "Non-stationary" if kpss_result[1] < 0.05 else "Stationary"
        },
    ])
    return summary.round(4)

def gridsearch_arima(train, test, p_values, q_values, horizon, filename):
    """
    Perform ARIMA(p,1,q) grid search and plot nRMSE heatmap.
    Returns: grid_df (DataFrame with metrics) and (best_p, best_q)
    """
    results = []

    for p in p_values:
        for q in q_values:
            try:
                fitted = fit_arima(train, order=(p, 1, q), seasonal_order=(0, 0, 0, 0))
                if fitted is None:
                    continue
                fc = forecast_arima(fitted, horizon, test.index)
                m = evaluate_forecast(test.values, fc.values)
                results.append({"p": p, "q": q, **m})
                print(f"ARIMA({p},1,{q})  MAE={m['MAE']:.4f}  nRMSE={m['nRMSE']:.4f}")
            except Exception as e:
                print(f"ARIMA({p},1,{q}) failed: {e}")

    if not results:
        print("No successful ARIMA fits.")
        return pd.DataFrame(), None, None

    grid_df = pd.DataFrame(results).sort_values("nRMSE")
    print("\nTop ARIMA configurations:")
    grid_df.head()

    # heatmap
    try:
        heat = px.imshow(
            grid_df.pivot(index="p", columns="q", values="nRMSE"),
            text_auto=".3f",
            aspect="auto",
            title="ARIMA(p,1,q) grid search – nRMSE"
        )
        heat.update_layout(**PLOT_STYLE)
        fig_path = IMAGES_DIR + filename
        heat.write_image(fig_path, format="svg", width=900, height=600)
        heat.show()
        print(f"Heatmap saved: {fig_path}")
    except Exception as e:
        print(f"Could not plot heatmap: {e}")

    best_p = int(grid_df.iloc[0]["p"])
    best_q = int(grid_df.iloc[0]["q"])
    return grid_df, best_p, best_q

def fit_arima(series, order, seasonal_order):
    """Train ARIMA/SARIMA model."""
    try:
        model = ARIMA(series, order=order, seasonal_order=seasonal_order, freq='h')
        return model.fit()
    except Exception as e:
        print(f"Fit failed for order={order}, seasonal={seasonal_order}: {e}")
        return None

def forecast_arima(model, horizon, index):
    """Forecast given a fitted ARIMA model."""
    fc = model.forecast(steps=horizon)
    return pd.Series(fc.values, index=index, name="forecast")


def evaluate_forecast(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / (y_true.max() - y_true.min() + 1e-12)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "nRMSE": nrmse, "MAPE": mape}

def walk_forward_daily(df, order, seasonal, days=7, horizon=24):
    preds, metrics = [], []

    for i in range(days):
        cutoff = df["timestamp"].max() - pd.Timedelta(days=days - i)
        tr = df.loc[df["timestamp"] < cutoff].set_index("timestamp")["demand"]
        te = df.loc[
            (df["timestamp"] >= cutoff) &
            (df["timestamp"] < cutoff + pd.Timedelta(hours=horizon))
        ].copy()

        if te.empty:
            continue

        fitted = fit_arima(tr, order, seasonal)
        if fitted is None:
            continue

        fc = forecast_arima(fitted, horizon, te["timestamp"])
        m = evaluate_forecast(te["demand"].values, fc.values)
        m["day"] = i + 1
        metrics.append(m)

        preds.append(pd.DataFrame({
            "timestamp": te["timestamp"],
            "y_true": te["demand"].values,
            "y_pred": fc.values,
            "day": i + 1,
        }))

    return (
        pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        pd.DataFrame(metrics) if metrics else pd.DataFrame(),
    )

# == RESIDUAL DIAGNOSTICS
def residual_diagnostics(residuals, model_name="Model"):
    """Return ACF and histogram for residuals."""
    r_acf = acf(residuals.dropna(), nlags=24)
    acf_df = pd.DataFrame({"lag": range(len(r_acf)), "value": r_acf})

    fig_acf = make_subplots(rows=1, cols=1, subplot_titles=("Residuals ACF",))
    fig_acf.add_trace(go.Bar(x=acf_df["lag"], y=acf_df["value"]))
    fig_acf.update_layout(title=f"Residuals ACF – {model_name}", **PLOT_STYLE)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=residuals.values, nbinsx=30, name="Residuals"))
    fig_hist.update_layout(title=f"Residual histogram – {model_name}", **PLOT_STYLE)

    return fig_acf, fig_hist

def train_xgboost(X_train, y_train, X_val=None, y_val=None, params=None, seed=42):
    y_train = np.asarray(y_train, dtype=np.float32).ravel()

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_train, y_train), (X_val, y_val)]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
        **(params or {})
    )

    model.fit(X_train, y_train, **fit_kwargs)
    return model, getattr(model, "evals_result_", {})


def show_table_info(df, name="Dataset"):
    """Compact summary of columns, types, and missing values."""
    print(f"\n{name.upper()} SUMMARY")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Time span: {df['timestamp'].min()} -> {df['timestamp'].max()}\n")

    info = []
    for col in df.columns:
        if col != "timestamp":
            na_pct = df[col].isna().mean() * 100
            info.append((col, str(df[col].dtype), f"{na_pct:5.2f}%"))
    summary = pd.DataFrame(info, columns=["Column", "Type", "NA %"])
    print(summary.to_string(index=False))
    print()
    return summary


def plot_forecast(timestamp, y_true, y_pred, model_name="XGBoost", filename=None):
    """Plot and optionally save forecast results in academic style."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamp, y=y_true,
        mode="lines", name="Actual",
    ))
    fig.add_trace(go.Scatter(
        x=timestamp, y=y_pred,
        mode="lines", name=f"Forecast",
        line=dict(dash="dash")
    ))

    fig.update_layout(
        title=f"Forecast – {model_name}",
        xaxis_title="Hour",
        yaxis_title="Demand",
        **PLOT_STYLE
    )

    if filename:
        save_fig_plotly(fig, filename, width=1000, height=500)
    return fig

def add_time_related_features(df):
    df = df.copy()
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    else:
        ts = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"index": "timestamp"})
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.dayofweek
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["cooling_degree"] = np.clip(df["temperature"] - 18, 0, None)
    df["heating_degree"] = np.clip(18 - df["temperature"], 0, None)
    return df


def split_7_consecutive_days(forecast_df: pd.DataFrame):
    """Tagasta 7 päeva algus ja lõpp. Eeldab 7 järjestikust päeva tunnitarkusega."""
    df = forecast_df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.normalize()
    days = sorted(df["date"].unique().tolist())
    ranges = []
    for d in days:
        start = pd.Timestamp(d)
        end = start + pd.Timedelta(hours=23)
        ranges.append((start, end))
    return ranges


def baseline_naive(last_value: float, horizon_index: pd.DatetimeIndex) -> pd.Series:
    """Naive: prognoos on viimane täheldatud väärtus enne horisonti."""
    return pd.Series(np.repeat(float(last_value), len(horizon_index)), index=horizon_index)


def baseline_seasonal_naive(history: pd.Series, horizon_index: pd.DatetimeIndex) -> pd.Series:
    """Seasonal naive: väärtus t-24 kui olemas, muidu viimane väärtus enne horisonti."""
    res = []
    last_val = float(history.iloc[-1])
    hist = history.copy()
    hist.index = pd.to_datetime(hist.index)
    for ts in horizon_index:
        t_prev = ts - pd.Timedelta(hours=24)
        if t_prev in hist.index:
            res.append(float(hist.loc[t_prev]))
        else:
            res.append(last_val)
    return pd.Series(res, index=horizon_index)


def rolling_forecast_7days(
    train_full_df,
    forecast_df,
    feature_cols,
    target: str = "demand",
    arima_order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 24),
    xgb_params: dict = None,
):
    if feature_cols is None:
        feature_cols = FORECAST_FEATURES

    trn = add_time_related_features(train_full_df.reset_index() if "timestamp" not in train_full_df.columns else train_full_df)
    trn = trn.sort_values("timestamp")
    trn_X_all = trn[feature_cols].astype(float)
    trn_y_all = trn[target].astype(float)

    days = split_7_consecutive_days(forecast_df)
    all_rows = []
    metric_rows = []

    for day_idx, (start, end) in enumerate(days, 1):
        # treening andmed kuni eelmise tunni lõpuni
        cutoff = start - pd.Timedelta(hours=1)
        train_slice = trn[trn["timestamp"] <= cutoff].copy()
        if train_slice.empty:
            continue

        mask = (forecast_df["timestamp"] >= start) & (forecast_df["timestamp"] <= end)
        fc_day = forecast_df.loc[mask].copy()
        horizon_index = pd.to_datetime(fc_day["timestamp"])
        y_true = fc_day[target].values.astype(float)

        # SARIMA
        y_hist = train_slice.set_index("timestamp")[target].astype(float)
        sarima_fit = fit_arima(y_hist, order=arima_order, seasonal_order=seasonal_order)
        if sarima_fit is not None:
            sarima_pred = forecast_arima(sarima_fit, len(horizon_index), horizon_index).values
            m = evaluate_forecast(y_true, sarima_pred)
            metric_rows.append({"day_idx": day_idx, "model_name": "BestStat", **m})
            all_rows.append(pd.DataFrame({
                "timestamp": horizon_index, "model_name": "BestStat",
                "y_true": y_true, "y_pred": sarima_pred, "day_idx": day_idx
            }))

        # XGBoost
        X_train = train_slice[feature_cols].astype(float)
        y_train = train_slice[target].astype(float)
        X_test = fc_day[feature_cols].astype(float)
        xgb_model, _ = train_xgboost(X_train, y_train, params=xgb_params or {})
        xgb_pred = xgb_model.predict(X_test)
        m = evaluate_forecast(y_true, xgb_pred)
        metric_rows.append({"day_idx": day_idx, "model_name": "XGBoost", **m})
        all_rows.append(pd.DataFrame({
            "timestamp": horizon_index, "model_name": "XGBoost",
            "y_true": y_true, "y_pred": xgb_pred, "day_idx": day_idx
        }))

        # Naive
        naive_pred = baseline_naive(y_hist.iloc[-1], horizon_index)
        m = evaluate_forecast(y_true, naive_pred.values)
        metric_rows.append({"day_idx": day_idx, "model_name": "Naive", **m})
        all_rows.append(pd.DataFrame({
            "timestamp": horizon_index, "model_name": "Naive",
            "y_true": y_true, "y_pred": naive_pred.values, "day_idx": day_idx
        }))

        # Seasonal naive
        seas_pred = baseline_seasonal_naive(y_hist, horizon_index)
        m = evaluate_forecast(y_true, seas_pred.values)
        metric_rows.append({"day_idx": day_idx, "model_name": "SeasonalNaive", **m})
        all_rows.append(pd.DataFrame({
            "timestamp": horizon_index, "model_name": "SeasonalNaive",
            "y_true": y_true, "y_pred": seas_pred.values, "day_idx": day_idx
        }))

    predictions_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    metrics_by_day_df = pd.DataFrame(metric_rows) if metric_rows else pd.DataFrame()

    if not metrics_by_day_df.empty:
        metrics_summary_df = (
            metrics_by_day_df
            .groupby("model_name")[["MAE", "RMSE", "nRMSE"]]
            .agg(MAE_mean=("MAE", "mean"),
                 MAE_std=("MAE", "std"),
                 RMSE_mean=("RMSE", "mean"),
                 RMSE_std=("RMSE", "std"),
                 nRMSE_mean=("nRMSE", "mean"),
                 nRMSE_std=("nRMSE", "std"))
            .reset_index()
        )
    else:
        metrics_summary_df = pd.DataFrame()

    return predictions_df, metrics_by_day_df, metrics_summary_df