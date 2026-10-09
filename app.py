from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# SmartGrid Energy Demand Forecasting
# 24-hour and 7-day (168-hour) recursive forecast dashboard
# ---------------------------------------------------------

st.set_page_config(
    page_title="SmartGrid Energy Forecasting",
    page_icon="⚡",
    layout="wide",
)

DATA_CANDIDATES = [
    Path("data/PJME_hourly.csv"),
    Path("PJME_hourly.csv"),
]

FEATURES = ["lag_1", "lag_24", "lag_168", "hour", "dayofweek"]


@st.cache_data
def load_data():
    data_path = next((p for p in DATA_CANDIDATES if p.exists()), None)
    if data_path is None:
        raise FileNotFoundError(
            "PJME_hourly.csv was not found. Put it inside the 'data' folder "
            "or in the same folder as app.py."
        )

    data = pd.read_csv(data_path)
    required = {"Datetime", "PJME_MW"}
    if not required.issubset(data.columns):
        raise ValueError(
            "CSV must contain these columns: Datetime and PJME_MW."
        )

    data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
    data["PJME_MW"] = pd.to_numeric(data["PJME_MW"], errors="coerce")
    data = data.dropna(subset=["Datetime", "PJME_MW"])
    data = data.sort_values("Datetime").drop_duplicates(
        subset=["Datetime"], keep="last"
    )
    data = data.reset_index(drop=True)

    if len(data) < 200:
        raise ValueError("The dataset needs at least 200 valid hourly rows.")

    return data


def make_features(frame):
    """Create row-based lag features and calendar features."""
    out = frame.copy()
    out["lag_1"] = out["PJME_MW"].shift(1)
    out["lag_24"] = out["PJME_MW"].shift(24)
    out["lag_168"] = out["PJME_MW"].shift(168)
    out["hour"] = out["Datetime"].dt.hour
    out["dayofweek"] = out["Datetime"].dt.dayofweek
    return out.dropna(subset=FEATURES + ["PJME_MW"])


@st.cache_resource
def train_model(data_signature, feature_frame):
    # data_signature is passed so Streamlit invalidates cache when data changes.
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(feature_frame[FEATURES], feature_frame["PJME_MW"])
    return model


def build_model(data):
    feature_frame = make_features(data)
    signature = (len(data), str(data["Datetime"].max()), float(data["PJME_MW"].sum()))
    model = train_model(signature, feature_frame)
    return model, feature_frame


def evaluate_holdout(data):
    featured = make_features(data)
    split = int(len(featured) * 0.8)
    train = featured.iloc[:split]
    test = featured.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train["PJME_MW"])
    predictions = model.predict(test[FEATURES])
    mae = mean_absolute_error(test["PJME_MW"], predictions)
    rmse = np.sqrt(mean_squared_error(test["PJME_MW"], predictions))

    result = pd.DataFrame({
        "Datetime": test["Datetime"].to_numpy(),
        "Actual (MW)": test["PJME_MW"].to_numpy(),
        "Predicted (MW)": predictions,
    })
    return mae, rmse, result


def recursive_forecast(model, data, forecast_start, latest_mw, demand_24_mw, demand_168_mw, horizon):
    """
    Recursive row-based forecast. The user-provided demand values seed the
    latest lag values; each new prediction is then reused as forecast history.
    """
    history = data["PJME_MW"].astype(float).tolist()

    # Seed the final history positions with user-provided lag observations.
    history[-1] = float(latest_mw)
    history[-24] = float(demand_24_mw)
    history[-168] = float(demand_168_mw)

    forecasts = []
    timestamps = []

    for step in range(horizon):
        timestamp = pd.Timestamp(forecast_start) + pd.Timedelta(hours=step)
        row = pd.DataFrame([{
            "lag_1": history[-1],
            "lag_24": history[-24],
            "lag_168": history[-168],
            "hour": timestamp.hour,
            "dayofweek": timestamp.dayofweek,
        }])

        predicted_mw = float(model.predict(row[FEATURES])[0])
        predicted_mw = max(0.0, predicted_mw)

        forecasts.append(predicted_mw)
        timestamps.append(timestamp)
        history.append(predicted_mw)

    return pd.DataFrame({
        "Datetime": timestamps,
        "Predicted Demand (MW)": forecasts,
    })


def make_line_chart(data, x, y, title, y_label="Demand (MW)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x],
        y=data[y],
        mode="lines",
        name=y,
        line={"width": 2},
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date and time",
        yaxis_title=y_label,
        hovermode="x unified",
        height=420,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


st.title("⚡ SmartGrid Energy Demand Forecasting")
st.caption(
    "Hourly electricity-demand analysis using Random Forest and time-series lag features."
)

try:
    dataset = load_data()
except Exception as exc:
    st.error(str(exc))
    st.stop()

model, feature_frame = build_model(dataset)

with st.sidebar:
    st.header("Forecast Settings")
    horizon_choice = st.radio(
        "Choose prediction period",
        ["Next 24 Hours", "Next 7 Days (168 Hours)"],
        index=0,
    )
    horizon = 24 if horizon_choice == "Next 24 Hours" else 168

    latest_dataset_time = pd.Timestamp(dataset["Datetime"].max())
    default_start = latest_dataset_time + pd.Timedelta(hours=1)
    start_date = st.date_input("Forecast start date", value=default_start.date())
    start_hour = st.selectbox("Forecast start hour", list(range(24)), index=default_start.hour)
    forecast_start = pd.Timestamp(start_date) + pd.Timedelta(hours=int(start_hour))

    st.info(
        "The sample PJME dataset ends in 2018. Choose the forecast start date "
        "you want to display. This does not make the historical dataset live/current."
    )

tab_overview, tab_forecast, tab_evaluation, tab_peaks = st.tabs([
    "Data Overview",
    "Forecast",
    "Model Evaluation",
    "Peak Load Analysis",
])

with tab_overview:
    st.subheader("Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid records", f"{len(dataset):,}")
    c2.metric("Average demand", f"{dataset['PJME_MW'].mean():,.0f} MW")
    c3.metric("Maximum recorded demand", f"{dataset['PJME_MW'].max():,.0f} MW")

    st.write(
        f"**Data period:** {dataset['Datetime'].min():%Y-%m-%d %H:%M} to "
        f"{dataset['Datetime'].max():%Y-%m-%d %H:%M}"
    )
    st.dataframe(dataset.tail(10), use_container_width=True)

    daily = dataset.set_index("Datetime")["PJME_MW"].resample("D").mean().dropna().reset_index()
    st.plotly_chart(
        make_line_chart(daily, "Datetime", "PJME_MW", "Daily Average Electricity Demand"),
        use_container_width=True,
    )

with tab_forecast:
    st.subheader(f"{horizon_choice} Forecast")
    st.write(
        "Enter known demand observations in MW. Use actual measured values when available. "
        "If you use example values, treat the output as a demonstration only."
    )

    with st.form("forecast_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            latest_mw = st.number_input(
                "Latest Demand (MW)",
                min_value=0.0,
                value=float(dataset["PJME_MW"].iloc[-1]),
                step=100.0,
            )
        with col2:
            demand_24_mw = st.number_input(
                "Demand 24 Observations Ago (MW)",
                min_value=0.0,
                value=float(dataset["PJME_MW"].iloc[-25]),
                step=100.0,
            )
        with col3:
            demand_168_mw = st.number_input(
                "Demand 168 Observations Ago (MW)",
                min_value=0.0,
                value=float(dataset["PJME_MW"].iloc[-169]),
                step=100.0,
            )

        submitted = st.form_submit_button("Forecast Demand", type="primary")

    if submitted:
        forecast = recursive_forecast(
            model=model,
            data=dataset,
            forecast_start=forecast_start,
            latest_mw=latest_mw,
            demand_24_mw=demand_24_mw,
            demand_168_mw=demand_168_mw,
            horizon=horizon,
        )

        next_mw = forecast["Predicted Demand (MW)"].iloc[0]
        max_idx = forecast["Predicted Demand (MW)"].idxmax()
        min_idx = forecast["Predicted Demand (MW)"].idxmin()
        max_mw = forecast.loc[max_idx, "Predicted Demand (MW)"]
        min_mw = forecast.loc[min_idx, "Predicted Demand (MW)"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Next Hour Demand", f"{next_mw:,.2f} MW")
        m2.metric("Maximum Forecast Demand", f"{max_mw:,.2f} MW")
        m3.metric("Minimum Forecast Demand", f"{min_mw:,.2f} MW")

        st.plotly_chart(
            make_line_chart(
                forecast,
                "Datetime",
                "Predicted Demand (MW)",
                f"Predicted Electricity Demand — {horizon} Hours",
            ),
            use_container_width=True,
        )

        st.dataframe(forecast, use_container_width=True, hide_index=True)
        csv_data = forecast.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Forecast CSV",
            data=csv_data,
            file_name=f"smartgrid_forecast_{horizon}_hours.csv",
            mime="text/csv",
        )

        st.caption(
            "Forecast values are model estimates, not guaranteed real-world demand. "
            "Recursive forecasts can accumulate error, especially over 168 hours."
        )
    else:
        st.info("Enter the demand observations and click Forecast Demand to generate predictions.")

with tab_evaluation:
    st.subheader("Model Evaluation on Historical Holdout Data")
    with st.spinner("Evaluating Random Forest on the historical test set..."):
        mae, rmse, evaluation = evaluate_holdout(dataset)

    a, b = st.columns(2)
    a.metric("Holdout MAE", f"{mae:,.2f} MW")
    b.metric("Holdout RMSE", f"{rmse:,.2f} MW")

    # Show a manageable recent slice for a responsive chart.
    chart_data = evaluation.tail(min(1500, len(evaluation)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data["Datetime"], y=chart_data["Actual (MW)"],
        mode="lines", name="Actual demand",
    ))
    fig.add_trace(go.Scatter(
        x=chart_data["Datetime"], y=chart_data["Predicted (MW)"],
        mode="lines", name="Predicted demand",
    ))
    fig.update_layout(
        title="Actual vs Predicted Demand (Recent Holdout Rows)",
        xaxis_title="Date and time",
        yaxis_title="Demand (MW)",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This is a chronological holdout evaluation. Forecast-tab predictions are recursive "
        "and are a different task from this one-step holdout evaluation."
    )

with tab_peaks:
    st.subheader("Historical Peak Load Analysis")
    peak_row = dataset.loc[dataset["PJME_MW"].idxmax()]
    low_row = dataset.loc[dataset["PJME_MW"].idxmin()]

    p1, p2 = st.columns(2)
    p1.metric("Historical Peak Demand", f"{peak_row['PJME_MW']:,.0f} MW")
    p1.write(f"Recorded at: {peak_row['Datetime']}")
    p2.metric("Historical Minimum Demand", f"{low_row['PJME_MW']:,.0f} MW")
    p2.write(f"Recorded at: {low_row['Datetime']}")

    hourly = dataset.assign(Hour=dataset["Datetime"].dt.hour).groupby("Hour")["PJME_MW"].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly["Hour"], y=hourly["PJME_MW"], name="Average demand"))
    fig.update_layout(
        title="Average Demand by Hour of Day",
        xaxis_title="Hour of day (0–23)",
        yaxis_title="Average demand (MW)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Project note: PJME hourly data is historical and ends in 2018. For operational forecasting, "
    "use current validated demand data and retrain/validate the model for the intended forecast horizon."
)
