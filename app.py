from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="SmartGrid Energy Forecasting",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ SmartGrid Energy Forecasting")
st.caption(
    "Electricity demand forecasting using Time Series features "
    "and Random Forest Machine Learning."
)

DATA_CANDIDATES = [
    Path("data/PJME_hourly.csv"),
    Path("PJME_hourly.csv"),
]

FEATURES = ["lag_1", "lag_24", "lag_168", "hour", "dayofweek"]


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    data_path = next(
        (path for path in DATA_CANDIDATES if path.exists()),
        None,
    )

    if data_path is None:
        raise FileNotFoundError(
            "PJME_hourly.csv was not found. "
            "Place it in the repository root or the data folder."
        )

    data = pd.read_csv(data_path)

    required_columns = {"Datetime", "PJME_MW"}
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required CSV columns: {sorted(missing_columns)}"
        )

    data["Datetime"] = pd.to_datetime(
        data["Datetime"], errors="coerce"
    )
    data["PJME_MW"] = pd.to_numeric(
        data["PJME_MW"], errors="coerce"
    )

    data = data.dropna(subset=["Datetime", "PJME_MW"])
    data = data.sort_values("Datetime")
    data = data.drop_duplicates(subset=["Datetime"], keep="last")
    data = data.reset_index(drop=True)

    if len(data) <= 168:
        raise ValueError(
            "The dataset must contain more than 168 valid hourly records."
        )

    return data


try:
    dataset = load_data()
except Exception as error:
    st.error(f"Unable to load the dataset: {error}")
    st.stop()


# --------------------------------------------------
# 3. CREATE TIME SERIES FEATURES
# --------------------------------------------------
def make_features(data):
    featured = data.copy()

    featured["lag_1"] = featured["PJME_MW"].shift(1)
    featured["lag_24"] = featured["PJME_MW"].shift(24)
    featured["lag_168"] = featured["PJME_MW"].shift(168)
    featured["hour"] = featured["Datetime"].dt.hour
    featured["dayofweek"] = featured["Datetime"].dt.dayofweek

    featured = featured.dropna(
        subset=FEATURES + ["PJME_MW"]
    ).reset_index(drop=True)

    return featured


# --------------------------------------------------
# 4. TRAIN RANDOM FOREST MODEL
# --------------------------------------------------
@st.cache_resource
def train_model(data_signature, _feature_frame):
    model = RandomForestRegressor(
        n_estimators=40,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )

    model.fit(
        _feature_frame[FEATURES],
        _feature_frame["PJME_MW"],
    )

    return model


def build_model(data):
    feature_frame = make_features(data)

    signature = (
        len(data),
        str(data["Datetime"].max()),
        float(data["PJME_MW"].sum()),
    )

    # The signature is the cache key.
    # The leading underscore avoids hashing the full feature DataFrame.
    model = train_model(signature, feature_frame)

    return model, feature_frame


# --------------------------------------------------
# 5. RECURSIVE FORECAST
# --------------------------------------------------
def recursive_forecast(data, model, horizon):
    history = data.copy().sort_values("Datetime").reset_index(drop=True)

    values = history["PJME_MW"].astype(float).tolist()
    last_datetime = history["Datetime"].iloc[-1]

    forecast_rows = []

    for step in range(1, horizon + 1):
        next_datetime = last_datetime + pd.Timedelta(hours=step)

        features = pd.DataFrame(
            [
                {
                    "lag_1": values[-1],
                    "lag_24": values[-24],
                    "lag_168": values[-168],
                    "hour": next_datetime.hour,
                    "dayofweek": next_datetime.dayofweek,
                }
            ]
        )

        prediction = float(model.predict(features[FEATURES])[0])

        # Electricity demand cannot be negative.
        prediction = max(0.0, prediction)

        values.append(prediction)

        forecast_rows.append(
            {
                "Datetime": next_datetime,
                "Predicted Demand (MW)": prediction,
            }
        )

    return pd.DataFrame(forecast_rows)


# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------
@st.cache_data
def evaluate_holdout(data):
    featured = make_features(data)

    split = int(len(featured) * 0.8)

    train = featured.iloc[:split]
    test = featured.iloc[split:]

    if train.empty or test.empty:
        raise ValueError(
            "Not enough data available for model evaluation."
        )

    evaluation_model = RandomForestRegressor(
        n_estimators=40,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=1,
    )

    evaluation_model.fit(
        train[FEATURES],
        train["PJME_MW"],
    )

    predictions = evaluation_model.predict(test[FEATURES])

    mae = mean_absolute_error(
        test["PJME_MW"], predictions
    )

    rmse = np.sqrt(
        mean_squared_error(test["PJME_MW"], predictions)
    )

    results = pd.DataFrame(
        {
            "Datetime": test["Datetime"].to_numpy(),
            "Actual (MW)": test["PJME_MW"].to_numpy(),
            "Predicted (MW)": predictions,
        }
    )

    return mae, rmse, results


# --------------------------------------------------
# 7. PLOTLY LINE CHART
# --------------------------------------------------
def make_line_chart(data, x_column, y_column, chart_title):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode="lines",
            name=y_column,
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=chart_title,
        xaxis_title="Date and Time",
        yaxis_title="Electricity Demand (MW)",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )

    return fig


# --------------------------------------------------
# 8. PREPARE MODEL
# --------------------------------------------------
try:
    with st.spinner("Preparing the forecasting model..."):
        model, feature_frame = build_model(dataset)
except Exception as error:
    st.error(f"Unable to prepare the model: {error}")
    st.stop()


# --------------------------------------------------
# 9. SUMMARY METRICS
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", f"{len(dataset):,}")

with col2:
    st.metric(
        "Average Demand",
        f"{dataset['PJME_MW'].mean():,.0f} MW",
    )

with col3:
    st.metric(
        "Latest Actual Demand",
        f"{dataset['PJME_MW'].iloc[-1]:,.0f} MW",
    )

st.divider()


# --------------------------------------------------
# 10. APPLICATION TABS
# --------------------------------------------------
tab_overview, tab_forecast, tab_evaluation, tab_peaks = st.tabs(
    [
        "Data Overview",
        "Forecast",
        "Model Evaluation",
        "Peak Load Analysis",
    ]
)


# --------------------------------------------------
# TAB 1: DATA OVERVIEW
# --------------------------------------------------
with tab_overview:
    st.subheader("Historical Electricity Demand")

    st.write(
        "This dataset contains historical hourly electricity "
        "demand measurements."
    )

    st.dataframe(
        dataset.tail(10),
        width="stretch",
        hide_index=True,
    )

    daily = (
        dataset.set_index("Datetime")["PJME_MW"]
        .resample("D")
        .mean()
        .reset_index()
    )

    st.plotly_chart(
        make_line_chart(
            daily,
            "Datetime",
            "PJME_MW",
            "Daily Average Electricity Demand",
        ),
        width="stretch",
    )


# --------------------------------------------------
# TAB 2: FORECAST
# --------------------------------------------------
with tab_forecast:
    st.subheader("Future Electricity Demand Forecast")

    with st.form("forecast_form"):
        horizon = st.selectbox(
            "Select forecast period",
            options=[24, 168],
            format_func=lambda value: (
                "Next 24 Hours" if value == 24
                else "Next 7 Days (168 Hours)"
            ),
        )

        submit_forecast = st.form_submit_button(
            "Generate Forecast"
        )

    if submit_forecast:
        try:
            with st.spinner("Generating forecast..."):
                forecast = recursive_forecast(
                    dataset, model, horizon
                )

            st.success("Forecast generated successfully.")

            st.metric(
                "Average Predicted Demand",
                f"{forecast['Predicted Demand (MW)'].mean():,.0f} MW",
            )

            st.plotly_chart(
                make_line_chart(
                    forecast,
                    "Datetime",
                    "Predicted Demand (MW)",
                    f"Predicted Electricity Demand — {horizon} Hours",
                ),
                width="stretch",
            )

            st.dataframe(
                forecast,
                width="stretch",
                hide_index=True,
            )

            csv_data = forecast.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name=f"smartgrid_forecast_{horizon}_hours.csv",
                mime="text/csv",
            )

        except Exception as error:
            st.error(f"Unable to generate forecast: {error}")


# --------------------------------------------------
# TAB 3: MODEL EVALUATION
# --------------------------------------------------
with tab_evaluation:
    st.subheader("Model Performance Evaluation")

    st.write(
        "The model is evaluated on the later 20% of the "
        "time-ordered feature dataset."
    )

    if st.button("Evaluate Model"):
        try:
            with st.spinner("Evaluating model..."):
                mae, rmse, evaluation_results = evaluate_holdout(
                    dataset
                )

            metric1, metric2 = st.columns(2)

            with metric1:
                st.metric("MAE", f"{mae:,.2f} MW")

            with metric2:
                st.metric("RMSE", f"{rmse:,.2f} MW")

            st.caption(
                "Lower MAE and RMSE generally indicate smaller prediction errors."
            )

            st.plotly_chart(
                make_line_chart(
                    evaluation_results.tail(500),
                    "Datetime",
                    "Actual (MW)",
                    "Actual Electricity Demand — Last 500 Test Records",
                ),
                width="stretch",
            )

            comparison_fig = go.Figure()

            comparison_fig.add_trace(
                go.Scatter(
                    x=evaluation_results["Datetime"].tail(500),
                    y=evaluation_results["Actual (MW)"].tail(500),
                    mode="lines",
                    name="Actual Demand",
                )
            )

            comparison_fig.add_trace(
                go.Scatter(
                    x=evaluation_results["Datetime"].tail(500),
                    y=evaluation_results["Predicted (MW)"].tail(500),
                    mode="lines",
                    name="Predicted Demand",
                )
            )

            comparison_fig.update_layout(
                title="Actual vs Predicted Demand",
                xaxis_title="Date and Time",
                yaxis_title="Electricity Demand (MW)",
                height=420,
                margin=dict(l=20, r=20, t=60, b=20),
                hovermode="x unified",
            )

            st.plotly_chart(
                comparison_fig,
                width="stretch",
            )

            st.dataframe(
                evaluation_results.tail(20),
                width="stretch",
                hide_index=True,
            )

        except Exception as error:
            st.error(f"Unable to evaluate model: {error}")


# --------------------------------------------------
# TAB 4: PEAK LOAD ANALYSIS
# --------------------------------------------------
with tab_peaks:
    st.subheader("Peak Electricity Demand Analysis")

    peak_count = st.selectbox(
        "Number of highest-demand records",
        options=[10, 20, 50],
        index=0,
    )

    peak_data = (
        dataset.nlargest(peak_count, "PJME_MW")
        .sort_values("PJME_MW", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(
        peak_data,
        width="stretch",
        hide_index=True,
    )

    peak_fig = go.Figure()

    peak_fig.add_trace(
        go.Bar(
            x=peak_data["Datetime"].dt.strftime(
                "%Y-%m-%d %H:%M"
            ),
            y=peak_data["PJME_MW"],
            name="Demand",
        )
    )

    peak_fig.update_layout(
        title=f"Top {peak_count} Peak Electricity Demand Records",
        xaxis_title="Date and Time",
        yaxis_title="Electricity Demand (MW)",
        height=450,
        margin=dict(l=20, r=20, t=60, b=80),
        xaxis_tickangle=-45,
    )

    st.plotly_chart(
        peak_fig,
        width="stretch",
    )

    peak_record = dataset.loc[dataset["PJME_MW"].idxmax()]

    st.info(
        f"Highest recorded demand: {peak_record['PJME_MW']:,.0f} MW "
        f"on {peak_record['Datetime']}."
    )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "SmartGrid Energy Forecasting | Python • Pandas • "
    "Scikit-learn • Random Forest • Streamlit • Plotly"
)
