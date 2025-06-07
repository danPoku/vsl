import streamlit as st
from pipeline import (
    load_data,
    compute_metrics,
    compute_metrics_in_currency,
    train_model,
    segment_clients,
    train_premium_regressor,
    risk_exposure_summary,
    detect_anomalies,
)

# Cache loading of the CSV so repeated runs are faster. The cach key is the
# file path which allows switching between datasets if required
@st.cache_data
def get_data(path: str):
    return load_data(path)

st.title("Insurance Brokerage Analytics Dashboard")

data_path = st.text_input("CSV data file", "visal_data_cleaned.csv")
run = st.button("Run Analysis")

# Load the dataframe when the button is pressed and remember it across reruns
df = None
if run:
    with st.spinner("Loading data..."):
        df = get_data(data_path)
    st.session_state['df'] = df
elif 'df' in st.session_state:
    df = st.session_state['df']

if df is not None:
    currencies = sorted(df["currency"].unique())
    currency_filter = st.selectbox("Filter dataset by currency", ["All"] + currencies)
    view_currency = st.selectbox(
        "Convert metrics to", ["Original", "USD", "GHS"], index=0
    )

    if currency_filter != "All":
        filtered = df[df["currency"] == currency_filter]
    else:
        filtered = df

    st.subheader("Simple Metrics")

    if view_currency == "USD":
        metrics = compute_metrics_in_currency(filtered, "USD")
    elif view_currency == "GHS":
        metrics = compute_metrics_in_currency(filtered, "GHS")
    else:
        metrics = compute_metrics(filtered, None if currency_filter == "All" else currency_filter)
    for key, value in metrics.items():
        st.write(f"**{key.replace('_', ' ').title()}**: {value}")

    st.subheader("Payment Status Model")
    model, acc, report = train_model(df)
    st.write(f"Accuracy: {acc:.2f}")
    st.json(report)

    st.subheader("Client Segmentation")
    clusters, segment_value = segment_clients(df)
    st.write("Segment Lifetime Value:")
    st.json(segment_value)
    st.write("Segment assignments (first 5):")
    st.dataframe(clusters.head())

    st.subheader("Premium Regression")
    _, mse = train_premium_regressor(df)
    st.write(f"Mean Squared Error: {mse:.2f}")

    st.subheader("Risk Exposure Summary")
    exposure = risk_exposure_summary(df)
    st.json(exposure)

    st.subheader("Potential Anomalies")
    st.dataframe(detect_anomalies(df))
else:
    st.info("Click 'Run Analysis' to load the data and see metrics.")
