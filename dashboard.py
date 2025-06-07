import streamlit as st
from pipeline import (
    load_data,
    compute_metrics,
    train_model,
    segment_clients,
    train_premium_regressor,
    risk_exposure_summary,
    detect_anomalies,
)

st.title("Insurance Brokerage Analytics Dashboard")

data_path = st.text_input("CSV data file", "visal_data_cleaned.csv")
run = st.button("Run Analysis")

if run:
    with st.spinner("Loading data..."):
        df = load_data(data_path)

    st.subheader("Simple Metrics")
    metrics = compute_metrics(df)
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
