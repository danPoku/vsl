import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    mean_squared_error,
)

# Rough average exchange rates to USD used for conversion. These values are
# placeholders for demonstration. In a real system these would come from a
# reliable historical FX data source.
CONVERSION_TO_USD = {
    "USD": 1.0,
    "GHS": 0.086,  # 1 Ghana cedi ~= 0.086 USD
    "EUR": 1.1,
    "GBP": 1.3,
    "Le": 0.000045,
    "XOF": 0.0016,
    "D": 0.016,
}


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    # parse dates
    for col in ['offer_date', 'start_period', 'end_period']:
        df[col] = pd.to_datetime(df[col])
    # add policy duration in days
    df['policy_duration'] = (df['end_period'] - df['start_period']).dt.days
    return df


def compute_metrics(df: pd.DataFrame, currency: str | None = None) -> dict:
    """Compute simple domain metrics from the dataframe.

    If ``currency`` is provided the dataframe is first filtered to rows
    matching that currency before aggregation.
    """
    if currency is not None:
        df = df[df["currency"] == currency]

    metrics = {}
    metrics["total_policies"] = len(df)
    metrics["total_premium"] = df["fac_premium"].sum()
    metrics["average_premium"] = df["fac_premium"].mean()
    metrics["average_outstanding_balance"] = df["outstanding_balance"].mean()
    metrics["payment_status_counts"] = df["payment_status"].value_counts().to_dict()
    metrics["top_outstanding_by_business"] = (
        df.groupby("business_name")["outstanding_balance"].sum()
        .sort_values(ascending=False)
        .head(5)
        .to_dict()
    )
    return metrics


def compute_metrics_in_currency(df: pd.DataFrame, target: str) -> dict:
    """Convert numeric columns to ``target`` currency then compute metrics."""
    if target not in ("USD", "GHS"):
        raise ValueError("target must be 'USD' or 'GHS'")

    numeric_cols = [
        "fac_sum_insured",
        "fac_premium",
        "brokerage",
        "nic_levy",
        "amount_due",
        "commission",
        "amount_paid",
        "outstanding_balance",
    ]

    df_conv = df.copy()

    rates = df_conv["currency"].map(CONVERSION_TO_USD).fillna(1.0)
    if target == "GHS":
        rates /= CONVERSION_TO_USD["GHS"]
    factors = rates
    for col in numeric_cols:
        df_conv[col] = df_conv[col] * factors

    # After conversion, treat everything as the same currency
    df_conv["currency"] = target
    return compute_metrics(df_conv)


def train_model(df: pd.DataFrame):
    """Train a logistic regression model to predict payment_status."""
    X = df[
        [
            'currency',
            'fac_sum_insured',
            'fac_premium',
            'amount_due',
            'commission',
            'amount_paid',
            'outstanding_balance',
            'policy_duration',
            'reinsured',
        ]
    ]
    y = df['payment_status']

    preprocess = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), ['currency', 'reinsured'])],
        remainder='passthrough',
    )

    model = Pipeline([
        ('preprocess', preprocess),
        ('clf', LogisticRegression(max_iter=1000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return model, accuracy, report


def segment_clients(df: pd.DataFrame, n_clusters: int = 3):
    """Cluster clients to discover segments and lifetime value."""
    features = df[[
        'fac_sum_insured',
        'fac_premium',
        'commission',
        'outstanding_balance',
    ]]
    scaler = StandardScaler()
    km = KMeans(n_clusters=n_clusters, random_state=42)
    pipeline = Pipeline([
        ('scale', scaler),
        ('kmeans', km),
    ])
    clusters = pipeline.fit_predict(features)
    df = df.copy()
    df['segment'] = clusters
    value = df.groupby('segment')['commission'].sum().to_dict()
    return df[['policy_no', 'segment']], value


def train_premium_regressor(df: pd.DataFrame):
    """Fit a regression model suggesting premium levels."""
    X = df[[
        'fac_sum_insured',
        'brokerage',
        'nic_levy',
        'commission',
        'currency',
    ]]
    y = df['fac_premium']

    preprocess = ColumnTransformer(
        [('cat', OneHotEncoder(handle_unknown='ignore'), ['currency'])],
        remainder='passthrough',
    )

    model = Pipeline([
        ('preprocess', preprocess),
        ('reg', LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, mse


def risk_exposure_summary(df: pd.DataFrame):
    """Aggregate exposures by business line and currency."""
    summary = (
        df.groupby(['business_name', 'currency'])['fac_sum_insured']
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    return summary.to_dict()


def detect_anomalies(df: pd.DataFrame, threshold: float = 3.0):
    """Flag outlier policies based on numeric columns."""
    numeric = ['fac_premium', 'brokerage', 'nic_levy', 'commission']
    z = ((df[numeric] - df[numeric].mean()) / df[numeric].std()).abs()
    anomalies = df[(z > threshold).any(axis=1)]
    return anomalies[['policy_no'] + numeric].head(5)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Insurance brokerage analytics")
    parser.add_argument('--data', default='visal_data_cleaned.csv', help='Path to CSV data file')
    args = parser.parse_args()

    df = load_data(args.data)
    metrics = compute_metrics(df)
    print("Simple Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    model, acc, report = train_model(df)
    print(f"\nPayment Status Model Accuracy: {acc:.4f}")
    for label, stats in report.items():
        if isinstance(stats, dict):
            print(f"{label}: precision={stats['precision']:.2f}, recall={stats['recall']:.2f}, f1={stats['f1-score']:.2f}")

    clusters, segment_value = segment_clients(df)
    print("\nClient Segmentation (first 5):")
    print(clusters.head())
    print("Segment Lifetime Value:", segment_value)

    prem_model, mse = train_premium_regressor(df)
    print(f"\nPremium Regression MSE: {mse:.2f}")

    print("\nRisk Exposure Summary:")
    print(risk_exposure_summary(df))

    print("\nPotential Anomalies:")
    print(detect_anomalies(df))


if __name__ == '__main__':
    main()
