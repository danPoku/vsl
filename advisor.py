# fac_check_app.py
# ────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="VisalRE Premium Check", page_icon="📊")

# ── 1. Load pickled model (cached) ──────────────────────────────────────────


@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model(Path(__file__).with_name("visal_re_predictor.pkl"))

# ── 2. Utility: currency formatter ─────────────────────────────────────────


def fmt_currency(val: float, cur: str) -> str:
    symbol = "GHS" if cur == "GHS" else "$"
    return f"{symbol} {val:,.2f}"


# ── 3. Sidebar  –  data entry ──────────────────────────────────────────────
with st.sidebar:
    st.header("Enter policy details")

    business = st.selectbox(
        "Policy Type",
        [
            "CONTRACTOR'S ALL RISKS",
            "Assets All Risks",
            "Commercial Fire",
            "Goods In Transit",
            "Motor Comprehensive",
            "Performance Bond",
            "Erection All Risks",
            "Machinery Breakdown",
            "Money Insurance",
            "Motor Comprehensive Fleet",
            "Public Liability",
            "Group Personal Accident",
            "Directors & Officers Liability",
            "Removal Bond",
            "General Exportation Bond",
            "Industrial Fire",
            "General Premises Bond",
            "Workmen's Compensation",
            "Advance Payment Guarantee",
            "Security Bond(Customs & Excise Bond)",
            "Temporary Importation Bond",
            "Professional Indemnity",
            "Bankers Blanket Indemnity",
            "Fire & Allied Perils",
            "Motor Third Party Fire & Theft",
            "Contractors Plant & Machinery",
            "Motor Third Party Liability",
            "Fidelity Guarantee",
            "Public/Product Liability",
            "Business Interruption",
            "Cash In Transit",
            "Boiler & Pressure Plant",
            "Bankers Blanket Bond",
            "Plant & Machinery",
            "Bid Bond",
            "Customs Warehouse Bond",
            "Electronic Equipment Insurance",
            "Fire Assets All Risks and Business Interruption",
            "Fire Loss Profit",
            "Property All Risks, Machinery Breakdown & Business Interruption",
            "Motor Comprehensive (Automobile Fac Facility)",
            "Contractual Liability",
            "Advance Payment Bond",
            "Fire",
            "Petroleum Bond",
            "Motor Comprehensive Fleet (Automobile Fac Facility)",
            "Environmental, Social, Health and Safety (ESHS) Performance Bond",
            "Deterioration of Stock",
            "Cole Class Of Business",
            "Temporary Exportation/Importation Bond",
            "Transit Bond",

        ]
    )

    sum_ins = st.number_input(
        "Sum Insured",
        min_value=0.0, step=1000.0, format="%.2f"
    )

    # User supplies premium figure, rate is auto-derived
    fac_premium_input = st.number_input(
        "Facultative Premium",
        min_value=0.0, step=100.0, format="%.2f"
    )

    currency = st.selectbox("Currency", ["GHS", "USD"])

    brokerage = st.number_input("Brokerage",  min_value=0.0,
                                value=200.0, step=10.0)
    commission = st.number_input("Commission %", min_value=0.0, max_value=100.0,
                                 value=26.0, step=0.1)

    reinsurer = st.selectbox(
        "Reinsurer",
        [
            "Vanguard Assurance Company Limited",
            "Bedrock Insurance Company Limited",
            "Best Assurance Company Limited",
            "Loyalty Insurance Company Limited",
            "SIC Insurance PLC",
            "Millennium Insurance Company Limited",
            "RegencyNem Insurance Company",
            "Ghana Union Assurance Limited",
            "Unique Insurance Company Limited",
            "Donewell Insurance Limited",
            "Hollard Insurance Ghana Limited",
            "Enterprise Insurance Company Limited",
            "Coronation Insurance Ghana Limited",
            "Glico General Insurance Company Limited",
            "Phoenix Insurance Company Ghana Limited",
            "Serene Insurance Company Limited",
            "Sanlamallianz General Insurance Ghana",
            "Quality Insurance Company",
            "Imperial General Assurance Company Limited",
            "Provident Insurance Limited Company",
            "Nsia Insurance Company Limited",
            "Sunu Assurances Ghana Limited",
            "Star Assurance Limited",
            "Prime Insurance Company Limited",
            "Priority Insurance Company Limited",
            "Cole Insurance Company Limited",

        ]
    )


    # Calculate quoted fac rate and brokerage
    quoted_fac_rate = (fac_premium_input / sum_ins * 100) if sum_ins else 0.0
    quoted_brokerage_rate = (brokerage / fac_premium_input * 100) if fac_premium_input else 0.0
    st.markdown("---")
    st.write("**Quoted Facultative Rate(%)**")
    st.info(f"{quoted_fac_rate:.2f}")
    st.markdown("---")
    st.write("**Brokerage Rate(%)**")
    st.info(f"{quoted_brokerage_rate:.2f}")

    predict_btn = st.button("Advise")

# ── 4. Main panel – prediction & advisories ────────────────────────────────
st.title("📊 Reinsurance Quotation Index")

if predict_btn:
    # build feature frame for the model
    row = pd.DataFrame([{
        "fac_sum_insured": sum_ins,
        "business_name":   business,
        "currency":        currency,
        "brokerage":       brokerage,
        "commission":      commission,
        "reinsured":       reinsurer
    }])

    # run prediction
    pred_prem = float(model.predict(row)[0])
    pred_rate = pred_prem / sum_ins if sum_ins else 0
    gap = fac_premium_input - pred_prem
    gap_pct = (gap / pred_prem) * 100 if pred_prem else 0

    # ── Results metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # flag colours and text
    if abs(gap_pct) <= 10:
        band, colour, flag = "ok",   "green",  "✅ Within normal range."
    elif gap_pct < -10:
        band, colour, flag = "under", "orange", "⚠ Under-priced."
    else:
        band, colour, flag = "over", "red",    "❌ Over-priced."

        # style tweaks so metric content doesn't clip
    st.markdown(
        """
        <style>
            /* ensure metric text wraps instead of clipping */
            div[data-testid="stMarkdownContainer"] div[data-testid="metric-container"] {
                overflow-wrap: break-word;
                white-space: normal;
                font-size: 100px;
            }
            div[data-testid="stMetricValue"]
            div[data-testid="stMarkdownContainer"] {
                overflow-wrap: break-word;
                white-space: normal;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # display flag as the premium comment
    col1.markdown("**Premium Comment**")
    col1.markdown(
        f"<span style='color:{colour}; font-weight:bold'>{flag}</span>",
        unsafe_allow_html=True,
    )

    col2.metric("Average Acceptable Market Rate",   f"{pred_rate:.2%}")
    col3.metric("Predicted vs Actual Gap %",            f"{gap_pct:+.1f}%")

    # predicted premium range guidance
    if pred_prem >= fac_premium_input:
        range_low, range_high = fac_premium_input, pred_prem
    else:
        range_low, range_high = pred_prem, fac_premium_input

    range_txt = f"{fmt_currency(range_low, currency)} – {fmt_currency(range_high, currency)}"
    col4.metric("Visal Model Rating Guide", range_txt)

    # ── Advisory panel ─────────────────────────────────────────────────────
    advice = {
        "ok": [
            # Cedant
            "✔ Fair price. You pay about what the market expects.",
            # Broker
            "✓ Easy placement. Commission is fine and client should be happy.",
            # Reinsurer
            "✓ Fair return for risk taken. Low chance of push-back."
        ],
        "under": [
            # Cedant
            "⚠ Cheap now, but you might keep too much risk. A big loss could hurt.",
            # Broker
            "⚠ Lower commission and reinsurer may refuse. Be ready to justify the low rate.",
            # Reinsurer
            "⚠ Premium may not cover claims. Profit at risk."
        ],
        "over": [
            # Cedant
            "❌ You’re paying more than the model price. Ties up extra cash.",
            # Broker
            "❗ Higher commission, but client could say no or delay payment.",
            # Reinsurer
            "💰 Extra premium today, yet higher chance the policy is renegotiated or canceled."
        ]
    }
    cedant_msg, broker_msg, reins_msg = advice[band]

    st.subheader("Implications")
    cA, cB, cC = st.columns(3)
    cA.info(f"💼 **Cedant**\n\n{cedant_msg}")
    cB.warning(f"🤝 **Broker**\n\n{broker_msg}")
    cC.error(f"🏢 **Reinsurer**\n\n{reins_msg}")

else:
    st.write("⬅ Configure the policy on the left, then click **Advise** to see the benchmark premium and guidance.")
