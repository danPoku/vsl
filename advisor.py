# fac_check_app.py
# ────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import math

st.set_page_config(page_title="VisalRE Premium Check", page_icon="📊")

# ── 1. Load pickled model (cached) ──────────────────────────────────────────


@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model(Path(__file__).with_name("visal_re_predictor.pkl"))

# ── Load reinsurer default-band lookup ────────────────────────────────
@st.cache_data
def load_band_lookup(csv_path: Path):
    df_band = (
        pd.read_csv(csv_path)
          .assign(reinsured=lambda d: d["reinsured"].str.strip())
          .assign(band=lambda d: d["band_x"].str.lower())
    )
    return dict(zip(df_band["reinsured"], df_band["band"]))

band_lookup = load_band_lookup(Path(__file__).with_name("prem_adequacy_with_bands.csv"))

band_desc = {
    "low":        "Low defaulter",
    "moderate":   "Moderate defaulter",
    "high":       "High defaulter",
}

@st.cache_data
def load_error_bands(csv_path: Path) -> dict:
    """Return {LOB: (lo80, hi80)} from lob_error_bands.csv."""
    df = pd.read_csv(csv_path).set_index("business_name")
    return {lob: (row.lo80, row.hi80) for lob, row in df.iterrows()}

@st.cache_data
def load_model_meta(json_path: Path) -> dict:
    import json
    return json.loads(Path(json_path).read_text())

bands_dict = load_error_bands(Path(__file__).with_name("lob_error_bands.csv"))
meta       = load_model_meta(Path(__file__).with_name("model_meta.json"))
MAE        = meta["mae"]          # overall MAE saved during training

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

    risk_occupation = st.selectbox(
        "Risk Occupation",
        [
            "Agribusiness",
            "Aluminium Processing",
            "Bank",
            "BDC Petroleum Company",
            "Cable Manufacturing",
            "Car Dealership",
            "Ceramics Manufacturing",
            "Construction",
            "Corrugated Cardboard Production",
            "Canned Seafoods Manufacturing",
            "Education",
            "Energy Generation",
            "Ethanol Distillery",
            "Food Manufacturing",
            "Flour Production",
            "Furniture Store",
            "Gold Refinery",
            "Gold Trading / Refinery",
            "Heavy-Duty Vehicle Dealership",
            "Hotelier",
            "ICT Solutions",
            "ID Manufacturing / Processing",
            "Import / Export of Electrical Products",
            "IT Solutions",
            "Logistics",
            "Mining",
            "Packaging Solutions",
            "Petroleum Retailing",
            "Petroleum Trading",
            "Pipes Manufacturing",
            "Pre-fab Building Manufacturer",
            "Real Estate",
            "Residential Unit",
            "Restaurant",
            "Restaurant Chain",
            "Security Services",
            "Solar Panel Distribution",
            "Steel Manufacturing",
            "Supermarket",
            "Timber Processing",
            "Warehouse / Warehousing",
            "Wholesale Food Distributors",
        ]
    )

    currency = st.selectbox(
        "Currency", ["GHS", "USD", "EUR"])

    sum_ins = st.number_input(
        "Sum Insured",
        min_value=0.0, step=1000.0, format="%.2f"
    )

    # User supplies premium figure, rate is auto-derived
    premium_input = st.number_input(
        "Premium",
        min_value=0.0, step=100.0, format="%.2f"
    )

    brokerage = st.number_input("Brokerage",  min_value=0.0,
                                value=200.0, step=10.0)
    commission = st.number_input("Commission %", min_value=0.0, max_value=100.0,
                                 value=26.0, step=0.1)

    insurer = st.selectbox(
        "Insurer",
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
    quoted_fac_rate = (premium_input / sum_ins * 100) if sum_ins else 0.0
    quoted_brokerage_rate = (
        brokerage / premium_input * 100) if premium_input else 0.0
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
        "reinsured":       insurer
    }])

    # run prediction
    pred_prem = float(model.predict(row)[0])
    pred_rate = pred_prem / sum_ins if sum_ins else 0
    gap = premium_input - pred_prem
    gap_pct = (gap / pred_prem) * 100 if pred_prem else 0

    # ── Results metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    # flag colours and text
    lob_lo, lob_hi = bands_dict.get(business, (-10, 10))  # fallback if LOB missing

    if gap_pct < lob_lo:
        price_band, colour, flag = "under", "orange", "⚠ Under-priced."
    elif gap_pct > lob_hi:
        price_band, colour, flag = "over",  "red",    "❌ Over-priced."
    else:
        price_band, colour, flag = "ok",    "green",  "✅ Within normal range."
    
    band = price_band

        # style tweaks so metric content doesn't clip
    st.markdown(
        """
        <style>
            div[data-testid="stMarkdownContainer"] {
                overflow-wrap: break-word;
                white-space: normal;
                font-size: 15px;
            }
            div[data-testid="stMetricValue"] {
                overflow-wrap: break-word;
                white-space: normal;
                font-size: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Metrics display
    col1.markdown("**Premium Comment**")
    col1.markdown(
        f"<span style='color:{colour}; font-weight:bold'>{flag}</span>",
        unsafe_allow_html=True,
    )

    col2.metric("Average Acceptable Market Rate",   f"{pred_rate:.2%}")

    # predicted premium range guidance - ±1.96·MAE (95 % error band)
    range_low  = max(0, pred_prem - 1.96 * MAE)
    range_high = pred_prem + 1.96 * MAE 

    range_txt = f"{fmt_currency(range_low, currency)} – {fmt_currency(range_high, currency)}"
    col3.metric("Visal Model Rating Guide", range_txt)
    
    # Reinsurer default band
    band_key  = band_lookup.get(insurer, None)
    default_txt = band_desc.get(band_key, "No data available")
    col4.metric("Insurer Premium Payment Profile", default_txt)

    # ── Advisory panel ─────────────────────────────────────────────────────
    advice_matrix = {
        "ok": {
            "low": [
                "👍 Rate is fair and the insurer usually pays on time – good deal.",
                "👍 Smooth placement; normal commission, low collection risk.",
                "👍 Fair premium and prompt payer – business as usual."
            ],
            "moderate": [
                "👍 Rate is fine but this insurer can be slow in payment – set clear due dates.",
                "🙂 Deal works, yet chase invoices quickly.",
                "⚠ Fair rate; settle premiums fast to keep terms."
            ],
            "high": [
                "⚠ Good rate, yet insurer often pays late – add strict credit terms.",
                "⚠ Commission OK, but expect follow-ups on payment.",
                "⚠ Rate okay, history of arrears – cash before cover where possible."
            ],
        },
        "under": {
            "low": [
                "⚠ Cheap cover from a reliable payer – be sure limits are enough.",
                "😐 Lower commission but quick cash; confirm scope is adequate.",
                "⚠ Thin premium – watch insurer's claims ratio."
            ],
            "moderate": [
                "⚠ Cheap price and payer sometimes late – keep retention small.",
                "⚠ Discounted premium; send reminders early.",
                "⚠ Low premium; pay promptly to avoid stricter terms."
            ],
            "high": [
                "🚩 Very cheap but insurer often pays late – ask for deposit or bank guarantee.",
                "🚩 Low commission plus high collection risk – rethink placement.",
                "🚩 Premium may not cover risk; insist on cash up-front."
            ],
        },
        "over": {
            "low": [
                "❗ You’re paying more than needed, even with a good payer – negotiate down.",
                "❗ Higher commission, but client may object; be ready.",
                "🙂 Extra premium for you, payment likely on time – still overpriced."
            ],
            "moderate": [
                "❗ Pricey and payer sometimes late – ask for a discount or staged payments.",
                "❗ Commission up, but expect slower cash; manage client expectations.",
                "⚠ High rate; pay quickly to keep cover active."
            ],
            "high": [
                "🚨 Expensive and chronic late payer – high financial risk; consider other markets.",
                "🚨 Commission good, but collection will be tough – advise cash before cover.",
                "🚩 Overpriced cover; past arrears mean tight credit control or decline."
            ],
        },
    }

    # ── ❷ Look up the three messages ------------------------------------------------
    band_key  = band_lookup.get(insurer, None)            # "low" / "moderate" / "high"
    band_key  = band_key if band_key in {"low","moderate","high"} else "low"

    cedant_msg, broker_msg, reins_msg = advice_matrix[band][band_key]

    st.subheader("Implications")
    cA, cB, cC = st.columns(3)
    cA.info(f"💼 **Cedant/Insurer**\n\n{cedant_msg}")
    cB.warning(f"🤝 **Broker**\n\n{broker_msg}")
    cC.error(f"🏢 **Reinsurance Market**\n\n{reins_msg}")

else:
    st.write("⬅ Configure the policy on the left, then click **Advise** to see the benchmark premium and guidance.")