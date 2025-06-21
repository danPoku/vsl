import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from db import log_submission_gsheets

st.set_page_config(page_title="VisalRE Premium Check", page_icon="📊")
# ── 0. Initialize database ────────────────────────────────────────────────
# init_db()

# ── 1. Load pickled model (cached) ──────────────────────────────────────────


@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


model = load_model(Path(__file__).parent.parent / "models" /
                   "predictors" / "visal_re_predictor.pkl")
broker_model = load_model(Path(__file__).parent.parent / "models" /
                          "predictors" / "brokerage_predictor.pkl")

# ── Load reinsurer default-band lookup ────────────────────────────────


@st.cache_data
def load_band_lookup(csv_path: Path):
    df_band = (
        pd.read_csv(csv_path)
          .assign(reinsured=lambda d: d["reinsured"].str.strip())
          .assign(band=lambda d: d["band_x"].str.lower())
    )
    return dict(zip(df_band["reinsured"], df_band["band"]))


band_lookup = load_band_lookup(Path(__file__).parent.parent / "data" /
                               "prem_adequacy_with_bands.csv")

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


bands_dict = load_error_bands(
    Path(__file__).parent.parent / "data" / "lob_error_bands.csv")
# Load model metadata
meta = load_model_meta(Path(__file__).parent.parent / "models" / "meta" /
                       "model_meta_v2.json")
broker_meta = load_model_meta(Path(__file__).parent.parent / "models" / "meta" /
                              "broker_model_meta.json")

# Extract metadata
PREM_CONFIDENCE_INTERVAL = 1.96 # 95% confidence interval
BRK_CONFIDENCE_INTERVAL = 1.28 # 80% confidence interval
                                
MAE = meta["mae"]          # overall MAE saved during training
MAE_PCT = meta["mae_pct"]    # overall MAE saved during training, as percentage
BROKER_MAE = broker_meta["mae"]   # overall MAE saved during training

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
        min_value=0.0, step=1000.0
    )

    # User supplies premium figure, rate is auto-derived
    premium_input = st.number_input(
        "Premium",
        min_value=0.0, step=100.0
    )

    brokerage = st.number_input("Brokerage Rate %",  min_value=0.0, max_value=100.0,
                                value=5.0, step=1.0)
    commission = st.number_input("Commission %", min_value=0.0, max_value=100.0,
                                 value=26.0, step=0.1)
    other_deductions = st.number_input(
        "Other Deductions %",
        min_value=0.0, step=10.0, value=0.0
    )

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
    quoted_brokerage_fee = (brokerage/100) * premium_input
    st.markdown("---")
    st.write("**Placement Rate(%)**")
    st.info(f"{quoted_fac_rate:.2f}")
    st.markdown("---")
    st.write("**Placement Brokerage Fee**")
    st.info(fmt_currency(quoted_brokerage_fee, currency))
    st.markdown("---")
    st.write("**Total Deductions**")
    st.info(f"{(brokerage + commission + other_deductions):.2f}%")

    predict_btn = st.button("Advise")

# ── 4. Main panel – prediction & advisories ────────────────────────────────
st.title("📊 Reinsurance Placement Index")

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

    bro_row = pd.DataFrame([{
        "fac_sum_insured": sum_ins,
        "fac_premium":     premium_input,
        "business_name":   business,
        "commission":      commission,
        "reinsured":       insurer
    }])

    # run premium prediction
    pred_prem = float(model.predict(row)[0])
    pred_rate = pred_prem / sum_ins if sum_ins else 0
    gap = premium_input - pred_prem
    gap_pct = (gap / pred_prem) * 100 if pred_prem else 0

    # Sound premium threshold (not less than 50% of predicted premium)
    SOUND_PREMIUM_THRESHOLD = 0.5
    is_premium_sound = (premium_input >= SOUND_PREMIUM_THRESHOLD * pred_prem)

    # ── Results metrics ────────────────────────────────────────────────────
    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

    # flag colours and text
    lob_lo, lob_hi = bands_dict.get(
        business, (-10, 10))  # fallback if LOB missing

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

    # predicted premium range guidance - ±1.96·MAE (95 % error band)
    range_low = pred_prem - PREM_CONFIDENCE_INTERVAL * MAE
    if range_low < 0:
        # Use percentage-based formula if lower bound is negative
        range_low = max(1, pred_prem * (1 - PREM_CONFIDENCE_INTERVAL * MAE_PCT))
    else:
        # Use absolute value formula if lower bound is non-negative
        range_low = max(0, range_low)
    range_high = pred_prem + PREM_CONFIDENCE_INTERVAL * MAE
    # range_low = max(1, pred_prem *(1 - PREM_CONFIDENCE_INTERVAL * MAE_PCT))
    # range_high = pred_prem * (1 + PREM_CONFIDENCE_INTERVAL * MAE_PCT)

    # Reinsurer default band
    band_key = band_lookup.get(insurer, None)
    default_txt = band_desc.get(band_key, "No data available")

    # Metrics display
    if sum_ins > 0:
        row1_col1.markdown("**Premium Comment**")
        row1_col1.markdown(
            f"<span style='color:{colour}; font-weight:bold'>{flag}</span>",
            unsafe_allow_html=True,
        )

        row1_col2.metric("Average Acceptable Market Rate",
                         f"{pred_rate:.2%}")

        range_txt = f"{fmt_currency(range_low, currency)} – {fmt_currency(range_high, currency)}"
        row1_col3.metric("Visal Model Rating Guide", range_txt)

        row1_col4.metric("Insurer Premium Payment Profile", default_txt)
    else:
        st.warning("Enter a valid sum insured for model predictions")

    # ── Row 2 – brokerage KPIs ─────────────────────────────────────────────
    # run brokerage prediction
    pred_broker_fee = float(broker_model.predict(bro_row)[
        0])               # currency
    pred_broker_rate = (pred_broker_fee / premium_input *
                        100) if premium_input else 0
    broker_gap_pct = 100 * (quoted_brokerage_fee - pred_broker_fee) / \
        pred_broker_fee if pred_broker_fee else 0

    # brokerage fairness – using ±1.28·MAE band
    br_range_low = max(0, pred_broker_fee - BRK_CONFIDENCE_INTERVAL * BROKER_MAE)
    br_range_high = pred_broker_fee + BRK_CONFIDENCE_INTERVAL * BROKER_MAE

    if quoted_brokerage_fee < br_range_low:
        br_colour, br_flag = "orange", "⚠ Low brokerage"
    elif quoted_brokerage_fee > br_range_high:
        br_colour, br_flag = "red",    "❌ High brokerage"
    else:
        br_colour, br_flag = "green",  "✅ Fair brokerage"

    br_col1, br_col2, br_col3 = st.columns(3)

    # Show brokerage metrics if premium is sound
    if is_premium_sound:
        # 1 brokerage comment chip
        br_col1.markdown("**Brokerage Comment**")
        br_col1.markdown(
            f"<span style='color:{br_colour}; font-weight:bold'>{br_flag}</span>",
            unsafe_allow_html=True
        )
        # 2 predicted brokerage rate
        br_col2.metric("Predicted Brokerage Rate", f"{pred_broker_rate:.2f}%")
    else:
        st.warning(
            "Brokerage guidance is not shown as premium entered is either invalid or extremely below the model's predicted range")

    # ──────────────────────────────────────────────────────────────────────────
    #  A D V I S O R Y   E N G I N E   (premium · brokerage · deductions · default)
    #  --------------------------------------------------------------------------
    #  1.  classify each metric → simple bands
    #  2.  derive a placement-difficulty score
    #  3.  build plain-English messages for cedant, broker, reinsurer
    # ──────────────────────────────────────────────────────────────────────────

    # 1️⃣  ----- BAND CLASSIFIERS ----------------------------------------------

    # premium_band already computed earlier  →  price_band  (under | ok | over)

    # brokerage band  (low | fair | high)
    if quoted_brokerage_fee < br_range_low:
        br_band = "low"
    elif quoted_brokerage_fee > br_range_high:
        br_band = "high"
    else:
        br_band = "fair"

    # deductions band  (acceptable | low | high)
    total_deduct_pct = brokerage + commission          # add other deductions here
    if 28 <= total_deduct_pct <= 38:
        ded_band = "acceptable"
    elif total_deduct_pct < 28:
        ded_band = "low"
    else:
        ded_band = "high"

    # insurer payment default band  → band_key  ("low"|"moderate"|"high") loaded earlier
        # ── RQS – Reinsurance Quotability Score ----------------------------------
    price_score = {"under": 0.5, "ok": 1.0, "over": 0.8}[price_band]
    broker_score = {"low": 0.8, "fair": 1.0, "high": 0.6}[br_band]

    if ded_band == "acceptable":
        ded_score = 1.0
    elif ded_band == "low":
        ded_score = 0.9
    elif total_deduct_pct <= 45:
        ded_score = 0.6
    else:                                 # >45 %
        ded_score = 0.3

    default_score = {"low": 1.0, "moderate": 0.7, "high": 0.4}[band_key]

    hi_risk_lobs = {
        "Aviation", "Marine", "Performance Bond", "Energy Generation",
        "Motor Comprehensive (Automobile Fac Facility)"
    }
    lob_score = 0.6 if business in hi_risk_lobs else 1.0

    RQS = round(100 * (
        0.30 * price_score
        + 0.15 * broker_score
        + 0.15 * ded_score
        + 0.30 * default_score
        + 0.10 * lob_score
    ), 1)

    # ── Map RQS → band, colour, one-liner -----------------------------------
    if RQS >= 90:
        rqs_band = "A – Excellent"
        rqs_colour = "green"
        rqs_comment = "Top-tier submission; place confidently."
    elif RQS >= 75:                   # 75-89
        rqs_band = "B – Strong / Preferred"
        rqs_colour = "#CEFA05"  # limegreen
        rqs_comment = "Attractive risk; place with minor tweaks."
    elif RQS >= 60:                   # 60-74
        rqs_band = "C – Borderline / Conditional"
        rqs_colour = "orange"
        rqs_comment = "Placement feasible, but needs concessions or extra info."
    else:                             # 0-59
        rqs_band = "D – Weak / Decline"
        rqs_colour = "red"
        rqs_comment = "Outside appetite; likely decline."

    # 3 RQS (new)
    if is_premium_sound:
        br_col3.metric("Reinsurance Placement Score", f"{RQS}/100")
        br_col3.markdown(
            f"<span style='color:{rqs_colour}; font-weight:bold'>{rqs_band}</span><br>"
            f"<span style='font-size:0.85rem'>{rqs_comment}</span>",
            unsafe_allow_html=True)
    else:
        st.warning(
            "Placement score is not calculated as premium entered is is either invalid or extremely below the model's predicted range.")

    # 2️⃣  ----- PLACEMENT DIFFICULTY SCORE ------------------------------------
    score = 0
    score += {"under": 2, "ok": 0, "over": 1}[price_band]
    score += {"low": 0, "fair": 0, "high": 1}[br_band]
    score += {"acceptable": 0, "low": 1, "high": 1}[ded_band]
    score += {"low": 0, "moderate": 1, "high": 2}[band_key]

    if score <= 1:
        difficulty = "easy"
    elif score <= 3:
        difficulty = "moderate"
    else:
        difficulty = "difficult"

    # 3️⃣  ----- MESSAGE TEMPLATES ---------------------------------------------

    cedant_tmpl = {
        ("under",):  "Premium is below model range – reinsurers may load or decline.",
        ("ok",):     "Premium sits in the fair range.",
        ("over",):   "Premium is above model range – client may be overpaying.",
    }

    brokerage_tmpl = {
        ("low",):    "Brokerage below peer level; revenue impact but appeals to reinsurer.",
        ("fair",):   "Brokerage within peer range.",
        ("high",):   "High brokerage – reinsurer will ask for justification.",
    }

    ded_tmpl = {
        "acceptable": "Total deductions are within the 28-38 % comfort zone.",
        "low":        "Deductions below norm – reinsurer keeps more net premium.",
        "high":       "Deductions above norm – reinsurer margin is thin.",
    }

    default_tmpl = {
        "low":       "Insurer pays promptly.",
        "moderate":  "Insurer sometimes late – credit terms needed.",
        "high":      "Insurer often late – cash-before-cover likely.",
    }

    difficulty_msg = {
        "easy":      "Placement looks easy.",
        "moderate":  "Placement may need negotiation.",
        "difficult": "Placement will be difficult – prepare alternatives.",
    }

    # 4️⃣  ----- BUILD FINAL ADVICE TEXTS --------------------------------------
    cedant_msg = " ".join([
        cedant_tmpl[(price_band,)],
        brokerage_tmpl[(br_band,)],
        ded_tmpl[ded_band],
        f"{difficulty_msg[difficulty]}"
    ])

    broker_msg = (
        f"{difficulty_msg[difficulty]}  "
        f"Reasons: price: **{price_band}**, brokerage: **{br_band}**, "
        f"deductions: **{ded_band}**, insurer premium payment default profile **{band_key}**."
    )

    reins_msg = " ".join([
        default_tmpl[band_key],
        # cedant_tmpl[(price_band,)],
        ded_tmpl[ded_band],
        brokerage_tmpl[(br_band,)]
    ])

    # ── DISPLAY ----------------------------------------------------------------
    st.subheader("Implications")
    if sum_ins > 0:
        cA, cB, cC = st.columns(3)
        cA.info(f"💼 **Cedant / Insurer**\n\n{cedant_msg}")
        cB.warning(f"🤝 **Broker**\n\n{broker_msg}")
        cC.error(f"🏢 **Reinsurer**\n\n{reins_msg}")
    else:
        st.warning("No advisory available")

    # ── Log submission to database ───────────────────────────────────────
    log_data = {
        "fac_sum_insured": sum_ins,
        "business_name": business,
        "risk_occupation": risk_occupation,
        "currency": currency,
        "brokerage": brokerage,
        "commission": commission,
        "reinsured": insurer,
        "premium_input": premium_input,
        "pred_prem": pred_prem,
        "pred_rate": pred_rate,
        "prem_mae": MAE,
        "prem_confidence_interval": PREM_CONFIDENCE_INTERVAL,
        "prem_range_low": range_low,
        "prem_range_high": range_high,
        "quoted_brokerage_fee": quoted_brokerage_fee,
        "pred_broker_fee": pred_broker_fee,
        "pred_broker_rate": pred_broker_rate,
        "broker_mae": BROKER_MAE,
        "broker_confidence_interval": BRK_CONFIDENCE_INTERVAL,
        "br_range_low": br_range_low,
        "br_range_high": br_range_high,
    }
    # log_submission(log_data)
    log_submission_gsheets(log_data)
else:
    st.write("⬅ Configure the policy on the left, then click **Advise** to see the benchmark premium and guidance.")
