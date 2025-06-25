"""
VisalRE Premium Check – state-safe
"""
import json
from pathlib import Path
import pickle
import yaml

import pandas as pd
import streamlit as st

from db import log_submission_gsheets
from rpc import rps, rps_band

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="VisalRE Premium Check", page_icon="📊")

# ── -----------------------------------------------------------------------
# 1.  LOAD RESOURCES (cached)                                               
# ── -----------------------------------------------------------------------

@st.cache_resource
def load_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_band_lookup(csv_path: Path):
    df = (
        pd.read_csv(csv_path)
          .assign(reinsured=lambda d: d["reinsured"].str.strip())
          .assign(band=lambda d: d["band_x"].str.lower())
    )
    return dict(zip(df["reinsured"], df["band"]))

@st.cache_data
def load_error_bands(csv_path: Path):
    df = pd.read_csv(csv_path).set_index("business_name")
    return {lob: (row.lo80, row.hi80) for lob, row in df.iterrows()}

@st.cache_data
def load_meta(json_path: Path):
    return json.loads(Path(json_path).read_text())

@st.cache_data
def load_options_yaml(yaml_path: Path):
    with open(yaml_path, "r") as f:
        options = yaml.safe_load(f)
    return options

ROOT = Path(__file__).parent.parent
model          = load_model(ROOT / "models" / "predictors" / "visal_re_predictor.pkl")
broker_model   = load_model(ROOT / "models" / "predictors" / "brokerage_predictor.pkl")
band_lookup    = load_band_lookup(ROOT / "data"   / "prem_adequacy_with_bands.csv")
bands_dict     = load_error_bands(ROOT / "data"   / "lob_error_bands.csv")
meta           = load_meta(ROOT  / "models" / "meta" / "model_meta_v2.json")
broker_meta    = load_meta(ROOT  / "models" / "meta" / "broker_model_meta.json")
options        = load_options_yaml(ROOT / "data" / "options.yml")

# ── -----------------------------------------------------------------------
# 2.  STATIC CONSTANTS                                                      
# ── -----------------------------------------------------------------------
band_desc  = {"low": "Low defaulter", "moderate": "Moderate defaulter", "high": "High defaulter"}
PREM_CONFIDENCE_INTERVAL, BRK_CONFIDENCE_INTERVAL = 1.96, 1.28
MAE, MAE_PCT, BROKER_MAE = meta["mae"], meta["mae_pct"], broker_meta["mae"]

# … (POLICY_OPTIONS, OCC_OPTIONS, INSURER_OPTIONS remain unchanged) …
# --- snip for brevity – paste the same large lists here unchanged ----------
POLICY_OPTIONS = options["policy_options"]
OCC_OPTIONS    = options["occ_options"]
INSURER_OPTIONS = options["insurer_options"]

DEFAULTS = {
    "policy": POLICY_OPTIONS[0],
    "risk_occupation": OCC_OPTIONS[0],
    "currency": "GHS",
    "insurer": INSURER_OPTIONS[0],
    "sum_insured": 0.0,
    "premium": 0.0,
    "brokerage": 3.0,
    "commission": 26.0,
    "other_deductions": 0.0,
}

# ── -----------------------------------------------------------------------
# 3.  SESSION INITIALISATION & HELPERS                                      
# ── -----------------------------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "inputs" not in st.session_state:
    st.session_state.inputs = DEFAULTS.copy()

# ── Validation helper ────────────────────────────────────────────────────
def all_required_inputs_present() -> bool:
    """Return True only when every critical input has a sensible value."""
    return (
        st.session_state.policy.strip()                   # policy type chosen
        and st.session_state.currency.strip()             # currency chosen
        and st.session_state.sum_insured > 0              # positive sum insured
        and st.session_state.premium > 0                  # positive premium
        and st.session_state.brokerage >= 0               # brokerage entered
        and st.session_state.commission >= 0              # commission entered
        and st.session_state.insurer.strip()              # insurer chosen
    )

def fmt_currency(x: float, cur: str):
    return f"{'GHS' if cur == 'GHS' else '$'} {x:,.2f}"

def reset_form():
    """Callback for the *Reset* button."""
    st.session_state.update(DEFAULTS)
    st.session_state.results = None

# ── -----------------------------------------------------------------------
# 4.  SIDEBAR – DATA ENTRY                                                  
# ── -----------------------------------------------------------------------
with st.sidebar:
    st.header("Enter policy details")

    # Widgets are keyed exactly to DEFAULTS keys
    business         = st.selectbox("Policy Type", POLICY_OPTIONS,      key="policy")
    risk_occupation  = st.selectbox("Risk Occupation", OCC_OPTIONS,     key="risk_occupation")
    currency         = st.selectbox("Currency", ["GHS", "USD", "EUR"],  key="currency")
    sum_ins          = st.number_input("Sum Insured",   min_value=0.0,  step=1_000.0, key="sum_insured")
    premium_input    = st.number_input("Premium",       min_value=0.0,  step=100.0,   key="premium")
    brokerage        = st.number_input("Brokerage Rate %", 0.0, 100.0,  key="brokerage")
    commission       = st.number_input("Commission %",     0.0, 100.0,  key="commission")
    other_deductions = st.number_input("Other Deductions %", 0.0, step=10.0,
                                       key="other_deductions", value=0.0)
    insurer          = st.selectbox("Insurer", INSURER_OPTIONS, key="insurer")

    # Quick calculations
    quoted_fac_rate       = (premium_input / sum_ins * 100) if sum_ins else 0.0
    quoted_brokerage_fee  = (brokerage / 100) * premium_input
    st.markdown("---")
    st.info(f"**Placement Rate (%)**  \n{quoted_fac_rate:.2f}")
    st.markdown("---")
    st.info(f"**Placement Brokerage Fee**  \n{fmt_currency(quoted_brokerage_fee, currency)}")
    st.markdown("---")
    st.info(f"**Total Deductions**  \n{(brokerage + commission + other_deductions):.2f}%")

    # Action buttons
    advise_btn = st.button("Advise", type="primary")
    st.button("Reset",  type="secondary", on_click=reset_form)
    
    # ── footer ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("Powered by **Vaarde AI**")
    
    
# ── -----------------------------------------------------------------------
# 5.  MAIN PANEL                                                            
# ── -----------------------------------------------------------------------
st.title("Reinsurance Placement Index")

# --------------------------------------------------------------------------
# Compute & store results (on demand)                                       
# --------------------------------------------------------------------------
if advise_btn:
    if not all_required_inputs_present():
        st.error("Please fill in: Policy Type, Currency, Sum Insured, "
                 "Premium, Brokerage %, Commission %, and Insurer before "
                 "requesting advice.")
        st.stop() 

    # ---- Build feature frames -------------------------------------------
    row = pd.DataFrame([{
        "fac_sum_insured": sum_ins,
        "business_name":   business,
        "currency":        currency,
        "brokerage":       quoted_brokerage_fee,
        "commission":      commission,
        "reinsured":       insurer,
    }])

    bro_row = pd.DataFrame([{
        "fac_sum_insured": sum_ins,
        "fac_premium":     premium_input,
        "business_name":   business,
        "commission":      commission,
        "reinsured":       insurer,
    }])

    # ---- Model predictions ---------------------------------------------
    pred_prem        = float(model.predict(row)[0])
    # st.write("**Visal Model Predicted Premium:**", fmt_currency(pred_prem, currency))
    pred_rate        = pred_prem / sum_ins if sum_ins else 0
    gap              = premium_input - pred_prem
    gap_pct          = (gap / pred_prem) * 100 if pred_prem else 0

    SOUND_PREMIUM_THRESHOLD = 0.5
    is_premium_sound = premium_input >= SOUND_PREMIUM_THRESHOLD * pred_prem

    # ---- Price band -----------------------------------------------------

    OVERRIDE_BANDS = {
        "Boiler & Pressure Plant",
        "Business Interruption",
        "Money Insurance",
        "Motor Third Party Fire & Theft",
        "Performance Bond",
        "Petroleum Bond",
        "Professional Indemnity",
        "Public Liability",
        "Transit Bond",
    }

    if business in OVERRIDE_BANDS:
        lob_lo, lob_hi = -10, 10
    else:
        lob_lo, lob_hi = bands_dict.get(business, (-10, 10))

    # lob_lo, lob_hi   = bands_dict.get(business, (-10, 10)) 
    if gap_pct < lob_lo:
        price_band, colour, flag = "under", "orange", "⚠ Under-priced."
    elif gap_pct > lob_hi:
        price_band, colour, flag = "over",  "red",    "❌ Over-priced."
    else:
        price_band, colour, flag = "ok",    "green",  "✅ Within normal range."

    # ---- Premium range (±1.96 · MAE) ------------------------------------
    range_low  = max(1, pred_prem * (1 - PREM_CONFIDENCE_INTERVAL * MAE_PCT)) \
                 if (pred_prem - PREM_CONFIDENCE_INTERVAL * MAE) < 0 \
                 else max(0, pred_prem - PREM_CONFIDENCE_INTERVAL * MAE)
    range_high = pred_prem + PREM_CONFIDENCE_INTERVAL * MAE

    # ---- Insurer default profile ---------------------------------------
    band_key   = band_lookup.get(insurer, None)
    default_txt = band_desc.get(band_key, "No data available")

    # ---- Brokerage predictions -----------------------------------------
    pred_broker_fee  = float(broker_model.predict(bro_row)[0])
    pred_broker_rate = (pred_broker_fee / premium_input * 100) if premium_input else 0
    broker_gap_pct   = 100 * (quoted_brokerage_fee - pred_broker_fee) / pred_broker_fee \
                       if pred_broker_fee else 0
    br_range_low     = max(0, pred_broker_fee - BRK_CONFIDENCE_INTERVAL * BROKER_MAE)
    br_range_high    = pred_broker_fee + BRK_CONFIDENCE_INTERVAL * BROKER_MAE
    if quoted_brokerage_fee < br_range_low:
        br_colour, br_flag, br_band = "orange", "⚠ Low brokerage",  "low"
    elif quoted_brokerage_fee > br_range_high:
        br_colour, br_flag, br_band = "red",    "❌ High brokerage", "high"
    else:
        br_colour, br_flag, br_band = "green",  "✅ Fair brokerage", "fair"

    # ---- Deductions band -----------------------------------------------
    total_deduct_pct = brokerage + commission
    if 28 <= total_deduct_pct <= 38:
        ded_band = "acceptable"
    elif total_deduct_pct < 28:
        ded_band = "low"
    else:
        ded_band = "high"

    # ---- RPS ------------------------------------------------------------
    RPS_VAL = rps(gap_pct, broker_gap_pct, total_deduct_pct, band_key, business)
    rps_letter, rps_colour, rps_comment = rps_band(RPS_VAL)

    # ---- Difficulty -----------------------------------------------------
    score = (
        {"under": 2, "ok": 0, "over": 1}[price_band] +
        {"low": 0, "fair": 0, "high": 1}[br_band] +
        {"acceptable": 0, "low": 1, "high": 1}[ded_band] +
        {"low": 0, "moderate": 1, "high": 2}[band_key]
    )
    difficulty = "easy" if score <= 1 else "moderate" if score <= 3 else "difficult"

    # ---- Advisory messages ---------------------------------------------
    cedant_msg  = {
        "under": "Premium is below model range - reinsurers may load or decline.",
        "ok":    "Premium sits in the fair range.",
        "over":  "Premium is above model range – client may be overpaying.",
    }[price_band]

    brokerage_msg = {
        "low":  "Brokerage below peer level; revenue impact but appeals to reinsurer.",
        "fair": "Brokerage within peer range.",
        "high": "High brokerage – reinsurer will ask for justification.",
    }[br_band]

    ded_msg = {
        "acceptable": "Total deductions are within the 28-38 % comfort zone.",
        "low":        "Deductions below norm – reinsurer keeps more net premium.",
        "high":       "Deductions above norm – reinsurer margin is thin.",
    }[ded_band]

    default_msg = {
        "low":      "Insurer pays promptly.",
        "moderate": "Insurer sometimes late – credit terms needed.",
        "high":     "Insurer often late – cash-before-cover likely.",
    }[band_key]

    difficulty_msg = {
        "easy":      "Placement looks easy.",
        "moderate":  "Placement may need negotiation.",
        "difficult": "Placement will be difficult – prepare alternatives.",
    }[difficulty]

    broker_msg = (
        f"{difficulty_msg}  Reasons: price **{price_band}**, "
        f"brokerage **{br_band}**, deductions **{ded_band}**, "
        f"insurer default profile **{band_key}**."
    )

    reins_msg = " ".join([default_msg, ded_msg, brokerage_msg])
    
    # ── DISPLAY  •  METRIC GRID ────────────────────────────────────────────────
    # Styling for metric grid
    st.markdown(
        """
        <style>
        /* metric titles wrap */
        div[data-testid="stMarkdownContainer"] {
            white-space: normal;      
            overflow-wrap: break-word; 
        }

        /* slightly smaller value font + allow wrap for very wide figures   */
        div[data-testid="stMetricValue"] {
            font-size: 12px;    
            white-space: normal;
            overflow-wrap: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Row 1 – premium-centric metrics
    row1_col1, row1_col2, row1_col3 = st.columns(3, gap="small")

    # row1_col1.markdown("**Premium Comment**")
    # row1_col1.markdown(
    #     f"<span style='color:{colour}; font-weight:bold'>{flag}</span>",
    #     unsafe_allow_html=True
    # )
    
    row1_col1.metric("**Visal Model Predicted Premium**", fmt_currency(pred_prem, currency))

    row1_col2.metric("**Average Acceptable Market Rate**", f"{pred_rate:.2%}")

    range_txt = f"{fmt_currency(range_low, currency)} – {fmt_currency(range_high, currency)}"
    row1_col3.metric("**Visal Model Rating Guide**", range_txt)

    # Row 2 – brokerage metrics   (only if the premium itself is sensible)
    br_col1, br_col2, br_col3 = st.columns(3, gap="small")

    if is_premium_sound:
        # br_col1.markdown("**Brokerage Comment**")
        # br_col1.markdown(
        #     f"<span style='color:{br_colour}; font-weight:bold'>{br_flag}</span>",
        #     unsafe_allow_html=True
        # )
        br_col1.metric("**Insurer Premium Payment Profile**", default_txt)
        br_col2.metric("**Predicted Brokerage Rate**", f"{pred_broker_rate:.2f}%")
        br_col3.metric("**Reinsurance Placement Score**", f"{RPS_VAL}/100")
        br_col3.markdown(
            f"<span style='color:{rps_colour}; font-weight:bold'>"
            f"{rps_letter} – {rps_comment}</span>",
            unsafe_allow_html=True
        )
    else:
        st.warning(
            "Brokerage guidance and placement score are hidden "
            "because the quoted premium is outside the model’s credible range."
        )

    # ---- Display metrics ------------------------------------------------
    st.subheader("Implications")
    if sum_ins > 0:
        c1, c2, c3 = st.columns(3)
        c1.info(f"💼 **Cedant / Insurer**\n\n{cedant_msg} {brokerage_msg} {ded_msg} {difficulty_msg}")
        c2.warning(f"🤝 **Broker**\n\n{broker_msg}")
        c3.error(f"🏢 **Reinsurer**\n\n{reins_msg}")
    else:
        st.warning("No advisory available – enter a valid sum insured.")

    # ---- Persist & log --------------------------------------------------
    st.session_state.results = {
        "premium":  {"pred_rate": pred_rate,      "range": (range_low, range_high),
                     "flag": flag, "colour": colour, "band": price_band},
        "brokerage": {"pred_rate": pred_broker_rate, "flag": br_flag,
                      "colour": br_colour, "band": br_band},
        "rps": {"value": RPS_VAL, "letter": rps_letter,
                "colour": rps_colour, "comment": rps_comment},
        "difficulty": difficulty,
        "messages": {"cedant": cedant_msg, "broker": broker_msg, "reins": reins_msg},
    }
    st.session_state.inputs = {k: st.session_state[k] for k in DEFAULTS}

    log_submission_gsheets({
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
    })

# --------------------------------------------------------------------------
# Show placeholder if no results yet                                        
# --------------------------------------------------------------------------
if st.session_state.results is None and not advise_btn:
    st.write("⬅ Configure the policy on the left, then click **Advise** to see guidance.")
