# Reinsurance Placement Score engine

from math import exp

# weight constants
W_PRICE   = 0.30
W_BRK     = 0.15
W_DED     = 0.15
W_DEF     = 0.25
W_LOB     = 0.15

HI_RISK_LOBS = {
    "Aviation", "Marine", "Performance Bond",
    "Energy Generation", "Motor Comprehensive (Automobile Fac Facility)",
}

def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def price_score(gap_pct: float) -> float:
    """
    gap_pct  = 100 · (Quoted - Predicted) / Predicted
    Perfect 0% → 1.0 ; ±10% → 0.8 ; ±25% → 0.3 ; worse → floor 0.1
    """
    return clamp(1.0 - abs(gap_pct) / 50, 0.1)

def brokerage_score(diff_pct: float) -> float:
    """
    diff_pct = 100 · (QuotedFee - PredictedFee) / PredictedFee
    Within ±10% → 1.0 ; cheaper for reinsurer is good (slight bonus)
    """
    if diff_pct < -30:        # far below model
        return 0.8
    if diff_pct < -10:
        return 1.05           # broker cheap – reinsurer likes
    if diff_pct < 10:
        return 1.0
    if diff_pct < 25:
        return 0.7
    return 0.4                # very high brokerage

def deduction_score(ded_pct: float) -> float:
    """
    28-38% best; score decays quadratically outside the window.
    """
    if 28 <= ded_pct <= 38:
        return 1.0
    dist = min(abs(ded_pct - 28), abs(ded_pct - 38))
    return clamp(1.0 - (dist / 15) ** 2, 0.2)

def default_score(band: str) -> float:
    return {"low": 1.0, "moderate": 0.7, "high": 0.3}.get(band, 0.5)

def lob_score(lob: str) -> float:
    return 0.6 if lob in HI_RISK_LOBS else 1.0

def rps(gap_pct: float,
        br_diff_pct: float,
        ded_pct: float,
        default_band: str,
        lob: str) -> float:
    """Return 0-100 Reinsurance Placement Score."""
    s = (
        W_PRICE * price_score(gap_pct) +
        W_BRK   * brokerage_score(br_diff_pct) +
        W_DED   * deduction_score(ded_pct) +
        W_DEF   * default_score(default_band) +
        W_LOB   * lob_score(lob)
    )
    return round(100 * s, 1)

def rps_band(score: float):
    if score >= 90:  return "A", "#008000", "Top-tier; place confidently."
    if score >= 75:  return "B", "#2b8a3e", "Attractive; place with minor tweaks."
    if score >= 60:  return "C", "orange",  "Quote possible but needs concessions."
    return            "D", "red",           "Outside appetite; expect decline."
