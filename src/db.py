import sqlite3
from datetime import datetime

DB_PATH = "database/submissions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            fac_sum_insured REAL,
            business_name TEXT,
            risk_occupation TEXT,
            currency TEXT,
            brokerage REAL,
            commission REAL,
            reinsured TEXT,
            premium_input REAL,
            pred_prem REAL,
            pred_rate REAL,
            prem_mae REAL,
            confidence_interval REAL,
            prem_range_low REAL,
            prem_range_high REAL,
            quoted_brokerage_fee REAL,
            pred_broker_fee REAL,
            pred_broker_rate REAL,
            broker_mae REAL,
            br_range_low REAL,
            br_range_high REAL
        )
    """)
    conn.commit()
    conn.close()

def log_submission(data: dict):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO submissions (
            timestamp, fac_sum_insured, business_name, risk_occupation, currency, brokerage, commission,
            reinsured, premium_input, pred_prem, pred_rate, prem_mae, confidence_interval, 
            prem_range_low, prem_range_high, quoted_brokerage_fee, pred_broker_fee, 
            pred_broker_rate, broker_mae, br_range_low, br_range_high
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        data.get("fac_sum_insured"),
        data.get("business_name"),
        data.get("risk_occupation"),
        data.get("currency"),
        data.get("brokerage"),
        data.get("commission"),
        data.get("reinsured"),
        data.get("premium_input"),
        data.get("pred_prem"),
        data.get("pred_rate"),
        data.get("prem_mae"),
        data.get("confidence_interval"),
        data.get("prem_range_low"),
        data.get("prem_range_high"),
        data.get("quoted_brokerage_fee"),
        data.get("pred_broker_fee"),
        data.get("pred_broker_rate"),
        data.get("broker_mae"),
        data.get("br_range_low"),
        data.get("br_range_high")
    ))
    conn.commit()
    conn.close()