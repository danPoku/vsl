import os
import sqlite3
from datetime import datetime
import gspread
import json
from google.oauth2.service_account import Credentials
import streamlit as st

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
            prem_confidence_interval REAL,
            prem_range_low REAL,
            prem_range_high REAL,
            quoted_brokerage_fee REAL,
            pred_broker_fee REAL,
            pred_broker_rate REAL,
            broker_mae REAL,
            broker_confidence_interval REAL,
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
            reinsured, premium_input, pred_prem, pred_rate, prem_mae, prem_confidence_interval, 
            prem_range_low, prem_range_high, quoted_brokerage_fee, pred_broker_fee, 
            pred_broker_rate, broker_mae, broker_confidence_interval, br_range_low, br_range_high
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        data.get("prem_confidence_interval"),
        data.get("prem_range_low"),
        data.get("prem_range_high"),
        data.get("quoted_brokerage_fee"),
        data.get("pred_broker_fee"),
        data.get("pred_broker_rate"),
        data.get("broker_mae"),
        data.get("broker_confidence_interval"),
        data.get("br_range_low"),
        data.get("br_range_high")
    ))
    conn.commit()
    conn.close()
    

# Google Service Account 
SERVICE_ACCOUNT_INFO = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT_INFO"])

# Google Sheet
SHEET_NAME = st.secrets["GOOGLE_SHEET_NAME"]

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

def get_sheet():
    creds = Credentials.from_service_account_info(SERVICE_ACCOUNT_INFO, scopes=scope)
    client = gspread.authorize(creds)
    sheet = client.open(SHEET_NAME).sheet1 
    return sheet

def log_submission_gsheets(data: dict):
    sheet = get_sheet()
    # Count non-empty rows (assuming first row is header)
    existing_records = len([row for row in sheet.get_all_values() if any(row)])
    next_id = existing_records
    row = [
        next_id,
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
        data.get("broker_confidence_interval"),
        data.get("br_range_low"),
        data.get("br_range_high"),
    ]
    sheet.append_row(row)