from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import aiohttp
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import re
from lxml import etree
import asyncio
from typing import List, Dict
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
USER_AGENT = "13F_Old (13Fnew@example.com)"
CACHE_FILE = "fund_managers_cache.json"

# Global variable for fund managers
FUND_MANAGERS = []

# Data fetching and parsing functions
async def fetch_filings(cik: str, session: aiohttp.ClientSession) -> List[Dict]:
    filings_url = BASE_URL.format(cik=str(cik).zfill(10))
    headers = {"User-Agent": USER_AGENT}
    async with session.get(filings_url, headers=headers) as response:
        if response.status != 200:
            logger.error(f"Failed to fetch filings for CIK {cik}: {response.status}")
            return []
        data = await response.json()
    
    filings = []
    recent = data.get("filings", {}).get("recent", {})
    for i in range(len(recent.get("accessionNumber", []))):
        if recent["form"][i] == "13F-HR":
            filing = {
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "period_of_report": recent["reportDate"][i] or recent["periodOfReport"][i],
            }
            filings.append(filing)
    logger.info(f"CIK {cik} - Retrieved {len(filings)} filings: {[f['period_of_report'] for f in filings]}")
    return filings[:5]  # Fetching 5 quarters

async def fetch_filing_details(cik: str, filing: Dict, session: aiohttp.ClientSession) -> List[Dict]:
    accession_number = filing["accession_number"]
    accession_number_formatted = accession_number.replace("-", "")
    period_of_report = filing["period_of_report"]
    filing_date = filing["filing_date"]

    base_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number_formatted}/"
    file_url = base_url + f"{accession_number}.txt"

    async with session.get(file_url, headers={"User-Agent": USER_AGENT}) as response:
        if response.status != 200:
            logger.error(f"Failed to fetch filing details for {accession_number}: {response.status}")
            index_url = base_url + f"{accession_number}-index.html"
            async with session.get(index_url, headers={"User-Agent": USER_AGENT}) as index_response:
                if index_response.status != 200:
                    logger.error(f"Index page also unavailable for {accession_number}: {index_response.status}")
                    return []
                index_content = await index_response.text()
                txt_match = re.search(r'href="/Archives/edgar/data/\d+/\d+/([^"]+\.txt)"', index_content)
                if txt_match:
                    file_url = f"https://www.sec.gov{txt_match.group(1)}"
                    async with session.get(file_url, headers={"User-Agent": USER_AGENT}) as retry_response:
                        if retry_response.status != 200:
                            logger.error(f"Retry fetch failed for {file_url}: {retry_response.status}")
                            return []
                        content = await retry_response.text()
                else:
                    logger.error(f"No .txt file link found in index for {accession_number}")
                    return []
        else:
            content = await response.text()

    name_match = re.search(r"COMPANY CONFORMED NAME:\s*(.*?)\n", content)
    filing_manager_name = name_match.group(1) if name_match else "Unknown"

    documents = content.split("<DOCUMENT>")
    for doc in documents:
        if not doc.strip():
            continue
        type_match = re.search(r"<TYPE>\s*(.*)", doc)
        if type_match and "INFORMATION TABLE" in type_match.group(1).strip().upper():
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL)
            if text_match:
                text_content = text_match.group(1)
                xml_match = re.search(r"<XML>(.*?)</XML>", text_content, re.DOTALL)
                if xml_match:
                    return parse_info_table_xml(
                        xml_match.group(1), accession_number, cik, filing_date,
                        period_of_report, filing_manager_name
                    )
                return parse_text_info_table(
                    text_content, accession_number, cik, filing_date,
                    period_of_report, filing_manager_name
                )
    logger.error(f"No valid INFORMATION TABLE found in {accession_number}")
    return []

def parse_info_table_xml(content: str, accession_number: str, cik: str, filing_date: str, period_of_report: str, filing_manager_name: str) -> List[Dict]:
    parser = etree.XMLParser(recover=True)
    try:
        root = etree.fromstring(content.encode(), parser=parser)
        info_tables = root.xpath('.//*[local-name()="infoTable"]')
        holdings = []
        for table in info_tables:
            holding = {
                "accession_number": accession_number,
                "cik": cik,
                "filing_date": filing_date,
                "period_of_report": period_of_report,
                "filing_manager_name": filing_manager_name,
                "name_of_issuer": table.xpath('.//*[local-name()="nameOfIssuer"]/text()')[0] or "",
                "cusip": table.xpath('.//*[local-name()="cusip"]/text()')[0] or "",
                "value": float(table.xpath('.//*[local-name()="value"]/text()')[0] or 0) * 1000,
                "sshprnamt": float(table.xpath('.//*[local-name()="shrsOrPrnAmt"]/*[local-name()="sshPrnamt"]/text()')[0] or 0),
            }
            holdings.append(holding)
        return holdings
    except Exception as e:
        logger.error(f"XML parsing error for {accession_number}: {e}")
        return []

def parse_text_info_table(text_content: str, accession_number: str, cik: str, filing_date: str, period_of_report: str, filing_manager_name: str) -> List[Dict]:
    data_list = []
    lines = text_content.splitlines()
    headers = None
    for i, line in enumerate(lines):
        if "NAME OF ISSUER" in line.upper():
            headers = re.findall(r"\b\w+\b", line)
            start_index = i + 2
            break
    if not headers:
        logger.error(f"No headers found in plain text table for {accession_number}")
        return []

    for line in lines[start_index:]:
        if not line.strip():
            continue
        fields = re.findall(r"\S+", line)
        if len(fields) >= len(headers):
            data = dict(zip(headers, fields))
            parsed_data = {
                "accession_number": accession_number,
                "cik": cik,
                "filing_date": filing_date,
                "period_of_report": period_of_report,
                "filing_manager_name": filing_manager_name,
                "name_of_issuer": data.get("NAME", ""),
                "cusip": data.get("CUSIP", ""),
                "value": float(data.get("VALUE", 0)) * 1000,
                "sshprnamt": float(data.get("SHARES", 0)),
            }
            data_list.append(parsed_data)
    return data_list

async def get_holdings(cik: str) -> List[Dict]:
    async with aiohttp.ClientSession() as session:
        filings = await fetch_filings(cik, session)
        tasks = [fetch_filing_details(cik, filing, session) for filing in filings]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

def analyze_holdings(data: List[Dict]) -> Dict:
    df = pd.DataFrame(data)
    if df.empty:
        return {"top_holdings": [], "changes": [], "all_holdings": [], "manager_name": "Unknown"}
    
    df["value_fixed"] = np.where(
        (df["sshprnamt"] == 0) | ((df["value"] * 1000 / df["sshprnamt"]) > 1000),
        df["value"] / 1000,
        df["value"],
    )
    
    df["period_of_report"] = pd.to_datetime(df["period_of_report"])
    df["quarter"] = df["period_of_report"].dt.to_period("Q").astype(str)
    
    all_quarters = sorted(df["quarter"].unique())
    logger.info(f"All fetched quarters: {all_quarters}")
    
    # Ensure Q1 2024 is included if available
    target_quarters = ['2024Q1']  # Start with Q1 2024
    recent_quarters = sorted(df["quarter"].unique(), reverse=True)
    for q in recent_quarters:
        if q not in target_quarters and len(target_quarters) < 4:
            target_quarters.append(q)
    target_quarters = sorted(target_quarters)  # Sort for consistency
    logger.info(f"Selected quarters including Q1 2024: {target_quarters}")
    df = df[df["quarter"].isin(target_quarters)]
    
    latest_quarter = df["quarter"].max()
    earliest_quarter = df["quarter"].min()
    top_holdings = df[df["quarter"] == latest_quarter].groupby("name_of_issuer").agg({
        "value_fixed": "sum",
        "sshprnamt": "sum"
    }).nlargest(5, "value_fixed").reset_index()
    
    grouped = df.groupby(["name_of_issuer", "quarter"]).agg({
        "value_fixed": "sum",
        "sshprnamt": "sum"
    }).reset_index()
    grouped["prev_value"] = grouped.groupby("name_of_issuer")["value_fixed"].shift(1)
    grouped["value_change"] = np.abs(grouped["value_fixed"] - grouped["prev_value"].fillna(0))
    grouped["prev_sshprnamt"] = grouped.groupby("name_of_issuer")["sshprnamt"].shift(1)
    grouped["sshprnamt_change"] = np.abs(grouped["sshprnamt"] - grouped["prev_sshprnamt"].fillna(0))

    all_holdings = df.groupby(["name_of_issuer", "quarter"]).agg({
        "value_fixed": "sum",
        "sshprnamt": "sum",
        "accession_number": "first"  # Include accession_number for linking to filings
    }).reset_index()
    all_holdings["prev_value"] = all_holdings.groupby("name_of_issuer")["value_fixed"].shift(1)
    all_holdings["value_change"] = all_holdings["value_fixed"] - all_holdings["prev_value"]
    all_holdings["prev_sshprnamt"] = all_holdings.groupby("name_of_issuer")["sshprnamt"].shift(1)
    all_holdings["sshprnamt_change"] = all_holdings["sshprnamt"] - all_holdings["prev_sshprnamt"]

    # Get the last two quarters for "New Position" status
    unique_quarters = sorted(all_holdings["quarter"].unique())
    last_two_quarters = unique_quarters[-2:] if len(unique_quarters) >= 2 else unique_quarters

    all_holdings["status"] = all_holdings.apply(
        lambda row: "New Position" if (pd.isna(row["prev_value"]) and row["quarter"] in last_two_quarters) else "", 
        axis=1
    )

    for col in ["value_fixed", "sshprnamt"]:
        if col in top_holdings.columns:
            top_holdings[col] = top_holdings[col].replace([np.inf, -np.inf, np.nan], None)
    for col in ["value_fixed", "sshprnamt", "value_change", "sshprnamt_change", "prev_value", "prev_sshprnamt"]:
        if col in all_holdings.columns:
            all_holdings[col] = all_holdings[col].replace([np.inf, -np.inf, np.nan], None)
    for col in ["value_fixed", "sshprnamt", "prev_value", "value_change", "prev_sshprnamt", "sshprnamt_change"]:
        if col in grouped.columns:
            grouped[col] = grouped[col].replace([np.inf, -np.inf, np.nan], None)
    df = df.replace([np.inf, -np.inf, np.nan], None)

    return {
        "top_holdings": top_holdings.to_dict(orient="records"),
        "changes": grouped.to_dict(orient="records"),
        "all_holdings": all_holdings.to_dict(orient="records"),
        "manager_name": df["filing_manager_name"].iloc[0].title() if not df.empty else "Unknown"
    }

async def load_fund_manager_data() -> List[Dict]:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            if cached_data:
                logger.info("Loaded fund managers from cache")
                return cached_data
        except Exception as e:
            logger.error(f"Failed to load cache from {CACHE_FILE}: {e}")

    urls = [
        "https://www.sec.gov/Archives/edgar/full-index/2025/QTR1/company.idx",
        "https://www.sec.gov/Archives/edgar/full-index/2024/QTR4/company.idx"
    ]
    headers = {"User-Agent": USER_AGENT}
    data = []

    async with aiohttp.ClientSession() as session:
        for url in urls:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Failed to fetch {url}: {response.status}")
                        continue
                    content = await response.text(encoding="utf-8")
                    lines = content.splitlines()
                    
                    form_types = ["13F-HR", "13F-HR/A", "13F-NT", "13F-NT/A", "13F-CTR", "13F-CTR/A"]
                    for line in lines[11:]:
                        parts = re.split(r'\s+', line.strip())
                        if len(parts) >= 5 and parts[-4] in form_types:
                            # Clean and capitalize the name
                            raw_name = " ".join(parts[:-4])
                            # Expanded regex to remove common suffixes
                            cleaned_name = re.sub(r'\b(INC|Shares|LLC|Corp|Co|Insu|Insurance|Homestate)\b', '', raw_name, flags=re.IGNORECASE).strip()
                            # Apply title case after cleaning
                            titled_name = cleaned_name.title()
                            data.append({"cik": parts[-3], "name": titled_name})
            except Exception as e:
                logger.error(f"Error fetching or processing {url}: {e}")
                continue

    if not data:
        logger.error("No fund manager data retrieved from any URL")
        return []

    # Remove duplicates based on the cleaned and titled name, keeping the first occurrence
    df = pd.DataFrame(data).drop_duplicates(subset="name", keep="first")
    result = df.to_dict("records")
    
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(result, f)
        logger.info("Cached fund managers to file")
    except Exception as e:
        logger.error(f"Failed to cache fund managers: {e}")
    
    return result

@app.on_event("startup")
async def startup_event():
    global FUND_MANAGERS
    FUND_MANAGERS = await load_fund_manager_data()
    logger.info(f"Initialized FUND_MANAGERS with {len(FUND_MANAGERS)} entries")

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEC 13F Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root[data-theme="light"] {
            --bg-color: #f4f7fa;
            --text-color: #2d3748;
            --card-bg: #ffffff;
            --sidebar-bg: linear-gradient(180deg, #ffffff, #f8fafc);
            --border-color: #e2e8f0;
            --hover-bg: #edf2f7;
            --table-header-bg: #edf2f7;
            --new-position-bg: #e6fffa;
            --shadow-color: rgba(0, 0, 0, 0.05);
            --btn-primary-bg: #4a90e2;
            --btn-primary-hover: #357abd;
            --btn-success-bg: #38a169;
            --btn-success-hover: #2f855a;
            --placeholder-color: #6b7280;
            --pagination-bg: #ffffff;
            --pagination-active-bg: #4a90e2;
            --pagination-active-text: #ffffff;
            --pagination-disabled-text: #6b7280;
            --toggle-bg: #e2e8f0;
            --toggle-text-color: #f59e0b; /* Sun emoji color */
            --new-position-bg: #e6fffa; /* Light teal background for pill */
            --new-position-text: #2b6cb0; /* Darker teal text for contrast */
            --new-position-border: #b2f5ea;
        }

        :root[data-theme="dark"] {
            --bg-color: #1a1a1a;
            --text-color: #e2e8f0;
            --card-bg: #2d2d2d;
            --sidebar-bg: linear-gradient(180deg, #2d2d2d, #262626);
            --border-color: #404040;
            --hover-bg: #404040;
            --table-header-bg: #333333;
            --new-position-bg: #1a4a3c;
            --shadow-color: rgba(0, 0, 0, 0.2);
            --btn-primary-bg: #2b6cb0; /* Darker blue */
            --btn-primary-hover: #4a90e2; /* Original blue */
            --btn-success-bg: #38a169;
            --btn-success-hover: #48b17a;
            --placeholder-color: #9ca3af;
            --pagination-bg: #2d2d2d;
            --pagination-active-bg: #4a90e2;
            --pagination-active-text: #ffffff;
            --pagination-disabled-text: #6b7280;
            --toggle-bg: #404040;
            --toggle-text-color: #60a5fa; /* Moon emoji color */
            --new-position-bg: #2c7a7b; /* Darker teal for dark mode */
            --new-position-text: #e6fffa; /* Light teal text for contrast */
            --new-position-border: #4a9a9b;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: background-color 0.3s ease, color 0.3s ease;
        }

        .sidebar {
            height: 100vh;
            position: fixed;
            top: 0;
            left: 0;
            width: 280px;
            padding: 30px 20px;
            background: var(--sidebar-bg);
            box-shadow: 2px 0 15px var(--shadow-color);
            transition: transform 0.3s ease, background 0.3s ease;
            z-index: 1000; /* Base z-index */
        }


        .sidebar-overlay {
            display: none; /* Hidden by default */
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5); /* Semi-transparent black */
            z-index: 1250; /* Below active sidebar (1300), above toggle (1200) */
            transition: opacity 0.3s ease;
        }

        .sidebar-overlay.active {
            display: block;
            opacity: 1;
        }

        .sidebar.active {
            transform: translateX(0);
            z-index: 1300; /* Above everything when active */
        }

        .toggle-btn {
            display: none; /* Hidden by default on desktop */
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 1200;
            font-size: 24px; /* Icon size */
            padding: 8px 12px 12px 12px; /* Top: 8px, Right: 12px, Bottom: 12px, Left: 12px */
            background-color: var(--btn-primary-bg);
            border: none;
            border-radius: 8px;
            color: #ffffff;
            transition: background-color 0.3s ease;
        }

        .toggle-btn:hover {
            background-color: var(--btn-primary-hover); /* Use theme variable for hover */
        }

        #manager-name {
            margin-top: 0; /* Remove any default top margin that might push it up */
        }

        .sidebar h2 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 20px;
        }

        .main-content {
            margin-left: 280px;
            padding: 40px;
            transition: margin-left 0.3s ease;
        }

        .card {
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 20px var(--shadow-color);
            background: var(--card-bg);
            padding: 20px;
            transition: transform 0.2s ease, background-color 0.3s ease;
            color: var(--text-color);
            position: relative; /* Helps contain children */
        }

        .card:hover {
            transform: translateY(-5px);
        }

        /* Specific styles for chart containers */
        .chart-container {
            position: relative; /* Allows absolute positioning of canvas if needed */
            height: 400px; /* Fixed height for desktop */
            width: 100%; /* Full width of parent */
        }

        /* Style both canvases directly */
        #top-holdings-chart, #changes-chart {
            width: 100% !important; /* Override Chart.js inline styles */
            height: 100% !important; /* Fill container height */
        }

        .form-control {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 10px;
            background-color: var(--card-bg);
            color: var(--text-color);
            transition: border-color 0.2s ease, background-color 0.3s ease, color 0.3s ease;
        }

        .form-control:focus {
            border-color: var(--btn-primary-bg);
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
            background-color: var(--card-bg);
            color: var(--text-color);
        }

        .form-control::placeholder {
            color: var(--placeholder-color);
            opacity: 1;
        }

        .form-select {
            border-radius: 8px;
            border: 1px solid var(--border-color);
            padding: 10px;
            background-color: var(--card-bg);
            color: var(--text-color);
            transition: border-color 0.2s ease, background-color 0.3s ease, color 0.3s ease;
            background-image: linear-gradient(45deg, transparent 50%, var(--text-color) 50%),
                              linear-gradient(135deg, var(--text-color) 50%, transparent 50%);
            background-position: calc(100% - 20px) calc(1em + 2px),
                                 calc(100% - 15px) calc(1em + 2px);
            background-size: 5px 5px, 5px 5px;
            background-repeat: no-repeat;
        }

        .form-select:focus {
            border-color: var(--btn-primary-bg);
            box-shadow: 0 0 0 3px rgba(74, 144, 226, 0.1);
            background-color: var(--card-bg);
            color: var(--text-color);
            outline: none;
        }

        .form-select option {
            background-color: var(--card-bg);
            color: var(--text-color);
        }

        .btn-primary {
            background-color: var(--btn-primary-bg);
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.2s ease;
        }

        .btn-primary:hover {
            background-color: var(--btn-primary-hover);
        }

        .btn-success {
            background-color: var(--btn-success-bg);
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            transition: background-color 0.2s ease;
        }

        .btn-success:hover {
            background-color: var(--btn-success-hover);
        }

        #suggestions, #company-suggestions {
            position: absolute;
            z-index: 1000;
            width: 100%;
            max-height: 200px;
            overflow-y: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px var(--shadow-color);
            background: var(--card-bg);
        }

        .dropdown-item {
            padding: 10px 15px;
            color: var(--text-color);
            transition: background-color 0.2s ease;
        }

        .dropdown-item:hover {
            background-color: var(--hover-bg);
        }

        .table {
            border-radius: 8px;
            overflow: hidden;
            color: var(--text-color);
        }

        .table th {
            background-color: var(--table-header-bg);
            color: var(--text-color);
            font-weight: 600;
        }

        .table td {
            vertical-align: middle;
            background-color: var(--card-bg);
            color: var(--text-color);
        }

        .new-position {
            background-color: var(--card-bg); /* Keep row background as card-bg */
        }
        .status-pill {
            display: inline-block;
            background-color: var(--new-position-bg);
            color: var(--new-position-text);
            padding: 4px 12px;
            border-radius: 16px; /* Rounded corners for pill shape */
            font-weight: 500;
            font-size: 0.9em;
            border: 1px solid var(--new-position-border);
            margin-right: 8px; /* Space between pill and link */
        }

        #loading-indicator {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: rgba(0, 0, 0, 0.8);
            color: #ffffff;
            padding: 15px 25px;
            border-radius: 8px;
            z-index: 2000;
            font-weight: 600;
        }

        .sortable {
            cursor: pointer;
            user-select: none;
            position: relative;
        }

        .sortable:hover {
            background-color: var(--hover-bg);
        }

        .sortable::after {
            content: "↕";
            font-size: 0.8em;
            margin-left: 5px;
            opacity: 0.5;
        }

        .pagination {
            margin-top: 20px;
        }

        .page-item .page-link {
            background-color: var(--pagination-bg);
            color: var(--text-color);
            border: 1px solid var(--border-color);
            transition: background-color 0.2s ease, color 0.2s ease;
        }

        .page-item.active .page-link {
            background-color: var(--pagination-active-bg);
            color: var(--pagination-active-text);
            border-color: var(--pagination-active-bg);
        }

        .page-item:not(.active) .page-link:hover {
            background-color: var(--hover-bg);
            color: var(--text-color);
        }

        .page-item.disabled .page-link {
            background-color: var(--pagination-bg);
            color: var(--pagination-disabled-text);
            border-color: var(--border-color);
            pointer-events: none;
        }

        /* Moon/Sun Emoji Toggle (xAI-inspired) */
        .theme-toggle {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1100;
            background: var(--toggle-bg);
            border: 1px solid var(--border-color);
            border-radius: 50%;
            width: 44px;
            height: 44px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.3s ease, box-shadow 0.3s ease;
            font-size: 24px; /* Larger emoji */
            color: var(--toggle-text-color);
            box-shadow: 0 2px 6px var(--shadow-color);
        }

        .theme-toggle:hover {
            background: var(--hover-bg);
            transform: scale(1.1);
            box-shadow: 0 4px 12px var(--shadow-color);
        }

        .theme-toggle:active {
            transform: scale(0.95);
        }

@media (max-width: 768px) {
    .sidebar {
        transform: translateX(-280px);
    }
    .sidebar.active {
        transform: translateX(0);
        z-index: 1300;
    }
    .main-content {
        margin-left: 0;
        padding: 70px 20px 20px 20px; /* Increased from 60px to 70px */
    }
    .toggle-btn {
        display: block;
    }
    .theme-toggle {
        top: 60px;
        right: 15px;
    }
    .sidebar-overlay {
        display: none;
    }
    .sidebar-overlay.active {
        display: block;
    }
    .card {
        padding: 15px;
    }
    .chart-container {
        height: 300px; /* Smaller fixed height for mobile */
    }
}
    </style>
</head>
<body>
    <div id="loading-indicator">Loading...</div>
    <button id="mobile-toggle" class="btn btn-primary toggle-btn" onclick="toggleSidebar()">☰</button>
    <div id="sidebar-overlay" class="sidebar-overlay" onclick="toggleSidebar()"></div> <!-- New overlay -->
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle Theme">🌙</div>
    <div class="sidebar">
        <h2>SEC 13F Dashboard</h2>
        <div class="mb-4 position-relative">
            <input type="text" id="fund-manager-search" class="form-control" placeholder="Search fund manager...">
            <div id="suggestions" class="dropdown-menu"></div>
        </div>
        <button class="btn btn-primary w-100 mb-3" onclick="toggleFilters()">Toggle Filters</button>
        <div id="filter-section" class="filter-section card" style="display: none;">
            <h4 class="mb-3">Filters</h4>
            <div class="mb-3 position-relative">
                <label for="company-filter" class="form-label">Company Name</label>
                <input type="text" id="company-filter" class="form-control" placeholder="e.g., AMAZON">
                <div id="company-suggestions" class="dropdown-menu"></div>
            </div>
            <div class="mb-3">
                <label for="start-quarter" class="form-label">Start Quarter</label>
                <select id="start-quarter" class="form-select">
                    <option value="">Select Start Quarter</option>
                </select>
            </div>
            <div class="mb-3">
                <label for="end-quarter" class="form-label">End Quarter</label>
                <select id="end-quarter" class="form-select">
                    <option value="">Select End Quarter</option>
                </select>
            </div>
            <button class="btn btn-success w-100" onclick="applyFilters()">Apply Filters</button>
        </div>
    </div>
    <div class="main-content">
        <h2 id="manager-name" class="mb-4">Select a Fund Manager</h2>
        <div class="row g-4">
            <div class="col-md-6">
                <div class="card">
                    <h3 class="mb-3">Top 10 Holdings</h3>
                    <div class="chart-container">
                        <canvas id="top-holdings-chart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <h3 class="mb-3">Quarterly Changes</h3>
                    <div class="chart-container">
                        <canvas id="changes-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <h3 class="mb-3">All Holdings</h3>
                    <div class="table-responsive">
                        <table class="table table-striped" id="holdings-table">
                            <thead>
                                <tr>
                                    <th class="sortable" data-column="name_of_issuer" data-direction="asc" onclick="sortTable('name_of_issuer')">Issuer</th>
                                    <th class="sortable" data-column="value_fixed" data-direction="asc" onclick="sortTable('value_fixed')">Value ($)</th>
                                    <th class="sortable" data-column="sshprnamt" data-direction="asc" onclick="sortTable('sshprnamt')">Shares</th>
                                    <th class="sortable" data-column="value_change" data-direction="asc" onclick="sortTable('value_change')">Value Change ($)</th>
                                    <th class="sortable" data-column="sshprnamt_change" data-direction="asc" onclick="sortTable('sshprnamt_change')">Share Change</th>
                                    <th class="sortable" data-column="quarter" data-direction="asc" onclick="sortTable('quarter')">Quarter</th>
                                    <th class="sortable" data-column="status" data-direction="asc" onclick="sortTable('status')">Status</th>
                                </tr>
                            </thead>
                            <tbody id="holdings-tbody"></tbody>
                        </table>
                    </div>
                    <nav>
                        <ul class="pagination justify-content-center mt-3" id="pagination"></ul>
                    </nav>
                </div>
            </div>
        </div>
    </div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<script>
    // Load theme from localStorage on page load
    document.addEventListener('DOMContentLoaded', () => {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        updateThemeIcon();
        updateChartTheme(savedTheme);
    });

    function toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateThemeIcon();
        updateChartTheme(newTheme);
    }

    function updateThemeIcon() {
        const toggle = document.querySelector('.theme-toggle');
        const theme = document.documentElement.getAttribute('data-theme');
        toggle.textContent = theme === 'dark' ? '☀️' : '🌙';
    }

    function updateChartTheme(theme) {
        const chartOptions = {
            light: {
                scales: {
                    y: { 
                        grid: { color: 'rgba(0, 0, 0, 0.1)' },
                        ticks: { color: '#2d3748' }
                    },
                    x: { 
                        grid: { color: 'rgba(0, 0, 0, 0.1)' },
                        ticks: { color: '#2d3748' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#2d3748' }
                    },
                    title: { color: '#2d3748' }
                }
            },
            dark: {
                scales: {
                    y: { 
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#e2e8f0' }
                    },
                    x: { 
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#e2e8f0' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#e2e8f0' }
                    },
                    title: { color: '#e2e8f0' }
                }
            }
        };

        if (topHoldingsChart) {
            topHoldingsChart.options = { ...topHoldingsChart.options, ...chartOptions[theme] };
            topHoldingsChart.update();
        }
        if (changesChart) {
            changesChart.options = { ...changesChart.options, ...chartOptions[theme] };
            changesChart.update();
        }
    }

    function toggleSidebar() {
        const sidebar = document.querySelector('.sidebar');
        const toggleBtn = document.getElementById('mobile-toggle');
        const overlay = document.getElementById('sidebar-overlay');
        
        sidebar.classList.toggle('active');
        overlay.classList.toggle('active');
        
        // Update button icon based on sidebar state
        if (sidebar.classList.contains('active')) {
            toggleBtn.textContent = '✕'; // Close icon when sidebar is open
        } else {
            toggleBtn.textContent = '☰'; // Menu icon when sidebar is closed
        }
    }

    let selectedCik = null;
    const itemsPerPage = 10;
    let currentPage = 1;
    let allHoldings = [];
    let filteredHoldings = [];
    let topHoldingsChart = null;
    let changesChart = null;
    let availableCompanies = [];
    let availableQuarters = [];

    const issuerColorMap = {};

    function getIssuerColor(issuer) {
        if (!issuerColorMap[issuer]) {
            let hash = 0;
            for (let i = 0; i < issuer.length; i++) {
                hash = issuer.charCodeAt(i) + ((hash << 5) - hash);
            }
            const hue = Math.abs(hash) % 360;
            issuerColorMap[issuer] = `hsl(${hue}, 50%, 50%)`;
        }
        return issuerColorMap[issuer];
    }

    function showLoading() {
        document.getElementById('loading-indicator').style.display = 'block';
    }

    function hideLoading() {
        document.getElementById('loading-indicator').style.display = 'none';
    }

    document.getElementById('fund-manager-search').addEventListener('input', async function() {
        const term = this.value;
        if (term.length < 2) {
            document.getElementById('suggestions').style.display = 'none';
            return;
        }
        showLoading();
        try {
            const response = await fetch(`/api/suggestions?term=${term}`);
            const data = await response.json();
            hideLoading();
            const suggestions = document.getElementById('suggestions');
            suggestions.innerHTML = '';
            data.results.forEach(fm => {
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                item.textContent = fm.name;
                item.onclick = () => {
                    this.value = fm.name;
                    selectedCik = fm.cik;
                    suggestions.style.display = 'none';
                    loadData();
                };
                suggestions.appendChild(item);
            });
            suggestions.style.display = 'block';
        } catch (error) {
            hideLoading();
            console.error('Error fetching suggestions:', error);
        }
    });

    document.getElementById('company-filter').addEventListener('input', function() {
        const term = this.value.toLowerCase();
        if (term.length < 2) {
            document.getElementById('company-suggestions').style.display = 'none';
            return;
        }
        const suggestions = document.getElementById('company-suggestions');
        suggestions.innerHTML = '';
        availableCompanies
            .filter(company => company.toLowerCase().includes(term))
            .slice(0, 10)
            .forEach(company => {
                const item = document.createElement('div');
                item.className = 'dropdown-item';
                item.textContent = company;
                item.onclick = () => {
                    this.value = company;
                    suggestions.style.display = 'none';
                    applyFilters();
                };
                suggestions.appendChild(item);
            });
        suggestions.style.display = 'block';
    });

    async function loadData() {
        if (!selectedCik) return;
        showLoading();
        try {
            const response = await fetch(`/api/data/${selectedCik}`);
            const data = await response.json();
            hideLoading();
            
            document.getElementById('manager-name').textContent = data.manager_name;

            allHoldings = data.all_holdings;
            filteredHoldings = [...allHoldings];

            availableQuarters = [...new Set(allHoldings.map(h => h.quarter))].sort();
            const startSelect = document.getElementById('start-quarter');
            const endSelect = document.getElementById('end-quarter');
            startSelect.innerHTML = '<option value="">Select Start Quarter</option>';
            endSelect.innerHTML = '<option value="">Select End Quarter</option>';
            availableQuarters.forEach(quarter => {
                const optionStart = document.createElement('option');
                optionStart.value = quarter;
                optionStart.textContent = quarter;
                startSelect.appendChild(optionStart);
                const optionEnd = document.createElement('option');
                optionEnd.value = quarter;
                optionEnd.textContent = quarter;
                endSelect.appendChild(optionEnd);
            });

            availableCompanies = [...new Set(allHoldings.map(h => h.name_of_issuer))].sort();

            renderAllVisualizations();
            updateChartTheme(document.documentElement.getAttribute('data-theme') || 'light');
        } catch (error) {
            hideLoading();
            console.error('Error loading data:', error);
        }
    }

    function toggleFilters() {
        const filterSection = document.getElementById('filter-section');
        filterSection.style.display = filterSection.style.display === 'block' ? 'none' : 'block';
    }

    function applyFilters() {
        const companyFilter = document.getElementById('company-filter').value.toLowerCase();
        const startQuarter = document.getElementById('start-quarter').value;
        const endQuarter = document.getElementById('end-quarter').value;

        filteredHoldings = allHoldings.filter(holding => {
            const matchesCompany = !companyFilter || holding.name_of_issuer.toLowerCase().includes(companyFilter);
            const matchesStart = !startQuarter || holding.quarter >= startQuarter;
            const matchesEnd = !endQuarter || holding.quarter <= endQuarter;
            return matchesCompany && matchesStart && matchesEnd;
        });

        renderAllVisualizations();
    }

    function renderAllVisualizations() {
        renderTopHoldingsChart();
        renderChangesChart();
        renderTable(1);
    }

    function renderTopHoldingsChart() {
    if (topHoldingsChart) topHoldingsChart.destroy();

    if (!filteredHoldings.length) {
        const topCtx = document.getElementById('top-holdings-chart').getContext('2d');
        topHoldingsChart = new Chart(topCtx, {
            type: 'bar',
            data: {
                labels: ['No Data'],
                datasets: [{
                    label: 'Value ($)',
                    data: [0],
                    backgroundColor: 'rgba(54, 162, 235, 0.6)'
                }]
            },
            options: {
                scales: { y: { beginAtZero: true } },
                responsive: true, // Ensure responsiveness
                maintainAspectRatio: false // Allow stretching within container
            }
        });
        return;
    }

    const sortedQuarters = [...new Set(filteredHoldings.map(h => h.quarter))].sort();
    const latestQuarter = sortedQuarters[sortedQuarters.length - 1];

    const topHoldingsData = filteredHoldings
        .filter(h => h.quarter === latestQuarter)
        .reduce((acc, curr) => {
            const issuer = curr.name_of_issuer;
            if (!acc[issuer]) acc[issuer] = { value_fixed: 0, sshprnamt: 0 };
            acc[issuer].value_fixed += curr.value_fixed || 0;
            acc[issuer].sshprnamt += curr.sshprnamt || 0;
            return acc;
        }, {});
    
    const topHoldings = Object.entries(topHoldingsData)
        .map(([name_of_issuer, data]) => ({ name_of_issuer, ...data }))
        .sort((a, b) => b.value_fixed - a.value_fixed)
        .slice(0, 10);

    const topCtx = document.getElementById('top-holdings-chart').getContext('2d');
    topHoldingsChart = new Chart(topCtx, {
        type: 'bar',
        data: {
            labels: topHoldings.length ? topHoldings.map(h => h.name_of_issuer) : ['No Data'],
            datasets: [{
                label: 'Value ($)',
                data: topHoldings.length ? topHoldings.map(h => h.value_fixed || 0) : [0],
                backgroundColor: 'rgba(54, 162, 235, 0.6)'
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Value ($)' } }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top', // Match Quarterly Changes
                    labels: {
                        font: { size: 10 },
                        padding: 5,
                        boxWidth: 20,
                        usePointStyle: true
                    },
                    maxHeight: 50
                },
                title: {
                    display: true,
                    text: 'Top 10 Holdings',
                    font: { size: 14 }
                }
            },
            responsive: true, // Ensure chart adjusts to container
            maintainAspectRatio: false // Allow stretching within container
        }
    });
    updateChartTheme(document.documentElement.getAttribute('data-theme') || 'light');
}

    function renderChangesChart() {
    if (changesChart) changesChart.destroy();

    if (!filteredHoldings.length) {
        const changesCtx = document.getElementById('changes-chart').getContext('2d');
        changesChart = new Chart(changesCtx, {
            type: 'bar',
            data: {
                labels: ['No Data'],
                datasets: [{
                    label: 'Shares',
                    data: [0],
                    backgroundColor: 'rgba(255, 99, 132, 0.6)'
                }]
            },
            options: { scales: { y: { beginAtZero: true } } }
        });
        return;
    }

    const quarters = [...new Set(filteredHoldings.map(h => h.quarter))].sort();

    const groupedShares = filteredHoldings.reduce((acc, curr) => {
        const key = `${curr.name_of_issuer}-${curr.quarter}`;
        if (!acc[key]) {
            acc[key] = { name_of_issuer: curr.name_of_issuer, quarter: curr.quarter, sshprnamt: 0 };
        }
        acc[key].sshprnamt += curr.sshprnamt || 0;
        return acc;
    }, {});

    const sharesData = Object.values(groupedShares);

    const issuerTotals = filteredHoldings.reduce((acc, curr) => {
        acc[curr.name_of_issuer] = (acc[curr.name_of_issuer] || 0) + (curr.sshprnamt || 0);
        return acc;
    }, {});
    const topIssuers = Object.entries(issuerTotals)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(entry => entry[0]);

    const datasets = topIssuers.map(issuer => {
        const data = quarters.map(quarter => {
            const entry = sharesData.find(d => d.name_of_issuer === issuer && d.quarter === quarter);
            return entry ? (entry.sshprnamt || 0) : 0;
        });
        return {
            label: issuer,
            data: data,
            backgroundColor: getIssuerColor(issuer)
        };
    });

    const changesCtx = document.getElementById('changes-chart').getContext('2d');
    changesChart = new Chart(changesCtx, {
        type: 'bar',
        data: {
            labels: quarters,
            datasets: datasets.length ? datasets : [{ label: 'Shares', data: [0], backgroundColor: 'rgba(255, 99, 132, 0.6)' }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, title: { display: true, text: 'Shares' } },
                x: { stacked: false }
            },
            indexAxis: 'x',
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        font: { size: 10 },
                        padding: 5,
                        boxWidth: 20,
                        usePointStyle: true
                    },
                    maxHeight: 50
                },
                title: {
                    display: true,
                    text: 'Shares Held by Top Issuers',
                    font: { size: 14 }
                }
            },
            responsive: true, // Chart adjusts to container size
            maintainAspectRatio: false // Allow stretching within container
        }
    });
    updateChartTheme(document.documentElement.getAttribute('data-theme') || 'light');
}

    function sortTable(column) {
        const th = document.querySelector(`th[data-column="${column}"]`);
        const currentDirection = th.getAttribute("data-direction");
        const newDirection = currentDirection === "asc" ? "desc" : "asc";
        th.setAttribute("data-direction", newDirection);

        document.querySelectorAll(".sortable").forEach(header => {
            if (header !== th) header.setAttribute("data-direction", "asc");
        });

        filteredHoldings.sort((a, b) => {
            let valA = a[column];
            let valB = b[column];

            if (valA === null || valA === undefined) valA = column.includes("change") || column === "value_fixed" || column === "sshprnamt" ? -Infinity : "";
            if (valB === null || valB === undefined) valB = column.includes("change") || column === "value_fixed" || column === "sshprnamt" ? -Infinity : "";

            if (column === "value_fixed" || column === "sshprnamt" || column === "value_change" || column === "sshprnamt_change") {
                return newDirection === "asc" ? valA - valB : valB - valA;
            }

            valA = valA.toString().toLowerCase();
            valB = valB.toString().toLowerCase();
            return newDirection === "asc" 
                ? valA.localeCompare(valB) 
                : valB.localeCompare(valA);
        });

        renderAllVisualizations();
    }

    function renderTable(page) {
        currentPage = page;
        const tbody = document.getElementById('holdings-tbody');
        tbody.innerHTML = '';
        const start = (page - 1) * itemsPerPage;
        const end = start + itemsPerPage;
        filteredHoldings.slice(start, end).forEach(holding => {
            const tr = document.createElement('tr');
            tr.className = holding.status === "New Position" ? 'new-position' : '';
            const valueChangeStyle = holding.value_change < 0 ? 'style="color: red"' : '';
            const shareChangeStyle = holding.sshprnamt_change < 0 ? 'style="color: red"' : '';
            const accessionNumber = holding.accession_number; 
            const accessionNumberNoHyphens = accessionNumber.replace(/-/g, '');
            const filingUrl = `https://www.sec.gov/Archives/edgar/data/${selectedCik}/${accessionNumberNoHyphens}/${accessionNumber}-index.html`;
            
            // Style "New Position" as a pill with the link beside it
            const statusContent = holding.status 
                ? `<span class="status-pill">${holding.status}</span><a href="${filingUrl}" target="_blank" title="View Filing">🔗</a>`
                : `<span style="display: block; text-align: right;"><a href="${filingUrl}" target="_blank" title="View Filing">🔗</a></span>`;
            
            tr.innerHTML = `
                <td>${holding.name_of_issuer}</td>
                <td>${holding.value_fixed === null ? 'N/A' : holding.value_fixed.toLocaleString()}</td>
                <td>${holding.sshprnamt === null ? 'N/A' : holding.sshprnamt.toLocaleString()}</td>
                <td ${valueChangeStyle}>${holding.value_change === null ? 'N/A' : holding.value_change.toLocaleString()}</td>
                <td ${shareChangeStyle}>${holding.sshprnamt_change === null ? 'N/A' : holding.sshprnamt_change.toLocaleString()}</td>
                <td>${holding.quarter}</td>
                <td>${statusContent}</td>
            `;
            tbody.appendChild(tr);
        });
        renderPagination();
    }

    function renderPagination() {
        const totalPages = Math.ceil(filteredHoldings.length / itemsPerPage);
        const pagination = document.getElementById('pagination');
        pagination.innerHTML = '';
        
        addPageItem(1, 1 === currentPage);
        if (currentPage > 3) addEllipsis();
        const startPage = Math.max(2, currentPage - 2);
        const endPage = Math.min(totalPages - 1, currentPage + 2);
        for (let i = startPage; i <= endPage; i++) {
            addPageItem(i, i === currentPage);
        }
        if (currentPage < totalPages - 2) addEllipsis();
        if (totalPages > 1) addPageItem(totalPages, totalPages === currentPage);

        function addPageItem(page, isActive) {
            const li = document.createElement('li');
            li.className = `page-item ${isActive ? 'active' : ''}`;
            li.innerHTML = `<a class="page-link" href="#" onclick="renderTable(${page}); return false;">${page}</a>`;
            pagination.appendChild(li);
        }

        function addEllipsis() {
            const li = document.createElement('li');
            li.className = 'page-item disabled';
            li.innerHTML = '<span class="page-link">...</span>';
            pagination.appendChild(li);
        }
    }
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return INDEX_HTML

@app.get("/api/suggestions")
async def get_suggestions(term: str):
    if not FUND_MANAGERS:
        logger.error("FUND_MANAGERS is empty in get_suggestions")
        raise HTTPException(status_code=500, detail="No fund manager data available")
    logger.info(f"Searching FUND_MANAGERS with {len(FUND_MANAGERS)} entries for term: {term}")
    results = [fm for fm in FUND_MANAGERS if term.lower() in fm["name"].lower()]
    return {"results": results[:10]}

@app.get("/api/data/{cik}")
async def get_data(cik: str):
    holdings = await get_holdings(cik)
    if not holdings:
        raise HTTPException(status_code=404, detail="No data found for this CIK")
    analysis = analyze_holdings(holdings)
    return analysis

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT not set
    uvicorn.run(app, host="0.0.0.0", port=port)