#!/usr/bin/env python3
"""
Slovak Election Poll Analysis - Static HTML Generator

Generates a static HTML site with interactive charts that can be
hosted on GitHub Pages or any static file server.

Usage:
    python generate_html.py --output-dir docs/
    python generate_html.py --output-dir docs/ --threshold 5 --forecast 6
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Import from trend_analysis
from trend_analysis import (
    load_all_polls,
    flatten_to_timeseries,
    flatten_mandates_to_timeseries,
    calculate_monthly_averages,
    calculate_monthly_seat_averages,
    calculate_kalman_averages,
    kalman_forecast,
    run_kalman_filter,
    calculate_party_correlations,
    calculate_volatility,
    get_current_estimates,
)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Party colors matching Slovak political colors
PARTY_COLORS = {
    "PS": "#6B2D5C",
    "SMER": "#E31E24",
    "HLAS": "#003DA5",
    "REP": "#1a1a1a",
    "SLOV": "#00A651",
    "SAS": "#FFD700",
    "KDH": "#0066CC",
    "DEM": "#FF6600",
    "SNS": "#003366",
    "Aliancia": "#009933",
    "ROD": "#8B0000",
}

# Political blocks
GOVERNMENT_PARTIES = ["SMER", "HLAS", "SNS", "REP"]
OPPOSITION_PARTIES = ["PS", "SAS", "DEM", "SLOV", "KDH"]
CATHOLIC_PARTIES = ["KDH", "DEM", "SLOV", "ROD"]


def get_raw_chart_data(records: List[Dict], threshold: float, forecast_months: int) -> Dict:
    """Generate raw (non-Kalman) data for Chart.js visualization."""
    monthly_series = calculate_monthly_averages(records)

    parties_data = {}
    for party, series in monthly_series.items():
        if not series:
            continue

        current = series[-1][1]
        dates = [d.strftime("%Y-%m-%d") for d, _ in series]
        values = [round(v, 2) for _, v in series]

        parties_data[party] = {
            "current": round(current, 2),
            "trend": 0,  # No trend calculation for raw data
            "aboveThreshold": current >= threshold,
            "color": PARTY_COLORS.get(party, "#666666"),
            "isGovernment": party in GOVERNMENT_PARTIES,
            "isOpposition": party in OPPOSITION_PARTIES,
            "isCatholic": party in CATHOLIC_PARTIES,
            "dates": dates,
            "values": values,
            "uncertainties": [0] * len(values),
            "forecast": {"dates": [], "values": [], "upper": [], "lower": []}
        }

    return parties_data


def get_chart_data(records: List[Dict], threshold: float, forecast_months: int) -> Dict:
    """Generate data for Chart.js visualization."""
    kalman_series = calculate_kalman_averages(records)

    # Get all dates
    all_dates = sorted(set(d for series in kalman_series.values() for d, _, _ in series))

    parties_data = {}
    for party, series in kalman_series.items():
        if not series:
            continue

        current = series[-1][1]

        # Store all parties but mark if above threshold
        dates = [d.strftime("%Y-%m-%d") for d, _, _ in series]
        values = [round(v, 2) for _, v, _ in series]
        uncertainties = [round(u, 2) for _, _, u in series]

        # Get forecast
        forecast = kalman_forecast(records, party, forecast_months)
        forecast_dates = []
        forecast_values = []
        forecast_upper = []
        forecast_lower = []

        if forecast:
            for d, v, u in forecast:
                forecast_dates.append(d.strftime("%Y-%m-%d"))
                forecast_values.append(round(v, 2))
                forecast_upper.append(round(v + 1.96 * u, 2))
                forecast_lower.append(round(v - 1.96 * u, 2))

        # Calculate trend
        kf, results = run_kalman_filter(records, party)
        velocity = 0
        if kf:
            _, velocity, _ = kf.get_state()

        parties_data[party] = {
            "current": round(current, 2),
            "trend": round(velocity, 3),
            "aboveThreshold": current >= threshold,
            "color": PARTY_COLORS.get(party, "#666666"),
            "isGovernment": party in GOVERNMENT_PARTIES,
            "isOpposition": party in OPPOSITION_PARTIES,
            "isCatholic": party in CATHOLIC_PARTIES,
            "dates": dates,
            "values": values,
            "uncertainties": uncertainties,
            "forecast": {
                "dates": forecast_dates,
                "values": forecast_values,
                "upper": forecast_upper,
                "lower": forecast_lower,
            }
        }

    return parties_data


def get_blocks_data(records: List[Dict], forecast_months: int) -> Dict:
    """Generate coalition blocks data."""
    kalman_series = calculate_kalman_averages(records)

    all_dates = sorted(set(d for series in kalman_series.values() for d, _, _ in series))

    gov_data = {"dates": [], "values": [], "upper": [], "lower": []}
    opp_data = {"dates": [], "values": [], "upper": [], "lower": []}

    for date in all_dates:
        gov_sum, opp_sum = 0, 0
        gov_var, opp_var = 0, 0

        for party, series in kalman_series.items():
            for d, v, u in series:
                if d == date:
                    if party in GOVERNMENT_PARTIES:
                        gov_sum += v
                        gov_var += u ** 2
                    elif party in OPPOSITION_PARTIES:
                        opp_sum += v
                        opp_var += u ** 2

        date_str = date.strftime("%Y-%m-%d")
        gov_data["dates"].append(date_str)
        gov_data["values"].append(round(gov_sum, 2))
        gov_data["upper"].append(round(gov_sum + 1.96 * np.sqrt(gov_var), 2))
        gov_data["lower"].append(round(gov_sum - 1.96 * np.sqrt(gov_var), 2))

        opp_data["dates"].append(date_str)
        opp_data["values"].append(round(opp_sum, 2))
        opp_data["upper"].append(round(opp_sum + 1.96 * np.sqrt(opp_var), 2))
        opp_data["lower"].append(round(opp_sum - 1.96 * np.sqrt(opp_var), 2))

    # Add forecast
    last_date = all_dates[-1] if all_dates else datetime.now()
    gov_forecast = {"dates": [], "values": []}
    opp_forecast = {"dates": [], "values": []}

    for i in range(1, forecast_months + 1):
        forecast_date = last_date + timedelta(days=30 * i)
        date_str = forecast_date.strftime("%Y-%m-%d")
        gov_sum, opp_sum = 0, 0

        for party in GOVERNMENT_PARTIES:
            forecast = kalman_forecast(records, party, forecast_months)
            if forecast and i <= len(forecast):
                _, v, _ = forecast[i - 1]
                gov_sum += v

        for party in OPPOSITION_PARTIES:
            forecast = kalman_forecast(records, party, forecast_months)
            if forecast and i <= len(forecast):
                _, v, _ = forecast[i - 1]
                opp_sum += v

        gov_forecast["dates"].append(date_str)
        gov_forecast["values"].append(round(gov_sum, 2))
        opp_forecast["dates"].append(date_str)
        opp_forecast["values"].append(round(opp_sum, 2))

    return {
        "government": {**gov_data, "forecast": gov_forecast},
        "opposition": {**opp_data, "forecast": opp_forecast},
    }


def get_seats_data(polls: Dict, records: List[Dict], forecast_months: int) -> Dict:
    """Generate parliament seats data."""
    mandate_records = flatten_mandates_to_timeseries(polls)
    seat_series = calculate_monthly_seat_averages(mandate_records)

    seat_dates = sorted(set(d for series in seat_series.values() for d, _ in series))

    gov_seats = {"dates": [], "values": []}
    opp_seats = {"dates": [], "values": []}

    for date in seat_dates:
        gov_sum, opp_sum = 0, 0
        for party, series in seat_series.items():
            for d, v in series:
                if d == date:
                    if party in GOVERNMENT_PARTIES:
                        gov_sum += v
                    elif party in OPPOSITION_PARTIES:
                        opp_sum += v

        date_str = date.strftime("%Y-%m-%d")
        gov_seats["dates"].append(date_str)
        gov_seats["values"].append(int(gov_sum))
        opp_seats["dates"].append(date_str)
        opp_seats["values"].append(int(opp_sum))

    # Forecast seats
    last_date = seat_dates[-1] if seat_dates else datetime.now()
    gov_forecast = {"dates": [], "values": []}
    opp_forecast = {"dates": [], "values": []}

    for i in range(1, forecast_months + 1):
        forecast_date = last_date + timedelta(days=30 * i)
        date_str = forecast_date.strftime("%Y-%m-%d")
        gov_sum, opp_sum = 0, 0

        for party in GOVERNMENT_PARTIES:
            forecast = kalman_forecast(records, party, forecast_months)
            if forecast and i <= len(forecast):
                _, pct, _ = forecast[i - 1]
                if pct >= 5:
                    gov_sum += pct * 1.7

        for party in OPPOSITION_PARTIES:
            forecast = kalman_forecast(records, party, forecast_months)
            if forecast and i <= len(forecast):
                _, pct, _ = forecast[i - 1]
                if pct >= 5:
                    opp_sum += pct * 1.7

        gov_forecast["dates"].append(date_str)
        gov_forecast["values"].append(int(gov_sum))
        opp_forecast["dates"].append(date_str)
        opp_forecast["values"].append(int(opp_sum))

    return {
        "government": {**gov_seats, "forecast": gov_forecast},
        "opposition": {**opp_seats, "forecast": opp_forecast},
    }


def get_correlation_data(records: List[Dict], threshold: float) -> Dict:
    """Get correlation analysis data."""
    parties, corr_matrix = calculate_party_correlations(records)

    if corr_matrix is None or len(parties) < 2:
        return {"matrix": [], "parties": [], "correlations": []}

    # Filter to parties above threshold
    kalman_series = calculate_kalman_averages(records)
    filtered_parties = []
    filtered_indices = []
    for i, party in enumerate(parties):
        if party in kalman_series and kalman_series[party]:
            current = kalman_series[party][-1][1]
            if current >= threshold:
                filtered_parties.append(party)
                filtered_indices.append(i)

    if len(filtered_parties) < 2:
        return {"matrix": [], "parties": [], "correlations": []}

    # Build filtered matrix
    matrix = []
    for i in filtered_indices:
        row = []
        for j in filtered_indices:
            val = corr_matrix[i, j]
            row.append(round(val, 2) if not np.isnan(val) else 0)
        matrix.append(row)

    # Extract notable correlations
    correlations = []
    for i, p1 in enumerate(filtered_parties):
        for j, p2 in enumerate(filtered_parties):
            if i < j:
                corr = matrix[i][j]
                if abs(corr) > 0.3:
                    correlations.append({
                        "party1": p1,
                        "party2": p2,
                        "value": corr,
                        "type": "positive" if corr > 0 else "negative"
                    })

    correlations.sort(key=lambda x: abs(x["value"]), reverse=True)

    return {
        "matrix": matrix,
        "parties": filtered_parties,
        "correlations": correlations[:10],
    }


def get_volatility_data(records: List[Dict], threshold: float) -> List[Dict]:
    """Get volatility analysis data."""
    volatility = calculate_volatility(records)

    result = []
    for party, data in volatility.items():
        if data["mean"] < threshold:
            continue

        cv = data["coef_variation"]
        result.append({
            "party": party,
            "mean": round(data["mean"], 1),
            "std": round(data["std_dev"], 2),
            "range": round(data["range"], 1),
            "cv": round(cv, 1),
            "color": PARTY_COLORS.get(party, "#666666"),
            "stability": "high" if cv < 10 else "medium" if cv < 20 else "low"
        })

    result.sort(key=lambda x: x["cv"])
    return result


def get_scenario_data(records: List[Dict]) -> List[Dict]:
    """Pre-generate scenario analysis results."""
    try:
        from sk_election_model import calculate_mandates, ModelParams

        # Load weights
        try:
            with open("data/weights.json", "r") as f:
                params = ModelParams.from_dict(json.load(f))
        except FileNotFoundError:
            params = ModelParams()

        # Get current estimates
        current = get_current_estimates(records)
        current_seats = calculate_mandates(current, params)

        # Calculate current block totals
        current_gov = sum(current_seats.get(p, 0) for p in GOVERNMENT_PARTIES)
        current_opp = sum(current_seats.get(p, 0) for p in OPPOSITION_PARTIES)

        scenarios = []

        # Load scenarios from file if exists
        scenario_definitions = []
        try:
            with open("data/scenarios.json", "r") as f:
                scenario_definitions = json.load(f).get("scenarios", [])
        except FileNotFoundError:
            # Default scenarios
            scenario_definitions = [
                {
                    "name": "HLAS below threshold",
                    "description": "What if HLAS falls below 5%?",
                    "modifications": {"HLAS": 4.5}
                },
                {
                    "name": "SNS enters parliament",
                    "description": "What if SNS rises above 5%?",
                    "modifications": {"SNS": 5.5}
                },
                {
                    "name": "HLAS voters to SMER",
                    "description": "3% of voters switch from HLAS to SMER",
                    "modifications": {"from": "HLAS", "to": "SMER", "amount": 3}
                },
                {
                    "name": "REP surge",
                    "description": "Republika gains 3 percentage points",
                    "modifications": {"REP": "+3"}
                },
                {
                    "name": "Government collapse",
                    "description": "SMER drops 5%, split between PS and REP",
                    "modifications": {"SMER": "-5", "PS": "+2.5", "REP": "+2.5"}
                },
            ]

        for scenario_def in scenario_definitions:
            name = scenario_def.get("name", "Unknown")
            description = scenario_def.get("description", "")
            modifications_raw = scenario_def.get("modifications", {})

            # Parse modifications
            modifications = {}
            for party, value in modifications_raw.items():
                if party in ["from", "to", "amount"]:
                    continue
                if isinstance(value, str):
                    if value.startswith('+') or value.startswith('-'):
                        modifications[party] = current.get(party, 0) + float(value)
                    else:
                        modifications[party] = float(value)
                else:
                    modifications[party] = float(value)

            # Handle transfer syntax
            if "from" in modifications_raw and "to" in modifications_raw:
                from_party = modifications_raw["from"]
                to_party = modifications_raw["to"]
                amount = float(modifications_raw.get("amount", 0))
                modifications[from_party] = current.get(from_party, 0) - amount
                modifications[to_party] = current.get(to_party, 0) + amount

            # Apply to current state
            scenario_state = current.copy()
            for party, value in modifications.items():
                scenario_state[party] = max(0, value)

            # Calculate seats
            scenario_seats = calculate_mandates(scenario_state, params)

            # Calculate block totals
            scenario_gov = sum(scenario_seats.get(p, 0) for p in GOVERNMENT_PARTIES)
            scenario_opp = sum(scenario_seats.get(p, 0) for p in OPPOSITION_PARTIES)

            # Build changes list
            changes = []
            for party, new_val in modifications.items():
                old_val = current.get(party, 0)
                old_seats = current_seats.get(party, 0)
                new_seats = scenario_seats.get(party, 0)
                changes.append({
                    "party": party,
                    "oldPct": round(old_val, 1),
                    "newPct": round(new_val, 1),
                    "pctChange": round(new_val - old_val, 1),
                    "oldSeats": old_seats,
                    "newSeats": new_seats,
                    "seatChange": new_seats - old_seats,
                })

            # All party seats
            all_seats = []
            for party in sorted(scenario_seats.keys(), key=lambda p: scenario_seats[p], reverse=True):
                if scenario_seats[party] > 0 or current_seats.get(party, 0) > 0:
                    all_seats.append({
                        "party": party,
                        "seats": scenario_seats[party],
                        "change": scenario_seats[party] - current_seats.get(party, 0),
                        "color": PARTY_COLORS.get(party, "#666666"),
                    })

            scenarios.append({
                "name": name,
                "description": description,
                "changes": changes,
                "seats": all_seats,
                "blocks": {
                    "government": {
                        "seats": scenario_gov,
                        "change": scenario_gov - current_gov,
                        "hasMajority": scenario_gov >= 76,
                        "hasConstitutional": scenario_gov >= 90,
                    },
                    "opposition": {
                        "seats": scenario_opp,
                        "change": scenario_opp - current_opp,
                        "hasMajority": scenario_opp >= 76,
                    }
                }
            })

        return scenarios

    except Exception as e:
        print(f"Warning: Could not generate scenarios: {e}")
        return []


def generate_html(output_dir: str, threshold: float = 5.0, forecast_months: int = 6):
    """Generate complete static HTML site with interactive charts."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    polls = load_all_polls("data")
    records = flatten_to_timeseries(polls)

    # Get date range
    all_dates = [r["date"] for r in records]
    date_start = min(all_dates).strftime("%B %Y")
    date_end = max(all_dates).strftime("%B %Y")

    print("Generating chart data...")
    # Always generate 12 months forecast so user can select different lengths
    max_forecast = 12
    parties_data = get_chart_data(records, threshold, max_forecast)
    raw_parties_data = get_raw_chart_data(records, threshold, max_forecast)
    blocks_data = get_blocks_data(records, max_forecast)
    seats_data = get_seats_data(polls, records, max_forecast)

    print("Calculating statistics...")
    correlation_data = get_correlation_data(records, threshold)
    volatility_data = get_volatility_data(records, threshold)

    print("Generating scenarios...")
    scenarios_data = get_scenario_data(records)

    # Current estimates for display
    current_estimates = get_current_estimates(records)

    # Generate timestamp
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Translations
    translations = {
        "en": {
            "title": "Slovak Election Poll Analysis",
            "subtitle": "Interactive Kalman-filtered trends and statistical analysis",
            "threshold": "Threshold",
            "forecast": "Forecast",
            "months": "months",
            "trends": "Trends",
            "blocks": "Blocks",
            "seats": "Seats",
            "scenarios": "Scenarios",
            "correlations": "Correlations",
            "volatility": "Volatility",
            "partyTrends": "Party Trends",
            "kalmanFiltered": "Kalman Filtered",
            "rawData": "Raw Data",
            "clickToHighlight": "Click on a party in the legend to highlight it. Dashed lines show forecasts with 95% confidence intervals.",
            "coalitionBlocks": "Coalition Blocks",
            "government": "Government",
            "opposition": "Opposition",
            "parliamentSeats": "Parliament Seats",
            "simpleMajority": "Simple majority",
            "constitutionalMajority": "Constitutional majority (2/3)",
            "whatIfScenarios": "What-If Scenarios",
            "clickScenario": "Click on a scenario to see detailed seat projections. These are pre-calculated based on current polling data.",
            "partyCorrelations": "Party Correlations",
            "pollingVolatility": "Polling Volatility",
            "party": "Party",
            "mean": "Mean",
            "stdDev": "Std Dev",
            "cv": "CV",
            "stability": "Stability",
            "high": "High",
            "medium": "Medium",
            "low": "Low",
            "before": "Before",
            "after": "After",
            "changes": "Changes",
            "allSeats": "All Seats",
            "hasMajority": "Has Majority",
            "noMajority": "No Majority",
            "govMajority": "Gov Majority",
            "oppLeads": "Opp Leads",
            "notableCorrelations": "Notable Correlations",
            "moveTogether": "move together",
            "inverseRelationship": "inverse relationship",
            "dataSources": "Data sources: Wikipedia opinion polling, various Slovak polling agencies",
            "analysisPowered": "Analysis powered by Kalman filtering",
            "downloadJson": "Download JSON Data",
            "generatedWith": "Generated with",
            "updated": "Updated",
            "useKalman": "Use Kalman Filter",
        },
        "sk": {
            "title": "Analýza volebných prieskumov SR",
            "subtitle": "Interaktívne trendy s Kalmanovým filtrom a štatistická analýza",
            "threshold": "Prah",
            "forecast": "Predpoveď",
            "months": "mesiacov",
            "trends": "Trendy",
            "blocks": "Bloky",
            "seats": "Mandáty",
            "scenarios": "Scenáre",
            "correlations": "Korelácie",
            "volatility": "Volatilita",
            "partyTrends": "Trendy strán",
            "kalmanFiltered": "Kalmanov filter",
            "rawData": "Surové dáta",
            "clickToHighlight": "Kliknite na stranu v legende pre zvýraznenie. Prerušované čiary zobrazujú predpovede s 95% intervalom spoľahlivosti.",
            "coalitionBlocks": "Koaličné bloky",
            "government": "Vláda",
            "opposition": "Opozícia",
            "parliamentSeats": "Mandáty v parlamente",
            "simpleMajority": "Jednoduchá väčšina",
            "constitutionalMajority": "Ústavná väčšina (2/3)",
            "whatIfScenarios": "Čo ak scenáre",
            "clickScenario": "Kliknite na scenár pre zobrazenie detailných projekcií mandátov. Tieto sú prepočítané na základe aktuálnych prieskumov.",
            "partyCorrelations": "Korelácie strán",
            "pollingVolatility": "Volatilita prieskumov",
            "party": "Strana",
            "mean": "Priemer",
            "stdDev": "Štd. odch.",
            "cv": "KV",
            "stability": "Stabilita",
            "high": "Vysoká",
            "medium": "Stredná",
            "low": "Nízka",
            "before": "Pred",
            "after": "Po",
            "changes": "Zmeny",
            "allSeats": "Všetky mandáty",
            "hasMajority": "Má väčšinu",
            "noMajority": "Nemá väčšinu",
            "govMajority": "Väčšina vlády",
            "oppLeads": "Vedie opozícia",
            "notableCorrelations": "Významné korelácie",
            "moveTogether": "pohybujú sa spolu",
            "inverseRelationship": "inverzný vzťah",
            "dataSources": "Zdroje dát: Wikipedia, rôzne slovenské agentúry",
            "analysisPowered": "Analýza pomocou Kalmanovho filtra",
            "downloadJson": "Stiahnuť JSON dáta",
            "generatedWith": "Generované pomocou",
            "updated": "Aktualizované",
            "useKalman": "Použiť Kalmanov filter",
        }
    }

    # Build complete data object
    site_data = {
        "meta": {
            "generatedAt": generated_at,
            "dateRange": {"start": date_start, "end": date_end},
            "defaultThreshold": threshold,
            "defaultForecast": forecast_months,
        },
        "parties": parties_data,
        "partiesRaw": raw_parties_data,
        "blocks": blocks_data,
        "seats": seats_data,
        "correlations": correlation_data,
        "volatility": volatility_data,
        "scenarios": scenarios_data,
        "translations": translations,
        "config": {
            "partyColors": PARTY_COLORS,
            "governmentParties": GOVERNMENT_PARTIES,
            "oppositionParties": OPPOSITION_PARTIES,
            "catholicParties": CATHOLIC_PARTIES,
        }
    }

    print("Generating HTML...")
    html = generate_html_template(site_data)

    # Write HTML
    index_path = output_path / "index.html"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated: {index_path}")

    # Write JSON data
    data_path = output_path / "data.json"
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(site_data, f, indent=2, cls=NumpyEncoder)
    print(f"Generated: {data_path}")


def generate_html_template(data: Dict) -> str:
    """Generate the HTML template with embedded data and Chart.js."""

    data_json = json.dumps(data, cls=NumpyEncoder)

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slovak Election Poll Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>
    <style>
        :root {
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --danger: #e74c3c;
            --warning: #f39c12;
            --light: #ecf0f1;
            --dark: #2c3e50;
            --gov-color: #e74c3c;
            --opp-color: #3498db;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: var(--dark);
            background: #f5f6fa;
        }

        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }

        header {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            padding: 30px 20px;
            text-align: center;
        }

        header h1 { font-size: 2.2em; margin-bottom: 5px; }
        header .subtitle { opacity: 0.9; }

        .meta {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 15px;
            flex-wrap: wrap;
        }

        .meta-item {
            background: rgba(255,255,255,0.2);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85em;
        }

        .controls {
            background: white;
            padding: 15px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 30px;
            flex-wrap: wrap;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .control-group {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .control-group label { font-weight: 500; font-size: 0.9em; }

        .control-group input[type="range"] { width: 100px; }

        .control-group select, .control-group input {
            padding: 6px 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 0.9em;
        }

        .nav-links {
            display: flex;
            gap: 10px;
        }

        .nav-links a {
            color: var(--primary);
            text-decoration: none;
            padding: 6px 12px;
            border-radius: 5px;
            font-size: 0.9em;
            transition: background 0.2s;
        }

        .nav-links a:hover { background: var(--light); }

        .card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 25px;
            overflow: hidden;
        }

        .card-header {
            background: var(--primary);
            color: white;
            padding: 12px 20px;
            font-size: 1.1em;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-header .badge {
            background: rgba(255,255,255,0.2);
            padding: 3px 10px;
            border-radius: 10px;
            font-size: 0.8em;
        }

        .card-body { padding: 20px; }

        .chart-container {
            position: relative;
            height: 400px;
            width: 100%;
        }

        .chart-container.small { height: 300px; }

        .party-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 15px;
            justify-content: center;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
            font-size: 0.9em;
        }

        .legend-item:hover { transform: scale(1.05); }
        .legend-item.highlighted { border-color: var(--dark); box-shadow: 0 2px 8px rgba(0,0,0,0.2); }
        .legend-item.dimmed { opacity: 0.3; }

        .legend-color {
            width: 14px;
            height: 14px;
            border-radius: 50%;
        }

        .legend-item .trend {
            font-size: 0.8em;
            margin-left: 4px;
        }

        .legend-item .trend.up { color: var(--success); }
        .legend-item .trend.down { color: var(--danger); }

        .block-indicator {
            font-size: 0.7em;
            padding: 2px 6px;
            border-radius: 3px;
            margin-left: 5px;
        }

        .block-indicator.gov { background: var(--gov-color); color: white; }
        .block-indicator.opp { background: var(--opp-color); color: white; }

        .two-column {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 25px;
        }

        @media (max-width: 900px) {
            .two-column { grid-template-columns: 1fr; }
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--light);
        }

        th { background: var(--light); font-weight: 600; font-size: 0.9em; }
        tr:hover { background: #f8f9fa; }

        .rising { color: var(--success); }
        .falling { color: var(--danger); }
        .stable { color: var(--warning); }

        .high { color: var(--success); }
        .medium { color: var(--warning); }
        .low { color: var(--danger); }

        .scenario-card {
            border: 1px solid var(--light);
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
        }

        .scenario-header {
            background: var(--light);
            padding: 12px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .scenario-header:hover { background: #dfe6e9; }

        .scenario-header h4 { margin: 0; font-size: 1em; }

        .scenario-content {
            padding: 15px;
            display: none;
        }

        .scenario-content.active { display: block; }

        .scenario-blocks {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }

        .scenario-block {
            flex: 1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .scenario-block.gov { background: rgba(231, 76, 60, 0.1); border: 2px solid var(--gov-color); }
        .scenario-block.opp { background: rgba(52, 152, 219, 0.1); border: 2px solid var(--opp-color); }

        .scenario-block .seats { font-size: 2em; font-weight: bold; }
        .scenario-block .change { font-size: 0.9em; }
        .scenario-block .status { margin-top: 5px; font-size: 0.85em; }

        .correlation-matrix {
            display: grid;
            gap: 2px;
            margin: 20px auto;
            max-width: 600px;
        }

        .correlation-cell {
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75em;
            font-weight: 500;
            border-radius: 3px;
        }

        .correlation-label {
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            font-size: 0.8em;
            background: var(--light);
        }

        footer {
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.85em;
        }

        footer a { color: var(--secondary); }

        .lang-switch { gap: 5px; }

        .lang-btn {
            padding: 6px 12px;
            border: 2px solid var(--primary);
            background: white;
            color: var(--primary);
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85em;
            transition: all 0.2s;
        }

        .lang-btn.active {
            background: var(--primary);
            color: white;
        }

        .lang-btn:hover:not(.active) { background: var(--light); }

        .toggle-label {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.9em;
        }

        .toggle-label input { display: none; }

        .toggle-slider {
            width: 40px;
            height: 22px;
            background: #ccc;
            border-radius: 11px;
            position: relative;
            transition: background 0.2s;
        }

        .toggle-slider::before {
            content: '';
            position: absolute;
            width: 18px;
            height: 18px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.2s;
        }

        .toggle-label input:checked + .toggle-slider {
            background: var(--secondary);
        }

        .toggle-label input:checked + .toggle-slider::before {
            transform: translateX(18px);
        }

        .info-box {
            background: #e8f4fd;
            border-left: 4px solid var(--secondary);
            padding: 12px 15px;
            margin: 15px 0;
            font-size: 0.9em;
        }

        .block-legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 15px 0;
            font-size: 0.9em;
        }

        .block-legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .block-legend-color {
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <header>
        <h1>🇸🇰 Slovak Election Poll Analysis</h1>
        <p class="subtitle">Interactive Kalman-filtered trends and statistical analysis</p>
        <div class="meta">
            <span class="meta-item" id="dateRange"></span>
            <span class="meta-item" id="lastUpdated"></span>
        </div>
    </header>

    <div class="controls">
        <div class="control-group lang-switch">
            <button id="langEN" class="lang-btn active" onclick="setLanguage('en')">EN</button>
            <button id="langSK" class="lang-btn" onclick="setLanguage('sk')">SK</button>
        </div>
        <div class="control-group">
            <label data-i18n="threshold">Threshold:</label>
            <input type="range" id="thresholdSlider" min="0" max="10" step="0.5" value="5">
            <span id="thresholdValue">5%</span>
        </div>
        <div class="control-group">
            <label data-i18n="forecast">Forecast:</label>
            <select id="forecastSelect">
                <option value="3">3</option>
                <option value="6" selected>6</option>
                <option value="9">9</option>
                <option value="12">12</option>
            </select>
            <span data-i18n="months">months</span>
        </div>
        <div class="control-group">
            <label class="toggle-label">
                <input type="checkbox" id="kalmanToggle" checked>
                <span class="toggle-slider"></span>
                <span data-i18n="useKalman">Kalman Filter</span>
            </label>
        </div>
        <div class="nav-links">
            <a href="#trends" data-i18n="trends">Trends</a>
            <a href="#blocks" data-i18n="blocks">Blocks</a>
            <a href="#seats" data-i18n="seats">Seats</a>
            <a href="#scenarios" data-i18n="scenarios">Scenarios</a>
            <a href="#correlations" data-i18n="correlations">Correlations</a>
            <a href="#volatility" data-i18n="volatility">Volatility</a>
        </div>
    </div>

    <div class="container">
        <!-- Party Trends -->
        <section id="trends" class="card">
            <div class="card-header">
                📈 Party Trends
                <span class="badge">Kalman Filtered</span>
            </div>
            <div class="card-body">
                <div class="info-box">
                    Click on a party in the legend to highlight it. Dashed lines show forecasts with 95% confidence intervals.
                </div>
                <div class="chart-container">
                    <canvas id="trendsChart"></canvas>
                </div>
                <div class="party-legend" id="partyLegend"></div>
            </div>
        </section>

        <!-- Coalition Blocks -->
        <section id="blocks" class="card">
            <div class="card-header">
                ⚖️ Coalition Blocks
            </div>
            <div class="card-body">
                <div class="block-legend">
                    <div class="block-legend-item">
                        <div class="block-legend-color" style="background: var(--gov-color);"></div>
                        <span><strong>Government:</strong> SMER + HLAS + SNS + REP</span>
                    </div>
                    <div class="block-legend-item">
                        <div class="block-legend-color" style="background: var(--opp-color);"></div>
                        <span><strong>Opposition:</strong> PS + SAS + DEM + SLOV + KDH</span>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="blocksChart"></canvas>
                </div>
            </div>
        </section>

        <!-- Parliament Seats -->
        <section id="seats" class="card">
            <div class="card-header">
                🏛️ Parliament Seats
            </div>
            <div class="card-body">
                <div class="chart-container">
                    <canvas id="seatsChart"></canvas>
                </div>
                <div class="info-box">
                    <strong>76 seats</strong> = Simple majority | <strong>90 seats</strong> = Constitutional majority (2/3)
                </div>
            </div>
        </section>

        <!-- Scenarios -->
        <section id="scenarios" class="card">
            <div class="card-header">
                🎭 What-If Scenarios
                <span class="badge" id="scenarioCount"></span>
            </div>
            <div class="card-body">
                <div class="info-box">
                    Click on a scenario to see detailed seat projections. These are pre-calculated based on current polling data.
                </div>
                <div id="scenariosList"></div>
            </div>
        </section>

        <div class="two-column">
            <!-- Correlations -->
            <section id="correlations" class="card">
                <div class="card-header">🔗 Party Correlations</div>
                <div class="card-body">
                    <div id="correlationMatrix"></div>
                    <div id="correlationList"></div>
                </div>
            </section>

            <!-- Volatility -->
            <section id="volatility" class="card">
                <div class="card-header">📊 Polling Volatility</div>
                <div class="card-body">
                    <table id="volatilityTable">
                        <thead>
                            <tr>
                                <th>Party</th>
                                <th>Mean</th>
                                <th>Std Dev</th>
                                <th>CV</th>
                                <th>Stability</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </section>
        </div>
    </div>

    <footer>
        <p>Data sources: Wikipedia opinion polling, various Slovak polling agencies</p>
        <p>Analysis powered by Kalman filtering | <a href="data.json">Download JSON Data</a></p>
        <p>Generated with <a href="https://github.com/anthropics/claude-code" target="_blank">Claude Code</a></p>
    </footer>

    <script>
        // Embedded data
        const DATA = ''' + data_json + ''';

        // State
        let highlightedParty = null;
        let currentThreshold = DATA.meta.defaultThreshold;
        let currentForecast = DATA.meta.defaultForecast;
        let currentLang = localStorage.getItem('sk-polls-lang') || 'en';
        let useKalman = localStorage.getItem('sk-polls-kalman') !== 'false';

        // Charts
        let trendsChart, blocksChart, seatsChart;

        // Translation function
        function t(key) {
            return DATA.translations[currentLang][key] || DATA.translations['en'][key] || key;
        }

        function setLanguage(lang) {
            currentLang = lang;
            localStorage.setItem('sk-polls-lang', lang);

            // Update button states
            document.getElementById('langEN').classList.toggle('active', lang === 'en');
            document.getElementById('langSK').classList.toggle('active', lang === 'sk');

            // Update all translated elements
            document.querySelectorAll('[data-i18n]').forEach(el => {
                const key = el.getAttribute('data-i18n');
                el.textContent = t(key);
            });

            // Update dynamic content
            updateTranslatedContent();
        }

        function updateTranslatedContent() {
            // Update header
            document.querySelector('header h1').textContent = '🇸🇰 ' + t('title');
            document.querySelector('header .subtitle').textContent = t('subtitle');

            // Update card headers
            document.querySelector('#trends .card-header').innerHTML = `📈 ${t('partyTrends')} <span class="badge">${useKalman ? t('kalmanFiltered') : t('rawData')}</span>`;
            document.querySelector('#blocks .card-header').innerHTML = `⚖️ ${t('coalitionBlocks')}`;
            document.querySelector('#seats .card-header').innerHTML = `🏛️ ${t('parliamentSeats')}`;
            document.querySelector('#scenarios .card-header').innerHTML = `🎭 ${t('whatIfScenarios')} <span class="badge" id="scenarioCount">${DATA.scenarios.length} ${t('scenarios').toLowerCase()}</span>`;
            document.querySelector('#correlations .card-header').textContent = '🔗 ' + t('partyCorrelations');
            document.querySelector('#volatility .card-header').textContent = '📊 ' + t('pollingVolatility');

            // Update info boxes
            document.querySelector('#trends .info-box').textContent = t('clickToHighlight');
            document.querySelector('#scenarios .info-box').textContent = t('clickScenario');
            document.querySelector('#seats .info-box').innerHTML = `<strong>76 ${t('seats').toLowerCase()}</strong> = ${t('simpleMajority')} | <strong>90 ${t('seats').toLowerCase()}</strong> = ${t('constitutionalMajority')}`;

            // Update block legend
            document.querySelector('.block-legend').innerHTML = `
                <div class="block-legend-item">
                    <div class="block-legend-color" style="background: var(--gov-color);"></div>
                    <span><strong>${t('government')}:</strong> SMER + HLAS + SNS + REP</span>
                </div>
                <div class="block-legend-item">
                    <div class="block-legend-color" style="background: var(--opp-color);"></div>
                    <span><strong>${t('opposition')}:</strong> PS + SAS + DEM + SLOV + KDH</span>
                </div>
            `;

            // Update volatility table header
            const volHeader = document.querySelector('#volatilityTable thead tr');
            volHeader.innerHTML = `
                <th>${t('party')}</th>
                <th>${t('mean')}</th>
                <th>${t('stdDev')}</th>
                <th>${t('cv')}</th>
                <th>${t('stability')}</th>
            `;

            // Update footer
            document.querySelector('footer').innerHTML = `
                <p>${t('dataSources')}</p>
                <p>${t('analysisPowered')} | <a href="data.json">${t('downloadJson')}</a></p>
                <p>${t('generatedWith')} <a href="https://github.com/anthropics/claude-code" target="_blank">Claude Code</a></p>
            `;

            // Re-render dynamic elements
            renderScenarios();
            renderCorrelations();
            renderVolatility();
        }

        function getActivePartyData() {
            return useKalman ? DATA.parties : DATA.partiesRaw;
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            // Set meta info
            document.getElementById('dateRange').textContent =
                `📅 ${DATA.meta.dateRange.start} - ${DATA.meta.dateRange.end}`;
            document.getElementById('lastUpdated').textContent =
                `🕐 ${t('updated')}: ${DATA.meta.generatedAt}`;

            // Initialize controls
            document.getElementById('thresholdSlider').value = currentThreshold;
            document.getElementById('thresholdValue').textContent = currentThreshold + '%';
            document.getElementById('forecastSelect').value = currentForecast;
            document.getElementById('kalmanToggle').checked = useKalman;

            // Set initial language
            setLanguage(currentLang);

            // Event listeners
            document.getElementById('thresholdSlider').addEventListener('input', (e) => {
                currentThreshold = parseFloat(e.target.value);
                document.getElementById('thresholdValue').textContent = currentThreshold + '%';
                updateCharts();
            });

            document.getElementById('forecastSelect').addEventListener('change', (e) => {
                currentForecast = parseInt(e.target.value);
                renderTrendsChart();
                renderBlocksChart();
                renderSeatsChart();
            });

            document.getElementById('kalmanToggle').addEventListener('change', (e) => {
                useKalman = e.target.checked;
                localStorage.setItem('sk-polls-kalman', useKalman);
                document.querySelector('#trends .card-header .badge').textContent = useKalman ? t('kalmanFiltered') : t('rawData');
                renderTrendsChart();
                renderBlocksChart();
                renderSeatsChart();
            });

            // Render everything
            renderTrendsChart();
            renderBlocksChart();
            renderSeatsChart();
            renderScenarios();
            renderCorrelations();
            renderVolatility();
        });

        function getVisibleParties() {
            const partyData = getActivePartyData();
            return Object.entries(partyData)
                .filter(([_, data]) => data.current >= currentThreshold)
                .sort((a, b) => b[1].current - a[1].current);
        }

        function renderTrendsChart() {
            const ctx = document.getElementById('trendsChart').getContext('2d');
            const parties = getVisibleParties();

            const datasets = [];

            parties.forEach(([party, data]) => {
                // Confidence band (only for Kalman mode when party is highlighted or no highlight)
                if (useKalman && data.uncertainties && (!highlightedParty || highlightedParty === party)) {
                    // Upper bound
                    datasets.push({
                        label: party + ' (upper)',
                        data: data.dates.map((d, i) => ({ x: d, y: data.values[i] + 1.96 * data.uncertainties[i] })),
                        borderColor: 'transparent',
                        backgroundColor: 'transparent',
                        pointRadius: 0,
                        fill: false,
                        tension: 0.3,
                    });
                    // Lower bound with fill to upper
                    datasets.push({
                        label: party + ' (lower)',
                        data: data.dates.map((d, i) => ({ x: d, y: data.values[i] - 1.96 * data.uncertainties[i] })),
                        borderColor: 'transparent',
                        backgroundColor: data.color + '15',
                        pointRadius: 0,
                        fill: '-1',  // Fill to previous dataset
                        tension: 0.3,
                    });
                }

                // Main line - smooth for Kalman, straight for raw
                datasets.push({
                    label: party,
                    data: data.dates.map((d, i) => ({ x: d, y: data.values[i] })),
                    borderColor: data.color,
                    backgroundColor: data.color + '20',
                    borderWidth: highlightedParty === party ? 4 : 2,
                    pointRadius: highlightedParty === party ? 4 : (useKalman ? 2 : 4),
                    tension: useKalman ? 0.3 : 0,  // Smooth curve for Kalman, straight lines for raw
                    fill: false,
                    hidden: highlightedParty && highlightedParty !== party,
                });

                // Forecast line (only with Kalman filter)
                if (useKalman && data.forecast.dates.length > 0) {
                    const lastDate = data.dates[data.dates.length - 1];
                    const lastValue = data.values[data.values.length - 1];

                    // Filter forecast by selected months
                    const forecastDates = data.forecast.dates.slice(0, currentForecast);
                    const forecastValues = data.forecast.values.slice(0, currentForecast);

                    if (forecastDates.length > 0) {
                        datasets.push({
                            label: party + ' (forecast)',
                            data: [
                                { x: lastDate, y: lastValue },
                                ...forecastDates.map((d, i) => ({ x: d, y: forecastValues[i] }))
                            ],
                            borderColor: data.color,
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            tension: 0.3,
                            fill: false,
                            hidden: highlightedParty && highlightedParty !== party,
                        });
                    }
                }
            });

            if (trendsChart) {
                trendsChart.destroy();
            }

            trendsChart = new Chart(ctx, {
                type: 'line',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        intersect: false,
                        mode: 'index',
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: (ctx) => {
                                    const label = ctx.dataset.label.replace(' (forecast)', '');
                                    return `${label}: ${ctx.parsed.y.toFixed(1)}%`;
                                }
                            }
                        },
                        annotation: {
                            annotations: {
                                threshold: {
                                    type: 'line',
                                    yMin: 5,
                                    yMax: 5,
                                    borderColor: 'rgba(128, 128, 128, 0.5)',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    label: {
                                        content: '5% threshold',
                                        enabled: true,
                                        position: 'end',
                                    }
                                }
                            }
                        }
                    },
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'month' },
                            title: { display: true, text: 'Date' }
                        },
                        y: {
                            min: 0,
                            max: 30,
                            title: { display: true, text: 'Poll Percentage (%)' }
                        }
                    }
                }
            });

            // Render legend
            renderPartyLegend(parties);
        }

        function renderPartyLegend(parties) {
            const container = document.getElementById('partyLegend');
            container.innerHTML = '';

            parties.forEach(([party, data]) => {
                const item = document.createElement('div');
                item.className = 'legend-item';
                if (highlightedParty === party) item.classList.add('highlighted');
                if (highlightedParty && highlightedParty !== party) item.classList.add('dimmed');

                item.style.background = data.color + '20';

                const trendClass = data.trend > 0.05 ? 'up' : data.trend < -0.05 ? 'down' : '';
                const trendIcon = data.trend > 0.05 ? '↑' : data.trend < -0.05 ? '↓' : '→';

                let blockBadge = '';
                if (data.isGovernment) blockBadge = '<span class="block-indicator gov">GOV</span>';
                else if (data.isOpposition) blockBadge = '<span class="block-indicator opp">OPP</span>';

                item.innerHTML = `
                    <span class="legend-color" style="background: ${data.color}"></span>
                    <span>${party}</span>
                    <span class="trend ${trendClass}">${data.current}% ${trendIcon}</span>
                    ${blockBadge}
                `;

                item.addEventListener('click', () => {
                    highlightedParty = highlightedParty === party ? null : party;
                    renderTrendsChart();
                });

                container.appendChild(item);
            });
        }

        function renderBlocksChart() {
            const ctx = document.getElementById('blocksChart').getContext('2d');
            const blocks = DATA.blocks;

            // Filter forecast by selected months
            const govForecastDates = blocks.government.forecast.dates.slice(0, currentForecast);
            const govForecastValues = blocks.government.forecast.values.slice(0, currentForecast);
            const oppForecastDates = blocks.opposition.forecast.dates.slice(0, currentForecast);
            const oppForecastValues = blocks.opposition.forecast.values.slice(0, currentForecast);

            const datasets = [
                {
                    label: t('government'),
                    data: blocks.government.dates.map((d, i) => ({ x: d, y: blocks.government.values[i] })),
                    borderColor: 'var(--gov-color)',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: useKalman ? 0.3 : 0,
                },
                {
                    label: t('opposition'),
                    data: blocks.opposition.dates.map((d, i) => ({ x: d, y: blocks.opposition.values[i] })),
                    borderColor: 'var(--opp-color)',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: useKalman ? 0.3 : 0,
                }
            ];

            // Add forecast lines only when Kalman is enabled
            if (useKalman && govForecastDates.length > 0) {
                datasets.push({
                    label: t('government') + ' (forecast)',
                    data: [
                        { x: blocks.government.dates[blocks.government.dates.length - 1],
                          y: blocks.government.values[blocks.government.values.length - 1] },
                        ...govForecastDates.map((d, i) => ({ x: d, y: govForecastValues[i] }))
                    ],
                    borderColor: 'var(--gov-color)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                });
                datasets.push({
                    label: t('opposition') + ' (forecast)',
                    data: [
                        { x: blocks.opposition.dates[blocks.opposition.dates.length - 1],
                          y: blocks.opposition.values[blocks.opposition.values.length - 1] },
                        ...oppForecastDates.map((d, i) => ({ x: d, y: oppForecastValues[i] }))
                    ],
                    borderColor: 'var(--opp-color)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                });
            }

            if (blocksChart) blocksChart.destroy();

            blocksChart = new Chart(ctx, {
                type: 'line',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            labels: {
                                filter: (item) => !item.text.includes('forecast')
                            }
                        },
                        annotation: {
                            annotations: {
                                majority: {
                                    type: 'line',
                                    yMin: 50,
                                    yMax: 50,
                                    borderColor: 'rgba(39, 174, 96, 0.7)',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                }
                            }
                        }
                    },
                    scales: {
                        x: { type: 'time', time: { unit: 'month' } },
                        y: { title: { display: true, text: 'Combined Percentage (%)' } }
                    }
                }
            });
        }

        function renderSeatsChart() {
            const ctx = document.getElementById('seatsChart').getContext('2d');
            const seats = DATA.seats;

            // Filter forecast by selected months
            const govForecastDates = seats.government.forecast.dates.slice(0, currentForecast);
            const govForecastValues = seats.government.forecast.values.slice(0, currentForecast);
            const oppForecastDates = seats.opposition.forecast.dates.slice(0, currentForecast);
            const oppForecastValues = seats.opposition.forecast.values.slice(0, currentForecast);

            const datasets = [
                {
                    label: t('government'),
                    data: seats.government.dates.map((d, i) => ({ x: d, y: seats.government.values[i] })),
                    borderColor: 'var(--gov-color)',
                    backgroundColor: 'rgba(231, 76, 60, 0.3)',
                    borderWidth: 3,
                    pointRadius: 4,
                    fill: true,
                    tension: useKalman ? 0.3 : 0,
                },
                {
                    label: t('opposition'),
                    data: seats.opposition.dates.map((d, i) => ({ x: d, y: seats.opposition.values[i] })),
                    borderColor: 'var(--opp-color)',
                    backgroundColor: 'rgba(52, 152, 219, 0.3)',
                    borderWidth: 3,
                    pointRadius: 4,
                    fill: true,
                    tension: useKalman ? 0.3 : 0,
                }
            ];

            // Add forecast lines only when Kalman is enabled
            if (useKalman && govForecastDates.length > 0) {
                datasets.push({
                    label: t('government') + ' (forecast)',
                    data: [
                        { x: seats.government.dates[seats.government.dates.length - 1],
                          y: seats.government.values[seats.government.values.length - 1] },
                        ...govForecastDates.map((d, i) => ({ x: d, y: govForecastValues[i] }))
                    ],
                    borderColor: 'var(--gov-color)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                });
                datasets.push({
                    label: t('opposition') + ' (forecast)',
                    data: [
                        { x: seats.opposition.dates[seats.opposition.dates.length - 1],
                          y: seats.opposition.values[seats.opposition.values.length - 1] },
                        ...oppForecastDates.map((d, i) => ({ x: d, y: oppForecastValues[i] }))
                    ],
                    borderColor: 'var(--opp-color)',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3,
                });
            }

            if (seatsChart) seatsChart.destroy();

            seatsChart = new Chart(ctx, {
                type: 'line',
                data: { datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        annotation: {
                            annotations: {
                                majority: {
                                    type: 'line',
                                    yMin: 76,
                                    yMax: 76,
                                    borderColor: 'rgba(39, 174, 96, 0.7)',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    label: { content: 'Majority (76)', enabled: true }
                                },
                                constitutional: {
                                    type: 'line',
                                    yMin: 90,
                                    yMax: 90,
                                    borderColor: 'rgba(243, 156, 18, 0.7)',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    label: { content: 'Constitutional (90)', enabled: true }
                                }
                            }
                        }
                    },
                    scales: {
                        x: { type: 'time', time: { unit: 'month' } },
                        y: {
                            min: 0,
                            max: 160,
                            title: { display: true, text: 'Parliament Seats' }
                        }
                    }
                }
            });
        }

        function renderScenarios() {
            const container = document.getElementById('scenariosList');
            const scenarios = DATA.scenarios;

            container.innerHTML = scenarios.map((scenario, idx) => `
                <div class="scenario-card">
                    <div class="scenario-header" onclick="toggleScenario(${idx})">
                        <h4>${scenario.name}</h4>
                        <span>${scenario.blocks.government.hasMajority ? '🔴 ' + t('govMajority') : '🔵 ' + t('oppLeads')}</span>
                    </div>
                    <div class="scenario-content" id="scenario-${idx}">
                        <p style="color: #666; margin-bottom: 15px;">${scenario.description}</p>

                        <h5 style="margin-bottom: 10px;">${t('changes')}:</h5>
                        <table>
                            <thead>
                                <tr>
                                    <th>${t('party')}</th>
                                    <th>${t('before')}</th>
                                    <th>${t('after')}</th>
                                    <th>Δ ${t('seats')}</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${scenario.changes.map(c => `
                                    <tr>
                                        <td><strong>${c.party}</strong></td>
                                        <td>${c.oldPct}%</td>
                                        <td>${c.newPct}%</td>
                                        <td class="${c.seatChange > 0 ? 'rising' : c.seatChange < 0 ? 'falling' : ''}">
                                            ${c.seatChange > 0 ? '+' : ''}${c.seatChange}
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>

                        <div class="scenario-blocks">
                            <div class="scenario-block gov">
                                <div>${t('government')}</div>
                                <div class="seats">${scenario.blocks.government.seats}</div>
                                <div class="change ${scenario.blocks.government.change >= 0 ? 'rising' : 'falling'}">
                                    ${scenario.blocks.government.change >= 0 ? '+' : ''}${scenario.blocks.government.change} ${t('seats').toLowerCase()}
                                </div>
                                <div class="status">
                                    ${scenario.blocks.government.hasMajority ? '✅ ' + t('hasMajority') : '❌ ' + t('noMajority')}
                                </div>
                            </div>
                            <div class="scenario-block opp">
                                <div>${t('opposition')}</div>
                                <div class="seats">${scenario.blocks.opposition.seats}</div>
                                <div class="change ${scenario.blocks.opposition.change >= 0 ? 'rising' : 'falling'}">
                                    ${scenario.blocks.opposition.change >= 0 ? '+' : ''}${scenario.blocks.opposition.change} ${t('seats').toLowerCase()}
                                </div>
                                <div class="status">
                                    ${scenario.blocks.opposition.hasMajority ? '✅ ' + t('hasMajority') : ''}
                                </div>
                            </div>
                        </div>

                        <h5 style="margin: 15px 0 10px;">${t('allSeats')}:</h5>
                        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                            ${scenario.seats.map(s => `
                                <span style="background: ${s.color}20; border: 2px solid ${s.color}; padding: 4px 10px; border-radius: 15px; font-size: 0.85em;">
                                    ${s.party}: ${s.seats}
                                    ${s.change !== 0 ? `<span class="${s.change > 0 ? 'rising' : 'falling'}">(${s.change > 0 ? '+' : ''}${s.change})</span>` : ''}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                </div>
            `).join('');
        }

        function toggleScenario(idx) {
            const content = document.getElementById(`scenario-${idx}`);
            content.classList.toggle('active');
        }

        function renderCorrelations() {
            const corr = DATA.correlations;

            if (corr.parties.length === 0) {
                document.getElementById('correlationMatrix').innerHTML = '<p>Not enough data for correlation analysis.</p>';
                return;
            }

            // Render matrix
            const n = corr.parties.length;
            const matrixContainer = document.getElementById('correlationMatrix');

            let gridStyle = `grid-template-columns: 50px repeat(${n}, 1fr);`;
            let html = `<div class="correlation-matrix" style="${gridStyle}">`;

            // Header row
            html += '<div></div>';
            corr.parties.forEach(p => {
                html += `<div class="correlation-label">${p}</div>`;
            });

            // Data rows
            corr.parties.forEach((p1, i) => {
                html += `<div class="correlation-label">${p1}</div>`;
                corr.parties.forEach((p2, j) => {
                    const val = corr.matrix[i][j];
                    const color = val > 0
                        ? `rgba(52, 152, 219, ${Math.abs(val)})`
                        : `rgba(231, 76, 60, ${Math.abs(val)})`;
                    const textColor = Math.abs(val) > 0.5 ? 'white' : 'black';
                    html += `<div class="correlation-cell" style="background: ${color}; color: ${textColor}">${val.toFixed(2)}</div>`;
                });
            });

            html += '</div>';
            matrixContainer.innerHTML = html;

            // Render list
            const listContainer = document.getElementById('correlationList');
            if (corr.correlations.length > 0) {
                html = `<div style="margin-top: 20px;"><h5>${t('notableCorrelations')}:</h5><ul style="margin-top: 10px;">`;
                corr.correlations.forEach(c => {
                    const icon = c.type === 'positive' ? '📈' : '📉';
                    const desc = c.type === 'positive' ? t('moveTogether') : t('inverseRelationship');
                    html += `<li style="margin: 8px 0;">${icon} <strong>${c.party1}</strong> ↔ <strong>${c.party2}</strong>: ${c.value.toFixed(2)} (${desc})</li>`;
                });
                html += '</ul></div>';
                listContainer.innerHTML = html;
            }
        }

        function renderVolatility() {
            const tbody = document.querySelector('#volatilityTable tbody');
            const data = DATA.volatility.filter(v => v.mean >= currentThreshold);

            tbody.innerHTML = data.map(v => {
                const stabilityText = t(v.stability);
                return `
                <tr>
                    <td>
                        <span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background: ${v.color}; margin-right: 8px;"></span>
                        <strong>${v.party}</strong>
                    </td>
                    <td>${v.mean}%</td>
                    <td>±${v.std}%</td>
                    <td>${v.cv}%</td>
                    <td class="${v.stability}">${stabilityText.charAt(0).toUpperCase() + stabilityText.slice(1)}</td>
                </tr>
            `}).join('');
        }

        function updateCharts() {
            renderTrendsChart();
            renderVolatility();
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</body>
</html>'''

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate static HTML for Slovak election analysis")
    parser.add_argument("--output-dir", type=str, default="docs",
                        help="Output directory for HTML files (default: docs/)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Default threshold percentage (default: 5.0)")
    parser.add_argument("--forecast", type=int, default=6,
                        help="Number of months to forecast (default: 6)")
    args = parser.parse_args()

    generate_html(args.output_dir, args.threshold, args.forecast)
    print("\nDone! You can serve the site with:")
    print(f"  cd {args.output_dir} && python -m http.server 8000")


if __name__ == "__main__":
    main()
