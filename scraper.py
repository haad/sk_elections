#!/usr/bin/env python3
"""
Slovak Election Poll Scraper

Scrapes polling data from Wikipedia and converts to the JSON format
used by the trend analysis tool.

Sources:
- https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Slovak_parliamentary_election
"""

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import pandas as pd


# Party name mappings (Wikipedia names -> our short names)
PARTY_MAPPING = {
    # Progressive Slovakia
    "PS": "PS",
    "Progresívne Slovensko": "PS",
    "Progressive Slovakia": "PS",

    # SMER
    "Smer": "SMER",
    "SMER": "SMER",
    "Smer–SD": "SMER",
    "Smer-SD": "SMER",
    "Smer–SSD": "SMER",

    # HLAS
    "Hlas": "HLAS",
    "HLAS": "HLAS",
    "Hlas–SD": "HLAS",
    "Hlas-SD": "HLAS",

    # Republika
    "Republika": "REP",
    "REP": "REP",
    "R": "REP",

    # Slovensko (Slovakia movement)
    "Slovensko": "SLOV",
    "S": "SLOV",
    "SLOV": "SLOV",
    "Slovensko (party)": "SLOV",

    # SaS (Freedom and Solidarity)
    "SaS": "SAS",
    "SAS": "SAS",
    "Sloboda a Solidarita": "SAS",

    # KDH (Christian Democrats)
    "KDH": "KDH",
    "Kresťanskodemokratické hnutie": "KDH",

    # Democrats
    "Demokrati": "DEM",
    "DEM": "DEM",
    "D": "DEM",

    # SNS
    "SNS": "SNS",
    "Slovenská národná strana": "SNS",

    # Magyar Alliance
    "Aliancia": "Aliancia",
    "MA": "Aliancia",
    "Szövetség": "Aliancia",
    "Magyar Szövetség": "Aliancia",

    # Sme Rodina
    "Sme rodina": "ROD",
    "SR": "ROD",
    "ROD": "ROD",
    "Sme Rodina": "ROD",
}

# Polling agency mappings
AGENCY_MAPPING = {
    "AKO": "ako",
    "Focus": "focus",
    "FOCUS": "focus",
    "Median": "median",
    "MEDIAN": "median",
    "NMS": "nms",
    "NMS Market Research": "nms",
    "Ipsos": "ipsos",
    "IPSOS": "ipsos",
    "Polis": "polis",
    "POLIS": "polis",
    "MVK": "mvk",
    "Infostat": "infostat",
    "INFOSTAT": "infostat",
}


def fetch_wikipedia_page(url: str) -> str:
    """Fetch Wikipedia page content using the Wikipedia API."""
    import time
    from urllib.parse import urlparse, unquote

    # Extract page title from URL
    parsed = urlparse(url)
    page_title = parsed.path.split('/wiki/')[-1]
    page_title = unquote(page_title)

    # Use Wikipedia API to get HTML content
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'parse',
        'page': page_title,
        'format': 'json',
        'prop': 'text',
        'disableeditsection': 'true',
    }

    headers = {
        'User-Agent': 'PollingDataScraper/1.0 (Slovak Election Analysis; contact@example.com)',
        'Accept': 'application/json',
    }

    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()

    if 'error' in data:
        raise ValueError(f"Wikipedia API error: {data['error'].get('info', 'Unknown error')}")

    html = data['parse']['text']['*']
    return html


def parse_percentage(value: str) -> Optional[float]:
    """Parse percentage value from string."""
    if not value or value in ['–', '—', '-', 'N/A', '']:
        return None

    # Remove % sign and whitespace
    value = value.replace('%', '').strip()

    # Handle range values (take average)
    if '–' in value or '-' in value:
        parts = re.split(r'[–-]', value)
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                return None

    try:
        return float(value)
    except ValueError:
        return None


def parse_date(date_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse date string and return (year, month, day)."""
    date_str = date_str.strip()

    # Common date patterns
    patterns = [
        (r'(\d{1,2})\s*[-–]\s*(\d{1,2})\s+(\w+)\s+(\d{4})', 'range_dmy'),  # "18-20 Nov 2025"
        (r'(\d{1,2})\s+(\w+)\s+(\d{4})', 'dmy'),  # "20 Nov 2025"
        (r'(\w+)\s+(\d{1,2})[-–](\d{1,2}),?\s+(\d{4})', 'range_mdy'),  # "Nov 18-20, 2025"
        (r'(\w+)\s+(\d{4})', 'my'),  # "November 2025"
        (r'(\d{4})-(\d{2})-(\d{2})', 'iso'),  # "2025-11-20"
    ]

    months = {
        'jan': 1, 'january': 1, 'január': 1,
        'feb': 2, 'february': 2, 'február': 2,
        'mar': 3, 'march': 3, 'marec': 3,
        'apr': 4, 'april': 4, 'apríl': 4,
        'may': 5, 'máj': 5,
        'jun': 6, 'june': 6, 'jún': 6,
        'jul': 7, 'july': 7, 'júl': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'október': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    for pattern, fmt in patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            try:
                if fmt == 'range_dmy':
                    day = int(match.group(2))  # Take end date
                    month = months.get(match.group(3).lower()[:3], None)
                    year = int(match.group(4))
                elif fmt == 'dmy':
                    day = int(match.group(1))
                    month = months.get(match.group(2).lower()[:3], None)
                    year = int(match.group(3))
                elif fmt == 'range_mdy':
                    month = months.get(match.group(1).lower()[:3], None)
                    day = int(match.group(3))  # Take end date
                    year = int(match.group(4))
                elif fmt == 'my':
                    month = months.get(match.group(1).lower()[:3], None)
                    day = 15  # Default to mid-month
                    year = int(match.group(2))
                elif fmt == 'iso':
                    year = int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))

                if month and 1 <= month <= 12:
                    return (year, month, day)
            except (ValueError, AttributeError):
                continue

    return None


def extract_tables_from_html(html: str) -> List[pd.DataFrame]:
    """Extract polling tables from Wikipedia HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    tables = []

    # Find all wikitables
    for table in soup.find_all('table', class_='wikitable'):
        try:
            # Convert to pandas DataFrame
            from io import StringIO
            df = pd.read_html(StringIO(str(table)))[0]
            tables.append(df)
        except Exception as e:
            print(f"Warning: Could not parse table: {e}")
            continue

    return tables


def identify_polling_table(df: pd.DataFrame) -> Tuple[bool, str]:
    """Check if a DataFrame looks like a polling table.

    Returns (is_polling_table, table_type) where table_type is 'percentages' or 'seats'.
    """
    # Check for common polling table indicators
    columns_str = ' '.join(str(c).lower() for c in df.columns.tolist())

    indicators = ['date', 'polling', 'firm', 'smer', 'ps', 'hlas', 'sample']
    matches = sum(1 for ind in indicators if ind in columns_str)

    if matches < 2:
        return False, ''

    # Determine if this is a seats table (has Gov./Opp. columns or integer values)
    if 'gov' in columns_str or 'opp' in columns_str:
        return True, 'seats'

    # Check if values look like percentages (have decimals) or seats (integers)
    # Look at first few data rows and multiple party columns
    if len(df) > 0:
        party_keywords = ['smer', 'ps', 'hlas', 'kdh', 'sas', 'sns']
        decimal_count = 0
        integer_count = 0

        for col in df.columns:
            col_lower = str(col).lower()
            if any(pk in col_lower for pk in party_keywords):
                # Check first 3 rows for this column
                for row_idx in range(min(3, len(df))):
                    try:
                        val = df.iloc[row_idx][col]
                        if pd.notna(val):
                            val_str = str(val).replace('%', '').strip()
                            val_float = float(val_str)
                            # Check if it has a true decimal component
                            if '.' in val_str and val_float != int(val_float):
                                decimal_count += 1
                            elif val_float == int(val_float) and val_float > 10:
                                integer_count += 1
                    except (ValueError, TypeError):
                        pass

        # If we found any decimals, it's percentages
        # Seats are always whole numbers, percentages often have decimals
        if decimal_count > 0:
            return True, 'percentages'
        elif integer_count > 3:
            # Multiple large integers suggest seats
            return True, 'seats'

    return True, 'percentages'


def parse_integer(value: str) -> Optional[int]:
    """Parse integer value (for seats) from string."""
    if not value or value in ['–', '—', '-', 'N/A', '', '0']:
        return None

    value = value.strip()

    try:
        return int(float(value))
    except ValueError:
        return None


def parse_polling_table(df: pd.DataFrame, table_type: str = 'percentages') -> List[Dict]:
    """Parse a polling DataFrame into structured records.

    Args:
        df: DataFrame containing polling data
        table_type: 'percentages' for vote share, 'seats' for mandate projections
    """
    records = []

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(str(c) for c in col).strip() for col in df.columns]

    # Find column mappings
    date_col = None
    agency_col = None
    party_cols = {}

    for col in df.columns:
        col_lower = str(col).lower()

        if any(x in col_lower for x in ['date', 'fieldwork', 'datum']):
            date_col = col
        elif any(x in col_lower for x in ['polling firm', 'agency', 'institute', 'firm']):
            agency_col = col
        else:
            # Check if it's a party column
            for wiki_name, short_name in PARTY_MAPPING.items():
                if wiki_name.lower() in col_lower:
                    party_cols[col] = short_name
                    break

    if not date_col or not party_cols:
        return records

    # Parse each row
    for _, row in df.iterrows():
        date_parsed = parse_date(str(row.get(date_col, '')))
        if not date_parsed:
            continue

        year, month, day = date_parsed

        # Get agency
        agency = 'unknown'
        if agency_col and pd.notna(row.get(agency_col)):
            agency_raw = str(row[agency_col])
            for wiki_agency, short_agency in AGENCY_MAPPING.items():
                if wiki_agency.lower() in agency_raw.lower():
                    agency = short_agency
                    break

        # Get party data (percentages or seats depending on table_type)
        data = {}
        for col, party_name in party_cols.items():
            if table_type == 'seats':
                value = parse_integer(str(row.get(col, '')))
            else:
                value = parse_percentage(str(row.get(col, '')))
            if value is not None:
                data[party_name] = value

        if data:
            record = {
                'year': year,
                'month': month,
                'day': day,
                'agency': agency,
            }
            if table_type == 'seats':
                record['mandaty'] = data
            else:
                record['vysledky'] = data
            records.append(record)

    return records


def scrape_wikipedia_polls(url: str) -> List[Dict]:
    """Scrape polling data from Wikipedia page."""
    print(f"Fetching: {url}")
    html = fetch_wikipedia_page(url)

    print("Extracting tables...")
    tables = extract_tables_from_html(html)
    print(f"Found {len(tables)} tables")

    percentage_records = []
    seats_records = []

    for i, df in enumerate(tables):
        is_polling, table_type = identify_polling_table(df)
        if is_polling:
            print(f"Parsing {table_type} table {i+1}...")
            records = parse_polling_table(df, table_type)
            print(f"  Found {len(records)} records")
            if table_type == 'seats':
                seats_records.extend(records)
            else:
                percentage_records.extend(records)

    # Merge seats data into percentage records where they match
    merged_records = merge_percentage_and_seats(percentage_records, seats_records)

    return merged_records


def merge_percentage_and_seats(percentage_records: List[Dict], seats_records: List[Dict]) -> List[Dict]:
    """Merge seats data into percentage records based on matching date and agency."""
    # Create a lookup for seats by (year, month, agency)
    seats_lookup = {}
    for record in seats_records:
        key = (record['year'], record['month'], record['agency'])
        seats_lookup[key] = record.get('mandaty', {})

    # Merge into percentage records
    merged = []
    for record in percentage_records:
        key = (record['year'], record['month'], record['agency'])
        new_record = record.copy()
        if key in seats_lookup:
            new_record['mandaty'] = seats_lookup[key]
        merged.append(new_record)

    # Count how many got mandaty
    with_mandaty = sum(1 for r in merged if 'mandaty' in r)
    print(f"  Merged {with_mandaty} records with seat projections")

    return merged


def convert_to_data_format(records: List[Dict]) -> Dict:
    """Convert scraped records to the data/YYYY.json format."""
    data = {}

    for record in records:
        year = str(record['year'])
        month = f"{record['month']:02d}"
        agency = record['agency']

        if year not in data:
            data[year] = {}
        if month not in data[year]:
            data[year][month] = {}

        # If agency already exists for this month, update or skip
        if agency not in data[year][month]:
            data[year][month][agency] = {
                'vysledky': record.get('vysledky', {}),
                'mandaty': record.get('mandaty', None)
            }

    return data


def merge_with_existing(new_data: Dict, data_dir: str = "data") -> Dict:
    """Merge scraped data with existing data files."""
    merged = {}

    # Load existing data
    data_path = Path(data_dir)
    for json_file in data_path.glob("*.json"):
        if json_file.name in ['weights.json', 'input.json']:
            continue

        year = json_file.stem
        with open(json_file, 'r', encoding='utf-8') as f:
            merged[year] = json.load(f)

    # Merge new data
    for year, months in new_data.items():
        if year not in merged:
            merged[year] = {}

        for month, agencies in months.items():
            if month not in merged[year]:
                merged[year][month] = {}

            for agency, data in agencies.items():
                if agency not in merged[year][month]:
                    merged[year][month][agency] = data
                    print(f"  Added: {year}/{month}/{agency}")

    return merged


def save_data(data: Dict, data_dir: str = "data"):
    """Save data to year-based JSON files."""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)

    for year, year_data in data.items():
        file_path = data_path / f"{year}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(year_data, f, ensure_ascii=False, indent=2)
        print(f"Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Slovak election polls from Wikipedia")
    parser.add_argument("--url", type=str,
                        default="https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Slovak_parliamentary_election",
                        help="Wikipedia URL to scrape")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for data files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be scraped without saving")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing data files")
    parser.add_argument("--list-sources", action="store_true",
                        help="List available Wikipedia sources")
    args = parser.parse_args()

    if args.list_sources:
        print("Available Wikipedia sources:")
        print("  1. Next Slovak parliamentary election (current):")
        print("     https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Slovak_parliamentary_election")
        print("  2. 2023 Slovak parliamentary election:")
        print("     https://en.wikipedia.org/wiki/Opinion_polling_for_the_2023_Slovak_parliamentary_election")
        return

    # Scrape data
    records = scrape_wikipedia_polls(args.url)

    if not records:
        print("No polling data found!")
        return

    print(f"\nTotal records scraped: {len(records)}")

    # Convert to our format
    new_data = convert_to_data_format(records)

    if args.dry_run:
        print("\n--- DRY RUN ---")
        print(json.dumps(new_data, ensure_ascii=False, indent=2))
        return

    if args.merge:
        print("\nMerging with existing data...")
        data = merge_with_existing(new_data, args.output_dir)
    else:
        data = new_data

    # Save
    save_data(data, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
