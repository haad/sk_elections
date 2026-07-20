# Claude Code Guidelines for sk_elections

## Project Overview

Slovak election polling analysis tool that scrapes Wikipedia for poll data, applies trend analysis, and generates seat projections for the Slovak parliament (150 seats).

## Directory Structure

```
sk_elections/
├── data/                    # Poll data in JSON format
│   ├── YYYY.json           # Polls by year (2023, 2024, 2025)
│   ├── weights.json        # Seat calculation parameters
│   └── scenarios.json      # What-if analysis scenarios
├── docs/                   # Generated HTML site
├── scraper.py             # Wikipedia poll scraper
├── trend_analysis.py      # Main analysis script
├── generate_html.py       # HTML site generator
└── _/                     # Legacy/experimental scripts (ignore)
```

## Data Format

### Poll Data (data/YYYY.json)

```json
{
  "MM": {
    "agency": {
      "vysledky": {
        "PARTY": percentage
      },
      "mandaty": {
        "PARTY": seats
      }
    }
  }
}
```

- `MM` = month (01-12)
- `vysledky` = poll percentages (0-100)
- `mandaty` = seat projections (sum should equal 150)

### Party Codes

| Code | Party |
|------|-------|
| PS | Progresívne Slovensko |
| SMER | Smer-SD |
| HLAS | Hlas-SD |
| SLOV | Slovensko (Slovakia movement) |
| REP | Republika |
| SAS | Sloboda a Solidarita |
| KDH | Kresťanskodemokratické hnutie |
| SNS | Slovenská národná strana |
| DEM | Demokrati |
| Aliancia | Magyar Szövetség |
| ROD | Sme Rodina |

### Polling Agencies

`ako`, `focus`, `median`, `nms`, `ipsos`, `polis`

## Key Rules

### Electoral System

- Slovak parliament has **150 seats**
- **5% threshold** required to enter parliament
- Parties below 5% get **zero seats** - their votes are redistributed
- Seat projections must sum to exactly 150

### Scraper (scraper.py)

Wikipedia source: `https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Slovak_parliamentary_election`

When modifying the scraper:
- Wikipedia tables have multi-level headers - handle column flattening carefully
- Percentage tables have decimals (23.6%), seats tables have integers (40)
- Skip columns: ĽSNS, Others, Lead
- Map "OĽaNO and Friends" → SLOV for seat projections
- Map "Republic" → REP (Wikipedia uses English name in seats table)
- Use word boundary matching to avoid partial matches (SNS vs ĽSNS)

Run scraper:
```bash
.venv/bin/python scraper.py --dry-run    # Preview changes
.venv/bin/python scraper.py --merge      # Update data files
```

### Data Validation

After any data changes, validate:
1. All JSON files parse correctly
2. Seat projections sum to 150
3. Percentages are in valid range (0-100)
4. Party codes are consistent

```bash
.venv/bin/python -c "
import json
from pathlib import Path
for f in Path('data').glob('[0-9]*.json'):
    data = json.load(open(f))
    for month, polls in data.items():
        for agency, poll in polls.items():
            if 'mandaty' in poll and poll['mandaty']:
                total = sum(poll['mandaty'].values())
                if abs(total - 150) > 5:
                    print(f'{f.stem}/{month}/{agency}: {total} seats')
"
```

## Workflow

1. **Fetching new polls**: Run `scraper.py --merge`
2. **Validate data**: Check seat totals sum to 150
3. **Run analysis**: `trend_analysis.py`
4. **Generate site**: `generate_html.py`
5. **Commit**: Include both data and code changes

## Common Issues

### Scraper finds 0 records
- Check `identify_polling_table()` - it may misclassify percentages as seats
- Tables with all integer-like values (17.0) can be misidentified

### Wrong party values
- Check `PARTY_MAPPING` for missing Wikipedia names
- Avoid single-letter mappings that match too broadly
- Check column order - multi-level headers can confuse matching

### Seat totals don't sum to 150
- A party above 5% is missing from mandaty
- Check if Wikipedia table structure changed
- Verify party name mapping for seats table (uses different names)
