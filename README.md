
# Slovak Parliament Mandate Allocator 🇸🇰

This Python project approximates the allocation of parliamentary seats (mandates) in the Slovak National Council using polling data. It implements a modified Hagenbach-Bischoff method, enhanced with parameter tuning, simulation capabilities, trend analysis, and automated data scraping.

---

## 🧠 Purpose

The goal is to simulate and tune a mandate calculation model that closely reflects real-world predictions published by Slovak media such as [SME.sk](https://sme.sk) and [Denník N](https://dennikn.sk).

---

## 📁 Project Structure

```
data/
├── 2023.json            # Historical polling data
├── 2024.json            # Polling data by month/agency
├── 2025.json            # Current polling data
├── weights.json         # Auto-generated model weights after training
├── input.json           # Example input for simulation mode
sk_election_model.py     # Main model for training and simulation
trend_analysis.py        # Trend analysis with Kalman filtering
scraper.py               # Wikipedia poll scraper
```

---

## ⚙️ Features

- ✅ Training mode with `scipy.optimize` for best parameter fitting
- ✅ Simulation mode for user-defined voter preference inputs
- ✅ Block summarization: Government / Opposition / Catholic parties
- ✅ Fault-tolerant data loading and mandate calculation
- ✅ Customizable quota/remainder weights
- ✅ Trend analysis with linear regression and Kalman filtering
- ✅ Statistical analysis: correlations, volatility, Monte Carlo simulation
- ✅ Automated Wikipedia scraper for polling data (percentages and seat projections)

---

## 🚀 Usage

### 1. 📊 Train the model on real-world polls

```bash
python sk_election_model.py training
```

- Trains the model using all polls in the `data/` folder.
- Saves the optimized weights to `data/weights.json`.

---

### 2. 🔮 Simulate mandates for hypothetical voter preferences

Prepare a JSON file like `data/input.json`:

```json
{
  "PS": 22.5,
  "SMER": 20,
  "REP": 10.3,
  "HLAS": 9.8,
  "SLOV": 7.5,
  "SAS": 7.1,
  "KDH": 6,
  "DEM": 5.2,
  "Aliancia": 3.6,
  "SNS": 2.7,
  "ROD": 2
}
```

Then run:

```bash
python sk_election_model.py simulation --weights data/weights.json --input data/input.json
```

The output will show:
- Party mandates
- Block summaries (government, opposition, catholic)
- Majority/constitution status

---

### 3. 📈 Trend Analysis

Analyze polling trends with linear regression or Kalman filtering:

```bash
# Basic trend analysis
python trend_analysis.py

# With Kalman filtering for smoother trends
python trend_analysis.py --kalman

# Show interactive graphs
python trend_analysis.py --graph

# Save graphs to output directory
python trend_analysis.py --graph --output-dir output/

# Full statistical analysis
python trend_analysis.py --full-stats --graph
```

#### Trend Analysis Options

| Flag | Description |
|------|-------------|
| `--kalman` | Use Kalman filtering instead of linear regression |
| `--graph` | Generate graphs (interactive or saved to `--output-dir`) |
| `--output-dir DIR` | Save graphs to specified directory |
| `--forecast N` | Forecast N months ahead (default: 6) |
| `--threshold N` | Only show parties above N% (default: 5.0) |
| `--correlations` | Show party correlation analysis |
| `--volatility` | Show polling volatility metrics |
| `--monte-carlo` | Run Monte Carlo seat simulation |
| `--scenarios` | Run scenario (what-if) analysis |
| `--full-stats` | Run all statistical analyses |

---

### 4. 🌐 Scrape Polling Data from Wikipedia

Automatically fetch the latest polling data from Wikipedia:

```bash
# Preview what would be scraped (dry run)
python scraper.py --dry-run

# Scrape and merge with existing data
python scraper.py --merge

# List available Wikipedia sources
python scraper.py --list-sources
```

The scraper fetches both vote percentages (`vysledky`) and seat projections (`mandaty`) from:
- [Opinion polling for the next Slovak parliamentary election](https://en.wikipedia.org/wiki/Opinion_polling_for_the_next_Slovak_parliamentary_election)

---

## 🧱 Political Blocks

Block classification used in simulation:

- **Government**: SMER, SNS, HLAS, REP
- **Opposition**: PS, SAS, DEM, SLOV, KDH
- **Catholic**: DEM, KDH, SLOV, ROD

---

## 🗃️ Data Format for Polling Files

Each file like `2025.json` contains:

```json
{
  "08": {
    "ipsos": {
      "vysledky": { "PS": 22.5, "SMER": 20, ... },
      "mandaty":  { "PS": 38, "SMER": 34, ... }
    },
    ...
  }
}
```

If `mandaty` are missing, that poll will be ignored during training.

---

## 📌 Requirements

- Python 3.9+
- `scipy`, `sklearn`, `numpy`, `matplotlib`
- For scraper: `requests`, `beautifulsoup4`, `pandas`, `lxml`

Install with:

```bash
pip install -r requirements.txt

# Or with uv
uv pip install scipy scikit-learn numpy matplotlib requests beautifulsoup4 pandas lxml
```

---

## 🧪 Next Ideas

- Per-agency error tracking
- Support CSV/HTML export
- Web UI for interactive simulations
- Additional data sources for scraping

---

## 📝 License

MIT License

---

Made with ❤️ to understand Slovak politics better.
