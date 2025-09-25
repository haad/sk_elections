
# Slovak Parliament Mandate Allocator 🇸🇰

This Python project approximates the allocation of parliamentary seats (mandates) in the Slovak National Council using polling data. It implements a modified Hagenbach-Bischoff method, enhanced with parameter tuning and simulation capabilities.

---

## 🧠 Purpose

The goal is to simulate and tune a mandate calculation model that closely reflects real-world predictions published by Slovak media such as [SME.sk](https://sme.sk) and [Denník N](https://dennikn.sk).

---

## 📁 Project Structure

```
data/
├── 2025.json            # Polling data by month/agency
├── 2024.json            # Older polling data
├── weights.json         # Auto-generated model weights after training
├── input.json           # Example input for simulation mode
self_learning_model_cli_with_optimizer.py
```

---

## ⚙️ Features

- ✅ Training mode with `scipy.optimize` for best parameter fitting.
- ✅ Simulation mode for user-defined voter preference inputs.
- ✅ Block summarization: Government / Opposition / Catholic parties.
- ✅ Fault-tolerant data loading and mandate calculation.
- ✅ Customizable quota/remainder weights.

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
- `scipy`, `sklearn`, `numpy`

Install with:

```bash
pip install -r requirements.txt
```

---

## 🧪 Next Ideas

- Add visualizations (convergence, heatmaps)
- Per-agency error tracking
- Support CSV/HTML export
- Web UI for interactive simulations

---

## 📝 License

MIT License

---

Made with ❤️ to understand Slovak politics better.
