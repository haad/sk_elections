
import argparse
import json
import os
from pathlib import Path
from typing import Dict
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

class ModelParams:
    def __init__(self, quota_offset=0.0, remainder_weight=1.0, threshold=5.0, favor_small_parties=False):
        self.quota_offset = quota_offset
        self.remainder_weight = remainder_weight
        self.threshold = threshold
        self.favor_small_parties = favor_small_parties

    def to_dict(self):
        return {
            "quota_offset": self.quota_offset,
            "remainder_weight": self.remainder_weight,
            "threshold": self.threshold,
            "favor_small_parties": self.favor_small_parties
        }

    @staticmethod
    def from_dict(data):
        return ModelParams(
            quota_offset=data.get("quota_offset", 0.0),
            remainder_weight=data.get("remainder_weight", 1.0),
            threshold=data.get("threshold", 5.0),
            favor_small_parties=data.get("favor_small_parties", False)
        )

def load_polls_from_folder(folder_path="data") -> Dict:
    combined_polls = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and "weights" not in filename and "test" not in filename:
            year = os.path.splitext(filename)[0]
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                combined_polls[year] = data
    return combined_polls

def calculate_mandates(results_percent: Dict[str, float], params: ModelParams,
                       total_votes=1_000_000, total_mandates=150) -> Dict[str, int]:
    votes = {party: percent / 100 * total_votes for party, percent in results_percent.items()}
    above_threshold = {p: v for p, v in votes.items() if results_percent[p] >= params.threshold}
    valid_votes = sum(above_threshold.values())

    quota = valid_votes / (total_mandates + 1)
    quota *= (1 + params.quota_offset)

    mandates = {}
    remainders = {}
    assigned = 0

    for party, v in above_threshold.items():
        m = int(v // quota)
        mandates[party] = m
        assigned += m

        percent = results_percent[party]
        small_party_bonus = 1.0
        if params.favor_small_parties:
            small_party_bonus = 1 + (max(0, 10 - percent) / 100)

        remainders[party] = (v - m * quota) * params.remainder_weight * small_party_bonus

    remaining = total_mandates - assigned
    sorted_remainders = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
    for i in range(remaining):
        mandates[sorted_remainders[i][0]] += 1

    return mandates

def evaluate_model(polls: Dict, params: ModelParams):
    errors = []

    for year, months in polls.items():
        for month, agencies in months.items():
            if not isinstance(agencies, dict):
                print(f"⚠️  Skipping invalid data in {year}/{month} – expected dict, got {type(agencies)}")
                continue
            for agency, data in agencies.items():
                if not isinstance(data, dict) or 'mandaty' not in data or 'vysledky' not in data:
                    continue
                input_data = data['vysledky']
                expected = data['mandaty']
                prediction = calculate_mandates(input_data, params)

                all_parties = set(expected.keys()) | set(prediction.keys())
                real = [expected.get(p, 0) for p in all_parties]
                pred = [prediction.get(p, 0) for p in all_parties]
                errors.append(mean_absolute_error(real, pred))

    return np.mean(errors) if errors else 9999

def loss_fn(x, polls):
    params = ModelParams(
        quota_offset=x[0],
        remainder_weight=x[1],
        threshold=5.0,
        favor_small_parties=True
    )
    return evaluate_model(polls, params)

def summarize_blocks(mandates: Dict[str, int]):
    government = {"SMER", "SNS", "HLAS", "REP"}
    opposition = {"PS", "SAS", "DEM", "SLOV", "KDH"}
    catholic = {"DEM", "KDH", "SLOV", "ROD"}

    sum_government = sum(mandates.get(p, 0) for p in government)
    sum_opposition = sum(mandates.get(p, 0) for p in opposition)
    sum_catholic = sum(mandates.get(p, 0) for p in catholic)

    print("\nBlock Summary:")
    print(f"  Government block: {sum_government} mandates")
    print(f"  Opposition block: {sum_opposition} mandates")
    print(f"  Catholic block:   {sum_catholic} mandates")

    if sum_government >= 90:
        print("  ✅ Government block can change the constitution.")
    elif sum_government >= 76:
        print("  ✅ Government block can form a majority.")
    else:
        print("  ❌ Government block cannot form a majority.")

def main():
    parser = argparse.ArgumentParser(description="Slovak Parliament Mandate Allocator with Optimizer")
    parser.add_argument("mode", choices=["training", "simulation"], help="Mode to run: training or simulation")
    parser.add_argument("--weights", type=str, help="Path to weights.json file for simulation")
    parser.add_argument("--input", type=str, help="Path to input JSON file with party percentages")
    args = parser.parse_args()

    if args.mode == "training":
        polls = load_polls_from_folder("data")
        print("🔍 Optimizing parameters...")
        bounds = [(-0.05, 0.05), (0.5, 2.0)]
        initial_guess = [0.0, 1.0]
        result = minimize(loss_fn, initial_guess, args=(polls,), bounds=bounds, method='L-BFGS-B')

        if result.success:
            best = ModelParams(quota_offset=result.x[0], remainder_weight=result.x[1], favor_small_parties=True)
            print("✅ Optimization complete!")
            print("Best parameters found:")
            print(json.dumps(best.to_dict(), indent=2))
            Path("data").mkdir(parents=True, exist_ok=True)
            with open("data/weights.json", "w", encoding="utf-8") as f:
                json.dump(best.to_dict(), f, ensure_ascii=False, indent=2)
        else:
            print("❌ Optimization failed.")
            print(result)

    elif args.mode == "simulation":
        if not args.weights or not args.input:
            print("❌ Simulation mode requires both --weights and --input.")
            return
        with open(args.weights, "r", encoding="utf-8") as f:
            params = ModelParams.from_dict(json.load(f))
        with open(args.input, "r", encoding="utf-8") as f:
            percent_data = json.load(f)
        mandates = calculate_mandates(percent_data, params)
        print(json.dumps(mandates, ensure_ascii=False, indent=2))
        summarize_blocks(mandates)

if __name__ == "__main__":
    main()
