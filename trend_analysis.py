import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class KalmanFilter:
    """
    Kalman filter for poll tracking.

    State: [support_percentage, trend_velocity]
    - support_percentage: current estimated true support
    - trend_velocity: rate of change per time unit (month)
    """

    def __init__(self, initial_state: float, process_noise: float = 0.5,
                 measurement_noise: float = 2.0, initial_velocity: float = 0.0):
        """
        Initialize Kalman filter.

        Args:
            initial_state: Initial estimate of support percentage
            process_noise: How much we expect true support to change (higher = more volatile)
            measurement_noise: How noisy polls are (higher = less trust in individual polls)
            initial_velocity: Initial trend velocity
        """
        # State vector [support, velocity]
        self.x = np.array([initial_state, initial_velocity])

        # State covariance matrix (uncertainty in our estimates)
        self.P = np.array([
            [4.0, 0.0],   # Initial uncertainty in support
            [0.0, 0.25]   # Initial uncertainty in velocity
        ])

        # Process noise covariance (how much true state changes between measurements)
        self.Q = np.array([
            [process_noise, 0.0],
            [0.0, process_noise * 0.1]
        ])

        # Measurement noise variance
        self.R = measurement_noise ** 2

        # Measurement matrix (we only observe support, not velocity)
        self.H = np.array([[1.0, 0.0]])

        self.history = []

    def predict(self, dt: float = 1.0):
        """
        Predict step: project state forward in time.

        Args:
            dt: Time step (in months)
        """
        # State transition matrix
        F = np.array([
            [1.0, dt],   # support = support + velocity * dt
            [0.0, 1.0]   # velocity stays same (random walk)
        ])

        # Predict state
        self.x = F @ self.x

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q * dt

    def update(self, measurement: float, measurement_noise: Optional[float] = None):
        """
        Update step: incorporate new poll measurement.

        Args:
            measurement: Observed poll percentage
            measurement_noise: Override default measurement noise (for pollster weighting)
        """
        R = (measurement_noise ** 2) if measurement_noise else self.R

        # Innovation (measurement residual)
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + R

        # Kalman gain
        K = self.P @ self.H.T / S

        # Update state
        self.x = self.x + K.flatten() * y

        # Update covariance
        self.P = (np.eye(2) - K @ self.H) @ self.P

        # Store history
        self.history.append({
            'state': self.x.copy(),
            'covariance': self.P.copy(),
            'support': self.x[0],
            'velocity': self.x[1],
            'uncertainty': np.sqrt(self.P[0, 0])
        })

    def get_state(self) -> Tuple[float, float, float]:
        """Return current (support, velocity, uncertainty)."""
        return self.x[0], self.x[1], np.sqrt(self.P[0, 0])

    def forecast(self, months_ahead: int, velocity_decay: float = 0.85) -> List[Tuple[float, float]]:
        """
        Forecast future values with uncertainty.

        Args:
            months_ahead: Number of months to forecast
            velocity_decay: Decay factor for velocity (0.85 = velocity reduces by 15% each month)
                           This prevents unrealistic linear extrapolation

        Returns: List of (predicted_support, uncertainty) tuples
        """
        forecasts = []
        x_pred = self.x.copy()
        P_pred = self.P.copy()

        for i in range(months_ahead):
            # Decay velocity over time (mean reversion)
            decay = velocity_decay ** (i + 1)

            F = np.array([
                [1.0, decay],  # support = support + decayed_velocity
                [0.0, velocity_decay]  # velocity decays
            ])

            x_pred = F @ x_pred
            # Reduce process noise growth for forecasts
            P_pred = F @ P_pred @ F.T + self.Q * 0.5
            forecasts.append((x_pred[0], np.sqrt(P_pred[0, 0])))

        return forecasts


def run_kalman_filter(records: List[Dict], party: str,
                      process_noise: float = 0.3,
                      measurement_noise: float = 2.0) -> Tuple[KalmanFilter, List[Dict]]:
    """
    Run Kalman filter on poll data for a specific party.

    Returns: (filter, filtered_results)
    """
    # Get records for this party, sorted by date
    party_records = sorted(
        [r for r in records if r['party'] == party],
        key=lambda x: (x['date'], x['agency'])
    )

    if not party_records:
        return None, []

    # Initialize filter with first measurement
    initial_value = party_records[0]['percent']
    kf = KalmanFilter(initial_value, process_noise, measurement_noise)

    results = []
    current_date = party_records[0]['date']

    for record in party_records:
        # Calculate time delta in months
        dt = (record['date'] - current_date).days / 30.0

        if dt > 0:
            kf.predict(dt)
            current_date = record['date']

        # Update with measurement
        kf.update(record['percent'])

        support, velocity, uncertainty = kf.get_state()
        results.append({
            'date': record['date'],
            'agency': record['agency'],
            'raw_poll': record['percent'],
            'filtered_support': support,
            'velocity': velocity,
            'uncertainty': uncertainty
        })

    return kf, results


def calculate_kalman_averages(records: List[Dict],
                               process_noise: float = 0.3,
                               measurement_noise: float = 2.0) -> Dict[str, List[Tuple[datetime, float, float]]]:
    """
    Run Kalman filter for all parties and return filtered time series.

    Returns: Dict mapping party -> List of (date, filtered_value, uncertainty)
    """
    parties = set(r['party'] for r in records)
    party_series = {}

    for party in parties:
        kf, results = run_kalman_filter(records, party, process_noise, measurement_noise)
        if results:
            # Group by date and take last filtered value for each date
            by_date = {}
            for r in results:
                by_date[r['date']] = (r['filtered_support'], r['uncertainty'])

            party_series[party] = [
                (date, val, unc) for date, (val, unc) in sorted(by_date.items())
            ]

    return party_series


def kalman_forecast(records: List[Dict], party: str, months_ahead: int = 6,
                    process_noise: float = 0.3,
                    measurement_noise: float = 2.0) -> List[Tuple[datetime, float, float]]:
    """
    Get Kalman filter forecast for a party.

    Returns: List of (date, predicted_support, uncertainty)
    """
    kf, results = run_kalman_filter(records, party, process_noise, measurement_noise)
    if not kf or not results:
        return []

    last_date = results[-1]['date']
    forecasts = kf.forecast(months_ahead)

    forecast_results = []
    for i, (support, uncertainty) in enumerate(forecasts):
        future_date = datetime(
            last_date.year + (last_date.month + i) // 12,
            (last_date.month + i) % 12 + 1,
            15
        )
        forecast_results.append((future_date, support, uncertainty))

    return forecast_results


###############################################################################
# STATISTICAL ANALYSIS FUNCTIONS
###############################################################################

def calculate_party_correlations(records: List[Dict]) -> Tuple[List[str], np.ndarray]:
    """
    Calculate correlation matrix between parties based on poll movements.

    Returns: (party_names, correlation_matrix)
    """
    # Build time series for each party
    parties = sorted(set(r['party'] for r in records))

    # Group by date and calculate average per party per date
    date_party_values = {}
    for r in records:
        date = r['date']
        if date not in date_party_values:
            date_party_values[date] = {}
        if r['party'] not in date_party_values[date]:
            date_party_values[date][r['party']] = []
        date_party_values[date][r['party']].append(r['percent'])

    # Average values per date
    dates = sorted(date_party_values.keys())
    party_series = {p: [] for p in parties}

    for date in dates:
        for party in parties:
            if party in date_party_values[date]:
                party_series[party].append(np.mean(date_party_values[date][party]))
            else:
                party_series[party].append(np.nan)

    # Convert to numpy array and calculate correlations
    data_matrix = np.array([party_series[p] for p in parties])

    # Calculate correlation matrix (handling NaN values)
    n_parties = len(parties)
    corr_matrix = np.zeros((n_parties, n_parties))

    for i in range(n_parties):
        for j in range(n_parties):
            mask = ~(np.isnan(data_matrix[i]) | np.isnan(data_matrix[j]))
            if mask.sum() > 2:
                corr_matrix[i, j] = np.corrcoef(data_matrix[i][mask], data_matrix[j][mask])[0, 1]
            else:
                corr_matrix[i, j] = np.nan

    return parties, corr_matrix


def get_filtered_correlation_data(records: List[Dict], threshold: float = 5.0):
    """Get correlation matrix filtered to parties above threshold."""
    parties, corr_matrix = calculate_party_correlations(records)

    # Filter to parties above threshold
    kalman_series = calculate_kalman_averages(records)
    main_parties = []
    for party in parties:
        if party in kalman_series and kalman_series[party]:
            current = kalman_series[party][-1][1]
            if current >= threshold:
                main_parties.append(party)

    # Build filtered correlation matrix
    n = len(main_parties)
    filtered_matrix = np.zeros((n, n))
    for i, p1 in enumerate(main_parties):
        for j, p2 in enumerate(main_parties):
            idx_i = parties.index(p1)
            idx_j = parties.index(p2)
            filtered_matrix[i, j] = corr_matrix[idx_i, idx_j]

    return main_parties, filtered_matrix


def print_correlations(records: List[Dict], threshold: float = 5.0):
    """Print party correlation analysis."""
    print("\n" + "=" * 70)
    print("PARTY CORRELATIONS")
    print("=" * 70)
    print("\nPositive = move together, Negative = inverse (voters switching)")

    parties, corr_matrix = calculate_party_correlations(records)

    # Filter to parties above threshold
    kalman_series = calculate_kalman_averages(records)
    main_parties = []
    for party in parties:
        if party in kalman_series and kalman_series[party]:
            current = kalman_series[party][-1][1]
            if current >= threshold:
                main_parties.append(party)

    # ANSI color codes
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def colorize(val):
        if np.isnan(val):
            return f"{'N/A':>8}"
        if val == 1.0:
            return f"{BOLD}{'1.00':>8}{RESET}"
        if val >= 0.5:
            return f"{GREEN}{val:>8.2f}{RESET}"
        elif val <= -0.5:
            return f"{RED}{val:>8.2f}{RESET}"
        elif val >= 0.3 or val <= -0.3:
            return f"{YELLOW}{val:>8.2f}{RESET}"
        else:
            return f"{val:>8.2f}"

    # Print correlation matrix with colors
    print(f"\n{'':12}", end="")
    for p in main_parties:
        print(f"{p:>8}", end="")
    print()
    print("-" * (12 + 8 * len(main_parties)))

    for i, p1 in enumerate(main_parties):
        print(f"{p1:<12}", end="")
        for j, p2 in enumerate(main_parties):
            idx_i = parties.index(p1)
            idx_j = parties.index(p2)
            corr = corr_matrix[idx_i, idx_j]
            print(colorize(corr), end="")
        print()

    # Find strongest positive and negative correlations
    print("\nStrongest correlations:")
    correlations = []
    for i, p1 in enumerate(main_parties):
        for j, p2 in enumerate(main_parties):
            if i < j:
                idx_i = parties.index(p1)
                idx_j = parties.index(p2)
                corr = corr_matrix[idx_i, idx_j]
                if not np.isnan(corr):
                    correlations.append((p1, p2, corr))

    correlations.sort(key=lambda x: x[2], reverse=True)

    # Get velocity/trend for each party
    party_trends = {}
    for party in main_parties:
        kf, results = run_kalman_filter(records, party)
        if kf:
            _, velocity, _ = kf.get_state()
            party_trends[party] = velocity

    # Positive correlations - moving together
    print(f"\n  {GREEN}Moving together (positive correlation + trend):{RESET}")
    for p1, p2, corr in correlations[:5]:
        if corr > 0.3:  # Only significant positive correlations
            trend1 = party_trends.get(p1, 0)
            trend2 = party_trends.get(p2, 0)
            avg_trend = (trend1 + trend2) / 2

            if avg_trend > 0.1:
                direction = "both rising ↑"
            elif avg_trend < -0.1:
                direction = "both falling ↓"
            else:
                direction = "both stable →"

            print(f"    {p1} ↔ {p2}: {GREEN}{corr:+.2f}{RESET} "
                  f"({direction}: {p1} {trend1:+.2f}, {p2} {trend2:+.2f}%/mo)")

    # Negative correlations - voter flow
    print(f"\n  {RED}Voter flow (negative correlation + trend):{RESET}")

    for p1, p2, corr in correlations[-5:]:  # Show more negative correlations
        if corr < -0.3:  # Only significant negative correlations
            trend1 = party_trends.get(p1, 0)
            trend2 = party_trends.get(p2, 0)

            # Determine flow direction: from falling party to rising party
            if trend1 < trend2:
                # p1 falling, p2 rising → flow from p1 to p2
                from_party, to_party = p1, p2
                from_trend, to_trend = trend1, trend2
            else:
                # p2 falling, p1 rising → flow from p2 to p1
                from_party, to_party = p2, p1
                from_trend, to_trend = trend2, trend1

            print(f"    {from_party} → {to_party}: {RED}{corr:.2f}{RESET} "
                  f"('{from_party}' {from_trend:+.2f}%/mo, '{to_party}' {to_trend:+.2f}%/mo)")


def plot_correlations(records: List[Dict], output_file: str = None, threshold: float = 5.0):
    """Plot correlation matrix as a heatmap."""
    main_parties, corr_matrix = get_filtered_correlation_data(records, threshold)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Correlation", rotation=-90, va="bottom", fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(main_parties)))
    ax.set_yticks(np.arange(len(main_parties)))
    ax.set_xticklabels(main_parties, fontsize=11)
    ax.set_yticklabels(main_parties, fontsize=11)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add correlation values as text
    for i in range(len(main_parties)):
        for j in range(len(main_parties)):
            val = corr_matrix[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            text = ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                          color=color, fontsize=10, fontweight='bold')

    ax.set_title("Party Correlation Matrix\n(Red = negative/switching, Blue = positive/together)",
                 fontsize=14, fontweight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(main_parties) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(main_parties) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Correlation chart saved to: {output_file}")
    else:
        plt.show()

    plt.close()


def calculate_volatility(records: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate volatility metrics for each party.

    Returns dict with: std_dev, coef_variation, range, trend_stability
    """
    parties = set(r['party'] for r in records)
    volatility = {}

    for party in parties:
        party_records = [r for r in records if r['party'] == party]
        if len(party_records) < 3:
            continue

        values = [r['percent'] for r in party_records]

        # Basic statistics
        mean_val = np.mean(values)
        std_dev = np.std(values)
        coef_var = (std_dev / mean_val * 100) if mean_val > 0 else 0
        val_range = max(values) - min(values)

        # Month-to-month changes
        sorted_records = sorted(party_records, key=lambda x: x['date'])
        monthly_changes = []
        prev_val = None
        prev_date = None
        for r in sorted_records:
            if prev_val is not None and prev_date != r['date']:
                monthly_changes.append(abs(r['percent'] - prev_val))
            prev_val = r['percent']
            prev_date = r['date']

        avg_change = np.mean(monthly_changes) if monthly_changes else 0

        volatility[party] = {
            'mean': mean_val,
            'std_dev': std_dev,
            'coef_variation': coef_var,
            'range': val_range,
            'min': min(values),
            'max': max(values),
            'avg_monthly_change': avg_change
        }

    return volatility


def print_volatility(records: List[Dict], threshold: float = 5.0):
    """Print volatility analysis."""
    print("\n" + "=" * 70)
    print("VOLATILITY ANALYSIS")
    print("=" * 70)
    print("\nHigher volatility = less predictable, higher forecast uncertainty")

    volatility = calculate_volatility(records)

    # Filter and sort by coefficient of variation
    filtered = [(p, v) for p, v in volatility.items() if v['mean'] >= threshold]
    filtered.sort(key=lambda x: x[1]['coef_variation'], reverse=True)

    print(f"\n{'Party':<10} {'Mean':>8} {'Std Dev':>8} {'CV %':>8} {'Range':>8} {'Min-Max':>12} {'Stability':<12}")
    print("-" * 75)

    for party, v in filtered:
        stability = "Very Stable" if v['coef_variation'] < 5 else \
                   "Stable" if v['coef_variation'] < 10 else \
                   "Moderate" if v['coef_variation'] < 15 else \
                   "Volatile" if v['coef_variation'] < 20 else "Very Volatile"

        print(f"{party:<10} {v['mean']:>7.1f}% {v['std_dev']:>7.2f}% {v['coef_variation']:>7.1f}% "
              f"{v['range']:>7.1f}% {v['min']:>5.1f}-{v['max']:>5.1f}% {stability:<12}")

    print("\nCV% = Coefficient of Variation (std dev / mean × 100)")


def monte_carlo_simulation(records: List[Dict], params, n_simulations: int = 10000,
                           threshold: float = 5.0) -> Dict:
    """
    Run Monte Carlo simulation for seat projections.

    Uses Kalman filter uncertainty estimates to sample possible outcomes.
    """
    from sk_election_model import calculate_mandates, ModelParams

    # Get current Kalman estimates with uncertainty
    parties = set(r['party'] for r in records)
    estimates = {}

    for party in parties:
        kf, results = run_kalman_filter(records, party)
        if kf and results:
            support, velocity, uncertainty = kf.get_state()
            estimates[party] = {
                'mean': support,
                'std': uncertainty
            }

    # Run simulations
    government = {"SMER", "SNS", "HLAS", "REP"}
    opposition = {"PS", "SAS", "DEM", "SLOV", "KDH"}

    gov_seats = []
    opp_seats = []
    party_seats = {p: [] for p in estimates.keys()}
    threshold_passes = {p: 0 for p in estimates.keys()}

    for _ in range(n_simulations):
        # Sample poll values from uncertainty distribution
        sampled_polls = {}
        for party, est in estimates.items():
            # Sample from normal distribution, clamp to valid range
            sample = np.random.normal(est['mean'], est['std'])
            sample = max(0, min(100, sample))
            sampled_polls[party] = sample

        # Calculate seats
        try:
            seats = calculate_mandates(sampled_polls, params)
        except:
            continue

        # Track results
        gov_total = sum(seats.get(p, 0) for p in government)
        opp_total = sum(seats.get(p, 0) for p in opposition)

        gov_seats.append(gov_total)
        opp_seats.append(opp_total)

        for party in estimates.keys():
            party_seats[party].append(seats.get(party, 0))
            if sampled_polls[party] >= threshold:
                threshold_passes[party] += 1

    return {
        'gov_seats': np.array(gov_seats),
        'opp_seats': np.array(opp_seats),
        'party_seats': {p: np.array(s) for p, s in party_seats.items()},
        'threshold_prob': {p: c / n_simulations for p, c in threshold_passes.items()},
        'n_simulations': n_simulations
    }


def print_monte_carlo(records: List[Dict], params, n_simulations: int = 10000,
                      threshold: float = 5.0):
    """Print Monte Carlo simulation results."""
    print("\n" + "=" * 70)
    print(f"MONTE CARLO SIMULATION ({n_simulations:,} simulations)")
    print("=" * 70)

    results = monte_carlo_simulation(records, params, n_simulations, threshold)

    gov_seats = results['gov_seats']
    opp_seats = results['opp_seats']

    # Coalition probabilities
    gov_majority = np.mean(gov_seats >= 76) * 100
    gov_constitutional = np.mean(gov_seats >= 90) * 100
    opp_majority = np.mean(opp_seats >= 76) * 100
    opp_constitutional = np.mean(opp_seats >= 90) * 100

    print("\nCoalition Probabilities:")
    print("-" * 40)
    print(f"  Government majority (≥76):        {gov_majority:>5.1f}%")
    print(f"  Government constitutional (≥90):  {gov_constitutional:>5.1f}%")
    print(f"  Opposition majority (≥76):        {opp_majority:>5.1f}%")
    print(f"  Opposition constitutional (≥90):  {opp_constitutional:>5.1f}%")

    print("\nSeat Projections (median [5th-95th percentile]):")
    print("-" * 50)
    print(f"  Government: {np.median(gov_seats):>3.0f} seats [{np.percentile(gov_seats, 5):>3.0f} - {np.percentile(gov_seats, 95):>3.0f}]")
    print(f"  Opposition: {np.median(opp_seats):>3.0f} seats [{np.percentile(opp_seats, 5):>3.0f} - {np.percentile(opp_seats, 95):>3.0f}]")

    # Party-level results
    print("\nParty Seat Projections:")
    print("-" * 60)
    print(f"{'Party':<10} {'Median':>8} {'5th %':>8} {'95th %':>8} {'P(≥5%)':>10}")
    print("-" * 60)

    # Sort by median seats
    party_results = []
    for party, seats in results['party_seats'].items():
        if len(seats) > 0:
            median = np.median(seats)
            p5 = np.percentile(seats, 5)
            p95 = np.percentile(seats, 95)
            prob = results['threshold_prob'].get(party, 0) * 100
            party_results.append((party, median, p5, p95, prob))

    party_results.sort(key=lambda x: x[1], reverse=True)

    for party, median, p5, p95, prob in party_results:
        if median > 0 or prob > 50:
            print(f"{party:<10} {median:>7.0f} {p5:>8.0f} {p95:>8.0f} {prob:>9.1f}%")


def run_scenario(records: List[Dict], params, scenario_name: str,
                 modifications: Dict[str, float], threshold: float = 5.0):
    """
    Run a what-if scenario with modified party percentages.

    modifications: dict of party -> new percentage (or delta if starts with +/-)
    """
    from sk_election_model import calculate_mandates, summarize_blocks

    # Get current Kalman estimates
    current = {}
    for party in set(r['party'] for r in records):
        kf, results = run_kalman_filter(records, party)
        if kf and results:
            support, _, _ = kf.get_state()
            current[party] = support

    # Apply modifications
    scenario = current.copy()
    for party, value in modifications.items():
        scenario[party] = value

    # Calculate seats
    seats = calculate_mandates(scenario, params)

    print(f"\n--- Scenario: {scenario_name} ---")
    print("\nModified polls:")
    for party, value in modifications.items():
        original = current.get(party, 0)
        diff = value - original
        print(f"  {party}: {original:.1f}% → {value:.1f}% ({diff:+.1f}%)")

    print("\nResulting seats:")
    for party, s in sorted(seats.items(), key=lambda x: x[1], reverse=True):
        orig_seats = calculate_mandates(current, params).get(party, 0)
        diff = s - orig_seats
        diff_str = f"({diff:+d})" if diff != 0 else ""
        print(f"  {party}: {s} {diff_str}")

    summarize_blocks(seats)


def print_scenarios(records: List[Dict], params, threshold: float = 5.0):
    """Run and print common scenario analyses."""
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)

    # Get current estimates
    current = {}
    for party in set(r['party'] for r in records):
        kf, results = run_kalman_filter(records, party)
        if kf and results:
            support, _, _ = kf.get_state()
            current[party] = support

    # Scenario 1: HLAS drops below 5%
    run_scenario(records, params, "HLAS drops below 5%",
                 {"HLAS": 4.5}, threshold)

    # Scenario 2: SNS rises above 5%
    run_scenario(records, params, "SNS rises above 5%",
                 {"SNS": 5.5}, threshold)

    # Scenario 3: HLAS collapses, voters go to SMER
    hlas_current = current.get("HLAS", 8)
    smer_current = current.get("SMER", 17)
    run_scenario(records, params, "HLAS collapse → SMER",
                 {"HLAS": 3.0, "SMER": smer_current + (hlas_current - 3.0)}, threshold)

    # Scenario 4: REP continues rising
    rep_current = current.get("REP", 10)
    run_scenario(records, params, "REP surge (+3%)",
                 {"REP": rep_current + 3.0}, threshold)

    # Scenario 5: PS + DEM merger (combined support)
    ps_current = current.get("PS", 23)
    dem_current = current.get("DEM", 5.5)
    run_scenario(records, params, "PS + DEM combined list",
                 {"PS": ps_current + dem_current, "DEM": 0}, threshold)


def load_all_polls(folder_path: str = "data") -> Dict:
    """Load all poll data from JSON files."""
    combined = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json") and filename not in ["weights.json", "input.json"]:
            year = os.path.splitext(filename)[0]
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                combined[year] = json.load(f)
    return combined


def flatten_to_timeseries(polls: Dict) -> List[Dict]:
    """Flatten nested poll data to a list of records with dates."""
    records = []
    for year, months in polls.items():
        for month, agencies in months.items():
            if not isinstance(agencies, dict):
                continue
            for agency, data in agencies.items():
                if not isinstance(data, dict) or "vysledky" not in data:
                    continue
                date = datetime(int(year), int(month), 15)  # Mid-month date
                for party, percent in data["vysledky"].items():
                    records.append({
                        "date": date,
                        "year": int(year),
                        "month": int(month),
                        "agency": agency,
                        "party": party,
                        "percent": percent
                    })
    return sorted(records, key=lambda x: (x["date"], x["party"]))


def flatten_mandates_to_timeseries(polls: Dict) -> List[Dict]:
    """Flatten nested poll mandate data to a list of records with dates."""
    records = []
    for year, months in polls.items():
        for month, agencies in months.items():
            if not isinstance(agencies, dict):
                continue
            for agency, data in agencies.items():
                if not isinstance(data, dict) or "mandaty" not in data:
                    continue
                mandaty = data["mandaty"]
                if not mandaty:  # Skip empty mandate data
                    continue
                date = datetime(int(year), int(month), 15)  # Mid-month date
                for party, seats in mandaty.items():
                    records.append({
                        "date": date,
                        "year": int(year),
                        "month": int(month),
                        "agency": agency,
                        "party": party,
                        "seats": seats
                    })
    return sorted(records, key=lambda x: (x["date"], x["party"]))


def calculate_monthly_seat_averages(records: List[Dict]) -> Dict[str, List[Tuple[datetime, float]]]:
    """Calculate monthly average seats per party across all agencies."""
    # Group by (year, month, party)
    grouped = {}
    for r in records:
        key = (r["year"], r["month"], r["party"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r["seats"])

    # Calculate averages and organize by party
    party_series = {}
    for (year, month, party), values in grouped.items():
        avg = np.mean(values)
        date = datetime(year, month, 15)
        if party not in party_series:
            party_series[party] = []
        party_series[party].append((date, avg))

    # Sort each series by date
    for party in party_series:
        party_series[party] = sorted(party_series[party], key=lambda x: x[0])

    return party_series


def calculate_monthly_averages(records: List[Dict]) -> Dict[str, List[Tuple[datetime, float]]]:
    """Calculate monthly averages per party across all agencies."""
    # Group by (year, month, party)
    grouped = {}
    for r in records:
        key = (r["year"], r["month"], r["party"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r["percent"])

    # Calculate averages and organize by party
    party_series = {}
    for (year, month, party), values in grouped.items():
        avg = np.mean(values)
        date = datetime(year, month, 15)
        if party not in party_series:
            party_series[party] = []
        party_series[party].append((date, avg))

    # Sort each series by date
    for party in party_series:
        party_series[party] = sorted(party_series[party], key=lambda x: x[0])

    return party_series


def calculate_trend(dates: List[datetime], values: List[float]) -> Tuple[float, float, float]:
    """Calculate linear trend using linear regression.

    Returns: (slope per month, intercept, r_squared)
    """
    # Convert dates to numeric (months from first date)
    x = np.array([(d - dates[0]).days / 30.0 for d in dates])
    y = np.array(values)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, r_value ** 2


def forecast_linear(dates: List[datetime], values: List[float], months_ahead: int = 6) -> List[Tuple[datetime, float]]:
    """Forecast future values using linear regression."""
    slope, intercept, _ = calculate_trend(dates, values)

    last_date = dates[-1]
    x_last = (last_date - dates[0]).days / 30.0

    forecasts = []
    for i in range(1, months_ahead + 1):
        future_x = x_last + i
        future_value = intercept + slope * future_x
        # Clamp to reasonable range
        future_value = max(0, min(100, future_value))
        future_date = datetime(
            last_date.year + (last_date.month + i - 1) // 12,
            (last_date.month + i - 1) % 12 + 1,
            15
        )
        forecasts.append((future_date, future_value))

    return forecasts


def calculate_moving_average(values: List[float], window: int = 3) -> List[float]:
    """Calculate simple moving average."""
    if len(values) < window:
        return values
    ma = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        ma.append(np.mean(values[start:i + 1]))
    return ma


def print_trend_summary(party_series: Dict[str, List[Tuple[datetime, float]]], threshold: float = 0.0):
    """Print summary of trends for all parties."""
    print("\n" + "=" * 70)
    print("TREND SUMMARY" + (f" (parties above {threshold}%)" if threshold > 0 else ""))
    print("=" * 70)

    trends = []
    for party, series in party_series.items():
        if len(series) < 3:
            continue
        dates, values = zip(*series)
        slope, _, r_squared = calculate_trend(list(dates), list(values))
        current = values[-1]
        if current >= threshold:
            trends.append((party, current, slope, r_squared))

    # Sort by current percentage
    trends.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Party':<12} {'Current %':>10} {'Trend/month':>12} {'R²':>8} {'Direction':<15}")
    print("-" * 60)

    for party, current, slope, r_squared in trends:
        direction = "Rising" if slope > 0.1 else "Falling" if slope < -0.1 else "Stable"
        trend_symbol = "↑" if slope > 0.1 else "↓" if slope < -0.1 else "→"
        print(f"{party:<12} {current:>9.1f}% {slope:>+11.2f}% {r_squared:>7.2f} {trend_symbol} {direction:<15}")


def print_forecast(party_series: Dict[str, List[Tuple[datetime, float]]], months_ahead: int = 6,
                   threshold: float = 0.0):
    """Print forecast for all parties."""
    print("\n" + "=" * 70)
    print(f"FORECAST ({months_ahead} months ahead)")
    print("=" * 70)

    forecasts = {}
    for party, series in party_series.items():
        if len(series) < 3:
            continue
        dates, values = zip(*series)
        current = values[-1]
        if current >= threshold:
            forecast = forecast_linear(list(dates), list(values), months_ahead)
            forecasts[party] = forecast

    if not forecasts:
        print("No parties above threshold.")
        return

    # Get forecast dates from first party
    first_party = list(forecasts.keys())[0]
    forecast_dates = [f[0] for f in forecasts[first_party]]

    # Print header
    print(f"\n{'Party':<12}", end="")
    for d in forecast_dates:
        print(f" {d.strftime('%b %Y'):>10}", end="")
    print()
    print("-" * (12 + 11 * len(forecast_dates)))

    # Sort by latest current value
    sorted_parties = sorted(
        forecasts.keys(),
        key=lambda p: party_series[p][-1][1],
        reverse=True
    )

    for party in sorted_parties:
        print(f"{party:<12}", end="")
        for _, value in forecasts[party]:
            print(f" {value:>9.1f}%", end="")
        print()


def plot_trends(party_series: Dict[str, List[Tuple[datetime, float]]],
                seat_series: Dict[str, List[Tuple[datetime, float]]],
                output_file: str = None,
                include_forecast: bool = True,
                months_ahead: int = 6,
                threshold: float = 0.0):
    """Plot trends for all parties."""

    # Define party colors (Slovak political colors)
    party_colors = {
        "PS": "#6B2D5C",      # Purple
        "SMER": "#E31E24",    # Red
        "HLAS": "#003DA5",    # Blue
        "REP": "#000000",     # Black
        "SLOV": "#00A651",    # Green
        "SAS": "#FFD700",     # Yellow/Gold
        "KDH": "#0066CC",     # Blue
        "DEM": "#FF6600",     # Orange
        "SNS": "#003366",     # Dark blue
        "Aliancia": "#009933", # Green
        "ROD": "#8B0000",     # Dark red
    }

    # Filter to main parties based on threshold
    main_parties = {}
    for party, series in party_series.items():
        current = series[-1][1] if series else 0
        if current >= threshold:
            main_parties[party] = series

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))

    # Plot 1: All parties trend lines
    ax1.set_title("Slovak Election Polls - Party Trends", fontsize=14, fontweight='bold')
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='5% threshold')

    for party, series in main_parties.items():
        dates, values = zip(*series)
        color = party_colors.get(party, None)
        ax1.plot(dates, values, 'o-', label=party, color=color, markersize=4, linewidth=2)

        # Add forecast
        if include_forecast and len(series) >= 3:
            forecast = forecast_linear(list(dates), list(values), months_ahead)
            f_dates, f_values = zip(*forecast)
            ax1.plot(f_dates, f_values, '--', color=color, alpha=0.5, linewidth=1.5)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Poll Percentage (%)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 30)

    # Add vertical line for "now"
    last_date = max(max(d for d, _ in series) for series in main_parties.values())
    if include_forecast:
        ax1.axvline(x=last_date, color='red', linestyle=':', alpha=0.7, label='Latest data')

    # Plot 2: Government vs Opposition blocks (percentage)
    ax2.set_title("Coalition Blocks - Poll Percentage", fontsize=14, fontweight='bold')

    government = {"SMER", "SNS", "HLAS", "REP"}
    opposition = {"PS", "SAS", "DEM", "SLOV", "KDH"}

    # Calculate block totals per month
    all_dates = sorted(set(d for series in party_series.values() for d, _ in series))
    gov_totals = []
    opp_totals = []

    for date in all_dates:
        gov_sum = 0
        opp_sum = 0
        for party, series in party_series.items():
            for d, v in series:
                if d == date:
                    if party in government:
                        gov_sum += v
                    elif party in opposition:
                        opp_sum += v
        gov_totals.append((date, gov_sum))
        opp_totals.append((date, opp_sum))

    gov_dates, gov_values = zip(*gov_totals)
    opp_dates, opp_values = zip(*opp_totals)

    ax2.plot(gov_dates, gov_values, 'o-', label='Government (SMER+HLAS+SNS+REP)',
             color='red', markersize=5, linewidth=2)
    ax2.plot(opp_dates, opp_values, 'o-', label='Opposition (PS+SAS+DEM+SLOV+KDH)',
             color='blue', markersize=5, linewidth=2)

    # Add forecast for blocks
    if include_forecast:
        gov_forecast = forecast_linear(list(gov_dates), list(gov_values), months_ahead)
        opp_forecast = forecast_linear(list(opp_dates), list(opp_values), months_ahead)

        gf_dates, gf_values = zip(*gov_forecast)
        of_dates, of_values = zip(*opp_forecast)

        ax2.plot(gf_dates, gf_values, '--', color='red', alpha=0.5, linewidth=1.5)
        ax2.plot(of_dates, of_values, '--', color='blue', alpha=0.5, linewidth=1.5)
        ax2.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7)

    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50% mark')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Combined Percentage (%)")
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Parliament Seats by Block
    ax3.set_title("Parliament Seats - Government vs Opposition", fontsize=14, fontweight='bold')

    # Calculate seat totals per month
    seat_dates = sorted(set(d for series in seat_series.values() for d, _ in series))
    gov_seats = []
    opp_seats = []

    for date in seat_dates:
        gov_sum = 0
        opp_sum = 0
        for party, series in seat_series.items():
            for d, v in series:
                if d == date:
                    if party in government:
                        gov_sum += v
                    elif party in opposition:
                        opp_sum += v
        gov_seats.append((date, gov_sum))
        opp_seats.append((date, opp_sum))

    if gov_seats and opp_seats:
        gs_dates, gs_values = zip(*gov_seats)
        os_dates, os_values = zip(*opp_seats)

        ax3.plot(gs_dates, gs_values, 'o-', label='Government (SMER+HLAS+SNS+REP)',
                 color='red', markersize=5, linewidth=2)
        ax3.plot(os_dates, os_values, 'o-', label='Opposition (PS+SAS+DEM+SLOV+KDH)',
                 color='blue', markersize=5, linewidth=2)

        # Add forecast for seat blocks
        if include_forecast and len(gs_values) >= 3:
            gov_seat_forecast = forecast_linear(list(gs_dates), list(gs_values), months_ahead)
            opp_seat_forecast = forecast_linear(list(os_dates), list(os_values), months_ahead)

            gsf_dates, gsf_values = zip(*gov_seat_forecast)
            osf_dates, osf_values = zip(*opp_seat_forecast)

            ax3.plot(gsf_dates, gsf_values, '--', color='red', alpha=0.5, linewidth=1.5)
            ax3.plot(osf_dates, osf_values, '--', color='blue', alpha=0.5, linewidth=1.5)

        # Find the last date with seat data
        last_seat_date = max(gs_dates)
        ax3.axvline(x=last_seat_date, color='gray', linestyle=':', alpha=0.7)

    ax3.axhline(y=76, color='green', linestyle='--', alpha=0.5, label='Majority (76)')
    ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Constitutional (90)')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Parliament Seats")
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.set_ylim(0, 160)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {output_file}")
    else:
        plt.show()


def plot_kalman_trends(records: List[Dict],
                       seat_series: Dict[str, List[Tuple[datetime, float]]],
                       output_file: str = None,
                       months_ahead: int = 6,
                       process_noise: float = 0.3,
                       measurement_noise: float = 2.0,
                       threshold: float = 0.0):
    """Plot Kalman-filtered trends with confidence intervals."""

    # Define party colors
    party_colors = {
        "PS": "#6B2D5C",
        "SMER": "#E31E24",
        "HLAS": "#003DA5",
        "REP": "#000000",
        "SLOV": "#00A651",
        "SAS": "#FFD700",
        "KDH": "#0066CC",
        "DEM": "#FF6600",
        "SNS": "#003366",
        "Aliancia": "#009933",
        "ROD": "#8B0000",
    }

    # Get Kalman filtered data
    kalman_series = calculate_kalman_averages(records, process_noise, measurement_noise)

    # Filter to main parties based on threshold
    main_parties = {}
    for party, series in kalman_series.items():
        if series:
            current = series[-1][1]  # Latest filtered value
            if current >= threshold:
                main_parties[party] = series

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16))

    # Plot 1: Kalman filtered party trends with confidence bands
    ax1.set_title("Slovak Election Polls - Kalman Filtered Trends (with 95% CI)", fontsize=14, fontweight='bold')
    ax1.axhline(y=5, color='gray', linestyle='--', alpha=0.5, label='5% threshold')

    government = {"SMER", "SNS", "HLAS", "REP"}
    opposition = {"PS", "SAS", "DEM", "SLOV", "KDH"}

    for party, series in main_parties.items():
        dates = [d for d, _, _ in series]
        values = [v for _, v, _ in series]
        uncertainties = [u for _, _, u in series]

        color = party_colors.get(party, None)

        # Plot filtered line
        ax1.plot(dates, values, '-', label=party, color=color, linewidth=2)

        # Plot confidence band (95% CI = ~2 standard deviations)
        upper = [v + 1.96 * u for v, u in zip(values, uncertainties)]
        lower = [v - 1.96 * u for v, u in zip(values, uncertainties)]
        ax1.fill_between(dates, lower, upper, color=color, alpha=0.15)

        # Add forecast with growing uncertainty
        forecast = kalman_forecast(records, party, months_ahead, process_noise, measurement_noise)
        if forecast:
            f_dates = [d for d, _, _ in forecast]
            f_values = [v for _, v, _ in forecast]
            f_unc = [u for _, _, u in forecast]

            ax1.plot(f_dates, f_values, '--', color=color, alpha=0.7, linewidth=1.5)

            f_upper = [v + 1.96 * u for v, u in zip(f_values, f_unc)]
            f_lower = [v - 1.96 * u for v, u in zip(f_values, f_unc)]
            ax1.fill_between(f_dates, f_lower, f_upper, color=color, alpha=0.08)

    # Add vertical line for latest data
    last_date = max(max(d for d, _, _ in series) for series in main_parties.values())
    ax1.axvline(x=last_date, color='red', linestyle=':', alpha=0.7)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Poll Percentage (%)")
    ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=1)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax1.set_ylim(0, 30)

    # Plot 2: Coalition blocks with Kalman filtering
    ax2.set_title("Coalition Blocks - Kalman Filtered (with 95% CI)", fontsize=14, fontweight='bold')

    # Calculate block totals using Kalman filtered values
    all_dates = sorted(set(d for series in kalman_series.values() for d, _, _ in series))

    gov_totals = []
    opp_totals = []
    gov_unc = []
    opp_unc = []

    for date in all_dates:
        gov_sum = 0
        opp_sum = 0
        gov_var = 0
        opp_var = 0

        for party, series in kalman_series.items():
            for d, v, u in series:
                if d == date:
                    if party in government:
                        gov_sum += v
                        gov_var += u ** 2
                    elif party in opposition:
                        opp_sum += v
                        opp_var += u ** 2

        gov_totals.append((date, gov_sum))
        opp_totals.append((date, opp_sum))
        gov_unc.append(np.sqrt(gov_var))
        opp_unc.append(np.sqrt(opp_var))

    gov_dates, gov_values = zip(*gov_totals)
    opp_dates, opp_values = zip(*opp_totals)

    # Plot with confidence bands
    ax2.plot(gov_dates, gov_values, '-', label='Government (SMER+HLAS+SNS+REP)',
             color='red', linewidth=2)
    gov_upper = [v + 1.96 * u for v, u in zip(gov_values, gov_unc)]
    gov_lower = [v - 1.96 * u for v, u in zip(gov_values, gov_unc)]
    ax2.fill_between(gov_dates, gov_lower, gov_upper, color='red', alpha=0.15)

    ax2.plot(opp_dates, opp_values, '-', label='Opposition (PS+SAS+DEM+SLOV+KDH)',
             color='blue', linewidth=2)
    opp_upper = [v + 1.96 * u for v, u in zip(opp_values, opp_unc)]
    opp_lower = [v - 1.96 * u for v, u in zip(opp_values, opp_unc)]
    ax2.fill_between(opp_dates, opp_lower, opp_upper, color='blue', alpha=0.15)

    # Add forecast for coalition blocks
    gov_forecast_totals = []
    opp_forecast_totals = []
    gov_forecast_unc = []
    opp_forecast_unc = []

    # Generate forecast dates
    for i in range(1, months_ahead + 1):
        forecast_date = last_date + timedelta(days=30 * i)
        gov_sum = 0
        opp_sum = 0
        gov_var = 0
        opp_var = 0

        for party in government:
            forecast = kalman_forecast(records, party, months_ahead, process_noise, measurement_noise)
            if forecast and i <= len(forecast):
                _, v, u = forecast[i - 1]
                gov_sum += v
                gov_var += u ** 2

        for party in opposition:
            forecast = kalman_forecast(records, party, months_ahead, process_noise, measurement_noise)
            if forecast and i <= len(forecast):
                _, v, u = forecast[i - 1]
                opp_sum += v
                opp_var += u ** 2

        gov_forecast_totals.append((forecast_date, gov_sum))
        opp_forecast_totals.append((forecast_date, opp_sum))
        gov_forecast_unc.append(np.sqrt(gov_var))
        opp_forecast_unc.append(np.sqrt(opp_var))

    if gov_forecast_totals:
        gf_dates, gf_values = zip(*gov_forecast_totals)
        of_dates, of_values = zip(*opp_forecast_totals)

        ax2.plot(gf_dates, gf_values, '--', color='red', alpha=0.7, linewidth=1.5)
        gf_upper = [v + 1.96 * u for v, u in zip(gf_values, gov_forecast_unc)]
        gf_lower = [v - 1.96 * u for v, u in zip(gf_values, gov_forecast_unc)]
        ax2.fill_between(gf_dates, gf_lower, gf_upper, color='red', alpha=0.08)

        ax2.plot(of_dates, of_values, '--', color='blue', alpha=0.7, linewidth=1.5)
        of_upper = [v + 1.96 * u for v, u in zip(of_values, opp_forecast_unc)]
        of_lower = [v - 1.96 * u for v, u in zip(of_values, opp_forecast_unc)]
        ax2.fill_between(of_dates, of_lower, of_upper, color='blue', alpha=0.08)

    ax2.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50% mark')
    ax2.axvline(x=last_date, color='gray', linestyle=':', alpha=0.7)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Combined Percentage (%)")
    ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Parliament Seats
    ax3.set_title("Parliament Seats - Government vs Opposition", fontsize=14, fontweight='bold')

    seat_dates = sorted(set(d for series in seat_series.values() for d, _ in series))
    gov_seats = []
    opp_seats = []

    for date in seat_dates:
        gov_sum = 0
        opp_sum = 0
        for party, series in seat_series.items():
            for d, v in series:
                if d == date:
                    if party in government:
                        gov_sum += v
                    elif party in opposition:
                        opp_sum += v
        gov_seats.append((date, gov_sum))
        opp_seats.append((date, opp_sum))

    if gov_seats and opp_seats:
        gs_dates, gs_values = zip(*gov_seats)
        os_dates, os_values = zip(*opp_seats)

        ax3.plot(gs_dates, gs_values, 'o-', label='Government',
                 color='red', markersize=5, linewidth=2)
        ax3.plot(os_dates, os_values, 'o-', label='Opposition',
                 color='blue', markersize=5, linewidth=2)

        last_seat_date = max(gs_dates)
        ax3.axvline(x=last_seat_date, color='gray', linestyle=':', alpha=0.7)

        # Add forecast for seats based on percentage forecasts
        # Use a simple conversion: seats ≈ percentage * 1.5 (rough approximation for parties above threshold)
        if len(gs_values) >= 3:
            gov_seat_forecast = []
            opp_seat_forecast = []

            for i in range(1, months_ahead + 1):
                forecast_date = last_seat_date + timedelta(days=30 * i)
                gov_seats_sum = 0
                opp_seats_sum = 0

                for party in government:
                    forecast = kalman_forecast(records, party, months_ahead, process_noise, measurement_noise)
                    if forecast and i <= len(forecast):
                        _, pct, _ = forecast[i - 1]
                        # Approximate seats from percentage (parties above 5% get ~1.5x their percentage in seats)
                        if pct >= 5:
                            gov_seats_sum += pct * 1.7
                        elif pct >= 3:
                            gov_seats_sum += pct * 0.5

                for party in opposition:
                    forecast = kalman_forecast(records, party, months_ahead, process_noise, measurement_noise)
                    if forecast and i <= len(forecast):
                        _, pct, _ = forecast[i - 1]
                        if pct >= 5:
                            opp_seats_sum += pct * 1.7
                        elif pct >= 3:
                            opp_seats_sum += pct * 0.5

                gov_seat_forecast.append((forecast_date, gov_seats_sum))
                opp_seat_forecast.append((forecast_date, opp_seats_sum))

            if gov_seat_forecast:
                gsf_dates, gsf_values = zip(*gov_seat_forecast)
                osf_dates, osf_values = zip(*opp_seat_forecast)

                ax3.plot(gsf_dates, gsf_values, '--', color='red', alpha=0.5, linewidth=1.5)
                ax3.plot(osf_dates, osf_values, '--', color='blue', alpha=0.5, linewidth=1.5)

    ax3.axhline(y=76, color='green', linestyle='--', alpha=0.5, label='Majority (76)')
    ax3.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='Constitutional (90)')
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Parliament Seats")
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax3.set_ylim(0, 160)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nChart saved to: {output_file}")
    else:
        plt.show()


def print_kalman_summary(records: List[Dict], process_noise: float = 0.3,
                         measurement_noise: float = 2.0, threshold: float = 0.0):
    """Print Kalman filter summary for all parties."""
    print("\n" + "=" * 70)
    print("KALMAN FILTER ESTIMATES" + (f" (parties above {threshold}%)" if threshold > 0 else ""))
    print("=" * 70)

    parties = set(r['party'] for r in records)
    results = []

    for party in parties:
        kf, filtered = run_kalman_filter(records, party, process_noise, measurement_noise)
        if kf and filtered:
            support, velocity, uncertainty = kf.get_state()
            if support >= threshold:
                results.append((party, support, velocity, uncertainty))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Party':<12} {'Estimate':>10} {'95% CI':>16} {'Velocity':>12} {'Direction':<10}")
    print("-" * 65)

    for party, support, velocity, uncertainty in results:
        ci_low = support - 1.96 * uncertainty
        ci_high = support + 1.96 * uncertainty
        direction = "Rising" if velocity > 0.1 else "Falling" if velocity < -0.1 else "Stable"
        trend_symbol = "↑" if velocity > 0.1 else "↓" if velocity < -0.1 else "→"

        print(f"{party:<12} {support:>9.1f}% [{ci_low:>5.1f}-{ci_high:>5.1f}%] {velocity:>+10.2f}%/mo {trend_symbol} {direction:<10}")


def print_kalman_forecast(records: List[Dict], months_ahead: int = 6,
                          process_noise: float = 0.3, measurement_noise: float = 2.0,
                          threshold: float = 0.0):
    """Print Kalman-based forecasts with uncertainty."""
    print("\n" + "=" * 70)
    print(f"KALMAN FORECAST ({months_ahead} months ahead, with 95% CI)")
    print("=" * 70)

    parties = set(r['party'] for r in records)
    forecasts = {}

    for party in parties:
        kf, _ = run_kalman_filter(records, party, process_noise, measurement_noise)
        if kf:
            support, _, _ = kf.get_state()
            if support >= threshold:
                forecast = kalman_forecast(records, party, months_ahead, process_noise, measurement_noise)
                if forecast:
                    forecasts[party] = forecast

    if not forecasts:
        print("No forecast data available.")
        return

    # Get dates from first party
    first_party = list(forecasts.keys())[0]
    forecast_dates = [d for d, _, _ in forecasts[first_party]]

    # Sort parties by current estimate
    sorted_parties = sorted(
        forecasts.keys(),
        key=lambda p: forecasts[p][0][1] if forecasts[p] else 0,
        reverse=True
    )

    # Print header
    print(f"\n{'Party':<12}", end="")
    for d in forecast_dates:
        print(f" {d.strftime('%b %Y'):>12}", end="")
    print()
    print("-" * (12 + 13 * len(forecast_dates)))

    for party in sorted_parties:
        print(f"{party:<12}", end="")
        for _, value, unc in forecasts[party]:
            ci_range = 1.96 * unc
            print(f" {value:>5.1f}±{ci_range:>4.1f}%", end="")
        print()


def print_input_data(party_series: Dict[str, List[Tuple[datetime, float]]], threshold: float = 0.0):
    """Print the input data summary."""
    print("=" * 70)
    print("INPUT DATA SUMMARY")
    print("=" * 70)

    # Find date range
    all_dates = sorted(set(d for series in party_series.values() for d, _ in series))
    print(f"\nData period: {all_dates[0].strftime('%B %Y')} - {all_dates[-1].strftime('%B %Y')}")
    print(f"Number of months: {len(all_dates)}")
    print(f"Number of parties tracked: {len(party_series)}")

    print(f"\nLatest poll averages ({all_dates[-1].strftime('%B %Y')}):")
    print("-" * 40)

    latest = []
    for party, series in party_series.items():
        for date, value in series:
            if date == all_dates[-1]:
                latest.append((party, value))

    latest.sort(key=lambda x: x[1], reverse=True)
    for party, value in latest:
        if threshold > 0 and value < threshold:
            continue
        threshold_mark = "✓" if value >= 5 else "✗"
        print(f"  {party:<12} {value:>6.1f}%  {threshold_mark}")


def main():
    parser = argparse.ArgumentParser(description="Slovak Election Poll Trend Analysis")
    parser.add_argument("--forecast", type=int, default=6,
                        help="Number of months to forecast (default: 6)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for charts (creates if doesn't exist)")
    parser.add_argument("--kalman", action="store_true",
                        help="Use Kalman filtering for smoothing and forecasting")
    parser.add_argument("--process-noise", type=float, default=0.2,
                        help="Kalman process noise (default: 0.2, higher = more volatile)")
    parser.add_argument("--measurement-noise", type=float, default=1.5,
                        help="Kalman measurement noise (default: 1.5, higher = less trust in polls)")
    parser.add_argument("--threshold", type=float, default=5.0,
                        help="Only show parties above this percentage (default: 5.0)")

    # Statistical analysis flags
    parser.add_argument("--correlations", action="store_true",
                        help="Show party correlation analysis")
    parser.add_argument("--volatility", action="store_true",
                        help="Show volatility analysis")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulation for seat probabilities")
    parser.add_argument("--simulations", type=int, default=10000,
                        help="Number of Monte Carlo simulations (default: 10000)")
    parser.add_argument("--scenarios", action="store_true",
                        help="Run scenario analysis (what-if)")
    parser.add_argument("--full-stats", action="store_true",
                        help="Run all statistical analyses")
    parser.add_argument("--graph", action="store_true",
                        help="Generate graphs/charts (saves to --output-dir if specified, otherwise shows interactively)")
    args = parser.parse_args()

    # Determine output file path
    output_file = None
    if args.output_dir:
        from pathlib import Path
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = "trends_kalman.png" if args.kalman else "trends.png"
        output_file = str(output_path / filename)

    # Load and process data
    polls = load_all_polls("data")
    records = flatten_to_timeseries(polls)
    party_series = calculate_monthly_averages(records)

    # Load and process seat/mandate data
    mandate_records = flatten_mandates_to_timeseries(polls)
    seat_series = calculate_monthly_seat_averages(mandate_records)

    # Print analysis
    print_input_data(party_series, args.threshold)

    # Check if we should generate graphs
    generate_graphs = args.graph

    if args.kalman:
        # Use Kalman filtering
        print_kalman_summary(records, args.process_noise, args.measurement_noise, args.threshold)
        print_kalman_forecast(records, args.forecast, args.process_noise, args.measurement_noise, args.threshold)

        if generate_graphs:
            plot_kalman_trends(records, seat_series, output_file, args.forecast,
                               args.process_noise, args.measurement_noise, args.threshold)
    else:
        # Use simple linear regression
        print_trend_summary(party_series, args.threshold)
        print_forecast(party_series, args.forecast, args.threshold)

        if generate_graphs:
            plot_trends(party_series, seat_series, output_file, months_ahead=args.forecast,
                        threshold=args.threshold)

    # Statistical analyses
    if args.correlations or args.full_stats:
        print_correlations(records, args.threshold)
        if generate_graphs:
            corr_file = None
            if args.output_dir:
                from pathlib import Path
                corr_file = str(Path(args.output_dir) / "correlations.png")
            plot_correlations(records, corr_file, args.threshold)

    if args.volatility or args.full_stats:
        print_volatility(records, args.threshold)

    if args.monte_carlo or args.full_stats:
        # Load model parameters
        import json
        try:
            with open("data/weights.json", "r") as f:
                from sk_election_model import ModelParams
                params = ModelParams.from_dict(json.load(f))
        except FileNotFoundError:
            from sk_election_model import ModelParams
            params = ModelParams()
        print_monte_carlo(records, params, args.simulations, args.threshold)

    if args.scenarios or args.full_stats:
        # Load model parameters
        import json
        try:
            with open("data/weights.json", "r") as f:
                from sk_election_model import ModelParams
                params = ModelParams.from_dict(json.load(f))
        except FileNotFoundError:
            from sk_election_model import ModelParams
            params = ModelParams()
        print_scenarios(records, params, args.threshold)


if __name__ == "__main__":
    main()
