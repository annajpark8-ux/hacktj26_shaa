"""
================================================================================
QAOA Heart-Organ Matching  —  Full CSV-to-Match Pipeline
================================================================================

PIPELINE:
    1. Read donor info (ABO, age, BSA, lat/lon).
    2. Read recipient CSV (name, abo, age, bsa, cpra, wait, lat, lon, urgency).
    3. FILTER — eliminate incompatible candidates:
         a. ABO blood type incompatibility
         b. Travel time > 5 hours (haversine distance / transport speed)
         c. Donor BSA < 70% of recipient BSA (undersized organ)
    4. BUILD Recipients from surviving candidates, computing:
         - bsa_similarity: how close donor/recipient BSA are (0–1)
         - distance_time_hours: estimated travel time from coordinates
         - is_child: age < 18
    5. RUN QAOA to select the optimal recipient.

INSTALL:
    pip install numpy scipy

    For the Qiskit version, also: pip install qiskit qiskit-aer

nd ON DISTANCE:
    Travel time is estimated as haversine great-circle distance
    divided by an average transport speed (default 80 km/h for
    helicopter door-to-door).  Adjust TRANSPORT_SPEED_KMH for
    your transport mode.
================================================================================
"""

import csv
import numpy as np
from collections import Counter
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from math import radians, sin, cos, sqrt, atan2
import time
import sys
import os


# =============================================================================
# 1. DATA MODELS
# =============================================================================

@dataclass
class Donor:
    """The donated heart and its donor's attributes."""
    abo: str            # e.g. "A+", "O-", "AB+"
    age: int
    bsa: float          # body surface area in m²
    latitude: float     # hospital latitude
    longitude: float    # hospital longitude


@dataclass
class CsvCandidate:
    """Raw candidate row from the CSV before any filtering."""
    name: str
    abo: str
    age: int
    bsa: float
    cpra: float         # 0–100
    waiting_time_days: float
    latitude: float
    longitude: float
    urgency: int        # 1–6 (1 = most urgent)


@dataclass
class Recipient:
    """
    A filtered, QAOA-ready candidate.

    Fields:
        name:                patient name
        bsa_similarity:      how well the donor organ size matches (0–1)
        urgency_level:       1–6 (1 = most urgent)
        waiting_time_days:   days on the waitlist
        distance_time_hours: estimated travel time donor→recipient hospital
        is_child:            True if age < 18
        cpra_score:          0–100 (panel reactive antibodies)
    """
    name: str
    bsa_similarity: float
    urgency_level: int
    waiting_time_days: float
    distance_time_hours: float
    is_child: bool
    cpra_score: float


@dataclass
class MatchingWeights:
    """Policy weights for the 6 QAOA factors."""
    bsa_similarity: float = 0.20   # organ size match
    urgency: float        = 0.25   # medical urgency
    waiting_time: float   = 0.15   # fairness
    distance: float       = 0.15   # organ viability (ischemic time)
    pediatric: float      = 0.10   # pediatric priority
    cpra: float           = 0.15   # sensitized patient priority


# =============================================================================
# 2. CSV READING
# =============================================================================

def read_candidates_csv(filepath: str) -> List[CsvCandidate]:
    """
    Parse a CSV file into CsvCandidate objects.

    Expected columns (case-insensitive, order-independent):
        name, abo, age, bsa, cpra, waiting_time_days,
        latitude, longitude, urgency
    """
    candidates = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        # Normalize column names: strip whitespace, lowercase
        reader.fieldnames = [col.strip().lower().replace(' ', '_') for col in reader.fieldnames]

        for row in reader:
            candidates.append(CsvCandidate(
                name=row['name'].strip(),
                abo=row['abo'].strip().upper(),
                age=int(row['age'].strip()),
                bsa=float(row['bsa'].strip()),
                cpra=float(row['cpra'].strip()),
                waiting_time_days=float(row['waiting_time_days'].strip()),
                latitude=float(row['latitude'].strip()),
                longitude=float(row['longitude'].strip()),
                urgency=int(row['urgency'].strip()),
            ))

    return candidates


# =============================================================================
# 3. ABO BLOOD TYPE COMPATIBILITY
# =============================================================================
#
# Heart transplant ABO rules (recipient can receive from):
#   O  can receive from:  O only
#   A  can receive from:  A, O
#   B  can receive from:  B, O
#   AB can receive from:  A, B, AB, O  (universal recipient)
#
# Rh factor (+/-) is generally NOT a barrier for solid organ
# transplants (unlike blood transfusions), so we strip it.
# =============================================================================

# Maps each recipient base type → set of compatible donor base types
ABO_COMPATIBILITY = {
    'O':  {'O'},
    'A':  {'A', 'O'},
    'B':  {'B', 'O'},
    'AB': {'A', 'B', 'AB', 'O'},
}


def _strip_rh(abo: str) -> str:
    """
    Extract the base ABO type, ignoring Rh factor.
    'A+' → 'A',  'AB-' → 'AB',  'O+' → 'O'
    """
    return abo.replace('+', '').replace('-', '')


def is_abo_compatible(donor_abo: str, recipient_abo: str) -> bool:
    """Check if the donor's blood type is compatible with the recipient."""
    donor_base = _strip_rh(donor_abo)
    recipient_base = _strip_rh(recipient_abo)

    compatible_donors = ABO_COMPATIBILITY.get(recipient_base, set())
    return donor_base in compatible_donors


# =============================================================================
# 4. DISTANCE / TRAVEL TIME  —  Haversine + estimated transport speed
# =============================================================================
#
# Since the CSV provides lat/lon directly, we just compute the
# great-circle distance between donor and recipient hospitals
# using the Haversine formula, then divide by an average transport
# speed to estimate travel time.
#
# TRANSPORT_SPEED_KMH = 80 km/h is a conservative estimate for
# helicopter organ transport (cruise speed ~250 km/h but total
# door-to-door including prep, takeoff, landing, and ground
# transport averages much lower).
#
# Adjust this constant if your transport mode differs:
#   Fixed-wing aircraft:  ~200 km/h effective
#   Ground ambulance:     ~60–80 km/h (varies by region/traffic)
#   Helicopter:           ~80–120 km/h door-to-door
# =============================================================================

TRANSPORT_SPEED_KMH = 80.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth.

    Uses the Haversine formula:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2·atan2(√a, √(1−a))
        d = R·c

    Returns distance in kilometers.
    """
    R = 6371.0  # Earth's mean radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def estimate_travel_time_hours(
    donor_lat: float,
    donor_lon: float,
    recipient_lat: float,
    recipient_lon: float,
) -> float:
    """
    Estimate travel time in hours between donor and recipient hospitals.

    Args:
        donor_lat, donor_lon:         donor hospital coordinates.
        recipient_lat, recipient_lon: recipient hospital coordinates.

    Returns:
        Estimated travel time in hours.
    """
    distance_km = _haversine_km(donor_lat, donor_lon, recipient_lat, recipient_lon)
    return distance_km / TRANSPORT_SPEED_KMH


# =============================================================================
# 5. BSA SIMILARITY
# =============================================================================
#
# Body Surface Area matching ensures the donor heart is appropriately
# sized for the recipient's body.
#
# Hard filter: donor BSA must be >= 70% of recipient BSA
#   (an undersized organ can't support the recipient's circulation)
#
# Soft score: BSA similarity for QAOA ranking (0–1)
#   We use:  similarity = 1 - |donor_bsa - recip_bsa| / max(donor_bsa, recip_bsa)
#   This gives 1.0 for a perfect match and decreases as the sizes diverge.
#   An oversized organ is generally acceptable (better than undersized),
#   so this penalizes both directions but the hard filter already
#   removed dangerously undersized cases.
# =============================================================================

def is_bsa_compatible(donor_bsa: float, recipient_bsa: float) -> bool:
    """
    Hard filter: reject if donor organ is too small for the recipient.
    Donor BSA must be at least 70% of recipient BSA.
    """
    return donor_bsa >= 0.70 * recipient_bsa


def compute_bsa_similarity(donor_bsa: float, recipient_bsa: float) -> float:
    """
    Soft score: how close are the body sizes? Returns 0–1.
    1.0 = identical BSA, decreasing as they diverge.
    """
    max_bsa = max(donor_bsa, recipient_bsa)
    if max_bsa == 0:
        return 0.0
    return 1.0 - abs(donor_bsa - recipient_bsa) / max_bsa


# =============================================================================
# 6. FILTERING PIPELINE
# =============================================================================

def filter_and_build_recipients(
    donor: Donor,
    candidates: List[CsvCandidate],
    max_travel_hours: float = 5.0,
    verbose: bool = True,
) -> Tuple[List[Recipient], dict]:
    """
    Apply all hard filters and build Recipient objects for QAOA.

    Filters (in order):
        1. ABO incompatibility
        2. Travel time > max_travel_hours
        3. Donor BSA < 70% of recipient BSA

    Args:
        donor: the organ donor's attributes.
        candidates: raw CSV candidates.
        max_travel_hours: maximum acceptable travel time.
        verbose: print filtering details.

    Returns:
        recipients: list of Recipient objects that passed all filters.
        filter_log: dict with counts and details of eliminated candidates.
    """
    filter_log = {
        'total_candidates': len(candidates),
        'eliminated_abo': [],
        'eliminated_distance': [],
        'eliminated_bsa': [],
        'passed': [],
    }

    recipients = []

    if verbose:
        print(f"\n  FILTERING {len(candidates)} CANDIDATES")
        print(f"  Donor: ABO={donor.abo}, Age={donor.age}, BSA={donor.bsa:.2f}")
        print(f"  Donor location: ({donor.latitude:.4f}, {donor.longitude:.4f})")
        print(f"  Max travel time: {max_travel_hours}h")
        print(f"  Transport speed: {TRANSPORT_SPEED_KMH} km/h")
        print(f"  Min donor/recipient BSA ratio: 70%")
        print("  " + "-" * 65)

    for cand in candidates:
        # ---- Filter 1: ABO compatibility ----
        if not is_abo_compatible(donor.abo, cand.abo):
            filter_log['eliminated_abo'].append(cand.name)
            if verbose:
                print(f"    ✗ {cand.name:<20} ABO incompatible "
                      f"(donor {donor.abo} → recipient {cand.abo})")
            continue

        # ---- Filter 2: Travel time ----
        travel_hours = estimate_travel_time_hours(
            donor.latitude, donor.longitude,
            cand.latitude, cand.longitude,
        )

        if travel_hours > max_travel_hours:
            filter_log['eliminated_distance'].append(cand.name)
            if verbose:
                print(f"    ✗ {cand.name:<20} Too far: {travel_hours:.1f}h "
                      f"(max {max_travel_hours}h)")
            continue

        # ---- Filter 3: BSA compatibility ----
        if not is_bsa_compatible(donor.bsa, cand.bsa):
            bsa_ratio = donor.bsa / cand.bsa * 100
            filter_log['eliminated_bsa'].append(cand.name)
            if verbose:
                print(f"    ✗ {cand.name:<20} Undersized organ: donor BSA "
                      f"{donor.bsa:.2f} is {bsa_ratio:.0f}% of recipient "
                      f"{cand.bsa:.2f} (need ≥70%)")
            continue

        # ---- Passed all filters — build Recipient ----
        bsa_sim = compute_bsa_similarity(donor.bsa, cand.bsa)
        is_child = cand.age < 18

        recipient = Recipient(
            name=cand.name,
            bsa_similarity=bsa_sim,
            urgency_level=cand.urgency,
            waiting_time_days=cand.waiting_time_days,
            distance_time_hours=travel_hours,
            is_child=is_child,
            cpra_score=cand.cpra,
        )
        recipients.append(recipient)
        filter_log['passed'].append(cand.name)

        if verbose:
            child_tag = " [CHILD]" if is_child else ""
            print(f"    ✓ {cand.name:<20} ABO={cand.abo:>3}  "
                  f"BSA={bsa_sim:.2f}  "
                  f"Travel={travel_hours:.1f}h  "
                  f"Urgency={cand.urgency}  "
                  f"CPRA={cand.cpra:.0f}%{child_tag}")

    if verbose:
        print("  " + "-" * 65)
        print(f"  RESULT: {len(recipients)} candidates passed "
              f"(eliminated {len(candidates) - len(recipients)})")
        print(f"    ABO incompatible:   {len(filter_log['eliminated_abo'])}")
        print(f"    Too far:            {len(filter_log['eliminated_distance'])}")
        print(f"    BSA undersized:     {len(filter_log['eliminated_bsa'])}")

    return recipients, filter_log


# =============================================================================
# 7. SCORE NORMALIZATION  (updated for new Recipient fields)
# =============================================================================

def normalize_scores(recipients: List[Recipient]) -> np.ndarray:
    """
    Normalize all 6 factors to [0,1]. Returns (n, 6) array.

    Columns:
        0: bsa_similarity     — already [0,1], pass through
        1: urgency            — 1→1.0 (most urgent), 6→0.0 (INVERTED)
        2: waiting_time       — min-max normalized (longer = higher)
        3: distance_time      — INVERTED (closer = higher)
        4: pediatric          — binary 0/1
        5: cpra               — scale to [0,1] (higher = more priority)
    """
    n = len(recipients)
    scores = np.zeros((n, 6))

    bsa_sims   = np.array([r.bsa_similarity for r in recipients])
    urgencies  = np.array([r.urgency_level for r in recipients], dtype=float)
    wait_times = np.array([r.waiting_time_days for r in recipients], dtype=float)
    distances  = np.array([r.distance_time_hours for r in recipients], dtype=float)
    pediatrics = np.array([1.0 if r.is_child else 0.0 for r in recipients])
    cpras      = np.array([r.cpra_score for r in recipients])

    # Col 0: BSA similarity — already [0,1]
    scores[:, 0] = bsa_sims

    # Col 1: Urgency — INVERTED (1=most urgent→1.0, 6=least→0.0)
    scores[:, 1] = (6.0 - urgencies) / 5.0

    # Col 2: Waiting time — min-max (longer wait = higher priority)
    wt_min, wt_max = wait_times.min(), wait_times.max()
    scores[:, 2] = (wait_times - wt_min) / (wt_max - wt_min) if wt_max > wt_min else 0.5

    # Col 3: Distance time — INVERTED (closer = better for organ viability)
    d_min, d_max = distances.min(), distances.max()
    scores[:, 3] = 1.0 - (distances - d_min) / (d_max - d_min) if d_max > d_min else 0.5

    # Col 4: Pediatric bonus — binary
    scores[:, 4] = pediatrics

    # Col 5: CPRA — scale 0–100 → 0–1 (higher = harder to match = more priority)
    scores[:, 5] = cpras / 100.0

    return scores


def compute_composite_scores(
    recipients: List[Recipient],
    weights: MatchingWeights,
) -> np.ndarray:
    """Weighted sum of normalized factors → one scalar per candidate."""
    scores = normalize_scores(recipients)
    w = np.array([
        weights.bsa_similarity, weights.urgency, weights.waiting_time,
        weights.distance, weights.pediatric, weights.cpra,
    ])
    w /= w.sum()
    return scores @ w


# =============================================================================
# 8. QAOA ENGINE  (shot-based, from-scratch — swap for Qiskit version if desired)
# =============================================================================

def _bitstring_to_indices(z: int, n: int) -> List[int]:
    return [i for i in range(n) if (z >> i) & 1]


def _build_cost_diagonal(composite_scores, penalty_strength):
    n = len(composite_scores)
    dim = 2 ** n
    cost = np.zeros(dim)
    for z in range(dim):
        selected = _bitstring_to_indices(z, n)
        k = len(selected)
        reward = sum(composite_scores[i] for i in selected)
        penalty = penalty_strength * (k - 1) ** 2
        cost[z] = reward - penalty
    return cost


def _evolve_qaoa_state(cost_diagonal, gammas, betas, n):
    dim = 2 ** n
    p = len(gammas)
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    for l in range(p):
        state = np.exp(-1j * gammas[l] * cost_diagonal) * state
        cos_b = np.cos(betas[l])
        sin_b = np.sin(betas[l])
        for i in range(n):
            new_state = np.zeros_like(state)
            for z in range(dim):
                z_flipped = z ^ (1 << i)
                new_state[z] += cos_b * state[z] - 1j * sin_b * state[z_flipped]
            state = new_state

    return state


def _sample_shots(state, n_shots):
    probs = np.abs(state) ** 2
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return np.random.choice(len(probs), size=n_shots, p=probs)


def _cvar_objective(costs, alpha=0.25):
    sorted_costs = np.sort(costs)[::-1]
    k = max(1, int(np.ceil(alpha * len(sorted_costs))))
    return sorted_costs[:k].mean()


def qaoa_optimize_shots(
    composite_scores, p=3, penalty_strength=10.0,
    n_shots=4096, n_shots_final=16384, cvar_alpha=0.25, n_restarts=20,
):
    n = len(composite_scores)
    dim = 2 ** n
    cost_diagonal = _build_cost_diagonal(composite_scores, penalty_strength)
    eval_count = [0]

    def shot_objective(params):
        gammas, betas = params[:p], params[p:]
        state = _evolve_qaoa_state(cost_diagonal, gammas, betas, n)
        samples = _sample_shots(state, n_shots)
        costs = cost_diagonal[samples]
        eval_count[0] += 1
        return -_cvar_objective(costs, cvar_alpha)

    best_params = None
    best_value = float('inf')

    print(f"\n  [QAOA] p={p}, shots={n_shots}, CVaR α={cvar_alpha}, restarts={n_restarts}")

    for restart in range(n_restarts):
        result = minimize(
            shot_objective,
            np.random.uniform(0, 2 * np.pi, size=2 * p),
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.5},
        )
        if result.fun < best_value:
            best_value = result.fun
            best_params = result.x

    print(f"  [QAOA] Done. {eval_count[0]} evaluations.")

    # Final measurement
    gammas_opt, betas_opt = best_params[:p], best_params[p:]
    final_state = _evolve_qaoa_state(cost_diagonal, gammas_opt, betas_opt, n)
    final_samples = _sample_shots(final_state, n_shots_final)

    raw_counts = Counter(int(s) for s in final_samples)

    valid_counts: Dict[int, int] = {}
    invalid_count = 0
    invalid_breakdown: Dict[int, int] = {}

    for z, count in raw_counts.items():
        selected = _bitstring_to_indices(z, n)
        k = len(selected)
        if k == 1:
            valid_counts[selected[0]] = valid_counts.get(selected[0], 0) + count
        else:
            invalid_count += count
            invalid_breakdown[k] = invalid_breakdown.get(k, 0) + count

    total_valid = sum(valid_counts.values())

    if valid_counts:
        ranked = sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)
        best_candidate = ranked[0][0]
        best_votes = ranked[0][1]
    else:
        best_candidate = int(np.argmax(composite_scores))
        best_votes = 0

    winner_fraction = best_votes / n_shots_final
    valid_fraction = total_valid / n_shots_final
    se = np.sqrt(winner_fraction * (1 - winner_fraction) / n_shots_final)

    runner_up_idx = ranked[1][0] if len(ranked) >= 2 else None
    runner_up_votes = ranked[1][1] if len(ranked) >= 2 else 0
    margin = (best_votes - runner_up_votes) / n_shots_final

    final_costs = cost_diagonal[final_samples]

    info = {
        'optimal_params': best_params,
        'qaoa_depth': p,
        'n_candidates': n,
        'n_shots_final': n_shots_final,
        'cvar_alpha': cvar_alpha,
        'penalty_strength': penalty_strength,
        'total_circuit_evaluations': eval_count[0],
        'valid_vote_counts': valid_counts,
        'invalid_shot_count': invalid_count,
        'invalid_breakdown': invalid_breakdown,
        'total_valid_shots': total_valid,
        'total_invalid_shots': invalid_count,
        'valid_fraction': valid_fraction,
        'winner_votes': best_votes,
        'winner_fraction': winner_fraction,
        'winner_std_error': se,
        'winner_95ci': (max(0, winner_fraction - 1.96*se), min(1, winner_fraction + 1.96*se)),
        'runner_up_index': runner_up_idx,
        'runner_up_votes': runner_up_votes,
        'win_margin': margin,
        'cost_mean': final_costs.mean(),
        'cost_std': final_costs.std(),
        'cost_max': final_costs.max(),
        'cost_min': final_costs.min(),
        'final_cvar': _cvar_objective(final_costs, cvar_alpha),
        'all_valid_probabilities': {
            idx: cnt / total_valid if total_valid > 0 else 0
            for idx, cnt in valid_counts.items()
        },
    }

    return best_candidate, composite_scores[best_candidate], info


# =============================================================================
# 9. FULL PIPELINE
# =============================================================================

def match_heart_from_csv(
    donor: Donor,
    csv_filepath: str,
    weights: MatchingWeights = None,
    max_travel_hours: float = 5.0,
    qaoa_depth: int = 5,
    penalty_strength: float = 10.0,
    n_shots: int = 4096,
    n_shots_final: int = 16384,
    cvar_alpha: float = 0.25,
    n_restarts: int = 30,
) -> dict:
    """
    Complete pipeline: CSV → filter → QAOA → winner.

    Args:
        donor: the organ donor.
        csv_filepath: path to the recipient CSV.
        weights: policy weighting (defaults if None).
        max_travel_hours: hard cutoff for travel time.
        qaoa_depth: QAOA circuit depth p.
        penalty_strength: constraint penalty λ.
        n_shots: shots per optimization evaluation.
        n_shots_final: shots for the final round.
        cvar_alpha: CVaR fraction.
        n_restarts: optimizer restarts.

    Returns:
        dict with selected recipient, scores, filter log, QAOA info.
    """
    if weights is None:
        weights = MatchingWeights()

    # Step 1: Read CSV
    candidates = read_candidates_csv(csv_filepath)

    # Step 2: Filter and build Recipients
    recipients, filter_log = filter_and_build_recipients(
        donor, candidates, max_travel_hours,
    )

    if len(recipients) == 0:
        print("\n  ✗ NO COMPATIBLE RECIPIENTS after filtering.")
        return {'selected_recipient': None, 'filter_log': filter_log}

    if len(recipients) == 1:
        print(f"\n  Only one compatible recipient: {recipients[0].name}")
        return {
            'selected_recipient': recipients[0],
            'selected_index': 0,
            'composite_scores': compute_composite_scores(recipients, weights),
            'filter_log': filter_log,
            'qaoa_info': None,
        }

    # Step 3: Compute composite scores
    composite = compute_composite_scores(recipients, weights)
    classical_best = int(np.argmax(composite))

    # Step 4: Run QAOA
    best_idx, best_score, info = qaoa_optimize_shots(
        composite,
        p=qaoa_depth,
        penalty_strength=penalty_strength,
        n_shots=n_shots,
        n_shots_final=n_shots_final,
        cvar_alpha=cvar_alpha,
        n_restarts=n_restarts,
    )

    return {
        'selected_recipient': recipients[best_idx],
        'selected_index': best_idx,
        'all_recipients': recipients,
        'composite_scores': composite,
        'normalized_factors': normalize_scores(recipients),
        'qaoa_info': info,
        'filter_log': filter_log,
        'classical_best_index': classical_best,
        'qaoa_matches_classical': best_idx == classical_best,
    }


# =============================================================================
# 10. REPORTING
# =============================================================================

def print_report(result: dict, weights: MatchingWeights):
    """Pretty-print the full pipeline results."""
    if result.get('selected_recipient') is None:
        print("\n  No recipient selected — all candidates were filtered out.")
        return

    factor_names = [
        "BSA Match", "Urgency", "Waiting Time",
        "Distance (inv)", "Pediatric", "CPRA"
    ]
    w_arr = np.array([
        weights.bsa_similarity, weights.urgency, weights.waiting_time,
        weights.distance, weights.pediatric, weights.cpra,
    ])
    w_arr /= w_arr.sum()
    info = result.get('qaoa_info')
    recipients = result['all_recipients']

    print("\n" + "=" * 85)
    print("  QAOA HEART-ORGAN MATCHING  —  Full Pipeline Results")
    print("=" * 85)

    print("\n  POLICY WEIGHTS:")
    for name, w in zip(factor_names, w_arr):
        print(f"    {name:<20s}: {w:.3f}")

    norm = result['normalized_factors']
    comp = result['composite_scores']
    votes = info['valid_vote_counts'] if info else {}
    total_valid = info['total_valid_shots'] if info else 0

    print(f"\n  SURVIVING CANDIDATES ({len(recipients)}):")
    print(f"  {'#':<4} {'Name':<20} ", end="")
    for fn in factor_names:
        print(f"{fn[:6]:>7}", end="")
    print(f"  {'Score':>7}  {'Votes':>7}  {'Prob':>7}")
    print("  " + "-" * 110)

    for i, r in enumerate(recipients):
        marker = " ★" if i == result['selected_index'] else "  "
        v = votes.get(i, 0)
        prob = v / total_valid if total_valid > 0 else 0
        print(f"  {i:<4} {r.name:<20} ", end="")
        for j in range(6):
            print(f"{norm[i, j]:>7.3f}", end="")
        print(f"  {comp[i]:>7.4f}  {v:>7,}  {prob:>7.4f}{marker}")

    if info:
        print(f"\n  MEASUREMENT STATISTICS:")
        print(f"    Valid shots:      {info['total_valid_shots']:>8,} / "
              f"{info['n_shots_final']:,}  ({info['valid_fraction']:.1%})")
        print(f"    Invalid shots:    {info['total_invalid_shots']:>8,}")

        ci = info['winner_95ci']
        print(f"\n  WINNER ANALYSIS:")
        print(f"    ★ Selected:       {result['selected_recipient'].name}")
        print(f"    Composite score:  {comp[result['selected_index']]:.4f}")
        print(f"    Votes:            {info['winner_votes']:,} / "
              f"{info['total_valid_shots']:,} valid")
        print(f"    Win probability:  {info['winner_fraction']:.4f}  "
              f"± {info['winner_std_error']:.4f}  "
              f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")

        print(f"\n  SHOT HISTOGRAM:")
        max_votes = max(votes.values()) if votes else 1
        bar_width = 40
        for i in range(len(recipients)):
            v = votes.get(i, 0)
            bar_len = int(bar_width * v / max_votes) if max_votes > 0 else 0
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            marker = " ★" if i == result['selected_index'] else "  "
            print(f"    {recipients[i].name:<20} |{bar}| {v:>6,}{marker}")

        print(f"\n  CLASSICAL VERIFICATION:")
        print(f"    Classical optimum:  #{result['classical_best_index']} "
              f"({recipients[result['classical_best_index']].name})")
        print(f"    QAOA agrees:       "
              f"{'✓ YES' if result['qaoa_matches_classical'] else '✗ NO'}")

    print("=" * 85)


# =============================================================================
# 11. DEMO
# =============================================================================

def main():
    # ---- Define the donor ----
    donor = Donor(
        abo="A+",
        age=35,
        bsa=1.80,
        latitude=38.8838,    # Virginia Hospital Center, Arlington, VA
        longitude=-77.1050,
    )

    # ---- Path to recipient CSV ----
    csv_path = os.path.join(os.path.dirname(__file__) or '.', 'recipients.csv')

    # ---- Run the full pipeline ----
    np.random.seed(42)
    t0 = time.time()

    result = match_heart_from_csv(
        donor=donor,
        csv_filepath=csv_path,
        weights=MatchingWeights(
            bsa_similarity=0.20,
            urgency=0.25,
            waiting_time=0.15,
            distance=0.15,
            pediatric=0.10,
            cpra=0.15,
        ),
        max_travel_hours=5.0,
        qaoa_depth=5,
        penalty_strength=10.0,
        n_shots=4096,
        n_shots_final=16384,
        cvar_alpha=0.25,
        n_restarts=30,
    )

    elapsed = time.time() - t0
    print_report(result, MatchingWeights(0.20, 0.25, 0.15, 0.15, 0.10, 0.15))
    print(f"\n  Total runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()