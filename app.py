import csv
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from math import radians, sin, cos, sqrt, atan2
import time
import sys
import os

import io
import threading
import webbrowser
import traceback

# ---- FastAPI ----
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

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
    waiting_time: float   = 0.18   # fairness
    distance: float       = 0.10   # organ viability (ischemic time)
    pediatric: float      = 0.10   # pediatric priority
    cpra: float           = 0.17   # sensitized patient priority


# =============================================================================
# 2. CSV READING
# =============================================================================

def read_candidates_csv(filepath: str) -> List[CsvCandidate]:
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


def read_candidates_from_text(text: str) -> List[CsvCandidate]:
    candidates = []

    reader = csv.DictReader(io.StringIO(text))
    reader.fieldnames = [col.strip().lower().replace(' ', '_') for col in reader.fieldnames]

    for row in reader:
        try:
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
        except (KeyError, ValueError) as e:
            continue

    return candidates

# =============================================================================
# 3. ABO BLOOD TYPE COMPATIBILITY
# =============================================================================

# Maps each recipient base type to set of compatible donor base types
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

    distance_km = _haversine_km(donor_lat, donor_lon, recipient_lat, recipient_lon)
    return distance_km / TRANSPORT_SPEED_KMH


# =============================================================================
# 5. BSA SIMILARITY
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

    for cand in candidates:
        # ---- Filter 1: ABO compatibility ----
        if not is_abo_compatible(donor.abo, cand.abo):
            filter_log['eliminated_abo'].append(cand.name)
            continue

        # ---- Filter 2: Travel time ----
        travel_hours = estimate_travel_time_hours(
            donor.latitude, donor.longitude,
            cand.latitude, cand.longitude,
        )

        if travel_hours > max_travel_hours:
            filter_log['eliminated_distance'].append(cand.name)
            continue

        # ---- Filter 3: BSA compatibility ----
        if not is_bsa_compatible(donor.bsa, cand.bsa):
            bsa_ratio = donor.bsa / cand.bsa * 100
            filter_log['eliminated_bsa'].append(cand.name)
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

    return recipients, filter_log


# =============================================================================
# 7. SCORE NORMALIZATION  
# =============================================================================

def normalize_scores(recipients: List[Recipient]) -> np.ndarray:
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

    # Col 1: Urgency — INVERTED (1=most urgent->1.0, 6=least→0.0)
    scores[:, 1] = (6.0 - urgencies) / 5.0

    # Col 2: Waiting time — min-max (longer wait = higher priority)
    wt_min, wt_max = wait_times.min(), wait_times.max()
    scores[:, 2] = (wait_times - wt_min) / (wt_max - wt_min) if wt_max > wt_min else 0.5

    # Col 3: Distance time — INVERTED (closer = better for organ viability)
    d_min, d_max = distances.min(), distances.max()
    scores[:, 3] = 1.0 - (distances - d_min) / (d_max - d_min) if d_max > d_min else 0.5

    # Col 4: Pediatric bonus — binary
    scores[:, 4] = pediatrics

    # Col 5: CPRA — scale 0–100 -> 0–1 (higher = harder to match = more priority)
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
# 8. QAOA ENGINE  —  Qiskit + AerSimulator
# =============================================================================

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator


def _bitstring_to_indices(z: int, n: int) -> List[int]:
    """Return which qubits are |1⟩ in the n-bit integer z."""
    return [i for i in range(n) if (z >> i) & 1]


def _build_cost_diagonal(composite_scores, penalty_strength):
    """
    Build the full diagonal cost vector for post-processing shots.
    C(z) = Σ_i c_i·z_i − λ·(Σ_i z_i − 1)²
    """
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


def _compute_hamiltonian_coefficients(
    composite_scores: np.ndarray,
    penalty_strength: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Decompose cost Hamiltonian into Pauli Z/ZZ coefficients.

    C = Σ_i (c_i + λ) Z̃_i  −  λ Σ_{i<j} Z̃_i Z̃_j  −  λ

    Substituting Z̃ = (I−Z)/2 yields linear (Z_i) and quadratic (Z_i Z_j)
    coefficients that map directly to RZ and RZZ gate angles.

    Returns:
        h_linear:    shape (n,)    — coefficient of Z_i
        h_quadratic: shape (n, n)  — coefficient of Z_i Z_j (upper triangle)
        h_offset:    float         — constant (ignorable global phase)
    """
    n = len(composite_scores)
    lam = penalty_strength

    h_linear = np.zeros(n)
    offset = 0.0

    # Linear terms from (c_i + λ) · (I - Z_i) / 2
    for i in range(n):
        coeff = composite_scores[i] + lam
        h_linear[i] = -coeff / 2.0
        offset += coeff / 2.0

    # Quadratic terms from -λ · (I - Z_i - Z_j + Z_i Z_j) / 4
    h_quadratic = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            h_quadratic[i, j] = -lam / 4.0
            h_linear[i] += lam / 4.0
            h_linear[j] += lam / 4.0
            offset -= lam / 4.0

    offset -= lam
    return h_linear, h_quadratic, offset


def _build_qaoa_circuit(
    n: int,
    p: int,
    h_linear: np.ndarray,
    h_quadratic: np.ndarray,
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    """
    Build a parameterized QAOA circuit using Qiskit gates.

    Architecture: |0⟩^n → H^⊗n → [Cost Layer → Mixer Layer] × p → Measure

    Cost layer:   RZ(2γ·h_linear[i]) per qubit,
                  RZZ(2γ·h_quadratic[i,j]) per pair (decomposed as CX-RZ-CX).
    Mixer layer:  RX(2β) per qubit.
    """
    gammas = ParameterVector('γ', p)
    betas  = ParameterVector('β', p)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.barrier()

    for l in range(p):
        # ---- Cost unitary e^{-iγC} ----
        for i in range(n):
            if abs(h_linear[i]) > 1e-10:
                qc.rz(2 * gammas[l] * h_linear[i], i)

        for i in range(n):
            for j in range(i + 1, n):
                if abs(h_quadratic[i, j]) > 1e-10:
                    theta = 2 * gammas[l] * h_quadratic[i, j]
                    qc.cx(i, j)
                    qc.rz(theta, j)
                    qc.cx(i, j)

        qc.barrier()

        # ---- Mixer unitary e^{-iβB} ----
        for i in range(n):
            qc.rx(2 * betas[l], i)

        qc.barrier()

    qc.measure(range(n), range(n))
    return qc, gammas, betas


def _cvar_from_counts(
    counts: dict,
    cost_diagonal: np.ndarray,
    n: int,
    alpha: float = 0.25,
) -> float:
    """
    Compute CVaR_α from Qiskit measurement counts.

    Qiskit bitstrings are big-endian (leftmost = highest qubit),
    so we reverse to match our little-endian cost_diagonal indexing.
    """
    cost_count_pairs = []
    for bitstring, count in counts.items():
        z_le = int(bitstring[::-1], 2)
        cost_count_pairs.append((cost_diagonal[z_le], count))

    cost_count_pairs.sort(key=lambda x: x[0], reverse=True)

    total_shots = sum(c for _, c in cost_count_pairs)
    k = max(1, int(np.ceil(alpha * total_shots)))

    accumulated = 0
    weighted_sum = 0.0
    for cost_val, count in cost_count_pairs:
        take = min(count, k - accumulated)
        weighted_sum += cost_val * take
        accumulated += take
        if accumulated >= k:
            break

    return weighted_sum / accumulated if accumulated > 0 else 0.0


def qaoa_optimize_qiskit(
    composite_scores, p=3, penalty_strength=10.0,
    n_shots=4096, n_shots_final=16384, cvar_alpha=0.25, n_restarts=20,
):
   
    n = len(composite_scores)

    # Hamiltonian decomposition
    h_linear, h_quadratic, h_offset = _compute_hamiltonian_coefficients(
        composite_scores, penalty_strength
    )

    # Build circuit once
    qc, gamma_params, beta_params = _build_qaoa_circuit(n, p, h_linear, h_quadratic)


    # Initialize simulator
    simulator = AerSimulator(method='automatic')

    # Cost diagonal for post-processing
    cost_diagonal = _build_cost_diagonal(composite_scores, penalty_strength)


    # ---- Shot-based objective ----
    def qiskit_objective(params):
        gamma_vals = params[:p]
        beta_vals  = params[p:]

        param_dict = {}
        for l in range(p):
            param_dict[gamma_params[l]] = gamma_vals[l]
            param_dict[beta_params[l]]  = beta_vals[l]

        bound_qc = qc.assign_parameters(param_dict)
        job = simulator.run(bound_qc, shots=n_shots)
        counts = job.result().get_counts()
        cvar = _cvar_from_counts(counts, cost_diagonal, n, cvar_alpha)

        return -cvar

    # ---- Classical optimization with random restarts ----
    best_params = None
    best_value = float('inf')


    for restart in range(n_restarts):
        result = minimize(
            qiskit_objective,
            np.random.uniform(0, 2 * np.pi, size=2 * p),
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.5},
        )
        if result.fun < best_value:
            best_value = result.fun
            best_params = result.x


    # ---- Final measurement round ----
    gamma_vals = best_params[:p]
    beta_vals  = best_params[p:]
    param_dict = {}
    for l in range(p):
        param_dict[gamma_params[l]] = gamma_vals[l]
        param_dict[beta_params[l]]  = beta_vals[l]

    bound_qc = qc.assign_parameters(param_dict)

    job = simulator.run(bound_qc, shots=n_shots_final)
    final_counts = job.result().get_counts()

    # ---- Parse measurement histogram ----
    valid_counts: Dict[int, int] = {}
    invalid_count = 0
    invalid_breakdown: Dict[int, int] = {}

    for bitstring, count in final_counts.items():
        z = int(bitstring[::-1], 2)  # reverse Qiskit big-endian
        selected = _bitstring_to_indices(z, n)
        k = len(selected)
        if k == 1:
            valid_counts[selected[0]] = valid_counts.get(selected[0], 0) + count
        else:
            invalid_count += count
            invalid_breakdown[k] = invalid_breakdown.get(k, 0) + count

    total_valid = sum(valid_counts.values())

    # ---- Majority vote ----
    if valid_counts:
        ranked = sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)
        best_candidate = ranked[0][0]
        best_votes = ranked[0][1]
    else:
        best_candidate = int(np.argmax(composite_scores))
        best_votes = 0

    info = {
        'valid_vote_counts': valid_counts,
        'total_valid_shots': total_valid,
        'winner_votes': best_votes,
    }

    return best_candidate, composite_scores[best_candidate], info


# =============================================================================
# 9. FASTAPI WEB APP
# =============================================================================

app = FastAPI(title="QAOA Heart-Organ Matching")

# Load recipients CSV once at startup
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recipients.csv')
CANDIDATES = read_candidates_csv(CSV_PATH)


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return FRONTEND_HTML

@app.get("/style.css")
def serve_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'style.css')
    return FileResponse(css_path, media_type="text/css")

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/match")
async def api_match(
    donor_abo: str = Form(...),
    donor_age: int = Form(...),
    donor_bsa: float = Form(...),
    donor_latitude: float = Form(...),
    donor_longitude: float = Form(...),
    w_bsa: float = Form(0.20),
    w_urgency: float = Form(0.25),
    w_waiting: float = Form(0.15),
    w_distance: float = Form(0.15),
    w_pediatric: float = Form(0.10),
    w_cpra: float = Form(0.15),
    qaoa_depth: int = Form(5),
    penalty_strength: float = Form(10.0),
    n_shots: int = Form(4096),
    n_shots_final: int = Form(16384),
    cvar_alpha: float = Form(0.25),
    n_restarts: int = Form(20),
    max_travel_hours: float = Form(5.0),
):
    t0 = time.time()

    try:
        # ---- Parse donor ----
        donor = Donor(
            abo=donor_abo.strip().upper(),
            age=donor_age,
            bsa=donor_bsa,
            latitude=donor_latitude,
            longitude=donor_longitude,
        )

        # ---- Use pre-loaded candidates ----
        candidates = CANDIDATES

        if len(candidates) == 0:
            raise HTTPException(status_code=400, detail="recipients.csv contains no valid candidates.")

        # ---- Weights ----
        weights = MatchingWeights(
            bsa_similarity=w_bsa,
            urgency=w_urgency,
            waiting_time=w_waiting,
            distance=w_distance,
            pediatric=w_pediatric,
            cpra=w_cpra,
        )

        # ---- Filter (uses filter_and_build_recipients from section 6) ----
        recipients, filter_log = filter_and_build_recipients(
            donor, candidates, max_travel_hours,
        )

        # Build eliminated list for the response
        eliminated = []
        for name in filter_log.get('eliminated_abo', []):
            cand = next((c for c in candidates if c.name == name), None)
            eliminated.append({
                "name": name,
                "reason": "ABO Incompatible",
                "detail": f"Donor {donor.abo} -> Recipient {cand.abo if cand else '?'}",
            })
        for name in filter_log.get('eliminated_distance', []):
            eliminated.append({
                "name": name,
                "reason": "Too Far",
                "detail": f"Exceeds {max_travel_hours}h travel time",
            })
        for name in filter_log.get('eliminated_bsa', []):
            eliminated.append({
                "name": name,
                "reason": "BSA Undersized",
                "detail": f"Donor BSA {donor.bsa:.2f} < 70% of recipient's",
            })

        if len(recipients) == 0:
            raise HTTPException(
                status_code=422,
                detail=f"All {len(candidates)} candidates were eliminated by filters.",
            )

        # ---- Compute scores (uses normalize_scores + compute_composite_scores from section 7) ----
        composite = compute_composite_scores(recipients, weights)

        # ---- Single candidate — no QAOA needed ----
        if len(recipients) == 1:
            r = recipients[0]
            return {
                "success": True,
                "winner_name": r.name,
                "winner_composite_score": round(float(composite[0]), 4),
                "total_candidates_csv": len(candidates),
                "candidates_after_filter": 1,
                "eliminated": eliminated,
                "candidates": [{
                    "name": r.name,
                    "composite_score": round(float(composite[0]), 4),
                    "votes": 1, "probability": 1.0, "is_winner": True,
                    "is_child": r.is_child,
                    "urgency_level": r.urgency_level,
                    "bsa_similarity": round(r.bsa_similarity, 3),
                    "distance_time_hours": round(r.distance_time_hours, 2),
                    "cpra_score": r.cpra_score,
                    "waiting_time_days": r.waiting_time_days,
                }],
                "runtime_seconds": round(time.time() - t0, 2),
            }

        # ---- Run QAOA (uses qaoa_optimize_qiskit from section 8) ----
        np.random.seed(42)
        best_idx, best_score, info = qaoa_optimize_qiskit(
            composite,
            p=qaoa_depth,
            penalty_strength=penalty_strength,
            n_shots=n_shots,
            n_shots_final=n_shots_final,
            cvar_alpha=cvar_alpha,
            n_restarts=n_restarts,
        )


        votes = info['valid_vote_counts']
        total_valid = info['total_valid_shots']

        candidate_results = []
        for i, r in enumerate(recipients):
            v = votes.get(i, 0)
            prob = v / total_valid if total_valid > 0 else 0
            candidate_results.append({
                "name": r.name,
                "composite_score": round(float(composite[i]), 4),
                "votes": v,
                "probability": round(prob, 4),
                "is_winner": (i == best_idx),
                "is_child": r.is_child,
                "urgency_level": r.urgency_level,
                "bsa_similarity": round(r.bsa_similarity, 3),
                "distance_time_hours": round(r.distance_time_hours, 2),
                "cpra_score": r.cpra_score,
                "waiting_time_days": r.waiting_time_days,
            })

        return {
            "success": True,
            "winner_name": recipients[best_idx].name,
            "winner_composite_score": round(float(composite[best_idx]), 4),
            "total_candidates_csv": len(candidates),
            "candidates_after_filter": len(recipients),
            "eliminated": eliminated,
            "candidates": candidate_results,
            "runtime_seconds": round(time.time() - t0, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 10. FRONTEND HTML
# =============================================================================

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QAOA Heart-Organ Matching</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:        #05080f;
  --surface:   #090e1a;
  --border:    rgba(180,210,255,0.08);
  --border-hi: rgba(180,210,255,0.18);
  --text:      #d8e8ff;
  --muted:     #5a7299;
  --accent:    #4f9fff;
  --accent2:   #c084fc;
  --red:       #ff4f6a;
  --green:     #22d3a0;
  --warn:      #f59e0b;
  --glow:      rgba(79,159,255,0.15);
  --glow2:     rgba(192,132,252,0.10);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  overflow-x: hidden;
}

/* ── Animated background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 10% 0%,   rgba(79,159,255,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 90% 100%,  rgba(192,132,252,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 40% 40% at 50% 50%,   rgba(34,211,160,0.03) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

/* Subtle grid overlay */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(79,159,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(79,159,255,0.03) 1px, transparent 1px);
  background-size: 48px 48px;
  pointer-events: none;
  z-index: 0;
}

/* ── Header ── */
header {
  position: relative;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 40px;
  border-bottom: 1px solid var(--border);
  backdrop-filter: blur(12px);
  background: rgba(5,8,15,0.7);
}

.logo {
  display: flex;
  align-items: center;
  gap: 14px;
}

.logo-icon {
  width: 36px;
  height: 36px;
  position: relative;
  flex-shrink: 0;
}

.logo-icon svg {
  width: 100%;
  height: 100%;
  filter: drop-shadow(0 0 8px rgba(79,159,255,0.6));
}

.logo-text {
  font-family: 'DM Serif Display', serif;
  font-size: 18px;
  letter-spacing: 0.02em;
  color: var(--text);
  line-height: 1.1;
}

.logo-text span {
  display: block;
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  font-weight: 300;
  color: var(--muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 2px;
}

.status-pill {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--green);
  background: rgba(34,211,160,0.06);
  border: 1px solid rgba(34,211,160,0.2);
  border-radius: 20px;
  padding: 6px 14px;
}

.status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Navigation ── */
nav {
  position: relative;
  z-index: 10;
  display: flex;
  gap: 0;
  padding: 0 40px;
  border-bottom: 1px solid var(--border);
  background: rgba(5,8,15,0.5);
  backdrop-filter: blur(8px);
}

nav button {
  background: none;
  border: none;
  color: var(--muted);
  padding: 16px 22px;
  cursor: pointer;
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  border-bottom: 2px solid transparent;
  transition: all 0.2s;
  position: relative;
}

nav button:hover { color: var(--text); }

nav button.on {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

/* ── Main layout ── */
main {
  position: relative;
  z-index: 1;
  max-width: 780px;
  margin: 0 auto;
  padding: 40px 40px 80px;
}

/* ── Pages ── */
.page { display: none; }
.page.on { display: block; animation: fadeIn 0.3s ease; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ── About page ── */
.about-hero {
  margin-bottom: 40px;
  padding-bottom: 40px;
  border-bottom: 1px solid var(--border);
}

.about-hero h2 {
  font-family: 'DM Serif Display', serif;
  font-size: 38px;
  line-height: 1.15;
  color: var(--text);
  margin-bottom: 16px;
}

.about-hero h2 em {
  font-style: italic;
  color: var(--accent);
}

.about-hero p {
  font-size: 15px;
  line-height: 1.75;
  color: var(--muted);
  max-width: 580px;
}

.about-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 32px;
}

.about-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px;
  transition: border-color 0.2s;
}

.about-card:hover { border-color: var(--border-hi); }

.about-card-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 8px;
}

.about-card h3 {
  font-family: 'DM Serif Display', serif;
  font-size: 16px;
  color: var(--text);
  margin-bottom: 8px;
}

.about-card p {
  font-size: 13px;
  line-height: 1.6;
  color: var(--muted);
}

/* ── Section label ── */
.section-label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 10px;
}

.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Form ── */
.form-block {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 24px;
  margin-bottom: 16px;
}

.row   { display: grid; grid-template-columns: 1fr 1fr;     gap: 16px; }
.row3  { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }

.field { display: flex; flex-direction: column; }

label {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 7px;
}

input, select {
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
  padding: 9px 12px;
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  outline: none;
  transition: border-color 0.2s, box-shadow 0.2s;
  -webkit-appearance: none;
}

input:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(79,159,255,0.1);
}

select option { background: #0e1525; }

hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 20px 0;
}

details summary {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  cursor: pointer;
  user-select: none;
  transition: color 0.2s;
}

details summary:hover { color: var(--accent); }
details[open] summary { color: var(--accent); margin-bottom: 16px; }

.adv-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

/* ── Run button ── */
.btn-run {
  width: 100%;
  background: linear-gradient(135deg, var(--accent) 0%, #3a8aff 100%);
  border: none;
  border-radius: 10px;
  color: #fff;
  padding: 14px;
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  cursor: pointer;
  transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
  box-shadow: 0 4px 24px rgba(79,159,255,0.25);
  position: relative;
  overflow: hidden;
  margin-top: 4px;
}

.btn-run::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.12) 0%, transparent 60%);
}

.btn-run:hover:not(:disabled) {
  opacity: 0.9;
  transform: translateY(-1px);
  box-shadow: 0 8px 32px rgba(79,159,255,0.35);
}

.btn-run:active:not(:disabled) { transform: translateY(0); }
.btn-run:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

/* ── Spinner ── */
.spinner {
  display: none;
  width: 14px; height: 14px;
  border: 2px solid rgba(255,255,255,0.3);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  margin: 0 auto;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Error ── */
.err {
  min-height: 20px;
  color: var(--red);
  font-family: 'DM Mono', monospace;
  font-size: 12px;
  margin-top: 10px;
  padding: 0 2px;
}

/* ── Results ── */
#output { margin-top: 32px; }

.winner-card {
  background: linear-gradient(135deg, rgba(79,159,255,0.07) 0%, rgba(192,132,252,0.05) 100%);
  border: 1px solid rgba(79,159,255,0.3);
  border-radius: 16px;
  padding: 28px;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}

.winner-card::before {
  content: '';
  position: absolute;
  top: -40px; right: -40px;
  width: 160px; height: 160px;
  background: radial-gradient(circle, rgba(79,159,255,0.12) 0%, transparent 70%);
  pointer-events: none;
}

.winner-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 10px;
}

.winner-name {
  font-family: 'DM Serif Display', serif;
  font-size: 30px;
  color: var(--text);
  margin-bottom: 12px;
  line-height: 1.1;
}

.winner-meta {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.winner-stat {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.winner-stat-val {
  font-family: 'DM Mono', monospace;
  font-size: 18px;
  font-weight: 500;
  color: var(--accent);
}

.winner-stat-label {
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.divider-meta {
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  color: var(--muted);
  margin-bottom: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}

/* ── Candidate cards ── */
.cand {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  margin-bottom: 8px;
  transition: border-color 0.2s, transform 0.15s;
  cursor: default;
}

.cand:hover { border-color: var(--border-hi); transform: translateX(2px); }
.cand.win   { border-color: rgba(79,159,255,0.35); }

.cand-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.cand-name {
  font-size: 14px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.cand-score {
  font-family: 'DM Mono', monospace;
  font-size: 13px;
  color: var(--accent);
}

.tag {
  font-family: 'DM Mono', monospace;
  font-size: 9px;
  padding: 2px 8px;
  border-radius: 10px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: 500;
}

.tag.winner-tag { background: rgba(79,159,255,0.12); color: var(--accent); border: 1px solid rgba(79,159,255,0.25); }
.tag.child-tag  { background: rgba(192,132,252,0.10); color: var(--accent2); border: 1px solid rgba(192,132,252,0.2); }

.cand-meta {
  font-size: 12px;
  color: var(--muted);
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.cand-meta span {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* ── Vote bar ── */
.vote-bar-wrap {
  margin-top: 8px;
  height: 3px;
  background: rgba(255,255,255,0.05);
  border-radius: 2px;
  overflow: hidden;
}

.vote-bar {
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
  transition: width 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.win .vote-bar { background: linear-gradient(90deg, var(--accent), #7dd3fc); }

/* ── Eliminated section ── */
.elim-section {
  margin-top: 24px;
}

.elim-toggle {
  background: none;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--muted);
  padding: 9px 16px;
  font-family: 'DM Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  cursor: pointer;
  width: 100%;
  text-align: left;
  transition: border-color 0.2s, color 0.2s;
}
.elim-toggle:hover { border-color: var(--border-hi); color: var(--text); }

.elim-list { margin-top: 8px; display: none; }
.elim-list.open { display: block; }

.elim-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 12px;
  border-radius: 8px;
  font-size: 12px;
  margin-bottom: 4px;
  background: rgba(255,79,106,0.04);
  border: 1px solid rgba(255,79,106,0.1);
}

.elim-name { color: var(--muted); }
.elim-reason {
  font-family: 'DM Mono', monospace;
  font-size: 10px;
  color: var(--red);
  letter-spacing: 0.05em;
}
.elim-detail { font-size: 11px; color: rgba(255,79,106,0.5); }

/* ── Responsive ── */
@media (max-width: 600px) {
  header { padding: 16px 20px; }
  nav { padding: 0 20px; }
  main { padding: 24px 20px 60px; }
  .about-cards { grid-template-columns: 1fr; }
  .row, .row3 { grid-template-columns: 1fr; }
  .adv-grid { grid-template-columns: 1fr; }
  .winner-meta { gap: 14px; }
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M18 6C18 6 8 10 8 18C8 22.4 11.6 26 16 26H20C24.4 26 28 22.4 28 18C28 10 18 6 18 6Z" stroke="#4f9fff" stroke-width="1.5" fill="rgba(79,159,255,0.1)"/>
        <circle cx="18" cy="18" r="3" fill="#4f9fff" opacity="0.8"/>
        <line x1="18" y1="10" x2="18" y2="26" stroke="#4f9fff" stroke-width="0.75" stroke-dasharray="2 2" opacity="0.4"/>
        <line x1="10" y1="18" x2="26" y2="18" stroke="#4f9fff" stroke-width="0.75" stroke-dasharray="2 2" opacity="0.4"/>
        <circle cx="18" cy="10" r="1.5" fill="#c084fc" opacity="0.7"/>
        <circle cx="18" cy="26" r="1.5" fill="#c084fc" opacity="0.7"/>
        <circle cx="10" cy="18" r="1.5" fill="#c084fc" opacity="0.7"/>
        <circle cx="26" cy="18" r="1.5" fill="#c084fc" opacity="0.7"/>
      </svg>
    </div>
    <div class="logo-text">
      QAOA Matching
      <span>Quantum Heart Transplant</span>
    </div>
  </div>
  <div class="status-pill">
    <div class="status-dot"></div>
    System Online
  </div>
</header>

<nav>
  <button class="on" onclick="tab(this,'about')">About</button>
  <button onclick="tab(this,'match')">Match</button>
</nav>

<main>

  <!-- ── ABOUT ── -->
  <div id="about" class="page on">
    <div class="about-hero">
      <h2>Quantum-optimised<br><em>organ matching</em></h2>
      <p>This system uses the Quantum Approximate Optimisation Algorithm (QAOA) running on Qiskit's AerSimulator to select the optimal heart transplant recipient from a pre-screened waitlist — balancing urgency, compatibility, fairness, and organ viability in a single pass.</p>
    </div>

    <div class="section-label">How it works</div>
    <div class="about-cards">
      <div class="about-card">
        <div class="about-card-label">Step 01</div>
        <h3>Hard Filtering</h3>
        <p>Candidates are eliminated by ABO blood type incompatibility, travel time exceeding the ischaemic window, and BSA size mismatch.</p>
      </div>
      <div class="about-card">
        <div class="about-card-label">Step 02</div>
        <h3>Score Encoding</h3>
        <p>Six clinical factors — urgency, BSA similarity, CPRA, wait time, distance, and paediatric status — are normalised into a cost Hamiltonian.</p>
      </div>
      <div class="about-card">
        <div class="about-card-label">Step 03</div>
        <h3>QAOA Circuit</h3>
        <p>A parameterised quantum circuit is compiled once and optimised over many shots using CVaR as the objective to focus on high-quality solutions.</p>
      </div>
      <div class="about-card">
        <div class="about-card-label">Step 04</div>
        <h3>Decision</h3>
        <p>A final high-shot measurement round produces a probability distribution; the majority-vote winner across valid (single-selection) bitstrings is selected.</p>
      </div>
    </div>
  </div>

  <!-- ── MATCH ── -->
  <div id="match" class="page">

    <div class="section-label">Donor information</div>
    <div class="form-block">
      <div class="row" style="margin-bottom:16px">
        <div class="field">
          <label>Blood Type</label>
          <select id="d_abo">
            <option>A+</option><option>A-</option><option>B+</option><option>B-</option>
            <option>AB+</option><option>AB-</option><option>O+</option><option>O-</option>
          </select>
        </div>
        <div class="field">
          <label>Age</label>
          <input type="number" id="d_age" value="35">
        </div>
      </div>
      <div class="row3" style="margin-bottom:16px">
        <div class="field">
          <label>BSA (m²)</label>
          <input type="number" id="d_bsa" value="1.80" step="0.01">
        </div>
        <div class="field">
          <label>Latitude</label>
          <input type="number" id="d_lat" value="38.8838" step="0.0001">
        </div>
        <div class="field">
          <label>Longitude</label>
          <input type="number" id="d_lon" value="-77.1050" step="0.0001">
        </div>
      </div>
      <div class="field">
        <label>Max Travel Time (hours)</label>
        <input type="number" id="max_h" value="5" step="0.5">
      </div>
    </div>

    <div class="section-label">Algorithm settings</div>
    <div class="form-block">
      <details>
        <summary>Advanced parameters</summary>
        <div class="adv-grid">
          <div class="field">
            <label>QAOA Depth</label>
            <input type="number" id="q_depth" value="5" min="1" max="10">
          </div>
          <div class="field">
            <label>Restarts</label>
            <input type="number" id="q_restarts" value="20" min="5">
          </div>
          <div class="field">
            <label>Shots / eval</label>
            <input type="number" id="q_shots" value="4096" min="512" step="512">
          </div>
          <div class="field">
            <label>CVaR α</label>
            <input type="number" id="q_alpha" value="0.25" min="0.05" max="1" step="0.05">
          </div>
        </div>
      </details>
    </div>

    <button class="btn-run" id="btn" onclick="run()">
      Run Quantum Match
    </button>
    <div class="err" id="err"></div>

    <div id="output"></div>
  </div>

</main>

<script>
function tab(el, id) {
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('on'));
  el.classList.add('on');
  document.querySelectorAll('.page').forEach(p => p.classList.remove('on'));
  document.getElementById(id).classList.add('on');
}

async function run() {
  const btn = document.getElementById('btn');
  btn.disabled = true;
  btn.textContent = 'Running…';
  document.getElementById('err').textContent = '';
  document.getElementById('output').innerHTML = '';

  const fd = new FormData();
  fd.append('donor_abo',       document.getElementById('d_abo').value);
  fd.append('donor_age',       document.getElementById('d_age').value);
  fd.append('donor_bsa',       document.getElementById('d_bsa').value);
  fd.append('donor_latitude',  document.getElementById('d_lat').value);
  fd.append('donor_longitude', document.getElementById('d_lon').value);
  fd.append('w_bsa',           '0.20');
  fd.append('w_urgency',       '0.25');
  fd.append('w_waiting',       '0.15');
  fd.append('w_distance',      '0.15');
  fd.append('w_pediatric',     '0.10');
  fd.append('w_cpra',          '0.15');
  fd.append('qaoa_depth',      document.getElementById('q_depth').value);
  fd.append('n_shots',         document.getElementById('q_shots').value);
  fd.append('n_shots_final',   '16384');
  fd.append('cvar_alpha',      document.getElementById('q_alpha').value);
  fd.append('n_restarts',      document.getElementById('q_restarts').value);
  fd.append('max_travel_hours',document.getElementById('max_h').value);
  fd.append('penalty_strength','10.0');

  try {
    const res = await fetch('/api/match', { method: 'POST', body: fd });
    const d   = await res.json();
    if (!res.ok) { document.getElementById('err').textContent = d.detail || 'Error'; return; }

    const maxVotes = Math.max(...d.candidates.map(c => c.votes), 1);
    const sorted   = [...d.candidates].sort((a, b) => b.votes - a.votes);

    let h = `
      <div class="winner-card">
        <div class="winner-eyebrow">Selected Recipient</div>
        <div class="winner-name">${d.winner_name}</div>
        <div class="winner-meta">
          <div class="winner-stat">
            <div class="winner-stat-val">${d.winner_composite_score}</div>
            <div class="winner-stat-label">Score</div>
          </div>
          <div class="winner-stat">
            <div class="winner-stat-val">${d.candidates_after_filter} / ${d.total_candidates_csv}</div>
            <div class="winner-stat-label">Passed filters</div>
          </div>
          <div class="winner-stat">
            <div class="winner-stat-val">${d.runtime_seconds}s</div>
            <div class="winner-stat-label">Runtime</div>
          </div>
        </div>
      </div>

      <div class="section-label" style="margin-top:8px">Candidates</div>
    `;

    sorted.forEach(c => {
      const pct = ((c.votes / maxVotes) * 100).toFixed(1);
      h += `
        <div class="cand ${c.is_winner ? 'win' : ''}">
          <div class="cand-top">
            <div class="cand-name">
              ${c.name}
              ${c.is_winner ? '<span class="tag winner-tag">Winner</span>' : ''}
              ${c.is_child  ? '<span class="tag child-tag">Child</span>'  : ''}
            </div>
            <div class="cand-score">${c.composite_score}</div>
          </div>
          <div class="cand-meta">
            <span>Urgency ${c.urgency_level}</span>
            <span>BSA ${c.bsa_similarity}</span>
            <span>${c.distance_time_hours}h travel</span>
            <span>CPRA ${c.cpra_score}%</span>
            <span>${c.waiting_time_days}d wait</span>
            <span>${c.votes} votes (${(c.probability * 100).toFixed(1)}%)</span>
          </div>
          <div class="vote-bar-wrap">
            <div class="vote-bar" style="width:${pct}%"></div>
          </div>
        </div>
      `;
    });

    if (d.eliminated && d.eliminated.length > 0) {
      h += `
        <div class="elim-section">
          <button class="elim-toggle" onclick="toggleElim(this)">
            ▸ ${d.eliminated.length} candidate${d.eliminated.length > 1 ? 's' : ''} eliminated by filters
          </button>
          <div class="elim-list">
            ${d.eliminated.map(e => `
              <div class="elim-item">
                <span class="elim-name">${e.name}</span>
                <span>
                  <span class="elim-reason">${e.reason}</span>
                  <span class="elim-detail"> — ${e.detail}</span>
                </span>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    document.getElementById('output').innerHTML = h;

  } catch (e) {
    document.getElementById('err').textContent = 'Connection failed: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Quantum Match';
  }
}

function toggleElim(btn) {
  const list = btn.nextElementSibling;
  list.classList.toggle('open');
  btn.textContent = list.classList.contains('open')
    ? btn.textContent.replace('▸', '▾')
    : btn.textContent.replace('▾', '▸');
}
</script>
</body>
</html>"""

# =============================================================================
# YES RUN RUN RUN ITS DONE HALLELUJOAH
# # =============================================================================

if __name__ == "__main__":
    PORT = 8000

    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")