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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Mount the static folder so /static/style.css etc. are served automatically
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at the root
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse("static/qore_web.html")

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
# YES RUN RUN RUN ITS DONE HALLELUJOAH
# # =============================================================================

if __name__ == "__main__":
    PORT = 8000

    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")