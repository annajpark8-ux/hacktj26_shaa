"""
================================================================
Heart Organ Matching via QAOA
================================================================
Mirrors the NordIQuEst Flight Scheduling QAOA Tutorial:
  https://nordiquest.net/application-library/training-material/
  qas2024/notebooks/flight_scheduling_optimization_qaoa.html

Pipeline:
  [Classical]  Sort donors by recency of death (most recent first)
  [Classical]  Hard-filter recipient pool per donor:
                 — blood type incompatibility (ABO rules)
                 — size mismatch (adult vs child)
               HLA mismatches NOT a hard filter for hearts
               (unlike kidneys); they penalise the priority score.

  [Quantum — per donor, on the filtered pool]
  Step 1.1  Build constraint matrix A  (1 row: "assign exactly 1")
  Step 1.2  QUBO:  Q = A^T A - 2 diag(A^T 1)   (tutorial formula)
                   Q -= lambda_s * diag(priority_scores)
  Step 1.3  Ising: b_i = -sum_j (Q_ij + Q_ji)   (tutorial formula)
  Step 2    Hamiltonian via generate_pauli_terms  (tutorial function)
  Step 3    QAOAAnsatz(cost_operator=H, reps=p)  (tutorial style)
            Optimise gamma/beta with COBYLA
            Sample final circuit, filter valid bitstrings
  Step 4    Post-process: pick lowest-cost valid bitstring
            Remove matched recipient from pool
================================================================
Install:
  pip install qiskit qiskit-aer scipy numpy
================================================================
"""

import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize

from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit import transpile
from qiskit_aer import AerSimulator


# ================================================================
# SECTION 0 — SYNTHETIC DATA
# ================================================================
# 6 donors x 12 recipients.
# Blood + size filtering leaves 4-8 candidates per donor, keeping
# qubit counts small for fast Aer simulation.

BLOOD_TYPES = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
BLOOD_FREQ  = [0.374, 0.066, 0.357, 0.063, 0.085, 0.015, 0.034, 0.006]

HLA_A_POOL  = [1, 2, 3, 11, 24, 25, 26, 29, 30, 31]
HLA_B_POOL  = [7, 8, 13, 14, 15, 18, 27, 35, 37, 38, 39, 40, 41, 42, 44]
HLA_DR_POOL = [1, 3, 4, 7, 8, 9, 10, 11, 12, 13]

CITIES = [
    ("New York",     40.71,  -74.01),
    ("Los Angeles",  34.05, -118.24),
    ("Chicago",      41.88,  -87.63),
    ("Houston",      29.76,  -95.37),
    ("Phoenix",      33.45, -112.07),
    ("Philadelphia", 39.95,  -75.17),
    ("Boston",       42.36,  -71.06),
    ("Detroit",      42.33,  -83.05),
    ("Dallas",       32.78,  -96.80),
    ("Atlanta",      33.75,  -84.39),
    ("Miami",        25.77,  -80.19),
    ("Seattle",      47.61, -122.33),
    ("Denver",       39.74, -104.99),
    ("San Francisco",37.77, -122.42),
    ("Washington DC",38.91,  -77.04),
    ("Las Vegas",    36.17, -115.14),
    ("Minneapolis",  44.98,  -93.26),
    ("Portland",     45.52, -122.68),
    ("Nashville",    36.16,  -86.78),
    ("Austin",       30.27,  -97.74),
]

# ABO compatibility: recipient blood -> acceptable donor blood types
BLOOD_COMPAT = {
    "O+":  {"O+", "O-"},
    "O-":  {"O-"},
    "A+":  {"A+", "A-", "O+", "O-"},
    "A-":  {"A-", "O-"},
    "B+":  {"B+", "B-", "O+", "O-"},
    "B-":  {"B-", "O-"},
    "AB+": {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"},
    "AB-": {"A-", "B-", "AB-", "O-"},
}


def generate_data(n_donors=20, n_recipients=50, seed=7):
    rng = np.random.default_rng(seed)
    now = datetime.now()

    def make_hla():
        return {
            "A":  sorted(rng.choice(HLA_A_POOL,  2, replace=False).tolist()),
            "B":  sorted(rng.choice(HLA_B_POOL,  2, replace=False).tolist()),
            "DR": sorted(rng.choice(HLA_DR_POOL, 2, replace=False).tolist()),
        }

    donors = []
    for i in range(n_donors):
        city = CITIES[rng.integers(len(CITIES))]
        donors.append({
            "id":      f"D-{i+1:03d}",
            "died_at": now - timedelta(hours=float(rng.uniform(0.5, 48.0))),
            "blood":   rng.choice(BLOOD_TYPES, p=BLOOD_FREQ),
            "hla":     make_hla(),
            "size":    rng.choice(["adult", "child"], p=[0.80, 0.20]),
            "city":    city[0], "lat": city[1], "lon": city[2],
        })

    recipients = []
    for i in range(n_recipients):
        city = CITIES[rng.integers(len(CITIES))]
        recipients.append({
            "id":      f"R-{i+1:03d}",
            "blood":   rng.choice(BLOOD_TYPES, p=BLOOD_FREQ),
            "hla":     make_hla(),
            "size":    rng.choice(["adult", "child"], p=[0.80, 0.20]),
            "cpra":    int(rng.integers(0, 100)),
            "urgency": int(rng.integers(1, 5)),
            "days":    int(rng.integers(30, 1500)),
            "city":    city[0], "lat": city[1], "lon": city[2],
        })

    return donors, recipients


# ================================================================
# SECTION 1 — CLASSICAL COMPATIBILITY & SCORING
# ================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R  = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a  = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def hla_mismatches(donor, recipient):
    """Donor antigens absent from recipient across A, B, DR loci. Max=6."""
    mm = 0
    for locus in ["A", "B", "DR"]:
        mm += len(set(donor["hla"][locus]) - set(recipient["hla"][locus]))
    return mm


def hard_filter(donor, pool):
    """
    Classical pre-filter. Hard constraints for heart transplant:
      1. ABO blood type compatibility
      2. Size match (adult/child)
    HLA is NOT a hard filter for hearts — penalises the score instead.
    Returns: (compatible_list, rejected_dict)
    """
    compatible, rejected = [], {}
    for r in pool:
        reasons = []
        if donor["blood"] not in BLOOD_COMPAT.get(r["blood"], set()):
            reasons.append(f"blood ({donor['blood']} -> {r['blood']})")
        if donor["size"] != r["size"]:
            reasons.append(f"size ({donor['size']} != {r['size']})")
        if reasons:
            rejected[r["id"]] = reasons
        else:
            compatible.append(r)
    return compatible, rejected


def priority_score(donor, recipient):
    """
    Weighted priority score 0-100.
    Higher = match this recipient first.

    CPRA       40 pts  (highly sensitised = hardest to re-match)
    Urgency    30 pts  (medical status 1-4)
    Wait time  20 pts  (days on waitlist, normalised to 4 years)
    Distance   10 pts  (closer = less ischaemic time)
    HLA        -2 pts per mismatch
    """
    mm         = hla_mismatches(donor, recipient)
    w_cpra     = (recipient["cpra"] / 100) * 40
    w_urgency  = (recipient["urgency"] / 4) * 30
    w_days     = min(recipient["days"] / 1460, 1.0) * 20
    dist       = haversine_km(donor["lat"], donor["lon"],
                              recipient["lat"], recipient["lon"])
    w_distance = max(0.0, 1.0 - dist / 3000.0) * 10
    w_hla      = -2.0 * mm
    return round(max(0.0, w_cpra + w_urgency + w_days + w_distance + w_hla), 3)


# ================================================================
# SECTION 2 — QUBO  (tutorial formula verbatim)
# ================================================================
#
# Variables: x_i in {0,1},  x_i=1 -> "assign donor to recipient i"
# Constraint: exactly one recipient ->  A x = 1,  A = [[1,1,...,1]]
#
# Tutorial equation 11:
#   Q = A^T A - 2 diag(A^T 1)
#
# Full QUBO:
#   Q = lambda_c * (A^T A - 2 diag(A^T 1))  -  lambda_s * diag(scores)

def build_QUBO(scores, lambda_c=10.0, lambda_s=1.0):
    N      = len(scores)
    A      = np.ones((1, N))
    id_vec = np.ones((1, 1))
    Q      = lambda_c * (A.T @ A - 2 * np.diag((A.T @ id_vec).flatten()))
    Q     -= lambda_s * np.diag(scores)
    return Q


# ================================================================
# SECTION 3 — QUBO -> HAMILTONIAN  (tutorial function verbatim)
# ================================================================

def generate_pauli_terms(Q, b):
    """
    Construct cost Hamiltonian Pauli terms.
    COPIED DIRECTLY from the NordIQuEst tutorial.
    b_i = -sum_j (Q_ij + Q_ji)
    """
    N = len(b)
    pauli_list = []

    for i in range(N - 1):
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                paulis = ["I"] * N
                paulis[i] = "Z"
                paulis[j] = "Z"
                coeff = 2 * Q[i, j] / 4
                pauli_list.append(("".join(paulis)[::-1], coeff))

    for i in range(N):
        if b[i] != 0:
            paulis = ["I"] * N
            paulis[i] = "Z"
            coeff = b[i] / 4
            pauli_list.append(("".join(paulis)[::-1], coeff))

    return pauli_list


def qubo_to_hamiltonian(Q):
    b = -np.array([Q[i, :] + Q[:, i] for i in range(Q.shape[0])]).sum(axis=0)
    pauli_terms = generate_pauli_terms(Q, b)
    if not pauli_terms:
        pauli_terms = [("I" * Q.shape[0], 0.0)]
    return SparsePauliOp.from_list(pauli_terms)


# ================================================================
# SECTION 4 — QAOA  (QAOAAnsatz + COBYLA, tutorial style)
# ================================================================

def run_qaoa(Q, reps=2, shots=4096, seed=42):
    """
    QAOA for a single-donor matching problem.
    Tutorial steps 3.1 -> 3.3 -> 4.1 -> 4.2.
    """
    # Step 2: Hamiltonian
    cost_hamiltonian = qubo_to_hamiltonian(Q)

    # Step 3.1: QAOAAnsatz (tutorial line)
    circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    circuit.measure_all()
    
    # Check circuit size — if too wide for simulator, use statevector method
    if len(circuit.qubits) > 29:
        print(f"      Warning: circuit has {len(circuit.qubits)} qubits (max 29) — using statevector sim")
        sim = AerSimulator(method='statevector', seed_simulator=seed)
    else:
        sim = AerSimulator(seed_simulator=seed)

    # Step 3.2: Classical optimiser (COBYLA)
    def expectation(params):
        bound      = circuit.assign_parameters(dict(zip(circuit.parameters, params)))
        transpiled = transpile(bound, sim, optimization_level=1)
        counts     = sim.run(transpiled, shots=shots).result().get_counts()
        total      = sum(counts.values())
        cost       = sum(
            cnt * float(np.array([int(b) for b in reversed(bs)]) @ Q @
                        np.array([int(b) for b in reversed(bs)]))
            for bs, cnt in counts.items()
        )
        return cost / total

    np.random.seed(seed)
    init_params = np.random.uniform(0, np.pi, len(circuit.parameters))
    opt_result  = minimize(expectation, init_params, method="COBYLA",
                           options={"maxiter": 300, "rhobeg": 0.5})

    # Step 3.3: Final sample
    bound_final      = circuit.assign_parameters(dict(zip(circuit.parameters, opt_result.x)))
    transpiled_final = transpile(bound_final, sim, optimization_level=1)
    final_counts     = sim.run(transpiled_final, shots=shots*2).result().get_counts()

    # Steps 4.1 + 4.2: Filter valid bitstrings, pick best
    print(f"      Top outcomes  (valid = exactly one |1>):")
    best_idx, best_cost = None, np.inf
    for bs, cnt in sorted(final_counts.items(), key=lambda x: -x[1])[:8]:
        x    = np.array([int(b) for b in reversed(bs)])
        prob = cnt / (shots * 2)
        note = ""
        if x.sum() == 1:
            cost = float(x @ Q @ x)
            note = f"  valid  cost={cost:.2f}"
            if cost < best_cost:
                best_cost = cost
                best_idx  = int(np.argmax(x))
        print(f"        |{bs}>  p={prob:.3f}{note}")

    if best_idx is None:
        print("      No valid QAOA state — classical fallback.")
        best_idx = int(np.argmax(-np.diag(Q)))

    return best_idx, opt_result


# ================================================================
# SECTION 5 — MAIN PIPELINE
# ================================================================

def run_pipeline(n_donors=6, n_recipients=12,
                 qaoa_reps=2, qaoa_shots=4096, seed=7):

    W = 70
    def sep(): print("─" * W)
    def header(t): print("=" * W); print(f"  {t}"); print("=" * W)

    header("HEART ORGAN MATCHING  —  Hybrid Classical / QAOA")

    donors, recipients = generate_data(n_donors, n_recipients, seed)
    now = datetime.now()

    print(f"\n  Donors:           {n_donors}")
    print(f"  Recipients:       {n_recipients}")
    print(f"  QAOA reps (p):    {qaoa_reps}  ->  {2*qaoa_reps} parameters")
    print(f"  Shots per run:    {qaoa_shots}")

    print("\n  DONOR BANK:")
    for d in donors:
        hrs = (now - d["died_at"]).total_seconds() / 3600
        print(f"    {d['id']}  {d['blood']:4s}  {d['size']:5s}  "
              f"{hrs:.1f}h ago  [{d['city']}]")

    print("\n  RECIPIENT POOL:")
    for r in recipients:
        print(f"    {r['id']}  {r['blood']:4s}  {r['size']:5s}  "
              f"cpra={r['cpra']:3d}%  urg={r['urgency']}  "
              f"days={r['days']:4d}  [{r['city']}]")

    sorted_donors  = sorted(donors, key=lambda d: d["died_at"], reverse=True)
    recipient_pool = list(recipients)
    assignments    = []
    unmatched      = []

    print("\n  DONOR ORDER (most recent death first):")
    for i, d in enumerate(sorted_donors, 1):
        hrs = (now - d["died_at"]).total_seconds() / 3600
        print(f"    {i}. {d['id']}  {hrs:.1f}h since death")

    for donor in sorted_donors:
        hrs = (now - donor["died_at"]).total_seconds() / 3600
        print(f"\n{'='*W}")
        print(f"  DONOR {donor['id']}  |  {donor['blood']}  {donor['size']}  "
              f"|  {hrs:.1f}h since death  |  {donor['city']}")
        print("=" * W)

        if not recipient_pool:
            print("  Pool exhausted.")
            unmatched.append(donor["id"])
            continue

        # STEP 1: Classical hard filter
        print(f"\n  [Classical]  Hard filter  (pool = {len(recipient_pool)})")
        compatible, rejected = hard_filter(donor, recipient_pool)
        for rid, reasons in rejected.items():
            print(f"    x  {rid}:  {';  '.join(reasons)}")
        print(f"  Compatible: {len(compatible)}")
        
        # Cap compatible pool at 20 to avoid excessive qubits in QAOA
        if len(compatible) > 20:
            # Sort by priority score and keep top 20
            temp_scores = np.array([priority_score(donor, r) for r in compatible])
            top_indices = np.argsort(-temp_scores)[:20]
            capped = [compatible[i] for i in sorted(top_indices)]
            print(f"  Capped to top 20 by priority score (was {len(compatible)})")
            compatible = capped

        if not compatible:
            print("  No compatible recipients.")
            unmatched.append(donor["id"])
            continue

        # STEP 1.1: Priority scores
        scores = np.array([priority_score(donor, r) for r in compatible])
        print(f"\n  [Classical]  Priority scores")
        print(f"  {'ID':<9} {'CPRA':>5} {'Urg':>4} {'Days':>5} "
              f"{'HLA_mm':>7} {'Dist(km)':>9} {'Score':>7}")
        print("  " + "-" * 50)
        for r, s in zip(compatible, scores):
            dist = haversine_km(donor["lat"], donor["lon"], r["lat"], r["lon"])
            mm   = hla_mismatches(donor, r)
            print(f"  {r['id']:<9} {r['cpra']:>4}% {r['urgency']:>4} "
                  f"{r['days']:>5} {mm:>7} {dist:>9.0f} {s:>7.2f}")

        # STEP 1.2: QUBO
        N = len(compatible)
        Q = build_QUBO(scores, lambda_c=10.0, lambda_s=1.0)
        print(f"\n  [QUBO]  {N}x{N} matrix  ({N} qubits needed)")
        print("  Q =")
        for row in Q:
            print("    " + "  ".join(f"{v:8.2f}" for v in row))

        # STEP 1.3: Hamiltonian
        H = qubo_to_hamiltonian(Q)
        print(f"\n  [Hamiltonian]  {len(H)} Pauli terms")
        print(f"  {H}")

        # STEPS 3-4: QAOA
        if N == 1:
            print("\n  Single candidate — assigned directly.")
            best_idx = 0
            opt      = None
        else:
            print(f"\n  [QAOA]  QAOAAnsatz  reps={qaoa_reps}  shots={qaoa_shots}")
            best_idx, opt = run_qaoa(Q, reps=qaoa_reps,
                                     shots=qaoa_shots, seed=seed)
            if opt:
                print(f"      COBYLA converged={opt.success}  "
                      f"evals={opt.nfev}  cost={opt.fun:.3f}")

        # Assign and remove from pool
        matched = compatible[best_idx]
        dist_km = haversine_km(donor["lat"], donor["lon"],
                               matched["lat"], matched["lon"])
        recipient_pool = [r for r in recipient_pool if r["id"] != matched["id"]]

        assignments.append({
            "donor": donor["id"], "d_blood": donor["blood"],
            "d_city": donor["city"], "recipient": matched["id"],
            "r_blood": matched["blood"], "r_city": matched["city"],
            "score": scores[best_idx], "cpra": matched["cpra"],
            "urgency": matched["urgency"], "days": matched["days"],
            "hla_mm": hla_mismatches(donor, matched), "dist_km": round(dist_km, 1),
        })

        print(f"\n  MATCHED:")
        print(f"    Donor:     {donor['id']} ({donor['blood']}, {donor['city']})")
        print(f"    Recipient: {matched['id']} ({matched['blood']}, {matched['city']})")
        print(f"    Score={scores[best_idx]:.2f}  CPRA={matched['cpra']}%  "
              f"urgency={matched['urgency']}  days={matched['days']}  "
              f"HLA_mm={hla_mismatches(donor,matched)}  dist={dist_km:.0f}km")
        print(f"    Pool remaining: {len(recipient_pool)}")

    # Summary
    print(f"\n{'='*W}")
    print("  FINAL SUMMARY")
    print("=" * W)
    print(f"\n  Matches:           {len(assignments)} / {n_donors}")
    print(f"  Unmatched donors:  {len(unmatched)}"
          + (f"  {unmatched}" if unmatched else ""))
    print(f"  Waiting patients:  {len(recipient_pool)}")

    print(f"\n  {'Donor':<8} {'D.Bld':<6} {'Recipient':<11} {'R.Bld':<6} "
          f"{'Score':>6}  {'CPRA':>5} {'Urg':>4} {'Days':>5} "
          f"{'HLA_mm':>7} {'Dist(km)':>9}")
    print("  " + "-" * 68)
    for a in assignments:
        print(f"  {a['donor']:<8} {a['d_blood']:<6} {a['recipient']:<11} "
              f"{a['r_blood']:<6} {a['score']:>6.2f}  "
              f"{a['cpra']:>4}% {a['urgency']:>4} {a['days']:>5} "
              f"{a['hla_mm']:>7} {a['dist_km']:>9.1f}")

    return assignments


if __name__ == "__main__":
    run_pipeline(
        n_donors     = 20,
        n_recipients = 50,
        qaoa_reps    = 2,
        qaoa_shots   = 4096,
        seed         = 7,
    )
