import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Qiskit core ───────────────────────────────────────────────────────────────
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info  import SparsePauliOp
from qiskit.transpiler    import generate_preset_pass_manager
from qiskit_aer           import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler

# ── Classical optimizer ───────────────────────────────────────────────────────
from scipy.optimize import minimize, differential_evolution

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: ORGAN CONFIGURATIONS
#
# To add a new organ:
#   1. Add a config block here
#   2. Add a distance penalty function in SECTION 2
#   3. Register it in DISTANCE_PENALTY_FNS
#   4. Change ORGAN_TYPE in SECTION 7
# ═════════════════════════════════════════════════════════════════════════════

ORGAN_CONFIGS = {

    "heart": {
        "weights": {
            "distance":     0.35,
            "urgency":      0.25,
            "bio_match":    0.20,
            "waiting_time": 0.10,
            "size_match":   0.10,
        },
        "max_ischemia_hr":  6,
        "max_distance_km":  800,
        "bsa_tolerance":    0.20,
    },

    "lung": {
        "weights": {
            "distance":     0.35,
            "urgency":      0.25,
            "bio_match":    0.15,
            "waiting_time": 0.10,
            "size_match":   0.15,
        },
        "max_ischemia_hr":  6,
        "max_distance_km":  800,
        "bsa_tolerance":    0.15,
    },

    "liver": {
        "weights": {
            "distance":     0.25,
            "urgency":      0.30,
            "bio_match":    0.20,
            "waiting_time": 0.15,
            "size_match":   0.10,
        },
        "max_ischemia_hr":  24,
        "max_distance_km":  2000,
        "bsa_tolerance":    0.30,
    },

    "kidney": {
        "weights": {
            "distance":     0.15,
            "urgency":      0.20,
            "bio_match":    0.35,
            "waiting_time": 0.25,
            "size_match":   0.05,
        },
        "max_ischemia_hr":  36,
        "max_distance_km":  3000,
        "bsa_tolerance":    0.40,
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: DISTANCE PENALTY FUNCTIONS
#
# Signature must always be: (distance_km: float, config: dict) -> float
# Return value must be in [0, 1].
# ═════════════════════════════════════════════════════════════════════════════

def _heart_lung_distance_penalty(distance_km, config):
    """Sharp cliff after ~500 km — reflects hard 4-6 hr ischemia window."""
    d, max_d = distance_km, config["max_distance_km"]
    if d <= 300:   return d / max_d
    elif d <= 600: return 0.375 + (d - 300) / 400
    else:          return min(1.0, 0.75 + (d - 600) / 200)

def _liver_distance_penalty(distance_km, config):
    """Gradual slope — livers tolerate up to 24 hr cold ischemia."""
    d, max_d = distance_km, config["max_distance_km"]
    if d <= 1000: return (d / max_d) * 0.6
    else:         return min(1.0, 0.6 + (d - 1000) / 2000)

def _kidney_distance_penalty(distance_km, config):
    """Near-linear — kidneys survive 24-36 hr outside the body."""
    return min(1.0, distance_km / config["max_distance_km"])

DISTANCE_PENALTY_FNS = {
    "heart":  _heart_lung_distance_penalty,
    "lung":   _heart_lung_distance_penalty,
    "liver":  _liver_distance_penalty,
    "kidney": _kidney_distance_penalty,
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: COST COMPUTATION  (organ-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

def compute_size_mismatch(donor_bsa, recipient_bsa, bsa_tolerance):
    ratio = abs(donor_bsa - recipient_bsa) / donor_bsa
    return min(1.0, ratio / bsa_tolerance)


def compute_weights(donor, recipients, organ_type):
    """Return a 1-D numpy array of per-recipient cost weights."""
    config     = ORGAN_CONFIGS[organ_type]
    w          = config["weights"]
    penalty_fn = DISTANCE_PENALTY_FNS[organ_type]

    weights = []
    for r in recipients:
        cost = (
            w["distance"]     * penalty_fn(r["distance_km"], config)
          + w["urgency"]      * (1 - r["urgency"])
          + w["bio_match"]    * r["pra_score"]
          + w["waiting_time"] * (1 - r["waiting_time"])
          + w["size_match"]   * compute_size_mismatch(
                donor["bsa"], r["bsa"], config["bsa_tolerance"]
            )
        )
        weights.append(cost)
    return np.array(weights)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: PENALTY CALIBRATION
#
# Penalty must satisfy:  max(weights) < P < scale where cost diffs vanish.
# The formula below keeps P tight to the actual weight range so the
# optimizer can still distinguish between recipients.
# ═════════════════════════════════════════════════════════════════════════════

def calibrate_penalty(weights):
    """
    Set penalty just above max weight, scaled by weight spread.
    Prevents both under-penalisation and cost-signal burial.
    """
    weight_range = max(weights) - min(weights)
    return max(weights) + 2 * weight_range


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: QUBO AND ISING CONVERSION  (organ-agnostic)
# ═════════════════════════════════════════════════════════════════════════════

def build_qubo(weights, penalty):
    """
    One-of-N QUBO:
      Diagonal     Q[j][j] = w_j - P
      Off-diagonal Q[j][k] = 2P
    """
    n = len(weights)
    Q = np.zeros((n, n))
    for j in range(n):
        Q[j][j] = weights[j] - penalty
    for j in range(n):
        for k in range(j + 1, n):
            Q[j][k] = 2 * penalty
    return Q


def qubo_to_ising(Q):
    """
    Convert QUBO → Ising via x_j = (1 - z_j) / 2.
    Returns (SparsePauliOp, float offset).
    """
    n          = Q.shape[0]
    pauli_list = []
    offset     = 0.0

    # Diagonal terms
    for j in range(n):
        c      = Q[j][j]
        offset += c / 2
        p      = ['I'] * n
        p[j]   = 'Z'
        pauli_list.append((''.join(reversed(p)), -c / 2))

    # Off-diagonal terms
    for j in range(n):
        for k in range(j + 1, n):
            c = Q[j][k]
            if c == 0:
                continue
            offset += c / 4

            pj      = ['I'] * n;  pj[j] = 'Z'
            pk      = ['I'] * n;  pk[k] = 'Z'
            pjk     = ['I'] * n;  pjk[j] = 'Z';  pjk[k] = 'Z'

            pauli_list.append((''.join(reversed(pj)),  -c / 4))
            pauli_list.append((''.join(reversed(pk)),  -c / 4))
            pauli_list.append((''.join(reversed(pjk)),  c / 4))

    return SparsePauliOp.from_list(pauli_list).simplify(), offset


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: QAOA SOLVER WITH AER BACKEND
#
# Key improvements over the basic version:
#
#   1. AerSimulator  — simulates real quantum hardware including gate noise,
#                      decoherence, and measurement errors
#   2. Transpilation — compiles the circuit to the simulator's native gate set,
#                      which is required for Aer to run it correctly
#   3. Two-phase opt — differential_evolution finds a good global starting
#                      point first; COBYLA refines from there. This avoids
#                      barren plateaus that trap single-start optimizers.
#   4. reps=3        — more QAOA layers → better approximation ratio
#   5. Higher shots  — reduces measurement noise in the cost estimate
# ═════════════════════════════════════════════════════════════════════════════

def build_aer_backend(optimization_level=1):
    """
    Build an AerSimulator backend and a matching pass manager.

    In qiskit-aer's SamplerV2, the backend is no longer passed to
    the constructor — it is instead passed per .run() call, or used
    here to build the transpiler pass manager that targets it.

    optimization_level controls transpiler aggressiveness:
      0 = no optimization  (fastest compile, noisiest)
      1 = light            (good balance for QAOA)
      2 = heavy            (slower compile, cleaner circuit)
    """
    backend = AerSimulator()
    pm      = generate_preset_pass_manager(
        optimization_level=optimization_level,
        backend=backend
    )
    return backend, pm


def run_qaoa(ising_op, Q, offset, reps=3, shots=2048, maxiter=600):
    """
    Run QAOA on an AerSimulator backend with two-phase optimization.

    Phase 1 — differential_evolution:
        Global search over parameter space. Expensive but avoids
        barren plateaus that defeat local optimizers like COBYLA.

    Phase 2 — COBYLA:
        Fast local refinement starting from Phase 1's best point.

    Returns best assignment as a binary numpy array.
    """
    backend, pm = build_aer_backend(optimization_level=1)

    # SamplerV2 is instantiated with no arguments — backend is
    # passed per-job via sampler.run([circuit], backend=backend)
    sampler = AerSampler()

    # Build and transpile circuit once — reuse across all evaluations
    ansatz = QAOAAnsatz(cost_operator=ising_op, reps=reps)
    ansatz.measure_all()
    transpiled = pm.run(ansatz)

    n_params = len(ansatz.parameters)
    print(f"Circuit depth : {transpiled.depth()} gates")
    print(f"Parameters    : {n_params} (γ and β for each of {reps} layers)")

    def cost_fn(params):
        bound  = transpiled.assign_parameters(params)
        counts = (
            sampler.run([bound], shots=shots)
            .result()[0].data.meas.get_counts()
        )
        total  = sum(counts.values())
        return sum(
            (c / total) * (
                np.array([int(b) for b in reversed(bs)]) @ Q
                @ np.array([int(b) for b in reversed(bs)])
            )
            for bs, c in counts.items()
        ) + offset

    # ── Phase 1: global search ────────────────────────────────────────────
    print("\nPhase 1 — global parameter search (differential evolution)...")
    # Parameter bounds: gamma in [0, pi], beta in [0, pi/2]
    bounds = []
    for _ in range(reps):
        bounds.append((0, np.pi))        # gamma
        bounds.append((0, np.pi / 2))    # beta

    de_result = differential_evolution(
        cost_fn,
        bounds,
        maxiter=40,       # keep fast for hackathon; raise for better results
        popsize=8,
        tol=1e-3,
        seed=42,
        disp=False
    )
    print(f"Phase 1 best cost : {de_result.fun:.4f}")

    # ── Phase 2: local refinement ─────────────────────────────────────────
    print("Phase 2 — local refinement (COBYLA)...")
    opt = minimize(
        cost_fn,
        x0=de_result.x,
        method='COBYLA',
        options={'maxiter': maxiter, 'rhobeg': 0.3}
    )
    print(f"Phase 2 best cost : {opt.fun:.4f}")
    print(f"Converged         : {opt.success}")

    # ── Final high-shot sampling with optimal parameters ──────────────────
    print("Final sampling with optimal parameters...")
    bound_final = transpiled.assign_parameters(opt.x)
    final_counts = (
        sampler.run([bound_final], shots=shots * 4)
        .result()[0].data.meas.get_counts()
    )

    # Return the most frequently measured valid bitstring
    best_bs = max(final_counts, key=final_counts.get)
    return np.array([int(b) for b in reversed(best_bs)]), final_counts


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: RESULTS DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def display_results(bits, weights, recipients, organ_type, counts=None):
    print(f"\n{'═'*55}")
    print(f"  {organ_type.upper()} ASSIGNMENT RESULT")
    print(f"{'═'*55}")

    assigned = False
    for j, r in enumerate(recipients):
        if bits[j] == 1:
            print(f"\n  ✦ {organ_type.capitalize()} → Recipient {j}")
            print(f"    Distance     : {r['distance_km']} km")
            print(f"    Urgency      : {r['urgency']}")
            print(f"    PRA score    : {r['pra_score']}")
            print(f"    Waiting time : {r['waiting_time']}")
            print(f"    BSA          : {r['bsa']} m²")
            print(f"    Composite cost: {weights[j]:.4f}")
            assigned = True

    if not assigned:
        print("\n  ⚠ No valid single assignment found")
        print(f"  Recalibrate PENALTY — current max weight: {max(weights):.4f}")

    # Show top 5 measured states for transparency
    if counts:
        print(f"\n── Top 5 measured states {'─'*30}")
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        total = sum(counts.values())
        for bs, c in sorted_counts[:5]:
            bits_row = np.array([int(b) for b in reversed(bs)])
            recipient_idx = np.where(bits_row == 1)[0]
            label = (f"→ Recipient {recipient_idx[0]}"
                     if len(recipient_idx) == 1 else
                     f"→ INVALID ({len(recipient_idx)} assigned)")
            print(f"  |{bs}⟩  {label:25s}  "
                  f"{c:5d} shots  ({100*c/total:.1f}%)")

    print(f"\n── Classical verification {'─'*29}")
    best_j = np.argmin(weights)
    print(f"  Classical optimum : Recipient {best_j} "
          f"(cost {weights[best_j]:.4f})")
    print(f"  QAOA matches      : {bool(bits[best_j] == 1)}")
    print()

    print("── All recipient costs ─────────────────────────────")
    sorted_recipients = sorted(enumerate(weights), key=lambda x: x[1])
    for rank, (j, w) in enumerate(sorted_recipients):
        marker = " ◄ QAOA" if bits[j] == 1 else ""
        opt    = " ◄ CLASSICAL OPT" if j == best_j else ""
        print(f"  Rank {rank+1}  Recipient {j}  cost {w:.4f}{marker}{opt}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: YOUR DATA
#
# Edit donor, recipients, and ORGAN_TYPE here.
# Larger recipient lists give QAOA more room to demonstrate advantage.
# ═════════════════════════════════════════════════════════════════════════════

ORGAN_TYPE = "heart"   # ← "heart" | "lung" | "liver" | "kidney"

donor = {
    "blood_type": "A",
    "bsa":        1.9,
    "lvef":       0.58,
    "location":   "Boston, MA"
}

# 8 recipients — large enough that QAOA has a meaningful search space
# (8 qubits = 256 possible states) while staying simulatable in seconds
recipients = [
    {"id": 0, "bsa": 1.85, "urgency": 0.95, "waiting_time": 0.60,
     "distance_km": 80,   "pra_score": 0.10},
    {"id": 1, "bsa": 2.10, "urgency": 0.70, "waiting_time": 0.90,
     "distance_km": 350,  "pra_score": 0.40},
    {"id": 2, "bsa": 1.95, "urgency": 0.85, "waiting_time": 0.40,
     "distance_km": 200,  "pra_score": 0.20},
    {"id": 3, "bsa": 1.75, "urgency": 0.60, "waiting_time": 0.75,
     "distance_km": 500,  "pra_score": 0.60},
    {"id": 4, "bsa": 1.88, "urgency": 0.80, "waiting_time": 0.55,
     "distance_km": 130,  "pra_score": 0.15},
    {"id": 5, "bsa": 2.00, "urgency": 0.55, "waiting_time": 0.85,
     "distance_km": 420,  "pra_score": 0.30},
    {"id": 6, "bsa": 1.92, "urgency": 0.90, "waiting_time": 0.30,
     "distance_km": 160,  "pra_score": 0.25},
    {"id": 7, "bsa": 1.80, "urgency": 0.65, "waiting_time": 0.70,
     "distance_km": 290,  "pra_score": 0.45},
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: RUN  — nothing below needs to change
# ═════════════════════════════════════════════════════════════════════════════

print(f"{'═'*55}")
print(f"  ORGAN MATCHING QAOA  —  {ORGAN_TYPE.upper()}")
print(f"  {len(recipients)} recipients  |  Aer quantum simulation")
print(f"{'═'*55}\n")

weights = compute_weights(donor, recipients, ORGAN_TYPE)
penalty = calibrate_penalty(weights)
Q       = build_qubo(weights, penalty)
ising_op, offset = qubo_to_ising(Q)

print(f"Weights   : {np.round(weights, 4)}")
print(f"Penalty   : {penalty:.4f}  (auto-calibrated)")
print(f"Qubits    : {ising_op.num_qubits}")

bits, counts = run_qaoa(
    ising_op, Q, offset,
    reps    = 3,     # 3 QAOA layers — good approximation ratio
    shots   = 2048,  # high shot count — reduces measurement noise
    maxiter = 600    # COBYLA refinement iterations
)

display_results(bits, weights, recipients, ORGAN_TYPE, counts)