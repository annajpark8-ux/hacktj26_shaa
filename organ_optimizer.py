import numpy as np
import pandas as pd
from collections import Counter
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

# ---- Qiskit imports ----
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator


# =============================================================================
# 1. DATA MODEL
# =============================================================================

@dataclass
class Recipient:
    """One pre-screened compatible transplant candidate."""
    name: str
    compatibility_score: float   # 0.0–1.0
    urgency_level: int           # 1–4
    waiting_time_days: int
    distance_km: float
    is_child: bool
    cpra_score: float            # 0–100


@dataclass
class MatchingWeights:
    """Policy weights for each factor (normalized internally)."""
    compatibility: float = 0.25
    urgency: float       = 0.25
    waiting_time: float  = 0.15
    distance: float      = 0.15
    pediatric: float     = 0.10
    cpra: float          = 0.10


# =============================================================================
# 2. SCORE NORMALIZATION & COMPOSITE  (unchanged)
# =============================================================================

def normalize_scores(recipients: List[Recipient]) -> np.ndarray:
    """Normalize all 6 factors to [0,1]. Returns (n, 6) array."""
    n = len(recipients)
    scores = np.zeros((n, 6))

    compatibilities = np.array([r.compatibility_score for r in recipients])
    urgencies       = np.array([r.urgency_level for r in recipients], dtype=float)
    wait_times      = np.array([r.waiting_time_days for r in recipients], dtype=float)
    distances       = np.array([r.distance_km for r in recipients], dtype=float)
    pediatrics      = np.array([1.0 if r.is_child else 0.0 for r in recipients])
    cpras           = np.array([r.cpra_score for r in recipients])

    scores[:, 0] = compatibilities
    scores[:, 1] = (6.0 - urgencies) / 5.0

    wt_min, wt_max = wait_times.min(), wait_times.max()
    scores[:, 2] = (wait_times - wt_min) / (wt_max - wt_min) if wt_max > wt_min else 0.5

    d_min, d_max = distances.min(), distances.max()
    scores[:, 3] = 1.0 - (distances - d_min) / (d_max - d_min) if d_max > d_min else 0.5

    scores[:, 4] = pediatrics
    scores[:, 5] = cpras / 100.0
    return scores


def compute_composite_scores(
    recipients: List[Recipient],
    weights: MatchingWeights,
) -> np.ndarray:
    """Weighted sum of normalized factors → one scalar per candidate."""
    scores = normalize_scores(recipients)
    w = np.array([
        weights.compatibility, weights.urgency, weights.waiting_time,
        weights.distance, weights.pediatric, weights.cpra,
    ])
    w /= w.sum()
    return scores @ w


# =============================================================================
# 3. COST HAMILTONIAN ENCODING
# =============================================================================

def _compute_hamiltonian_coefficients(
    composite_scores: np.ndarray,
    penalty_strength: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    
    n = len(composite_scores)
    lam = penalty_strength

    # ---- Linear coefficients (in Z_i basis) ----
    # From  (c_i + λ) · Z̃_i  =  (c_i + λ) · (I - Z_i) / 2
    # The Z_i coefficient is  -(c_i + λ) / 2
    # The I coefficient is     (c_i + λ) / 2  (absorbed into offset)
    h_linear = np.zeros(n)
    offset = 0.0

    for i in range(n):
        coeff = composite_scores[i] + lam
        h_linear[i] = -coeff / 2.0
        offset += coeff / 2.0

    # ---- Quadratic coefficients (in Z_i Z_j basis) ----
    # From  -λ · Z̃_i Z̃_j  =  -λ · (I - Z_i - Z_j + Z_i Z_j) / 4
    # Z_i Z_j coefficient:  -λ / 4
    # Z_i coefficient:       +λ / 4   (added to linear)
    # Z_j coefficient:       +λ / 4   (added to linear)
    # I coefficient:         -λ / 4   (added to offset)
    h_quadratic = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            h_quadratic[i, j] = -lam / 4.0
            h_linear[i] += lam / 4.0
            h_linear[j] += lam / 4.0
            offset -= lam / 4.0

    # ---- Global offset from the -λ constant in C ----
    offset -= lam

    return h_linear, h_quadratic, offset


# =============================================================================
# 4. QISKIT CIRCUIT CONSTRUCTION
# =============================================================================

def build_qaoa_circuit(
    n: int,
    p: int,
    h_linear: np.ndarray,
    h_quadratic: np.ndarray,
) -> Tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    
    # ---- Create parameter vectors ----
    # These are symbolic — Qiskit compiles the circuit once, then
    # we bind concrete values during each optimizer iteration.
    gammas = ParameterVector('γ', p)
    betas  = ParameterVector('β', p)

    qc = QuantumCircuit(n, n)  # n qubits, n classical bits

    # ---- Initial state: |+⟩^⊗n ----
    qc.h(range(n))
    qc.barrier()  # visual separator in circuit diagrams

    for l in range(p):
        # ======== COST LAYER (e^{-iγC}) ========
        #
        # For a diagonal Hamiltonian in the Z basis, the cost unitary
        # decomposes into single-qubit RZ and two-qubit RZZ gates.
        #
        # RZ(θ) = e^{-iθZ/2}   so for coefficient h, we need θ = 2γ·h
        # RZZ(θ) = e^{-iθ Z⊗Z /2}  same convention

        # Single-qubit Z rotations
        for i in range(n):
            if abs(h_linear[i]) > 1e-10:  # skip negligible terms
                qc.rz(2 * gammas[l] * h_linear[i], i)

        # Two-qubit ZZ interactions
        # RZZ is not a native gate on most hardware, so we decompose:
        #   RZZ(θ) = CX(i,j) · RZ(θ, j) · CX(i,j)
        # This is exact and uses only 2 CX gates.
        for i in range(n):
            for j in range(i + 1, n):
                if abs(h_quadratic[i, j]) > 1e-10:
                    theta = 2 * gammas[l] * h_quadratic[i, j]
                    qc.cx(i, j)
                    qc.rz(theta, j)
                    qc.cx(i, j)

        qc.barrier()

        # ======== MIXER LAYER (e^{-iβB}) ========
        #
        # B = Σ_i X_i  →  e^{-iβB} = Π_i e^{-iβX_i} = Π_i RX(2β)
        #
        # This is the standard transverse-field mixer.  On hardware,
        # RX is typically a native gate, so no further decomposition needed.
        for i in range(n):
            qc.rx(2 * betas[l], i)

        qc.barrier()

    # ---- Measurement ----
    qc.measure(range(n), range(n))

    return qc, gammas, betas


# =============================================================================
# 5. CVaR OBJECTIVE  (same math, applied to Qiskit measurement counts)
# =============================================================================

def _bitstring_to_indices(z: int, n: int) -> List[int]:
    """Return which qubits are |1⟩ in the n-bit integer z."""
    return [i for i in range(n) if (z >> i) & 1]


def _build_cost_diagonal(
    composite_scores: np.ndarray,
    penalty_strength: float,
) -> np.ndarray:
    """
    Build the full diagonal cost vector for post-processing shot results.
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


def _cvar_from_counts(
    counts: dict,
    cost_diagonal: np.ndarray,
    n: int,
    alpha: float = 0.25,
) -> float:
    """
    Compute CVaR_α from Qiskit measurement counts.

    Args:
        counts: dict from Qiskit result, e.g. {"01001": 137, "00010": 245, ...}
                Qiskit uses BIG-ENDIAN bit ordering: leftmost bit = highest qubit.
        cost_diagonal: precomputed cost for each basis state.
        n: number of qubits.
        alpha: CVaR fraction.

    Returns:
        cvar: mean of top α-fraction of shot costs.
    """
    # Expand counts into a list of (cost, count) pairs
    cost_count_pairs = []
    for bitstring, count in counts.items():
        # Qiskit bitstrings are big-endian: bit 0 is rightmost.
        # Convert to integer matching our little-endian convention.
        z = int(bitstring, 2)
        # Reverse bit order: Qiskit's q[0] is the rightmost character
        # but our cost_diagonal indexes q[0] as bit 0 (LSB), which
        # matches int(bitstring, 2) with reversed string.
        z_le = int(bitstring[::-1], 2)
        cost_count_pairs.append((cost_diagonal[z_le], count))

    # Sort by cost DESCENDING (best first)
    cost_count_pairs.sort(key=lambda x: x[0], reverse=True)

    # Compute total shots
    total_shots = sum(c for _, c in cost_count_pairs)
    k = max(1, int(np.ceil(alpha * total_shots)))

    # Accumulate the top-k shots
    accumulated = 0
    weighted_sum = 0.0
    for cost_val, count in cost_count_pairs:
        take = min(count, k - accumulated)
        weighted_sum += cost_val * take
        accumulated += take
        if accumulated >= k:
            break

    return weighted_sum / accumulated if accumulated > 0 else 0.0


# =============================================================================
# 6. QAOA OPTIMIZER  —  Qiskit AerSimulator
# =============================================================================

def qaoa_optimize_qiskit(
    composite_scores: np.ndarray,
    p: int = 3,
    penalty_strength: float = 10.0,
    n_shots: int = 1024,
    n_shots_final: int = 8192,
    cvar_alpha: float = 0.25,
    n_restarts: int = 10,
) -> Tuple[int, float, dict]:
    """
    Run QAOA on Qiskit's AerSimulator with shot-based measurement.

    Workflow:
        1. Decompose cost Hamiltonian into Pauli Z/ZZ coefficients.
        2. Build a parameterized QuantumCircuit (compiled once).
        3. Classical optimization loop:
             a. Bind (γ, β) values into the circuit.
             b. Execute on AerSimulator with n_shots.
             c. Parse measurement counts → CVaR objective.
             d. COBYLA adjusts parameters.
        4. Final measurement round with n_shots_final.
        5. Majority vote among valid bitstrings → winner.

    Args:
        composite_scores: per-candidate weighted scores.
        p: QAOA circuit depth.
        penalty_strength: constraint penalty λ.
        n_shots: shots per optimizer evaluation.
        n_shots_final: shots for the final decision round.
        cvar_alpha: CVaR top-fraction parameter.
        n_restarts: classical optimizer restarts.

    Returns:
        best_candidate, best_score, info_dict
    """
    n = len(composite_scores)

    # ---- 6a. Hamiltonian decomposition ----
    h_linear, h_quadratic, h_offset = _compute_hamiltonian_coefficients(
        composite_scores, penalty_strength
    )

    # ---- 6b. Build parameterized circuit (compiled ONCE) ----
    qc, gamma_params, beta_params = build_qaoa_circuit(n, p, h_linear, h_quadratic)

    print(f"  [QISKIT] Circuit: {qc.num_qubits} qubits, depth {qc.depth()}, "
          f"{qc.count_ops()} gates")
    print(f"  [QISKIT] Parameters: {qc.num_parameters} "
          f"({p} γ + {p} β)")

    # ---- 6c. Initialize AerSimulator ----
    #
    simulator = AerSimulator(method='automatic')

    # ---- 6d. Precompute cost diagonal for post-processing ----
    cost_diagonal = _build_cost_diagonal(composite_scores, penalty_strength)

    # ---- 6e. Evaluation counter ----
    eval_count = [0]

    # ---- 6f. Shot-based objective function ----
    def qiskit_objective(params):
        """
        Bind parameters → run on AerSimulator → parse counts → CVaR.

        This is the inner loop of the variational algorithm.
        On real hardware, this function would submit a job to the QPU.
        """
        gamma_vals = params[:p]
        beta_vals  = params[p:]

        # Build the parameter binding dictionary
        # Maps each Qiskit Parameter object to a concrete float value
        param_dict = {}
        for l in range(p):
            param_dict[gamma_params[l]] = gamma_vals[l]
            param_dict[beta_params[l]]  = beta_vals[l]

        # Bind parameters into the circuit (returns a new circuit, no recompile)
        bound_qc = qc.assign_parameters(param_dict)

        # Execute on AerSimulator
        # .run() returns a Job; .result() blocks until complete;
        # .get_counts() returns {"bitstring": count} dict.
        job = simulator.run(bound_qc, shots=n_shots)
        result = job.result()
        counts = result.get_counts()

        # Compute CVaR from the measurement counts
        cvar = _cvar_from_counts(counts, cost_diagonal, n, cvar_alpha)

        eval_count[0] += 1
        return -cvar  # negate for minimization

    # ---- 6g. Classical optimization with random restarts ----
    #
    # COBYLA is gradient-free and handles the stochastic (shot-noisy)
    # landscape well.  For production use, consider:
    #   • SPSA: designed for noisy function evaluations
    #   • NFT (Nakanishi-Fujii-Todo): parameter-shift-like, good for QAOA
    #   • Bayesian optimization for expensive evaluations

    best_params = None
    best_value = float('inf')

    print(f"  [QISKIT] Starting optimization: p={p}, shots={n_shots}, "
          f"CVaR α={cvar_alpha}, restarts={n_restarts}")

    for restart in range(n_restarts):
        init_params = np.random.uniform(0, 2 * np.pi, size=2 * p)

        result = minimize(
            qiskit_objective,
            init_params,
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.5},
        )

        if result.fun < best_value:
            best_value = result.fun
            best_params = result.x

    print(f"  [QISKIT] Optimization complete. {eval_count[0]} circuit executions.")

    # ---- 6h. FINAL MEASUREMENT ROUND ----
    gamma_vals = best_params[:p]
    beta_vals  = best_params[p:]

    param_dict = {}
    for l in range(p):
        param_dict[gamma_params[l]] = gamma_vals[l]
        param_dict[beta_params[l]]  = beta_vals[l]

    bound_qc = qc.assign_parameters(param_dict)

    print(f"  [QISKIT] Final measurement: {n_shots_final:,} shots on AerSimulator...")
    job = simulator.run(bound_qc, shots=n_shots_final)
    final_result = job.result()
    final_counts = final_result.get_counts()

    # ---- 6i. Parse measurement histogram ----
    valid_counts: Dict[int, int] = {}
    invalid_count = 0
    invalid_breakdown: Dict[int, int] = {}

    for bitstring, count in final_counts.items():
        # Convert Qiskit big-endian bitstring to our candidate indices
        z = int(bitstring[::-1], 2)  # reverse for little-endian
        selected = _bitstring_to_indices(z, n)
        k = len(selected)
        if k == 1:
            cand = selected[0]
            valid_counts[cand] = valid_counts.get(cand, 0) + count
        else:
            invalid_count += count
            invalid_breakdown[k] = invalid_breakdown.get(k, 0) + count

    total_valid = sum(valid_counts.values())

    # ---- 6j. Majority vote winner ----
    if valid_counts:
        ranked = sorted(valid_counts.items(), key=lambda x: x[1], reverse=True)
        best_candidate = ranked[0][0]
        best_votes = ranked[0][1]
    else:
        best_candidate = int(np.argmax(composite_scores))
        best_votes = 0
        print("  [QISKIT] WARNING: No valid measurements! Falling back to classical.")

    # ---- 6k. Statistical analysis ----
    winner_fraction = best_votes / n_shots_final if n_shots_final > 0 else 0
    valid_fraction = total_valid / n_shots_final if n_shots_final > 0 else 0
    se = np.sqrt(winner_fraction * (1 - winner_fraction) / n_shots_final) \
        if n_shots_final > 0 else 0
    ci_low = max(0, winner_fraction - 1.96 * se)
    ci_high = min(1, winner_fraction + 1.96 * se)

    if len(ranked) >= 2:
        runner_up_idx = ranked[1][0]
        runner_up_votes = ranked[1][1]
        margin = (best_votes - runner_up_votes) / n_shots_final
    else:
        runner_up_idx = None
        runner_up_votes = 0
        margin = winner_fraction

    # ---- 6l. Cost distribution from final shots ----
    final_costs = []
    for bitstring, count in final_counts.items():
        z = int(bitstring[::-1], 2)
        final_costs.extend([cost_diagonal[z]] * count)
    final_costs = np.array(final_costs)

    final_cvar = _cvar_from_counts(final_counts, cost_diagonal, n, cvar_alpha)

    # ---- 6m. Package diagnostics ----
    info = {
        'optimal_params': best_params,
        'gammas': gamma_vals,
        'betas': beta_vals,
        'qaoa_depth': p,
        'n_candidates': n,
        'n_shots_optimization': n_shots,
        'n_shots_final': n_shots_final,
        'cvar_alpha': cvar_alpha,
        'penalty_strength': penalty_strength,
        'total_circuit_evaluations': eval_count[0],

        # Qiskit-specific metadata
        'circuit_depth': qc.depth(),
        'gate_counts': dict(qc.count_ops()),
        'simulator_method': simulator._options.get('method', 'automatic'),

        # Raw Qiskit counts (for external analysis)
        'raw_qiskit_counts': dict(final_counts),

        # Parsed measurement results
        'valid_vote_counts': valid_counts,
        'invalid_shot_count': invalid_count,
        'invalid_breakdown': invalid_breakdown,
        'total_valid_shots': total_valid,
        'total_invalid_shots': invalid_count,
        'valid_fraction': valid_fraction,

        # Winner statistics
        'winner_votes': best_votes,
        'winner_fraction': winner_fraction,
        'winner_std_error': se,
        'winner_95ci': (ci_low, ci_high),
        'runner_up_index': runner_up_idx,
        'runner_up_votes': runner_up_votes,
        'win_margin': margin,

        # Cost function statistics
        'cost_mean': final_costs.mean(),
        'cost_std': final_costs.std(),
        'cost_max': final_costs.max(),
        'cost_min': final_costs.min(),
        'final_cvar': final_cvar,

        # Per-candidate probabilities
        'all_valid_probabilities': {
            idx: cnt / total_valid if total_valid > 0 else 0
            for idx, cnt in valid_counts.items()
        },
    }

    return best_candidate, composite_scores[best_candidate], info


# =============================================================================
# 7. FULL MATCHING PIPELINE
# =============================================================================

def match_heart_to_recipient(
    recipients: List[Recipient],
    weights: MatchingWeights = None,
    qaoa_depth: int = 3,
    penalty_strength: float = 10.0,
    n_shots: int = 4096,
    n_shots_final: int = 16384,
    cvar_alpha: float = 0.25,
    n_restarts: int = 20,
) -> dict:
    """End-to-end Qiskit QAOA heart matching pipeline."""
    if weights is None:
        weights = MatchingWeights()

    composite = compute_composite_scores(recipients, weights)
    classical_best = int(np.argmax(composite))

    best_idx, best_score, info = qaoa_optimize_qiskit(
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
        'composite_scores': composite,
        'normalized_factors': normalize_scores(recipients),
        'qaoa_info': info,
        'classical_best_index': classical_best,
        'qaoa_matches_classical': best_idx == classical_best,
    }


# =============================================================================
# 8. REPORTING
# =============================================================================

def print_report(result: dict, recipients: List[Recipient], weights: MatchingWeights):
    """Pretty-print matching results with Qiskit circuit and shot details."""
    factor_names = [
        "Compatibility", "Urgency", "Waiting Time",
        "Distance (inv)", "Pediatric", "CPRA"
    ]
    w_arr = np.array([
        weights.compatibility, weights.urgency, weights.waiting_time,
        weights.distance, weights.pediatric, weights.cpra,
    ])
    w_arr /= w_arr.sum()
    info = result['qaoa_info']

    print("=" * 80)
    print("  QAOA HEART-ORGAN MATCHING  —  Qiskit AerSimulator Backend")
    print("=" * 80)

    print("\n  POLICY WEIGHTS:")
    for name, w in zip(factor_names, w_arr):
        print(f"    {name:<20s}: {w:.3f}")

    print(f"\n  QISKIT CIRCUIT INFO:")
    print(f"    Qubits:                     {info['n_candidates']}")
    print(f"    Circuit depth:              {info['circuit_depth']}")
    print(f"    Gate counts:                {info['gate_counts']}")
    print(f"    QAOA layers (p):            {info['qaoa_depth']}")
    print(f"    Simulator method:           {info['simulator_method']}")

    print(f"\n  OPTIMIZATION CONFIG:")
    print(f"    Shots per evaluation:       {info['n_shots_optimization']:,}")
    print(f"    Final measurement shots:    {info['n_shots_final']:,}")
    print(f"    CVaR α:                     {info['cvar_alpha']}")
    print(f"    Penalty strength (λ):       {info['penalty_strength']}")
    print(f"    Total circuit executions:   {info['total_circuit_evaluations']:,}")

    norm = result['normalized_factors']
    comp = result['composite_scores']
    votes = info['valid_vote_counts']
    total_valid = info['total_valid_shots']

    print(f"\n  CANDIDATE SCORES & MEASUREMENT RESULTS:")
    print(f"  {'#':<4} {'Name':<18} ", end="")
    for fn in factor_names:
        print(f"{fn[:6]:>7}", end="")
    print(f"  {'Score':>7}  {'Votes':>7}  {'Prob':>7}")
    print("  " + "-" * 105)

    for i, r in enumerate(recipients):
        marker = " ★" if i == result['selected_index'] else "  "
        v = votes.get(i, 0)
        prob = v / total_valid if total_valid > 0 else 0
        print(f"  {i:<4} {r.name:<18} ", end="")
        for j in range(6):
            print(f"{norm[i, j]:>7.3f}", end="")
        print(f"  {comp[i]:>7.4f}  {v:>7,}  {prob:>7.4f}{marker}")

    print(f"\n  MEASUREMENT STATISTICS:")
    print(f"    Valid shots:      {info['total_valid_shots']:>8,} / {info['n_shots_final']:,}  "
          f"({info['valid_fraction']:.1%})")
    print(f"    Invalid shots:    {info['total_invalid_shots']:>8,}  ", end="")
    if info['invalid_breakdown']:
        parts = [f"{k}-selected: {v}" for k, v in sorted(info['invalid_breakdown'].items())]
        print(f"({', '.join(parts)})")
    else:
        print("(none)")

    ci = info['winner_95ci']
    print(f"\n  WINNER ANALYSIS:")
    print(f"    ★ Selected:       {result['selected_recipient'].name}")
    print(f"    Composite score:  {comp[result['selected_index']]:.4f}")
    print(f"    Votes:            {info['winner_votes']:,} / {info['total_valid_shots']:,} valid")
    print(f"    Win probability:  {info['winner_fraction']:.4f}  "
          f"± {info['winner_std_error']:.4f}  "
          f"(95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
    print(f"    Win margin:       {info['win_margin']:.4f}  "
          f"(over {recipients[info['runner_up_index']].name if info['runner_up_index'] is not None else 'N/A'})")

    print(f"\n  COST FUNCTION DISTRIBUTION (final shots):")
    print(f"    Mean:      {info['cost_mean']:>8.4f}")
    print(f"    Std Dev:   {info['cost_std']:>8.4f}")
    print(f"    Min:       {info['cost_min']:>8.4f}")
    print(f"    Max:       {info['cost_max']:>8.4f}")
    print(f"    CVaR₀.₂₅:  {info['final_cvar']:>8.4f}")

    print(f"\n  SHOT HISTOGRAM (valid measurements):")
    max_votes = max(votes.values()) if votes else 1
    bar_width = 40
    for i in range(len(recipients)):
        v = votes.get(i, 0)
        bar_len = int(bar_width * v / max_votes) if max_votes > 0 else 0
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        marker = " ★" if i == result['selected_index'] else "  "
        print(f"    {recipients[i].name:<18} |{bar}| {v:>6,}{marker}")

    print(f"\n  CLASSICAL VERIFICATION:")
    print(f"    Classical optimum:  #{result['classical_best_index']} "
          f"({recipients[result['classical_best_index']].name})")
    print(f"    QAOA agrees:       {'✓ YES' if result['qaoa_matches_classical'] else '✗ NO'}")
    print("=" * 80)


# =============================================================================
# 9. DEMO
# =============================================================================

def main():
    recipients = [
        Recipient("Alice Thompson", 0.92, 3, 540, 150.0, False, 45.0),
        Recipient("Ben Ortega",     0.85, 4, 210, 320.0, False, 78.0),
        Recipient("Clara Johansson",0.78, 2, 890,  50.0, True,  92.0),
        Recipient("David Kim",      0.95, 2, 1100, 80.0, False, 15.0),
        Recipient("Elena Vasquez",  0.88, 4, 365, 500.0, False, 60.0),
    ]

    weights = MatchingWeights(
        compatibility=0.25, urgency=0.25, waiting_time=0.15,
        distance=0.15, pediatric=0.10, cpra=0.10,
    )

    np.random.seed(42)
    t0 = time.time()

    result = match_heart_to_recipient(
        recipients,
        weights=weights,
        qaoa_depth=5,
        penalty_strength=10.0,
        n_shots=4096,
        n_shots_final=16384,
        cvar_alpha=0.25,
        n_restarts=30,
    )

    elapsed = time.time() - t0
    print_report(result, recipients, weights)
    print(f"\n  Total runtime: {elapsed:.2f}s")

if __name__ == "__main__":
    main()