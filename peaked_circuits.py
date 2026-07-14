"""Peaked-circuit construction from Aaronson & Zhang, arXiv:2404.14493.

Reproduces the numerical experiment of Section 3, "Numerical results on the
peakability of RQCs" (see Fig. 2b for the circuit layout):

  * a 1D brick-wall circuit with open boundaries, made of
  * ``tau_r`` layers of *fixed* Haar-random two-qubit gates, followed by
  * ``tau_p`` layers of *parameterized* two-qubit gates (full SU(4) each),
  * whose parameters are optimized to maximize the peak weight (Eq. 9)

        delta_{0^n}(C(theta)) = |<0^n| C(theta) |0^n>|^2 .

Following the paper and its official code
(https://github.com/yuxuanzhang1995/Peaked-circuits), the optimization runs
batches of independently initialized Adam runs and keeps the best result, to
mitigate barren plateaus (Section 4), with a step-decay learning-rate
schedule and early stopping. For the default configuration (n = 12,
tau_r = 40, tau_p = 10) the paper reports delta ~ 0.2, averaged over 100
circuit instances (Fig. 2c/2d); use ``--instances`` to average likewise.

Known deviation from the official code (same physics, different numerics):
the official code optimizes raw 4x4 unitaries kept unitary by a Cayley
retraction; here each gate is PennyLane's ``qml.ArbitraryUnitary`` (15
Pauli-basis angles), which realizes any two-qubit unitary up to global
phase, so the model class is identical. At theta = 0 the gate is the
identity, so the small random initialization starts the peaking section
near the identity, exactly like the official code.

Usage:
    python peaked_circuits.py                      # paper configuration
    python peaked_circuits.py --output best.npz    # also save the best angles
    python peaked_circuits.py --num-qubits 6 --random-layers 12 \
        --peaking-layers 3 --restarts 1 --max-steps 300   # quick smoke test
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.stats import unitary_group

logger = logging.getLogger(__name__)

#: Dimension of SU(4) as a real manifold: any two-qubit gate has 15 parameters.
PARAMETERS_PER_SU4_GATE = 15

#: The (n, tau_r, tau_p) configuration for which the paper reports delta ~ 0.2.
PAPER_CONFIGURATION = (12, 40, 10)

#: A layer of Haar-random 4x4 unitaries, one per qubit pair of that layer.
HaarLayer = list[np.ndarray]

#: Differentiable map theta -> delta_{0^n}(C(theta)).
PeakWeightFunction = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CircuitConfig:
    """Geometry of the peaked circuit (Fig. 2b of the paper)."""

    num_qubits: int = 12
    num_random_layers: int = 40  # tau_r
    num_peaking_layers: int = 10  # tau_p

    def __post_init__(self) -> None:
        if self.num_qubits < 2:
            raise ValueError("A brick-wall circuit needs at least 2 qubits.")
        if self.num_random_layers < 0 or self.num_peaking_layers < 1:
            raise ValueError("Layer counts must be tau_r >= 0 and tau_p >= 1.")

    @property
    def num_peaking_parameters(self) -> int:
        """Total number of trainable parameters in the peaking section."""
        num_gates = sum(
            len(brick_wall_pairs(self.num_random_layers + layer, self.num_qubits))
            for layer in range(self.num_peaking_layers)
        )
        return num_gates * PARAMETERS_PER_SU4_GATE


@dataclass(frozen=True)
class OptimizerConfig:
    """Settings for the batched Adam optimization (Sections 3 and 4).

    The official code uses Adam (lr = 1e-3) on raw unitaries with a
    StepLR(300, 0.5) schedule, up to 5000 iterations and early stopping once
    the per-step loss change falls below tolerance. The defaults below play
    the same role, tuned for the 15-angle gate parameterization used here.
    """

    num_restarts: int = 3
    max_steps: int = 2000
    learning_rate: float = 0.05
    decay_every: int = 300  # halve the step size every this many steps
    decay_factor: float = 0.5
    convergence_tol: float = 1e-8  # early stop on per-step |cost change|
    min_steps: int = 100  # never early-stop before this many steps
    initial_scale: float = 0.1  # std-dev of the random parameter initialization
    log_every: int = 50

    def __post_init__(self) -> None:
        if self.num_restarts < 1 or self.max_steps < 1:
            raise ValueError("num_restarts and max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.decay_every < 1 or not 0 < self.decay_factor <= 1:
            raise ValueError("Need decay_every >= 1 and 0 < decay_factor <= 1.")
        if self.convergence_tol < 0 or self.initial_scale < 0:
            raise ValueError("convergence_tol and initial_scale must be >= 0.")


@dataclass(frozen=True)
class ExperimentResult:
    """Outcome of one peaked-circuit optimization."""

    peak_weight: float
    baseline_peak_weight: float  # random section alone, expected ~ 2^-n
    parameters: np.ndarray


def brick_wall_pairs(layer_index: int, num_qubits: int) -> tuple[tuple[int, int], ...]:
    """Qubit pairs coupled by one brick-wall layer with open boundaries.

    Even-indexed layers couple (0, 1), (2, 3), ...; odd-indexed layers are
    shifted by one qubit and couple (1, 2), (3, 4), ...
    """
    first_qubit = layer_index % 2
    return tuple((q, q + 1) for q in range(first_qubit, num_qubits - 1, 2))


def sample_haar_random_layers(
    config: CircuitConfig, rng: np.random.Generator
) -> list[HaarLayer]:
    """Draw the tau_r fixed layers: one Haar-random SU(4) matrix per gate."""
    return [
        [
            unitary_group.rvs(4, random_state=rng)
            for _ in brick_wall_pairs(layer, config.num_qubits)
        ]
        for layer in range(config.num_random_layers)
    ]


def _apply_random_section(config: CircuitConfig, haar_layers: list[HaarLayer]) -> None:
    """Apply the tau_r fixed Haar-random brick-wall layers."""
    for layer_index, layer_gates in enumerate(haar_layers):
        pairs = brick_wall_pairs(layer_index, config.num_qubits)
        for wires, gate in zip(pairs, layer_gates, strict=True):
            qml.QubitUnitary(gate, wires=wires)


def _apply_peaking_section(config: CircuitConfig, theta) -> None:
    """Apply the tau_p trainable layers, continuing the brick-wall pattern.

    Each gate is ``qml.ArbitraryUnitary``: a universal two-qubit gate with 15
    Pauli-basis angles, equal to the identity at zero angles.
    """
    gate_params = theta.reshape(-1, PARAMETERS_PER_SU4_GATE)
    gate_index = 0
    for layer in range(config.num_peaking_layers):
        layer_index = config.num_random_layers + layer
        for wires in brick_wall_pairs(layer_index, config.num_qubits):
            qml.ArbitraryUnitary(gate_params[gate_index], wires=wires)
            gate_index += 1


def build_peak_weight_fn(
    config: CircuitConfig, haar_layers: list[HaarLayer]
) -> PeakWeightFunction:
    """Return a differentiable function theta -> delta_{0^n}(C(theta)) (Eq. 9)."""
    device = qml.device("default.qubit", wires=config.num_qubits)
    all_zeros = np.zeros(config.num_qubits, dtype=int)

    @qml.qnode(device, interface="autograd", diff_method="backprop")
    def peak_weight(theta):
        _apply_random_section(config, haar_layers)
        _apply_peaking_section(config, theta)
        return qml.expval(qml.Projector(all_zeros, wires=range(config.num_qubits)))

    return peak_weight


def random_section_peak_weight(
    config: CircuitConfig, haar_layers: list[HaarLayer]
) -> float:
    """Peak weight of the random section alone; ~2^-n for a deep RQC."""
    device = qml.device("default.qubit", wires=config.num_qubits)
    all_zeros = np.zeros(config.num_qubits, dtype=int)

    @qml.qnode(device)
    def peak_weight():
        _apply_random_section(config, haar_layers)
        return qml.expval(qml.Projector(all_zeros, wires=range(config.num_qubits)))

    return float(peak_weight())


def maximize_peak_weight(
    peak_weight_fn: PeakWeightFunction,
    num_parameters: int,
    optimizer_config: OptimizerConfig,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Run a batch of independently initialized Adam runs and keep the best.

    Random restarts are the paper's mitigation for barren plateaus and local
    optima (Section 4). Each run decays the Adam step size by
    ``decay_factor`` every ``decay_every`` steps and stops early once the
    per-step cost change drops below ``convergence_tol``.

    Returns ``(best_peak_weight, best_parameters)``.
    """

    def cost(theta):
        return -peak_weight_fn(theta)  # minimizing -delta maximizes delta

    best_value = -np.inf
    best_theta: np.ndarray | None = None

    for restart in range(1, optimizer_config.num_restarts + 1):
        theta = pnp.array(
            rng.normal(0.0, optimizer_config.initial_scale, num_parameters),
            requires_grad=True,
        )
        optimizer = qml.AdamOptimizer(stepsize=optimizer_config.learning_rate)
        previous_cost = np.inf

        for step in range(1, optimizer_config.max_steps + 1):
            # step_and_cost returns the cost at theta *before* this update.
            theta, current_cost = optimizer.step_and_cost(cost, theta)
            current_cost = float(current_cost)

            if step % optimizer_config.log_every == 0:
                logger.info(
                    "restart %d/%d  step %4d/%d  peak weight = %.4f",
                    restart,
                    optimizer_config.num_restarts,
                    step,
                    optimizer_config.max_steps,
                    float(peak_weight_fn(theta)),
                )
            if (
                step > optimizer_config.min_steps
                and abs(previous_cost - current_cost) < optimizer_config.convergence_tol
            ):
                logger.info(
                    "restart %d converged after %d steps (per-step |cost change| "
                    "< %.0e)",
                    restart,
                    step,
                    optimizer_config.convergence_tol,
                )
                break
            previous_cost = current_cost
            if step % optimizer_config.decay_every == 0:
                optimizer.stepsize *= optimizer_config.decay_factor

        final_value = float(peak_weight_fn(theta))
        logger.info("restart %d finished with peak weight %.4f", restart, final_value)
        if final_value > best_value:
            best_value = final_value
            best_theta = np.asarray(theta)

    assert best_theta is not None  # num_restarts >= 1 guarantees at least one run
    return best_value, best_theta


def run_experiment(
    circuit_config: CircuitConfig,
    optimizer_config: OptimizerConfig,
    seed: int | None = None,
) -> ExperimentResult:
    """Build one random circuit instance and optimize its peaking layers."""
    rng = np.random.default_rng(seed)
    haar_layers = sample_haar_random_layers(circuit_config, rng)

    baseline = random_section_peak_weight(circuit_config, haar_layers)
    logger.info(
        "n = %d, tau_r = %d, tau_p = %d, %d trainable parameters",
        circuit_config.num_qubits,
        circuit_config.num_random_layers,
        circuit_config.num_peaking_layers,
        circuit_config.num_peaking_parameters,
    )
    logger.info(
        "peak weight of the random section alone: %.2e (1/2^n = %.2e)",
        baseline,
        2.0**-circuit_config.num_qubits,
    )

    peak_weight_fn = build_peak_weight_fn(circuit_config, haar_layers)
    start = time.perf_counter()
    best_value, best_theta = maximize_peak_weight(
        peak_weight_fn,
        circuit_config.num_peaking_parameters,
        optimizer_config,
        rng,
    )
    elapsed = time.perf_counter() - start

    is_paper_configuration = (
        circuit_config.num_qubits,
        circuit_config.num_random_layers,
        circuit_config.num_peaking_layers,
    ) == PAPER_CONFIGURATION
    logger.info("optimization took %.1f s", elapsed)
    logger.info(
        "best peak weight after optimization: %.4f%s",
        best_value,
        " (paper: ~0.2 for this configuration)" if is_paper_configuration else "",
    )
    return ExperimentResult(
        peak_weight=best_value,
        baseline_peak_weight=baseline,
        parameters=best_theta,
    )


def instance_seeds(base_seed: int, num_instances: int) -> list[int]:
    """Derive one RNG seed per circuit instance, deterministically.

    A single instance keeps ``base_seed`` unchanged, so single-instance runs
    stay reproducible from the command line; multiple instances get
    independent sub-seeds spawned from it.
    """
    if num_instances == 1:
        return [base_seed]
    return [
        int(s) for s in np.random.SeedSequence(base_seed).generate_state(num_instances)
    ]


def save_result(
    path: str, circuit_config: CircuitConfig, result: ExperimentResult, seed: int
) -> None:
    """Persist the best angles and the metadata needed to rebuild the circuit."""
    np.savez(
        path,
        parameters=result.parameters,
        peak_weight=result.peak_weight,
        baseline_peak_weight=result.baseline_peak_weight,
        num_qubits=circuit_config.num_qubits,
        num_random_layers=circuit_config.num_random_layers,
        num_peaking_layers=circuit_config.num_peaking_layers,
        seed=seed,
    )
    logger.info("saved best parameters to %s", path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--num-qubits", type=int, default=12, help="n (paper: 10, 12)")
    parser.add_argument(
        "--random-layers", type=int, default=40, help="tau_r (paper: 40)"
    )
    parser.add_argument(
        "--peaking-layers", type=int, default=10, help="tau_p (paper: 10)"
    )
    parser.add_argument("--restarts", type=int, default=3, help="independent Adam runs")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="maximum Adam steps per restart (early stopping may end sooner)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="initial Adam step size (halved every 300 steps)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help="independent circuit instances to average over (paper Fig. 2d: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="base RNG seed (circuit + initialization)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="save the best instance's parameters and metadata to this .npz file",
    )
    args = parser.parse_args(argv)
    if args.instances < 1:
        parser.error("--instances must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    args = parse_args(argv)
    circuit_config = CircuitConfig(
        num_qubits=args.num_qubits,
        num_random_layers=args.random_layers,
        num_peaking_layers=args.peaking_layers,
    )
    optimizer_config = OptimizerConfig(
        num_restarts=args.restarts,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )

    results: list[tuple[int, ExperimentResult]] = []
    for index, seed in enumerate(instance_seeds(args.seed, args.instances), start=1):
        if args.instances > 1:
            logger.info("=== instance %d/%d (seed %d) ===", index, args.instances, seed)
        results.append(
            (seed, run_experiment(circuit_config, optimizer_config, seed=seed))
        )

    if args.instances > 1:
        peak_weights = np.array([result.peak_weight for _, result in results])
        logger.info(
            "average peak weight over %d instances: %.4f +/- %.4f (max %.4f)",
            len(peak_weights),
            peak_weights.mean(),
            peak_weights.std() / np.sqrt(len(peak_weights)),
            peak_weights.max(),
        )

    if args.output is not None:
        best_seed, best_result = max(results, key=lambda item: item[1].peak_weight)
        save_result(args.output, circuit_config, best_result, best_seed)


if __name__ == "__main__":
    main()
