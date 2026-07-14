"""Sanity tests for peaked_circuits.py.

Run with
    python test_peaked_circuits.py
or, if pytest is installed,
    pytest test_peaked_circuits.py
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from scipy.optimize import minimize
from scipy.stats import unitary_group

from peaked_circuits import (
    PARAMETERS_PER_SU4_GATE,
    CircuitConfig,
    OptimizerConfig,
    brick_wall_pairs,
    build_peak_weight_fn,
    random_section_peak_weight,
    sample_haar_random_layers,
)


def gate_matrix(params: np.ndarray) -> np.ndarray:
    """Matrix of the trainable two-qubit gate used in the peaking section."""
    return qml.matrix(qml.ArbitraryUnitary(params, wires=[0, 1]))


def test_brick_wall_pairs_layout() -> None:
    assert brick_wall_pairs(0, 6) == ((0, 1), (2, 3), (4, 5))
    assert brick_wall_pairs(1, 6) == ((1, 2), (3, 4))
    assert brick_wall_pairs(2, 5) == ((0, 1), (2, 3))  # open boundary, odd n
    assert brick_wall_pairs(1, 2) == ()  # odd layer on 2 qubits is empty


def test_gate_at_zero_is_identity() -> None:
    """Zero angles must give the identity, matching the official-code init."""
    assert np.allclose(gate_matrix(np.zeros(PARAMETERS_PER_SU4_GATE)), np.eye(4))


def test_gate_is_universal() -> None:
    """The 15-angle gate must reach arbitrary two-qubit unitaries."""
    rng = np.random.default_rng(0)
    for _ in range(2):
        target = unitary_group.rvs(4, random_state=rng)

        def infidelity(params: np.ndarray, target: np.ndarray = target) -> float:
            overlap = np.trace(target.conj().T @ gate_matrix(params))
            return 1.0 - abs(overlap) / 4.0

        best = min(
            minimize(
                infidelity,
                rng.uniform(-np.pi, np.pi, PARAMETERS_PER_SU4_GATE),
                method="L-BFGS-B",
            ).fun
            for _ in range(5)
        )
        assert best < 1e-6, f"gate could not fit a Haar target (residual {best})"


def test_initial_peak_weight_equals_baseline() -> None:
    """At theta = 0 the peaking section is the identity, so the objective
    starts exactly at the random-section baseline."""
    config = CircuitConfig(num_qubits=4, num_random_layers=3, num_peaking_layers=2)
    haar_layers = sample_haar_random_layers(config, np.random.default_rng(1))
    peak_weight_fn = build_peak_weight_fn(config, haar_layers)
    baseline = random_section_peak_weight(config, haar_layers)
    initial = float(peak_weight_fn(np.zeros(config.num_peaking_parameters)))
    assert np.isclose(initial, baseline)


def test_same_seed_gives_same_circuit() -> None:
    config = CircuitConfig(num_qubits=4, num_random_layers=3, num_peaking_layers=2)
    theta = np.random.default_rng(2).normal(0.0, 0.3, config.num_peaking_parameters)
    values = [
        float(
            build_peak_weight_fn(
                config, sample_haar_random_layers(config, np.random.default_rng(7))
            )(theta)
        )
        for _ in range(2)
    ]
    assert values[0] == values[1]


def test_config_validation() -> None:
    invalid_configs = (
        lambda: CircuitConfig(num_qubits=1),
        lambda: CircuitConfig(num_peaking_layers=0),
        lambda: OptimizerConfig(learning_rate=0.0),
        lambda: OptimizerConfig(decay_factor=1.5),
    )
    for make_config in invalid_configs:
        try:
            make_config()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = sorted(
        (name, fn)
        for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    for name, fn in tests:
        fn()
        print(f"{name}: OK")
    print(f"all {len(tests)} tests passed")
