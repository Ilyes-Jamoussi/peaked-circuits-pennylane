"""Reproduce Fig. 2c of arXiv:2404.14493 from a saved optimization result.

Plots the sorted output distribution of the random circuit alone against the
same circuit with its optimized peaking layers: the bulk of the two
distributions coincides, while the probability of the peaked string is boosted
by orders of magnitude.

Usage:
    python plot_distribution.py results/best.npz -o results/distribution.png

Requires matplotlib (plotting only; not needed to run the experiment).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml

from peaked_circuits import (
    CircuitConfig,
    _apply_peaking_section,
    _apply_random_section,
    sample_haar_random_layers,
)


def output_distributions(path: str) -> tuple[CircuitConfig, np.ndarray, np.ndarray]:
    """Rebuild the circuit saved in ``path`` and return both distributions.

    Returns ``(config, random_probs, peaked_probs)``, the output probabilities
    of the random section alone and of the full optimized circuit.
    """
    data = np.load(path)
    config = CircuitConfig(
        num_qubits=int(data["num_qubits"]),
        num_random_layers=int(data["num_random_layers"]),
        num_peaking_layers=int(data["num_peaking_layers"]),
    )
    # Same RNG consumption order as run_experiment: the Haar layers are drawn
    # first, so the seed alone reproduces the exact circuit.
    rng = np.random.default_rng(int(data["seed"]))
    haar_layers = sample_haar_random_layers(config, rng)
    device = qml.device("default.qubit", wires=config.num_qubits)

    @qml.qnode(device)
    def random_probs():
        _apply_random_section(config, haar_layers)
        return qml.probs(wires=range(config.num_qubits))

    @qml.qnode(device)
    def peaked_probs(theta):
        _apply_random_section(config, haar_layers)
        _apply_peaking_section(config, theta)
        return qml.probs(wires=range(config.num_qubits))

    theta = data["parameters"]
    return config, np.asarray(random_probs()), np.asarray(peaked_probs(theta))


def plot(
    config: CircuitConfig,
    random_probs: np.ndarray,
    peaked_probs: np.ndarray,
    output: str,
) -> None:
    """Save the sorted-distribution comparison plot (cf. paper Fig. 2c)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    basis = np.arange(random_probs.size)
    ax.semilogy(basis, np.sort(random_probs)[::-1], label="random layers", lw=1)
    ax.semilogy(
        basis,
        np.sort(peaked_probs)[::-1],
        label=f"random + peaking layers  (delta = {peaked_probs[0]:.3f})",
        lw=1,
    )
    ax.set_xlabel("basis states (sorted by probability)")
    ax.set_ylabel("output probability")
    ax.set_title(
        f"n = {config.num_qubits}, tau_r = {config.num_random_layers}, "
        f"tau_p = {config.num_peaking_layers}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"saved {output}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result", help=".npz file produced by peaked_circuits.py --output"
    )
    parser.add_argument("-o", "--output", default="results/distribution.png")
    args = parser.parse_args(argv)
    plot(*output_distributions(args.result), output=args.output)


if __name__ == "__main__":
    main()
