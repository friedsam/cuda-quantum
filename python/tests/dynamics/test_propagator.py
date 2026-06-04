import numpy as np
import cudaq
from cudaq import spin, Schedule


def test_propagator_constant_x_hamiltonian():
    cudaq.reset_target()

    t_final = 0.1
    propagator = cudaq.propagator(
        spin.x(0),
        {0: 2},
        Schedule([0.0, t_final], ["t"]),
    )

    expected = np.array([
        [np.cos(t_final), -1j * np.sin(t_final)],
        [-1j * np.sin(t_final), np.cos(t_final)],
    ])

    np.testing.assert_allclose(propagator, expected, atol=1e-12)
