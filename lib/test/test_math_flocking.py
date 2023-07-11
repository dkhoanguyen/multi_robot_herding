import numpy as np
from multi_robot_herding.utils.utils import MathUtils
from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
from multi_robot_herding.environment.spawner import Spawner
import pytest


@pytest.fixture
def generate_flock():
    config = {
        'follow_cursor': False,
        'sensing_range': 1000,
        'danger_range': 110,
        'initial_consensus': [1000, 350]
    }
    flock = MathematicalFlock(**config)
    qi = np.array([1, 2])
    qj = np.array([[3, 4], [5, 6], [7, 8]])
    pi = np.array([9, 10])
    pj = np.array([[11, 12], [13, 14], [15, 16]])
    yield flock, qi, qj, pi, pj


def test_get_a_ij(generate_flock):
    flock, qi, qj, pi, pj = generate_flock
    result = flock._get_a_ij(qi, qj, MathematicalFlock.ALPHA_RANGE)
    assert np.allclose(result, np.array([[1.], [1.], [1.]]))


def test_get_n_ij(generate_flock):
    flock, qi, qj, pi, pj = generate_flock
    result = flock._get_n_ij(qi, qj)
    assert np.allclose(result, np.array([[1.49071198, 1.49071198],
                                         [1.95180015, 1.95180015],
                                         [2.09529089, 2.09529089]]))


def test_gradient_term(generate_flock):
    flock, qi, qj, pi, pj = generate_flock
    result = flock._gradient_term(
        c=3.46410161514, qi=qi, qj=qj,
        r=MathematicalFlock.ALPHA_RANGE,
        d=MathematicalFlock.ALPHA_DISTANCE
    )
    assert np.allclose(result, np.array([-95.91318642, -95.91318642]))


def test_consensus_term(generate_flock):
    flock, qi, qj, pi, pj = generate_flock
    result = flock._velocity_consensus_term(
        c=3.46410161514,
        qi=qi, qj=qj,
        pi=pi, pj=pj,
        r=MathematicalFlock.ALPHA_RANGE)
    assert np.allclose(result, np.array([41.56921938, 41.56921938]))


def test_group_objective_term(generate_flock):
    flock, qi, qj, pi, pj = generate_flock
    target = np.array([20, 30])
    result = flock._group_objective_term(
        c1=MathematicalFlock.C1_gamma,
        c2=MathematicalFlock.C2_gamma,
        pos=target,
        qi=qi,
        pi=pi)
    assert np.allclose(result, np.array([0.96816679, 0.52467832]))


if __name__ == '__main__':
    pytest.main()
