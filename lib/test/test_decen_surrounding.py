import numpy as np
from multi_robot_herding.utils.utils import MathUtils
from multi_robot_herding.behavior.decentralised_surrounding import *
import pytest
import yaml


@pytest.fixture
def generate_behavior():
    config = 'default_config.yml'
    # Read yaml and extract configuration
    with open(f'/home/khoa/mr_ws/src/multi_robot_herding/lib/config/{config}', 'r') as file:
        config = yaml.safe_load(file)
    behavior_config = config['behavior']
    surround_config = behavior_config['math_formation']['params']
    qi = np.array([1, 2])
    qj = np.array([[3, 4], [5, 6], [7, 8]])
    pi = np.array([9, 10])
    pj = np.array([[11, 12], [13, 14], [15, 16]])

    surround_config["Cs"] = 1
    surround_config["Cr"] = 1
    surround_config["potential_func"]["edge_follow"]["c"] = 10
    surround_config["potential_func"]["edge_follow"]["m"] = 20
    dec_surround = DecentralisedSurrounding(**surround_config)
    yield dec_surround, qi, qj, pi, pj


def test_potential_edge(generate_behavior):
    dec_surround, qi, qj, pi, pj = generate_behavior
    result = dec_surround._potential_edge_following(
        qi=qi, qj=qj, d=130, gain=1)
    assert np.allclose(result, np.array([-444950.54236284, -444950.54236284]))


if __name__ == '__main__':
    pytest.main()
