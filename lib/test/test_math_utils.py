import numpy as np
from multi_robot_herding.utils.utils import MathUtils
from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
import pytest


def test_sigma_1_case_1():
    z = np.array([1, 2, 3, 4, 5])
    result = MathUtils.sigma_1(z)
    assert np.allclose(result, np.array(
        [0.70710678, 0.89442719, 0.9486833, 0.9701425, 0.98058068]))


def test_sigma_norm():
    z = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    result = MathUtils.sigma_norm(z)
    assert np.allclose(result, np.array([
        [2.24744871],
        [8.70828693],
        [16.64582519],
        [25.07135583]]))


def test_sigma_norm_grad():
    z = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    result = MathUtils.sigma_norm_grad(z)
    assert np.allclose(result, np.array([
        [0.81649658, 1.63299316],
        [1.60356745, 2.13808994],
        [1.87646656, 2.25175988],
        [1.99593082, 2.28106379]]))


def test_bump_function():
    z = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
    result = MathUtils.bump_function(z)
    assert np.allclose(result, np.array(
        [1, 1, 0.85355339, 0.5, 0.14644661, 0]))


def test_phi():
    # z = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    z = np.array([1, 2, 3, 4, 5])
    result = MathUtils.phi(z)
    assert np.allclose(result, np.array(
        [3.53553391, 4.47213595, 4.74341649, 4.8507125,  4.90290338]))


def test_phi_alpha():
    z = np.array([1, 2, 3, 4, 5])
    result = MathUtils.phi_alpha(z)
    assert np.allclose(result, np.array(
        [-4.99981385, -4.9998106,  -4.99980726, -4.99980383, -4.99980031]))


if __name__ == '__main__':
    # pytest.main()
    test_phi_alpha()
