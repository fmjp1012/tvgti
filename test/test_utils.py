# test_utils.py

import numpy as np
from ..utils import mat_to_vec, mat_to_vech, vech_to_mat, duplication_matrix


def test_mat_to_vec():
    # Test case 1
    S = np.array([[1, 2],
                  [3, 4]])
    expected = np.array([1, 2, 3, 4])
    result = mat_to_vec(S)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"

    # Test case 2
    S = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = mat_to_vec(S)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"


def test_mat_to_vech():
    # Test case 1
    S = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    expected = np.array([1, 2, 3, 5, 6, 9])
    result = mat_to_vech(S)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"

    # Test case 2
    S = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    expected = np.array([1, 2, 3, 4, 6, 7, 8, 11, 12, 16])
    result = mat_to_vech(S)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"


def test_vech_to_mat():
    # Test case 1
    s = np.array([1, 2, 3, 4, 5, 6])
    expected = np.array([[1, 2, 3],
                         [2, 4, 5],
                         [3, 5, 6]])
    result = vech_to_mat(s)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"

    # Test case 2
    s = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    expected = np.array([[1, 2, 3, 4],
                         [2, 5, 6, 7],
                         [3, 6, 8, 9],
                         [4, 7, 9, 10]])
    result = vech_to_mat(s)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"


def test_duplication_matrix():
    # Test case 1
    n = 2
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    result = duplication_matrix(n)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"

    n = 3
    expected = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])

    result = duplication_matrix(n)
    assert np.array_equal(
        result, expected), f"expected {expected}, but got {result}"


if __name__ == "__main__":
    import pytest
    pytest.main()
