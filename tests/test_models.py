"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest

from inflammation.models import daily_mean, daily_max, daily_min


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
    ],
)
def test_daily_mean_integers(test_input, expected):
    """Test that mean function works for an array of positive integers."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), np.array(expected))


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),
        ([[1, 6], [23, 7], [17, 6]], [23, 7]),
    ],
)
def test_daily_max_integers(test_input, expected):
    """Test that mean function works for an array of positive integers."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test_input), np.array(expected))

@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),
        ([[1, 6], [23, 7], [17, 6]], [1, 6]),
    ],
)
def test_daily_min_integers(test_input, expected):
    """Test that mean function works for an array of positive integers."""

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test_input), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([["Hello", "there"], ["General", "Kenobi"]])
