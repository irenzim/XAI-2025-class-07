import numpy as np
import pytest

from src import LinearAlgebraOperations


def test_matrix_multiply():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    result = LinearAlgebraOperations.matrix_multiply(A, B)
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_array_equal(
        result, expected, err_msg="Matrix multiplication failed"
    )


def test_dot_product():
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]
    result = LinearAlgebraOperations.dot_product(vector1, vector2)
    expected = 32  # 1*4 + 2*5 + 3*6
    assert result == expected, "Dot product failed"
