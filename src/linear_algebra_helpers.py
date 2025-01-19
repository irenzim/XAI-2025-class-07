import numpy as np


class LinearAlgebraHelpers:
    """
    Class for basic linear algebra operations.
    """

    @staticmethod
    def matrix_multiply(A, B):
        """Multiply two matrices

        Parameters
        ----------
        A (list of lists or np.ndarray): First matrix.
        B (list of lists or np.ndarray): Second matrix.

        Returns
        ----------
        np.ndarray: The result of matrix multiplication
        """

        return np.dot(A, B)

    @staticmethod
    def dot_product(vector1, vector2):
        """Compute the dot product of two vectors.

        Parameters
        ----------
        vector1 : list or nd.array
            First vector.
        vector2 : list or nd.array
            Second vector.

        Returns
        ----------
        float: The dot product of two vectors.
        """
        np.dot(vector1, vector2)
