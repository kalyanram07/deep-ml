import numpy as np


def matrix_dot_vector(a: list[list[float]], b: list[float]) -> list[float] | int:
    """Computes the dot product of a matrix and a vector using NumPy.

    Args:
        a: A 2D list representing the matrix.
        b: A 1D list representing the vector.

    Returns:
        The resulting vector as a list, or -1 if dimensions are incompatible.
    """
    try:
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
    except Exception:
        return -1

    # Ensure 'a' is a 2D matrix, 'b' is a 1D vector, and neither is ragged/empty
    if a_arr.ndim != 2 or b_arr.ndim != 1 or a_arr.size == 0 or b_arr.size == 0:
        return -1

    # Check if number of columns in 'a' equals length of 'b'
    if a_arr.shape[1] != b_arr.shape[0]:
        return -1

    # Matrix-vector multiplication using the @ operator (or np.dot)
    return (a_arr @ b_arr).tolist()