"""
utils.py — Matrix helpers and validation utilities.
"""

import numpy as np


# ──────────────────────────────────────────────
# Constructors
# ──────────────────────────────────────────────

def make_identity(n: int) -> np.ndarray:
    """Return an (n×n) identity matrix."""
    return np.eye(n)


def make_zero(rows: int, cols: int | None = None) -> np.ndarray:
    """Return a zero matrix of shape (rows × cols). Square if cols is None."""
    cols = cols if cols is not None else rows
    return np.zeros((rows, cols))


# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────

def validate_matrix(M: np.ndarray, name: str, expected_shape: tuple) -> None:
    """
    Assert that M is a NumPy array with the expected shape.

    Raises
    ------
    TypeError  : if M is not an ndarray
    ValueError : if shape does not match
    """
    if not isinstance(M, np.ndarray):
        raise TypeError(f"'{name}' must be a numpy ndarray, got {type(M).__name__}.")
    if M.shape != expected_shape:
        raise ValueError(
            f"'{name}' has shape {M.shape}, expected {expected_shape}."
        )


def validate_positive_definite(M: np.ndarray, name: str) -> None:
    """
    Assert that M is symmetric positive-definite.

    Raises
    ------
    ValueError : if the matrix is not symmetric or not positive-definite
    """
    if not np.allclose(M, M.T):
        raise ValueError(f"'{name}' must be symmetric.")
    eigenvalues = np.linalg.eigvalsh(M)
    if np.any(eigenvalues <= 0):
        raise ValueError(f"'{name}' must be positive-definite (all eigenvalues > 0).")


# ──────────────────────────────────────────────
# Numerics
# ──────────────────────────────────────────────

def safe_invert(M: np.ndarray) -> np.ndarray:
    """
    Invert M using numpy. Falls back to pseudo-inverse if singular.
    """
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M)


def symmetrize(M: np.ndarray) -> np.ndarray:
    """Force M to be exactly symmetric: M = (M + Mᵀ) / 2."""
    return (M + M.T) / 2


def nearest_positive_definite(M: np.ndarray) -> np.ndarray:
    """
    Find the nearest positive-definite matrix to M (Higham, 1988).
    Useful for covariance repair after numerical drift.
    """
    B = symmetrize(M)
    _, s, Vt = np.linalg.svd(B)
    H = Vt.T @ np.diag(s) @ Vt
    M2 = (B + H) / 2
    M3 = symmetrize(M2)

    # Ensure positive-definiteness via small diagonal bump
    spacing = np.spacing(np.linalg.norm(M3))
    I = np.eye(M3.shape[0])
    k = 1
    while not _is_positive_definite(M3):
        min_eig = np.min(np.linalg.eigvalsh(M3))
        M3 += I * (-min_eig * k ** 2 + spacing)
        k += 1

    return M3


def _is_positive_definite(M: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False
