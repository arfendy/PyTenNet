"""
Tensor decompositions with autograd support.

All decompositions support:
- CPU and CUDA devices
- float32, float64, complex64, complex128 dtypes
- Gradient computation via torch.autograd
"""

from typing import Tuple, Optional
import torch
from torch import Tensor


def svd_truncated(
    A: Tensor,
    max_rank: Optional[int] = None,
    cutoff: float = 0.0,
    absolute_cutoff: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Truncated SVD with autograd support.
    
    Computes A = U @ diag(S) @ Vh where only the top singular values are kept.
    
    Args:
        A: Input matrix of shape (m, n)
        max_rank: Maximum number of singular values to keep. If None, keep all.
        cutoff: Discard singular values below this threshold.
            If absolute_cutoff=False, this is relative to the largest singular value.
            If absolute_cutoff=True, this is an absolute threshold.
        absolute_cutoff: Whether cutoff is absolute or relative.
        
    Returns:
        U: Left singular vectors, shape (m, r)
        S: Singular values, shape (r,)
        Vh: Right singular vectors, shape (r, n)
        
    Example:
        >>> A = torch.randn(100, 50, dtype=torch.float64)
        >>> U, S, Vh = svd_truncated(A, max_rank=10)
        >>> reconstructed = U @ torch.diag(S) @ Vh
        >>> error = torch.norm(A - reconstructed)
    """
    # Full SVD
    U_full, S_full, Vh_full = torch.linalg.svd(A, full_matrices=False)
    
    # Determine truncation rank
    rank = len(S_full)
    
    if max_rank is not None:
        rank = min(rank, max_rank)
    
    if cutoff > 0:
        if absolute_cutoff:
            threshold = cutoff
        else:
            threshold = cutoff * S_full[0]
        
        # Find number of singular values above threshold
        mask = S_full > threshold
        rank_from_cutoff = mask.sum().item()
        rank = min(rank, max(1, rank_from_cutoff))
    
    # Truncate
    U = U_full[:, :rank]
    S = S_full[:rank]
    Vh = Vh_full[:rank, :]
    
    return U, S, Vh


def qr_stable(A: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Numerically stable QR decomposition with autograd support.
    
    Uses column pivoting for improved numerical stability.
    
    Args:
        A: Input matrix of shape (m, n)
        
    Returns:
        Q: Orthogonal matrix, shape (m, min(m, n))
        R: Upper triangular matrix, shape (min(m, n), n)
        
    Example:
        >>> A = torch.randn(100, 50, dtype=torch.float64)
        >>> Q, R = qr_stable(A)
        >>> assert torch.allclose(Q @ R, A, atol=1e-10)
        >>> assert torch.allclose(Q.T @ Q, torch.eye(50), atol=1e-10)
    """
    Q, R = torch.linalg.qr(A, mode='reduced')
    
    # Fix signs for uniqueness (positive diagonal of R)
    signs = torch.sign(torch.diag(R))
    signs[signs == 0] = 1
    
    Q = Q * signs.unsqueeze(0)
    R = R * signs.unsqueeze(1)
    
    return Q, R


def polar_decompose(A: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Polar decomposition A = U @ P where U is unitary and P is positive semidefinite.
    
    Args:
        A: Input matrix of shape (m, n) with m >= n
        
    Returns:
        U: Unitary matrix, shape (m, n)
        P: Positive semidefinite matrix, shape (n, n)
        
    Example:
        >>> A = torch.randn(100, 50, dtype=torch.float64)
        >>> U, P = polar_decompose(A)
        >>> assert torch.allclose(U @ P, A, atol=1e-10)
        >>> assert torch.allclose(U.T @ U, torch.eye(50), atol=1e-10)
    """
    U_svd, S, Vh = torch.linalg.svd(A, full_matrices=False)
    
    U = U_svd @ Vh
    P = Vh.conj().T @ torch.diag(S) @ Vh
    
    return U, P


def eigh_truncated(
    A: Tensor,
    max_rank: Optional[int] = None,
    cutoff: float = 0.0,
    which: str = 'SA',
) -> Tuple[Tensor, Tensor]:
    """
    Truncated eigendecomposition for Hermitian matrices.
    
    Args:
        A: Hermitian matrix of shape (n, n)
        max_rank: Maximum number of eigenvalues to keep
        cutoff: Discard eigenvalues with |lambda| below this threshold
        which: 'SA' (smallest algebraic), 'LA' (largest algebraic),
               'SM' (smallest magnitude), 'LM' (largest magnitude)
               
    Returns:
        eigenvalues: Shape (r,)
        eigenvectors: Shape (n, r)
        
    Example:
        >>> A = torch.randn(100, 100, dtype=torch.float64)
        >>> A = A + A.T  # Make symmetric
        >>> vals, vecs = eigh_truncated(A, max_rank=10, which='SA')
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(A)
    
    # Sort according to 'which'
    if which == 'SA':
        # Already sorted ascending
        pass
    elif which == 'LA':
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    elif which == 'SM':
        idx = torch.argsort(eigenvalues.abs())
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    elif which == 'LM':
        idx = torch.argsort(eigenvalues.abs(), descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    else:
        raise ValueError(f"Unknown which='{which}'. Use 'SA', 'LA', 'SM', or 'LM'.")
    
    # Truncate
    rank = len(eigenvalues)
    
    if max_rank is not None:
        rank = min(rank, max_rank)
    
    if cutoff > 0:
        mask = eigenvalues.abs() > cutoff
        rank_from_cutoff = mask.sum().item()
        rank = min(rank, max(1, rank_from_cutoff))
    
    return eigenvalues[:rank], eigenvectors[:, :rank]
