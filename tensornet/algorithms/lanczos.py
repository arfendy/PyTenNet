"""
Lanczos algorithm for finding ground states.

Iterative eigensolver for large sparse Hermitian matrices.
"""

from typing import Tuple, Optional, Callable
import torch
from torch import Tensor


def lanczos_ground_state(
    matvec: Callable[[Tensor], Tensor],
    v0: Tensor,
    num_iterations: int = 100,
    tol: float = 1e-10,
) -> Tuple[Tensor, Tensor]:
    """
    Find the ground state using Lanczos iteration.
    
    Args:
        matvec: Function computing H @ v for a vector v
        v0: Initial vector
        num_iterations: Maximum number of Lanczos iterations
        tol: Convergence tolerance
        
    Returns:
        (eigenvalue, eigenvector)
    """
    dtype = v0.dtype
    device = v0.device
    n = v0.numel()
    
    # Normalize initial vector
    v = v0.flatten()
    v = v / torch.norm(v)
    
    # Lanczos vectors and tridiagonal matrix elements
    V = torch.zeros(n, num_iterations + 1, dtype=dtype, device=device)
    alpha = torch.zeros(num_iterations, dtype=dtype, device=device)
    beta = torch.zeros(num_iterations, dtype=dtype, device=device)
    
    V[:, 0] = v
    
    for j in range(num_iterations):
        # w = H @ v_j
        w = matvec(V[:, j].reshape(v0.shape)).flatten()
        
        # alpha_j = v_j^T @ w
        alpha[j] = torch.dot(V[:, j], w)
        
        # w = w - alpha_j * v_j - beta_{j-1} * v_{j-1}
        w = w - alpha[j] * V[:, j]
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]
        
        # Reorthogonalize for numerical stability
        for i in range(j + 1):
            w = w - torch.dot(V[:, i], w) * V[:, i]
        
        # beta_j = ||w||
        beta[j] = torch.norm(w)
        
        if beta[j] < tol:
            # Converged - invariant subspace found
            num_iterations = j + 1
            break
        
        # v_{j+1} = w / beta_j
        V[:, j + 1] = w / beta[j]
    
    # Build tridiagonal matrix
    T = torch.zeros(num_iterations, num_iterations, dtype=dtype, device=device)
    for j in range(num_iterations):
        T[j, j] = alpha[j]
        if j > 0:
            T[j, j - 1] = beta[j - 1]
            T[j - 1, j] = beta[j - 1]
    
    # Diagonalize tridiagonal matrix
    eigvals, eigvecs = torch.linalg.eigh(T)
    
    # Ground state is smallest eigenvalue
    idx = torch.argmin(eigvals)
    E0 = eigvals[idx]
    
    # Transform back to original space
    psi = V[:, :num_iterations] @ eigvecs[:, idx]
    psi = psi / torch.norm(psi)
    
    return E0, psi.reshape(v0.shape)


def lanczos_eigsh(
    matvec: Callable[[Tensor], Tensor],
    v0: Tensor,
    k: int = 1,
    num_iterations: int = 100,
    tol: float = 1e-10,
    which: str = 'SA',
) -> Tuple[Tensor, Tensor]:
    """
    Find k eigenvalues/eigenvectors using Lanczos.
    
    Args:
        matvec: Function computing H @ v
        v0: Initial vector
        k: Number of eigenvalues to find
        num_iterations: Maximum iterations
        tol: Convergence tolerance
        which: 'SA' (smallest algebraic), 'LA' (largest algebraic)
        
    Returns:
        (eigenvalues, eigenvectors) with shapes (k,) and (*v0.shape, k)
    """
    dtype = v0.dtype
    device = v0.device
    n = v0.numel()
    
    # Normalize initial vector
    v = v0.flatten()
    v = v / torch.norm(v)
    
    # Lanczos vectors and tridiagonal matrix elements
    max_iter = min(num_iterations, n)
    V = torch.zeros(n, max_iter + 1, dtype=dtype, device=device)
    alpha = torch.zeros(max_iter, dtype=dtype, device=device)
    beta = torch.zeros(max_iter, dtype=dtype, device=device)
    
    V[:, 0] = v
    actual_iter = max_iter
    
    for j in range(max_iter):
        w = matvec(V[:, j].reshape(v0.shape)).flatten()
        alpha[j] = torch.dot(V[:, j], w)
        
        w = w - alpha[j] * V[:, j]
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]
        
        # Reorthogonalize
        for i in range(j + 1):
            w = w - torch.dot(V[:, i], w) * V[:, i]
        
        beta[j] = torch.norm(w)
        
        if beta[j] < tol:
            actual_iter = j + 1
            break
        
        V[:, j + 1] = w / beta[j]
    
    # Build and diagonalize tridiagonal matrix
    T = torch.zeros(actual_iter, actual_iter, dtype=dtype, device=device)
    for j in range(actual_iter):
        T[j, j] = alpha[j]
        if j > 0:
            T[j, j - 1] = beta[j - 1]
            T[j - 1, j] = beta[j - 1]
    
    eigvals, eigvecs_T = torch.linalg.eigh(T)
    
    # Select k eigenvalues
    if which == 'SA':
        indices = torch.argsort(eigvals)[:k]
    elif which == 'LA':
        indices = torch.argsort(eigvals, descending=True)[:k]
    else:
        raise ValueError(f"Unknown which='{which}'")
    
    selected_eigvals = eigvals[indices]
    
    # Transform eigenvectors back
    eigvecs = V[:, :actual_iter] @ eigvecs_T[:, indices]
    
    # Normalize
    for i in range(k):
        eigvecs[:, i] = eigvecs[:, i] / torch.norm(eigvecs[:, i])
    
    # Reshape
    eigvecs = eigvecs.reshape(*v0.shape, k)
    
    return selected_eigvals, eigvecs
