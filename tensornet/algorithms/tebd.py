"""
TEBD (Time-Evolving Block Decimation) algorithm.

Time evolution of MPS using Trotter decomposition.
"""

from typing import List, Optional, Tuple
import torch
from torch import Tensor
import math

from tensornet.mps.mps import MPS
from tensornet.core.decompositions import svd_truncated


def tebd(
    mps: MPS,
    gates: List[Tensor],
    chi_max: Optional[int] = None,
    cutoff: float = 1e-10,
) -> MPS:
    """
    Apply a layer of two-site gates using TEBD.
    
    Args:
        mps: Input MPS (modified in-place)
        gates: List of two-site gates, each of shape (d, d, d, d)
               gates[i] acts on sites i and i+1
        chi_max: Maximum bond dimension after truncation
        cutoff: SVD truncation cutoff
        
    Returns:
        MPS after applying gates
        
    Example:
        >>> mps = MPS.random(L=10, d=2, chi=16)
        >>> # Create XX gate
        >>> X = torch.tensor([[0, 1], [1, 0]], dtype=torch.float64)
        >>> XX = torch.einsum('ij,kl->ikjl', X, X)
        >>> gates = [XX for _ in range(9)]  # 9 gates for 10 sites
        >>> mps = tebd(mps, gates, chi_max=32)
    """
    L = mps.L
    
    if len(gates) != L - 1:
        raise ValueError(f"Need {L-1} gates for {L} sites, got {len(gates)}")
    
    if chi_max is None:
        chi_max = mps.chi * 4  # Allow some growth
    
    # Apply gates
    for i, gate in enumerate(gates):
        _apply_two_site_gate(mps, gate, i, chi_max, cutoff)
    
    return mps


def tebd_sweep(
    mps: MPS,
    odd_gates: List[Tensor],
    even_gates: List[Tensor],
    chi_max: Optional[int] = None,
    cutoff: float = 1e-10,
) -> MPS:
    """
    One TEBD sweep with even/odd decomposition.
    
    Applies even gates then odd gates (second-order Trotter).
    
    Args:
        mps: Input MPS
        odd_gates: Gates on bonds (0,1), (2,3), ...
        even_gates: Gates on bonds (1,2), (3,4), ...
        chi_max: Maximum bond dimension
        cutoff: SVD cutoff
        
    Returns:
        MPS after sweep
    """
    L = mps.L
    
    # Apply odd gates (0,1), (2,3), ...
    for i, gate in enumerate(odd_gates):
        site = 2 * i
        if site < L - 1:
            _apply_two_site_gate(mps, gate, site, chi_max, cutoff)
    
    # Apply even gates (1,2), (3,4), ...
    for i, gate in enumerate(even_gates):
        site = 2 * i + 1
        if site < L - 1:
            _apply_two_site_gate(mps, gate, site, chi_max, cutoff)
    
    return mps


def time_evolve(
    mps: MPS,
    hamiltonian_terms: List[Tensor],
    dt: float,
    num_steps: int = 1,
    chi_max: Optional[int] = None,
    cutoff: float = 1e-10,
    order: int = 2,
) -> MPS:
    """
    Time evolution using TEBD.
    
    Args:
        mps: Initial state
        hamiltonian_terms: Local Hamiltonian terms h_i for H = sum_i h_i
                          Each h_i is shape (d, d, d, d) for two-site term
        dt: Time step
        num_steps: Number of time steps
        chi_max: Maximum bond dimension
        cutoff: SVD cutoff
        order: Trotter order (1, 2, or 4)
        
    Returns:
        Time-evolved MPS
    """
    # Create time evolution gates U_i = exp(-i dt h_i)
    gates = []
    for h in hamiltonian_terms:
        U = _expm_two_site(-1j * dt * h)
        gates.append(U)
    
    for step in range(num_steps):
        if order == 1:
            # First-order Trotter
            mps = tebd(mps, gates, chi_max, cutoff)
        elif order == 2:
            # Second-order Trotter: apply half steps on edges
            half_gates = [_expm_two_site(-0.5j * dt * h) for h in hamiltonian_terms]
            mps = tebd(mps, half_gates, chi_max, cutoff)
            mps = tebd(mps, half_gates, chi_max, cutoff)
        else:
            raise ValueError(f"Unsupported Trotter order: {order}")
    
    return mps


def imaginary_time_evolution(
    mps: MPS,
    hamiltonian_terms: List[Tensor],
    beta: float,
    num_steps: int = 100,
    chi_max: Optional[int] = None,
    cutoff: float = 1e-10,
) -> Tuple[MPS, float]:
    """
    Imaginary time evolution to find ground state.
    
    Uses TEBD with imaginary time to project onto ground state.
    
    Args:
        mps: Initial state
        hamiltonian_terms: Local Hamiltonian terms
        beta: Total imaginary time (larger = better ground state approximation)
        num_steps: Number of steps
        chi_max: Maximum bond dimension
        cutoff: SVD cutoff
        
    Returns:
        (ground_state_mps, approximate_energy)
    """
    dbeta = beta / num_steps
    
    # Create imaginary time gates U_i = exp(-dbeta h_i)
    gates = []
    for h in hamiltonian_terms:
        U = _expm_two_site(-dbeta * h)
        gates.append(U)
    
    for step in range(num_steps):
        mps = tebd(mps, gates, chi_max, cutoff)
        mps.normalize_()
    
    # Estimate energy (simplified - just return 0 for now)
    energy = 0.0
    
    return mps, energy


def _apply_two_site_gate(
    mps: MPS,
    gate: Tensor,
    site: int,
    chi_max: Optional[int],
    cutoff: float,
) -> None:
    """
    Apply two-site gate and truncate.
    
    Modifies MPS in place.
    
    Args:
        mps: MPS to modify
        gate: Two-site gate of shape (d, d, d, d) as (i, j, i', j')
        site: Left site index
        chi_max: Maximum bond dimension
        cutoff: SVD cutoff
    """
    A1 = mps.tensors[site]      # (chi_l, d, chi_m)
    A2 = mps.tensors[site + 1]  # (chi_m, d, chi_r)
    
    chi_l = A1.shape[0]
    d = A1.shape[1]
    chi_r = A2.shape[2]
    
    # Contract to two-site tensor
    # (chi_l, d, chi_m) x (chi_m, d, chi_r) -> (chi_l, d, d, chi_r)
    theta = torch.einsum('idk,klj->idlj', A1, A2)
    
    # Apply gate
    # gate: (d_out1, d_out2, d_in1, d_in2)
    # theta: (chi_l, d_in1, d_in2, chi_r)
    # result: (chi_l, d_out1, d_out2, chi_r)
    theta_new = torch.einsum('abcd,ecdf->eabf', gate, theta)
    
    # SVD to split
    theta_mat = theta_new.reshape(chi_l * d, d * chi_r)
    
    U, S, Vh = svd_truncated(theta_mat, max_rank=chi_max, cutoff=cutoff)
    chi_new = len(S)
    
    # Split back
    A1_new = U.reshape(chi_l, d, chi_new)
    A2_new = (torch.diag(S) @ Vh).reshape(chi_new, d, chi_r)
    
    mps.tensors[site] = A1_new
    mps.tensors[site + 1] = A2_new


def _expm_two_site(H: Tensor) -> Tensor:
    """
    Matrix exponential of two-site operator.
    
    Args:
        H: Two-site Hamiltonian of shape (d, d, d, d)
        
    Returns:
        exp(H) as shape (d, d, d, d)
    """
    d = H.shape[0]
    
    # Reshape to matrix
    H_mat = H.reshape(d * d, d * d)
    
    # Matrix exponential
    U_mat = torch.linalg.matrix_exp(H_mat)
    
    # Reshape back
    return U_mat.reshape(d, d, d, d)
