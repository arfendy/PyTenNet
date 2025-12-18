"""
Standard quantum states as MPS.

Provides exact MPS constructions for analytically known states.
"""

from typing import Optional
import torch
from torch import Tensor
import math

from tensornet.mps.mps import MPS


def product_mps(
    state: Tensor,
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create product state MPS.
    
    |psi> = |s> tensor |s> tensor ... tensor |s>
    
    Args:
        state: Single-site state vector of shape (d,)
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        MPS with bond dimension 1
        
    Example:
        >>> up = torch.tensor([1.0, 0.0])
        >>> mps = product_mps(up, L=10)  # |0000000000>
    """
    if device is None:
        device = torch.device('cpu')
    
    state = state.to(dtype=dtype, device=device)
    d = len(state)
    
    # Normalize
    state = state / torch.norm(state)
    
    tensors = []
    for i in range(L):
        A = state.reshape(1, d, 1)
        tensors.append(A.clone())
    
    return MPS(tensors)


def ghz_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create GHZ state as MPS.
    
    |GHZ> = (|00...0> + |11...1>) / sqrt(2)
    
    This state has maximal entanglement entropy log(2) at every cut.
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        MPS with bond dimension 2
        
    Example:
        >>> mps = ghz_mps(L=5)
        >>> for bond in range(4):
        ...     S = mps.entanglement_entropy(bond)
        ...     print(f"S[{bond}] = {S:.6f}")  # Should all be ~0.693147
    """
    if device is None:
        device = torch.device('cpu')
    
    d = 2
    chi = 2
    
    tensors = []
    
    # First site: shape (1, 2, 2)
    A0 = torch.zeros(1, d, chi, dtype=dtype, device=device)
    A0[0, 0, 0] = 1.0 / math.sqrt(2)  # |0> -> bond 0
    A0[0, 1, 1] = 1.0 / math.sqrt(2)  # |1> -> bond 1
    tensors.append(A0)
    
    # Bulk sites: shape (2, 2, 2)
    for i in range(1, L - 1):
        A = torch.zeros(chi, d, chi, dtype=dtype, device=device)
        A[0, 0, 0] = 1.0  # bond 0, |0> -> bond 0
        A[1, 1, 1] = 1.0  # bond 1, |1> -> bond 1
        tensors.append(A)
    
    # Last site: shape (2, 2, 1)
    AL = torch.zeros(chi, d, 1, dtype=dtype, device=device)
    AL[0, 0, 0] = 1.0  # bond 0, |0>
    AL[1, 1, 0] = 1.0  # bond 1, |1>
    tensors.append(AL)
    
    return MPS(tensors)


def w_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create W state as MPS.
    
    |W> = (|100...0> + |010...0> + ... + |000...1>) / sqrt(L)
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        MPS with bond dimension 2
    """
    if device is None:
        device = torch.device('cpu')
    
    d = 2
    chi = 2
    norm = 1.0 / math.sqrt(L)
    
    tensors = []
    
    # First site: shape (1, 2, 2)
    A0 = torch.zeros(1, d, chi, dtype=dtype, device=device)
    A0[0, 0, 0] = 1.0       # |0> -> "no excitation yet" (bond 0)
    A0[0, 1, 1] = norm      # |1> -> "excitation happened" (bond 1)
    tensors.append(A0)
    
    # Bulk sites: shape (2, 2, 2)
    for i in range(1, L - 1):
        A = torch.zeros(chi, d, chi, dtype=dtype, device=device)
        A[0, 0, 0] = 1.0    # no excitation, stay in bond 0
        A[0, 1, 1] = norm   # create excitation, move to bond 1
        A[1, 0, 1] = 1.0    # already excited, stay in bond 1
        tensors.append(A)
    
    # Last site: shape (2, 2, 1)
    AL = torch.zeros(chi, d, 1, dtype=dtype, device=device)
    AL[0, 0, 0] = 0.0       # no excitation - not valid for W state
    AL[0, 1, 0] = norm      # create excitation at last site
    AL[1, 0, 0] = 1.0       # already excited, output |0>
    tensors.append(AL)
    
    return MPS(tensors)


def random_mps(
    L: int,
    d: int = 2,
    chi: int = 8,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    normalize: bool = True,
) -> MPS:
    """
    Create random MPS.
    
    Alias for MPS.random().
    
    Args:
        L: Number of sites
        d: Physical dimension
        chi: Bond dimension
        dtype: Data type
        device: Device
        normalize: Whether to normalize
        
    Returns:
        Random MPS
    """
    return MPS.random(L=L, d=d, chi=chi, dtype=dtype, device=device, normalize=normalize)
