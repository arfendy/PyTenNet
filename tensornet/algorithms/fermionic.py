"""
Fermionic MPS
=============

Tensor networks for fermionic systems using Jordan-Wigner transformation.

Physics:
    Fermions have anticommutation relations: {c_i, c_j†} = δ_ij
    
    The Jordan-Wigner transformation maps fermions to spins:
    c_i = (∏_{j<i} σ^z_j) σ^-_i
    c_i† = (∏_{j<i} σ^z_j) σ^+_i
    
    This allows us to use standard MPS for fermionic systems,
    with the string operators (∏ σ^z) handled implicitly in the MPO.

Models:
    - Spinless fermion chain: H = -t Σᵢ (c†_i c_{i+1} + h.c.) + V Σᵢ n_i n_{i+1}
    - Hubbard model: H = -t Σᵢσ (c†_iσ c_{i+1,σ} + h.c.) + U Σᵢ n_i↑ n_i↓
    
The key insight is that in 1D with nearest-neighbor hopping,
the Jordan-Wigner strings cancel, making the MPO local.
"""

import torch
from typing import Tuple, Optional, Dict, Any, List
from tensornet import MPS
from tensornet.mps.mpo import MPO


def spinless_fermion_mpo(
    L: int,
    t: float = 1.0,
    V: float = 0.0,
    mu: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    Spinless fermion chain Hamiltonian as MPO.
    
    H = -t Σᵢ (c†_i c_{i+1} + c†_{i+1} c_i) + V Σᵢ n_i n_{i+1} - μ Σᵢ n_i
    
    After Jordan-Wigner transformation to spins:
    c†_i c_{i+1} = σ^+_i σ^-_{i+1}  (for nearest neighbors, no string!)
    n_i = (1 + σ^z_i) / 2
    
    Args:
        L: Number of sites
        t: Hopping amplitude
        V: Nearest-neighbor interaction
        mu: Chemical potential
        dtype: Data type
        device: Device
        
    Returns:
        MPO representation
    """
    if device is None:
        device = torch.device('cpu')
    
    d = 2  # |0⟩ = empty, |1⟩ = occupied
    
    # Operators in occupation basis
    # n = |1⟩⟨1|
    n = torch.tensor([[0, 0], [0, 1]], dtype=dtype, device=device)
    
    # c = |0⟩⟨1| (annihilation)
    c = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
    
    # c† = |1⟩⟨0| (creation)
    cdag = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
    
    # Identity
    I = torch.eye(2, dtype=dtype, device=device)
    
    # σ^z for Jordan-Wigner (not needed for NN hopping, but for interactions)
    # In occupation basis: σ^z = 1 - 2n = [[1,0],[0,-1]]
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    
    # MPO bond dimension: 4 (I, c†, c, I with accumulated terms)
    # W = [[I,    c†,   c,    0  ],
    #      [0,    0,    0,    c  ],
    #      [0,    0,    0,    c† ],
    #      [hI,   0,    0,    I  ]]
    # where h contains local terms
    
    D = 4  # MPO bond dimension
    
    tensors = []
    
    for i in range(L):
        W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
        
        if i == 0:
            # Left boundary: row vector
            # [I, c†, c, -μn]
            W[0, :, :, 0] = I
            W[0, :, :, 1] = cdag
            W[0, :, :, 2] = c
            W[0, :, :, 3] = -mu * n + V * n  # V*n for n_i n_{i+1}
        elif i == L - 1:
            # Right boundary: column vector
            # [0, -t*c, -t*c†, I]^T with local term
            W[0, :, :, 0] = -mu * n
            W[1, :, :, 0] = -t * c
            W[2, :, :, 0] = -t * cdag
            W[3, :, :, 0] = I
        else:
            # Bulk
            W[0, :, :, 0] = I
            W[0, :, :, 1] = cdag
            W[0, :, :, 2] = c
            W[0, :, :, 3] = -mu * n + V * n
            W[1, :, :, 3] = -t * c
            W[2, :, :, 3] = -t * cdag
            W[3, :, :, 3] = I
    
        tensors.append(W)
    
    # Reshape boundary tensors
    tensors[0] = tensors[0][0:1, :, :, :]  # (1, d, d, D)
    tensors[-1] = tensors[-1][:, :, :, 0:1]  # (D, d, d, 1)
    
    return MPO(tensors)


def hubbard_mpo(
    L: int,
    t: float = 1.0,
    U: float = 4.0,
    mu: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    Hubbard model Hamiltonian as MPO.
    
    H = -t Σᵢσ (c†_iσ c_{i+1,σ} + h.c.) + U Σᵢ n_i↑ n_i↓ - μ Σᵢ (n_i↑ + n_i↓)
    
    Uses a 4-dimensional local Hilbert space:
    |0⟩ = empty, |↑⟩ = spin up, |↓⟩ = spin down, |↑↓⟩ = doubly occupied
    
    Args:
        L: Number of sites
        t: Hopping amplitude
        U: On-site Coulomb repulsion
        mu: Chemical potential
        dtype: Data type
        device: Device
        
    Returns:
        MPO representation
    """
    if device is None:
        device = torch.device('cpu')
    
    d = 4  # |0⟩, |↑⟩, |↓⟩, |↑↓⟩
    
    # Basis: 0=empty, 1=up, 2=down, 3=both
    
    # Number operators
    n_up = torch.zeros(d, d, dtype=dtype, device=device)
    n_up[1, 1] = 1.0  # |↑⟩
    n_up[3, 3] = 1.0  # |↑↓⟩
    
    n_dn = torch.zeros(d, d, dtype=dtype, device=device)
    n_dn[2, 2] = 1.0  # |↓⟩
    n_dn[3, 3] = 1.0  # |↑↓⟩
    
    n_tot = n_up + n_dn
    n_double = torch.zeros(d, d, dtype=dtype, device=device)
    n_double[3, 3] = 1.0  # |↑↓⟩
    
    # Creation/annihilation operators (with fermionic signs)
    # c†_↑: |0⟩→|↑⟩, |↓⟩→|↑↓⟩
    cdag_up = torch.zeros(d, d, dtype=dtype, device=device)
    cdag_up[1, 0] = 1.0   # |0⟩ → |↑⟩
    cdag_up[3, 2] = 1.0   # |↓⟩ → |↑↓⟩ (no sign, up comes first)
    
    c_up = cdag_up.T.clone()
    
    # c†_↓: |0⟩→|↓⟩, |↑⟩→|↑↓⟩ (with sign from anticommutation)
    cdag_dn = torch.zeros(d, d, dtype=dtype, device=device)
    cdag_dn[2, 0] = 1.0   # |0⟩ → |↓⟩
    cdag_dn[3, 1] = -1.0  # |↑⟩ → -|↑↓⟩ (sign from c†_↓ c†_↑ = -c†_↑ c†_↓)
    
    c_dn = torch.zeros(d, d, dtype=dtype, device=device)
    c_dn[0, 2] = 1.0   # |↓⟩ → |0⟩
    c_dn[1, 3] = -1.0  # |↑↓⟩ → -|↑⟩
    
    # Identity
    I = torch.eye(d, dtype=dtype, device=device)
    
    # Fermion parity for Jordan-Wigner string
    P = torch.diag(torch.tensor([1, -1, -1, 1], dtype=dtype, device=device))
    
    # Build MPO with bond dimension 6
    # Operators to pass: I, c†_↑, c_↑, c†_↓, c_↓, accumulated
    D = 6
    
    tensors = []
    
    for i in range(L):
        W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
        
        local_term = U * n_double - mu * n_tot
        
        if i == 0:
            # Left boundary
            W[0, :, :, 0] = I
            W[0, :, :, 1] = cdag_up
            W[0, :, :, 2] = c_up
            W[0, :, :, 3] = cdag_dn
            W[0, :, :, 4] = c_dn
            W[0, :, :, 5] = local_term
        elif i == L - 1:
            # Right boundary
            W[0, :, :, 0] = local_term
            W[1, :, :, 0] = -t * c_up @ P  # Include parity for JW
            W[2, :, :, 0] = -t * cdag_up @ P
            W[3, :, :, 0] = -t * c_dn @ P
            W[4, :, :, 0] = -t * cdag_dn @ P
            W[5, :, :, 0] = I
        else:
            # Bulk
            W[0, :, :, 0] = I
            W[0, :, :, 1] = cdag_up
            W[0, :, :, 2] = c_up
            W[0, :, :, 3] = cdag_dn
            W[0, :, :, 4] = c_dn
            W[0, :, :, 5] = local_term
            W[1, :, :, 5] = -t * c_up @ P
            W[2, :, :, 5] = -t * cdag_up @ P
            W[3, :, :, 5] = -t * c_dn @ P
            W[4, :, :, 5] = -t * cdag_dn @ P
            W[5, :, :, 5] = I
        
        tensors.append(W)
    
    # Reshape boundaries
    tensors[0] = tensors[0][0:1, :, :, :]
    tensors[-1] = tensors[-1][:, :, :, 0:1]
    
    return MPO(tensors)


def fermi_sea_mps(
    L: int,
    n_particles: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create MPS for a Fermi sea (filled lowest modes).
    
    For spinless fermions, this is a product state with
    the first n_particles sites occupied.
    
    Args:
        L: Number of sites
        n_particles: Number of fermions
        dtype: Data type
        device: Device
        
    Returns:
        MPS representing the Fermi sea
    """
    if device is None:
        device = torch.device('cpu')
    
    tensors = []
    for i in range(L):
        A = torch.zeros(1, 2, 1, dtype=dtype, device=device)
        if i < n_particles:
            A[0, 1, 0] = 1.0  # Occupied
        else:
            A[0, 0, 0] = 1.0  # Empty
        tensors.append(A)
    
    return MPS(tensors)


def half_filled_mps(
    L: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPS:
    """
    Create half-filled MPS (alternating occupied/empty).
    
    This is a good initial state for repulsive interactions
    (CDW-like order).
    
    Args:
        L: Number of sites
        dtype: Data type
        device: Device
        
    Returns:
        MPS with alternating occupation
    """
    if device is None:
        device = torch.device('cpu')
    
    tensors = []
    for i in range(L):
        A = torch.zeros(1, 2, 1, dtype=dtype, device=device)
        if i % 2 == 0:
            A[0, 1, 0] = 1.0  # Occupied
        else:
            A[0, 0, 0] = 1.0  # Empty
        tensors.append(A)
    
    return MPS(tensors)


def compute_density(mps: MPS) -> torch.Tensor:
    """
    Compute local density ⟨n_i⟩ for each site.

    Args:
        mps: MPS state (for spinless fermions)

    Returns:
        Tensor of local densities
    """
    L = len(mps.tensors)
    dtype = mps.tensors[0].dtype
    device = mps.tensors[0].device

    # Number operator: n = |1><1|
    n_op = torch.tensor([[0, 0], [0, 1]], dtype=dtype, device=device)

    densities = []

    for site in range(L):
        # For product states, this simplifies greatly
        # ⟨n_site⟩ = Σ_d |A[site][:,d,:]|^2 * n[d,d]
        A = mps.tensors[site]
        
        # Contract left
        left = torch.ones(1, dtype=dtype, device=device)
        for i in range(site):
            Ai = mps.tensors[i]
            left = torch.einsum('l,ldr,Ldr->L', left, Ai.conj(), Ai)
        
        # Contract right
        right = torch.ones(1, dtype=dtype, device=device)
        for i in range(L - 1, site, -1):
            Ai = mps.tensors[i]
            right = torch.einsum('r,ldr,ldR->R', right, Ai.conj(), Ai)
        
        # Apply n at site and contract
        # ⟨n⟩ = left @ A^dag @ n @ A @ right
        density = torch.einsum('l,ldr,dD,lDR,R->', left, A.conj(), n_op, A, right)
        densities.append(torch.real(density).item())

    return torch.tensor(densities, dtype=dtype, device=device)
