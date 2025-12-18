"""
Standard Hamiltonians as MPOs.

Provides analytical MPO constructions for common models.
"""

from typing import Optional
import torch
from torch import Tensor
from tensornet.mps.mpo import MPO


def pauli_matrices(
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Return Pauli matrices sigma_x, sigma_y, sigma_z.
    
    Returns:
        (sigma_x, sigma_y, sigma_z) each of shape (2, 2)
    """
    if device is None:
        device = torch.device('cpu')
    
    sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    sigma_y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype, device=device)
    sigma_z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    
    return sigma_x, sigma_y, sigma_z


def spin_operators(
    S: float = 0.5,
    dtype: torch.dtype = torch.complex128,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Return spin operators S_x, S_y, S_z for spin S.
    
    Args:
        S: Spin value (0.5, 1, 1.5, ...)
        
    Returns:
        (S_x, S_y, S_z, S_p, S_m) where S_p = S_x + i*S_y, S_m = S_x - i*S_y
    """
    if device is None:
        device = torch.device('cpu')
    
    d = int(2 * S + 1)
    
    # S_z is diagonal with values S, S-1, ..., -S
    m_values = torch.arange(S, -S - 1, -1, dtype=dtype, device=device)
    S_z = torch.diag(m_values)
    
    # S_+ and S_- are raising/lowering operators
    # S_+ |S, m> = sqrt(S(S+1) - m(m+1)) |S, m+1>
    S_p = torch.zeros(d, d, dtype=dtype, device=device)
    S_m = torch.zeros(d, d, dtype=dtype, device=device)
    
    for i in range(d - 1):
        m = m_values[i + 1]  # m of the state we're raising FROM
        coef = torch.sqrt(S * (S + 1) - m * (m + 1))
        S_p[i, i + 1] = coef
        S_m[i + 1, i] = coef
    
    S_x = (S_p + S_m) / 2
    S_y = (S_p - S_m) / (2j)
    
    return S_x, S_y, S_z, S_p, S_m


def heisenberg_mpo(
    L: int,
    J: float = 1.0,
    Jz: Optional[float] = None,
    h: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    Heisenberg XXZ chain Hamiltonian as MPO.
    
    H = J * sum_i (S^x_i S^x_{i+1} + S^y_i S^y_{i+1}) + Jz * sum_i S^z_i S^z_{i+1} + h * sum_i S^z_i
    
    For XXX model, set Jz = J (or leave as None, which defaults to J).
    
    Args:
        L: Number of sites
        J: XY coupling strength
        Jz: Z coupling strength (defaults to J)
        h: Magnetic field
        dtype: Data type (real dtypes use real representation)
        device: Device
        
    Returns:
        MPO representation of Hamiltonian
    """
    if device is None:
        device = torch.device('cpu')
    
    if Jz is None:
        Jz = J
    
    d = 2  # Spin-1/2
    D = 5  # MPO bond dimension for Heisenberg
    
    # Use real representation for Heisenberg (S^x S^x + S^y S^y = (S^+ S^- + S^- S^+)/2)
    # For float dtype, construct with real matrices
    
    I = torch.eye(d, dtype=dtype, device=device)
    Z = torch.zeros(d, d, dtype=dtype, device=device)
    
    # Pauli matrices (real parts for real dtype)
    Sx = torch.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
    Sz = torch.tensor([[0.5, 0], [0, -0.5]], dtype=dtype, device=device)
    
    # For XY coupling, use S^+ S^- + S^- S^+ = 2(S^x S^x + S^y S^y)
    Sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
    Sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
    
    tensors = []
    
    for i in range(L):
        if i == 0:
            # First site: row vector
            # W = [I, Sp, Sm, Sz, h*Sz]
            W = torch.zeros(1, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = Sp
            W[0, :, :, 2] = Sm
            W[0, :, :, 3] = Sz
            W[0, :, :, 4] = h * Sz
            
        elif i == L - 1:
            # Last site: column vector
            # W = [h*Sz + Jz*Sz + J/2*(Sp + Sm), J/2*Sm, J/2*Sp, Jz*Sz, I]^T
            W = torch.zeros(D, d, d, 1, dtype=dtype, device=device)
            W[0, :, :, 0] = h * Sz
            W[1, :, :, 0] = (J / 2) * Sm
            W[2, :, :, 0] = (J / 2) * Sp
            W[3, :, :, 0] = Jz * Sz
            W[4, :, :, 0] = I
            
        else:
            # Bulk: full MPO matrix
            # W = [[I,  Sp,    Sm,    Sz,    h*Sz  ],
            #      [0,  0,     0,     0,     J/2*Sm],
            #      [0,  0,     0,     0,     J/2*Sp],
            #      [0,  0,     0,     0,     Jz*Sz ],
            #      [0,  0,     0,     0,     I     ]]
            W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = Sp
            W[0, :, :, 2] = Sm
            W[0, :, :, 3] = Sz
            W[0, :, :, 4] = h * Sz
            W[1, :, :, 4] = (J / 2) * Sm
            W[2, :, :, 4] = (J / 2) * Sp
            W[3, :, :, 4] = Jz * Sz
            W[4, :, :, 4] = I
        
        tensors.append(W)
    
    return MPO(tensors)


def tfim_mpo(
    L: int,
    J: float = 1.0,
    g: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    Transverse-field Ising model as MPO.
    
    H = -J * sum_i Z_i Z_{i+1} - g * sum_i X_i
    
    Critical point at g = 1 (for J = 1).
    
    Args:
        L: Number of sites
        J: Ising coupling
        g: Transverse field strength
        dtype: Data type
        device: Device
        
    Returns:
        MPO representation
    """
    if device is None:
        device = torch.device('cpu')
    
    d = 2
    D = 3
    
    I = torch.eye(d, dtype=dtype, device=device)
    X = torch.tensor([[0, 1], [1, 0]], dtype=dtype, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=dtype, device=device)
    
    tensors = []
    
    for i in range(L):
        if i == 0:
            W = torch.zeros(1, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = Z
            W[0, :, :, 2] = -g * X
            
        elif i == L - 1:
            W = torch.zeros(D, d, d, 1, dtype=dtype, device=device)
            W[0, :, :, 0] = -g * X
            W[1, :, :, 0] = -J * Z
            W[2, :, :, 0] = I
            
        else:
            W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = Z
            W[0, :, :, 2] = -g * X
            W[1, :, :, 2] = -J * Z
            W[2, :, :, 2] = I
        
        tensors.append(W)
    
    return MPO(tensors)


def xx_mpo(
    L: int,
    J: float = 1.0,
    h: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    XX model as MPO.
    
    H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1}) + h * sum_i Z_i
    
    This is equivalent to free fermions and exactly solvable.
    
    Args:
        L: Number of sites
        J: Coupling strength
        h: Magnetic field
        dtype: Data type
        device: Device
        
    Returns:
        MPO representation
    """
    # XX model is Heisenberg with Jz = 0
    return heisenberg_mpo(L, J=J, Jz=0.0, h=h, dtype=dtype, device=device)


def xyz_mpo(
    L: int,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    XYZ model as MPO.
    
    H = sum_i (Jx * X_i X_{i+1} + Jy * Y_i Y_{i+1} + Jz * Z_i Z_{i+1}) + h * sum_i Z_i
    
    Args:
        L: Number of sites
        Jx: X coupling strength
        Jy: Y coupling strength
        Jz: Z coupling strength
        h: Magnetic field
        dtype: Data type
        device: Device
        
    Returns:
        MPO representation
    """
    if device is None:
        device = torch.device('cpu')
    
    d = 2
    D = 5  # Bond dimension: I, Sx, Sy, Sz, accumulated
    
    I = torch.eye(d, dtype=dtype, device=device)
    Sx = torch.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
    Sz = torch.tensor([[0.5, 0], [0, -0.5]], dtype=dtype, device=device)
    
    # Use S+ S- representation for Sy Sy term
    Sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
    Sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
    
    tensors = []
    
    for i in range(L):
        if i == 0:
            W = torch.zeros(1, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = Sp
            W[0, :, :, 2] = Sm
            W[0, :, :, 3] = Sz
            W[0, :, :, 4] = h * Sz
            
        elif i == L - 1:
            # Jx (Sx Sx) + Jy (Sy Sy) = (Jx+Jy)/4 (S+ S- + S- S+) + (Jx-Jy)/4 (S+ S+ + S- S-)
            # For real Jx=Jy: coefficient is Jx/2 for S+ S- and S- S+
            W = torch.zeros(D, d, d, 1, dtype=dtype, device=device)
            W[0, :, :, 0] = h * Sz
            W[1, :, :, 0] = ((Jx + Jy) / 4) * Sm
            W[2, :, :, 0] = ((Jx + Jy) / 4) * Sp
            W[3, :, :, 0] = Jz * Sz
            W[4, :, :, 0] = I
            
        else:
            W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = Sp
            W[0, :, :, 2] = Sm
            W[0, :, :, 3] = Sz
            W[0, :, :, 4] = h * Sz
            W[1, :, :, 4] = ((Jx + Jy) / 4) * Sm
            W[2, :, :, 4] = ((Jx + Jy) / 4) * Sp
            W[3, :, :, 4] = Jz * Sz
            W[4, :, :, 4] = I
        
        tensors.append(W)
    
    return MPO(tensors)


def bose_hubbard_mpo(
    L: int,
    n_max: int = 3,
    t: float = 1.0,
    U: float = 1.0,
    mu: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> MPO:
    """
    Bose-Hubbard model as MPO.
    
    H = -t * sum_i (b_i^dag b_{i+1} + h.c.) + (U/2) * sum_i n_i(n_i-1) - mu * sum_i n_i
    
    Args:
        L: Number of sites
        n_max: Maximum occupation per site (Fock space truncation)
        t: Hopping strength
        U: On-site interaction
        mu: Chemical potential
        dtype: Data type
        device: Device
        
    Returns:
        MPO representation
    """
    if device is None:
        device = torch.device('cpu')
    
    d = n_max + 1  # Local Hilbert space dimension
    D = 4  # Bond dimension: I, b, b^dag, accumulated
    
    # Build local operators
    # b |n> = sqrt(n) |n-1>
    # b^dag |n> = sqrt(n+1) |n+1>
    # n |n> = n |n>
    
    b = torch.zeros(d, d, dtype=dtype, device=device)
    bdag = torch.zeros(d, d, dtype=dtype, device=device)
    n = torch.zeros(d, d, dtype=dtype, device=device)
    I = torch.eye(d, dtype=dtype, device=device)
    
    for i in range(d):
        n[i, i] = i
        if i > 0:
            b[i - 1, i] = (i ** 0.5)
            bdag[i, i - 1] = (i ** 0.5)
    
    # On-site term: (U/2) n(n-1) - mu * n
    onsite = (U / 2) * (n @ n - n) - mu * n
    
    tensors = []
    
    for i in range(L):
        if i == 0:
            W = torch.zeros(1, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = b
            W[0, :, :, 2] = bdag
            W[0, :, :, 3] = onsite
            
        elif i == L - 1:
            W = torch.zeros(D, d, d, 1, dtype=dtype, device=device)
            W[0, :, :, 0] = onsite
            W[1, :, :, 0] = -t * bdag
            W[2, :, :, 0] = -t * b
            W[3, :, :, 0] = I
            
        else:
            W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
            W[0, :, :, 0] = I
            W[0, :, :, 1] = b
            W[0, :, :, 2] = bdag
            W[0, :, :, 3] = onsite
            W[1, :, :, 3] = -t * bdag
            W[2, :, :, 3] = -t * b
            W[3, :, :, 3] = I
        
        tensors.append(W)
    
    return MPO(tensors)