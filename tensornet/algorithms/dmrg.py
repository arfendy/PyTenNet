"""
DMRG (Density Matrix Renormalization Group) algorithm.

Two-site DMRG for finding ground states of 1D systems.
"""

from typing import Tuple, Optional, Dict, Any
import torch
from torch import Tensor

from tensornet.mps.mps import MPS
from tensornet.mps.mpo import MPO
from tensornet.core.decompositions import svd_truncated
from tensornet.algorithms.lanczos import lanczos_ground_state


def dmrg(
    mps: MPS,
    mpo: MPO,
    num_sweeps: int = 10,
    chi_max: int = 64,
    cutoff: float = 1e-10,
    tol: float = 1e-8,
    lanczos_iterations: int = 50,
) -> Tuple[MPS, float, Dict[str, Any]]:
    """
    Run two-site DMRG.
    
    Alias for dmrg_two_site().
    """
    return dmrg_two_site(
        mps=mps,
        mpo=mpo,
        num_sweeps=num_sweeps,
        chi_max=chi_max,
        cutoff=cutoff,
        tol=tol,
        lanczos_iterations=lanczos_iterations,
    )


def dmrg_two_site(
    mps: MPS,
    mpo: MPO,
    num_sweeps: int = 10,
    chi_max: int = 64,
    cutoff: float = 1e-10,
    tol: float = 1e-8,
    lanczos_iterations: int = 50,
) -> Tuple[MPS, float, Dict[str, Any]]:
    """
    Two-site DMRG algorithm for finding ground states.
    
    Args:
        mps: Initial MPS (will be modified)
        mpo: Hamiltonian as MPO
        num_sweeps: Number of sweeps
        chi_max: Maximum bond dimension
        cutoff: SVD truncation cutoff
        tol: Energy convergence tolerance
        lanczos_iterations: Max Lanczos iterations per site
        
    Returns:
        (ground_state_mps, ground_state_energy, info_dict)
        
    Example:
        >>> mps = MPS.random(L=20, d=2, chi=16)
        >>> H = heisenberg_mpo(L=20)
        >>> psi, E, info = dmrg_two_site(mps, H, num_sweeps=10, chi_max=64)
        >>> print(f"Ground state energy: {E}")
    """
    L = mps.L
    
    if mpo.L != L:
        raise ValueError(f"Length mismatch: MPS has {L} sites, MPO has {mpo.L}")
    
    # Initialize environments
    # L_env[i]: environment from left up to site i (exclusive)
    # R_env[i]: environment from right starting at site i+1
    L_env = _init_left_environments(mps, mpo)
    R_env = _init_right_environments(mps, mpo)
    
    energy = 0.0
    energy_history = []
    
    for sweep in range(num_sweeps):
        # Right sweep: sites 0 to L-2
        # Before right sweep, rebuild ALL right environments from current MPS
        R_env = _init_right_environments(mps, mpo)
        
        for i in range(L - 1):
            energy = _update_two_site(
                mps, mpo, i, L_env, R_env,
                chi_max, cutoff, lanczos_iterations,
                direction='right'
            )
        
        # Left sweep: sites L-2 to 0
        # Before left sweep, rebuild ALL left environments from current MPS
        L_env = _build_left_environments(mps, mpo)
        
        for i in range(L - 2, -1, -1):
            energy = _update_two_site(
                mps, mpo, i, L_env, R_env,
                chi_max, cutoff, lanczos_iterations,
                direction='left'
            )
        
        # Compute true energy from full contraction
        energy = _compute_energy(mps, mpo)
        energy_history.append(energy)
        
        # Check convergence
        if sweep > 0 and abs(energy_history[-1] - energy_history[-2]) < tol:
            break
    
    info = {
        'num_sweeps': sweep + 1,
        'converged': sweep < num_sweeps - 1,
        'energy_history': energy_history,
        'final_chi': mps.chi,
    }
    
    return mps, energy, info


def _compute_energy(mps: MPS, mpo: MPO) -> float:
    """Compute <psi|H|psi> via environment contraction."""
    L_env = torch.ones(1, 1, 1, dtype=mps.dtype, device=mps.device)
    
    for i in range(mps.L):
        A = mps.tensors[i]
        W = mpo.tensors[i]
        # L_new[c,x,e] = sum L[a,w,b] A*[a,d,c] W[w,d,f,x] A[b,f,e]
        temp1 = torch.einsum('awb,adc->wdbc', L_env, A.conj())
        temp2 = torch.einsum('wdbc,wdfx->bfcx', temp1, W)
        L_env = torch.einsum('bfcx,bfe->cxe', temp2, A)
    
    return L_env[0, 0, 0].real.item()


def _init_left_environments(mps: MPS, mpo: MPO) -> list:
    """Initialize left environment tensors (only boundary)."""
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    L_env = [None] * (L + 1)
    
    # Leftmost environment: trivial
    L_env[0] = torch.ones(1, 1, 1, dtype=dtype, device=device)
    
    return L_env


def _build_left_environments(mps: MPS, mpo: MPO) -> list:
    """Build all left environment tensors from current MPS."""
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    L_env = [None] * (L + 1)
    
    # Leftmost environment: trivial
    L_env[0] = torch.ones(1, 1, 1, dtype=dtype, device=device)
    
    # Build from left to right
    for i in range(L):
        L_env[i + 1] = _contract_left_env(mps.tensors[i], mpo.tensors[i], L_env[i])
    
    return L_env


def _init_right_environments(mps: MPS, mpo: MPO) -> list:
    """Initialize right environment tensors."""
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    R_env = [None] * (L + 1)
    
    # Rightmost environment: trivial
    R_env[L] = torch.ones(1, 1, 1, dtype=dtype, device=device)
    
    # Build from right to left
    for i in range(L - 1, -1, -1):
        R_env[i] = _contract_right_env(mps.tensors[i], mpo.tensors[i], R_env[i + 1])
    
    return R_env


def _contract_right_env(A: Tensor, W: Tensor, R: Tensor) -> Tensor:
    """
    Contract right environment.
    
    A: MPS tensor (chi_l, d, chi_r)
    W: MPO tensor (D_l, d_out, d_in, D_r)
    R: Right environment (chi_r, D_r, chi_r')
    
    Returns: New right environment (chi_l, D_l, chi_l')
    
    Computes: R'[a,w,b] = sum_{c,d,d',x,c'} A*[a,d,c] W[w,d,d',x] A[b,d',c'] R[c,x,c']
    """
    # Step 1: Contract A* with R: A*[a,d,c] R[c,x,c'] -> temp1[a,d,x,c']
    temp1 = torch.einsum('adc,cxe->adxe', A.conj(), R)
    
    # Step 2: Contract temp1 with W: temp1[a,d,x,c'] W[w,d,f,x] -> temp2[a,w,f,c']
    temp2 = torch.einsum('adxe,wdfx->awfe', temp1, W)
    
    # Step 3: Contract temp2 with A: temp2[a,w,f,c'] A[b,f,c'] -> R'[a,w,b]
    R_new = torch.einsum('awfe,bfe->awb', temp2, A)
    
    return R_new


def _contract_left_env(A: Tensor, W: Tensor, L: Tensor) -> Tensor:
    """
    Contract left environment.
    
    L: Left environment (chi_l, D_l, chi_l')
    A: MPS tensor (chi_l, d, chi_r)
    W: MPO tensor (D_l, d_out, d_in, D_r)
    
    Returns: New left environment (chi_r, D_r, chi_r')
    
    Computes: L'[c,x,c'] = sum_{a,d,d',w,b} L[a,w,b] A*[a,d,c] W[w,d,d',x] A[b,d',c']
    """
    # Step 1: Contract L with A*: L[a,w,b] A*[a,d,c] -> temp1[w,d,b,c]
    temp1 = torch.einsum('awb,adc->wdbc', L, A.conj())
    
    # Step 2: Contract temp1 with W: temp1[w,d,b,c] W[w,d,f,x] -> temp2[b,f,c,x]
    temp2 = torch.einsum('wdbc,wdfx->bfcx', temp1, W)
    
    # Step 3: Contract temp2 with A: temp2[b,f,c,x] A[b,f,e] -> L'[c,x,e]
    L_new = torch.einsum('bfcx,bfe->cxe', temp2, A)
    
    return L_new


def _update_two_site(
    mps: MPS,
    mpo: MPO,
    site: int,
    L_env: list,
    R_env: list,
    chi_max: int,
    cutoff: float,
    lanczos_iterations: int,
    direction: str,
) -> float:
    """
    Update two sites in DMRG.
    
    Returns the energy after update.
    """
    i = site
    L = mps.L
    
    # Get two-site tensor
    A1 = mps.tensors[i]      # (chi_l, d, chi_m)
    A2 = mps.tensors[i + 1]  # (chi_m, d, chi_r)
    
    chi_l = A1.shape[0]
    d = A1.shape[1]
    chi_r = A2.shape[2]
    
    # Form two-site tensor theta
    # (chi_l, d, chi_m) x (chi_m, d, chi_r) -> (chi_l, d, d, chi_r)
    theta = torch.einsum('idk,klj->idlj', A1, A2)
    theta_shape = theta.shape
    
    # Get current environments with dimension checks
    Le = L_env[i]
    Re = R_env[i + 2]
    
    # Verify environment dimensions match current MPS
    if Le is None or Le.shape[0] != chi_l or Le.shape[2] != chi_l:
        # Rebuild left environment from scratch
        Le = torch.ones(1, 1, 1, dtype=mps.dtype, device=mps.device)
        for j in range(i):
            Le = _contract_left_env(mps.tensors[j], mpo.tensors[j], Le)
        L_env[i] = Le
    
    if Re is None or Re.shape[0] != chi_r or Re.shape[2] != chi_r:
        # Rebuild right environment from scratch
        Re = torch.ones(1, 1, 1, dtype=mps.dtype, device=mps.device)
        for j in range(L - 1, i + 1, -1):
            Re = _contract_right_env(mps.tensors[j], mpo.tensors[j], Re)
        R_env[i + 2] = Re
    
    # Flatten for Lanczos
    theta_flat = theta.reshape(-1)
    
    W1 = mpo.tensors[i]      # (D_l, d_out, d_in, D_m)
    W2 = mpo.tensors[i + 1]  # (D_m, d_out, d_in, D_r)
    
    # Define effective Hamiltonian application
    def apply_Heff(v):
        """Apply effective Hamiltonian to two-site tensor.
        
        Index convention:
        - Le[bra_l, w, ket_l]: left environment
        - Re[bra_r, x, ket_r]: right environment  
        - theta[ket_l, s, t, ket_r]: two-site ket wavefunction
        - W1[w, p_out, p_in, w']: left MPO tensor
        - W2[w', q_out, q_in, x]: right MPO tensor
        
        Computes: (H @ theta)[bra_l, p, q, bra_r]
        """
        v_2site = v.reshape(theta_shape)
        
        # Le[bra_l, w, ket_l] @ theta[ket_l, s, t, ket_r] -> [bra_l, w, s, t, ket_r]
        temp1 = torch.einsum('bwk,kstr->bwstr', Le, v_2site)
        
        # [bra_l, w, s, t, ket_r] @ W1[w, p, s, w'] -> [bra_l, p, w', t, ket_r]
        temp2 = torch.einsum('bwstr,wpsm->bpmtr', temp1, W1)
        
        # [bra_l, p, w', t, ket_r] @ W2[w', q, t, x] -> [bra_l, p, q, x, ket_r]
        temp3 = torch.einsum('bpmtr,mqtx->bpqxr', temp2, W2)
        
        # [bra_l, p, q, x, ket_r] @ Re[bra_r, x, ket_r] -> [bra_l, p, q, bra_r]
        result = torch.einsum('bpqxr,cxr->bpqc', temp3, Re)
        
        return result.reshape(-1)
    
    # Find ground state with Lanczos
    E, theta_opt = lanczos_ground_state(
        apply_Heff,
        theta_flat,
        num_iterations=lanczos_iterations,
    )
    
    theta_opt = theta_opt.reshape(theta_shape)
    
    # SVD to split back into two tensors
    # (chi_l, d, d, chi_r) -> (chi_l * d, d * chi_r)
    theta_mat = theta_opt.reshape(chi_l * d, d * chi_r)
    
    U, S, Vh = svd_truncated(theta_mat, max_rank=chi_max, cutoff=cutoff)
    chi_new = len(S)
    
    if direction == 'right':
        # U becomes left tensor, S @ Vh becomes right tensor
        A1_new = U.reshape(chi_l, d, chi_new)
        A2_new = (torch.diag(S) @ Vh).reshape(chi_new, d, chi_r)
        
        mps.tensors[i] = A1_new
        mps.tensors[i + 1] = A2_new
        
        # Update left environment for next site
        L_env[i + 1] = _contract_left_env(A1_new, mpo.tensors[i], L_env[i])
        
    else:  # direction == 'left'
        # U @ S becomes left tensor, Vh becomes right tensor
        A1_new = (U @ torch.diag(S)).reshape(chi_l, d, chi_new)
        A2_new = Vh.reshape(chi_new, d, chi_r)
        
        mps.tensors[i] = A1_new
        mps.tensors[i + 1] = A2_new
        
        # Update right environment for next site
        R_env[i + 1] = _contract_right_env(A2_new, mpo.tensors[i + 1], R_env[i + 2])
    
    # Update max chi
    mps.chi = max(mps.chi, chi_new)
    
    return E.item() if isinstance(E, Tensor) else E
