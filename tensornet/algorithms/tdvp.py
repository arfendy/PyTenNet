"""
TDVP (Time-Dependent Variational Principle) algorithm.

Variational time evolution that keeps the MPS on the manifold
of fixed bond dimension, unlike TEBD which grows bond dimension.
"""

from typing import Tuple, Optional, Dict, Any, Callable
import torch
from torch import Tensor

from tensornet.mps.mps import MPS
from tensornet.mps.mpo import MPO
from tensornet.core.decompositions import svd_truncated


def tdvp_sweep(
    mps: MPS,
    mpo: MPO,
    dt: float,
    direction: str = 'right',
    normalize: bool = True,
) -> MPS:
    """
    Single TDVP sweep (one-site variant).
    
    Evolves the MPS by dt using the 1-site TDVP algorithm.
    This is the "projector-splitting" integrator.
    
    Args:
        mps: Input MPS state
        mpo: Hamiltonian MPO
        dt: Time step (can be real or imaginary)
        direction: 'right' or 'left' sweep direction
        normalize: Whether to normalize after sweep
    
    Returns:
        Time-evolved MPS
    """
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    # Build environments
    L_envs = _build_left_environments(mps, mpo)
    R_envs = _build_right_environments(mps, mpo)
    
    if direction == 'right':
        sites = range(L - 1)
    else:
        sites = range(L - 1, 0, -1)
    
    for i in sites:
        # Get current tensor
        A = mps.tensors[i]
        chi_l, d, chi_r = A.shape
        
        # Build effective Hamiltonian for site i
        Le = L_envs[i]
        Re = R_envs[i + 1]
        W = mpo.tensors[i]
        
        # Evolve A forward by dt/2
        def H_eff_A(v):
            return _apply_H_eff_one_site(v.reshape(A.shape), Le, W, Re).reshape(-1)
        
        A_new = _expm_lanczos(H_eff_A, A.reshape(-1), -1j * dt / 2)
        A_new = A_new.reshape(chi_l, d, chi_r)
        
        # QR decomposition to move orthogonality center
        if direction == 'right':
            # A[χ_l, d, χ_r] -> Q[χ_l*d, χ_r] R[χ_r, χ_r]
            A_mat = A_new.reshape(chi_l * d, chi_r)
            Q, R = torch.linalg.qr(A_mat)
            chi_new = Q.shape[1]
            mps.tensors[i] = Q.reshape(chi_l, d, chi_new)
            
            # Update left environment
            L_envs[i + 1] = _contract_left_env(mps.tensors[i], mpo.tensors[i], L_envs[i])
            
            # Evolve R backward by dt/2 (the "backward" part of projector splitting)
            Le_C = L_envs[i + 1]
            Re_C = R_envs[i + 1]
            
            def H_eff_C(v):
                return _apply_H_eff_zero_site(v.reshape(R.shape), Le_C, Re_C).reshape(-1)
            
            R_new = _expm_lanczos(H_eff_C, R.reshape(-1), +1j * dt / 2)
            R_new = R_new.reshape(chi_new, chi_r)
            
            # Absorb R into next site
            A_next = mps.tensors[i + 1]
            mps.tensors[i + 1] = torch.einsum('ab,bdc->adc', R_new, A_next)
            
        else:  # direction == 'left'
            # A[χ_l, d, χ_r] -> L[χ_l, χ_l] Q[χ_l, d*χ_r]
            A_mat = A_new.reshape(chi_l, d * chi_r)
            Q, R = torch.linalg.qr(A_mat.T)
            Q = Q.T
            R = R.T
            chi_new = Q.shape[0]
            mps.tensors[i] = Q.reshape(chi_new, d, chi_r)
            
            # Update right environment
            R_envs[i] = _contract_right_env(mps.tensors[i], mpo.tensors[i], R_envs[i + 1])
            
            # Evolve L backward
            Le_C = L_envs[i]
            Re_C = R_envs[i]
            
            def H_eff_C(v):
                return _apply_H_eff_zero_site(v.reshape(R.shape), Le_C, Re_C).reshape(-1)
            
            L_new = _expm_lanczos(H_eff_C, R.reshape(-1), +1j * dt / 2)
            L_new = L_new.reshape(chi_l, chi_new)
            
            # Absorb L into previous site
            A_prev = mps.tensors[i - 1]
            mps.tensors[i - 1] = torch.einsum('adb,bc->adc', A_prev, L_new)
    
    if normalize:
        mps.normalize_()
    
    return mps


def tdvp(
    mps: MPS,
    mpo: MPO,
    dt: float,
    num_steps: int = 1,
    normalize: bool = True,
) -> MPS:
    """
    TDVP time evolution (1-site, 2nd order).
    
    Uses symmetric sweep (right then left) for 2nd order accuracy.
    
    Args:
        mps: Initial MPS state
        mpo: Hamiltonian MPO
        dt: Time step per sweep pair
        num_steps: Number of time steps
        normalize: Normalize after each step
    
    Returns:
        Time-evolved MPS
        
    Example:
        >>> mps = MPS.random(L=20, d=2, chi=32, dtype=torch.complex128)
        >>> H = heisenberg_mpo(L=20)
        >>> psi_t = tdvp(mps, H, dt=0.1, num_steps=100)
    """
    for _ in range(num_steps):
        # Right sweep with dt/2
        mps = tdvp_sweep(mps, mpo, dt / 2, direction='right', normalize=False)
        # Left sweep with dt/2
        mps = tdvp_sweep(mps, mpo, dt / 2, direction='left', normalize=normalize)
    
    return mps


def tdvp2_sweep(
    mps: MPS,
    mpo: MPO,
    dt: float,
    chi_max: int = None,
    cutoff: float = 1e-12,
    direction: str = 'right',
) -> MPS:
    """
    Two-site TDVP sweep.
    
    Allows bond dimension to grow (like TEBD) while maintaining
    variational optimality.
    
    Args:
        mps: Input MPS
        mpo: Hamiltonian MPO
        dt: Time step
        chi_max: Maximum bond dimension (None = no limit)
        cutoff: SVD truncation cutoff
        direction: Sweep direction
    
    Returns:
        Time-evolved MPS
    """
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    L_envs = _build_left_environments(mps, mpo)
    R_envs = _build_right_environments(mps, mpo)
    
    if direction == 'right':
        sites = range(L - 1)
    else:
        sites = range(L - 2, -1, -1)
    
    for i in sites:
        # Two-site tensor
        A1 = mps.tensors[i]
        A2 = mps.tensors[i + 1]
        chi_l = A1.shape[0]
        d = A1.shape[1]
        chi_r = A2.shape[2]
        
        theta = torch.einsum('adb,bec->adec', A1, A2)
        theta_shape = theta.shape
        
        # Environments
        Le = L_envs[i]
        Re = R_envs[i + 2]
        W1 = mpo.tensors[i]
        W2 = mpo.tensors[i + 1]
        
        # Effective Hamiltonian for two sites
        def H_eff_2site(v):
            return _apply_H_eff_two_site(
                v.reshape(theta_shape), Le, W1, W2, Re
            ).reshape(-1)
        
        # Evolve forward using Lanczos expm
        # For real time: exp(-i H dt)
        # dt is already complex if needed
        theta_new = _expm_krylov(H_eff_2site, theta.reshape(-1), -1j * dt, num_iter=15)
        theta_new = theta_new.reshape(theta_shape)
        
        # SVD split
        theta_mat = theta_new.reshape(chi_l * d, d * chi_r)
        U, S, Vh = svd_truncated(theta_mat, max_rank=chi_max, cutoff=cutoff)
        chi_new = len(S)
        
        if direction == 'right':
            A1_new = U.reshape(chi_l, d, chi_new)
            A2_new = (torch.diag(S.to(Vh.dtype)) @ Vh).reshape(chi_new, d, chi_r)
            mps.tensors[i] = A1_new
            mps.tensors[i + 1] = A2_new
            L_envs[i + 1] = _contract_left_env(A1_new, W1, Le)
        else:
            A1_new = (U @ torch.diag(S.to(U.dtype))).reshape(chi_l, d, chi_new)
            A2_new = Vh.reshape(chi_new, d, chi_r)
            mps.tensors[i] = A1_new
            mps.tensors[i + 1] = A2_new
            R_envs[i + 1] = _contract_right_env(A2_new, W2, Re)
    
    return mps


def tdvp2(
    mps: MPS,
    mpo: MPO,
    dt: float,
    num_steps: int = 1,
    chi_max: int = None,
    cutoff: float = 1e-12,
) -> MPS:
    """
    Two-site TDVP for real-time evolution.
    
    Note: For strict energy conservation, use smaller dt.
    TDVP is variational but 2-site version has Trotter error.
    
    Args:
        mps: Initial MPS
        mpo: Hamiltonian MPO
        dt: Time step
        num_steps: Number of steps
        chi_max: Maximum bond dimension
        cutoff: SVD cutoff
    
    Returns:
        Time-evolved MPS
    """
    for _ in range(num_steps):
        mps = tdvp2_sweep(mps, mpo, dt / 2, chi_max, cutoff, direction='right')
        mps = tdvp2_sweep(mps, mpo, dt / 2, chi_max, cutoff, direction='left')
        mps.normalize_()
    
    return mps


def tdvp_ground_state(
    mps: MPS,
    mpo: MPO,
    chi_max: int = 64,
    num_sweeps: int = 20,
    dt: float = 0.1,
    tol: float = 1e-8,
) -> Tuple[MPS, float, Dict[str, Any]]:
    """
    Find ground state via imaginary-time TDVP.
    
    Uses 2-site TDVP with imaginary time evolution to find ground state.
    Unlike DMRG, this can sometimes escape local minima better.
    
    Args:
        mps: Initial MPS guess
        mpo: Hamiltonian MPO
        chi_max: Maximum bond dimension
        num_sweeps: Maximum number of sweeps
        dt: Imaginary time step (larger = faster but less stable)
        tol: Energy convergence tolerance
    
    Returns:
        (ground_state_mps, energy, info_dict)
    """
    energy_history = []
    
    for sweep in range(num_sweeps):
        # Imaginary time sweep: exp(-τH) where τ = dt (real)
        # This projects onto ground state
        mps = _imaginary_time_sweep(mps, mpo, dt, chi_max, direction='right')
        mps = _imaginary_time_sweep(mps, mpo, dt, chi_max, direction='left')
        mps.normalize_()
        
        # Compute energy
        energy = _compute_mpo_expectation(mps, mpo)
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


def _imaginary_time_sweep(
    mps: MPS,
    mpo: MPO,
    dt: float,
    chi_max: int,
    direction: str = 'right',
) -> MPS:
    """Single imaginary time evolution sweep."""
    L = mps.L
    
    L_envs = _build_left_environments(mps, mpo)
    R_envs = _build_right_environments(mps, mpo)
    
    if direction == 'right':
        sites = range(L - 1)
    else:
        sites = range(L - 2, -1, -1)
    
    for i in sites:
        A1 = mps.tensors[i]
        A2 = mps.tensors[i + 1]
        chi_l, d, _ = A1.shape
        chi_r = A2.shape[2]
        
        theta = torch.einsum('adb,bec->adec', A1, A2)
        theta_shape = theta.shape
        n = theta.numel()
        
        Le = L_envs[i]
        Re = R_envs[i + 2]
        W1 = mpo.tensors[i]
        W2 = mpo.tensors[i + 1]
        
        # For small systems, use direct power method instead of Krylov
        # Apply (1 - dt*H) repeatedly for stability
        theta_flat = theta.reshape(-1)
        
        def H_eff(v):
            return _apply_H_eff_two_site(v.reshape(theta_shape), Le, W1, W2, Re).reshape(-1)
        
        # Simple power iteration: θ' ≈ (1 - dt*H)^k θ for imaginary time
        # This is stable and projects to ground state
        for _ in range(3):
            Hv = H_eff(theta_flat)
            theta_flat = theta_flat - dt * Hv
            theta_flat = theta_flat / torch.norm(theta_flat)
        
        theta_new = theta_flat.reshape(theta_shape)
        
        # SVD split
        theta_mat = theta_new.reshape(chi_l * d, d * chi_r)
        U, S, Vh = svd_truncated(theta_mat, max_rank=chi_max, cutoff=1e-12)
        chi_new = len(S)
        
        if direction == 'right':
            mps.tensors[i] = U.reshape(chi_l, d, chi_new)
            mps.tensors[i + 1] = (torch.diag(S.to(Vh.dtype)) @ Vh).reshape(chi_new, d, chi_r)
            L_envs[i + 1] = _contract_left_env(mps.tensors[i], W1, Le)
        else:
            mps.tensors[i] = (U @ torch.diag(S.to(U.dtype))).reshape(chi_l, d, chi_new)
            mps.tensors[i + 1] = Vh.reshape(chi_new, d, chi_r)
            R_envs[i + 1] = _contract_right_env(mps.tensors[i + 1], W2, Re)
    
    return mps


def _compute_mpo_expectation(mps: MPS, mpo: MPO) -> float:
    """Compute <ψ|H|ψ>."""
    L_env = torch.ones(1, 1, 1, dtype=mps.dtype, device=mps.device)
    for i in range(mps.L):
        A = mps.tensors[i]
        W = mpo.tensors[i].to(mps.dtype)
        temp1 = torch.einsum('awb,adc->wdbc', L_env, A.conj())
        temp2 = torch.einsum('wdbc,wdfx->bfcx', temp1, W)
        L_env = torch.einsum('bfcx,bfe->cxe', temp2, A)
    return L_env[0, 0, 0].real.item()


# ============================================================================
# Helper functions
# ============================================================================

def _build_left_environments(mps: MPS, mpo: MPO) -> list:
    """Build all left environments."""
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    L_envs = [None] * (L + 1)
    L_envs[0] = torch.ones(1, 1, 1, dtype=dtype, device=device)
    
    for i in range(L):
        L_envs[i + 1] = _contract_left_env(mps.tensors[i], mpo.tensors[i], L_envs[i])
    
    return L_envs


def _build_right_environments(mps: MPS, mpo: MPO) -> list:
    """Build all right environments."""
    L = mps.L
    dtype = mps.dtype
    device = mps.device
    
    R_envs = [None] * (L + 1)
    R_envs[L] = torch.ones(1, 1, 1, dtype=dtype, device=device)
    
    for i in range(L - 1, -1, -1):
        R_envs[i] = _contract_right_env(mps.tensors[i], mpo.tensors[i], R_envs[i + 1])
    
    return R_envs


def _contract_left_env(A: Tensor, W: Tensor, L: Tensor) -> Tensor:
    """Contract left environment with one MPS/MPO site."""
    # L[a,w,b] A*[a,d,c] W[w,p,d,x] A[b,p,e] -> L'[c,x,e]
    W = W.to(A.dtype)
    tmp1 = torch.einsum('awb,adc->wdbc', L, A.conj())
    tmp2 = torch.einsum('wdbc,wpdx->pbcx', tmp1, W)
    L_new = torch.einsum('pbcx,bpe->cxe', tmp2, A)
    return L_new


def _contract_right_env(A: Tensor, W: Tensor, R: Tensor) -> Tensor:
    """Contract right environment with one MPS/MPO site."""
    # A*[a,d,c] W[w,p,d,x] A[b,p,e] R[c,x,e] -> R'[a,w,b]
    W = W.to(A.dtype)
    tmp1 = torch.einsum('adc,cxe->adxe', A.conj(), R)
    tmp2 = torch.einsum('adxe,wpdx->awpe', tmp1, W)
    R_new = torch.einsum('awpe,bpe->awb', tmp2, A)
    return R_new


def _apply_H_eff_one_site(A: Tensor, Le: Tensor, W: Tensor, Re: Tensor) -> Tensor:
    """Apply effective one-site Hamiltonian."""
    # Le[a,w,a'] A[a',d,b'] W[w,p,d,x] Re[b,x,b'] -> result[a,p,b]
    W = W.to(A.dtype)
    tmp1 = torch.einsum('awk,kdb->awdb', Le, A)
    tmp2 = torch.einsum('awdb,wpdx->apbx', tmp1, W)
    result = torch.einsum('apbx,bxc->apc', tmp2, Re)
    return result


def _apply_H_eff_zero_site(C: Tensor, Le: Tensor, Re: Tensor) -> Tensor:
    """Apply effective zero-site (bond) Hamiltonian."""
    # Le[a,w,a'] C[a',b'] Re[b,w,b'] -> result[a,b]
    # Note: zero-site H_eff contracts MPO indices together
    tmp = torch.einsum('awk,kl->awl', Le, C)
    result = torch.einsum('awl,bwl->ab', tmp, Re)
    return result


def _apply_H_eff_two_site(
    theta: Tensor, Le: Tensor, W1: Tensor, W2: Tensor, Re: Tensor
) -> Tensor:
    """Apply effective two-site Hamiltonian."""
    # theta[a,s,t,b], Le[a',w,a], W1[w,p,s,x], W2[x,q,t,y], Re[b',y,b]
    # -> result[a',p,q,b']
    W1, W2 = W1.to(theta.dtype), W2.to(theta.dtype)
    tmp1 = torch.einsum('awk,kstl->awstl', Le, theta)
    tmp2 = torch.einsum('awstl,wpsx->apxtl', tmp1, W1)
    tmp3 = torch.einsum('apxtl,xqty->apqyl', tmp2, W2)
    result = torch.einsum('apqyl,byl->apqb', tmp3, Re)
    return result


def _expm_krylov(
    matvec: Callable[[Tensor], Tensor],
    v: Tensor,
    dt: complex,
    num_iter: int = 20,
) -> Tensor:
    """
    Compute exp(dt * H) @ v using Krylov subspace method.
    
    Simplified version that's more robust for TDVP.
    """
    dtype = v.dtype
    device = v.device
    n = v.numel()
    
    v_flat = v.flatten()
    norm_v = torch.norm(v_flat)
    if norm_v < 1e-14:
        return v
    
    v_norm = v_flat / norm_v
    
    # Build Krylov basis and Hessenberg matrix
    num_iter = min(num_iter, n, 30)  # Cap iterations
    
    V = []
    V.append(v_norm)
    
    H = torch.zeros(num_iter + 1, num_iter, dtype=torch.complex128, device=device)
    
    for j in range(num_iter):
        w = matvec(V[j]).to(torch.complex128)
        
        # Arnoldi process
        for i in range(j + 1):
            H[i, j] = torch.dot(V[i].conj().to(torch.complex128), w)
            w = w - H[i, j] * V[i].to(torch.complex128)
        
        h_jp1_j = torch.norm(w)
        H[j + 1, j] = h_jp1_j
        
        if h_jp1_j > 1e-12 and j < num_iter - 1:
            V.append((w / h_jp1_j).to(dtype))
        else:
            num_iter = j + 1
            break
    
    # Compute exp(dt * H_m) where H_m is the m×m upper-left block
    H_m = H[:num_iter, :num_iter]
    exp_H = torch.linalg.matrix_exp(dt * H_m)
    
    # Result = norm_v * V_m @ exp_H @ e_1
    e1 = torch.zeros(num_iter, dtype=torch.complex128, device=device)
    e1[0] = 1.0
    
    y = exp_H @ e1
    
    # Combine: result = Σ_j y_j * V_j
    result = torch.zeros(n, dtype=torch.complex128, device=device)
    for j in range(min(len(V), num_iter)):
        result = result + y[j] * V[j].to(torch.complex128)
    
    result = result * norm_v
    
    # Return in original dtype
    return result.to(dtype)


def _expm_lanczos(
    matvec: Callable[[Tensor], Tensor],
    v: Tensor,
    dt: complex,
    num_iter: int = 20,
) -> Tensor:
    """
    Compute exp(dt * H) @ v using Lanczos approximation.
    
    This is the key operation for TDVP time evolution.
    """
    dtype = v.dtype
    device = v.device
    n = v.numel()
    
    # Handle real vs complex
    if not dtype.is_complex:
        # For imaginary time with real MPS, keep real
        if abs(dt.imag) < 1e-14:
            dt = dt.real
    
    v_flat = v.flatten()
    norm_v = torch.norm(v_flat)
    if norm_v < 1e-14:
        return v
    
    v_flat = v_flat / norm_v
    
    # Lanczos iteration
    num_iter = min(num_iter, n)
    V = torch.zeros(n, num_iter, dtype=dtype, device=device)
    alpha = torch.zeros(num_iter, dtype=dtype, device=device)
    beta = torch.zeros(num_iter - 1, dtype=dtype, device=device)
    
    V[:, 0] = v_flat
    
    for j in range(num_iter):
        w = matvec(V[:, j])
        
        alpha[j] = torch.dot(V[:, j].conj(), w).real
        
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]
        w = w - alpha[j] * V[:, j]
        
        # Reorthogonalize
        for i in range(j + 1):
            w = w - torch.dot(V[:, i].conj(), w) * V[:, i]
        
        if j < num_iter - 1:
            beta_j = torch.norm(w)
            if beta_j < 1e-14:
                num_iter = j + 1
                break
            beta[j] = beta_j.real
            V[:, j + 1] = w / beta_j
    
    # Build tridiagonal matrix
    T = torch.diag(alpha[:num_iter])
    if num_iter > 1:
        T += torch.diag(beta[:num_iter - 1], 1)
        T += torch.diag(beta[:num_iter - 1], -1)
    
    # Compute exp(dt * T)
    T = T.to(torch.complex128)
    exp_T = torch.linalg.matrix_exp(dt * T)
    
    # First column gives exp(dt*H)|v⟩ in Krylov basis
    e1 = torch.zeros(num_iter, dtype=torch.complex128, device=device)
    e1[0] = 1.0
    
    coeffs = exp_T @ e1
    
    # Transform back
    result = V[:, :num_iter].to(torch.complex128) @ coeffs
    result = result * norm_v
    
    # Cast back to original dtype if needed
    if not dtype.is_complex:
        result = result.real.to(dtype)
    else:
        result = result.to(dtype)
    
    return result.reshape(v.shape)
