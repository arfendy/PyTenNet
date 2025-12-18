"""
iDMRG (Infinite Density Matrix Renormalization Group) algorithm.

Find ground states directly in the thermodynamic limit using
translation-invariant MPS (iMPS).
"""

from typing import Tuple, Optional, Dict, Any, Callable
import torch
from torch import Tensor

from tensornet.core.decompositions import svd_truncated
from tensornet.algorithms.lanczos import lanczos_ground_state


class iMPS:
    """
    Infinite Matrix Product State.
    
    Represents a translation-invariant state using a unit cell of tensors.
    For simplicity, we use a 2-site unit cell (A, B) with left/right
    canonical forms.
    
    Attributes:
        A: Left-canonical tensor (chi, d, chi)
        B: Right-canonical tensor (chi, d, chi)
        C: Center matrix (chi, chi) - contains singular values
        chi: Bond dimension
        d: Physical dimension
    """
    
    def __init__(
        self,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        d: int = 2,
    ):
        """
        Initialize iMPS.
        
        Args:
            A: Left-canonical tensor (chi, d, chi)
            B: Right-canonical tensor (chi, d, chi)
            C: Center matrix (chi, chi)
            d: Physical dimension
        """
        self.A = A
        self.B = B
        self.C = C
        self.d = d
        self.chi = A.shape[0]
        self.dtype = A.dtype
        self.device = A.device
    
    @classmethod
    def random(
        cls,
        chi: int,
        d: int = 2,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ) -> 'iMPS':
        """Create random iMPS with specified bond dimension."""
        if device is None:
            device = torch.device('cpu')
            
        # Random tensors
        A = torch.randn(chi, d, chi, dtype=dtype, device=device)
        B = torch.randn(chi, d, chi, dtype=dtype, device=device)
        
        # Orthonormalize A (left-canonical)
        A_mat = A.reshape(chi * d, chi)
        Q, R = torch.linalg.qr(A_mat)
        A = Q[:, :chi].reshape(chi, d, chi)
        
        # Orthonormalize B (right-canonical)
        B_mat = B.reshape(chi, d * chi)
        Q, R = torch.linalg.qr(B_mat.T)
        B = Q[:, :chi].T.reshape(chi, d, chi)
        
        # Random center matrix
        C = torch.eye(chi, dtype=dtype, device=device)
        
        return cls(A, B, C, d)
    
    def entanglement_entropy(self) -> float:
        """Compute entanglement entropy from singular values."""
        # C contains the Schmidt values (up to normalization)
        U, S, Vh = torch.linalg.svd(self.C)
        S = S / S.norm()  # Normalize
        S2 = S ** 2
        # Remove zeros for log
        S2 = S2[S2 > 1e-15]
        entropy = -torch.sum(S2 * torch.log(S2))
        return entropy.item()
    
    def correlation_length(self) -> float:
        """
        Estimate correlation length from transfer matrix spectrum.
        
        ξ = -1 / ln|λ_2 / λ_1|
        """
        # Build transfer matrix T[α,α'] = Σ_σ A[α,σ,β] A*[α',σ,β']
        # where β indices are contracted
        A = self.A
        chi = self.chi
        d = self.d
        
        # T[α,α',β,β'] = A[α,σ,β] A*[α',σ,β']
        T = torch.einsum('asb,csb->acbd', A, A.conj())
        T = T.reshape(chi * chi, chi * chi)
        
        # Get eigenvalues
        eigvals = torch.linalg.eigvals(T)
        eigvals_abs = eigvals.abs()
        
        # Sort by magnitude
        sorted_vals, _ = torch.sort(eigvals_abs, descending=True)
        
        if len(sorted_vals) >= 2 and sorted_vals[1] > 1e-10:
            xi = -1.0 / torch.log(sorted_vals[1] / sorted_vals[0]).item()
            return abs(xi)
        else:
            return float('inf')


class iMPO:
    """
    Infinite Matrix Product Operator.
    
    Translation-invariant Hamiltonian for infinite systems.
    Uses a single bulk tensor W.
    """
    
    def __init__(self, W: Tensor, d: int = 2):
        """
        Initialize iMPO.
        
        Args:
            W: Bulk MPO tensor (D, d, d, D) where D is MPO bond dimension
            d: Physical dimension
        """
        self.W = W
        self.d = d
        self.D = W.shape[0]
        self.dtype = W.dtype
        self.device = W.device
    
    @classmethod
    def heisenberg(
        cls,
        J: float = 1.0,
        Jz: float = 1.0,
        h: float = 0.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ) -> 'iMPO':
        """
        Create Heisenberg XXZ iMPO.
        
        H = J Σ (Sx_i Sx_{i+1} + Sy_i Sy_{i+1}) + Jz Σ Sz_i Sz_{i+1} + h Σ Sz_i
        
        MPO structure (D=5):
            W[0,:,:,0] = I (identity)
            W[0,:,:,1] = S+
            W[0,:,:,2] = S-
            W[0,:,:,3] = Sz
            W[0,:,:,4] = h*Sz
            W[1,:,:,4] = J/2 * S-
            W[2,:,:,4] = J/2 * S+
            W[3,:,:,4] = Jz * Sz
            W[4,:,:,4] = I
        """
        if device is None:
            device = torch.device('cpu')
        
        d = 2
        D = 5
        
        # Spin-1/2 operators
        I = torch.eye(2, dtype=dtype, device=device)
        Sz = torch.tensor([[0.5, 0], [0, -0.5]], dtype=dtype, device=device)
        Sp = torch.tensor([[0, 1], [0, 0]], dtype=dtype, device=device)
        Sm = torch.tensor([[0, 0], [1, 0]], dtype=dtype, device=device)
        
        W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
        
        # Row 0 (left boundary)
        W[0, :, :, 0] = I
        W[0, :, :, 1] = Sp
        W[0, :, :, 2] = Sm
        W[0, :, :, 3] = Sz
        W[0, :, :, 4] = h * Sz
        
        # Middle rows
        W[1, :, :, 4] = (J / 2) * Sm
        W[2, :, :, 4] = (J / 2) * Sp
        W[3, :, :, 4] = Jz * Sz
        
        # Row 4 (right boundary)
        W[4, :, :, 4] = I
        
        return cls(W, d)
    
    @classmethod
    def tfim(
        cls,
        J: float = 1.0,
        g: float = 1.0,
        dtype: torch.dtype = torch.float64,
        device: torch.device = None,
    ) -> 'iMPO':
        """
        Create Transverse-Field Ising Model iMPO.
        
        H = -J Σ Sz_i Sz_{i+1} - g Σ Sx_i
        
        MPO structure (D=3):
            W[0,:,:,0] = I
            W[0,:,:,1] = Sz
            W[0,:,:,2] = -g*Sx
            W[1,:,:,2] = -J*Sz
            W[2,:,:,2] = I
        """
        if device is None:
            device = torch.device('cpu')
        
        d = 2
        D = 3
        
        I = torch.eye(2, dtype=dtype, device=device)
        Sz = torch.tensor([[0.5, 0], [0, -0.5]], dtype=dtype, device=device)
        Sx = torch.tensor([[0, 0.5], [0.5, 0]], dtype=dtype, device=device)
        
        W = torch.zeros(D, d, d, D, dtype=dtype, device=device)
        
        W[0, :, :, 0] = I
        W[0, :, :, 1] = Sz
        W[0, :, :, 2] = -g * Sx
        W[1, :, :, 2] = -J * Sz
        W[2, :, :, 2] = I
        
        return cls(W, d)


def idmrg(
    mpo: iMPO,
    chi_max: int = 32,
    num_iterations: int = 100,
    tol: float = 1e-10,
    lanczos_iterations: int = 30,
    verbose: bool = False,
) -> Tuple[iMPS, float, Dict[str, Any]]:
    """
    Infinite DMRG algorithm (growing variant).
    
    Finds ground state energy per site in the thermodynamic limit
    by growing the chain 2 sites at a time from the center.
    
    This implements the "infinite-system" DMRG algorithm where we:
    1. Start with a small system
    2. Add two sites in the middle
    3. Optimize the two-site wavefunction
    4. Truncate via SVD
    5. Update environments
    6. Repeat until convergence
    
    Args:
        mpo: Infinite MPO (Hamiltonian)
        chi_max: Maximum bond dimension
        num_iterations: Maximum number of iDMRG iterations
        tol: Convergence tolerance for energy per site
        lanczos_iterations: Lanczos iterations for eigensolver
        verbose: Print progress
    
    Returns:
        (imps, energy_per_site, info_dict)
        
    Example:
        >>> H = iMPO.heisenberg(J=1.0, Jz=1.0)
        >>> psi, e0, info = idmrg(H, chi_max=64)
        >>> print(f"E/site = {e0}")
    """
    dtype = mpo.dtype
    device = mpo.device
    d = mpo.d
    D = mpo.D
    W = mpo.W
    
    # Initialize with random two-site state
    chi = 1
    
    # Random initial wavefunction
    theta = torch.randn(chi, d, d, chi, dtype=dtype, device=device)
    theta = theta / theta.norm()
    
    # Left environment starts at MPO left boundary (row 0)
    L_env = torch.zeros(chi, D, chi, dtype=dtype, device=device)
    L_env[0, 0, 0] = 1.0
    
    # Right environment starts at MPO right boundary (col D-1)
    R_env = torch.zeros(chi, D, chi, dtype=dtype, device=device)
    R_env[0, D-1, 0] = 1.0
    
    energy_history = []
    A = None
    B = None
    
    for iteration in range(num_iterations):
        chi_L = L_env.shape[0]
        chi_R = R_env.shape[2]
        
        # Current theta shape
        theta_shape = (chi_L, d, d, chi_R)
        
        # Make sure theta has right shape
        if theta.shape != theta_shape:
            theta_new = torch.randn(theta_shape, dtype=dtype, device=device)
            theta_new = theta_new / theta_new.norm()
            theta = theta_new
        
        # Lanczos for ground state of H_eff
        def apply_Heff(v: Tensor) -> Tensor:
            """Apply effective two-site Hamiltonian."""
            v_2site = v.reshape(theta_shape)
            
            # Contract: L[a,w,a'] v[a',s,t,b'] W[w,p,s,x] W[x,q,t,y] R[b,y,b']
            # Result: [a, p, q, b]
            
            # L @ v: [a,w,a'] @ [a',s,t,b'] -> [a,w,s,t,b']
            tmp1 = torch.einsum('awk,kstl->awstl', L_env, v_2site)
            
            # @ W (site 1): [a,w,s,t,b'] @ [w,p,s,x] -> [a,p,x,t,b']
            tmp2 = torch.einsum('awstl,wpsx->apxtl', tmp1, W)
            
            # @ W (site 2): [a,p,x,t,b'] @ [x,q,t,y] -> [a,p,q,y,b']
            tmp3 = torch.einsum('apxtl,xqty->apqyl', tmp2, W)
            
            # @ R: [a,p,q,y,b'] @ [b,y,b'] -> [a,p,q,b]
            result = torch.einsum('apqyl,byl->apqb', tmp3, R_env)
            
            return result.reshape(-1)
        
        theta_flat = theta.reshape(-1)
        E_total, theta_opt = lanczos_ground_state(
            apply_Heff,
            theta_flat,
            num_iterations=lanczos_iterations,
        )
        
        # System has 2*(iteration+1) sites total
        num_sites = 2 * (iteration + 1)
        energy_per_site = E_total / num_sites
        energy_history.append(energy_per_site)
        
        if verbose and iteration % 5 == 0:
            print(f"Iter {iteration:3d}: L={num_sites:3d}, E/site = {energy_per_site:.10f}, χ = {chi_L}")
        
        # SVD split: theta[a, s, t, b] -> A[a, s, c] @ diag(S) @ B[c, t, b]
        theta_opt = theta_opt.reshape(theta_shape)
        theta_mat = theta_opt.reshape(chi_L * d, d * chi_R)
        
        U, S, Vh = svd_truncated(theta_mat, max_rank=chi_max, cutoff=1e-12)
        chi_new = len(S)
        
        # Normalize singular values
        S = S / S.norm()
        
        A = U.reshape(chi_L, d, chi_new)
        B = Vh.reshape(chi_new, d, chi_R)
        C = torch.diag(S)
        
        # Update left environment: L' = L @ A* @ W @ A
        # L'[c, x, c'] = L[a,w,a'] A*[a,s,c] W[w,p,s,x] A[a',p,c']
        L_env_new = _contract_left_env_inf(A, W, L_env)
        
        # Update right environment: R' = B* @ W @ B @ R
        # R'[c, x, c'] = B*[c,t,b] W[x,q,t,y] B[c',q,b'] R[b,y,b']
        R_env_new = _contract_right_env_inf(B, W, R_env)
        
        L_env = L_env_new
        R_env = R_env_new
        
        # Initial guess for next iteration: A @ sqrt(S) @ sqrt(S) @ B
        sqrt_S = torch.sqrt(S)
        A_center = torch.einsum('asc,c->asc', A, sqrt_S)
        B_center = torch.einsum('c,ctb->ctb', sqrt_S, B)
        theta = torch.einsum('asc,ctb->astb', A_center, B_center)
        
        # Check convergence
        if len(energy_history) > 10:
            recent = energy_history[-5:]
            if max(recent) - min(recent) < tol:
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break
    
    # Final iMPS
    if A is not None and B is not None:
        imps = iMPS(A, B, C, d)
    else:
        # Fallback
        imps = iMPS.random(chi=chi_new, d=d, dtype=dtype, device=device)
    
    info = {
        'num_iterations': iteration + 1,
        'converged': len(energy_history) > 10 and (max(energy_history[-5:]) - min(energy_history[-5:]) < tol),
        'energy_history': energy_history,
        'final_chi': chi_new if 'chi_new' in dir() else 1,
        'final_system_size': 2 * (iteration + 1),
    }
    
    return imps, energy_per_site, info


def _contract_left_env_inf(A: Tensor, W: Tensor, L: Tensor) -> Tensor:
    """
    Contract left environment with one site.
    
    L[a,w,b] A*[a,d,c] W[w,d',d,x] A[b,d',e] -> L'[c,x,e]
    """
    chi = A.shape[0]
    D = W.shape[0]
    
    # L[a,w,b] A*[a,d,c] -> tmp1[w,d,b,c]
    tmp1 = torch.einsum('awb,adc->wdbc', L, A.conj())
    
    # tmp1[w,d,b,c] W[w,f,d,x] -> tmp2[f,b,c,x]
    tmp2 = torch.einsum('wdbc,wfdx->fbcx', tmp1, W)
    
    # tmp2[f,b,c,x] A[b,f,e] -> L'[c,x,e]
    L_new = torch.einsum('fbcx,bfe->cxe', tmp2, A)
    
    return L_new


def _contract_right_env_inf(B: Tensor, W: Tensor, R: Tensor) -> Tensor:
    """
    Contract right environment with one site.
    
    R[c,x,e] B*[a,d,c] W[w,d',d,x] B[b,d',e] -> R'[a,w,b]
    """
    # R[c,x,e] B*[a,d,c] -> tmp1[x,e,a,d]
    tmp1 = torch.einsum('cxe,adc->xead', R, B.conj())
    
    # tmp1[x,e,a,d] W[w,f,d,x] -> tmp2[e,a,w,f]
    tmp2 = torch.einsum('xead,wfdx->eawf', tmp1, W)
    
    # tmp2[e,a,w,f] B[b,f,e] -> R'[a,w,b]
    R_new = torch.einsum('eawf,bfe->awb', tmp2, B)
    
    return R_new


def idmrg_energy_density(
    mpo: iMPO,
    chi_max: int = 32,
    num_iterations: int = 100,
    tol: float = 1e-10,
) -> float:
    """
    Convenience function to get just the ground state energy per site.
    
    Args:
        mpo: Infinite MPO
        chi_max: Maximum bond dimension
        num_iterations: Max iterations
        tol: Convergence tolerance
        
    Returns:
        Ground state energy per site
    """
    _, e0, _ = idmrg(mpo, chi_max=chi_max, num_iterations=num_iterations, tol=tol)
    return e0
