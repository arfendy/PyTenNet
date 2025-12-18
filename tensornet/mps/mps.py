"""
Matrix Product State (MPS) implementation.

An MPS represents a tensor of shape (d_1, d_2, ..., d_L) as a product of matrices:
    T[s1, s2, ..., sL] = A1[s1] @ A2[s2] @ ... @ AL[sL]
    
where each A_i[s_i] is a matrix of shape (chi_{i-1}, chi_i).
"""

from __future__ import annotations
from typing import List, Optional, Literal, Tuple
import torch
from torch import Tensor
import math

from tensornet.core.decompositions import svd_truncated, qr_stable


class MPS:
    """
    Matrix Product State representation.
    
    Stores a quantum state or tensor as a chain of tensors.
    Each tensor has shape (chi_left, d, chi_right) where:
        - chi_left: left bond dimension
        - d: physical dimension
        - chi_right: right bond dimension
        
    For boundary tensors:
        - First tensor: shape (1, d, chi)
        - Last tensor: shape (chi, d, 1)
    
    Attributes:
        tensors: List of MPS tensors
        L: Number of sites
        d: Physical dimension (assumed uniform)
        chi: Maximum bond dimension
        
    Example:
        >>> mps = MPS.random(L=10, d=2, chi=16)
        >>> print(mps.L, mps.chi)
        10 16
        >>> tensor = mps.to_tensor()
        >>> print(tensor.shape)
        torch.Size([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """
    
    def __init__(
        self,
        tensors: List[Tensor],
        canonical_form: Optional[str] = None,
        center: Optional[int] = None,
    ):
        """
        Initialize MPS from list of tensors.
        
        Args:
            tensors: List of tensors with shapes (chi_l, d, chi_r)
            canonical_form: 'left', 'right', 'mixed', or None
            center: Canonical center site (for mixed canonical form)
        """
        self.tensors = tensors
        self.L = len(tensors)
        self._canonical_form = canonical_form
        self._center = center
        
        # Infer physical dimension (assumed uniform)
        self.d = tensors[0].shape[1]
        
        # Compute maximum bond dimension
        self.chi = max(t.shape[2] for t in tensors[:-1]) if len(tensors) > 1 else 1
        
        # Dtype and device from first tensor
        self.dtype = tensors[0].dtype
        self.device = tensors[0].device
    
    @classmethod
    def random(
        cls,
        L: int,
        d: int = 2,
        chi: int = 8,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
        normalize: bool = True,
    ) -> MPS:
        """
        Create random MPS.
        
        Args:
            L: Number of sites
            d: Physical dimension
            chi: Bond dimension
            dtype: Data type
            device: Device
            normalize: Whether to normalize the state
            
        Returns:
            Random MPS
        """
        if device is None:
            device = torch.device('cpu')
        
        # Compute consistent bond dimensions
        # chi[i] is the bond dimension between site i and i+1
        bond_dims = []
        for i in range(L - 1):
            # Maximum from left: d^(i+1)
            max_left = d ** (i + 1)
            # Maximum from right: d^(L-i-1)
            max_right = d ** (L - i - 1)
            bond_dims.append(min(chi, max_left, max_right))
        
        tensors = []
        for i in range(L):
            chi_l = 1 if i == 0 else bond_dims[i - 1]
            chi_r = 1 if i == L - 1 else bond_dims[i]
            
            tensor = torch.randn(chi_l, d, chi_r, dtype=dtype, device=device)
            tensors.append(tensor)
        
        mps = cls(tensors)
        
        if normalize:
            mps.normalize_()
            
        return mps
    
    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        chi_max: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> MPS:
        """
        Convert dense tensor to MPS via successive SVDs.
        
        Args:
            tensor: Dense tensor of shape (d_1, d_2, ..., d_L)
            chi_max: Maximum bond dimension
            cutoff: Singular value cutoff
            
        Returns:
            MPS approximation of tensor
            
        Example:
            >>> tensor = torch.randn(2, 2, 2, 2)
            >>> mps = MPS.from_tensor(tensor, chi_max=4)
            >>> error = torch.norm(tensor - mps.to_tensor())
        """
        if tensor.ndim < 2:
            raise ValueError(f"Need at least 2D tensor, got {tensor.ndim}D")
        
        L = tensor.ndim
        dtype = tensor.dtype
        device = tensor.device
        
        tensors = []
        
        # Start with tensor reshaped as (d_0, d_1 * d_2 * ... * d_{L-1})
        remaining = tensor.reshape(tensor.shape[0], -1)
        chi_l = 1
        
        for i in range(L - 1):
            d_i = tensor.shape[i]
            
            # remaining has shape (chi_l * d_i, rest) after first iteration
            # For first iteration, it's (d_0, rest)
            
            # SVD: (chi_l * d_i, rest) = U @ S @ Vh
            U, S, Vh = svd_truncated(remaining, max_rank=chi_max, cutoff=cutoff)
            
            chi_r = len(S)
            
            # Store tensor: reshape U from (chi_l * d_i, chi_r) to (chi_l, d_i, chi_r)
            A = U.reshape(chi_l, d_i, chi_r)
            tensors.append(A)
            
            # Continue with S @ Vh: shape (chi_r, rest)
            remaining = torch.diag(S) @ Vh
            
            # Reshape for next iteration: (chi_r, d_{i+1} * ...) -> (chi_r * d_{i+1}, ...)
            if i < L - 2:
                d_next = tensor.shape[i + 1]
                rest = remaining.shape[1] // d_next
                remaining = remaining.reshape(chi_r * d_next, rest)
                chi_l = chi_r
            else:
                # For the last SVD, remaining becomes the last tensor
                chi_l = chi_r
        
        # Last tensor: remaining has shape (chi_l, d_{L-1})
        d_last = tensor.shape[-1]
        tensors.append(remaining.reshape(chi_l, d_last, 1))
        
        return cls(tensors)
    
    def to_tensor(self) -> Tensor:
        """
        Contract MPS to dense tensor.
        
        Returns:
            Dense tensor of shape (d, d, ..., d)
            
        Warning:
            Exponential in L - only use for small systems!
        """
        result = self.tensors[0]  # (1, d, chi)
        
        for i in range(1, self.L):
            # result: (..., chi)
            # tensors[i]: (chi, d, chi')
            result = torch.einsum('...i,ijk->...jk', result, self.tensors[i])
        
        # Remove boundary dimensions
        shape = [self.d] * self.L
        return result.reshape(shape)
    
    def norm(self) -> Tensor:
        """
        Compute norm of MPS without full contraction.
        
        Returns:
            ||psi||
        """
        return torch.sqrt(self.inner(self).real)
    
    def normalize_(self) -> MPS:
        """Normalize MPS in-place."""
        n = self.norm()
        if n > 0:
            # Divide one tensor by norm
            self.tensors[0] = self.tensors[0] / n
        return self
    
    def inner(self, other: MPS) -> Tensor:
        """
        Compute inner product <self|other>.
        
        Args:
            other: Another MPS
            
        Returns:
            Scalar inner product
        """
        if self.L != other.L:
            raise ValueError(f"MPS lengths don't match: {self.L} vs {other.L}")
        
        # Contract from left
        # Each tensor has shape (chi_l, d, chi_r)
        # E[a,b] = sum_s A*[a,s,c] B[b,s,c'] -> E'[c,c']
        
        A = self.tensors[0]   # (1, d, chi)
        B = other.tensors[0]  # (1, d, chi')
        
        # Sum over left bond (trivial) and physical index
        # A*: (1, d, chi), B: (1, d, chi') -> E: (chi, chi')
        E = torch.einsum('asc,asd->cd', A.conj(), B)
        
        for i in range(1, self.L):
            # E: (chi_l, chi_l')
            # A*: (chi_l, d, chi_r)
            # B: (chi_l', d, chi_r')
            E = torch.einsum(
                'ab,asc,bsd->cd',
                E,
                self.tensors[i].conj(),
                other.tensors[i]
            )
        
        return E.squeeze()
    
    def canonicalize(
        self,
        form: Literal['left', 'right', 'mixed'] = 'right',
        center: Optional[int] = None,
    ) -> MPS:
        """
        Transform to canonical form.
        
        Args:
            form: 'left', 'right', or 'mixed'
            center: Orthogonality center for mixed form
            
        Returns:
            self (modified in-place)
        """
        if form == 'left':
            self._canonicalize_left()
        elif form == 'right':
            self._canonicalize_right()
        elif form == 'mixed':
            if center is None:
                center = self.L // 2
            self._canonicalize_mixed(center)
        else:
            raise ValueError(f"Unknown form: {form}")
        
        self._canonical_form = form
        self._center = center
        return self
    
    def _canonicalize_left(self):
        """Left-canonicalize: sweep left to right with QR."""
        for i in range(self.L - 1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            
            # Reshape to (chi_l * d, chi_r)
            mat = A.reshape(chi_l * d, chi_r)
            
            # QR decomposition
            Q, R = qr_stable(mat)
            
            # Update tensors
            chi_new = Q.shape[1]
            self.tensors[i] = Q.reshape(chi_l, d, chi_new)
            self.tensors[i + 1] = torch.einsum('ij,jkl->ikl', R, self.tensors[i + 1])
    
    def _canonicalize_right(self):
        """Right-canonicalize: sweep right to left with QR."""
        for i in range(self.L - 1, 0, -1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            
            # Reshape to (chi_l, d * chi_r)
            mat = A.reshape(chi_l, d * chi_r)
            
            # QR on transpose
            Q, R = qr_stable(mat.T)
            
            # Q is (d * chi_r, chi_l), R is (chi_l, chi_l)
            chi_new = Q.shape[1]
            self.tensors[i] = Q.T.reshape(chi_new, d, chi_r)
            self.tensors[i - 1] = torch.einsum('ijk,kl->ijl', self.tensors[i - 1], R.T)
    
    def _canonicalize_mixed(self, center: int):
        """Mixed canonical form with center site."""
        # Left-canonicalize up to center
        for i in range(center):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            mat = A.reshape(chi_l * d, chi_r)
            Q, R = qr_stable(mat)
            chi_new = Q.shape[1]
            self.tensors[i] = Q.reshape(chi_l, d, chi_new)
            self.tensors[i + 1] = torch.einsum('ij,jkl->ikl', R, self.tensors[i + 1])
        
        # Right-canonicalize from end to center
        for i in range(self.L - 1, center, -1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            mat = A.reshape(chi_l, d * chi_r)
            Q, R = qr_stable(mat.T)
            chi_new = Q.shape[1]
            self.tensors[i] = Q.T.reshape(chi_new, d, chi_r)
            self.tensors[i - 1] = torch.einsum('ijk,kl->ijl', self.tensors[i - 1], R.T)
    
    def compress(
        self,
        chi_max: Optional[int] = None,
        cutoff: float = 0.0,
    ) -> MPS:
        """
        Compress MPS to smaller bond dimension.
        
        Args:
            chi_max: Maximum bond dimension
            cutoff: Singular value cutoff
            
        Returns:
            self (modified in-place)
        """
        # Right canonicalize first
        self.canonicalize('right')
        
        # Sweep left with SVD truncation
        for i in range(self.L - 1):
            A = self.tensors[i]
            chi_l, d, chi_r = A.shape
            
            mat = A.reshape(chi_l * d, chi_r)
            U, S, Vh = svd_truncated(mat, max_rank=chi_max, cutoff=cutoff)
            
            chi_new = len(S)
            self.tensors[i] = U.reshape(chi_l, d, chi_new)
            self.tensors[i + 1] = torch.einsum(
                'i,ij,jkl->ikl',
                S, Vh, self.tensors[i + 1]
            )
        
        # Update chi
        self.chi = max(t.shape[2] for t in self.tensors[:-1]) if self.L > 1 else 1
        
        return self
    
    def entanglement_entropy(self, bond: int) -> Tensor:
        """
        Compute von Neumann entanglement entropy at a bond.
        
        Args:
            bond: Bond index (0 to L-2)
            
        Returns:
            S = -sum_i lambda_i^2 log(lambda_i^2)
        """
        if bond < 0 or bond >= self.L - 1:
            raise ValueError(f"Bond {bond} out of range [0, {self.L - 2}]")
        
        # Canonicalize to mixed form at this bond
        self.canonicalize('mixed', center=bond)
        
        # Get singular values at the bond
        A = self.tensors[bond]
        chi_l, d, chi_r = A.shape
        mat = A.reshape(chi_l * d, chi_r)
        
        _, S, _ = torch.linalg.svd(mat, full_matrices=False)
        
        # Normalize singular values
        S = S / torch.norm(S)
        
        # von Neumann entropy
        S2 = S ** 2
        S2 = S2[S2 > 1e-14]  # Remove zeros
        entropy = -torch.sum(S2 * torch.log(S2))
        
        return entropy
    

    def entropy(self, bond: int) -> Tensor:
        """Alias for entanglement_entropy."""
        return self.entanglement_entropy(bond)

    def expectation_local(self, op: Tensor, site: int) -> Tensor:
        """
        Compute expectation value of local operator.
        
        Args:
            op: Local operator of shape (d, d)
            site: Site index
            
        Returns:
            <psi|O|psi>
        """
        # Contract everything except site
        # For efficiency, use mixed canonical form centered at site
        self.canonicalize('mixed', center=site)
        
        # At the center, we just need <A|O|A>
        A = self.tensors[site]  # (chi_l, d, chi_r)
        
        # O|A>
        OA = torch.einsum('ij,kjl->kil', op, A)
        
        # <A|O|A>
        result = torch.einsum('ijk,ijk->', A.conj(), OA)
        
        return result
    
    def bond_dimensions(self) -> List[int]:
        """Return list of bond dimensions."""
        return [t.shape[2] for t in self.tensors]
    
    def copy(self) -> MPS:
        """Create a deep copy."""
        return MPS(
            [t.clone() for t in self.tensors],
            canonical_form=self._canonical_form,
            center=self._center,
        )
    
    def __repr__(self) -> str:
        return (
            f"MPS(L={self.L}, d={self.d}, chi={self.chi}, "
            f"dtype={self.dtype}, device={self.device})"
        )




