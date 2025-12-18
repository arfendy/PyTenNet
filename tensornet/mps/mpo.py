"""
Matrix Product Operator (MPO) implementation.

An MPO represents an operator acting on a tensor product space.
"""

from __future__ import annotations
from typing import List, Optional
import torch
from torch import Tensor


class MPO:
    """
    Matrix Product Operator.
    
    Each tensor has shape (D_left, d_out, d_in, D_right) where:
        - D_left: left bond dimension
        - d_out: output physical dimension (bra)
        - d_in: input physical dimension (ket)
        - D_right: right bond dimension
        
    For boundary tensors:
        - First tensor: shape (1, d, d, D)
        - Last tensor: shape (D, d, d, 1)
    
    Attributes:
        tensors: List of MPO tensors
        L: Number of sites
        d: Physical dimension
        D: Maximum bond dimension
    """
    
    def __init__(self, tensors: List[Tensor]):
        """
        Initialize MPO from list of tensors.
        
        Args:
            tensors: List of tensors with shapes (D_l, d_out, d_in, D_r)
        """
        self.tensors = tensors
        self.L = len(tensors)
        self.d = tensors[0].shape[1]
        self.D = max(t.shape[3] for t in tensors[:-1]) if len(tensors) > 1 else 1
        self.dtype = tensors[0].dtype
        self.device = tensors[0].device
    
    @classmethod
    def identity(
        cls,
        L: int,
        d: int = 2,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> MPO:
        """Create identity MPO."""
        if device is None:
            device = torch.device('cpu')
        
        I = torch.eye(d, dtype=dtype, device=device)
        tensors = []
        
        for i in range(L):
            # Identity: delta_{s,s'}
            W = I.reshape(1, d, d, 1)
            tensors.append(W)
        
        return cls(tensors)
    
    def apply(self, mps) -> 'MPS':
        """
        Apply MPO to MPS: |psi'> = H|psi>.
        
        Args:
            mps: Input MPS
            
        Returns:
            Output MPS (may have larger bond dimension)
        """
        from tensornet.mps.mps import MPS
        
        if self.L != mps.L:
            raise ValueError(f"Length mismatch: MPO has {self.L}, MPS has {mps.L}")
        
        new_tensors = []
        
        for i in range(self.L):
            W = self.tensors[i]  # (D_l, d_out, d_in, D_r)
            A = mps.tensors[i]  # (chi_l, d_in, chi_r)
            
            # Contract over d_in
            # Result: (D_l, d_out, D_r, chi_l, chi_r)
            B = torch.einsum('Dojd,ijd->Doijd', W, A)
            
            # Reshape to (D_l * chi_l, d_out, D_r * chi_r)
            D_l, d_out, _, D_r = W.shape
            chi_l, _, chi_r = A.shape
            
            B = B.permute(0, 3, 1, 2, 4)  # (D_l, chi_l, d_out, D_r, chi_r)
            B = B.reshape(D_l * chi_l, d_out, D_r * chi_r)
            
            new_tensors.append(B)
        
        return MPS(new_tensors)
    
    def to_matrix(self) -> Tensor:
        """
        Contract MPO to dense matrix.
        
        Returns:
            Matrix of shape (d^L, d^L)
        """
        result = self.tensors[0]  # (1, d, d, D)
        
        for i in range(1, self.L):
            # Contract bond index
            result = torch.einsum('...D,Dijd->...ijd', result, self.tensors[i])
        
        # result shape: (1, d, d, d, d, ..., d, d, 1)
        # Reshape to (d^L, d^L)
        dim = self.d ** self.L
        
        # Rearrange: all d_out indices then all d_in indices
        result = result.squeeze(0).squeeze(-1)  # Remove boundary dims
        
        # result has shape (d, d, d, d, ...) alternating out/in
        # Need to separate out and in indices
        out_shape = [self.d] * self.L
        in_shape = [self.d] * self.L
        
        # Transpose to group output indices together, then input indices
        perm = list(range(0, 2 * self.L, 2)) + list(range(1, 2 * self.L, 2))
        result = result.permute(perm)
        
        return result.reshape(dim, dim)
    
    def __repr__(self) -> str:
        return f"MPO(L={self.L}, d={self.d}, D={self.D})"
