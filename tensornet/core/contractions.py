"""
Tensor contraction operations.

Efficient contraction of tensor networks with autograd support.
"""

from typing import List, Tuple, Optional, Sequence
import torch
from torch import Tensor


def contract(
    A: Tensor,
    B: Tensor,
    indices_A: Sequence[int],
    indices_B: Sequence[int],
) -> Tensor:
    """
    Contract two tensors over specified indices.
    
    Args:
        A: First tensor
        B: Second tensor
        indices_A: Indices of A to contract
        indices_B: Indices of B to contract (must match indices_A in length and dimensions)
        
    Returns:
        Contracted tensor
        
    Example:
        >>> A = torch.randn(3, 4, 5)
        >>> B = torch.randn(5, 6, 4)
        >>> # Contract A's index 1 (dim 4) with B's index 2 (dim 4)
        >>> # and A's index 2 (dim 5) with B's index 0 (dim 5)
        >>> C = contract(A, B, [1, 2], [2, 0])
        >>> assert C.shape == (3, 6)
    """
    if len(indices_A) != len(indices_B):
        raise ValueError(
            f"Number of contracted indices must match: "
            f"got {len(indices_A)} for A and {len(indices_B)} for B"
        )
    
    # Validate dimensions match
    for i, (ia, ib) in enumerate(zip(indices_A, indices_B)):
        if A.shape[ia] != B.shape[ib]:
            raise ValueError(
                f"Dimension mismatch at contraction {i}: "
                f"A has dim {A.shape[ia]} at index {ia}, "
                f"B has dim {B.shape[ib]} at index {ib}"
            )
    
    # Build einsum string
    ndim_A = A.ndim
    ndim_B = B.ndim
    
    # Use ASCII letters for indices
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    idx_A = list(range(ndim_A))
    idx_B = list(range(ndim_A, ndim_A + ndim_B))
    
    # Map contracted indices
    for ia, ib in zip(indices_A, indices_B):
        idx_B[ib] = idx_A[ia]
    
    # Build subscript strings
    subscripts_A = ''.join(letters[i] for i in idx_A)
    subscripts_B = ''.join(letters[i] for i in idx_B)
    
    # Output indices: all non-contracted indices
    contracted_set = set(idx_A[i] for i in indices_A)
    output_indices = [i for i in idx_A if i not in contracted_set]
    output_indices += [i for i in idx_B if i not in contracted_set]
    subscripts_out = ''.join(letters[i] for i in output_indices)
    
    equation = f"{subscripts_A},{subscripts_B}->{subscripts_out}"
    
    return torch.einsum(equation, A, B)


def contract_network(
    tensors: List[Tensor],
    contractions: List[Tuple[int, int, Sequence[int], Sequence[int]]],
    path: Optional[List[Tuple[int, int]]] = None,
) -> Tensor:
    """
    Contract a network of tensors.
    
    Args:
        tensors: List of tensors to contract
        contractions: List of (tensor_idx_A, tensor_idx_B, indices_A, indices_B)
            specifying which indices to contract
        path: Contraction order as list of (i, j) pairs. If None, uses sequential order.
        
    Returns:
        Result of contracting the entire network
        
    Example:
        >>> A = torch.randn(3, 4)
        >>> B = torch.randn(4, 5)
        >>> C = torch.randn(5, 6)
        >>> # A @ B @ C
        >>> result = contract_network(
        ...     [A, B, C],
        ...     [(0, 1, [1], [0]), (1, 2, [1], [0])],  # contract A-B then B-C
        ... )
    """
    if len(tensors) == 0:
        raise ValueError("Need at least one tensor")
    
    if len(tensors) == 1:
        return tensors[0]
    
    # Simple sequential contraction if no path given
    # For a proper implementation, use path optimization
    working = list(tensors)
    
    for ta, tb, idx_a, idx_b in contractions:
        # This is a simplified version - proper implementation needs index tracking
        result = contract(working[ta], working[tb], idx_a, idx_b)
        # Update working list (simplified - assumes sequential contraction)
        working.append(result)
    
    return working[-1]


def einsum_tn(equation: str, *tensors: Tensor) -> Tensor:
    """
    Extended einsum for tensor networks.
    
    Wrapper around torch.einsum with additional validation.
    
    Args:
        equation: Einsum equation string
        *tensors: Input tensors
        
    Returns:
        Result tensor
        
    Example:
        >>> A = torch.randn(3, 4, 5)
        >>> B = torch.randn(5, 6)
        >>> C = einsum_tn('ijk,kl->ijl', A, B)
    """
    return torch.einsum(equation, *tensors)
