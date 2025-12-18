"""
TensorNet: Native Tensor Networks for PyTorch
==============================================

A production-quality tensor network library providing:
- Matrix Product States (MPS)
- Matrix Product Operators (MPO)
- DMRG optimization
- TEBD time evolution
- Full autograd support
"""

__version__ = "0.2.0"

from tensornet.core.decompositions import (
    svd_truncated,
    qr_stable,
    polar_decompose,
    eigh_truncated,
)

from tensornet.core.contractions import (
    contract,
    contract_network,
)

from tensornet.mps.mps import MPS
from tensornet.mps.mpo import MPO
from tensornet.mps.hamiltonians import (
    heisenberg_mpo,
    tfim_mpo,
    xx_mpo,
    xyz_mpo,
    bose_hubbard_mpo,
)
from tensornet.mps.states import (
    ghz_mps,
    product_mps,
    random_mps,
)

from tensornet.algorithms.dmrg import dmrg, dmrg_two_site
from tensornet.algorithms.tebd import tebd, heisenberg_gates, tfim_gates, time_evolve, imaginary_time_evolution
from tensornet.algorithms.lanczos import lanczos_ground_state

__all__ = [
    # Version
    "__version__",
    # Decompositions
    "svd_truncated",
    "qr_stable", 
    "polar_decompose",
    "eigh_truncated",
    # Contractions
    "contract",
    "contract_network",
    # MPS/MPO
    "MPS",
    "MPO",
    # Hamiltonians
    "heisenberg_mpo",
    "tfim_mpo",
    "xx_mpo",
    # States
    "ghz_mps",
    "product_mps",
    "random_mps",
    # Algorithms
    "dmrg",
    "dmrg_two_site",
    "tebd",
    "heisenberg_gates",
    "tfim_gates",
    "time_evolve",
    "imaginary_time_evolution",
    "lanczos_ground_state",
]
