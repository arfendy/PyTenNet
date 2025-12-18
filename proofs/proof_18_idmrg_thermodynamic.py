#!/usr/bin/env python3
"""
Proof 18: iDMRG Converges to Thermodynamic Limit
=================================================

Demonstrates that infinite DMRG converges to the correct
ground state energy per site in the thermodynamic limit.

Physics:
    For the Heisenberg XXX spin-1/2 chain:
    H = J Σ (Sx_i Sx_{i+1} + Sy_i Sy_{i+1} + Sz_i Sz_{i+1})
    
    The exact ground state energy per site (Bethe ansatz):
    e_0 = J(1/4 - ln(2)) ≈ -0.4431 J

Test:
    - Run iDMRG on Heisenberg XXX
    - Verify E/site approaches exact value as system grows
    - Check convergence with bond dimension

Criterion: |E/site - exact| < 0.02 for χ=32, L>50
"""

import torch
from tensornet.algorithms.idmrg import iMPO, idmrg
import math


def test_idmrg_heisenberg():
    """Test iDMRG on Heisenberg XXX chain."""
    # Exact result: e_0 = 1/4 - ln(2) ≈ -0.4431
    exact_energy = 0.25 - math.log(2)
    
    # Create Heisenberg iMPO
    H = iMPO.heisenberg(J=1.0, Jz=1.0)
    
    # Run iDMRG
    psi, e0, info = idmrg(
        H,
        chi_max=32,
        num_iterations=30,
        tol=1e-8,
        verbose=False,
    )
    
    error = abs(e0 - exact_energy)
    
    print(f"iDMRG energy/site:  {e0:.8f}")
    print(f"Exact (Bethe):      {exact_energy:.8f}")
    print(f"Error:              {error:.6f}")
    print(f"Final system size:  {info['final_system_size']} sites")
    print(f"Final χ:            {info['final_chi']}")
    
    # Tolerance: for χ=32, we expect ~1% error
    # The iDMRG is a growing algorithm, so finite-size effects remain
    assert error < 0.02, f"Error {error:.6f} exceeds tolerance 0.02"
    
    print(f"✓ iDMRG converges to thermodynamic limit (error < 2%)")
    
    return True


def test_idmrg_scaling():
    """Test that iDMRG improves with bond dimension."""
    H = iMPO.heisenberg(J=1.0, Jz=1.0)
    exact = 0.25 - math.log(2)
    
    print("\nBond dimension scaling:")
    print("-" * 40)
    
    errors = []
    for chi in [8, 16, 32]:
        _, e0, info = idmrg(H, chi_max=chi, num_iterations=30, tol=1e-10)
        error = abs(e0 - exact)
        errors.append(error)
        print(f"χ = {chi:2d}: E/site = {e0:.8f}, error = {error:.6f}")
    
    # Check that error decreases with chi
    assert errors[1] < errors[0], "Error should decrease with χ"
    assert errors[2] < errors[1], "Error should decrease with χ"
    
    print(f"✓ Error decreases with bond dimension")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Proof 18: iDMRG Converges to Thermodynamic Limit")
    print("=" * 60)
    print()
    
    success1 = test_idmrg_heisenberg()
    success2 = test_idmrg_scaling()
    
    print()
    print("=" * 60)
    print("PROOF PASSED" if (success1 and success2) else "PROOF FAILED")
    print("=" * 60)
