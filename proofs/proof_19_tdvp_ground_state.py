#!/usr/bin/env python3
"""
Proof 19: TDVP Finds Ground State via Imaginary Time
=====================================================

Demonstrates that Time-Dependent Variational Principle (TDVP)
can find ground states through imaginary time evolution.

Physics:
    Imaginary time evolution: |ψ(τ)⟩ = e^{-Hτ} |ψ(0)⟩
    
    As τ → ∞, the state projects onto the ground state:
    |ψ(τ)⟩ → e^{-E₀τ} |ψ₀⟩
    
    TDVP performs this projection while staying on the MPS manifold.

Test:
    - Run imaginary-time TDVP on Heisenberg chain
    - Compare with DMRG ground state energy
    - Verify convergence

Criterion: |E_TDVP - E_DMRG| / L < 0.01
"""

import torch
from tensornet import MPS
from tensornet.mps.hamiltonians import heisenberg_mpo
from tensornet.algorithms.tdvp import tdvp_ground_state
from tensornet.algorithms.dmrg import dmrg


def test_tdvp_ground_state():
    """Test TDVP finds ground state."""
    L = 10
    chi_max = 32
    
    # Get DMRG reference
    mps_dmrg = MPS.random(L=L, d=2, chi=4)
    H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)
    _, E_dmrg, _ = dmrg(mps_dmrg, H, num_sweeps=15, chi_max=chi_max)
    
    # Run TDVP
    mps_tdvp = MPS.random(L=L, d=2, chi=4)
    _, E_tdvp, info = tdvp_ground_state(
        mps_tdvp, H,
        chi_max=chi_max,
        num_sweeps=30,
        dt=0.1,
    )
    
    error = abs(E_tdvp - E_dmrg) / L
    
    print(f"DMRG E/site:  {E_dmrg/L:.8f}")
    print(f"TDVP E/site:  {E_tdvp/L:.8f}")
    print(f"Error/site:   {error:.6f}")
    print(f"TDVP sweeps:  {info['num_sweeps']}")
    
    assert error < 0.01, f"Error {error:.6f} exceeds tolerance 0.01"
    
    print(f"✓ TDVP finds ground state (error < 1%)")
    
    return True


def test_tdvp_convergence():
    """Test TDVP energy decreases with sweeps."""
    L = 8
    
    mps = MPS.random(L=L, d=2, chi=4)
    H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)
    
    _, _, info = tdvp_ground_state(mps, H, chi_max=32, num_sweeps=20, dt=0.1)
    
    history = info['energy_history']
    
    # Check energy generally decreases
    decreasing = sum(1 for i in range(len(history)-1) if history[i+1] <= history[i] + 0.01)
    ratio = decreasing / (len(history) - 1)
    
    print(f"\nEnergy history (first 5): {[f'{e:.4f}' for e in history[:5]]}")
    print(f"Energy history (last 5):  {[f'{e:.4f}' for e in history[-5:]]}")
    print(f"Decreasing ratio: {ratio:.1%}")
    
    assert ratio > 0.7, f"Energy not decreasing consistently"
    
    print(f"✓ TDVP shows convergent behavior")
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Proof 19: TDVP Finds Ground State via Imaginary Time")
    print("=" * 60)
    print()
    
    success1 = test_tdvp_ground_state()
    success2 = test_tdvp_convergence()
    
    print()
    print("=" * 60)
    print("PROOF PASSED" if (success1 and success2) else "PROOF FAILED")
    print("=" * 60)
