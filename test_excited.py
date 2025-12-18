#!/usr/bin/env python3
"""Test excited states with Heisenberg model."""
from tensornet import MPS
from tensornet.mps.hamiltonians import heisenberg_mpo
from tensornet.algorithms.excited import find_excited_states

L = 8
H = heisenberg_mpo(L=L, J=1.0, Jz=1.0)

print('Finding states in Heisenberg model using Sz sectors...')
states, energies, info = find_excited_states(H, num_states=3, chi_max=32, num_sweeps=15)

for i, E in enumerate(energies):
    Sz = info["sectors"][i]
    print(f'E{i} = {E:.6f} (E/site = {E/L:.6f}) [Sz={Sz}]')

gaps = info["spectral_gaps"]
print(f'Gaps: {[f"{g:.4f}" for g in gaps]}')
