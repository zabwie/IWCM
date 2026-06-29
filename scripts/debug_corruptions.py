#!/usr/bin/env python3
"""Debug: compare mechanical vs oracle corruptions for a specific worldline."""
import sys; sys.path.insert(0, '.')
import pickle, numpy as np, torch
from src.tamg_simple import _corrupt_realistic

with open('data/compositional_grid.pkl', 'rb') as f:
    data = pickle.load(f)

# Pick a valid worldline
z0, A, Z = data['train_valid'][0]
Z_t = torch.from_numpy(Z).float().unsqueeze(0)  # (1, H, N, d)
H, N, d = Z_t.shape[1], Z_t.shape[2], Z_t.shape[3]
print(f"Z shape: {Z_t.shape}  H={H} N={N} d={d}")
print(f"Channel 15 (existence) before: {Z_t[0, :, :, 15].numpy().round(2)}")

# Mechanical delete
rng = np.random.RandomState(42)
Zc = _corrupt_realistic(Z_t.clone(), rng)
print(f"\nMechanical delete - diff: {(Zc != Z_t).any().item()}")
print(f"Channel 15 after: {Zc[0, :, :, 15].numpy().round(2)}")

# Compare with an oracle delete from train_corr
for item in data['train_corr']:
    enc, meta = item
    if meta['violation_type'] == 'delete':
        z0_o, A_o, Z_o = enc
        Z_o_t = torch.from_numpy(Z_o).float()
        print(f"\nOracle delete Z shape: {Z_o_t.shape}")
        print(f"Channel 15 (existence): {Z_o_t[:, :, 15].numpy().round(2)}")
        print(f"Channel 0-4 (type): {Z_o_t[0, :, :5].numpy().round(2)}")
        print(f"Channel 5-6 (pos): {Z_o_t[0, :, 5:7].numpy().round(2)}")
        print(f"Channel 7-8 (vel): {Z_o_t[0, :, 7:9].numpy().round(2)}")
        break

# Mechanical corrupted
print(f"\nMechanical corrupted:")
print(f"Channel 0-4 (type): {Zc[0, 0, :, :5].numpy().round(2)}")
print(f"Channel 5-6 (pos): {Zc[0, 0, :, 5:7].numpy().round(2)}")
print(f"Channel 7-8 (vel): {Zc[0, 0, :, 7:9].numpy().round(2)}")
