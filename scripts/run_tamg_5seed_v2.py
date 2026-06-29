#!/usr/bin/env python3
"""5-seed on-the-fly SimpleTAMG (best config from single-seed trial)."""
import sys; sys.path.insert(0, '.')
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch
from src.utils.seed import set_seed
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from src.tamg_simple import SimpleTAMG, load_compositional_grid

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data = load_compositional_grid('data/compositional_grid.pkl')
test_valid = data['test_valid']
test_corr = data['test_corr']
train_valid = data['train_valid']

def run(seed):
    set_seed(seed)
    model = SimpleTAMG(d_slot=ORACLE_SLOT_DIM, d_action=11,
                       hidden=128, num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv = len(train_valid)
    batch = 32
    all_v = [(z0.to(DEVICE), A.to(DEVICE), Z.to(DEVICE)) for z0, A, Z in train_valid]

    for ep in range(500):
        n_batches = max(1, nv // batch)
        for _ in range(n_batches):
            vi = np.random.choice(nv, min(batch, nv), replace=False)
            vz0 = torch.stack([all_v[i][0] for i in vi])
            vA  = torch.stack([all_v[i][1] for i in vi])
            vZ  = torch.stack([all_v[i][2] for i in vi])
            opt.zero_grad()
            loss = model.training_step(vz0, vA, vZ)
            if loss.item() == 0.0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    valid_scores = [model.energy_fn(z0.unsqueeze(0).to(DEVICE),
                     A.unsqueeze(0).to(DEVICE), Z.unsqueeze(0).to(DEVICE)).item()
                    for z0, A, Z in test_valid]

    by_type = defaultdict(list)
    for z0, A, Z, meta in test_corr:
        by_type[meta['violation_type']].append((z0, A, Z))

    results = {}
    for vtype, items in sorted(by_type.items()):
        cs = [model.energy_fn(z0.unsqueeze(0).to(DEVICE),
              A.unsqueeze(0).to(DEVICE), Z.unsqueeze(0).to(DEVICE)).item()
              for z0, A, Z in items]
        results[vtype] = roc_auc_score([0]*len(valid_scores)+[1]*len(cs),
                                        valid_scores+cs)

    all_corr = [model.energy_fn(z0.unsqueeze(0).to(DEVICE),
                 A.unsqueeze(0).to(DEVICE), Z.unsqueeze(0).to(DEVICE)).item()
                for items in by_type.values() for z0, A, Z in items]
    results['overall'] = roc_auc_score([0]*len(valid_scores)+[1]*len(all_corr),
                                        valid_scores+all_corr)
    return results

print('SimpleTAMG 5-seed (on-the-fly, hidden=128, 500 epochs)')
print('-' * 64)
all_results = {}
for seed in [42, 123, 456, 789, 1011]:
    r = run(seed)
    for k, v in r.items():
        all_results.setdefault(k, []).append(v)
    print(f'  seed={seed}: ov={r["overall"]:.4f}  del={r.get("delete",0):.4f}  '
          f'dup={r.get("duplicate",0):.4f}  tel={r.get("teleport",0):.4f}')

print('-' * 64)
for k in sorted(all_results.keys()):
    vals = all_results[k]
    m = ' *' if k == 'overall' else ''
    print(f'  {k:<14s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{m}')
ov = all_results['overall']
print(f'\n  OVERALL: {np.mean(ov):.4f} +/- {np.std(ov):.4f}')
print(f'  TARGET 0.95: {"PASS" if np.mean(ov) >= 0.95 else f"GAP={0.95 - np.mean(ov):.4f}"}')
