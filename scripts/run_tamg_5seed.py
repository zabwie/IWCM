#!/usr/bin/env python3
"""5-seed SimpleTAMG evaluation."""
import sys; sys.path.insert(0, '.')
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import torch
from src.utils.seed import set_seed
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from src.tamg_simple import SimpleTAMG, load_compositional_grid, _corrupt_realistic

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
data = load_compositional_grid('data/compositional_grid.pkl')
test_valid = data['test_valid']
test_corr = data['test_corr']
train_valid = data['train_valid']

def run(seed, epochs=500):
    set_seed(seed)
    model = SimpleTAMG(d_slot=ORACLE_SLOT_DIM, d_action=11,
                       hidden=256, num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    K = 8
    rng_pre = np.random.RandomState(seed + 1)
    all_pairs = []
    for z0, A, Z in train_valid:
        Z_exp = Z.unsqueeze(0)
        for _ in range(K):
            Zc = _corrupt_realistic(Z_exp.clone(), rng_pre)
            if (Zc != Z_exp).any():
                all_pairs.append((z0.clone(), A.clone(),
                                  Z_exp.squeeze(0).clone(), Zc.squeeze(0).clone()))

    all_pairs = [(z0.to(DEVICE), A.to(DEVICE), Zv.to(DEVICE), Zc.to(DEVICE))
                 for z0, A, Zv, Zc in all_pairs]
    n_pairs = len(all_pairs)
    batch = 32

    for ep in range(epochs):
        n_batches = max(1, n_pairs // batch)
        for _ in range(n_batches):
            pi = np.random.choice(n_pairs, min(batch, n_pairs), replace=False)
            z0_b = torch.stack([all_pairs[i][0] for i in pi])
            A_b  = torch.stack([all_pairs[i][1] for i in pi])
            Zv_b = torch.stack([all_pairs[i][2] for i in pi])
            Zc_b = torch.stack([all_pairs[i][3] for i in pi])
            opt.zero_grad()
            Ev = model.energy_fn(z0_b, A_b, Zv_b)
            Ec = model.energy_fn(z0_b, A_b, Zc_b)
            loss = (torch.relu(Ev + 0.5).mean() +
                    torch.relu(Ev + 1.0 - Ec).mean() +
                    0.001 * (Ev.pow(2).mean() + Ec.pow(2).mean()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    valid_scores = []
    for z0, A, Z in test_valid:
        s = model.energy_fn(z0.unsqueeze(0).to(DEVICE),
                            A.unsqueeze(0).to(DEVICE),
                            Z.unsqueeze(0).to(DEVICE)).item()
        valid_scores.append(s)

    by_type = defaultdict(list)
    for z0, A, Z, meta in test_corr:
        by_type[meta['violation_type']].append((z0, A, Z))

    results = {}
    for vtype, items in sorted(by_type.items()):
        corr_scores = [model.energy_fn(
            z0.unsqueeze(0).to(DEVICE),
            A.unsqueeze(0).to(DEVICE),
            Z.unsqueeze(0).to(DEVICE)).item()
            for z0, A, Z in items]
        labels = [0] * len(valid_scores) + [1] * len(corr_scores)
        results[vtype] = roc_auc_score(labels, valid_scores + corr_scores)

    all_corr = [model.energy_fn(
        z0.unsqueeze(0).to(DEVICE),
        A.unsqueeze(0).to(DEVICE),
        Z.unsqueeze(0).to(DEVICE)).item()
        for items in by_type.values() for z0, A, Z in items]
    labels_all = [0] * len(valid_scores) + [1] * len(all_corr)
    results['overall'] = roc_auc_score(labels_all, valid_scores + all_corr)
    return results

print('SimpleTAMG 5-seed (hidden=256, 500 epochs)')
print('-' * 60)
all_results = {}
for seed in [42, 123, 456, 789, 1011]:
    r = run(seed)
    for k, v in r.items():
        all_results.setdefault(k, []).append(v)
    print(f'  seed={seed}: overall={r["overall"]:.4f}  '
          f'del={r.get("delete",0):.4f}  dup={r.get("duplicate",0):.4f}  '
          f'swap={r.get("swap",0):.4f}')

print('-' * 60)
for k in sorted(all_results.keys()):
    vals = all_results[k]
    marker = ' *' if k == 'overall' else ''
    print(f'  {k:<14s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}{marker}')
ov = all_results['overall']
print(f'\n  OVERALL: {np.mean(ov):.4f} +/- {np.std(ov):.4f}')
gap = 0.95 - np.mean(ov)
if gap <= 0:
    print(f'  TARGET 0.95: PASS (+{-gap:.4f})')
else:
    print(f'  TARGET 0.95: GAP={gap:.4f}')
