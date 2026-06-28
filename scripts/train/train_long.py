#!/usr/bin/env python3
"""Extended training for paper-quality results — 200 epochs."""
import sys, torch, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.ac3.trainer import AC3Trainer
from src.ac3.corruptor import AC3Corruptor
from src.ac3.oracle import SymbolicOracle
from src.metrics.evaluation import metric_cross_surface_law_generalization, metric_valid_invalid_classification

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
GRID, D_STATE, D_ACTION, H = 8, 8*8*4, 11, 25

# Load training data
with open("data/trajectories_all_h25_n5000.pkl", "rb") as f:
    dataset = pickle.load(f)
print(f"Train: {len(dataset)} trajectories")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

# Larger model
model = IWCM(d_state=D_STATE, d_action=D_ACTION, hidden_dim=384)
corruptor = AC3Corruptor(d_state=D_STATE, hidden_dim=192)
oracle = SymbolicOracle()
print(f"IWCM: {model.count_parameters_str()} params")

trainer = AC3Trainer(model, corruptor, oracle, device=device,
    lr_world=5e-5, lr_corruptor=5e-5, num_mutations_per_sample=4,
    top_k_hard=4, accept_low=0.3, accept_high=0.8,
    grid_size=GRID, horizon=H)

# Load eval data
cs = pickle.load(open("data/cross_surface.pkl", "rb"))
def np_to_torch(trajs):
    return [(torch.from_numpy(z0).reshape(-1), torch.from_numpy(A),
             torch.from_numpy(Z).reshape(Z.shape[0], -1)) for z0, A, Z in trajs]
cs_valid_t = np_to_torch(cs.get("valid", [])[:200])
cs_train_t = {k: {"invalid": np_to_torch(v)} for k, v in cs["train"].items()}
cs_test_t = {k: {"invalid": np_to_torch(v)} for k, v in cs["test"].items()}

NUM_EPOCHS = 200
print(f"\nTraining {NUM_EPOCHS} epochs...")
best_ho = 0.0
for epoch in range(NUM_EPOCHS):
    m = trainer.train_epoch(dataloader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: loss_w={m['loss_world']:.4f} "
              f"E_valid={m['energy_valid']:.3f} E_invalid={m['energy_invalid']:.3f} "
              f"accept={m['accept_mean']:.3f} viol={m['violation_rate']:.2f}")
    if (epoch + 1) % 50 == 0:
        model.eval()
        cs_r = metric_cross_surface_law_generalization(model, cs_train_t, cs_test_t, device)
        ho = cs_r.get('held_out_accuracy', 0.0)
        ida = cs_r.get('in_distribution_accuracy', 0.0)
        print(f"  Cross-surface: ID={ida:.3f} HeldOut={ho:.3f} Gap={cs_r.get('generalization_gap',0):.3f}")
        if ho > best_ho:
            best_ho = ho
            model.save("outputs/checkpoints/iwcm_best_cross_surface.pt")
            print(f"  *** New best held-out: {ho:.3f} ***")

print(f"\nBest held-out accuracy: {best_ho:.3f}")
print(f"Model saved to outputs/checkpoints/iwcm_best_cross_surface.pt")
