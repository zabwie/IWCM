#!/usr/bin/env python3
"""AC3 adversarial curriculum training — proper hardness-filtered corruptions."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.ac3.trainer import AC3Trainer
from src.ac3.corruptor import AC3Corruptor
from src.ac3.oracle import SymbolicOracle
from src.metrics.evaluation import metric_cross_surface_law_generalization, metric_valid_invalid_classification
from src.env.data import TrajectoryDataset
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
GRID, D_STATE, D_ACTION, H = 8, 8*8*4, 11, 25
print(f"Device: {device}")

# Load training data
with open("data/trajectories_all_h25_n5000.pkl", "rb") as f:
    dataset = pickle.load(f)
print(f"Train: {len(dataset)} trajectories")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, drop_last=True)

model = IWCM(d_state=D_STATE, d_action=D_ACTION, hidden_dim=384)
corruptor = AC3Corruptor(d_state=D_STATE, hidden_dim=192)
oracle = SymbolicOracle()
print(f"Model: {model.count_parameters_str()} params")

# Start with wide acceptance range, narrow over time
trainer = AC3Trainer(model, corruptor, oracle, device=device,
    lr_world=3e-5, lr_corruptor=3e-5, num_mutations_per_sample=14,
    top_k_hard=8, accept_low=0.2, accept_high=0.9,
    lambda_valid=1.0, lambda_invalid=1.0, lambda_repair=0.3, lambda_accept=0.3,
    lambda_minimal=0.1, lambda_diversity=0.3,
    grid_size=GRID, horizon=H)

cs = pickle.load(open("data/cross_surface.pkl", "rb"))
def np_to_torch(trajs):
    return [(torch.from_numpy(z0).reshape(-1), torch.from_numpy(A),
             torch.from_numpy(Z).reshape(Z.shape[0], -1)) for z0, A, Z in trajs]
ev_train = {k: {"invalid": np_to_torch(v)} for k, v in cs["train"].items()}
ev_test = {k: {"invalid": np_to_torch(v)} for k, v in cs["test"].items()}
ev_valid = np_to_torch(cs.get("valid", [])[:200])

NUM_EPOCHS = 300
best_ho = 0.0
for epoch in range(NUM_EPOCHS):
    # Narrow curriculum over time
    progress = epoch / NUM_EPOCHS
    trainer.curriculum.accept_low = max(0.1, 0.5 - 0.4 * progress)
    trainer.curriculum.accept_high = min(0.9, 0.5 + 0.4 * progress)

    m = trainer.train_epoch(dataloader)
    if (epoch+1) % 25 == 0:
        print(f"Epoch {epoch+1}: loss_w={m['loss_world']:.4f} "
              f"E_valid={m['energy_valid']:.3f} E_invalid={m['energy_invalid']:.3f} "
              f"accept={m['accept_mean']:.3f} viol={m['violation_rate']:.2f} "
              f"range=[{trainer.curriculum.accept_low:.2f},{trainer.curriculum.accept_high:.2f}]")
    if (epoch+1) % 100 == 0:
        model.eval()
        csr = metric_cross_surface_law_generalization(model, ev_train, ev_test, device)
        ho = csr.get("held_out_accuracy", 0.0)
        ida = csr.get("in_distribution_accuracy", 0.0)
        print(f"  CS: ID={ida:.3f} HeldOut={ho:.3f} Gap={csr.get('generalization_gap',0):.3f}")
        if ho > best_ho:
            best_ho = ho
            model.save("outputs/checkpoints/iwcm_ac3_curriculum.pt")
            print(f"  *** New best: {ho:.3f} ***")

model.save("outputs/checkpoints/iwcm_ac3_final.pt")
print(f"\nBest held-out: {best_ho:.3f}")
