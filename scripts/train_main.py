#!/usr/bin/env python3
"""Paper-quality IWCM + AC3 training on full dataset."""
import sys, torch, pickle, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.ac3.trainer import AC3Trainer
from src.ac3.corruptor import AC3Corruptor
from src.ac3.oracle import SymbolicOracle
from src.metrics.evaluation import (
    metric_cross_surface_law_generalization,
    metric_valid_invalid_classification,
    metric_long_horizon_drift,
    RolloutModel,
)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
GRID, D_STATE, D_ACTION, H = 8, 8*8*4, 11, 25

# Load training data
with open("data/trajectories_all_h25_n5000.pkl", "rb") as f:
    dataset = pickle.load(f)
print(f"Train: {len(dataset)} trajectories")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

# Models
model = IWCM(d_state=D_STATE, d_action=D_ACTION, hidden_dim=256)
corruptor = AC3Corruptor(d_state=D_STATE, hidden_dim=128)
oracle = SymbolicOracle()
rollout = RolloutModel(D_STATE, D_ACTION).to(device)
print(f"IWCM: {model.count_parameters_str()} params")

trainer = AC3Trainer(model, corruptor, oracle, device=device,
    lr_world=1e-4, lr_corruptor=1e-4, num_mutations_per_sample=2,
    top_k_hard=4, accept_low=0.3, accept_high=0.8)

# Load eval data
cs = pickle.load(open("data/cross_surface.pkl", "rb"))
def np_to_torch(trajs):
    return [(torch.from_numpy(z0).reshape(-1), torch.from_numpy(A),
             torch.from_numpy(Z).reshape(Z.shape[0], -1)) for z0, A, Z in trajs]
cs_valid_t = np_to_torch(cs.get("valid", [])[:100])
cs_train_t = {k: {"invalid": np_to_torch(v)} for k, v in cs["train"].items()}
cs_test_t = {k: {"invalid": np_to_torch(v)} for k, v in cs["test"].items()}
cons_train_t = np_to_torch(cs["train"].get("conservation", [])[:100])

NUM_EPOCHS = 50
print(f"\nTraining {NUM_EPOCHS} epochs...")
for epoch in range(NUM_EPOCHS):
    m = trainer.train_epoch(dataloader)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}: loss_w={m['loss_world']:.4f} "
              f"E_valid={m['energy_valid']:.3f} E_invalid={m['energy_invalid']:.3f} "
              f"accept={m['accept_mean']:.3f}")
    if (epoch + 1) % 25 == 0:
        model.eval()
        cs_r = metric_cross_surface_law_generalization(model, cs_train_t, cs_test_t, device)
        print(f"  Cross-surface: held_out_acc={cs_r['held_out_accuracy']:.3f} gen_gap={cs_r['generalization_gap']:.3f}")

print("\nFinal Evaluation...")
model.eval()
cs_r = metric_cross_surface_law_generalization(model, cs_train_t, cs_test_t, device)
print(f"Cross-surface: in_dist={cs_r['in_distribution_accuracy']:.3f} "
      f"held_out={cs_r['held_out_accuracy']:.3f} gap={cs_r['generalization_gap']:.3f}")
for law, v in cs_r['per_law_breakdown'].items():
    print(f"  {law}: valid_acc={v['valid_accuracy']:.3f} invalid_rej={v['invalid_rejection']:.3f}")

cls_r = metric_valid_invalid_classification(model, cs_valid_t, {"conservation": cons_train_t}, device)
print(f"Classification: AUROC={cls_r['AUROC']:.3f} FPR={cls_r['FPR']:.3f}")

drift_r = metric_long_horizon_drift(model, rollout, cs_valid_t[:50],
                                     horizons=[10, 25, 50, 100], device=device)
print(f"Drift CVR: H=10:{drift_r['iwcm_cvr'].get(10,0):.3f} H=25:{drift_r['iwcm_cvr'].get(25,0):.3f} "
      f"H=50:{drift_r['iwcm_cvr'].get(50,0):.3f} H=100:{drift_r['iwcm_cvr'].get(100,0):.3f}")

model.save("outputs/checkpoints/iwcm_ac3_trained.pt")
print("\nModel saved. Training complete.")
