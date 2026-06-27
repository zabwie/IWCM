#!/usr/bin/env python3
"""Quick end-to-end training test — verifies the full IWCM+AC3 pipeline on GPU."""
import sys, torch, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.ac3.trainer import AC3Trainer
from src.ac3.corruptor import AC3Corruptor
from src.ac3.oracle import SymbolicOracle
from src.env.data import TrajectoryDataset
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.env.symbolic_state import SymbolicState, SymbolicTrajectory
from src.metrics.evaluation import RolloutModel, metric_cross_surface_law_generalization, metric_valid_invalid_classification
set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load data
data_path = "data/trajectories_key_door_simple_h25_n5000.pkl"
with open(data_path, "rb") as f:
    dataset = pickle.load(f)
print(f"Dataset: {len(dataset)} trajectories")

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
grid_size, d_state, d_action, H = 8, 8*8*4, 11, 25

# Models
world_model = IWCM(d_state=d_state, d_action=d_action, hidden_dim=256)
corruptor = AC3Corruptor(d_state=d_state, hidden_dim=128)
oracle = SymbolicOracle()
print(f"IWCM params: {world_model.count_parameters_str()}")

trainer = AC3Trainer(world_model, corruptor, oracle, device=device,
    lr_world=1e-4, lr_corruptor=1e-4, num_mutations_per_sample=2,
    top_k_hard=4, accept_low=0.3, accept_high=0.8)

print("Training 5 epochs...")
for epoch in range(5):
    metrics = trainer.train_epoch(dataloader)
    print(f"Epoch {epoch+1}: loss_w={metrics['loss_world']:.4f}, "
          f"loss_c={metrics['loss_corruptor']:.4f}, "
          f"E_valid={metrics['energy_valid']:.3f}, "
          f"E_invalid={metrics['energy_invalid']:.3f}, "
          f"accept={metrics['accept_mean']:.3f}")

print("\nQuick eval...")
world_model.eval()
# Load cross-surface data for evaluation
cs_path = "data/cross_surface.pkl"
with open(cs_path, "rb") as f:
    cs = pickle.load(f)
cs_train = {k: {"invalid": v} for k, v in cs["train"].items()}
cs_test = {k: {"invalid": v} for k, v in cs["test"].items()}
cs_valid = cs.get("valid", [])[:50]

# Metric 1: Cross-surface
# Metric 1: Cross-surface
def np_to_torch(trajs):
    result = []
    for z0, A, Z in trajs:
        z0_t = torch.from_numpy(z0).reshape(-1)  # flatten to (256,)
        A_t = torch.from_numpy(A)
        Z_t = torch.from_numpy(Z).reshape(Z.shape[0], -1)  # flatten to (H, 256)
        result.append((z0_t, A_t, Z_t))
    return result
cs_train_t = {k: {"invalid": np_to_torch(v)} for k, v in cs["train"].items()}
cs_test_t = {k: {"invalid": np_to_torch(v)} for k, v in cs["test"].items()}
cs_valid_t = np_to_torch(cs.get("valid", [])[:50])
cs_result = metric_cross_surface_law_generalization(world_model, cs_train_t, cs_test_t, device)
print(f"Cross-surface: held_out_acc={cs_result.get('held_out_accuracy',0):.3f}, "
      f"gen_gap={cs_result.get('generalization_gap',0):.3f}")

# Metric 2: Classification (quick)
if cs_valid and cs["train"].get("conservation", []):
    cons_train_t = np_to_torch(cs["train"]["conservation"][:50])
    cls_result = metric_valid_invalid_classification(
        world_model, cs_valid_t,
        {"conservation": cons_train_t}, device)
    print(f"Classification: AUROC={cls_result['AUROC']:.3f}, FPR={cls_result['FPR']:.3f}")

print("\nPipeline verified!")
