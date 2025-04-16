
import torch
import json
from train_model import SimpleCNN  # Make sure train_model.py is accessible

# Load the saved model (you can switch this to cnn_model.pth)
model_path = "models/robust_cnn_model.pth"
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Print weight names and shapes
print("\n[INFO] Model Parameters:")
for name, param in model.named_parameters():
    print(f"{name} -> Shape: {tuple(param.shape)}")
    print(f"Sample values (first 2): {param.view(-1)[:2].tolist()}\n")

# Optional: Export weights to JSON for inspection
export = input("Do you want to export weights to JSON? (yes/no): ").strip().lower()
if export == "yes":
    weights = {k: v.tolist() for k, v in model.state_dict().items()}
    with open("model_weights.json", "w") as f:
        json.dump(weights, f)
    print("[INFO] Model weights exported to model_weights.json")

