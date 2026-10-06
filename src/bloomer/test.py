import torch

from bloomer.data import test_loader
from bloomer.model import build_model
from bloomer.training import validate_epoch


def main() -> None:
    model = build_model()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model = model.to(device)

    accuracy, loss = validate_epoch(model, device, test_loader)
    print(f"test accuracy: {accuracy:.2f}%  test loss: {loss:.4f}")
