import torch

from bloomer.data import build_loader
from bloomer.model import load_model
from bloomer.training import validate_epoch


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = load_model("best_model.pt").to(device)

    test_loader = build_loader("test")

    accuracy, loss = validate_epoch(model, device, test_loader)
    print(f"test accuracy: {accuracy:.2f}%  test loss: {loss:.4f}")
