import torch

from bloomer.data import training_loader, validation_loader
from bloomer.model import build_model
from bloomer.report import print_checkpoint, print_epoch_report, print_header
from bloomer.training import train_epoch, validate_epoch


def main() -> None:
    model = build_model()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    model.load_state_dict(torch.load("checkpoint_model_frozen.pt"))

    optimizer = torch.optim.Adam(
        [
            {"params": model.classifier[3].parameters(), "lr": 1e-3},
            {"params": model.features[14:].parameters(), "lr": 1e-4},
        ]
    )

    # Unfreezes the last 3 layers of the backbone
    for param in model.features[14:].parameters():
        param.requires_grad = True

    print_header()

    min_validation_loss = float("inf")
    patience = 3
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, 21):
        train_loss = train_epoch(
            model,
            device,
            training_loader,
            optimizer,
        )
        accuracy, validation_loss = validate_epoch(model, device, validation_loader)

        print_epoch_report(epoch, train_loss, accuracy, validation_loss)

        if validation_loss < min_validation_loss:
            epochs_without_improvement = 0
            min_validation_loss = validation_loss

            best_epoch = epoch

            torch.save(model.state_dict(), "best_model.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print_checkpoint(best_epoch)
            break
