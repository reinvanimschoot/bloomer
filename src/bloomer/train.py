import torch

from bloomer.data import build_loader
from bloomer.model import build_model, unfreeze_last_blocks
from bloomer.report import (
    print_checkpoint,
    print_early_stop,
    print_epoch_report,
    print_header,
)
from bloomer.training import train_epoch, validate_epoch

FROZEN_CHECKPOINT = "checkpoint_model_frozen.pt"
FINAL_CHECKPOINT = "best_model.pt"
N_UNFROZEN_BLOCKS = 3


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    loaders = (build_loader("train"), build_loader("val"))

    # Stage 1: the backbone is frozen, only the new last layer learns.
    print("Stage 1: training the head")
    optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=1e-3)
    run_training(model, device, optimizer, loaders, FROZEN_CHECKPOINT, max_epochs=20)

    # Stage 2: also unfreeze the last blocks of the backbone, with a lower learning rate.
    print(f"\nStage 2: fine-tuning the last {N_UNFROZEN_BLOCKS} backbone blocks")
    unfrozen_blocks = unfreeze_last_blocks(model, N_UNFROZEN_BLOCKS)
    optimizer = torch.optim.Adam(
        [
            {"params": model.classifier[3].parameters(), "lr": 1e-3},
            {"params": unfrozen_blocks.parameters(), "lr": 1e-4},
        ]
    )
    run_training(model, device, optimizer, loaders, FINAL_CHECKPOINT, max_epochs=10)


def run_training(model, device, optimizer, loaders, save_path, max_epochs, patience=3):
    print_header()

    min_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    training_loader, validation_loader = loaders

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, device, training_loader, optimizer)
        accuracy, validation_loss = validate_epoch(model, device, validation_loader)

        print_epoch_report(epoch, train_loss, accuracy, validation_loss)

        if validation_loss < min_validation_loss:
            epochs_without_improvement = 0
            min_validation_loss = validation_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print_early_stop(patience)
            break

    print_checkpoint(best_epoch, save_path)

    # The model in memory has the weights of the last epoch, which may not be
    # the best one. Load the best weights back, so the model continues from those.
    model.load_state_dict(torch.load(save_path, map_location=device))
