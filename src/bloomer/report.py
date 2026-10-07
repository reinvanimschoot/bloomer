def print_header():
    print("__________________________________________________________")
    print("|  epoch   |  train loss  |  accuracy  | validation loss |")


def print_epoch_report(epoch, train_loss, accuracy, validation_loss):

    print("----------------------------------------------------------")
    print(
        f"|{epoch:^10}|{train_loss:^14.4f}|{accuracy:^12.2f}|{validation_loss:^17.4f}|"
    )


def print_early_stop(patience):
    print("----------------------------------------------------------")
    print(f"Validation loss hasn't improved for {patience} epochs, stopping early.")


def print_checkpoint(epoch, path):
    print("----------------------------------------------------------")
    print(f"Best model: epoch {epoch}, saved in {path}")
