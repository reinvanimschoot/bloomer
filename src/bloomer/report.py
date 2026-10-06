def print_header():
    print("__________________________________________________________")
    print("|  epoch   |  train loss  |  accuracy  | validation loss |")


def print_epoch_report(epoch, train_loss, accuracy, validation_loss):

    print("----------------------------------------------------------")
    print(
        f"|{epoch:^10}|{train_loss:^14.4f}|{accuracy:^12.2f}|{validation_loss:^17.4f}|"
    )


def print_checkpoint(epoch):

    print("----------------------------------------------------------")
    print("Validation loss increasing for 3 epochs.")
    print(
        f"Patience reached, model weights of epoch {epoch} are saved in best_model.pt"
    )
