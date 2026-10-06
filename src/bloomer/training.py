import torch
import torch.nn.functional as F


def train_epoch(model, device, loader, optimizer):
    total_loss = 0.0

    model.train()
    model.features.eval()
    for images_batch, labels_batch in loader:
        images = images_batch.to(device)
        labels = labels_batch.to(device)

        output = model(images)

        loss = F.cross_entropy(output, labels)
        total_loss += loss.item() * len(labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loader.dataset)

    return average_loss


def validate_epoch(model, device, loader):
    total_loss = 0
    total_correct_predictions = 0

    model.eval()
    with torch.no_grad():
        for images_batch, labels_batch in loader:
            images = images_batch.to(device)
            labels = labels_batch.to(device)

            output = model(images)

            loss = F.cross_entropy(output, labels)
            total_loss += loss.item() * len(labels)

            predicted = output.argmax(dim=1)
            correct = (predicted == labels).sum().item()
            total_correct_predictions += correct

    accuracy = (total_correct_predictions / len(loader.dataset)) * 100.0
    average_loss = total_loss / len(loader.dataset)

    return accuracy, average_loss
