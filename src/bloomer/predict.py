import torch

from bloomer.model import build_model


def load_model(path="best_model.pt"):
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    return model


def predict(model, image):
    with torch.no_grad():
        batch = image.unsqueeze(0)
        output = model(batch)

        prediction = output.argmax(dim=1).item()
        probabilities = torch.softmax(output, dim=1)

        return prediction, probabilities[0]
