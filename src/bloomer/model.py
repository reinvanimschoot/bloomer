from torch import nn
from torchvision import models

weights = models.MobileNet_V3_Large_Weights.DEFAULT


def build_model():
    weights = models.MobileNet_V3_Large_Weights.DEFAULT
    model = models.mobilenet_v3_large(weights=weights)

    # Freezes all parameters in the model so they don't get changed,
    # when the new last layer gets trained.
    for param in model.parameters():
        param.requires_grad = False

    # Replaces last layer of the head with a layer that has the size of our
    # desired classifiers.
    model.classifier[3] = nn.Linear(1280, 102)

    return model
