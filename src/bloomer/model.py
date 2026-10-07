import torch
from torch import nn
from torchvision import models


def build_model(pretrained=True):
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)

    # Freezes all parameters in the model so they don't get changed,
    # when the new last layer gets trained.
    for param in model.parameters():
        param.requires_grad = False

    # Replaces the last layer of the head with one that outputs
    # a score for each of the 102 flower classes.
    old_layer = model.classifier[3]
    assert isinstance(old_layer, nn.Linear)
    model.classifier[3] = nn.Linear(old_layer.in_features, 102)

    return model


def load_model(path="best_model.pt"):
    # No pretrained weights needed: they're replaced by the saved ones right away.
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()

    return model


def unfreeze_last_blocks(model, n):
    start_block = len(model.features) - n
    blocks = model.features[start_block:]

    for param in blocks.parameters():
        param.requires_grad = True

    return blocks
