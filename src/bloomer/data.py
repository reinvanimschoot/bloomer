import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = v2.Compose(
    [
        # cuts out a random part of the image and scales it to 224 x 224
        # every epoch the model sees a slightly different version of each photo
        # With only 10 training images per class, this helps a lot against overfitting
        v2.RandomResizedCrop(224),
        # mirrors the image half the time
        # A flower mirrored is still the same flower (as opposed to letters or digits)
        v2.RandomHorizontalFlip(),
        # Changes the container, not the values. It turns a PIL image into a PyTorch tensor.
        #  This changes two things:
        # - The axis order. PIL stores images as height x width x channels, while PyTorch
        #   uses a shape of (3, H, W).
        # - The type becomes a tensor, so PyTorch can work with it.
        v2.ToImage(),
        # Changes the values. It turns the whole numbers into floats.
        # scale=True also divides them by 255, so 0-255 becomes 0.0-1.0.
        # This is needed because neural networks compute in floats, not integers, and Normalize
        # expects values in the 0-1 range.
        v2.ToDtype(torch.float32, scale=True),
        # Subtracts the ImageNet mean and divides by the standard deviation, per colour channel.
        # Essentially, this rewrites each pixel's colour value as "how far above or below average is this?".
        # An average pixel becomes 0, a brighter one gets a positive number, a darker one a negative number.
        # This is needed because a pretrained model learned from images described that way, so we need
        # to describe our images to it in the same way.
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

eval_transform = v2.Compose(
    [
        # Scales the short side to 256 pixels.
        # No randomness so evaluation gives the same results every time.
        v2.Resize(256),
        # Cuts the middle 224x224 out.
        # No randomness so evaluation gives the same results every time.
        v2.CenterCrop(224),
        # See train_transform above
        v2.ToImage(),
        # See train_transform above
        v2.ToDtype(torch.float32, scale=True),
        # See train_transform above
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def build_loader(split, batch_size=32):
    # Only the training split gets random augmentation and shuffling.
    is_training = split == "train"
    transform = train_transform if is_training else eval_transform

    dataset = datasets.Flowers102(
        root="data", download=True, split=split, transform=transform
    )

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
