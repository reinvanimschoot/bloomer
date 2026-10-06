import matplotlib.pyplot as plt
from torchvision import datasets

training_set = datasets.Flowers102(root="data", split="train", download=True)

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    img, label = training_set[i * 10]
    ax.imshow(img)
    ax.set_title(f"label {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()
