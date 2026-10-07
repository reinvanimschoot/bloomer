# Bloomer

Bloomer is an educational project I built as a follow-up to Numberly. Numberly covered the basics of training a network but its model was built from scratch and only had two linear layers. I wanted to learn more about working with an existing, larger, pre-trained model and how to fine-tune it.

Bloomer was built by fine-tuning a pretrained MobileNetV3 model with PyTorch on the [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) dataset. The trained model runs entirely in the browser, so the live app is a static site with no backend.

It allows you to upload a photo of a flower and the model guesses which type it is. The current limitation is that it only recognizes 102 species, though.

⚠️ _The web part was built by Claude since it wasn't part of what I wanted to learn about._ ⚠️

**[Try it on Hugging Face Spaces →](https://huggingface.co/spaces/reinvanimschoot/bloomer)**

## How it works

### The model

Bloomer starts from MobileNetV3-Large, pretrained on ImageNet. A model like this has two parts:

- **The backbone** turns an image into 960 numbers that describe which visual patterns it contains: edges, textures, shapes.
- **The head** turns those numbers into a score per class.

The backbone is kept, since what it learned from over a million images is useful for flowers too. The head's final layer, which scored ImageNet's 1,000 classes, is replaced with a new one for the 102 flower species:

```
image → backbone (17 blocks) → 960 values → head → 102 scores, one per species
```

### The data

Oxford 102 has an unusual split: only 10 images per class for training and 10 for validation (1,020 each), and 6,149 for testing. With so little training data, starting from a pretrained backbone is what makes the task feasible at all.

Training images are augmented with random crops and horizontal flips, so the model never sees exactly the same image twice and can't simply memorize the training set.

### Training in two stages

1. **Head only.** The whole backbone is frozen and only the new final layer learns: 130,662 of the model's 4.3 million weights. After up to 20 epochs this reaches about **88%** validation accuracy.
2. **Fine-tuning.** Starting from the best model of stage 1, the last 3 of the backbone's 17 blocks are unfrozen as well (1.9 million trainable weights). They get a learning rate ten times lower than the head, so they're adjusted rather than overwritten. After up to 10 more epochs: about **91%**.

Unfreezing the last 6 blocks instead gave no improvement and a slightly higher validation loss: with 10 images per class, the extra freedom went into fitting the training images rather than learning anything new.

Throughout, the backbone's BatchNorm layers are kept in evaluation mode, so their statistics stay the ones learned on ImageNet.

- **Loss:** cross-entropy
- **Optimizer:** Adam; learning rate 0.001 for the head, 0.0001 for the unfrozen blocks
- **Batch size:** 32
- **Early stopping:** each stage stops when the validation loss hasn't improved for 3 epochs, and the best checkpoint is kept
- **Hardware:** trained on a MacBook GPU (PyTorch's MPS backend)

### Results

All decisions were made on the validation set. The test set was used once, at the end:

|            | Accuracy  | Loss     |
| ---------- | --------- | -------- |
| Validation | ??%       | ??       |
| **Test**   | **88.5%** | **0.40** |

The test score is a few points lower than validation. That's expected: the validation set was used to pick the best epoch and the number of unfrozen blocks, which makes its score slightly optimistic. The test set played no part in any decision, and with six times as many images, it covers more of the harder, less typical photos.

### Preprocessing

The pretrained backbone expects images in the same format it was trained on: 224 × 224 pixels, with each colour channel normalized using ImageNet's mean and standard deviation. For evaluation, images are resized so their short side is 256 pixels, and the central 224 × 224 square is cropped.

The app applies the same steps in JavaScript before predicting. Mismatched preprocessing doesn't cause an error, just quietly bad predictions, so this has to match the Python version exactly.

### Running in the browser

The trained PyTorch model is exported to [ONNX](https://onnx.ai/) and run in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). The model file (about 17 MB) is hosted in a Hugging Face model repository, and the page downloads it on load.

## Project structure

```
bloomer/
├── src/bloomer/
│   ├── data.py       # Oxford 102 transforms and data loaders
│   ├── model.py      # building the model (MobileNetV3, frozen, new head), unfreezing and loading
│   ├── training.py   # one training epoch, and validation
│   ├── report.py     # the per-epoch results table
│   ├── train.py      # the training command (both stages, early stopping, checkpoints)
│   ├── test.py       # the final evaluation on the test set
│   └── export.py     # exporting the model to ONNX
├── web/
│   ├── index.html    # the app: upload, preprocessing, inference
│   └── README.md     # configuration for the Hugging Face Space
└── pyproject.toml
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Train

```bash
uv run bloomer-train
```

Downloads Oxford 102 into `data/` on the first run (about 350 MB) and runs both training stages. The best model of stage 1 is saved to `checkpoint_model_frozen.pt`, the final model to `best_model.pt`.

### Test

```bash
uv run bloomer-test
```

Evaluates `best_model.pt` on the 6,149 test images.

### Export

```bash
uv run bloomer-export
```

Exports `best_model.pt` to `bloomer.onnx`, and checks that its output matches the PyTorch model.

### Run the app locally

```bash
uv run python -m http.server 8000 --directory web
```

Then open [http://localhost:8000](http://localhost:8000). The page loads the model from Hugging Face, and falls back to a `bloomer.onnx` in `web/` if that fails.

## Deployment

- **Model:** `bloomer.onnx` is uploaded to the [reinvanimschoot/bloomer](https://huggingface.co/reinvanimschoot/bloomer) model repository:

  ```bash
  hf upload reinvanimschoot/bloomer bloomer.onnx
  ```

- **App:** the `web/` folder is deployed as a static Hugging Face Space, using a git subtree push:

  ```bash
  git subtree push --prefix web space main
  ```

Model files (`*.pt`, `*.onnx`) and the dataset are kept out of git.

## Limitations

- The model only knows the 102 species in Oxford 102. Show it any other plant and it will still pick the closest match.
- Oxford's photos are mostly clear shots of a single flower. Real-world photos, with several flowers, busy backgrounds or unusual angles, will do noticeably worse than the test score suggests.
