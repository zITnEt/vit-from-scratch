# ViT from scratch

A Vision Transformer implemented from scratch in PyTorch, trained on CIFAR-10.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m training.train
```

CIFAR-10 downloads automatically to `data/` on first run. Checkpoints are
written to `checkpoints/` (`last.pt` every epoch, `best.pt` on validation
improvement).

## Inference

```bash
# Sanity check on a real CIFAR-10 test image (prints true vs. predicted)
python -m inference.classify

# Classify your own image file
python -m inference.classify path/to/image.png
```

Inputs are resized to 32x32 and normalized with the same CIFAR-10 statistics
used in training (`config.CIFAR_MEAN` / `config.CIFAR_STD`).

## Layout

```
config/      hyperparameters, paths, and shared constants
model/        patch embedding, attention, encoder, transformer
training/     dataset/transforms and the training loop
inference/    single-image classification
```
