import sys
import torch
from PIL import Image
from torchvision import datasets, transforms

from config import config, CHECKPOINT_DIR, DATA_DIR, CIFAR_MEAN, CIFAR_STD, CLASSES
from model.transformer import Transformer

CHECKPOINT = CHECKPOINT_DIR / "best.pt"

# Same normalization as training, with a Resize so arbitrary-sized files fit.
preprocess = transforms.Compose([
    transforms.Resize((config.img_size, config.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def load_model(device):
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"No checkpoint at {CHECKPOINT}. Train the model first: "
            f"python -m training.train"
        )

    model = Transformer(
        config.num_classes,
        config.d_model,
        config.num_heads,
        config.d_ff,
        config.num_encoder_layers,
        (config.img_size // config.patch_size) ** 2,
        config.img_size,
        config.patch_size,
        config.in_channels,
        config.dropout,
    )

    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def image_to_tensor(path):
    """Load an image file as a model-ready batch [1, 3, img_size, img_size]."""
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0)


def sample_from_cifar(index=0):
    """Return one CIFAR-10 test image as (tensor[1,3,32,32], true_label_name)."""
    test_set = datasets.CIFAR10(str(DATA_DIR), train=False, download=True, transform=preprocess)
    image, label = test_set[index]
    return image.unsqueeze(0), CLASSES[label]


@torch.no_grad()
def classify(model, image, device):
    """image: tensor [1, 3, img_size, img_size]. Returns (label_name, confidence)."""
    image = image.to(device)
    logits = model(image)                       # [1, num_classes]
    probs = logits.softmax(dim=1)[0]            # [num_classes]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx])


if __name__ == "__main__":
    device = config.device
    model = load_model(device)

    if len(sys.argv) > 1:
        image = image_to_tensor(sys.argv[1])
        label, conf = classify(model, image, device)
        print(f"{sys.argv[1]} -> {label} ({conf:.1%})")
    else:
        image, true_label = sample_from_cifar(index=0)
        label, conf = classify(model, image, device)
        print(f"true: {true_label} | predicted: {label} ({conf:.1%})")
