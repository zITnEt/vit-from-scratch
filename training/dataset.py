from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import config, DATA_DIR, CIFAR_MEAN, CIFAR_STD


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(config.img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def eval_transform():
    """Deterministic transform for validation/inference (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def build_loaders():
    """CIFAR-10 train/test DataLoaders with light train-set augmentation."""
    train_set = datasets.CIFAR10(str(DATA_DIR), train=True, download=True,
                                 transform=train_transform())
    test_set = datasets.CIFAR10(str(DATA_DIR), train=False, download=True,
                                transform=eval_transform())

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True)
    return train_loader, test_loader
