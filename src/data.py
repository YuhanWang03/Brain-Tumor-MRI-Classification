from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .prepare_data import ensure_dataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_data_dirs(data_root: str = "data"):
    root = ensure_dataset(data_root=data_root)

    training_dir = Path(root) / "Training"
    testing_dir = Path(root) / "Testing"

    if not training_dir.exists() or not testing_dir.exists():
        raise FileNotFoundError(
            "Cannot find dataset folders after local check / download."
        )

    return training_dir, testing_dir


def build_transforms(img_size: int = 256):
    """
    Build train/eval transforms.
    """
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.25, 0.25),
            shear=20,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_tfms, eval_tfms


def create_dataloaders(
    data_root: str = "data",
    img_size: int = 256,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    device: torch.device | None = None,
):
    """
    Create train/val/test dataloaders from the brain tumor MRI dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_dir, testing_dir = resolve_data_dirs(data_root=data_root)
    train_tfms, eval_tfms = build_transforms(img_size=img_size)

    full_train = ImageFolder(training_dir, transform=train_tfms)
    test_set = ImageFolder(testing_dir, transform=eval_tfms)

    class_names = full_train.classes
    class_to_idx = full_train.class_to_idx

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_set, val_subset = random_split(
        full_train,
        [n_train, n_val],
        generator=generator
    )

    # Validation set should not use augmentation
    val_base = ImageFolder(training_dir, transform=eval_tfms)
    val_set = Subset(val_base, val_subset.indices)

    pin_memory = (device.type == "cuda")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "training_dir": training_dir,
        "testing_dir": testing_dir,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "train_tfms": train_tfms,
        "eval_tfms": eval_tfms,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
    }
