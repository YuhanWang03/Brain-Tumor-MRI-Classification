from pathlib import Path
import shutil
import subprocess

DATASET_SLUG = "masoudnickparvar/brain-tumor-mri-dataset"


def _find_nested_dataset_root(root: Path):
    for training_path in root.rglob("Training"):
        candidate_root = training_path.parent
        if (candidate_root / "Testing").exists():
            return candidate_root
    return None


def ensure_dataset(data_root: str = "data"):
    data_dir = Path(data_root)

    train_dir = data_dir / "Training"
    test_dir = data_dir / "Testing"

    # Case 1: already in expected location
    if train_dir.exists() and test_dir.exists():
        print(f"Dataset found: {data_dir.resolve()}")
        return data_dir

    # Case 2: nested folder exists
    nested_root = _find_nested_dataset_root(data_dir) if data_dir.exists() else None
    if nested_root is not None:
        print(f"Dataset found in nested folder: {nested_root.resolve()}")
        return nested_root

    # Case 3: download from Kaggle
    data_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install it with `pip install kaggle`, "
            "then configure your Kaggle API credentials."
        )

    print("Dataset not found locally. Downloading from Kaggle...")
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "--path",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )

    if train_dir.exists() and test_dir.exists():
        print(f"Dataset downloaded to: {data_dir.resolve()}")
        return data_dir

    nested_root = _find_nested_dataset_root(data_dir)
    if nested_root is not None:
        print(f"Dataset downloaded to nested folder: {nested_root.resolve()}")
        return nested_root

    raise FileNotFoundError(
        "Download finished, but Training/Testing folders were not found."
    )
