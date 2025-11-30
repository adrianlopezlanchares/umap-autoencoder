import subprocess
import zipfile
from pathlib import Path
import struct
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
import shutil


# ==========================================================
# DATA DIRECTORIES
# ==========================================================
# DGX_ROOT = Path("/app")  # DGX root directory
DGX_ROOT = Path(__file__).resolve().parents[1]  # Local root directory
DATA_ROOT = DGX_ROOT / "data"
DATA_DIR = DATA_ROOT / "img_align_celeba"

ORIGINAL_DIR = DATA_DIR / "original"
CLEAN_DIR    = DATA_DIR / "clean"

# Ensure folders exist
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def download_celeba_data():
    """
    Safe version: Downloads CelebA only if not already present.
    Handles nested extracted folders correctly.
    """

    IMG_ROOT = Path("/app/data/img_align_celeba")
    ORIGINAL_DIR = IMG_ROOT / "original"

    # If images already there → skip download
    if ORIGINAL_DIR.exists() and any(ORIGINAL_DIR.glob("*.jpg")):
        print("CelebA original images already present. Skipping download.")
        return

    print("Downloading CelebA dataset from Kaggle...")

    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    ORIGINAL_DIR.mkdir(exist_ok=True)

    zip_path = IMG_ROOT / "celeba.zip"

    subprocess.run(
        [
            "curl",
            "-L",
            "-o",
            str(zip_path),
            "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset",
        ],
        check=True,
    )

    print("Unzipping...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(IMG_ROOT)

    # 🔍 FIND where the .jpg files actually are (handles nested folders)
    print("Locating extracted images...")
    candidate_dirs = [
        d for d in IMG_ROOT.rglob("*")
        if d.is_dir() and any(f.suffix.lower() == ".jpg" for f in d.iterdir())
    ]

    if not candidate_dirs:
        raise RuntimeError("Could not find any extracted .jpg files!")

    extracted_jpg_dir = candidate_dirs[0]
    print(f"Found images inside: {extracted_jpg_dir}")

    print("Moving images into 'original/' folder...")
    count = 0
    for img in extracted_jpg_dir.glob("*.jpg"):
        img.rename(ORIGINAL_DIR / img.name)
        count += 1

    print(f"Moved {count} images.")

    # Remove ONLY extraction directory (not original)
    shutil.rmtree(extracted_jpg_dir.parent, ignore_errors=True)

    zip_path.unlink()


def download_mnist_data() -> None:
    """
    Download the MNIST dataset from Kaggle to the project root, under /data.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "mnist" / "images"

    output_file = data_dir / "mnist-dataset.zip"

    if data_dir.exists():
        print("Data is already downloaded. Skipping download.")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "curl",
            "-L",
            "-o",
            str(output_file),
            "https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset",
        ],
        check=True,
    )

    print("Unzipping dataset...")
    with zipfile.ZipFile(output_file, "r") as z:
        z.extractall(data_dir)

    print("Deleting zip...")
    output_file.unlink()

    print("Done")


def get_celeba_image_np_arrays(
    process_size: float = 0.01,
    train_size: float = 0.8,
    image_type: str = "clean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from the CelebA dataset and return them as a numpy array.
    The images are resized to 64x64 pixels.

    Args:
        process_size: Percentage of files to process. Between 0 and 1. Defaults to 0.01.
        train_size: Percentage of images to use for training. Between 0 and 1. Defaults to 0.8.
        image_type: Type of images to load, either "original" or "clean". Defaults to "clean".

    Returns:
        np.ndarray: Array of images with shape (N, 64, 64, 3)
    """
    # === Select directory based on image_type ===
    if image_type not in ["original", "clean"]:
        raise ValueError("image_type must be 'original' or 'clean'")

    if image_type == "clean":
        imgs_dir = CLEAN_DIR
        print("→ Using CLEAN images (background removed)")
    else:
        imgs_dir = ORIGINAL_DIR
        print("→ Using ORIGINAL images")

    # === Read image list ===
    paths = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() == ".jpg"])

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {imgs_dir}. Did you download/clean them? Run 'celeba_clean_background.py' first.")

    print(f"→ Found {len(paths)} '{image_type}' images in: {imgs_dir}")

    # === Apply process_size (subsample) ===
    n = int(len(paths) * process_size)
    paths = paths[:n]
    print(f"→ Processing {len(paths)} images (process_size = {process_size})")

    # === Load and resize ===
    imgs = []
    for i, p in enumerate(paths, start=1):
        print(f"Loading image {i}/{len(paths)}          ", end="\r")
        img = Image.open(p).convert("RGB").resize((64, 64))
        imgs.append(np.array(img, dtype=np.float32) / 255.0)

    imgs = np.stack(imgs, axis=0)
    print(f"\n→ Loaded {imgs.shape[0]} images.")

    # === Train/Test split ===
    n_train = int(len(imgs) * train_size)
    train_imgs = imgs[:n_train]
    test_imgs = imgs[n_train:]

    print(f"→ Split into {len(train_imgs)} train and {len(test_imgs)} test images.\n")

    return train_imgs, test_imgs


def get_mnist_image_np_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from the MNIST dataset and return them as a numpy array.
    The images are resized to 28x28 pixels.

    Returns:
        np.ndarray: Array of images with shape (N, 28, 28)
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "mnist" / "images"
    data_dir.mkdir(exist_ok=True)
    train_images_dir = data_dir / "train-images-idx3-ubyte" / "train-images-idx3-ubyte"
    test_images_dir = data_dir / "t10k-images-idx3-ubyte" / "t10k-images-idx3-ubyte"

    with train_images_dir.open("rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}, expected 2051")

        train_images = np.frombuffer(f.read(), dtype=np.uint8)
        train_images = train_images.reshape(num_images, rows, cols)

    with test_images_dir.open("rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}, expected 2051")

        test_images = np.frombuffer(f.read(), dtype=np.uint8)
        test_images = test_images.reshape(num_images, rows, cols)

    # Normalize images to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    return train_images, test_images


def get_mnnist_labels_np_arrays() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load labels from the MNIST dataset and return them as a numpy array.

    Returns:
        np.ndarray: Array of labels with shape (N,)
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "mnist" / "images"
    data_dir.mkdir(exist_ok=True)
    train_labels_dir = data_dir / "train-labels-idx1-ubyte" / "train-labels-idx1-ubyte"
    test_labels_dir = data_dir / "t10k-labels-idx1-ubyte" / "t10k-labels-idx1-ubyte"

    with train_labels_dir.open("rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}, expected 2049")

        train_labels = np.frombuffer(f.read(), dtype=np.uint8)

    with test_labels_dir.open("rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}, expected 2049")

        test_labels = np.frombuffer(f.read(), dtype=np.uint8)

    # Turn labels from numbers into one-hot encoded vectors
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]

    return train_labels, test_labels


def get_cifar10_image_np_arrays(train: bool = True) -> np.ndarray:
    """
    Load images from the CIFAR-10 dataset and return them as a numpy array.
    The images are 32x32x3.

    Args:
        train: Whether to load training or test data.

    Returns:
        np.ndarray: Array of images with shape (N, 32, 32, 3) depending on format,
                    normalized to [0, 1].
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # CIFAR10 dataset
    dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=train,
        download=True,
        transform=None,  # We want raw numpy arrays first
    )

    # dataset.data is already a numpy array of shape (N, 32, 32, 3)
    images = dataset.data.astype(np.float32) / 255.0

    return images


def get_cifar10_labels_np_arrays(train: bool = True) -> np.ndarray:
    """
    Load labels from the CIFAR-10 dataset and return them as a numpy array (one-hot).

    Args:
        train: Whether to load training or test data.

    Returns:
        np.ndarray: Array of labels with shape (N, 10)
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(root=str(data_dir), train=train, download=True)

    labels = np.array(dataset.targets)

    # One-hot encode
    labels = np.eye(10)[labels]

    return labels


def get_fashion_mnist_image_np_arrays(train: bool = True) -> np.ndarray:
    """
    Load images from the Fashion MNIST dataset and return them as a numpy array.
    The images are 28x28.

    Args:
        train: Whether to load training or test data.

    Returns:
        np.ndarray: Array of images with shape (N, 28, 28),
                    normalized to [0, 1].
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # FashionMNIST dataset
    dataset = datasets.FashionMNIST(
        root=str(data_dir), train=train, download=True, transform=None
    )

    # dataset.data is a tensor of shape (N, 28, 28)
    images = dataset.data.numpy().astype(np.float32) / 255.0

    return images


def get_fashion_mnist_labels_np_arrays(train: bool = True) -> np.ndarray:
    """
    Load labels from the Fashion MNIST dataset and return them as a numpy array (one-hot).

    Args:
        train: Whether to load training or test data.

    Returns:
        np.ndarray: Array of labels with shape (N, 10)
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.FashionMNIST(root=str(data_dir), train=train, download=True)

    labels = dataset.targets.numpy()

    # One-hot encode
    labels = np.eye(10)[labels]

    return labels


class UMAPImageDataset(Dataset):
    def __init__(self, images: np.ndarray, umap_embeddings: np.ndarray) -> None:
        # images expected to be (N, H, W, C) or (N, C, H, W)
        # We'll convert to tensor. If it's (N, H, W, C), PyTorch usually expects (N, C, H, W).
        # We will handle permutation in __getitem__ or init if needed.
        self.images = torch.tensor(images)
        if self.images.ndim == 4 and self.images.shape[-1] == 3:
            # (N, H, W, C) -> (N, C, H, W)
            self.images = self.images.permute(0, 3, 1, 2)

        self.umap_embeddings = torch.tensor(umap_embeddings)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.umap_embeddings[idx]


class ClassificationDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.tensor(images)
        if self.images.ndim == 4 and self.images.shape[-1] == 3:
            # (N, H, W, C) -> (N, C, H, W)
            self.images = self.images.permute(0, 3, 1, 2)

        self.labels = torch.tensor(labels)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
