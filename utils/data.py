import subprocess
import zipfile
from pathlib import Path
import struct
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def download_celeba_data() -> None:
    """
    Download the CelebA dataset from Kaggle to the project root, under /data.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    img_folder = data_dir / "img_align_celeba"
    output_file = data_dir / "celeba-dataset.zip"

    if img_folder.exists():
        print("Data is already downloaded. Skipping download.")
        return

    subprocess.run(
        [
            "curl",
            "-L",
            "-o",
            str(output_file),
            "https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset",
        ],
        check=True,
    )

    print("Unzipping dataset...")
    with zipfile.ZipFile(output_file, "r") as z:
        z.extractall(data_dir)

    print("Deleting zip and unused files...")
    output_file.unlink()
    # remove list_attr_celeba.csv, list_bbox_celeba.csv, list_eval_partition.csv and list_landmarks_align_celeba.csv
    for file_name in [
        "list_attr_celeba.csv",
        "list_bbox_celeba.csv",
        "list_eval_partition.csv",
        "list_landmarks_align_celeba.csv",
    ]:
        file_path = data_dir / file_name
        if file_path.exists():
            file_path.unlink()

    print("Done")


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
    process_size: float = 0.01, train_size: float = 0.8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images from the CelebA dataset and return them as a numpy array.
    The images are resized to 64x64 pixels.

    Args:
        process_size: Percentage of files to process. Between 0 and 1. Defaults to 0.01.

    Returns:
        np.ndarray: Array of images with shape (N, 64, 64, 3)
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    imgs_dir = data_dir / "img_align_celeba" / "img_align_celeba"
    paths = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() == ".jpg"])
    n = int(len(paths) * process_size)
    paths = paths[:n]

    print("Loading images...")
    imgs = []
    i = 1
    for p in paths:
        print(f"Image {i}/{len(paths)}            ", end="\r")
        i += 1

        img = Image.open(p).convert("RGB").resize((64, 64))
        imgs.append(np.array(img, dtype=np.float32) / 255.0)

    imgs = np.stack(imgs, axis=0)  # (N, H, W, 3)
    print(f"{len(imgs)} images loaded")

    n_train = int(len(imgs) * train_size)
    train_imgs = imgs[:n_train]
    test_imgs = imgs[n_train:]

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


class UMAPImageDataset(Dataset):
    def __init__(self, images: np.ndarray, umap_embeddings: np.ndarray) -> None:
        self.images = torch.tensor(images)
        self.umap_embeddings = torch.tensor(umap_embeddings)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.umap_embeddings[idx]
