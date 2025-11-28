import subprocess
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def download_data() -> None:
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

    print("Deleting zip...")
    output_file.unlink()

    print("Done")


def get_image_np_arrays(process_size: float = 0.01) -> np.ndarray:
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

    return imgs


class UMAPImageDataset(Dataset):
    def __init__(self, images: np.ndarray, umap_embeddings: np.ndarray) -> None:
        self.images = torch.tensor(images)
        self.umap_embeddings = torch.tensor(umap_embeddings)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.umap_embeddings[idx]
