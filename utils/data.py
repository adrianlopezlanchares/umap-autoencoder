import subprocess
import zipfile
from pathlib import Path

import numpy as np
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


class UMAPImageDataset(Dataset):
    def __init__(self, images: np.ndarray, umap_embeddings: np.ndarray) -> None:
        self.images = torch.tensor(images)
        self.umap_embeddings = torch.tensor(umap_embeddings)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.umap_embeddings[idx]
