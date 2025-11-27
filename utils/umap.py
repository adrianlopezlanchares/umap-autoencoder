from pathlib import Path

import numpy as np
import umap
from PIL import Image


def get_umap_embeddings_of_dataset(
    embedding_dim: int = 2, process_size: float = 1.0
) -> np.ndarray:
    """
    Compute and save umap embeddings of the whole image dataset.
    The resulting embeddings are returned and saved under data/umap_embeddings/

    Args:
        embedding_dim: Dimension of the final umap embeddings
        process_size: Percentage of files to process. Between 0 and 1. Defaults to 1.

    Returns:
        np.ndarray: UMAP embeddings of the images
    """

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    imgs_dir = data_dir / "img_align_celeba" / "img_align_celeba"
    embeddings_dir = data_dir / "umap_embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    paths = sorted([p for p in imgs_dir.iterdir() if p.suffix.lower() == ".jpg"])
    n = int(len(paths) * process_size)
    paths = paths[:n]

    print("Loading images...")
    imgs = []
    i = 1
    for p in paths:
        print(f"Image {i}/{len(paths)}            ", end="\r")
        i += 1
        img = Image.open(p).convert("RGB")
        imgs.append(np.array(img, dtype=np.float32) / 255.0)
    imgs = np.stack(imgs, axis=0)  # (N, H, W, 3)
    flat_imgs = imgs.reshape(len(imgs), -1)  # (N, H*W*3)
    print(f"{len(flat_imgs)} images loaded")

    print("Getting UMAP embeddings...")
    reducer = umap.UMAP(n_components=embedding_dim)  # type: ignore
    embeddings = reducer.fit_transform(flat_imgs)

    print("Saving embeddings...")
    out_file = embeddings_dir / f"embeddings_{process_size}.npy"
    np.save(out_file, embeddings)  # type: ignore
    print(f"Embeddings saved to: {out_file}")

    return embeddings  # type: ignore


if __name__ == "__main__":
    get_umap_embeddings_of_dataset(process_size=0.1)
