from pathlib import Path

import umap
import numpy as np

from utils.data import get_image_np_arrays


def compute_umap_on_images(process_size: float = 0.01, embedding_dim: int = 2) -> None:
    """
    Compute UMAP embeddings on images from the CelebA dataset. Saves the embeddings as a .npy file.

    Args:
        process_size: Percentage of files to process. Between 0 and 1. Defaults to 0.01.
        embedding_dim: Dimension of the UMAP embeddings. Defaults to 2.
    """
    images = get_image_np_arrays(process_size=process_size)

    flat_images = images.reshape(images.shape[0], -1)
    reducer = umap.UMAP(n_components=embedding_dim)
    embeddings = reducer.fit_transform(flat_images, verbose=True)

    # Save embeddings to a file
    project_dir = Path("..").resolve()
    embeddings_file = (
        project_dir
        / "data"
        / "umap_embeddings"
        / f"embeddings_d{embedding_dim}_n{len(embeddings)}.npy"
    )
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, embeddings)
