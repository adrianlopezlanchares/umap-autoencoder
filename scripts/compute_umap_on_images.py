from pathlib import Path
import numpy as np
import umap
from utils.data import get_celeba_image_np_arrays

def compute_umap_on_images(
    dataset: str = "celeba",
    process_size: float = 0.01,
    embedding_dim: int = 2,
    image_type: str = "original"
) -> None:
    """
    Compute UMAP embeddings on CelebA (original or clean) and save them into:
    /app/data/img_align_celeba/umap_embeddings/{image_type}/
    """

    if dataset == "celeba":
        dataset = "img_align_celeba"

    # === Load ORIGINAL or CLEAN images ===
    train_imgs, test_imgs = get_celeba_image_np_arrays(
        process_size=process_size,
        image_type=image_type
    )

    full_imgs = np.concatenate([train_imgs, test_imgs], axis=0)
    flat_imgs = full_imgs.reshape(full_imgs.shape[0], -1)

    # === Run UMAP ===
    reducer = umap.UMAP(n_components=embedding_dim)
    embeddings = reducer.fit_transform(flat_imgs)

    # === Save inside correct persistent directory ===
    data_root = Path("/app/data")
    base_dir = data_root / dataset / "umap_embeddings" / image_type
    base_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{image_type}_embeddings_d{embedding_dim}_n{len(embeddings)}.npy"
    embeddings_path = base_dir / file_name

    np.save(embeddings_path, embeddings)

    print(f"UMAP embeddings ({image_type}) saved to: {embeddings_path}")
