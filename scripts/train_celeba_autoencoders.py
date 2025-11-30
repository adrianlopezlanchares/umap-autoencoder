import torch
import numpy as np
import umap


from utils.data import (
    download_celeba_data,
    get_celeba_image_np_arrays,
    UMAPImageDataset,
)
from utils.autoencoder import CelebAAutoencoder
from utils.loss import UMAPAutoencoderLoss
from utils.train_functions import (
    train_autoencoder,
    train_autoencoder_with_umap,
)

from pathlib import Path

DGX_ROOT = Path("/app")              # inside the container
DATA_DIR = DGX_ROOT / "data"         # maps to ~/workdata/GI/project/data
MODEL_DIR = DGX_ROOT / "models"      # maps to ~/workdata/GI/project/models

IMG_DIR     = DATA_DIR / "img_align_celeba"
ORIGINAL_DIR = IMG_DIR / "original"
CLEAN_DIR    = IMG_DIR / "clean"
UMAP_DIR     = IMG_DIR / "umap_embeddings"
UMAP_ORIGINAL_DIR = UMAP_DIR / "original"
UMAP_CLEAN_DIR    = UMAP_DIR / "clean"


def get_and_save_umap_embeddings(full_images, train_images, embedding_dim=1024, image_type="original"):
    flat_full_images = full_images.reshape(full_images.shape[0], -1)
    reducer = umap.UMAP(n_components=embedding_dim)
    full_embeddings = reducer.fit_transform(flat_full_images)

    train_embeddings = full_embeddings[:len(train_images)]
    test_embeddings = full_embeddings[len(train_images):]

    # Select the right UMAP directory
    if image_type == "clean":
        embeddings_dir = UMAP_CLEAN_DIR
    else:
        embeddings_dir = UMAP_ORIGINAL_DIR

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    embeddings_file = embeddings_dir / f"{image_type}_embeddings_d{embedding_dim}_n{len(full_images)}.npy"


    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, full_embeddings)

    return full_embeddings, train_embeddings, test_embeddings


def load_umap_embeddings(embedding_dim, num_images, image_type="original"):
    if image_type == "clean":
        embeddings_dir = UMAP_CLEAN_DIR
    else:
        embeddings_dir = UMAP_ORIGINAL_DIR

    file = embeddings_dir / f"{image_type}_embeddings_d{embedding_dim}_n{num_images}.npy"

    print(f"→ Loading UMAP embeddings from {file}")

    return np.load(file)

def get_train_test_dataloaders(
    train_images,
    test_images,
    train_embeddings,
    test_embeddings,
):
    flat_train_images = train_images.reshape(train_images.shape[0], -1)
    flat_test_images = test_images.reshape(test_images.shape[0], -1)

    flat_train_dataset = UMAPImageDataset(
        images=flat_train_images, umap_embeddings=train_embeddings
    )
    flat_test_dataset = UMAPImageDataset(
        images=flat_test_images, umap_embeddings=test_embeddings
    )

    flat_train_loader = torch.utils.data.DataLoader(
        flat_train_dataset, batch_size=128, shuffle=False
    )
    flat_test_loader = torch.utils.data.DataLoader(
        flat_test_dataset, batch_size=128, shuffle=False
    )

    return flat_train_loader, flat_test_loader


def main():
    download_celeba_data()

    print("Loading CelebA images...")
    # Choose whether to use original or cleaned images
    # Choose the image type for the entire run
    image_type = "clean"      # or "original"
    print(f"\n===== RUNNING PIPELINE WITH {image_type.upper()} IMAGES =====")


    if image_type == "clean":
        imgs_root = CLEAN_DIR
    else:
        imgs_root = ORIGINAL_DIR

    train_images, test_images = get_celeba_image_np_arrays(
        process_size=0.1, 
        image_type=image_type,
    )
    full_images = np.concatenate([train_images, test_images], axis=0)

    print("Getting UMAP embeddings...")
    embedding_dim = 1024

    if image_type == "clean":
        umap_path = UMAP_CLEAN_DIR
    else:
        umap_path = UMAP_ORIGINAL_DIR

    if umap_path.exists() and len(list(umap_path.glob("*.npy"))) > 0:
        print(f"UMAP embeddings found for '{image_type}', loading...")
        full_embeddings = load_umap_embeddings(
            embedding_dim=embedding_dim,
            num_images=len(full_images),
            image_type=image_type
        )

        # ← IMPORTANT FIX: reconstruct train/test embeddings
        train_embeddings = full_embeddings[:len(train_images)]
        test_embeddings  = full_embeddings[len(train_images):]

    else:
        print(f"No UMAP embeddings found for '{image_type}', computing...")
        full_embeddings, train_embeddings, test_embeddings = (
            get_and_save_umap_embeddings(
                full_images, train_images,
                embedding_dim=embedding_dim,
                image_type=image_type
            )
        )


    print("Preparing datasets...")
    flat_train_loader, flat_test_loader = get_train_test_dataloaders(
        train_images,
        test_images,
        train_embeddings,
        test_embeddings,
    )

    print("Starting normal autoencoder training...")
    # NORMAL AUTOENCODER TRAINING #######################################################
    autoencoder_input_dim = flat_train_loader.dataset.images.shape[1]
    normal_autoencoder = CelebAAutoencoder(
        input_dim=autoencoder_input_dim, embedding_dim=embedding_dim
    )
    normal_criterion = torch.nn.MSELoss()
    normal_optimizer = torch.optim.Adam(normal_autoencoder.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else device)
    print(f"Using device: {device}")

    # NORMAL AUTOENCODER EPOCHS
    normal_num_epochs = 100
    train_autoencoder(
        model=normal_autoencoder,
        dataloader=flat_train_loader,
        criterion=normal_criterion,
        optimizer=normal_optimizer,
        num_epochs=normal_num_epochs,
        device=device,
    )

    print("Saving normal autoencoder model...")
    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    normal_model_path = model_dir / "celeba_normal_autoencoder.pth"
    torch.save(normal_autoencoder.state_dict(), normal_model_path)

    print("Starting UMAP autoencoder training...")
    # UMAP AUTOENCODER TRAINING #######################################################
    umap_autoencoder = CelebAAutoencoder(
        input_dim=autoencoder_input_dim, embedding_dim=embedding_dim
    )

    umap_criterion = UMAPAutoencoderLoss(reconstruction_weight=0.5, umap_weight=0.5)
    umap_optimizer = torch.optim.Adam(umap_autoencoder.parameters(), lr=1e-3)

    # UMAP AUTOENCODER EPOCHS
    umap_num_epochs = 100
    train_autoencoder_with_umap(
        model=umap_autoencoder,
        dataloader=flat_train_loader,
        criterion=umap_criterion,
        optimizer=umap_optimizer,
        num_epochs=umap_num_epochs,
        device=device,
    )

    print("Saving UMAP autoencoder model...")
    umap_model_path = MODEL_DIR / "celeba_umap_autoencoder.pth"
    torch.save(umap_autoencoder.state_dict(), umap_model_path)

    print("All done!!")


if __name__ == "__main__":
    main()
