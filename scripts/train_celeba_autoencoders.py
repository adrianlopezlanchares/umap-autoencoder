from pathlib import Path

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


def get_and_save_umap_embeddings(full_images, train_images, embedding_dim=1024):
    flat_full_images = full_images.reshape(full_images.shape[0], -1)
    reducer = umap.UMAP(n_components=embedding_dim)
    full_embeddings = reducer.fit_transform(flat_full_images)

    train_embeddings = full_embeddings[: len(train_images)]
    test_embeddings = full_embeddings[len(train_images) :]

    # Save full embeddings to a file
    project_dir = Path("..").resolve()
    embeddings_file = (
        project_dir
        / "data"
        / "img_align_celeba"
        / "umap_embeddings"
        / f"embeddings_d{embedding_dim}_n{len(full_images)}.npy"
    )
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_file, full_embeddings)

    return full_embeddings, train_embeddings, test_embeddings


def load_umap_embeddings(embedding_dim, num_images):
    project_dir = Path(".").resolve()
    data_dir = project_dir / "data" / "img_align_celeba" / "umap_embeddings"
    embeddings_file = data_dir / f"embeddings_d{embedding_dim}_n{num_images}.npy"

    full_embeddings = np.load(embeddings_file)

    return full_embeddings


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
    # Ejecutar el 100% en el cluster?
    train_images, test_images = get_celeba_image_np_arrays(process_size=0.1)
    full_images = np.concatenate([train_images, test_images], axis=0)

    print("Getting UMAP embeddings...")
    embedding_dim = 1024

    project_dir = Path(".").resolve()
    data_dir = project_dir / "data" / "img_align_celeba" / "umap_embeddings"
    if data_dir.exists():
        print("UMAP embeddings found, loading from file...")
        full_embeddings = load_umap_embeddings(
            embedding_dim=embedding_dim, num_images=len(full_images)
        )
        train_embeddings = full_embeddings[: len(train_images)]
        test_embeddings = full_embeddings[len(train_images) :]
    else:
        print("UMAP embeddings not found, computing and saving new embeddings...")
        full_embeddings, train_embeddings, test_embeddings = (
            get_and_save_umap_embeddings(
                full_images, train_images, embedding_dim=embedding_dim
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
    normal_num_epochs = 1
    train_autoencoder(
        model=normal_autoencoder,
        dataloader=flat_train_loader,
        criterion=normal_criterion,
        optimizer=normal_optimizer,
        num_epochs=normal_num_epochs,
        device=device,
    )

    print("Saving normal autoencoder model...")
    project_dir = Path(".").resolve()
    models_dir = project_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    normal_model_path = models_dir / "celeba_normal_autoencoder.pth"
    torch.save(normal_autoencoder.state_dict(), normal_model_path)

    print("Starting UMAP autoencoder training...")
    # UMAP AUTOENCODER TRAINING #######################################################
    umap_autoencoder = CelebAAutoencoder(
        input_dim=autoencoder_input_dim, embedding_dim=embedding_dim
    )

    umap_criterion = UMAPAutoencoderLoss(reconstruction_weight=0.5, umap_weight=0.5)
    umap_optimizer = torch.optim.Adam(umap_autoencoder.parameters(), lr=1e-3)

    # UMAP AUTOENCODER EPOCHS
    umap_num_epochs = 1
    train_autoencoder_with_umap(
        model=umap_autoencoder,
        dataloader=flat_train_loader,
        criterion=umap_criterion,
        optimizer=umap_optimizer,
        num_epochs=umap_num_epochs,
        device=device,
    )

    print("Saving UMAP autoencoder model...")
    umap_model_path = models_dir / "celeba_umap_autoencoder.pth"
    torch.save(umap_autoencoder.state_dict(), umap_model_path)

    print("All done!!")


if __name__ == "__main__":
    main()
