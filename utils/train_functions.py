import torch

from utils.loss import UMAPAutoencoderLoss


def train_autoencoder(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> None:
    """
    Train the autoencoder model.

    Args:
        model (torch.nn.Module): The autoencoder model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    model.to(device)
    criterion.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_images)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_images.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")


def train_autoencoder_with_umap(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: UMAPAutoencoderLoss,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
) -> None:
    """
    Train the autoencoder model.

    Args:
        model (torch.nn.Module): The autoencoder model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to run the training on (CPU or GPU).
    """
    model.to(device)
    criterion.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_images, umap_targets in dataloader:
            batch_images = batch_images.to(device).float()
            umap_targets = umap_targets.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(
                reconstructed=outputs,
                original=batch_images,
                embedded=model.encoder(batch_images),
                umap_target=umap_targets,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_images.size(0)

        epoch_loss /= len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")


def test_autoencoder_reconstruction(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Test the autoencoder model.

    Args:
        model (torch.nn.Module): The autoencoder model to test.
        dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to run the testing on (CPU or GPU).

    Returns:
        float: Average reconstruction loss on the test set.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device).float()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_images)
            total_loss += loss.item() * batch_images.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    print(f"Test Reconstruction Loss: {average_loss:.4f}")
    return average_loss


def test_autoencoder_umap_embedding(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Test the autoencoder model's UMAP embedding.

    Args:
        model (torch.nn.Module): The autoencoder model to test.
        dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to run the testing on (CPU or GPU).

    Returns:
        float: Average UMAP embedding loss on the test set.
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    with torch.no_grad():
        for batch_images, umap_targets in dataloader:
            batch_images = batch_images.to(device).float()
            umap_targets = umap_targets.to(device).float()
            embeddings = model.encoder(batch_images)
            loss = criterion(embeddings, umap_targets)
            total_loss += loss.item() * batch_images.size(0)

    average_loss = total_loss / len(dataloader.dataset)
    print(f"Test UMAP Embedding Loss: {average_loss:.4f}")
    return average_loss
