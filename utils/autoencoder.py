import torch


class MNISTAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super(MNISTAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, embedding_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim),
            torch.nn.Sigmoid(),  # Assuming input images are normalized between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CelebAAutoencoder(torch.nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int):
        super(CelebAAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 8192),
            torch.nn.ReLU(),
            torch.nn.Linear(8192, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, embedding_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 8192),
            torch.nn.ReLU(),
            torch.nn.Linear(8192, input_dim),
            torch.nn.Sigmoid(),  # Assuming input images are normalized between 0 and 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
