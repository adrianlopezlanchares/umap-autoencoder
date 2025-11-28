import torch


class UMAPAutoencoderLoss(torch.nn.Module):
    def __init__(self, reconstruction_weight: float = 1.0, umap_weight: float = 1.0):
        super(UMAPAutoencoderLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.umap_weight = umap_weight
        self.reconstruction_loss_fn = torch.nn.MSELoss()
        # UMAP loss can be implemented as needed; placeholder here
        self.umap_loss_fn = torch.nn.MSELoss()  # Placeholder

    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        embedded: torch.Tensor,
        umap_target: torch.Tensor,
    ) -> torch.Tensor:
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed, original)
        umap_loss = self.umap_loss_fn(embedded, umap_target)
        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.umap_weight * umap_loss
        )
        return total_loss
