from torch import nn, Tensor
import torch

class GeneratorLoss(nn.Module):
    def __init__(self, wadv=1, wcon=50, wenc=1) -> None:
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(
        self, latent_i: Tensor, latent_o: Tensor, images: Tensor, fake: Tensor, pred_real: Tensor, pred_fake: Tensor
    ) -> Tensor:
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)

        loss = error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc
        return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real: Tensor, pred_fake: Tensor) -> Tensor:

        error_discriminator_real = self.loss_bce(
            pred_real, torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device)
        )
        error_discriminator_fake = self.loss_bce(
            pred_fake, torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device)
        )
        loss_discriminator = (error_discriminator_fake + error_discriminator_real) * 0.5
        return loss_discriminator

class ClassifierLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self, anchor, positive, negative):
        dist_ap = torch.cdist(anchor, positive, p=2)
        dist_an = torch.cdist(anchor, negative, p=2)

        hardest_positive_dist, _ = dist_ap.max(dim=1)
        hardest_negative_dist, _ = dist_an.min(dim=1)

        y = torch.ones_like(hardest_negative_dist)
        loss = self.ranking_loss(hardest_negative_dist, hardest_positive_dist, y)

        return loss
