from __future__ import annotations
from .torch_model import DiscriminatorClassifier, Ganomaly1d
from anomalib.models.components import AnomalyModule
from .generator import AnomalyGenerator
from .loss import GeneratorLoss, DiscriminatorLoss, ClassifierLoss
from torch import optim, Tensor
from typing import Union
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import logging

logger = logging.getLogger(__name__)

class AFT(AnomalyModule):
    def __init__(
        self,
        batch_size,
        input_size,
        n_feature,
        latent_vec_size,
        add_final_conv_layer: bool = True,
        wadv: int = 1,
        wcon: int = 50,
        wenc: int = 1,
        wt: float = 0.5,
        wf: float = 0.5,
        lr: int = 0.0002,
        lr_c: int = 0.001,
        beta1: int = 0.5,
        beta2: int = 0.999,
        mean: float = 0.0,
        std: float = 0.1,
        margin: float = 1.0,
        dataset_name: str = ""
    ):
        super().__init__()

        self.model: Ganomaly1d = Ganomaly1d(
            input_size=input_size,
            num_input_channel=1,
            n_feature=n_feature,
            latent_vec_size=latent_vec_size,
            add_final_conv_layer=add_final_conv_layer,
        ).cuda()
        checkpoint = torch.load(f"./results/aft/{dataset_name}/run/weights/model-ganomaly.ckpt")

        state_dict = checkpoint['state_dict']

        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[len('model.'):]
                new_state_dict[new_key] = value

        # 加载到模型中
        self.model.load_state_dict(new_state_dict)

        self.classifier: DiscriminatorClassifier = DiscriminatorClassifier(
            latent_vec_size=latent_vec_size
        )

        self.real_label = torch.ones(size=(batch_size,), dtype=torch.float32)
        self.fake_label = torch.zeros(size=(batch_size,), dtype=torch.float32)

        self.min_scores: Tensor = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores: Tensor = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable

        self.generator_loss = GeneratorLoss(wadv=wadv, wcon=wcon, wenc=wenc)
        self.discriminator_loss = DiscriminatorLoss()
        self.classifierloss = ClassifierLoss(margin=margin)

        self.learning_rate = lr
        self.lr_c = lr_c
        self.beta1 = beta1
        self.beta2 = beta2
        self.wt = wt
        self.wf = wf
        self.mean = mean
        self.std = std

    def _reset_min_max(self) -> None:
        """Resets min_max scores."""
        self.min_scores = torch.tensor(float("inf"), dtype=torch.float32)  # pylint: disable=not-callable
        self.max_scores = torch.tensor(float("-inf"), dtype=torch.float32)  # pylint: disable=not-callable


    def configure_optimizers(self):
        optimizer_c = optim.Adam(
            self.classifier.parameters(),
            lr=self.lr_c,
            weight_decay=1e-5
        )

        # optimizer_d = optim.Adam(
        #     self.model.discriminator.parameters(),
        #     lr=self.learning_rate,
        #     betas=(self.beta1, self.beta2),
        # )
        #
        # optimizer_g = optim.Adam(
        #     self.model.generator.parameters(),
        #     lr=self.learning_rate,
        #     betas=(self.beta1, self.beta2),
        # )

        return optimizer_c
        # return [optimizer_d, optimizer_g]


    def training_step(self, batch):
        # [64, 1600]
        self.model.eval()
        latent_i, latent_o, re_batch, gen_image = self.model(batch["image"].cuda())  # [64, 70, 49], [64, 1, 1600]
        latent_fake = AnomalyGenerator(latent=latent_i, mean=self.mean, std=self.std, wt=self.wt, wf=self.wf)

        anchor = self.classifier(latent_i)
        positive = self.classifier(latent_o)
        negative = self.classifier(latent_fake)

        anchor = anchor.squeeze()
        positive = positive.squeeze()
        negative = negative.squeeze()

        loss = self.classifierloss(anchor=anchor, positive=positive, negative=negative)
        self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

        # latent_i, latent_o, re_batch, gen_image = self.model(batch["image"].cuda())
        # pred_real, _ = self.model.discriminator(re_batch)
        #
        # if optimizer_idx == 0:  # Discriminator
        #     pred_fake, _ = self.model.discriminator(gen_image.detach())
        #     loss = self.discriminator_loss(pred_real, pred_fake)
        # else:  # Generator
        #     pred_fake, _ = self.model.discriminator(gen_image)
        #     loss = self.generator_loss(latent_i, latent_o, re_batch, gen_image, pred_real, pred_fake)
        #
        # self.log("train_loss", loss.item(), on_epoch=True, prog_bar=True, logger=True)
        # return {"loss": loss}

    def on_validation_start(self) -> None:
        """Reset min and max values for current validation epoch."""
        self._reset_min_max()
        return super().on_validation_start()

    def validation_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        latent_i, latent_o, re_batch, gen_image = self.model(batch["image"].cuda())

        embedding_i = self.classifier(latent_i)
        embedding_o = self.classifier(latent_o)

        batch["pred_scores"] = torch.mean(torch.pow((embedding_i - embedding_o), 2), dim=1).view(-1)

        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        # batch["pred_scores"] = self.model(batch["image"].cuda())
        # self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        # self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))

        return batch

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing validation outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().validation_epoch_end(outputs)
        return outputs

    def on_test_start(self) -> None:
        """Reset min max values before test batch starts."""
        self._reset_min_max()
        return super().on_test_start()

    def test_step(self, batch: dict[str, str | Tensor], batch_idx: int, *args, **kwargs) -> STEP_OUTPUT:
        """Update min and max scores from the current step."""
        super().test_step(batch, batch_idx)
        self.max_scores = max(self.max_scores, torch.max(batch["pred_scores"]))
        self.min_scores = min(self.min_scores, torch.min(batch["pred_scores"]))
        return batch

    def test_epoch_end(self, outputs: EPOCH_OUTPUT) -> EPOCH_OUTPUT:
        """Normalize outputs based on min/max values."""
        logger.info("Normalizing test outputs based on min/max values.")
        for prediction in outputs:
            prediction["pred_scores"] = self._normalize(prediction["pred_scores"])
        super().test_epoch_end(outputs)
        return outputs

    def _normalize(self, scores: Tensor) -> Tensor:
        """Normalize the scores based on min/max of entire dataset.

        Args:
            scores (Tensor): Un-normalized scores.

        Returns:
            Tensor: Normalized scores.
        """
        scores = (scores - self.min_scores.to(scores.device)) / (
            self.max_scores.to(scores.device) - self.min_scores.to(scores.device)
        )
        return scores

class AftLightning(AFT):
    def __init__(
        self, 
        hparams,
    ) -> None:
        super().__init__(
            batch_size=hparams.dataset.train_batch_size,
            input_size=hparams.model.input_size[0], # 1600
            n_feature=hparams.model.n_features, # 16
            latent_vec_size = hparams.model.latent_vec_size,    # 70
            wadv=hparams.model.wadv,
            wcon=hparams.model.wcon,
            wenc=hparams.model.wenc,
            lr=hparams.model.lr,
            lr_c=hparams.model.lr_c,
            beta1=hparams.model.beta1, 
            beta2=hparams.model.beta2,
            mean=hparams.model.mean,
            std=hparams.model.std,
            dataset_name=hparams.dataset.name,
            margin=hparams.model.margin,
            wt=hparams.model.wt,
            wf=hparams.model.wf
        )
        self.hparams: Union[DictConfig,ListConfig]
        self.save_hyperparameters(hparams)

    def configure_callbacks(self):
        early_stopping = EarlyStopping(
            monitor=self.hparams.model.early_stopping.metric,
            patience=self.hparams.model.early_stopping.patience,
            mode=self.hparams.model.early_stopping.mode,
        )
        return [early_stopping]