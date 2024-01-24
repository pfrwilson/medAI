from medsam_cancer_detection_corewise_simple import get_dataloaders
import torch
from torch import nn
from segment_anything.modeling.common import LayerNorm2d
from medAI.utils.masking_generator import MaskingGenerator
from copy import deepcopy
import wandb
from tqdm import tqdm
from pathlib import Path
from medAI.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
import os
from rich_argparse import RichHelpFormatter, ArgumentDefaultsRichHelpFormatter
import typing as tp
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import einops
import matplotlib.pyplot as plt
import typing as tp 
from simple_parsing import ArgumentParser, subgroups
from einops.layers.torch import Rearrange
import hydra


torch.autograd.set_detect_anomaly(True)


@hydra.main(config_path="conf", config_name="config")
def main(config):
    logging.basicConfig(level=logging.INFO)
    logging.info("===========================================")
    logging.info("Starting experiment")
    logging.info("===========================================")

    logging.info(f"Config: {config}")

    if config.debug: 
        global debug
        debug = True

    state = None
    if config.checkpoint_dir is not None:
        config.checkpoint_dir.mkdir(exist_ok=False, parents=True)
        if "experiment.ckpt" in os.listdir(config.checkpoint_dir):
            logging.info(f"Loading from checkpoint found in {config.checkpoint_dir}")
            state = torch.load(config.checkpoint_dir / "experiment.ckpt")

    logging.info("Loading data")
    train_loader, val_loader, test_loader = get_dataloaders(
        config.fold,
        config.n_folds,
        config.benign_to_cancer_ratio,
        debug=config.debug,
        augmentation=None,
    )
    logging.info(
        f"Train: {len(train_loader)} batches, {len(train_loader.dataset)} images"
    )
    logging.info(f"Val: {len(val_loader)} batches, {len(val_loader.dataset)} images")
    logging.info(f"Test: {len(test_loader)} batches, {len(test_loader.dataset)} images")

    logging.info("Building model")
    if isinstance(config.ssl_model, IBotStyleModelConfig):
        model = IBotStyleModel.from_config(config.ssl_model)
    elif isinstance(config.ssl_model, SimMIMModelConfig):
        model = SimMIMModel.from_config(config.ssl_model)

    torch.compile(model)
    model.cuda()
    logging.info(f"Model <{model.__class__.__name__}> built: ")
    logging.info(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    logging.info(
        f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if state is not None:
        model.load_state_dict(state["model"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        5 * len(train_loader),
        config.epochs * len(train_loader),
        warmup_start_lr=1e-9,
        eta_min=1e-7,
    )

    if state is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    wandb.init(
        project="sam_ssl_pretraining",
        config=config,
        dir=config.checkpoint_dir,
        id=state["wandb_id"] if state is not None else None,
    )
    if "SLURM_JOB_ID" in os.environ:
        wandb.config.update({"SLURM_JOB_ID": os.environ["SLURM_JOB_ID"]})

    wandb.watch(model, log_freq=100)

    start_epoch = state["epoch"] if state is not None else 0

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch}")

        if config.checkpoint_dir is not None:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "wandb_id": wandb.run.id,
                    "epoch": epoch,
                },
                config.checkpoint_dir / "experiment.ckpt",
            )

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            if i % config.log_image_every == 0:
                # show some examples of features PCA as a sanity check
                fig = model.show_example(batch[0].cuda())
                wandb.log(
                    {
                        "example": wandb.Image(
                            fig, caption="PCA Decomposition of a features"
                        ),
                        "epoch": epoch,
                    }
                )
                plt.close(fig)

            optimizer.zero_grad()
            loss = model(batch[0].cuda())
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            model.on_train_step_end()
            wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], 'epoch': epoch})

        if config.checkpoint_dir is not None:
            torch.save(
                model.student.encoder.state_dict(),
                config.checkpoint_dir / f"encoder_{epoch}.pth",
            )


def medsam_image_encoder():
    from segment_anything import sam_model_registry

    sam_model = sam_model_registry["vit_b"](
        checkpoint="/ssd005/projects/exactvu_pca/checkpoint_store/medsam_vit_b_cpu.pth"
    )
    image_encoder = sam_model.image_encoder
    return image_encoder


@torch.no_grad()
def do_ema_update(teacher, student, alpha=0.999):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(1 - alpha, student_param.data)


class SinkhornKnoppCenteringModule(nn.Module):
    """Does the sinkhorn-knopp algorithm for optimal transport.

    Details of the sinkhorn-knopp algorithm can be found in the method
    ``sinkhorn`` below. This module is a wrapper around that method that
    allows us to do the sinkhorn-knopp algorithm on minibatches of scores,
    while maintaining a queue of the old minibatch scores, in order to prevent the
    chance of too small minibatches.
    """

    def __init__(
        self,
        use_cache=False,
        cache_size: int = 1000,
        eps: float = 0.05,
        niters: int = 3,
        n_classes=1024,
    ):
        super().__init__()
        self.cache_size = cache_size
        self.eps = eps
        self.niters = niters
        self.use_cache = use_cache

        self.register_buffer("queue", torch.zeros(cache_size, n_classes))
        self.register_buffer("queue_size", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def forward(self, scores):
        B, N = scores.shape

        if self.use_cache:
            # roll the queue elements to make room
            torch.roll(self.queue, shifts=B, dims=0)
            # overwrite the oldest elements with the new scores in the first rows
            self.queue[:B] = scores
            # do the sinkhorn-knopp algorithm using the whole queue
            self.queue_size[:] = min(self.queue_size + B, self.cache_size)
            scores = self.queue[: self.queue_size[0]]
            Q = self.sinkhorn(scores, eps=self.eps, niters=self.niters)
            # return the first rows of the queue, which are the scores for the current batch
            return Q[:B]

        else:
            return self.sinkhorn(scores, eps=self.eps, niters=self.niters)

    def sinkhorn(self, scores, eps=0.05, niters=3):
        """Does the sinkhorn-knopp algorithm for optimal transport.

        Informally, it receives a NxM matrix of "scores". N is the minibatch
        dimension and M is the number of tokens. The i,j'th entry of the matrix
        tells us the "score" for how much the i'th minibatch element should be
        assigned to the j'th token. In principle this kind of input could strongly
        prefer one token over all others, but the sinkhorn-knopp algorithm will
        prevent this from happening by ensuring each row and column of the matrix is
        assigned more uniformly. In particular, each token has to be assigned to
        the same number of minibatch elements, and each minibatch element has to be
        assigned to the same number of tokens.

        Args:
            scores (torch.Tensor): NxM matrix of scores
            eps (float, optional): Epsilon for the sinkhorn-knopp algorithm. Defaults to 0.05.
            niters (int, optional): Number of iterations to run the sinkhorn-knopp algorithm. Defaults to 3.
        """
        Q = torch.exp(scores / eps).T
        Q /= torch.sum(Q)
        K, B = Q.shape
        u, r, c = (
            torch.zeros(K, device=scores.device),
            torch.ones(K, device=scores.device) / K,
            torch.ones(B, device=scores.device) / B,
        )
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).T


class SelfSupervisedEncoderTrainer(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.image_encoder = medsam_image_encoder()

    @torch.no_grad()
    def show_example(self, image):
        """Shows an example of the image and the PCA of the features"""
        image_features = self.image_encoder(image)
        B, C, H, W = image_features.shape

        # make the features into a matrix.
        image_features = einops.rearrange(image_features, "b c h w -> (b h w) c")

        # compute the pca of the features
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(image_features.cpu().numpy())
        # make PCA features between 0 and 1

        N, C = features_pca.shape
        assert C == 3, "pca should have 3 components"
        features_pca = einops.rearrange(
            features_pca, "(b h w) c -> b h w c", b=B, h=H, w=W
        )
        image = einops.rearrange(image, "b c h w -> b h w c").cpu().numpy()

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(image[0])
        ax[1].imshow(features_pca[0])

        return fig

    @abstractmethod
    def forward(self, image):
        """Returns the loss for the batch of images (B, C, H, W)"""

    def from_config(self, config):
        """Returns a new instance of this model from the given config"""
        raise NotImplementedError()

    def on_train_step_end(self):
        """Called at the end of each training step"""
    

class IBotStyleModel(SelfSupervisedEncoderTrainer):
    """Runs IBot-style SSL training on the medsam backbone.

    IBot does student-teacher training, with the student receiving masked
    inputs and the teacher receiving unmasked inputs. The student is trained
    to predict the correct

    enc
    """

    def __init__(
        self,
        proj_dims=512,
        num_classes=1024,
        feature_map_size=64,
        min_num_patches=16,
        max_num_patches=100,
        mask_ratio=0.3,
        ema_alpha=0.999,
        sinkhorn_knopp_centering_mode: tp.Literal["in_mask", "whole_image"] = "in_mask",
        token_matching_mode: tp.Literal["hard", "soft"] = "hard",
        temp=1,
    ):
        super().__init__()
        self.mask_gen = MaskingGenerator(
            (feature_map_size, feature_map_size),
            int(feature_map_size * feature_map_size * mask_ratio),
            min_num_patches=min_num_patches,
            max_num_patches=max_num_patches,
        )

        # projection head for "token class prediction"
        self.proj = nn.Sequential(
            nn.Conv2d(256, proj_dims, 1),
            LayerNorm2d(proj_dims),
            nn.Conv2d(512, num_classes, 1),
        )
        self.student = MaskableEncoderWithProjection(
            encoder=self.image_encoder, projector=self.proj, transformer_dim=768
        )

        self.teacher = deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad_(False)

        self.ema_alpha = ema_alpha
        self.temp = temp
        self.sinkhorn_knopp_centering_mode = sinkhorn_knopp_centering_mode
        self.token_matching_mode = token_matching_mode

        self.sinkhorn = SinkhornKnoppCenteringModule(n_classes=num_classes)

    def generate_masks(self, image):
        masks = []
        for _ in range(image.shape[0]):
            masks.append(torch.from_numpy(self.mask_gen()).bool().to(image.device))
        return torch.stack(masks)

    def forward(self, image):
        mask = self.generate_masks(image)
        with torch.no_grad():
            target_token_scores = self.teacher(
                image, mask=None
            )  # B, self.num_classes, 64, 64
            target_token_scores = target_token_scores.permute(
                0, 2, 3, 1
            )  # B, 64, 64, self.num_classes

            if self.sinkhorn_knopp_centering_mode == "in_mask":
                # mask out the tokens that are not in the mask - the output is of shape
                # N, self.num_classes where N is sum of number of tokens in each mask across
                # the whole batch. Note that this means we are centering the sinkhorn-knopp
                # algorithm on the tokens that are actually in the mask, and across the batch.
                target_token_scores = target_token_scores[mask]
                target_token_scores = self.sinkhorn(target_token_scores)
                target_dist = (target_token_scores / self.temp).softmax(-1)

            elif self.sinkhorn_knopp_centering_mode == "whole_image":
                # this time, we should do the sinkhorn knopp centering first. We
                # need to collapse the batch and spatial dimensions to do this.
                B, H, W, C = target_token_scores.shape
                B1, H1, W1 = mask.shape
                assert (
                    B == B1 and H == H1 and W == W1
                ), "mask shape must match image shape"
                target_token_scores = einops.rearrange(
                    target_token_scores, "b h w c -> (b h w) c"
                )
                mask = einops.rearrange(mask, "b h w -> (b h w)")
                target_token_scores = self.sinkhorn(target_token_scores)
                target_dist = (target_token_scores / self.temp).softmax(-1)
                target_dist = target_dist[mask]

            else:
                raise ValueError(
                    f"Unknown sinkhorn-knopp centering mode {self.sinkhorn_knopp_centering_mode}"
                )

        # now compute the student token scores
        student_token_scores = self.student(image, mask=mask)
        student_token_scores = student_token_scores.permute(0, 2, 3, 1)
        student_token_scores = student_token_scores[mask]

        if self.token_matching_mode == "soft":
            # compute the cross entropy loss between the student and teacher token scores
            loss = torch.sum(
                -target_dist * torch.log_softmax(student_token_scores, dim=-1), dim=-1
            ).mean()
        elif self.token_matching_mode == "hard":
            # discretize the target distribution to integer labels
            target_labels = torch.argmax(target_dist, dim=-1)
            loss = torch.nn.functional.cross_entropy(
                student_token_scores, target_labels
            )
        else:
            raise ValueError(f"Unknown token matching mode {self.token_matching_mode}")

        return loss

    def on_train_step_end(self):
        self.ema_update()

    def ema_update(self):
        do_ema_update(self.teacher, self.student, self.ema_alpha)

    @staticmethod
    def from_config(config: IBotStyleModelConfig):
        return IBotStyleModel(
            ema_alpha=config.ema_alpha,
            temp=config.temp,
            sinkhorn_knopp_centering_mode=config.sinkhorn_knopp_centering_mode,
            token_matching_mode=config.token_matching_mode,
        )


class SimMIMModel(SelfSupervisedEncoderTrainer):
    """Implements SimMIM - self supervised learning through masked patch reconstruction. 

    Currently ___DOES NOT WORK___. Not sure why. Even with no mask, the model cannot
    reconstruct the original image, even with LOTS of training. 
    """

    def __init__(
        self,
        mask_ratio=0.2,
    ):
        super().__init__()

        # generates the grid of masks
        self.masking_generator = MaskingGenerator(
            (64, 64), int(64 * 64 * mask_ratio), min_num_patches=16, max_num_patches=100
        )

        # tries to reconstruct the original image patches from the encoder output.
        # encoder output is B, 256, 64, 64
        # since the VIT patches the 1024x1024 image into the 64x64 grid, we need to
        # we need to construct a 16x16x3 patch from a (256) feature. We do this by linear 
        # projection, and rearranging
        self.projector: nn.Module = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256 * 3, 1),
        )

        self.maskable_model = MaskableEncoderWithProjection(
            encoder=self.image_encoder, projector=self.projector, transformer_dim=768
        )

    def forward(self, image): 
        B, C, H, W = image.shape 
        assert H == W == 1024, "image must be 1024x1024"
        assert C == 3, "image must have 3 channels"

        # generate the masks
        masks = []
        for _ in range(image.shape[0]):
            masks.append(torch.from_numpy(self.masking_generator()).bool().to(image.device))
        masks = torch.stack(masks) 
        # masks is B, 64, 64

        # get the projected embeddings (which can be reshaped to the "reconstruction" of the image)
        if debug: 
            masks = None
        projected_embeddings = self.maskable_model(image, mask=masks)

        if False: 
            # select projected image embeddings that are in the mask
            projected_embeddings = einops.rearrange(projected_embeddings, "b c h w -> b h w c")
            projected_embeddings = projected_embeddings[masks] # N x C

            # convert the image to flattened patches
            image_patches = einops.rearrange(image, "b c (h p1) (w p2) -> b h w (p1 p2 c)", p1=16, p2=16)
            image_patches = image_patches[masks] # N x (p1 p2 c)

        else: 
            reconstructed_image = einops.rearrange(projected_embeddings, "b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=16, p2=16)
            loss = torch.nn.functional.l1_loss(reconstructed_image, image)
            return loss 
        
        breakpoint()

        assert image_patches.shape[0] == projected_embeddings.shape[0], "number of patches must match number of embeddings"

        return torch.nn.functional.l1_loss(projected_embeddings, image_patches)

    def from_config(config: SimMIMModelConfig):
        return SimMIMModel()

    @torch.no_grad()
    def show_example(self, image):
        """Shows an example of the image and the PCA of the features"""
        image_features = self.image_encoder(image)
        masks = self.generate_masks(image)

        B, C, H, W = image_features.shape

        # make the features into a matrix.
        image_features = einops.rearrange(image_features, "b c h w -> (b h w) c")

        # compute the pca of the features
        from sklearn.decomposition import PCA

        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(image_features.cpu().numpy())
        # make PCA features between 0 and 1

        N, C = features_pca.shape
        assert C == 3, "pca should have 3 components"
        features_pca = einops.rearrange(
            features_pca, "(b h w) c -> b h w c", b=B, h=H, w=W
        )
        
        projected_embeddings = self.maskable_model(image, mask=masks) if not debug else self.maskable_model(image)
        image_reconstruction = einops.rearrange(projected_embeddings, "b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=16, p2=16)
        image_reconstruction = image_reconstruction.cpu().numpy()
        image_reconstruction = einops.rearrange(image_reconstruction, "b c h w -> b h w c")
        masks = masks.cpu().numpy()
        image = einops.rearrange(image, "b c h w -> b h w c").cpu().numpy()

        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(image[0])
        ax[1].imshow(features_pca[0])
        ax[2].imshow(image_reconstruction[0][..., 0])
        ax[3].imshow(masks[0])

        return fig

    def generate_masks(self, image): 
        # generate the masks
        masks = []
        for _ in range(image.shape[0]):
            masks.append(torch.from_numpy(self.masking_generator()).bool().to(image.device))
        masks = torch.stack(masks) 

        return masks


class MaskableEncoderWithProjection(nn.Module):
    """Wraps the medsam image encoder with a projection head and tokenization head.

    The input of this model will be a batch of images of shape (B, 3, 1024, 1024). The intermediate
    output of the model will be a batch of patch embeddings of shape (B, 256, 64, 64). The final output
    will be a batch of token embeddings of shape (B, 1024, 64, 64), where each position (i, :, j, k)
    is the vector of token ``scores`` for the patch at position (i, j, k).

    Args:
        projector (nn.Module, optional): A projection head to project the patch embeddings. This
            module receives the (B, 256, 64, 64) tensor of features coming from the (possibly masked)
            input image, and gives an output of any desired shape depending on the use case.
    """

    def __init__(self, encoder, projector, transformer_dim=768):
        super().__init__()
        self.encoder = encoder
        self.proj = projector
        self.mask_token = torch.nn.Parameter(torch.randn(transformer_dim))

    def forward(self, image, mask=None):
        embed = self.encoder.patch_embed(image)  # B, N, H, W

        if mask is not None:
            embed[mask] = self.mask_token

        # do the rest of the forward pass
        x = embed

        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed

        for blk in self.encoder.blocks:
            x = blk(x)

        x = self.encoder.neck(x.permute(0, 3, 1, 2))
        x = self.proj(x)
        return x


if __name__ == "__main__":
    config = parse_args()
    main(config)
