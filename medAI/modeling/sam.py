"""
Implements wrappers and registry for Segment Anything Model (SAM) models.
"""

import os

import torch
from torch import nn

from .medsam.segment_anything.modeling.image_encoder import (
    Attention,
    Block,
    ImageEncoderViT,
    MLPBlock,
)
from .segment_anything.build_sam import sam_model_registry

CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR")  # top level checkpoint directory
if CHECKPOINT_DIR is None:
    raise ValueError(
        "Environment variable CHECKPOINT_DIR must be set to the top level checkpoint directory"
    )


def build_sam():
    """Builds the sam-vit-b model."""
    checkpoint = os.path.join(CHECKPOINT_DIR, "sam_vit_b_01ec64.pth")
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    return model


def build_medsam():
    """
    Builds the MedSAM model by building the SAM model and loading the medsam checkpoint.
    """
    checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    return model


def build_sammed_2d():
    from argparse import Namespace

    import torch

    from .sam_med2d.segment_anything import sam_model_registry as sammed_model_registry

    args = Namespace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.image_size = 256
    args.encoder_adapter = True
    args.sam_checkpoint = os.path.join(CHECKPOINT_DIR, "sam-med2d_b.pth")
    model = sammed_model_registry["vit_b"](args).to(device)
    return model


def build_adapter_medsam_256():
    checkpoint = os.path.join(CHECKPOINT_DIR, "medsam_vit_b_cpu.pth")
    model = sam_model_registry["vit_b"](checkpoint=checkpoint)

    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=256
    )
    freeze_non_adapter_layers(model.image_encoder)
    return model


def build_adapter_sammed_2d():
    model = build_sammed_2d()
    freeze_non_adapter_layers(model.image_encoder)
    return model


def build_adapter_sam():
    model = build_sam()
    model.image_encoder = wrap_image_encoder_with_adapter(
        model.image_encoder, adapter_dim=256
    )
    freeze_non_adapter_layers(model.image_encoder)
    return model


class SAMForUnpromptedSegmentation(nn.Module):
    """
    Wraps the SAM model to do unprompted segmentation.

    Args:
        freeze_backbone (bool): If True, freezes the backbone of the model.
    """

    def __init__(self, sam_model):
        super().__init__()
        from segment_anything import sam_model_registry

        self.sam_model = sam_model

    def forward(self, image):
        image_feats = self.sam_model.image_encoder(image)
        sparse_embedding, dense_embedding = self.sam_model.prompt_encoder(
            None, None, None  # no prompt - find prostate
        )
        mask_logits = self.sam_model.mask_decoder.forward(
            image_feats,
            self.sam_model.prompt_encoder.get_dense_pe(),
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]
        return mask_logits

    def get_loss_and_score(self, mask_logits, gt_mask):
        """Computes the loss and performance score for the given mask logits and ground truth mask.

        Loss is the sum of dice and cross-entropy loss

        Args:
            mask_logits (torch.Tensor): The mask logits of shape (B, C, H, W).
            gt_mask (torch.Tensor): The ground truth mask of shape (B, C, H, W).

        Returns:
            tuple: (loss, dice_score) A tuple containing the loss and the dice score.
        """

        from ..metrics import dice_loss, dice_score

        B, C, H, W = mask_logits.shape

        gt_mask = torch.nn.functional.interpolate(gt_mask.float(), size=(H, W))

        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            mask_logits, gt_mask
        )
        _dice_loss = dice_loss(mask_logits.sigmoid(), gt_mask)
        loss = ce_loss + _dice_loss

        _dice_score = dice_score(mask_logits.sigmoid(), gt_mask)
        return loss, _dice_score


class Adapter(nn.Module):
    def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(feature_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, feature_dim)
        self.act = nn.GELU()

        # initializations to make it close to identity function
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


class AdapterAttn(nn.Module):
    def __init__(self, attn: Attention, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()
        self.attn = attn
        embedding_dim = attn.proj.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.adapter(x)
        return x


class AdapterMLPBlock(nn.Module):
    def __init__(self, mlp: MLPBlock, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()

        self.mlp = mlp
        embedding_dim = mlp.lin1.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.adapter(x)
        return x


def wrap_block_with_adapter(block: Block, adapter_dim: int, init_scale: float = 1e-3):
    block.attn = AdapterAttn(block.attn, adapter_dim, init_scale=init_scale)
    block.mlp = AdapterMLPBlock(block.mlp, adapter_dim, init_scale=init_scale)
    return block


def wrap_image_encoder_with_adapter(
    image_encoder: ImageEncoderViT, adapter_dim: int, init_scale: float = 1e-3
):
    new_blocks = torch.nn.ModuleList()
    for block in image_encoder.blocks:
        new_block = wrap_block_with_adapter(block, adapter_dim, init_scale=init_scale)
        new_blocks.append(new_block)

    image_encoder.blocks = new_blocks

    return image_encoder


def freeze_non_adapter_layers(model: nn.Module):
    for name, param in model.named_parameters():
        if "adapter" not in name.lower():
            param.requires_grad = False

    return model
