import torch
from einops import rearrange, repeat
from torch import nn
from dataclasses import dataclass


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    @dataclass
    class Output:
        """
        Core_predictions: B x C
        Core_labels: B x 1
        Patch_predictions: (N x C) where N is the sum over each image in the batch of the number of valid pixels.
        Patch_labels: (N x 1) where N is the sum over each image in the batch of the number of valid pixels.
        Core_indices: (N x 1) where N is the sum over each image in the batch of the number of valid pixels.
        """

        core_predictions: torch.Tensor | None = None
        patch_predictions: torch.Tensor | None = None
        patch_logits: torch.Tensor | None = None
        patch_labels: torch.Tensor | None = None
        core_indices: torch.Tensor | None = None
        core_labels: torch.Tensor | None = None

    def __init__(
        self, needle_mask_threshold: float = 0.5, prostate_mask_threshold: float = 0.5
    ):
        super().__init__()
        self.needle_mask_threshold = needle_mask_threshold
        self.prostate_mask_threshold = prostate_mask_threshold

    def forward(self, heatmap_logits, needle_mask, prostate_mask, label):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        # chunk the needle mask into patches according to the heatmap size
        needle_mask = rearrange(
            needle_mask, "b c (nh h) (nw w) -> b c nh nw h w", nh=H, nw=W
        )
        needle_mask = needle_mask.mean(dim=(-1, -2)) > self.needle_mask_threshold
        mask = needle_mask

        # if prostate mask is provided, use it to further mask the needle mask
        if prostate_mask is not None:
            prostate_mask = rearrange(
                prostate_mask, "b c (nh h) (nw w) -> b c nh nw h w", nh=H, nw=W
            )
            prostate_mask = (
                prostate_mask.mean(dim=(-1, -2)) > self.prostate_mask_threshold
            )
            mask = mask & prostate_mask
            if mask.sum() == 0:
                mask = needle_mask

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)
        label_rep = repeat(label, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0].bool()
        label_flattened = rearrange(label_rep, "b h w -> (b h w)", h=H, w=W)[
            ..., None
        ].float()
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        label = label_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_predictions = logits.sigmoid()
        patch_logits = logits
        patch_labels = label
        core_predictions = []
        core_labels = []

        for i in core_idx.unique().tolist():
            core_idx_i = core_idx == i
            logits_i = logits[core_idx_i]
            predictions_i = logits_i.sigmoid().mean(dim=0)
            core_predictions.append(predictions_i)
            core_labels.append(label[core_idx_i][0])

        core_predictions = torch.stack(core_predictions)
        core_labels = torch.stack(core_labels)

        return self.Output(
            core_predictions=core_predictions,
            core_labels=core_labels,
            patch_predictions=patch_predictions,
            patch_logits=patch_logits,
            patch_labels=patch_labels,
            core_indices=core_idx,
        )