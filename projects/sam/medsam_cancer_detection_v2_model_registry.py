from dataclasses import dataclass
from torch import nn 
import torch 
from abc import ABC, abstractmethod
from segment_anything import sam_model_registry
import matplotlib.pyplot as plt
from einops import rearrange, repeat
from medAI.utils.registry import Registry


model_registry = Registry('model_registry', help="Registry for cancer detection models")


@dataclass
class CancerDetectorOutput: 
    loss: torch.Tensor | None = None
    core_predictions: torch.Tensor | None = None
    patch_predictions: torch.Tensor | None = None
    patch_labels: torch.Tensor | None = None
    core_indices: torch.Tensor | None = None
    core_labels: torch.Tensor | None = None
    cancer_logits_map: torch.Tensor | None = None


class CancerDetectorBase(nn.Module, ABC): 
    @abstractmethod
    def forward(self, image, needle_mask, label, prostate_mask) -> CancerDetectorOutput:        
        ...

    @torch.no_grad()
    def show_example(self, image, needle_mask, label, prostate_mask=None):
        import matplotlib.pyplot as plt
        import numpy as np

        output: CancerDetectorOutput = self(image, needle_mask, label, prostate_mask=prostate_mask)

        logits = output.cancer_logits_map
        pred = logits.sigmoid()
        needle_mask = needle_mask.cpu()
        logits = logits.cpu()
        pred = pred.cpu()
        image = image.cpu()

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        [ax.set_axis_off() for ax in ax.flatten()]
        kwargs = dict(vmin=0, vmax=1, extent=(0, 46, 0, 28))
        ax[0].imshow(image[0].permute(1, 2, 0), **kwargs)
        if prostate_mask is not None: 
            prostate_mask = prostate_mask.cpu()
            ax[0].imshow(prostate_mask[0, 0], alpha=prostate_mask[0][0] * 0.3, cmap="Blues", **kwargs)
        ax[0].imshow(needle_mask[0, 0], alpha=needle_mask[0][0], cmap="Reds", **kwargs)
        ax[1].imshow(pred[0, 0], **kwargs)
        ax[1].set_title(f"label: {label[0].item()}")


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

    def __init__(self, needle_mask_threshold: float = 0.5, prostate_mask_threshold: float = 0.5): 
        super().__init__()
        self.needle_mask_threshold = needle_mask_threshold
        self.prostate_mask_threshold = prostate_mask_threshold

    def forward(self, heatmap_logits, needle_mask, prostate_mask, label):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape
        needle_mask = rearrange(
            needle_mask, "b c (nh h) (nw w) -> b c nh nw h w", nh=H, nw=W
        )
        needle_mask = needle_mask.mean(dim=(-1, -2)) > self.needle_mask_threshold
        mask = needle_mask

        if prostate_mask is not None: 
            prostate_mask = rearrange(
                prostate_mask, "b c (nh h) (nw w) -> b c nh nw h w", nh=H, nw=W
            )
            prostate_mask = prostate_mask.mean(dim=(-1, -2)) > self.prostate_mask_threshold
            mask = mask & prostate_mask
            if mask.sum() == 0:
                mask = needle_mask

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)
        label_rep = repeat(label, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[
            ..., 0
        ].bool()
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


@model_registry
class MedSAMCancerDetectorV2(nn.Module):
    def __init__(self, medsam_checkpoint: str | None = None, freeze_backbone: bool=False):
        super().__init__()
        self.medsam_model = sam_model_registry["vit_b"](
            checkpoint="/scratch/ssd004/scratch/pwilson/medsam_vit_b_cpu.pth"
        )
        if medsam_checkpoint is not None:
            self.medsam_model.load_state_dict(
                torch.load(medsam_checkpoint, map_location="cpu")
            )
        if freeze_backbone: 
            self.freeze_backbone()
        else: 
            self.thaw_backbone()

    def thaw_backbone(self):
        for param in self.medsam_model.image_encoder.parameters():
            param.requires_grad = True
        self._backbone_frozen = False

    def freeze_backbone(self):
        for param in self.medsam_model.image_encoder.parameters():
            param.requires_grad = False
        self._backbone_frozen = True

    def forward(self, image):
        with torch.no_grad() if self._backbone_frozen else torch.enable_grad():
            image_emb = self.medsam_model.image_encoder(image)
        sparse_emb, dense_emb = self.medsam_model.prompt_encoder(None, None, None)
        mask_logits = self.medsam_model.mask_decoder.forward(
            image_emb,
            self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_emb,
            dense_emb,
            False,
        )[0]
        return mask_logits


class LinearLayerMedsamCancerDetector(MedSAMCancerDetectorV2):
    def __init__(self):
        super().__init__()
        self.clf = torch.nn.Conv2d(256, 1, kernel_size=1, padding=0)

    def get_logits(self, image):
        image_emb = self.medsam_model.image_encoder(image)
        logits = self.clf(image_emb)
        return logits


@model_registry
class MedSAMCancerDetectorWithAdapters(MedSAMCancerDetectorV2): 
    def __init__(self, adapter_dim: int = 64, thaw_patch_embed: bool = False, 
                 adapter_init_scale: float = 1e-3):
        super().__init__()
        from models import wrap_image_encoder_with_adapter, freeze_non_adapter_layers
        wrap_image_encoder_with_adapter(self.medsam_model.image_encoder, adapter_dim=adapter_dim, init_scale=adapter_init_scale)
        freeze_non_adapter_layers(self.medsam_model.image_encoder)
        if thaw_patch_embed:
            for param in self.medsam_model.image_encoder.patch_embed.parameters(): 
                param.requires_grad = True


RESNET10_PATH = '/ssd005/projects/exactvu_pca/checkpoint_store/vicreg_resnet10_feature_extractor_fold0.pth'


class LinearEval(torch.nn.Module):
    def __init__(self, model, linear_layer, freeze_model=True): 
        super().__init__()
        self.model = model
        self.linear_layer = linear_layer
        self.freeze_model = freeze_model

    def forward(self, x):
        with torch.no_grad() if self.freeze_model else torch.enable_grad(): 
            x = self.model(x)
        x = self.linear_layer(x)
        return x

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_model:
            self.model.eval()


class SlidingWindowHeatmapGenerator(torch.nn.Module): 
    def __init__(self, window_predictor_module, input_shape, strides, window_size, intermediate_size):
        super().__init__()
        self.window_predictor_module = window_predictor_module
        self.input_shape = input_shape
        self.strides = strides
        self.window_size = window_size
        self.intermediate_size = intermediate_size

        # compute n_windows_h, n_windows_w
        self.n_windows_h = (input_shape[0] - window_size[0]) // strides[0] + 1
        self.n_windows_w = (input_shape[1] - window_size[1]) // strides[1] + 1

        # unfold module to extract windows
        self.unfold = torch.nn.Unfold(window_size, strides)

        # rearrange module to rearrange the windows into patches 
        from einops.layers.torch import Rearrange
        self.rearrange_patches = Rearrange('b (c n_h n_w) h w -> b c n_h n_w h w', n_h=self.n_windows_h, n_w=self.n_windows_w, h=window_size[0], w=window_size[1])


@model_registry
class ResnetSlidingWindowCancerDetector: 
    def __init__(self, freeze_model=True): 
        super().__init__()
        from trusnet.modeling.registry import resnet10_feature_extractor
        feature_extractor = resnet10_feature_extractor().cuda()
        feature_extractor.load_state_dict(torch.load(RESNET10_PATH))
        linear_layer = torch.nn.Linear(512, 1).cuda()
        self.model = LinearEval(feature_extractor, linear_layer, freeze_model=freeze_model)


    def forward(self, image, needle_mask, label, prostate_mask) -> CancerDetectorOutput:
        from medAI.utils import view_as_windows_torch
        
        bmode = image[:, [0], ...] # take only the first channel

        with torch.no_grad(): 
            B, C, H, W = bmode.shape 
            step_size = (int(H / 28), int(W / 46))
            window_size = step_size[0] * 5, step_size[1] * 5

            needle_mask = torch.nn.functional.interpolate(needle_mask, size=(H, W), mode='nearest')
            prostate_mask = torch.nn.functional.interpolate(prostate_mask, size=(H, W), mode='nearest')
            
            needle_mask = view_as_windows_torch(needle_mask, window_size, step_size)
            needle_mask = (needle_mask.mean(dim=(4, 5)) > 0.66)
            needle_mask = rearrange(needle_mask, 'b c nh nw -> (b nh nw) c')[..., 0]

            prostate_mask = view_as_windows_torch(prostate_mask, window_size, step_size)
            prostate_mask = (prostate_mask.mean(dim=(4, 5)) > 0.9)
            prostate_mask = rearrange(prostate_mask, 'b c nh nw -> (b nh nw) c')[..., 0]

            mask = needle_mask & prostate_mask
            if mask.sum() == 0:
                mask = needle_mask

            bmode = view_as_windows_torch(bmode, window_size, step_size)
            B, C, nH, nW, H, W = bmode.shape
            bmode = rearrange(bmode, 'b c nh nw h w -> (b nh nw) c h w')
            bmode = (bmode - bmode.mean(dim=(-1, -2, -3), keepdim=True)) / bmode.std(dim=(-1, -2, -3), keepdim=True)
            bmode = torch.nn.functional.interpolate(bmode, size=(256, 256), mode='bilinear', align_corners=False)

            label = repeat(label, 'b -> b nh nw', nh=nH, nw=nW)
            label = rearrange(label, 'b nh nw -> (b nh nw)')[mask]
            batch_idx = torch.arange(B, device=bmode.device)
            batch_idx = repeat(batch_idx, 'b -> b nh nw', nh=nH, nw=nW)
            batch_idx = rearrange(batch_idx, 'b nh nw -> (b nh nw)')[mask]

        logits = self.model(bmode)

        logits_map = rearrange(logits, '(b nh nw) c -> b c nh nw', b=B, nh=nH, nw=nW)
        logits = logits[mask]
        patch_predictions = logits.sigmoid()
        label = label[..., None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label.float())

        # breakpoint()

        core_pred = []
        core_label = []
        for i in range(B): 
            core_pred.append(logits[batch_idx == i].sigmoid().mean(dim=0))
            core_label.append(label[batch_idx == i][0])
        core_pred = torch.stack(core_pred)
        core_label = torch.stack(core_label)

        return CancerDetectorOutput(
            loss=loss,
            core_predictions=core_pred,
            core_labels=core_label,
            cancer_logits_map=logits_map, 
            patch_predictions=patch_predictions,
            patch_labels=label, 
            core_indices=batch_idx
        )   

