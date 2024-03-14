import torch
import typing as tp
from dataclasses import dataclass, field
from torch import nn
from enum import StrEnum
import logging
from pydantic import BaseModel


class PromptOptions(StrEnum):
    """Options for prompts that can be used in the ProstNFound model.
    
    Members: 
        task: Task prompt (not currently used, could be used to encode the task ID for multitask training)
        anatomical: Anatomical location prompt (Sextant biopsy loc)
        psa: PSA prompt 
        age: Age prompt
        family_history: Family history prompt
        prostate_mask: Prostate mask prompt
        sparse_cnn_patch_features: Sparse CNN patch features prompt
        data_independent_prompts: Data independent prompts
        approx_psa_density: Approximate PSA density prompt
        sparse_cnn_patch_features_rf: Sparse CNN patch features for RF images prompt
        dense_cnn_features: Dense CNN features prompt
        custom: Custom prompt
    """

    task = "task"
    anatomical = "anatomical"
    psa = "psa"
    age = "age"
    family_history = "family_history"
    prostate_mask = "prostate_mask"
    sparse_cnn_patch_features = "sparse_cnn_patch_features"
    data_independent_prompts = "data_independent_prompts"
    approx_psa_density = "approx_psa_density"
    sparse_cnn_patch_features_rf = "sparse_cnn_patch_features_rf"
    dense_cnn_features = "dense_cnn_features"
    custom = "custom"


class BackboneOptions(StrEnum):
    sam = "sam"
    medsam = "medsam"
    sam_med2d = "sam_med2d"
    adapter_medsam = "adapter_medsam"
    adapter_sam = "adapter_sam"
    adapter_sammed_2d = "adapter_sammed_2d"


class ProstNFoundConfig(BaseModel):
    """
    Configuration for the ProstNFound model.

    Args:
        n_tasks (int): Number of tasks. Defaults to 1.
        prompts (list[PromptOptions]): List of prompts to use. Defaults to [].
        prompt_dropout (float): Dropout probability for prompts. Defaults to 0.0.
            Prompt dropout is applied to each prompt independently, and is only applied during training.
            It is implemented by randomly replacing the prompt embeddings with a learned "null" prompt.
        sam_backbone (BackboneOptions): Backbone to use. Defaults to BackboneOptions.medsam.
        replace_patch_embed (bool): Whether to replace the patch embed. Defaults to False. Only applies to sam_med2d backbone.
        sparse_cnn_backbone_path (str): Path to the sparse CNN backbone. Defaults to None. Only applies if using a sparse CNN backbone.
        freeze_mask_decoder (bool): Whether to freeze the mask decoder. Defaults to False.
        freeze_image_encoder (bool): Whether to freeze the image encoder. Defaults to False.
        freeze_cnn (bool): Whether to freeze the CNN. Defaults to False.
        img_emb_dropout (float): Dropout probability for image embeddings. Defaults to 0.0.
        cnn_patches_whole_prostate (bool): Whether to use whole prostate for CNN patches. Defaults to False, or only using the needle
            mask for the CNN patches.
        pos_embed_cnn_patch (bool): Whether to use positional embeddings for CNN patches. Defaults to False. Only applies if using sparse CNN.
        pool_patch_features (tp.Literal[None, 'transformer', 'max', 'mean']): Pooling method for patch features. If using sparse CNN,
            this should be specified. Defaults to None.
    """

    n_tasks: int = 1
    prompts: list[PromptOptions] = []
    prompt_dropout: float = 0.0
    sam_backbone: BackboneOptions = BackboneOptions.medsam
    replace_patch_embed: bool = False
    sparse_cnn_backbone_path: str | None = None
    freeze_mask_decoder: bool = False
    freeze_image_encoder: bool = False
    freeze_cnn: bool = False
    img_emb_dropout: float = 0.0
    cnn_patches_whole_prostate: bool = False
    pos_embed_cnn_patch: bool = False
    pool_patch_features: tp.Literal[None, "transformer", "max", "mean"] = None


class ProstNFound(nn.Module):
    def __init__(
        self,
        config: ProstNFoundConfig,
    ):
        super().__init__()
        self.config = config

        if config.replace_patch_embed and config.sam_backbone != "sam_med2d":
            raise ValueError(
                "replace_patch_embed is only supported for sam_med2d backbone"
            )

        for p in config.prompts:
            if not p in PromptOptions:
                raise ValueError(
                    f"Unknown prompt option: {p}. Options are {PromptOptions}"
                )

        from medAI.modeling.sam import (
            build_adapter_medsam_256,
            build_adapter_sam,
            build_adapter_sammed_2d,
            build_medsam,
            build_sam,
            build_sammed_2d,
        )

        # BUILD BACKBONE
        if config.sam_backbone == "medsam":
            self.medsam_model = build_medsam()
            self.image_size_for_features = 1024
        elif config.sam_backbone == "adapter_medsam":
            self.medsam_model = build_adapter_medsam_256()
            self.image_size_for_features = 1024
        elif config.sam_backbone == "sam":
            self.medsam_model = build_sam()
            self.image_size_for_features = 1024
        elif config.sam_backbone == "adapter_sam":
            self.medsam_model = build_adapter_sam()
            self.image_size_for_features = 1024
        elif config.sam_backbone == "sam_med2d":
            self.medsam_model = build_sammed_2d()

            if config.replace_patch_embed:
                self.image_size_for_features = 1024
                # sammed_2d has a different input size. Let's hack the model to accept 1024x1024 images
                from einops.layers.torch import Rearrange

                new_patch_embed = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 64),
                    nn.GELU(),
                    nn.Conv2d(64, 768, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(32, 768),
                    nn.GELU(),
                    nn.MaxPool2d(4, 4),
                    Rearrange("b c h w -> b h w c"),
                )
                self.medsam_model.image_encoder.patch_embed = new_patch_embed
            else:
                # use the default patch embed which is designed for 256x256 images
                self.image_size_for_features = 256
        elif config.sam_backbone == "adapter_sammed_2d":
            self.medsam_model = build_adapter_sammed_2d()
            self.image_size_for_features = 256

        self.img_emb_dropout = nn.Dropout(config.img_emb_dropout)

        if config.freeze_image_encoder:
            logging.debug("Freezing image encoder")
            for param in self.medsam_model.image_encoder.parameters():
                param.requires_grad = False

        if config.freeze_mask_decoder:
            logging.debug("Freezing mask decoder")
            for param in self.medsam_model.mask_decoder.parameters():
                param.requires_grad = False

        # BUILD PROMPT MODULES
        EMBEDDING_DIM = 256

        # null prompt - used for prompt dropout
        self.null_prompt = nn.Parameter(torch.zeros(1, EMBEDDING_DIM))

        # used for multitask training, but not currently used
        self.task_prompt_module = nn.Embedding(config.n_tasks, EMBEDDING_DIM)

        # 6 anatomical locations (mid-lateral, mid-medial, apex-lateral, apex-medial, base-lateral, base-medial)
        self.anatomical_prompt_module = nn.Embedding(6, EMBEDDING_DIM)

        # embed floating point values to 256 dim
        self.psa_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )
        self.age_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )
        self.approx_psa_density_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )
        self.custom_prompt_module = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )

        # 3 values for family history: 0, 1, 2 (yes, no, unknown)
        self.family_history_prompt_module = nn.Embedding(3, EMBEDDING_DIM)

        # CNN for extracting patch features
        from timm.models.resnet import resnet10t

        model = resnet10t(
            in_chans=3,
        )
        model.fc = nn.Identity()
        model = nn.Sequential(nn.InstanceNorm2d(3), model)
        if config.sparse_cnn_backbone_path is not None:
            state = torch.load(config.sparse_cnn_backbone_path, map_location="cpu")
            model.load_state_dict(
                {
                    k.replace("backbone.", ""): v
                    for k, v in state.items()
                    if "backbone" in k
                }
            )
        self.patch_feature_cnn = model
        if config.freeze_cnn:
            for param in self.patch_feature_cnn.parameters():
                param.requires_grad = False

        from medAI.modeling.transformer import TransformerEncoder

        if config.pool_patch_features == "transformer":
            self.patch_aggregator = TransformerEncoder(
                n_layers=6, n_heads=8, d_model=256, d_feed_forward=256, dropout=0.1
            )
        else:
            self.patch_aggregator = nn.Identity()

        self.dense_feature_projection = nn.Conv2d(512, EMBEDDING_DIM, kernel_size=1)

        # project the CNN features to the prompt space
        # self.patch_feature_prompt_module = nn.Linear(512, EMBEDDING_DIM)
        self.patch_feature_prompt_module = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, EMBEDDING_DIM),
        )

        self.pad_token = nn.Parameter(
            torch.zeros(EMBEDDING_DIM)
        )  # for padding the number of patches to a fixed number in a minibatch

        # data independent prompts
        self.data_independent_prompts = nn.Parameter(torch.randn(1, 10, EMBEDDING_DIM))

    def forward(
        self,
        image,
        task_id=None,
        anatomical_location=None,
        psa=None,
        age=None,
        family_history=None,
        prostate_mask=None,
        needle_mask=None,
        approx_psa_density=None,
        rf_image=None,
        custom=None,
        return_prompt_embeddings=False,
    ):
        DEVICE = image.device
        B, C, H, W = image.shape
        if H != self.image_size_for_features or W != self.image_size_for_features:
            image_resized_for_features = torch.nn.functional.interpolate(
                image, size=(self.image_size_for_features, self.image_size_for_features)
            )
        else:
            image_resized_for_features = image
        image_feats = self.medsam_model.image_encoder(image_resized_for_features)
        image_feats = self.img_emb_dropout(image_feats)

        if "prostate_mask" in self.config.prompts:
            if (
                prostate_mask is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                mask = None
            else:
                B, C, H, W = prostate_mask.shape
                if H != 256 or W != 256:
                    prostate_mask = torch.nn.functional.interpolate(
                        prostate_mask, size=(256, 256)
                    )
                mask = prostate_mask
        else:
            mask = None

        # use the prompt encoder to get the prompt embeddings for the mask, if provided.
        # otherwise we will use our own custom prompt modules exclusively
        sparse_embedding, dense_embedding = self.medsam_model.prompt_encoder.forward(
            None, None, mask
        )
        # sparse embeddings will be an empty tensor if the mask is None,
        # and we need to repeat the embeddings for each image in the batch
        sparse_embedding = sparse_embedding.repeat_interleave(len(image), 0)

        if "dense_cnn_features" in self.config.prompts:
            dense_features = self.patch_feature_cnn[0](image)
            dense_features = self.patch_feature_cnn[1].forward_features(dense_features)
            dense_features = self.dense_feature_projection(dense_features)
            dense_features = torch.nn.functional.interpolate(
                dense_features, size=dense_embedding.shape[-2:]
            )
            if self.training:
                dense_features = torch.nn.functional.dropout(
                    dense_features, p=0.5, training=True
                )
            dense_embedding = dense_embedding + dense_features

        if "task" in self.config.prompts:
            task_embedding = self.task_prompt_module(task_id)
            task_embedding = task_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, task_embedding], dim=1)

        if "anatomical" in self.config.prompts:
            if (
                anatomical_location is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                anatomical_embedding = self.null_prompt.repeat_interleave(
                    len(task_id), 0
                )
            else:
                anatomical_embedding = self.anatomical_prompt_module(
                    anatomical_location
                )
            anatomical_embedding = anatomical_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, anatomical_embedding], dim=1
            )

        if "psa" in self.config.prompts:
            if (
                psa is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                psa_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                psa_embedding = self.psa_prompt_module(psa)
            psa_embedding = psa_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, psa_embedding], dim=1)

        if "approx_psa_density" in self.config.prompts:
            if (
                psa is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                approx_psa_density_embedding = self.null_prompt.repeat_interleave(
                    len(task_id), 0
                )
            else:
                approx_psa_density_embedding = self.approx_psa_density_prompt_module(
                    approx_psa_density
                )
            approx_psa_density_embedding = approx_psa_density_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, approx_psa_density_embedding], dim=1
            )

        if "age" in self.config.prompts:
            if (
                age is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                age_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                age_embedding = self.age_prompt_module(age)
            age_embedding = age_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, age_embedding], dim=1)

        if "family_history" in self.config.prompts:
            if (
                family_history is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                family_history = torch.ones_like(task_id) * 2  # this encodes "unknown"
            family_history_embedding = self.family_history_prompt_module(family_history)
            family_history_embedding = family_history_embedding[:, None, :]
            sparse_embedding = torch.cat(
                [sparse_embedding, family_history_embedding], dim=1
            )

        if "data_independent_prompts" in self.config.prompts:
            sparse_embedding = torch.cat(
                [
                    sparse_embedding,
                    self.data_independent_prompts.repeat_interleave(B, 0),
                ],
                dim=1,
            )

        if "custom" in self.config.prompts:
            if (
                custom is None
                or self.config.prompt_dropout > 0
                and self.training
                and torch.rand(1) < self.config.prompt_dropout
            ):
                custom_embedding = self.null_prompt.repeat_interleave(len(task_id), 0)
            else:
                custom_embedding = self.custom_prompt_module(custom)
            custom_embedding = custom_embedding[:, None, :]
            sparse_embedding = torch.cat([sparse_embedding, custom_embedding], dim=1)

        if "sparse_cnn_patch_features" in self.config.prompts:
            # we need to extract patches from the images.
            # patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_bmode(
            #     image, needle_mask, prostate_mask
            # )
            window_size=(128, 128)
            stride=(64, 64)
            if self.config.cnn_patches_whole_prostate:
                needle_mask_threshold = -1
                prostate_mask_threshold = 0.9
            else:
                needle_mask_threshold = 0.3
                prostate_mask_threshold = 0.9
            patch_cnn_sparse_embeddings = self.extract_patch_embeddings(
                image,
                window_size, 
                stride,
                needle_mask,
                prostate_mask,
                needle_mask_threshold, 
                prostate_mask_threshold,
                None 
            )
            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        if "sparse_cnn_patch_features_rf" in self.config.prompts:
            # patch_cnn_sparse_embeddings = self.get_cnn_patch_embedding_rf(
            #     rf_image, needle_mask, prostate_mask
            # )

            im_size_mm = 28, 46.06
            B, C, H, W = rf_image.shape
            logging.debug(f"RF shape: {image.shape}")
            im_size_px = H, W
            patch_size_mm = 5, 5
            if not self.config.cnn_patches_whole_prostate:
                patch_stride_mm = 1, 1
            else:
                patch_stride_mm = 2, 2
            patch_size_px = int(patch_size_mm[0] / im_size_mm[0] * im_size_px[0]), int(
                patch_size_mm[1] / im_size_mm[1] * im_size_px[1]
            )
            patch_stride_px = int(patch_stride_mm[0] / im_size_mm[0] * im_size_px[0]), int(
                patch_stride_mm[1] / im_size_mm[1] * im_size_px[1]
            )
            if self.config.cnn_patches_whole_prostate:
                needle_mask_threshold = -1
                prostate_mask_threshold = 0.9
            else:
                needle_mask_threshold = 0.6 
                prostate_mask_threshold = 0.9

            patch_cnn_sparse_embeddings = self.extract_patch_embeddings(
                rf_image,
                patch_size_px,
                patch_stride_px,
                needle_mask,
                prostate_mask,
                needle_mask_threshold,
                prostate_mask_threshold,
                (256, 256),
            )

            if patch_cnn_sparse_embeddings is not None:
                sparse_embedding = torch.cat(
                    [sparse_embedding, patch_cnn_sparse_embeddings], dim=1
                )

        # Compute the mask logits based on the prompt embeddings and image features
        mask_logits = self.medsam_model.mask_decoder.forward(
            image_feats,
            self.medsam_model.prompt_encoder.get_dense_pe(),
            sparse_embedding,
            dense_embedding,
            multimask_output=False,
        )[0]

        if return_prompt_embeddings:
            return mask_logits, sparse_embedding, dense_embedding
        else:
            return mask_logits

    def extract_patch_embeddings(
        self,
        image,
        patch_size_px,
        patch_stride_px,
        needle_mask,
        prostate_mask,
        needle_mask_threshold,
        prostate_mask_threshold,
        resize_patches_to=None,
    ):
        patches = []
        batch_indices = []
        positions = []

        B = len(image)
        for i in range(B):
            from medAI.utils.data.patch_extraction import PatchView

            im = image[i].permute(1, 2, 0).cpu().numpy()
            needle_mask_ = needle_mask[i].permute(1, 2, 0).cpu().numpy()
            prostate_mask_ = prostate_mask[i].permute(1, 2, 0).cpu().numpy()

            pv = PatchView.from_sliding_window(
                im,
                window_size=patch_size_px,
                stride=patch_stride_px,
                masks=[needle_mask_, prostate_mask_],
                thresholds=[needle_mask_threshold, prostate_mask_threshold],
                align_to="topright",
            )
            for position, patch in zip(pv.positions, pv):
                patches.append(torch.from_numpy(patch).permute(2, 0, 1))
                positions.append(torch.from_numpy(position))
                batch_indices.append(i)

        logging.debug(f"Extracted {len(patches)} patches from {B} images")
        # number of patches could be zero if there is no intersection with prostate. In this case, return None
        if len(patches) == 0:
            return None

        patches = torch.stack(patches).to(image.device)
        positions = torch.stack(positions).to(image.device)
        positions = positions[:, [1, 0]]
        batch_indices = torch.tensor(batch_indices)
 
        if resize_patches_to is not None:
            patches = torch.nn.functional.interpolate(
                patches, size=resize_patches_to, mode="bilinear"
            )
        
        # extract patch embeddings
        patch_cnn_output = self.patch_feature_cnn(patches)
        patch_cnn_output = self.patch_feature_prompt_module(patch_cnn_output)

        # maybe add pos embedding
        if self.config.pos_embed_cnn_patch:
            position_encoding_outputs = (
                self.medsam_model.prompt_encoder.pe_layer.forward_with_coords(
                    positions[None, ...], image_size=(1024, 1024)
                )[0]
            )
            patch_cnn_output = patch_cnn_output + position_encoding_outputs

        # for each batch, collect the patch embeddings
        sparse_embeddings_by_batch = []
        for i in range(B):
            patch_embeddings_for_batch = patch_cnn_output[batch_indices == i]  # N x 256
            if self.config.pool_patch_features == "mean":
                if len(patch_embeddings_for_batch) == 0:
                    return None  # no patches found
                patch_embeddings_for_batch = torch.mean(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                )
            elif self.config.pool_patch_features == "max":
                if len(patch_embeddings_for_batch) == 0:
                    return None
                patch_embeddings_for_batch = torch.max(
                    patch_embeddings_for_batch, dim=0, keepdim=True
                ).values
            sparse_embeddings_by_batch.append(patch_embeddings_for_batch)

        # pad to length of longest sequence
        max_len = max([len(e) for e in sparse_embeddings_by_batch])
        patch_cnn_sparse_embeddings = torch.zeros(B, max_len, 256, device=image.device)
        for i, e in enumerate(sparse_embeddings_by_batch):
            patch_cnn_sparse_embeddings[i, : len(e)] = e
            patch_cnn_sparse_embeddings[i, len(e) :] = self.pad_token[None, None, :]

        # maybe apply prompt dropout, for each patch independently
        # (if they are pooled, it will be applied to the pooled embeddings)
        if self.config.prompt_dropout > 0 and self.training:
            for i in range(patch_cnn_sparse_embeddings.shape[1]):
                if torch.rand(1) < self.config.prompt_dropout:
                    patch_cnn_sparse_embeddings[
                        :, i, :
                    ] = self.null_prompt.repeat_interleave(B, 0)

        B, N, C = patch_cnn_sparse_embeddings.shape
        if self.config.pool_patch_features == "transformer":
            patch_cnn_sparse_embeddings = self.patch_aggregator(
                patch_cnn_sparse_embeddings
            )
            patch_cnn_sparse_embeddings = patch_cnn_sparse_embeddings.mean(
                dim=1, keepdim=True
            )

        return patch_cnn_sparse_embeddings

    def train(self, mode: bool = True):
        super().train(mode)

        # if we are using a pretrained sparse CNN backbone, we need to set it to eval mode
        # even if we are in training mode so that batch norm statistics are not updated
        if (
            self.config.sparse_cnn_backbone_path is not None
            and self.patch_feature_cnn is not None
        ):
            self.patch_feature_cnn.eval()

    def get_params_groups(self):
        """Return the parameters groups for the optimizer,
        which will be used to set different learning rates for different parts of the model.

        Returns:
            Tuple[tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter], tp.List[torch.nn.Parameter]]:
                encoder_parameters, warmup_parameters, cnn_parameters
                (warmup_parameters are the parameters for the prompt modules and the prompt encoder and mask decoder)
        """

        from itertools import chain

        encoder_parameters = [
            p
            for (k, p) in self.medsam_model.image_encoder.named_parameters()
            if "neck" not in k
        ]
        warmup_parameters = chain(
            self.medsam_model.mask_decoder.parameters(),
            self.task_prompt_module.parameters(),
            self.anatomical_prompt_module.parameters(),
            self.psa_prompt_module.parameters(),
            self.age_prompt_module.parameters(),
            [self.null_prompt],
            [self.data_independent_prompts],
            self.family_history_prompt_module.parameters(),
            self.approx_psa_density_prompt_module.parameters(),
            self.patch_feature_prompt_module.parameters(),
            self.custom_prompt_module.parameters(),
            self.medsam_model.image_encoder.neck.parameters(),
            self.medsam_model.prompt_encoder.parameters(),
            [self.pad_token],
            self.dense_feature_projection.parameters(),
        )
        cnn_parameters = (
            self.patch_feature_cnn.parameters()
            if self.patch_feature_cnn is not None
            else []
        )

        return encoder_parameters, warmup_parameters, cnn_parameters
