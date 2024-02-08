"""

Adapted from https://github.com/bytedance/ibot
with major modifications to the model and data loading.

Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

"""

import os
import signal
import sys
from dataclasses import dataclass

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from medAI.utils import reproducibiliy
from medAI.utils.cosine_scheduler import cosine_scheduler
from medAI.utils.masking_generator import MaskingGenerator
from segment_anything.modeling.image_encoder import ImageEncoderViT
from torch.nn import functional as F
from tqdm import tqdm

import wandb
from src.data_factory import BModeDataFactoryV1, UASweepsDataFactory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class NCTDataOptions:
    """Options for selecting cores from the NCT dataset."""

    fold: int = 0
    n_folds: int = 5
    test_center: str = "UVA"
    undersample_benign_ratio: float | None = None


@dataclass
class Args:
    """Arguments for training"""

    batch_size: int = 1
    """Batch size to use for training"""

    use_nct_data: bool = True
    use_ua_unlabelled_data: bool = False
    crop_scale_1: tuple[float, float] = (0.5, 1)
    crop_scale_2: tuple[float, float] = (0.5, 1)
    random_gamma: tuple[float, float] | None = None
    nct_data_options: NCTDataOptions = NCTDataOptions()

    weight_decay: float = 0.04
    """Weight decay at the start of training. The weight decay is cosine annealed to `weight_decay_end` over the course
    of training."""
    weight_decay_end: float = 0.4
    """Weight decay at the end of training"""
    clip_grad: float = 3.0

    epochs: int = 30
    """Number of epochs to train for"""

    freeze_last_layer: int = 1
    """Number of epochs to freeze the last layer for. The last layer is the layer that is closest to the output."""

    lr: float = 0.0005
    """Learning rate at the end of linear warmup (highest LR used during training). The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256."""

    warmup_epochs: int = 5
    """Number of epochs to linearly warmup the learning rate to the specified learning rate."""

    min_lr: float = 1e-7
    """Minimum learning rate reached at the final iteration of training."""

    optimizer: str = "adamw"

    use_amp: bool = False
    """Whether to use automatic mixed precision (AMP) for training. This can speed up training and reduce memory usage"""

    momentum_teacher: float = 0.9999
    """Momentum to use for the teacher network. This is cosine annealed to 1 over the course of training.
    Although the original paper uses a momentum of 0.999, we use a higher momentum as recommended to handle 
    the smaller batch size.
    """

    project: str = "sam_ssl_pretraining"

    name: str = "train_medsam_dino_style"
    """Name of the wandb run. If None, don't log anything to wandb."""

    ckpt_dir: str = "./checkpoints"
    """Directory to save checkpoints to. If None, don't save any checkpoints."""

    log_image_freq: int | None = None
    """How often to log an example image to wandb. If None, don't log any images."""

    out_dim: int = 2048
    """Dimensionality of output for the [CLS] token"""
    patch_out_dim: int = 2048
    """Dimensionality of output for the patch tokens"""


def main(args: Args):
    # set up wandb and signal handlers for graceful exit
    wandb.init(project=args.project, name=args.name, config=args)

    state_path = os.path.join(args.ckpt_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
    else:
        state = None

    train_loader = build_dataloaders(
        include_nct=args.use_nct_data,
        include_ua=args.use_ua_unlabelled_data,
        batch_size=args.batch_size,
        nct_options=args.nct_data_options,
        crop_scale_1=args.crop_scale_1,
        crop_scale_2=args.crop_scale_2,
        random_gamma=args.random_gamma,
    )

    student = MedSAMIBot(out_dim=args.out_dim,
                         patch_out_dim=args.patch_out_dim).to(DEVICE)
    teacher = MedSAMIBot(out_dim=args.out_dim,
                         patch_out_dim=args.patch_out_dim).to(DEVICE)
    for param in teacher.parameters():
        param.requires_grad = False

    if state is not None:
        student.load_state_dict(state["student"])
        teacher.load_state_dict(state["teacher"])

    # parameter extraction:
    # in the dino paper they don't use weight decay on the bias and Norm parameters
    regularized = []
    not_regularized = []
    for name, param in student.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    params_groups = [
        {
            "params": regularized
        },
        {
            "params": not_regularized,
            "weight_decay": 0.0
        },
    ]

    lr = args.lr * args.batch_size / 256  # linear scaling rule
    optimizer = torch.optim.AdamW(params_groups)
    if state is not None:
        optimizer.load_state_dict(state["optimizer"])

    lr_schedule = cosine_scheduler(lr, args.min_lr, args.epochs,
                                   len(train_loader), args.warmup_epochs)
    wd_schedule = cosine_scheduler(args.weight_decay, args.weight_decay_end,
                                   args.epochs, len(train_loader))
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.epochs,
                                         len(train_loader))
    if args.use_amp:
        gradient_scaler = torch.cuda.amp.GradScaler()
    else:
        gradient_scaler = None

    loss_fn = iBOTLoss(
        out_dim=args.out_dim,
        patch_out_dim=args.patch_out_dim,
        ngcrops=2,
        nlcrops=0,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp2=0.04,
        teacher_temp2=0.07,
        warmup_teacher_temp_epochs=10,
        nepochs=args.epochs,
    ).to(DEVICE)
    if state is not None:
        loss_fn.load_state_dict(state["loss_fn"])

    start_epoch = 0 if state is None else state["epoch"]
    if state is not None:
        reproducibiliy.set_all_rng_states(state["rng_state"])

    for epoch in range(start_epoch, args.epochs):
        if args.ckpt_dir is not None:
            torch.save(
                {
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng_state": reproducibiliy.get_all_rng_states(),
                    "loss_fn": loss_fn.state_dict(),
                    "epoch": epoch,
                },
                state_path,
            )
        torch.save(teacher.state_dict(),
                   os.path.join(args.ckpt_dir, f"teacher_{epoch}.pt"))

        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            train_step = epoch * len(train_loader) + i

            X1, X2 = batch
            X1, X2 = X1.to(DEVICE), X2.to(DEVICE)
            # im = batch["bmode"]
            # generate masks
            masks = []
            for _ in range(X1.shape[0]):
                mask = MaskingGenerator(
                    (64, 64),
                    int(64 * 64 * 0.3),
                    min_num_patches=16,
                    max_num_patches=100,
                )()
                masks.append(torch.from_numpy(mask).bool())
            masks = torch.stack(masks).to(DEVICE)

            # X1, X2 = DataAugmentationDINO((0.5, 1))(im)

            # concatenate the two sets of images along
            # batch dimension to process them together
            X = torch.cat((X1, X2), dim=0)
            masks = torch.cat((masks, masks), dim=0)

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                student_output = student(X, masks)
                with torch.no_grad():
                    teacher_output = teacher(X)
                loss = loss_fn(student_output, teacher_output, None, masks,
                               epoch)["loss"]

                # set the lr and wd for the optimizer
                for j, param_group in enumerate(optimizer.param_groups):
                    param_group["lr"] = lr_schedule[train_step]
                    if j == 0:
                        param_group["weight_decay"] = wd_schedule[train_step]

            optimizer.zero_grad()
            if gradient_scaler is None:
                loss.backward()
                if args.clip_grad:
                    nn.utils.clip_grad_norm_(student.parameters(),
                                             args.clip_grad)
                optimizer.step()
            else:
                gradient_scaler.scale(loss).backward()
                if args.clip_grad:
                    gradient_scaler.unscale_(
                        optimizer
                    )  # unscale the gradients of optimizer's assigned params in-place
                    nn.utils.clip_grad_norm_(student.parameters(),
                                             args.clip_grad)
                gradient_scaler.step(optimizer)
                gradient_scaler.update()

            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": lr_schedule[train_step],
                    "wd": wd_schedule[train_step],
                    "momentum": momentum_schedule[train_step],
                    "epoch": epoch,
                },
            )

            # update teacher
            with torch.no_grad():
                m = momentum_schedule[train_step]
                for param_q, param_k in zip(student.parameters(),
                                            teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                if args.log_image_freq is not None and i % args.log_image_freq == 0:
                    show_example(teacher.image_encoder_wrapped, X1)
                    wandb.log({"example": wandb.Image(plt)})


@torch.no_grad()
def show_example(image_encoder, image):
    """Shows an example of the image and the PCA of the features"""
    image_encoder.eval()
    image_features = image_encoder.get_image_features(image)
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
    features_pca = einops.rearrange(features_pca,
                                    "(b h w) c -> b h w c",
                                    b=B,
                                    h=H,
                                    w=W)

    features_pca -= features_pca.min()
    features_pca /= features_pca.max()
    image = einops.rearrange(image, "b c h w -> b h w c").cpu().numpy()

    fig, ax = plt.subplots(B, 2)
    if B == 1:
        ax = ax[None, :]
    for i in range(B):
        ax[i, 0].imshow(image[i])
        ax[i, 1].imshow(features_pca[i])

    return fig


def build_dataloaders(
        include_nct=True,
        include_ua=False,
        batch_size=1,
        nct_options: NCTDataOptions = NCTDataOptions(),
        crop_scale_1=(0.5, 1),
        crop_scale_2=(0.5, 1),
        random_gamma=(0.7, 1.4),
):

    def transform(item):
        image = item["bmode"]
        image = torch.from_numpy(image.copy())[None,
                                               ...].repeat_interleave(3, 0)
        image1 = image.clone()
        image2 = image.clone()

        if random_gamma is not None:
            if np.random.rand() < 0.5:
                from torchvision.transforms.functional import adjust_gamma

                gamma = np.random.uniform(*random_gamma)
                image1 = adjust_gamma(image1, gamma, gain=gamma)

        if crop_scale_1 is not None:
            from torchvision.transforms import RandomResizedCrop

            crop1 = RandomResizedCrop((1024, 1024), scale=crop_scale_1)
            image1 = crop1(image1)

        if crop_scale_2 is not None:
            from torchvision.transforms import RandomResizedCrop

            crop2 = RandomResizedCrop((1024, 1024), scale=crop_scale_2)
            image2 = crop2(image2)

        image1 = image1.float() / 255
        image1 -= image1.min()
        image1 /= image1.max()

        image2 = image2.float() / 255
        image2 -= image2.min()
        image2 /= image2.max()

        return image1, image2

    datasets = []
    from medAI.datasets.nct2013.bmode_dataset import BModeDatasetV1
    from medAI.datasets.nct2013.cohort_selection import select_cohort
    from medAI.datasets.ua_unlabeled import UAUnlabeledImages

    if include_nct:
        train_cores = select_cohort(
            fold=nct_options.fold,
            n_folds=nct_options.n_folds,
            test_center=nct_options.test_center,
        )[0]
        datasets.append(BModeDatasetV1(train_cores, transform=transform))

    if include_ua:
        datasets.append(
            UAUnlabeledImages(
                root="/ssd005/projects/exactvu_pca/UA_extracted_data",
                transform=transform,
            ))

    dataset = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=8)


# IBot loss from ibot paper


# # Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class iBOTLoss(nn.Module):

    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        mim_start_epoch=0,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("center2", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp,
                        warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
        ))
        self.teacher_temp2_schedule = (np.concatenate((
            np.linspace(warmup_teacher_temp2, teacher_temp2,
                        warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
        )) if mim_start_epoch == 0 else np.concatenate((
            np.ones(mim_start_epoch) * warmup_teacher_temp2,
            np.linspace(warmup_teacher_temp2, teacher_temp2,
                        warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch) *
            teacher_temp2,
        )))

    def forward(self, student_output, teacher_output, student_local_cls,
                student_mask, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        teacher_cls, teacher_patch = teacher_output

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        # [CLS] and patch for global patches
        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center2) / temp2,
                                    dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        total_loss1, n_loss_terms1 = 0, 0
        total_loss2, n_loss_terms2 = 0, 0
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    loss2 = torch.sum(
                        -teacher_patch_c[q] *
                        F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss2 = torch.sum(loss2 * mask.float(),
                                      dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
                    total_loss2 += loss2.mean()
                    n_loss_terms2 += 1
                else:
                    loss1 = torch.sum(
                        -teacher_cls_c[q] *
                        F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss1 += loss1.mean()
                    n_loss_terms1 += 1

        total_loss1 = total_loss1 / n_loss_terms1 * self.lambda1
        total_loss2 = total_loss2 / n_loss_terms2 * self.lambda2
        total_loss = dict(cls=total_loss1,
                          patch=total_loss2,
                          loss=total_loss1 + total_loss2)
        self.update_center(teacher_cls, teacher_patch)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        """
        Update center used for teacher output.
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        cls_center = cls_center / (len(teacher_cls))
        self.center = self.center * self.center_momentum + cls_center * (
            1 - self.center_momentum)

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        patch_center = patch_center / (len(teacher_patch))
        self.center2 = self.center2 * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2)


class MedSAMIBot(nn.Module):

    def __init__(self, out_dim, patch_out_dim):
        # keep defaults for now

        super().__init__()
        from medAI.modeling.sam import build_medsam

        medsam_model = build_medsam()
        self.image_encoder_wrapped = ImageEncoderViTWithClassTokenAndMasking(
            medsam_model.image_encoder)
        self.image_encoder = self.image_encoder_wrapped.image_encoder  # for saving

        from src.heads_ibot import iBOTHead

        self.output_head = iBOTHead(
            in_dim=768,
            out_dim=out_dim,
            patch_out_dim=patch_out_dim,
            hidden_dim=1024,
            bottleneck_dim=256,
        )

    def forward(self, image, mask=None):
        outputs = self.image_encoder_wrapped(image, mask)
        outputs = self.output_head(outputs)
        return outputs


class ImageEncoderViTWithClassTokenAndMasking(nn.Module):

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        embedding_dim: int = 768,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.class_token_to_image_attns = nn.ModuleList([
            ClassTokenBlock(emb_dim=embedding_dim,
                            mlp_dim=embedding_dim * 4,
                            heads=8)
            for _ in range(len(self.image_encoder.blocks))
        ])
        self.mask_token = torch.nn.Parameter(torch.randn(embedding_dim))

    def forward(self, x, mask=None):
        x = self.image_encoder.patch_embed(x)

        if mask is not None:
            x[mask] = self.mask_token.to(x.dtype)

        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        cls_token = self.class_token.expand(x.shape[0], -1, -1)

        for blk, blk2 in zip(self.image_encoder.blocks,
                             self.class_token_to_image_attns):
            x = blk(x)
            cls_token = blk2(x, cls_token)

        # concatenate to typical output shape expected by vision transformers -
        # B, N, C where N is the number of patches + 1 (for the class token)
        # and class token is the first element along the N dimension
        B, H, W, C = x.shape
        x = x.reshape(B, H * W, C)
        x = torch.cat([cls_token, x], dim=1)

        return x

    def get_image_features(self, image):
        x = self.image_encoder.patch_embed(image)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed
        for blk in self.image_encoder.blocks:
            x = blk(x)
        # B H W C -> B C H W
        x = x.permute(0, 3, 1, 2)
        return x


class ClassTokenBlock(nn.Module):

    def __init__(self, emb_dim, mlp_dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                          num_heads=heads,
                                          batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, mlp_dim), nn.GELU(),
                                 nn.Linear(mlp_dim, emb_dim))

    def forward(self, image, cls_token):
        shortcut = cls_token
        B, H, W, C = image.shape
        image = image.reshape(B, H * W, C)
        cls_token = self.attn(cls_token, image, image)[0]
        cls_token = cls_token + shortcut
        cls_token = cls_token + self.mlp(cls_token)
        return cls_token


from torchvision import transforms


class DataAugmentationDINO(object):

    def __init__(self, global_crops_scale):
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(
                1024,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(
                1024,
                scale=global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        return crops


if __name__ == "__main__":
    from simple_parsing import ArgumentParser

    parser = ArgumentParser(
        description="Train the MedSAM model in the style of iBOT.")
    parser.add_arguments(Args, dest="args")
    args = parser.parse_args()
    main(args.args)
