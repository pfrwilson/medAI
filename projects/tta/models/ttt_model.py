import torch
import torch.nn as nn 
import torch.nn.functional as F
from dataclasses import dataclass, field
import timm
from timm.layers import create_classifier 
import learn2learn as l2l

from typing import List
from copy import deepcopy

@dataclass
class FeatureExtractorConfig:
    model_name: str = 'resnet10t'
    num_classes: int = 2
    in_chans: int = 1
    features_only: bool = True # return features only, not logits
    
    def __post_init__(self):
        valid_models = timm.list_models()
        if self.model_name not in valid_models:
            raise ValueError(f"'{self.model_name}' is not a valid model. Choose from timm.list_models(): {valid_models}")


@dataclass
class YShapeConfig:
    feature_extractor: FeatureExtractorConfig = FeatureExtractorConfig()
    num_groups: int = 8
    proj_hidden_size: int = 256
    proj_size: int = 128
    use_mlp_norm: bool = True


@dataclass
class TTTConfig:
    y_shape_config: YShapeConfig = YShapeConfig()
    adaptation_steps: int = 1
    beta_byol: float = 0.1 
    joint_training: bool = False


class PPModelMLP(nn.Module):
    """
    MLP for predictor/projector in PyTorch
    """
    def __init__(
        self,
        input_size,
        hidden_size,
        projection_size,
        group_norm_groups,
        use_mlp_norm,
        ):
        super().__init__()
        #TODO: add weight decay to the optimizer
        
        self.dense_in = nn.Linear(input_size, hidden_size)
        self.use_mlp_norm = use_mlp_norm

        if self.use_mlp_norm:
            self.bn1 = nn.GroupNorm(num_groups=group_norm_groups, num_channels=hidden_size)

        self.dense1_out = nn.Linear(hidden_size, projection_size)

    def forward(self, x):
        x = self.dense_in(x)
        if self.use_mlp_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dense1_out(x)
        
        return x


class YShapeModel(nn.Module):
    def __init__(self, config: YShapeConfig) -> None:
        super().__init__()

        # Create the feature extractpr
        self.feature_extractor: nn.Module = timm.create_model(
            config.feature_extractor.model_name,
            num_classes=config.feature_extractor.num_classes,
            in_chans=config.feature_extractor.in_chans,
            features_only=config.feature_extractor.features_only,
            norm_layer=lambda channels: nn.GroupNorm(num_groups=config.num_groups, num_channels=channels)
            )
        
        # Create the last batch norm layer
        self.last_bn = nn.GroupNorm(
            num_groups=config.num_groups,
            num_channels=self.feature_extractor.feature_info[-1]['num_chs'],
            )
        
        # Separate creation of classifier and global pool from feature extractor
        self.global_pool, self.pre_hidden = create_classifier(
            self.feature_extractor.feature_info[-1]['num_chs'],
            #config.feature_extractor.num_classes,
            config.proj_hidden_size,
            pool_type='avg'
            )
        
        
        self.projector = PPModelMLP(
            input_size=config.proj_hidden_size, # self.feature_extractor.feature_info[-1]['num_chs'],
            hidden_size=config.proj_hidden_size,
            projection_size=config.proj_size,
            use_mlp_norm=config.use_mlp_norm,
            group_norm_groups=config.num_groups,
        )
        
        self.predictor = PPModelMLP(
            input_size=config.proj_size,
            hidden_size=config.proj_hidden_size,
            projection_size=config.proj_size,
            use_mlp_norm=config.use_mlp_norm,
            group_norm_groups=config.num_groups,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.proj_hidden_size, config.feature_extractor.num_classes),
            nn.Softmax(dim=-1)
            )
    
    def ssl_forward_with_latents(self, latents, training=False):
        z_proj = self.projector(latents)
        r_pred = self.predictor(z_proj)
        
        return z_proj, r_pred
    
    def forward(self, x, training=False, use_predictor=False):
        self.feature_extractor.train() if training else self.feature_extractor.eval()
        self.classifier.train() if training else self.classifier.eval()
        
        latent = self.feature_extractor(x)[-1]
        latent = F.relu(self.last_bn(latent))
        latent = self.global_pool(latent)
        latent = F.relu(self.pre_hidden(latent)) 
        
        z_proj = self.projector(latent)
        r_pred = self.predictor(z_proj)
            
        logits = self.classifier(latent)    
        return latent, z_proj, r_pred, logits
            
        
class TTTModel(nn.Module):
    """Implements TTT model."""
    
    def __init__(self, ttt_config: TTTConfig) -> None:
        super().__init__()

        # Set the config
        self.adaptation_steps = ttt_config.adaptation_steps
        self.beta_byol = ttt_config.beta_byol
        self.joint_training = ttt_config.joint_training
        
        # Create YShapeModel as model
        self.model = YShapeModel(ttt_config.y_shape_config)
        
    def forward(self, batch_images_aug_1, batch_images_aug_2, batch_images, batch_labels, training=False):  
        batch_size = batch_images_aug_1.shape[0]
        num_support_imgs = batch_images_aug_1.shape[1]
        image_size = batch_images_aug_1.shape[-2:]

        # Concatenate the images in one big batch
        big_batch_images = torch.cat([
            batch_images_aug_1.reshape(-1, *image_size),
            batch_images_aug_2.reshape(-1, *image_size),
            batch_images[:, 0, ...],
            ], dim=0)
        
        # Prepare test time training 
        adaptation_steps = 1 if training else self.adaptation_steps
        model = self.model if training else deepcopy(self.model)
        optimizer = None if training else torch.optim.SGD(model.parameters(), lr=1e-4)
        
        # Train or test time train
        for i in range(adaptation_steps):
            # Forward
            _, big_batch_z_proj, big_batch_r_pred, big_batch_logits = model(
                big_batch_images[:, None, ...],
                training=True, # always train group norm
                )
            
            # Split the z_proj and r_pred back
            z_proj_aug_1 = big_batch_z_proj[:num_support_imgs*batch_size]
            p_pred_aug_1 = big_batch_r_pred[:num_support_imgs*batch_size]
            z_proj_aug_2 = big_batch_z_proj[num_support_imgs*batch_size:2*num_support_imgs*batch_size]
            p_pred_aug_2 = big_batch_r_pred[num_support_imgs*batch_size:2*num_support_imgs*batch_size]
            
            # Get the byol loss
            byol_losses = self.byol_loss_fn(p_pred_aug_1, z_proj_aug_2.detach()) + \
                self.byol_loss_fn(p_pred_aug_2, z_proj_aug_1.detach())
            batch_byol_loss = byol_losses.reshape(batch_size, num_support_imgs).mean(dim=1)
                
            # Get the totall loss
            batch_total_loss = self.beta_byol * batch_byol_loss
            
            # Add cross entropy loss if training
            if training:
                # Split the logits back
                batch_logits = big_batch_logits[2*num_support_imgs*batch_size:]
                # Get the cross entropy loss
                batch_ce_loss = F.cross_entropy(batch_logits, batch_labels, reduction='none')
                batch_total_loss += batch_ce_loss
                # Backpropagate the total loss for both training and testing
                batch_total_loss.mean().backward() 
            else:
                if not self.joint_training:
                    optimizer.zero_grad()
                    batch_total_loss.mean().backward()
                    optimizer.step()
        
        
        if not training:
            # Get test logits
            _, _, _, batch_logits = model(
                batch_images,
                training=False, # always train group norm
                )
            batch_ce_loss = F.cross_entropy(batch_logits, batch_labels, reduction='none')
            
        # Sum the losses
        ce_loss = batch_ce_loss.mean().item()
        byol_loss = batch_byol_loss.mean().item()
        total_loss = ce_loss + self.beta_byol * byol_loss
        
        return batch_logits.detach().cpu(), total_loss, ce_loss, byol_loss
   
    def byol_loss_fn(self, r, z):
        r = F.normalize(r, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (r * z).sum(dim=-1)
        

# TODO: add needle images itself to the byol loss