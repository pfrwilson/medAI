import torch
import torch.nn as nn 
import torch.nn.functional as F
from functorch import combine_state_for_ensemble 
from torch.func import stack_module_state, vmap
from dataclasses import dataclass, field
import timm
from timm.layers import create_classifier 
import learn2learn as l2l
from copy import deepcopy 

from typing import List

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
class MT3Config:
    y_shape_config: YShapeConfig = YShapeConfig()
    inner_steps: int = 1
    inner_lr: float = 0.001
    beta_byol: float = 0.1 


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
            
        
class _MT3Model(nn.Module):
    """Implements MT3 model."""
    
    def __init__(
        self,
        spec_model,
        inner_steps,
        inner_lr,
        beta_byol,
        meta_batch_size,
        training=False,
        ) -> None:
        super().__init__()

        # Set the config
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.beta_byol = beta_byol
        
        self.spec_model = spec_model
        
        self.meta_batch_size = meta_batch_size
        self.training = training
    
    def forward(self, *args, **kwargs):
        return self.outer_train_loop(*args, **kwargs)

    def outer_train_loop(
        self,
        task_imgs_aug1,
        task_imgs_aug2,
        images,
        labels,
        ):
        # Clone the model
        maml_model = l2l.algorithms.MAML(self.spec_model.detach().clone(), lr=self.inner_lr, first_order=False)
        spec_model = maml_model.clone()
        
        # Run the inner training loop
        spec_model, byol_loss = self.inner_train_loop(task_imgs_aug1, task_imgs_aug2, spec_model)
        
        # Get the predictions 
        _, _, _, logits = spec_model(images, training=self.training)
        
        # Get the cross entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Get the totall loss
        total_loss = ce_loss + self.beta_byol * byol_loss
        
        # Backpropagate the total loss
        if self.training:
            # retain_graph=True to avoid RuntimeError. 
            # Divide by meta_batch_size to get the average loss over the meta batch.
            (total_loss/self.meta_batch_size).backward(retain_graph=True) 
        
        grad_dict = {}
        for name, param in maml_model.named_parameters():
            grad_dict[name] = param.grad.data.detach().clone()
        
        return logits.detach().cpu(), total_loss.item(), ce_loss.item(), byol_loss.item(), grad_dict
    
    def inner_train_loop(self, images_aug_1, images_aug_2, spec_model: l2l.algorithms.MAML):
        with torch.enable_grad(): # enable gradient for the inner loop
            
            for name, param in spec_model.named_parameters():
                param.requires_grad = True
                
            images_aug_1.requires_grad = True
            images_aug_2.requires_grad = True
            
            for i in range(self.inner_steps + 1):              
                _, z1, r1, logits1 = spec_model(images_aug_1, training=True)
                _, z2, r2, logits2 = spec_model(images_aug_2, training=True)

                
                # with torch.no_grad():
                # _, z1, _ = spec_model(images_aug_1, use_predictor=False, training=True)
                # _, z2, _ = spec_model(images_aug_2, use_predictor=False, training=True)
            
                # Calculate the loss
                loss1 = self.byol_loss_fn(r1, z2.detach())
                loss2 = self.byol_loss_fn(r2, z1.detach())
                byol_loss = loss1 + loss2
                
                if i == self.inner_steps: # similar to mt3 original code (maybe useless)
                    break
                # Update the model
                spec_model.adapt(byol_loss.mean(), allow_unused=True)
        
            return spec_model, byol_loss.mean()
   
    def byol_loss_fn(self, r, z):
        r = F.normalize(r, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (r * z).sum(dim=-1)

    # def forward(self, func):
    #     def wrapper(*args, **kwargs):
    #         spec_model = self.model.clone()
    #         return func(spec_model, *args, **kwargs)
    #     return wrapper

    
class MT3Model(nn.Module):
    def __init__(self, mt3_config: MT3Config) -> None:
        super().__init__()

        self.mt3_config = mt3_config
                
        # Create YShapeModel
        self.model = YShapeModel(mt3_config.y_shape_config)
        # self.model = l2l.algorithms.MAML(self.y_shape_model, lr=self.mt3_config.inner_lr, first_order=False)
    
    def forward(self, batch_images_aug_1, batch_images_aug_2, batch_images, batch_labels, training=False):  
        meta_batch_size = batch_images_aug_1.shape[0]
        num_support_imgs = batch_images_aug_1.shape[1]
        
        # Clone _MT3Model to get a new model for each batch in meta batch
        _mt3_models = [
            _MT3Model(
                deepcopy(self.model), 
                self.mt3_config.inner_steps,
                self.mt3_config.inner_lr,
                self.mt3_config.beta_byol,
                meta_batch_size,
                training=training
                ) for _ in range(meta_batch_size)
            ]
            
        minibatches = (
            batch_images_aug_1[:, :, None, ...], # (meta_bz, num_support_imgs, 1, H, W)
            batch_images_aug_2[:, :, None, ...],
            batch_images,
            batch_labels,
            )
        
        # def wrapper(params, buffers, batch_images_aug_1, batch_images_aug_2, batch_images, batch_labels):
        #     return torch.func.functional_call(_mt3_models[0], (params, buffers), (batch_images_aug_1, batch_images_aug_2, batch_images, batch_labels))
        
        # params, buffers = stack_module_state(_mt3_models)
        
        # logits, total_loss, ce_loss, byol_loss, grad_dict = vmap(wrapper, (0, 0, 0, 0, 0, 0))(
        #     params,
        #     buffers,
        #     batch_images_aug_1[:, :, None, ...], # (meta_bz, num_support_imgs, 1, H, W)
        #     batch_images_aug_2[:, :, None, ...],
        #     batch_images,
        #     batch_labels,)
        
        
        fmodel, params, buffers = combine_state_for_ensemble(_mt3_models)
        model = vmap(fmodel, (0, 0, 0, 0, 0, 0))
        
        logits, total_loss, ce_loss, byol_loss = model(params, buffers,             
                                                       batch_images_aug_1[:, :, None, ...], # (meta_bz, num_support_imgs, 1, H, W)
                                                       batch_images_aug_2[:, :, None, ...],
                                                       batch_images,
                                                       batch_labels,)
               
        return logits, total_loss.mean(dim=0), ce_loss.mean(dim=0), byol_loss.mean(dim=0)
        
     
class MT3ANILModel(MT3Model):
    config_class = MT3Config
    config: MT3Config
    
    def __init__(self, mt3_config: MT3Config) -> None:
        super().__init__(mt3_config)
        self.model = None
        
        self.feature_extractor = self.y_shape_model.feature_extractor
        self.pre_hidden = self.y_shape_model.pre_hidden
        self.classifier = self.y_shape_model.classifier
        self.pre_hidden_ANIL = l2l.algorithms.MAML(self.pre_hidden, lr=self.inner_lr, first_order=False)
    
    def forward(self, batch_images_aug_1, batch_images_aug_2, batch_images, batch_labels, training=False):  
        meta_batch_size = batch_images_aug_1.shape[0]
        num_support_imgs = batch_images_aug_1.shape[1]
        image_size = batch_images_aug_1.shape[-2:]
        
        # Concatenate the images in one big batch
        big_batch_images = torch.cat([
            batch_images_aug_1.reshape(-1, *image_size),
            batch_images_aug_2.reshape(-1, *image_size),
            batch_images[:, 0, ...]
            ], dim=0)
        
        # Get the features of the big batch
        big_features = self.feature_extractor(big_batch_images[:, None, ...])[-1]
        big_features = F.relu(self.y_shape_model.last_bn(big_features))
        big_features = self.y_shape_model.global_pool(big_features)
        
        # Split the features back
        batch_features_aug_1 = big_features[
            :num_support_imgs*meta_batch_size
            ].reshape(meta_batch_size, num_support_imgs, -1)
        batch_features_aug_2 = big_features[
            num_support_imgs*meta_batch_size:2*num_support_imgs*meta_batch_size
            ].reshape(meta_batch_size, num_support_imgs, -1)
        batch_features = big_features[2*num_support_imgs*meta_batch_size:]
           
        
        # List of all specialized pre_hidden models
        spec_pre_hiddens = []
              
        # Append specialized pre_hidden models and latents to the lists
        for batch_idx in range(meta_batch_size):
            # Clone pre_hidden to get a new model for each batch in meta batch
            spec_pre_hidden = self.pre_hidden_ANIL.clone()
            spec_pre_hiddens.append(spec_pre_hidden)
        
        spec_pre_hiddens, batch_byol_loss = self.inner_train_loop(
            batch_features_aug_1, 
            batch_features_aug_2, 
            spec_pre_hiddens, 
            training=training
            )            

        # Get the predictions and cross entropy loss
        batch_latents = self.pre_hidden_ANIL(batch_features)
        batch_logits = self.classifier(batch_latents)
        batch_ce_loss = F.cross_entropy(batch_logits, batch_labels)
        
        # Get the totall loss
        total_loss = batch_ce_loss + self.beta_byol * batch_byol_loss
        
        total_loss_avg = total_loss.mean()
        ce_loss_avg = batch_ce_loss.mean()
        byol_loss_avg = batch_byol_loss.mean()
        
        # Backpropagate the total loss
        if training:
            # retain_graph=True to avoid RuntimeError. 
            # Divide by meta_batch_size to get the average loss over the meta batch.
            (total_loss_avg).backward(retain_graph=True) 
        
        return batch_logits.detach(), total_loss_avg.detach(), ce_loss_avg.detach(), byol_loss_avg.detach()
        
    def inner_train_loop(
        self, 
        batch_features_aug_1,
        batch_features_aug_2, 
        spec_pre_hiddens: List[l2l.algorithms.MAML], 
        training=False
        ):
        
        meta_batch_size = batch_features_aug_1.shape[0]
        num_support_imgs = batch_features_aug_1.shape[1]
        
        for i in range(self.inner_steps + 1):
            # List of latents after pre_hidden for each augmented image
            batch_latents_aug_1 = []
            batch_latents_aug_2 = []
            
            for batch_idx, spec_pre_hidden in enumerate(spec_pre_hiddens):
                # Forward features through the pre_hidden
                latents_aug_1 = spec_pre_hidden(batch_features_aug_1[batch_idx]) # shape [num_support_imgs, proj_hidden_size]
                latents_aug_2 = spec_pre_hidden(batch_features_aug_2[batch_idx]) # shape [num_support_imgs, proj_hidden_size]
                
                # Append the latents to the list
                batch_latents_aug_1.append(latents_aug_1)
                batch_latents_aug_2.append(latents_aug_2)
            
            # Concatenate the latents
            batch_latents_aug_1 = torch.cat(batch_latents_aug_1, dim=0)
            batch_latents_aug_2 = torch.cat(batch_latents_aug_2, dim=0)
            
            batch_z_proj, batch_r_pred = self.y_shape_model.ssl_forward_with_latents(
                torch.cat(
                    [batch_latents_aug_1,
                     batch_latents_aug_2],
                    dim=0
                    ),
                training=training
            )
            
            batch_z_proj1, batch_r_pred1 = batch_z_proj[:num_support_imgs*meta_batch_size], \
                batch_r_pred[:num_support_imgs*meta_batch_size]
            batch_z_proj2, batch_r_pred2 = batch_z_proj[num_support_imgs*meta_batch_size:], \
                batch_r_pred[num_support_imgs*meta_batch_size:]
                
            batch_loss1 = self.byol_loss_fn(batch_r_pred1, batch_z_proj2.detach()).reshape(meta_batch_size, num_support_imgs)
            batch_loss2 = self.byol_loss_fn(batch_r_pred2, batch_z_proj1.detach()).reshape(meta_batch_size, num_support_imgs)
            batch_byol_loss = batch_loss1 + batch_loss2
            
            for batch_idx, spec_pre_hidden in enumerate(spec_pre_hiddens):
                # Update the model
                if i == self.inner_steps: # similar to mt3 original code (maybe useless)
                    break
                spec_pre_hidden.adapt(batch_byol_loss[batch_idx].mean(), allow_unused=True)
            
        return spec_pre_hiddens, batch_byol_loss.mean(dim=1)
   