import torch
import torch.nn as nn 
import torch.nn.functional as F
from dataclasses import dataclass, field
import timm
from timm.layers import create_classifier 


@dataclass
class FeatureExtractorConfig:
    model_name: str = 'resnet14t'
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
    inner_lr: float = 0.1
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
        self.feature_extractor = timm.create_model(
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
            input_size=self.feature_extractor.feature_info[-1]['num_chs'],
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
            
    def forward(self, x, training=False, use_predictor=False):
        self.feature_extractor.train() if training else self.feature_extractor.eval()
        self.classifier.train() if training else self.classifier.eval()
        
        latent = self.feature_extractor(x)
        latent = F.relu(self.last_bn(latent))
        latent = self.global_pool(latent)
        latent = F.relu(self.pre_hidden(latent)) 
        
        proj = self.projector(latent)
        if use_predictor:
            proj = self.predictor(proj)
            
        logits = self.classifier(latent)    
        return latent, proj, logits
            
        
class MT3Model(nn.Module):
    """Implements MT3 model."""
    
    def __init__(self, mt3_config: MT3Config) -> None:
        super().__init__()

        # Set the config
        self.inner_steps = mt3_config.inner_steps
        self.inner_lr = mt3_config.inner_lr
        self.beta_byol = mt3_config.beta_byol
        
        # Create YShapeModel
        self.y_shape_model = YShapeModel(mt3_config.y_shape_config)
        
                
    def forward(self, x):
        # Define the forward pass
        features = self.feature_extractor(x)
        # Additional forward pass logic can be implemented here
        # ...

        return features

    # Placeholder for the inner training loop (BYOL-like updates)
    def inner_train_loop(self, images_aug_1, images_aug_2):
        pass

    # Placeholder for the method to update the meta model
    def update_meta_model(self, updated_model, model, gradients):
        pass

    # Placeholder for the BYOL loss function
    def byol_loss_fn(self, x, y):
        pass
        
        