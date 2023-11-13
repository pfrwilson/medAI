"""
Implementation of SimCLR 
by Paul Wilson
"""


from torch import nn 
import torch 
from torch import distributed as dist
from .mlp import MLP
from .dist import gather


class SimCLRLoss(nn.Module): 
    def __init__(self, temperature): 
        super().__init__() 
        self.temperature = temperature

    def to_targets_and_logits(self, z1, z2):
        """Converts the features to targets and logits for the contrastive matching task"""

        if dist.is_initialized(): 
            # we might be in a distributed setting, so we should concatenate
            # all the features from different processes
            z1 = gather(z1)
            z2 = gather(z2)

        B, D = z1.shape # batch size and feature dimension

        Z = torch.cat([z1, z2], dim=0) # concatenate the features

        # make the logits for the matching task. The logits for i matching with j is 
        # the cosine similarity between the i'th and j'th feature vectors, scaled by the temperature
        Z = Z / torch.norm(Z, p=2, dim=1, keepdim=True) # normalize the features
        logits = torch.matmul(Z, Z.T) / self.temperature

        # mask the logits for the i'th example's positive pair being i, since this is not a valid pair
        logits = self.mask_simclr_logits(logits)

        # make the targets for the matching task. The positive pair for i is i+batch_size if i < batch_size
        # and i-batch_size otherwise
        targets = self.make_simclr_targets(B, Z.device)

        return targets, logits

    def __call__(self, z1, z2): 
        """
        computes the loss for the contrastive matching task
        """
        targets, logits = self.to_targets_and_logits(z1, z2)

        # compute the cross entropy loss
        loss = nn.CrossEntropyLoss()(logits, targets)
        return loss

    def make_simclr_targets(self, batch_size, device): 
        """Makes the tensor indicating the proper positive pair for a given sample in the batch"""
        targets = torch.zeros(batch_size * 2, dtype=torch.long, device=device)
        for i in range(batch_size): 
            targets[i] = i + batch_size
            targets[i + batch_size] = i
        return targets

    def mask_simclr_logits(self, logits): 
        """Masks the logits for the i'th example's positive pair being i, since this is not a valid pair"""
        PSEUDO_INF = 1e9

        mask = torch.zeros(logits.shape, dtype=torch.bool, device=logits.device)
        mask.fill_diagonal_(1)
        logits[mask] = -PSEUDO_INF
        return logits
    

class SimCLR(nn.Module): 
    def __init__(self, backbone, proj_dims, temperature=0.1): 
        super().__init__()

        # according to the paper, we should synchronize the batch norm across the 
        # different GPUS to avoid "cheating"
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(backbone)

        self.backbone = backbone
        self.projector = MLP(*proj_dims)
        self.loss_func = SimCLRLoss(temperature=temperature)

    def forward(self, X1, X2):
        """
        Compute the simclr loss for the two batches of views
        """
        X1 = self.backbone(X1)
        X2 = self.backbone(X2)

        X1 = self.projector(X1)
        X2 = self.projector(X2)

        loss = self.loss_func(X1, X2)
        return loss
    

class SimCLRTrainingLogic(nn.Module): 
    def __init__(self, backbone, proj_dims, temperature=0.1): 
        super().__init__()
        self.simclr = SimCLR(backbone, proj_dims, temperature=temperature).cuda()
        