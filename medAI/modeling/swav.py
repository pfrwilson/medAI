from torch import nn 
import torch 
import logging


class SwAV(nn.Module): 
    def __init__(self, backbone, features_dim, n_prototypes, n_stored_features, projector_dims=None, eps=0.05, n_sinkhorn_iters=3): 
        super().__init__()

        self.backbone = backbone 
        self.features_dim=features_dim
        self.n_prototypes=n_prototypes
        self.n_stored_features = n_stored_features
        self.projector_dims = projector_dims
        self.eps = eps
        self.n_sinkhorn_iters=n_sinkhorn_iters

        # sync batchnorm in case of multi-gpu training, where model 
        # might be able to "cheat" by using batch statistics to determine 
        # matching codes:
        nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)

        if projector_dims is not None: 
            from .mlp import MLP
            self.projector = MLP(*projector_dims)
        else: 
            self.projector = nn.Identity()

        self.Q = nn.Parameter(torch.randn(n_prototypes, features_dim))
        self.Q.requires_grad_(False)
        self.q_frozen = True 
        self.register_buffer('feature_queue', torch.randn(n_stored_features, features_dim))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    def forward(self, X1, X2): 
        z1 = self.projector(self.backbone(X1))
        z2 = self.projector(self.backbone(X2))

        P1, C1 = self.compute_p_and_c(z1)
        P2, C2 = self.compute_p_and_c(z2)

        loss = self.compute_loss(P1, C2) + self.compute_loss(P2, C1)

        self.add_features_to_queue(z1)
        return loss

    @torch.no_grad()
    def add_features_to_queue(self, z):
        ptr = int(self.queue_ptr)
        B, n_feats = z.shape 

        self.feature_queue[ptr:ptr+B] = z.detach().clone()
        ptr = (ptr + B) % self.n_stored_features
        self.queue_ptr[0] = ptr

    def compute_loss(self, p, c): 
        p = p.softmax(-1)
        cross_entropy = -(c * torch.log(p)).sum(-1)
        loss = cross_entropy.mean()
        return loss

    def compute_p_and_c(self, z): 
        B, n_feats = z.shape 

        # compute match scores with prototypes 
        z = nn.functional.normalize(z, p=2, dim=-1)
        Q = nn.functional.normalize(self.Q, p=2, dim=-1)
        
        P = z @ Q.T
        # Pij is match between feature i and prototype j
        
        with torch.no_grad(): 
            # compute match scores with cached features and prototypes 
            Z = self.feature_queue 
            Z = nn.functional.normalize(Z, p=2, dim=-1)
            P_ref = Z @ Q.T
            P_ref = torch.cat([P, P_ref])

            C = sinkhorn(P_ref, self.eps, self.n_sinkhorn_iters)
            C = C[:B, :]
            C = C.softmax(-1)

        return P, C 

    def epoch_end(self, epoch): 
        if self.q_frozen:
            print('Unfreezing prototype parameters')
            logging.info('Unfreezing prototype parameters')
            self.Q.requires_grad_(True)
            self.q_frozen = False


def sinkhorn(scores, eps=0.05, niters=3):
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
    
