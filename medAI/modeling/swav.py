from torch import nn 
import torch 
import logging


class SwAV(nn.Module): 
    def __init__(self, backbone, features_dim, n_prototypes, n_stored_features, projector_dims=None, lambda_=20, n_sinkhorn_iters=10): 
        super().__init__()

        self.backbone = backbone 
        self.features_dim=features_dim
        self.n_prototypes=n_prototypes
        self.n_stored_features = n_stored_features
        self.projector_dims = projector_dims
        self.lambda_ = lambda_
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

            # compute optimal assignment of prototypes to features
            n_targets = P_ref.shape[0]
            n_sources = P_ref.shape[1]
            K = torch.exp(self.lambda_ * P_ref) # starting point for sinkhorn iterations 

            # each feature should be the target of partial assignments adding up to 1
            target_row_sum = torch.ones((n_targets, 1), dtype=P_ref.dtype, device=P_ref.device)

            # each source should be assigned to the targets uniformly, so that the total number of 
            # assignments by a certain target should sum to n_targets/n_sources
            target_column_sum = torch.ones((n_sources, 1), dtype=P_ref.dtype, device=P_ref.device) * n_targets / n_sources

            C = sinkhorn_knopp(K, target_row_sum, target_column_sum, n_iters=self.n_sinkhorn_iters)
            C = C[:B, :]

        return P, C 

    def epoch_end(self, epoch): 
        if self.q_frozen:
            print('Unfreezing prototype parameters')
            logging.info('Unfreezing prototype parameters')
            self.Q.requires_grad_(True)
            self.q_frozen = False


def sinkhorn_knopp(K, r, c, n_iters=10): 
    """Uses Sinkhorn Knopp iteration algorithm to compute the unique 
    matrix of the form P = diag(u) @ K @ diag(v) such that the row
    sums of P are r and the column sums of P are c.                                                                      
    """
    assert torch.sum(r) == torch.sum(c), "Marginals must sum to the same value"
    assert torch.all(torch.abs(K) == K), "Input matrix must be positive"

    u = torch.ones_like(r)
    v = torch.ones_like(c)

    for _ in range(n_iters): 
        u = r / (K @ v) 
        v = c / (K.T @ u)

    u = torch.diag_embed(u[:, 0]) 
    v = torch.diag_embed(v[:, 0]) 
    P = u @ K @ v 

    return P