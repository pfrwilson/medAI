from torch import nn 
import torch 
import copy 


class MoCo(nn.Module): 
    def __init__(self, backbone, k_dim, queue_size, momentum=0.999, temperature=0.01, projector_dims=None):
        super().__init__()
        if projector_dims is not None: 
            from .mlp import MLP
            projector = MLP(*projector_dims)
        else: 
            projector = nn.Identity()
        
        self.q_encoder = torch.nn.Sequential(backbone, projector)
        self.k_encoder = copy.deepcopy(self.q_encoder)

        self.features_dim = k_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        self.register_buffer('queue', torch.randn(queue_size, k_dim))
        self.register_buffer('queue_pointer', torch.zeros(0, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self): 
        for p1, p2 in zip(self.q_encoder.parameters(), self.k_encoder.parameters()): 
            p2.data = self.momentum * p2.data + (1 - self.momentum) * p1.data

    def _enqueue_and_dequeue(self, k): 
        batch_size = k.shape[0]
        ptr = int(self.queue_ptr)
        self.queue[ptr:ptr+batch_size, :] = k
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k): 
        q = self.q_encoder(im_q)
        q = nn.functional.normalize(q)

        with torch.no_grad(): 
            self._momentum_update()
            k = self.k_encoder(im_k)
            k = nn.functional.normalize(k, p=2)
            all_k = torch.concat([k, self.queue])
    
        logits = q @ all_k.T  # n X n_keys logits tensor 
        logits /= self.temperature

        # target for position i is position i 
        targets = torch.arange(len(q), dtype=torch.long, device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, targets)

        return loss 
    
        
