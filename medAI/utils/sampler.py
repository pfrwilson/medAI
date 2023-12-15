import torch 
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import math



class StatefulSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data = data_source
        self.shuffle = shuffle

        # initial dataloader index
        self.init_index()

    def init_index(self):

        if self.shuffle:
            self.indices = torch.randperm(len(self.data))
        else:
            self.indices = torch.arange(len(self.data))

        self.data_counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.data)

    def __next__(self):
        if self.data_counter == len(self.data):
            self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = \
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size
        return {
                'indices': self.indices,
                'data_counter': self.data_counter - prefetched_num,
            }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.data_counter = state_dict['data_counter']


class GenericStatefulSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, index_generator, n_samples):
        self.index_generator = index_generator
        self.n_samples = n_samples

        # initial dataloader index
        self.init_index()

    def init_index(self):
        self.indices = self.index_generator()
        self.data_counter = 0

    def __iter__(self):
        return self
    
    def __len__(self): 
        return self.n_samples

    def __next__(self):
        if self.data_counter == len(self.indices):
            self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter]
            self.data_counter += 1
            return int(ele)

    def state_dict(self, dataloader_iter=None):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = \
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size
        return {
                'indices': self.indices,
                'data_counter': self.data_counter - prefetched_num,
            }

    def load_state_dict(self, state_dict):
        self.indices = state_dict['indices']
        self.data_counter = state_dict['data_counter']


class GenericDistributedStatefulSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, index_generator, n_samples, world_size=None, rank=None):
        assert dist.is_initialized(), 'Distributed environment is not initialized.'
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()

        self.index_generator = index_generator
        self.n_samples = n_samples

        # initial dataloader index
        self.init_index()

    def generate_indices(self): 
        indices = self.index_generator()
        # drop last to make sure the number of samples is divisible by world_size 
        num_indices_after_truncation = len(indices) - (len(indices) % dist.get_world_size())
        indices = indices[:num_indices_after_truncation]
        return indices
    
    def init_index(self):
        indices = torch.tensor(self.generate_indices()).cuda()
        dist.broadcast(indices, 0)
        
        self.indices = indices
        self.data_counter = 0

    def __iter__(self):
        return self
    
    def __len__(self):
        return ( self.n_samples - (self.n_samples % self.world_size) ) // self.world_size
    
    def __next__(self):
        if self.data_counter == len(self):
            self.init_index()
            raise StopIteration()
        else:
            ele = self.indices[self.data_counter * self.world_size + self.rank]
            self.data_counter += 1
            return int(ele)

    def data_count(self, dataloader_iter):
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = \
                    (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size

        return self.data_counter - prefetched_num
    
    def state_dict(self, dataloader_iter):
        return {
                'indices': self.indices,
                'data_counter': self.data_count(dataloader_iter),
            }
    
    def load_state_dict(self, state): 
        self.indices = state['indices']
        self.data_counter = state['data_counter']
