import torch 
import logging


def view_as_windows_torch(image: torch.Tensor, window_size, step_size): 
    """
    Creates a view of the given image as a collection of windows of the given size and step size.
    This is a PyTorch implementation of skimage.util.view_as_windows.
    :param image: 4D tensor with dimensions (N, C, H, W)
    :param window_size: tuple of ints
    :param step_size: tuple of ints

    :return: 6D tensor with dimensions (N, C, H', W', window_size[0], window_size[1])
    """

    window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size)
    step_size = step_size if isinstance(step_size, tuple) else (step_size, step_size)

    assert image.ndim == 4, f'Image must be 4D tensor with dimensions (N, C, H, W), found {image.shape}'

    windowed_vertical = image.unfold(2, window_size[0], step_size[0])
    windowed = windowed_vertical.unfold(3, window_size[1], step_size[1])

    return windowed



class EarlyStopping:
    """Signals to stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.improvement = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            if self.verbose:
                logging.info(f'EarlyStopping - no improvement: {self.counter}/{self.patience}')
            self.improvement = False
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose: 
                    logging.info(f'EarlyStopping - patience reached: {self.counter}/{self.patience}')
                self.early_stop = True
        else:
            if self.verbose:
                logging.info(f'Early stopping - score improved from {self.best_score:.4f} to {score:.4f}')
            self.improvement = True
            self.best_score = score
            self.counter = 0

    def state_dict(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'improvement': self.improvement,
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_score = state_dict['best_score']
        self.early_stop = state_dict['early_stop']
        self.improvement = state_dict['improvement']
