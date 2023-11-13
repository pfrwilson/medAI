import torch 


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