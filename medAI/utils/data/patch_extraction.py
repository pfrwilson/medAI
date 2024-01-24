from typing import Any
from skimage.transform import resize
import numpy as np
from tqdm import tqdm


__all__ = ["PatchView"]


class PatchView:
    """A class representing a view of an image as a collection of patches.

    Access patches through the [] operator.

    Args:
        image (array-like): The image to be viewed as patches. If the image is 2D, it is assumed to be grayscale.
            If the image is 3D, it is assumed to be RGB, with the last dimension being the color channel.
        positions (array-like): A list of positions of the patches. Each position is a list of 4 integers:
            [x1, y1, x2, y2], where (x1, y1) is the top left corner of the patch and (x2, y2) is the bottom right corner.
    """

    _cache = {}

    def __init__(self, image, positions):
        self.image = image
        self.positions = positions

    def __getitem__(self, index):
        x1, y1, x2, y2 = self.positions[index]

        return self.image[x1:x2, y1:y2]

    def __len__(self):
        return len(self.positions)

    @staticmethod
    def _sliding_window_positions(image_size, window_size, stride, align_to="topleft"):
        """
        Generate a list of positions for a sliding window over an image.

        Args:
            image_size (tuple): The size of the image.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].

        Returns:
            positions (array-like): A list of positions of the patches. Each position is a list of 4 integers:
                [x1, y1, x2, y2], where (x1, y1) is the top left corner of the patch and (x2, y2) is the bottom right corner.
        """

        if (image_size, window_size, stride, align_to) not in PatchView._cache:
            if len(image_size) == 2:
                x, y = image_size
            else:
                x, y, _ = image_size

            k1, k2 = window_size
            s1, s2 = stride

            positions = np.mgrid[0 : x - k1 + 1 : s1, 0 : y - k2 + 1 : s2]

            # if the last window is not flush with the image, we may need to offset the image slightly
            lastx, lasty = positions[:, -1, -1]
            lastx += k1
            lasty += k2
            if "bottom" in align_to:
                positions[0, :, :] += x - lastx
            if "right" in align_to:
                positions[1, :, :] += y - lasty

            positions = positions.reshape(2, -1).T
            positions = np.concatenate([positions, positions + window_size], axis=1)

            PatchView._cache[(image_size, window_size, stride, align_to)] = positions

        return PatchView._cache[(image_size, window_size, stride, align_to)]

    @staticmethod
    def from_sliding_window(
        image, window_size, stride, align_to="topleft", masks=[], thresholds=[]
    ):
        """Generate a PatchView from a sliding window over an image.

        This factory method can be used to generate a PatchView from a sliding window over an image.
        The sliding window can be filtered by a list of masks. If the mean of the mask in a window is greater than the corresponding threshold, the window is kept.

        Args:
            image (array-like): The image to be viewed as patches.
                If the image is 2D, it is assumed to be grayscale;
                if the image is 3D, it is assumed to be RGB, with the last dimension being the color channel.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].
            masks (array-like): A list of masks to apply to the sliding window. If the mean of the mask in a window is greater than the corresponding threshold, the window is kept.
                The masks should be 2-dimensional.
            thresholds (array-like): A list of thresholds for the masks.

        Returns:
            PatchView: A PatchView object.

        """
        positions = PatchView._sliding_window_positions(
            image.shape, window_size, stride, align_to=align_to
        )

        for mask, threshold in zip(masks, thresholds):
            filtered_positions = []
            for x1, y1, x2, y2 in positions:
                X, Y = image.shape[:2]
                X_mask, Y_mask = mask.shape[:2]

                # if the mask is of a different shape than the image,
                # we need to adjust the coordinates to be relative to the mask
                if X != X_mask:
                    x1 = int(x1 / X * X_mask)
                    x2 = int(x2 / X * X_mask)
                if Y != Y_mask:
                    y1 = int(y1 / Y * Y_mask)
                    y2 = int(y2 / Y * Y_mask)

                if np.mean(mask[x1:x2, y1:y2]) >= threshold:
                    filtered_positions.append([x1, y1, x2, y2])

            positions = np.array(filtered_positions)

        return PatchView(image, positions)

    @staticmethod
    def build_collection_from_images_and_masks(
        image_list,
        window_size,
        stride,
        align_to="topleft",
        mask_lists=[],
        thresholds=[],
    ):
        """Generate a collection of PatchViews from a collection of images and masks.

        Because this will vectorize the mask intersection calculations, it is much faster than calling from_sliding_window multiple times.
        However, this method requires that all images and masks are of the same size.

        Args:
            image_list (array-like): A list of images to be viewed as patches.
                If the images are 2D, they are assumed to be grayscale;
                if the images are 3D, they are assumed to be RGB, with the last dimension being the color channel.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].
            mask_lists (array-like): A list of lists of masks to apply to the sliding window. If the mean of the mask in a window is greater 
                than the corresponding threshold, the window is kept. The masks should be 2-dimensional. If more then one list of masks is provided,
                they will be applied in order to filter the windows.
            thresholds (array-like): A list of thresholds for the masks.
        """

        n_images = len(image_list)
        H, W = image_list[0].shape[:2]
        position_candidates = PatchView._sliding_window_positions(
            image_list[0].shape, window_size, stride, align_to=align_to
        )

        n_position_candidates = len(position_candidates)
        valid_position_candidates = np.ones(
            (n_images, n_position_candidates), dtype=bool
        )

        for mask_list, threshold in zip(mask_lists, thresholds):
            valid_position_candidates_for_mask = np.zeros(
                (n_images, n_position_candidates), dtype=bool
            )
            mask_arr = np.stack(mask_list, axis=-1)

            for idx in tqdm(range(n_position_candidates), desc="Applying mask"):
                x1, y1, x2, y2 = position_candidates[idx]
                x1 = int(x1 / H * mask_arr.shape[0])
                x2 = int(x2 / H * mask_arr.shape[0])
                y1 = int(y1 / W * mask_arr.shape[1])
                y2 = int(y2 / W * mask_arr.shape[1])

                valid_position_candidates_for_mask[:, idx] = (
                    mask_arr[x1:x2, y1:y2].mean(axis=(0, 1)) > threshold
                )

            valid_position_candidates *= valid_position_candidates_for_mask

        patch_views = []
        for idx in tqdm(range(n_images), desc="Filtering positions"):
            positions_for_image = []
            for j in range(n_position_candidates):
                if valid_position_candidates[idx, j]:
                    position = position_candidates[j]
                    positions_for_image.append(position)

            patch_views.append(PatchView(image_list[idx], positions_for_image))

        return patch_views

    def show(self, ax=None):
        """Show the patch view by plotting the patches on top of the image.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes is created.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        image = self.image
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        for x1, y1, x2, y2 in self.positions:
            ax.plot([y1, y2, y2, y1, y1], [x1, x1, x2, x2, x1], "r")
        ax.axis("off")
        return ax
