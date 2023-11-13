import numpy as np
from einops import rearrange
import torch 
import matplotlib.pyplot as plt


def convert_physical_coordinate_to_pixel_coordinate(
    physical_coordinates,
    physical_extent,
    pixel_extent,
    physical_origin=(0, 0),
    round_to_int=True,
):
    """Converts a physical coordinate to a pixel coordinate.
    Args:
        physical_coordinates (tuple): the physical coordinates to convert
        physical_extent (tuple): the physical extent of the image
        physical_origin (tuple): the physical origin of the image
        pixel_extent (tuple): the pixel extent of the image
    Returns:
        tuple: the pixel coordinates
    """

    # convert everything to numpy arrays if they are not already
    physical_coordinates = np.array(physical_coordinates)
    physical_extent = np.array(physical_extent)
    physical_origin = np.array(physical_origin)
    pixel_extent = np.array(pixel_extent)

    # Normalize physical coordinates to [0, 1]
    normalized_coordinates = (physical_coordinates - physical_origin) / physical_extent
    # Convert normalized coordinates to pixel coordinates
    pixel_coordinates = normalized_coordinates * pixel_extent

    if round_to_int:
        pixel_coordinates = np.round(pixel_coordinates).astype(int)

    return pixel_coordinates


def sliding_window_slice_coordinates(
    window_size, strides, image_size, discard_from=None
):
    """Returns the slice coordinates for a sliding window of size window_size over
    and img of size img_size with stride lengths step_size.
    Args:
        window_size (tuple): the size of the sliding window
        strides (tuple): the stride lengths
        image_size (tuple): the size of the image
        discard_from (tuple):
            whether to discard from the start or end of the image. If None, defaults to "end" for all dimensions.
            If specified, must be a tuple of length len(window_size) with entries "start" or "end".
    Returns:
        np.ndarray: an array whose i'th entry contains the array whose j'th entry contains the slice coordinates in that dimension

    Example:
        >>> sliding_window_slice_coords((3, 3), (3, 3), (10, 10), discard_from=["end", "end"])
        array([[[0, 3],
                [3, 6]],
                [6, 9]],

                [[0, 3],
                [3, 6]],
                [6, 9]]])
        >>> x, y = sliding_window_slice_coords((3, 3), (3, 3), (10, 10), discard_from=["end", "end"])
        >>> A = np.fromfunction(lambda i, j: i + j, (10, 10))
        >>> x1, x2 = x[0]
        >>> y1, y2 = y[0]
        >>> A[x1:x2, y1:y2]
        array([[0., 1., 2.],
                [1., 2., 3.],
                [2., 3., 4.]])
    """

    inds = []
    assert len(window_size) == len(strides) == len(image_size)
    if discard_from is None:
        discard_from = ["end"] * len(window_size)
    for i in range(len(window_size)):
        startpos = np.arange(0, image_size[i], strides[i])
        while startpos[-1] + window_size[i] > image_size[i]:
            startpos = startpos[:-1]
        if discard_from[i] == "start":
            while startpos[-1] + window_size[i] < image_size[i]:
                startpos = startpos + 1
        inds.append(np.array([[j, j + window_size[i]] for j in startpos]))
    return tuple(inds)


def compute_n_windows_along_dimension(length, window_length, window_strides):
    return (length - window_length) // window_strides + 1


def show_image_and_patch(image, image_physical_extent, patch_topleft_physical, patch_topright_physical): 
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))



""" def sliding_window_heatmap_inference(
    model,
    img,
    img_extent_mm,
    window_size_mm,
    window_strides_mm,
    input_resolution,
    device=DEVICE,
):
    model.eval().to(device)

    h_patch, w_patch = convert_physical_coordinate_to_pixel_coordinate(
        window_size_mm, img_extent_mm, img.shape
    ).astype(int)

    h_stride, w_stride = convert_physical_coordinate_to_pixel_coordinate(
        window_strides_mm, img_extent_mm, img.shape
    ).astype(int)

    h, w = img.shape

    nh = compute_n_windows_along_dimension(h, h_patch, h_stride)
    ny = compute_n_windows_along_dimension(w, w_patch, w_stride)

    from torch import nn

    kernel_size = (h_patch, w_patch)
    stride = (h_stride, w_stride)

    fold_params = dict(kernel_size=kernel_size, stride=stride)
    fold = nn.Fold(output_size=img.shape, **fold_params)
    unfold = nn.Unfold(**fold_params)

    input = img[None, None, :, :]
    input = torch.tensor(input, dtype=torch.float32, device=device)
    input_ones = torch.ones(input.shape, dtype=input.dtype, device=input.device)
    divisor = fold(
        unfold(input_ones)
    )  # for each pixel, how many times was it folded over?

    # Compute the heatmap
    out = unfold(input)
    out = rearrange(
        out,
        "b (h_patch w_patch) (h w) -> (b h w) 1 h_patch w_patch",
        h=nh,
        w=ny,
        h_patch=h_patch,
        w_patch=w_patch,
    )

    from torchvision.transforms import Resize

    out = Resize(input_resolution)(out) """


"""
Helper classes for working with images in 2D space.

"""

# usual imports
import numpy as np
from typing import Optional, Any, Callable
from itertools import product
from skimage.transform import resize
from functools import cache


# represents an abstract n-dimensional lattice
class Lattice:
    def __init__(self, basis_matrix: np.ndarray, origin: np.ndarray = None):
        """
        basis_matrix: matrix of shape (d, n) where d is the dimension of the ambient space and n is the dimension of the lattice.
        the columns of the matrix are the basis vectors of the lattice.
        origin: vector of shape (d,) where d is the dimension of the ambient space
        """
        if origin is None:
            origin = np.zeros(basis_matrix.shape[0])
        origin = np.array(origin)
        self.basis_matrix = np.array(basis_matrix)
        self.inverse_basis_matrix = np.linalg.inv(basis_matrix)
        self.origin = origin.reshape(-1, 1)

    def _convert_input_points(self, points: np.ndarray):
        points = np.array(points)
        if len(points.shape) == 1:
            points = points.reshape(1, -1)
        points = points.T
        return points

    def _convert_output_points(self, points: np.ndarray):
        points = points.T
        if points.shape[0] == 1:
            points = points.squeeze()
        return points

    def lattice_coordinates_to_standard_coordinates(self, lattice_points: np.ndarray):
        """
        Maps lattice coordinates to standard coordinates.
        lattice_points: array of shape (n, d) where n is the number of points and d is the dimension of the lattice
        """
        lattice_points = self._convert_input_points(lattice_points)
        return self._convert_output_points(
            np.matmul(self.basis_matrix, lattice_points) + self.origin
        )

    def standard_coordinates_to_lattice_coordinates(
        self, standard_points: np.ndarray, snap_to_grid="floor"
    ):
        """
        Maps standard coordinates to lattice coordinates.
        standard_points: array of shape (n, d) where n is the number of points and d is the dimension of the ambient space
        snap_to_grid: "floor", "none", or "round"
        """
        standard_points = self._convert_input_points(standard_points)
        lattice_points = self._convert_output_points(
            np.matmul(self.inverse_basis_matrix, standard_points - self.origin)
        )
        if snap_to_grid == "round":
            lattice_points = np.round(lattice_points).astype(int)
        elif snap_to_grid == "floor" or snap_to_grid == True:
            lattice_points = np.floor(lattice_points).astype(int)
        elif snap_to_grid == "none":
            pass
        else:
            raise ValueError(f"Invalid snap_to_grid value: {snap_to_grid}")
        return lattice_points

    def lattice_point_to_new_lattice_point(
        self, lattice_points: np.ndarray, new_lattice: "Lattice"
    ):
        """
        Maps lattice points to new lattice points.
        lattice_points: array of shape (n, d) where n is the number of points and d is the dimension of the lattice
        """
        lattice_points = self._convert_input_points(lattice_points)
        return new_lattice.standard_coordinates_to_lattice_coordinates(
            self.lattice_coordinates_to_standard_coordinates(lattice_points)
        )

    def volume(self):
        """Returns the volume of a lattice cell."""
        return np.abs(np.linalg.det(self.basis_matrix))

    @staticmethod
    def orthogonal_lattice(spacings: np.ndarray, origin: np.ndarray = None):
        """Builds an orthogonal lattice with the given spacings and origin."""
        return Lattice(np.diag(spacings), origin)


class Image2D:
    """Represents a 2D image living in 2d space on top of a uniform orthogonal
    pixel grid."""

    def __init__(
        self,
        pixel_values: np.ndarray,
        pixel_spacing: Optional[np.ndarray] = None,
        physical_size: Optional[np.ndarray] = None,
        physical_origin_top_left: Optional[np.ndarray] = None,
        units: Optional[str] = None,
    ):
        self.pixel_values = pixel_values
        self.pixel_spacing = pixel_spacing
        self.physical_size = physical_size
        self.units = units
        if physical_origin_top_left is None:
            physical_origin_top_left = np.zeros(2)
        self.physical_origin = physical_origin_top_left

        # calculate pixel or physical size if not provided, if neither provided, raise error
        if self.pixel_spacing is None and self.physical_size is not None:
            self.pixel_spacing = self.physical_size / np.array(
                self.pixel_values.shape[:2]
            )
            self._physical_size_unknown = False
        elif self.physical_size is None and self.pixel_spacing is not None:
            self.physical_size = self.pixel_spacing * np.array(
                self.pixel_values.shape[:2]
            )
            self._physical_size_unknown = False
        elif self.physical_size is None and self.pixel_spacing is None:
            raise ValueError("Must provide either pixel or physical size")
        else:
            # make sure pixel and physical size are consistent
            assert np.allclose(
                self.physical_size,
                self.pixel_spacing * np.array(self.pixel_values.shape[:2]),
            ), "Pixel and physical size are inconsistent"

        self.pixel_lattice = Lattice.orthogonal_lattice(
            self.pixel_spacing, self.physical_origin
        )

        self.bottom_right = self.physical_origin + self.physical_size
        self.top_left = self.physical_origin

    def get_patch(
        self,
        top_left,
        bottom_right,
        return_pixels_only=False,
    ):
        """
        Returns a rectangular view of the image of size physical_size at origin physical_origin
        """
        # use lattice to convert points to pixel coordinates
        top_left_pixel = self.pixel_lattice.standard_coordinates_to_lattice_coordinates(
            top_left, snap_to_grid=True
        ).astype(int)
        bottom_right_pixel = (
            self.pixel_lattice.standard_coordinates_to_lattice_coordinates(
                bottom_right, snap_to_grid=True
            ).astype(int)
        )

        bottom_right = self.pixel_lattice.lattice_coordinates_to_standard_coordinates(
            bottom_right_pixel
        )
        top_left = self.pixel_lattice.lattice_coordinates_to_standard_coordinates(
            top_left_pixel
        )

        # get the view
        view = self.pixel_values[
            top_left_pixel[0] : bottom_right_pixel[0],
            top_left_pixel[1] : bottom_right_pixel[1],
        ]

        if return_pixels_only:
            return view
        else:
            # make a new image with the view, and the new pixel spacing
            return Image2D(
                view,
                self.pixel_spacing,
                physical_size=None,
                physical_origin_top_left=top_left,
                units=self.units,
            )

    def show_patch(self, top_left, bottom_right, **imshow_kwargs):    
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        self.show(ax[0], **imshow_kwargs)
        patch = self.get_patch(top_left, bottom_right)
        patch.show(ax[1], **imshow_kwargs)
        # show the patch outlined on the original image
        ax[0].plot(
            [top_left[1], top_left[1], bottom_right[1], bottom_right[1], top_left[1]],
            [top_left[0], bottom_right[0], bottom_right[0], top_left[0], top_left[0]],
            color="red",
        )

    def show(self, axis=None, **imshow_kwargs):
        import matplotlib.pyplot as plt

        if axis is None:
            axis = plt.gca()

        # # if the image is greyscale, use gray colormap
        # if len(self.pixel_values.shape) == 2:
        #     cmap = "gray"
        # else:
        #     # use default colormap
        #     cmap = None

        # physical extents of the image, matplotlib uses this to scale the image
        # order is left, right, bottom, top

        physical_extents = [
            self.physical_origin[1],
            self.physical_origin[1] + self.physical_size[1],
            self.physical_origin[0] + self.physical_size[0],
            self.physical_origin[0],
        ]

        axis.imshow(
            self.pixel_values, extent=physical_extents, **imshow_kwargs
        )
        axis.set_xlabel(f"J ({self.units})")
        axis.set_ylabel(f"I ({self.units})")

    def __repr__(self) -> str:
        return f"Image2D with shape {self.pixel_values.shape}, physical size {self.physical_size}, pixel spacing {self.pixel_spacing}, origin {self.physical_origin}, units {self.units}"

    def resize(self, new_pixel_size):
        """
        Returns a resized version of the same image
        """
        return Image2D(
            resize(self.pixel_values, new_pixel_size),
            physical_size=self.physical_size,
            pixel_spacing=None,
            physical_origin_top_left=self.physical_origin,
            units=self.units,
        )

    def pixel_sizes_match(self, other_image):
        """
        Returns true if the two images have the same physical size
        """
        return self.pixel_values.shape == other_image.pixel_values.shape

    def resize_like(self, other_image):
        """
        Returns a resized version of the same image
        """
        return self.resize(other_image.pixel_values.shape)


class Mask2D(Image2D):
    def __init__(
        self,
        pixel_values,
        physical_size=None,
        pixel_spacing=None,
        physical_origin_top_left=None,
        units=None,
    ):

        # convert pixel values to boolean
        pixel_values = pixel_values.astype(bool)

        super().__init__(
            pixel_values,
            physical_size=physical_size,
            pixel_spacing=pixel_spacing,
            physical_origin_top_left=physical_origin_top_left,
            units=units,
        )

    def show(
        self, axis=None, color="red", alpha=0.5, make_background_transparent=False
    ):

        import matplotlib.pyplot as plt
        import matplotlib.colors

        # colormap is one solid color
        # alpha is the transparency of the mask
        # make_background_transparent is a boolean which if true will make the background transparent

        # make the background transparent
        if make_background_transparent:
            pixel_values = self.pixel_values.copy().astype(float)
            pixel_values[pixel_values == 0] = np.nan
        else:
            pixel_values = self.pixel_values

        # make the colormap
        cmap = matplotlib.colors.ListedColormap([color])

        if axis is None:
            axis = plt.gca()

        # physical extents of the image, matplotlib uses this to scale the image
        # order is left, right, bottom, top

        physical_extents = [
            self.physical_origin[1],
            self.physical_origin[1] + self.physical_size[1],
            self.physical_origin[0] + self.physical_size[0],
            self.physical_origin[0],
        ]

        axis.imshow(
            pixel_values,
            extent=physical_extents,
            cmap=cmap,
            alpha=alpha,
        )

    def point_is_inside_mask(self, physical_point):
        """
        Tests if a physical point is inside the mask
        """
        pixel_point = self.pixel_lattice.standard_coordinates_to_lattice_coordinates(
            physical_point
        )
        return self.pixel_values[pixel_point[0], pixel_point[1]]

    def fraction_of_rectangle_inside_mask(self, top_left, top_right):
        """
        Returns the fraction of a physical rectangle that is inside the mask
        """
        # use the get rectangle method to get the pixel values of the rectangle
        pixel_values = self.get_patch(
            top_left, top_right, return_pixels_only=True
        )

        # return the fraction of pixels inside the mask
        return np.sum(pixel_values) / np.prod(pixel_values.shape)

    def rectangle_is_inside_mask(self, top_left, top_right, threshold=0.5):
        """
        Tests if a physical rectangle is inside the mask
        """
        # use the fraction of the rectangle inside the mask to test if the rectangle is inside the mask
        return self.fraction_of_rectangle_inside_mask(top_left, top_right) > threshold

    def fraction_of_image_intersection(self, image: Image2D):
        return self.fraction_of_rectangle_inside_mask(
            image.physical_origin, image.physical_origin + image.physical_size
        )

    def area(self):
        """
        Returns the area of the mask in physical units
        """
        return np.sum(self.pixel_values) * np.prod(self.pixel_spacing)

    # method to intersect two masks
    def intersect(self, other_mask):
        """
        Intersects two masks
        """
        # make sure sizes match
        assert self.pixel_sizes_match(other_mask), "Sizes do not match"

        return Mask2D(
            np.logical_and(self.pixel_values, other_mask.pixel_values),
            physical_size=self.physical_size,
            pixel_spacing=self.pixel_spacing,
            physical_origin_top_left=self.physical_origin,
            units=self.units,
        )

    # method to union two masks
    def union(self, other_mask):
        """
        Unions two masks
        """
        # makes sure sizes match
        assert self.pixel_sizes_match(other_mask), "Sizes do not match"

        return Mask2D(
            np.logical_or(self.pixel_values, other_mask.pixel_values),
            physical_size=self.physical_size,
            pixel_spacing=self.pixel_spacing,
            physical_origin_top_left=self.physical_origin,
            units=self.units,
        )

    def difference(self, other_mask):
        """
        Returns the difference of two masks
        """
        # makes sure sizes match
        assert self.pixel_sizes_match(other_mask), "Sizes do not match"

        return Mask2D(
            np.logical_and(self.pixel_values, np.logical_not(other_mask.pixel_values)),
            physical_size=self.physical_size,
            pixel_spacing=self.pixel_spacing,
            physical_origin_top_left=self.physical_origin,
            units=self.units,
        )

    def resize(self, new_pixel_size):
        """
        Resizes the mask to a new pixel size
        """
        # get the new pixel values
        new_pixel_values = resize(self.pixel_values, new_pixel_size, order=0)

        # return the new mask
        return Mask2D(
            new_pixel_values,
            physical_size=self.physical_size,
            physical_origin_top_left=self.physical_origin,
            units=self.units,
        )


class ImageCollection:
    def __init__(self, images: np.ndarray):
        self.images = images

    def apply(self, func: Callable[[Image2D], Any]):
        out = np.zeros_like(self.images)
        iterator = np.nditer(self.images, flags=["multi_index", "refs_ok"])
        for image in iterator:
            out[iterator.multi_index] = func(image.item())
        return ImageCollection(out) if isinstance(func(image.item()), Image2D) else out

    def iterator(self):
        return np.nditer(self.images, flags=["multi_index", "refs_ok"])

    def mask_intersections(self, mask: Mask2D):
        return self.apply(
            lambda image: mask.fraction_of_image_intersection(image)
        ).astype("float32")

    def __getitem__(self, item):
        return self.images[item]

    @staticmethod
    def image_to_raw_pixels(image: Image2D):
        return image.pixel_values

    def to_uniform_pixel_size(self, pixel_size):
        return self.apply(lambda image: image.resize(pixel_size))

    def raw_pixels(self):
        collection = self.apply(ImageCollection.image_to_raw_pixels)


class PatchViewGenerator:
    @staticmethod
    @cache
    def generate_sliding_window_coordinates(
        grid_shape: tuple,
        window_shape: tuple,
        strides,
        discard_patches_that_dont_fit_from="bl",
        as_grid_shaped_array=False,
    ):
        """
        Generates coordinates for a sliding window over a grid.
        grid_shape: shape of the grid
        window_shape: shape of the window
        strides: tuple of strides for each dimension
        discard_patches_that_dont_fit_from: string containing the following characters: "t", "b", "l", "r" (top, bottom, left, right)
        """
        if len(grid_shape) != len(window_shape):
            raise ValueError("grid_shape and window_shape must have the same length")
        if len(grid_shape) != len(strides):
            raise ValueError("grid_shape and strides must have the same length")
        if len(grid_shape) != 2:
            raise ValueError("only 2D grids are supported")
        if len(discard_patches_that_dont_fit_from) == 0:
            raise ValueError("discard_patches_that_dont_fit_from must not be empty")

        # discard patches that don't fit from the bottom left
        discard_from = discard_patches_that_dont_fit_from.lower()
        if "t" in discard_from and "b" in discard_from:
            raise ValueError(
                "discard_patches_that_dont_fit_from must not contain both t and b"
            )
        if "l" in discard_from and "r" in discard_from:
            raise ValueError(
                "discard_patches_that_dont_fit_from must not contain both l and r"
            )

        # compute the number of patches in each dimension
        n_patches = []
        for i in range(len(grid_shape)):
            n_patches.append((grid_shape[i] - window_shape[i]) // strides[i] + 1)

        # compute the coordinates of the patches
        coordinates = []
        for i in range(n_patches[0]):
            for j in range(n_patches[1]):
                coordinates.append((i * strides[0], j * strides[1]))

        # we will need to shift the grid coordinates into the correct corner
        shift = [0, 0]
        if "t" in discard_from:
            shift[0] = grid_shape[0] - (
                strides[0] * (n_patches[0] - 1) + window_shape[0]
            )
        if "l" in discard_from:
            shift[1] = grid_shape[1] - (
                strides[1] * (n_patches[1] - 1) + window_shape[1]
            )

        coordinates = np.array(coordinates) + np.array(shift).reshape(1, 2)

        if as_grid_shaped_array:
            coordinates = np.array(coordinates)
            coordinates = coordinates.reshape(n_patches[0], n_patches[1], 2)

        return coordinates

    @staticmethod
    def generate_patch_views(
        image: Image2D, patch_size, strides, discard_patches_that_dont_fit_from="bl"
    ):
        """
        Generates patch views of an image
        image: Image2D object
        patch_size: tuple containing the patch size in pixels
        strides: tuple containing the strides in pixels
        discard_patches_that_dont_fit_from: string containing the following characters: "t", "b", "l", "r" (top, bottom, left, right)
        as_grid_shaped_array: if True, the patch views are returned as a 3D array with shape (n_patches_x, n_patches_y, patch_size_x, patch_size_y)
        """
        # get the coordinates of the patches
        coordinates = PatchViewGenerator.generate_sliding_window_coordinates(
            tuple(image.physical_size),
            patch_size,
            strides,
            discard_patches_that_dont_fit_from,
            as_grid_shaped_array=True,
        )

        # get the patch views
        patch_views = np.zeros(coordinates.shape[:2], dtype=object)
        for i in range(coordinates.shape[0]):
            for j in range(coordinates.shape[1]):
                top_left = coordinates[i, j] + image.physical_origin
                bottom_right = coordinates[i, j] + patch_size + image.physical_origin
                patch_views[i, j] = image.get_patch(top_left, bottom_right)

        return ImageCollection(patch_views)


def show_all_patch_positions_on_image(image, positions, physical_image_shape): 
    plt.imshow(image, extent=(0, physical_image_shape[1], physical_image_shape[0], 0))
    for position in positions: 
        hmin, wmin, hmax, wmax = position 
        from matplotlib.patches import Rectangle
        # add rectangular patch which is partially transparent
        plt.gca().add_patch(Rectangle((wmin, hmin), wmax - wmin, hmax - hmin, alpha=0.5, color="red"))