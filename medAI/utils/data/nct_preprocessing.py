"""
This file contains the preprocessing steps for the NCT dataset, 
including the conversion from IQ data to RF data, and the
stitching of focal zones.

The preprocessing steps are based on the MATLAB code provided by
Exact Imaging and adjusted by Amoon Jamzad.
"""


from math import floor
import einops
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import pandas as pd
from itertools import product
from skimage.transform import resize
from scipy.signal import lfilter

# ===================================================
# DEFAULT PARAMETERS TAKEN FROM THE MATLAB SCRIPT
# ===================================================

# Setup the Rx frequency
# from @moon matlab code

F0 = 17.5e6 * 2  # Rx Quad 2x
DEFAULT_RECONSTRUCTION_FREQ = F0 / 2

# Stitch params
DEFAULT_PARAMS = {
    "Depth": np.array([7, 16, 22]),
    "Boundaries": np.array([11.5000, 19]),
    "Corrections": np.array([8, 0.0400, 14, 0.0200, 17, 0.0600, 21, 0.0350]),
}

AXIAL_EXTENT = 28  # mm, length of image along axial dimention
LATERAL_EXTENT = 46.08  # mm, length of image along lateral dimention


# Interpolation filter
FILTER = [
    -2.2737367544323206e-13,
    -2.4312130025236911e-03,
    -4.5593193912054630e-03,
    -6.0867394810202313e-03,
    -6.7775138305705696e-03,
    -6.4858523272732782e-03,
    -5.1777598602029684e-03,
    -2.9436086272198736e-03,
    0.0000000000000000e00,
    1.3571640261943685e-02,
    2.5784400964312226e-02,
    3.4906633308764867e-02,
    3.9457774229958886e-02,
    3.8379988421638700e-02,
    3.1186609072392457e-02,
    1.8075864528327656e-02,
    -9.0949470177292824e-13,
    -4.4992860027377901e-02,
    -8.7701098444085801e-02,
    -1.2218482731623226e-01,
    -1.4265809342759894e-01,
    -1.4396752281299996e-01,
    -1.2204652839409391e-01,
    -7.4315408141046646e-02,
    0.0000000000000000e00,
    1.3696806975167419e-01,
    2.9100578523139120e-01,
    4.5221821498853387e-01,
    6.0983636066157487e-01,
    7.5295791229928000e-01,
    8.7130541983333387e-01,
    9.5595626632984931e-01,
    1.0000000000009095e00,
    9.5595626632984931e-01,
    8.7130541983333387e-01,
    7.5295791229928000e-01,
    6.0983636066157487e-01,
    4.5221821498853387e-01,
    2.9100578523139120e-01,
    1.3696806975167419e-01,
    0.0000000000000000e00,
    -7.4315408141046646e-02,
    -1.2204652839409391e-01,
    -1.4396752281299996e-01,
    -1.4265809342759894e-01,
    -1.2218482731623226e-01,
    -8.7701098444085801e-02,
    -4.4992860027377901e-02,
    -9.0949470177292824e-13,
    1.8075864528327656e-02,
    3.1186609072392457e-02,
    3.8379988421638700e-02,
    3.9457774229958886e-02,
    3.4906633308764867e-02,
    2.5784400964312226e-02,
    1.3571640261943685e-02,
    0.0000000000000000e00,
    -2.9436086272198736e-03,
    -5.1777598602029684e-03,
    -6.4858523272732782e-03,
    -6.7775138305705696e-03,
    -6.0867394810202313e-03,
    -4.5593193912054630e-03,
    -2.4312130025236911e-03,
    -2.2737367544323206e-13,
]

DELAY = 32


def upsample(array, factor):
    """upsample by filling between with zeros in the 0'th dimension"""

    out_shape = list(array.shape)
    out_shape[0] *= factor

    out = np.zeros(out_shape)

    out[::factor] = array

    return out


def interpolate(array, factor=8):
    """interpolates the given array along axis 0"""

    if not factor == 8:
        raise NotImplementedError("Factors other than 8 are not supported")

    # upsample the array by adding zeros
    array = upsample(array, factor)

    # apply the interpolation filter
    array = lfilter(FILTER, 1, array, axis=0)

    # adjust for the delay
    array = array[DELAY:]

    return array


def iq_to_rf(Q, I):

    reconstruction_freq = DEFAULT_RECONSTRUCTION_FREQ

    fs = reconstruction_freq
    f_rf = fs  # reconstruct to the original Rx freq in the param file

    fs = fs * 2  # actual Rx freq is double because of Quad 2x
    IntFac = 8
    fs_int = fs * IntFac

    bmode_n_samples = Q.shape[0]

    interpolation_factor = 8

    t = np.arange(
        0, (bmode_n_samples * interpolation_factor) / fs_int, 1 / fs_int
    ).reshape(-1, *((1,) * (Q.ndim - 1)))

    t = t[:-DELAY]

    Q_interp = interpolate(Q, interpolation_factor)
    I_interp = interpolate(I, interpolation_factor)

    rf = np.real(
        np.sqrt(I_interp**2 + Q_interp**2)  # type:ignore
        * np.sin(2 * np.pi * f_rf * t + np.arctan2(Q_interp, I_interp))
    )

    return rf


def stitch_focal_zones(img, depth=30, offset=2, params=DEFAULT_PARAMS):

    imgout = np.zeros((img.shape[0], img.shape[1] // 3))
    bound1 = round((params["Boundaries"][0] - offset) / (depth - offset) * img.shape[0])
    bound2 = round((params["Boundaries"][1] - offset) / (depth - offset) * img.shape[0])

    imgout[: bound1 - 1, :] = img[: bound1 - 1, 0 : img.shape[1] : 3]
    imgout[bound1 : bound2 - 1, :] = img[bound1 : bound2 - 1, 1 : img.shape[1] : 3]
    imgout[bound2:, :] = img[bound2:, 2 : img.shape[1] : 3]

    gaincurve = np.zeros(img.shape[0])
    depthvals = np.round(
        (params["Corrections"][0::2] - offset) / (depth - offset) * img.shape[0]
    ).astype("int")
    corrvals = params["Corrections"][1::2]

    samples_per_mm = img.shape[0] / (depth - offset)

    gaincurve[depthvals[0] : bound1 + 1] = (
        np.arange(1, (bound1 - depthvals[0] + 1) + 1) * corrvals[0] / samples_per_mm
    )
    gaincurve[bound1 + 1 : depthvals[1] + 1] = (
        np.arange(1, (depthvals[1] - bound1) + 1) * corrvals[1] / samples_per_mm
    )
    gaincurve[depthvals[2] : bound2 + 1] = (
        np.arange(1, (bound2 - depthvals[2] + 1) + 1) * corrvals[2] / samples_per_mm
    )
    gaincurve[bound2 + 1 : depthvals[3] + 1] = (
        np.arange(1, depthvals[3] - bound2 + 1) * corrvals[3] / samples_per_mm
    )

    imgout = imgout * einops.repeat(
        10 ** (gaincurve / 20), "axial -> axial lateral", lateral=imgout.shape[1]
    )

    return imgout


def stack_focal_zones(img):
    return einops.rearrange(img, "axial (lateral zone) -> axial lateral zone", zone=3)


def to_bmode(rf):
    return np.log(1 + np.abs(hilbert(rf)))


def downsample_axial(img, factor):
    return resize(img, (img.shape[0] // factor, img.shape[1]), anti_aliasing=True)


def DEFAULT_PREPROCESS_TRANSFORM(iq) -> np.ndarray:
    """Default preprocessing - turns iq to rf, selects last
    frame only, decimates the signal in the axial direction,
    and stitches focal zones if necessary
    """

    from scipy.signal import decimate

    # first frame only
    rf = iq_to_rf(iq["Q"][..., 0], iq["I"][..., 0])

    if rf.shape[1] > 512:
        rf = stitch_focal_zones(rf)

    # decimation by factor of 4 does not lose frequency information
    rf = decimate(rf, 4, axis=0)

    return rf