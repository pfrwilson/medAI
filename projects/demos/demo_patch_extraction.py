PATH = "/ssd005/projects/exactvu_pca/nct2013/bmode/CRCEO-0004_LBM_Benign.npy"
import matplotlib.pyplot as plt
import numpy as np

A = np.load(PATH)
print(A.shape)
plt.imshow(A[..., 0], cmap='gray')

from medAI.datasets.nct2013.data_access import data_accessor
from medAI.utils.data.patch_extraction import PatchView

bmode_image= data_accessor.get_bmode_image('CRCEO-0004_LBM', frame_idx=0)
needle_mask = data_accessor.get_needle_mask('CRCEO-0004_LBM')
prostate_mask = data_accessor.get_prostate_mask('CRCEO-0004_LBM')

print(needle_mask.min(), needle_mask.max())

plt.imshow(needle_mask, cmap='gray')
from skimage.transform import resize

needle_mask = resize(needle_mask,
                     bmode_image.shape,
                     order=0,
                     anti_aliasing=False)
prostate_mask = resize(prostate_mask,
                       bmode_image.shape,
                       order=0,
                       anti_aliasing=False)

plt.imshow(needle_mask, cmap='gray')
plt.figure(); plt.imshow(prostate_mask, cmap='gray')

needle_mask.min(), needle_mask.max()

H_px, W_px = bmode_image.shape
H_mm, W_mm = 28, 46.06
WH_mm, WW_mm = 3, 3
SH_mm, SW_mm = 1, 1

WH_px = int(WH_mm / W_mm * W_px)
WW_px = int(WW_mm / W_mm * W_px)
SH_px = int(SH_mm / H_mm * H_px)
SW_px = int(SW_mm / W_mm * W_px)
print(WH_px, WW_px, SH_px, SW_px)
pv = PatchView.from_sliding_window(bmode_image, (WH_px, WW_px), (SH_px, SW_px),
                                   masks=[needle_mask, prostate_mask],
                                   thresholds=[0.7, 0.9])


pv.show()
plt.imshow(needle_mask, alpha=0.2)