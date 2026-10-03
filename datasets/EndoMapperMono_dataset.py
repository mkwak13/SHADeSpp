"""Self-supervised training dataset for EndoMapper -- NOT the same thing as
EndoMapper_dataset.py's EndoMapperSequence (that one is for evaluating
against COLMAP sparse points). This one feeds plain pre-undistorted video
frames into the standard monodepth2-style triplet training loop, the same
way HK_dataset.py/C3VD_dataset.py do -- self-supervised training needs no
ground truth (depth or otherwise), just sequential frames from a moving
camera, so it doesn't touch COLMAP at all.

Expects frames already undistorted and sequentially numbered by
prepare_data/cache_endomapper_undistorted.py:
    datasets/EndoMapper_undistorted/<patient>_<subseq>/000000.png, 000001.png, ...

Intrinsics are fixed per dataset instance (matching HK_dataset.py/
C3VD_dataset.py's convention), from the actual camera -- not a rough mean
guess here, since the cached sequences happen to share one real camera
calibration (see cache_endomapper_undistorted.py's printed mean_K).
"""
from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
from .mono_dataset import MonoDataset


class EndoMapperMonoInitDataset(MonoDataset):
    def __init__(self, *args, flipping=False, rotating=False, distorted=False,
                 inpaint_pseudo_gt_dir=None,
                 intrinsics=(0.364966, 0.488356, 0.630647, 0.499410), **kwargs):
        super(EndoMapperMonoInitDataset, self).__init__(*args, **kwargs)

        # flipping defaults to False (and shouldn't be turned on) -- the
        # principal point here is well off-center (cx=0.63) from the
        # balance=0.0 undistortion crop, so a horizontal flip would need a
        # correspondingly flipped cx that this fixed-K setup doesn't do.
        self.flipping = flipping
        self.rotating = rotating
        self.inpaint_pseudo_gt_dir = inpaint_pseudo_gt_dir

        self.K = np.array([[intrinsics[0], 0, intrinsics[2], 0],
                            [0, intrinsics[1], intrinsics[3], 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1440, 1080)

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip, do_rot):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip and self.flipping:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        if do_rot and self.rotating:
            angle = np.random.choice([pil.ROTATE_90, pil.ROTATE_180, pil.ROTATE_270])
            color = color.transpose(angle)
        return color


class EndoMapperMonoDataset(EndoMapperMonoInitDataset):
    def __init__(self, *args, **kwargs):
        super(EndoMapperMonoDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        return os.path.join(self.data_path, folder, f_str)
