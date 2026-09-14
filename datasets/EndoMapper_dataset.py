"""EndoMapper + COLMAP evaluation dataset loader.

STATUS: stub. EndoMapper access is pending (Synapse approval), so the exact
on-disk layout below is a best guess based on how EndoMapper + COLMAP
reconstructions are typically distributed. Everything that depends on the
real layout is marked with a `TODO(endomapper-data)` comment -- once the
data is downloaded, `ls` a sequence directory and fix those spots.

Assumed layout (one directory per sequence):

    data_root/
        Seq_001/
            meta-data/
                colmap/
                    cameras.bin (or cameras.txt)
                    images.bin  (or images.txt)
                    points3D.bin (or points3D.txt)
            <frames live somewhere under here -- TODO(endomapper-data)>
        Seq_002/
            ...

This loader does *not* subclass `datasets.mono_dataset.MonoDataset`: that
class is built around the train-time triplet (frame-1, frame0, frame+1)
sampling used by `trainer.py`, whereas here we only need, per *registered*
COLMAP image: the frame itself, its pose/intrinsics, and its sparse 3D
points. Keeping this standalone avoids distorting MonoDataset with
eval-only concerns.
"""
from __future__ import absolute_import, division, print_function

import os
import glob

import numpy as np
import PIL.Image as pil

from . import colmap_utils


def pil_loader(path):
    with open(path, "rb") as f:
        with pil.open(f) as img:
            return img.convert("RGB")


def list_sequences(data_root, colmap_subdir="meta-data/colmap"):
    """Find sequence directories under `data_root` that contain a COLMAP model."""
    sequences = []
    if not os.path.isdir(data_root):
        return sequences
    for name in sorted(os.listdir(data_root)):
        seq_dir = os.path.join(data_root, name)
        colmap_dir = os.path.join(seq_dir, colmap_subdir)
        if os.path.isdir(colmap_dir):
            try:
                colmap_utils.detect_model_format(colmap_dir)
                sequences.append(name)
            except FileNotFoundError:
                continue
    return sequences


class EndoMapperSequence(object):
    """One EndoMapper sequence: COLMAP model + the frames it references.

    Only frames that COLMAP actually registered (i.e. appear in images.bin/txt
    with a solved pose) are usable for evaluation, since those are the only
    ones with sparse 3D points to compare depth against.
    """

    def __init__(self, data_root, seq_name, colmap_subdir="meta-data/colmap",
                 frames_subdir=None, img_ext=None, model_ext=None):
        self.data_root = data_root
        self.seq_name = seq_name
        self.seq_dir = os.path.join(data_root, seq_name)
        self.colmap_dir = os.path.join(self.seq_dir, colmap_subdir)

        # TODO(endomapper-data): confirm the frames subdirectory and
        # extension once real sequences are available. Common possibilities
        # for EndoMapper-style releases are "frames/", "images/", or frames
        # extracted directly next to the sequence root. `frames_subdir=None`
        # falls back to searching a few likely candidates in
        # `_resolve_frame_path`.
        self.frames_subdir = frames_subdir
        self.img_ext = img_ext  # e.g. ".png" / ".jpg"; None = infer from COLMAP image name

        if not os.path.isdir(self.colmap_dir):
            raise FileNotFoundError(
                "No COLMAP directory found at {}".format(self.colmap_dir))

        self.cameras, self.images, self.points3D = colmap_utils.read_model(
            self.colmap_dir, ext=model_ext)

        # Only keep frames we can actually find on disk.
        self._image_ids = []
        self._frame_paths = {}
        for image_id, image in self.images.items():
            frame_path = self._resolve_frame_path(image.name)
            if frame_path is not None:
                self._image_ids.append(image_id)
                self._frame_paths[image_id] = frame_path
        self._image_ids.sort(key=lambda i: self.images[i].name)

        if len(self._image_ids) == 0:
            raise FileNotFoundError(
                "COLMAP model at {} has {} registered images, but none of "
                "their frame files could be located under {}. Check "
                "`frames_subdir`/`img_ext`, or extend "
                "EndoMapperSequence._resolve_frame_path().".format(
                    self.colmap_dir, len(self.images), self.seq_dir))

    def _resolve_frame_path(self, colmap_image_name):
        """Map a COLMAP `images.txt` NAME field to an actual file on disk.

        TODO(endomapper-data): once the real directory layout is known,
        collapse this to the single correct join and drop the guesswork.
        """
        candidates = []
        name = colmap_image_name
        base = os.path.basename(name)

        search_dirs = []
        if self.frames_subdir is not None:
            search_dirs.append(os.path.join(self.seq_dir, self.frames_subdir))
        else:
            search_dirs.extend([
                self.seq_dir,
                os.path.join(self.seq_dir, "frames"),
                os.path.join(self.seq_dir, "images"),
                os.path.join(self.seq_dir, "rgb"),
            ])

        for d in search_dirs:
            candidates.append(os.path.join(d, name))
            candidates.append(os.path.join(d, base))
            if self.img_ext is not None:
                stem = os.path.splitext(base)[0]
                candidates.append(os.path.join(d, stem + self.img_ext))

        for c in candidates:
            if os.path.isfile(c):
                return c
        return None

    def __len__(self):
        return len(self._image_ids)

    @property
    def image_ids(self):
        return list(self._image_ids)

    def frame_path(self, image_id):
        return self._frame_paths[image_id]

    def get_camera(self, image_id):
        image = self.images[image_id]
        return self.cameras[image.camera_id]

    def load_image(self, image_id):
        return pil_loader(self._frame_paths[image_id])

    def get_sparse_depth(self, image_id, min_depth=1e-6):
        """Reprojected sparse COLMAP points visible in this frame.

        Returns:
            pixels: (N, 2) float array of (u, v) pixel coordinates
            depths: (N,) float array of camera-space depth
            point3D_ids: (N,) int array of COLMAP point3D ids
        """
        image = self.images[image_id]
        camera = self.cameras[image.camera_id]
        return colmap_utils.project_points_to_image(
            image, camera, self.points3D, min_depth=min_depth)

    def __iter__(self):
        for image_id in self._image_ids:
            yield image_id


class EndoMapperColmapDataset(object):
    """Convenience wrapper iterating over every sequence under `data_root`.

    Usage:
        ds = EndoMapperColmapDataset("/path/to/EndoMapper", sequences=["Seq_001"])
        for seq in ds:
            for image_id in seq:
                img = seq.load_image(image_id)
                pixels, depths, ids = seq.get_sparse_depth(image_id)
    """

    def __init__(self, data_root, sequences=None, **sequence_kwargs):
        self.data_root = data_root
        self.sequence_kwargs = sequence_kwargs
        self.sequence_names = sequences if sequences is not None else list_sequences(data_root)

    def __len__(self):
        return len(self.sequence_names)

    def __iter__(self):
        for seq_name in self.sequence_names:
            yield EndoMapperSequence(self.data_root, seq_name, **self.sequence_kwargs)

    def __getitem__(self, idx):
        return EndoMapperSequence(self.data_root, self.sequence_names[idx], **self.sequence_kwargs)
