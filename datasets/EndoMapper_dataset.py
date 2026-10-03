"""EndoMapper + COLMAP evaluation dataset loader.

Matches the real downloaded layout (confirmed against
`datasets/EndoMapper/{pseudoGT,colmap_benchmark_frames}`, 8 patients / 161
reconstructed sub-sequences as of 2026-10-02):

    data_root/
        pseudoGT/
            <patient_id>/                e.g. "00033"
                <subseq_id>/              e.g. "13"
                    0/
                        cameras.bin
                        images.bin
                        points3D.bin
                    cameras.txt           (redundant text export of the same model)
                    images.txt
                    points3D.txt
                    database.db           (COLMAP feature DB, unused here)
                    RandT.npy / RandT.txt (per-frame world-to-cam 4x4s, in
                                            RandT.txt's filename order --
                                            redundant with images.bin's
                                            qvec/tvec, unused here)
        colmap_benchmark_frames/
            <patient_id>/
                <subseq_id>/
                    <image.name>.png      e.g. HCULB_00033_procedure_lossy_h264_001.png

A "sequence" here is one (patient_id, subseq_id) pair -- i.e. one COLMAP
reconstruction. Not every extracted frame under colmap_benchmark_frames/ is
necessarily registered in the model (COLMAP only localizes the frames it
could); `EndoMapperSequence` only exposes the registered ones, since those
are the only ones with sparse 3D points to evaluate against.

Cameras are OPENCV_FISHEYE in the data seen so far. `get_sparse_depth` uses
`colmap_utils.observed_points_with_depth`, which samples the *actual*
detected keypoint pixel coordinates rather than re-projecting through the
(distorted) camera model -- so this works correctly regardless of camera
model, without us having to implement fisheye distortion.
"""
from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np
import PIL.Image as pil

from . import colmap_utils


def pil_loader(path):
    with open(path, "rb") as f:
        with pil.open(f) as img:
            return img.convert("RGB")


def _has_colmap_model(colmap_dir, colmap_model_subdir):
    """True if a COLMAP model is reachable from `colmap_dir`, either directly
    (text export) or under `colmap_model_subdir` (the binary export, e.g. "0/").
    """
    for candidate in (os.path.join(colmap_dir, colmap_model_subdir), colmap_dir):
        try:
            colmap_utils.detect_model_format(candidate)
            return True
        except FileNotFoundError:
            continue
    return False


def list_sequences(data_root, colmap_root="pseudoGT", frames_root="colmap_benchmark_frames",
                    colmap_model_subdir="0"):
    """Find every (patient_id, subseq_id) pair that has both a COLMAP model
    (under colmap_root) and a frames directory (under frames_root).

    Returns a sorted list of "patient_id/subseq_id" strings.
    """
    sequences = []
    colmap_base = os.path.join(data_root, colmap_root)
    frames_base = os.path.join(data_root, frames_root)
    if not os.path.isdir(colmap_base) or not os.path.isdir(frames_base):
        return sequences

    for patient_id in sorted(os.listdir(colmap_base)):
        patient_colmap_dir = os.path.join(colmap_base, patient_id)
        if not os.path.isdir(patient_colmap_dir):
            continue
        for subseq_id in sorted(os.listdir(patient_colmap_dir)):
            colmap_dir = os.path.join(patient_colmap_dir, subseq_id)
            frames_dir = os.path.join(frames_base, patient_id, subseq_id)
            if os.path.isdir(frames_dir) and _has_colmap_model(colmap_dir, colmap_model_subdir):
                sequences.append("{}/{}".format(patient_id, subseq_id))
    return sequences


class EndoMapperSequence(object):
    """One EndoMapper (patient_id, subseq_id) COLMAP reconstruction + the
    registered frames it references.
    """

    def __init__(self, data_root, patient_id, subseq_id=None,
                 colmap_root="pseudoGT", frames_root="colmap_benchmark_frames",
                 colmap_model_subdir="0"):
        if subseq_id is None:
            # allow a single "patient_id/subseq_id" string, as returned by list_sequences()
            patient_id, subseq_id = patient_id.split("/", 1)

        self.data_root = data_root
        self.patient_id = patient_id
        self.subseq_id = subseq_id
        self.seq_name = "{}/{}".format(patient_id, subseq_id)

        colmap_dir = os.path.join(data_root, colmap_root, patient_id, subseq_id)
        self.frames_dir = os.path.join(data_root, frames_root, patient_id, subseq_id)

        model_dir = os.path.join(colmap_dir, colmap_model_subdir)
        if not os.path.isdir(model_dir) or not _has_colmap_model(model_dir, "."):
            model_dir = colmap_dir  # fall back to the sibling .txt export
        self.colmap_dir = model_dir

        if not os.path.isdir(self.frames_dir):
            raise FileNotFoundError("No frames directory at {}".format(self.frames_dir))

        self.cameras, self.images, self.points3D = colmap_utils.read_model(self.colmap_dir)

        self._image_ids = []
        self._frame_paths = {}
        for image_id, image in self.images.items():
            frame_path = os.path.join(self.frames_dir, image.name)
            if not os.path.isfile(frame_path):
                # fall back to matching by basename, in case `name` ever carries a subdir prefix
                frame_path = os.path.join(self.frames_dir, os.path.basename(image.name))
            if os.path.isfile(frame_path):
                self._image_ids.append(image_id)
                self._frame_paths[image_id] = frame_path
        self._image_ids.sort(key=lambda i: self.images[i].name)

        if len(self._image_ids) == 0:
            raise FileNotFoundError(
                "COLMAP model at {} has {} registered images, but none of their frame files "
                "could be found under {}.".format(self.colmap_dir, len(self.images), self.frames_dir))

        self._undistort_cache = {}  # camera_id -> (map1, map2, new_K)

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

    def _get_undistort_maps(self, camera, balance=0.0):
        key = (camera.id, balance)
        if key not in self._undistort_cache:
            self._undistort_cache[key] = colmap_utils.build_undistort_map(camera, balance=balance)
        return self._undistort_cache[key]

    def load_image_undistorted(self, image_id, balance=0.0):
        """Undistort this frame using its camera's own OPENCV_FISHEYE
        parameters (a no-op, returning the original image + K, for cameras
        that aren't fisheye-distorted). Use alongside `get_sparse_depth`'s
        `undistort=True` so predicted depth and GT points stay in the same
        pixel space.

        Returns (PIL.Image, new_K) -- new_K is the camera matrix of the
        undistorted image (3x3), needed to also map GT points into its space.
        """
        image = self.images[image_id]
        camera = self.cameras[image.camera_id]
        map1, map2, new_K = self._get_undistort_maps(camera, balance=balance)
        if map1 is None:
            return self.load_image(image_id), new_K

        img_bgr = cv2.imread(self._frame_paths[image_id])
        undist_bgr = colmap_utils.undistort_image(img_bgr, map1, map2)
        undist_rgb = cv2.cvtColor(undist_bgr, cv2.COLOR_BGR2RGB)
        return pil.fromarray(undist_rgb), new_K

    def get_sparse_depth(self, image_id, min_depth=1e-6, undistort=False, balance=0.0):
        """Sparse COLMAP points visible in this frame, at their actual
        detected pixel locations (correct even for the fisheye-distorted
        EndoMapper cameras -- see colmap_utils.observed_points_with_depth).

        Pass `undistort=True` (matching `load_image_undistorted`'s `balance`)
        to get pixel coordinates in the *undistorted* image's pixel space --
        required if the depth map you're comparing against was predicted on
        an undistorted frame.

        Returns:
            pixels: (N, 2) float array of (u, v) pixel coordinates
            depths: (N,) float array of camera-space depth
            point3D_ids: (N,) int array of COLMAP point3D ids
        """
        image = self.images[image_id]
        pixels, depths, point3D_ids = colmap_utils.observed_points_with_depth(
            image, self.points3D, min_depth=min_depth)
        if undistort and pixels.shape[0] > 0:
            camera = self.cameras[image.camera_id]
            pixels = colmap_utils.undistort_points(pixels, camera,
                                                     self._get_undistort_maps(camera, balance)[2])
        return pixels, depths, point3D_ids

    def __iter__(self):
        for image_id in self._image_ids:
            yield image_id


class EndoMapperColmapDataset(object):
    """Convenience wrapper iterating over every sequence under `data_root`.

    Usage:
        ds = EndoMapperColmapDataset("datasets/EndoMapper", sequences=["00033/13"])
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
