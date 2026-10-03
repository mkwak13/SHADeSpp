"""Pre-undistort EndoMapper frames for self-supervised training.

Self-supervised monodepth-style training needs no ground truth (depth or
otherwise) -- just temporally sequential frames from a moving camera, warped
via a *predicted* pose and depth and compared photometrically. But the
warping machinery (BackprojectDepth/Project3D) assumes a standard pinhole
camera, so feeding it raw fisheye-distorted frames would make the geometry
wrong near the edges. This undistorts ALL frames in each chosen sequence
(not just the COLMAP-registered ones -- training doesn't need poses or
sparse points, so there's no reason to throw away unregistered frames) using
that sequence's own OPENCV_FISHEYE camera params, and writes them under a
clean sequential numbering (00000.png, 00001.png, ...) so there are no gaps
and no filename ambiguity for MonoDataset's frame-index parsing.

Also prints a normalized mean intrinsics matrix (fx, fy, cx, cy as fractions
of width/height) across the chosen sequences' undistorted cameras, for use
as the fixed K in a new EndoMapperMonoDataset -- the same "single
mean-intrinsics K for the whole dataset" convention C3VD/HK already use,
since per-frame/per-sequence K isn't something MonoDataset's design supports.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from datasets import colmap_utils
from datasets.EndoMapper_dataset import EndoMapperSequence


def cache_sequence(data_root, seq_name, out_root, colmap_root="pseudoGT",
                    frames_root="colmap_benchmark_frames", colmap_model_subdir="0",
                    balance=0.0):
    seq = EndoMapperSequence(data_root, seq_name, colmap_root=colmap_root,
                              frames_root=frames_root, colmap_model_subdir=colmap_model_subdir)
    # Camera is per-image in principle but in practice one physical scope per
    # sequence -- grab it from any registered image.
    camera = seq.get_camera(seq.image_ids[0])
    map1, map2, new_K = colmap_utils.build_undistort_map(camera, balance=balance)

    patient_id, subseq_id = seq_name.split("/")
    out_dir = os.path.join(out_root, "{}_{}".format(patient_id, subseq_id))
    os.makedirs(out_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(data_root, frames_root, patient_id, subseq_id, "*.png")))
    for i, src_path in enumerate(frame_files):
        img_bgr = cv2.imread(src_path)
        if img_bgr is None:
            print("  skip (unreadable): {}".format(src_path))
            continue
        if map1 is not None:
            img_bgr = colmap_utils.undistort_image(img_bgr, map1, map2)
        out_path = os.path.join(out_dir, "{:06d}.png".format(i))
        cv2.imwrite(out_path, img_bgr)

    print("{}: {} frames -> {}  (new_K diag fx={:.1f} fy={:.1f} cx={:.1f} cy={:.1f}, size {}x{})".format(
        seq_name, len(frame_files), out_dir, new_K[0, 0], new_K[1, 1], new_K[0, 2], new_K[1, 2],
        camera.width, camera.height))

    return out_dir, len(frame_files), new_K, camera.width, camera.height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=str, default="datasets/EndoMapper")
    parser.add_argument("--out_root", type=str, default="datasets/EndoMapper_undistorted")
    parser.add_argument("--seq", type=str, nargs="+", required=True)
    parser.add_argument("--balance", type=float, default=0.0)
    args = parser.parse_args()

    results = []
    for seq_name in args.seq:
        results.append(cache_sequence(args.data_root, seq_name, args.out_root, balance=args.balance))

    Ks_norm = []
    for _, _, new_K, w, h in results:
        Ks_norm.append([new_K[0, 0] / w, new_K[1, 1] / h, new_K[0, 2] / w, new_K[1, 2] / h])
    mean_K = np.mean(Ks_norm, axis=0)
    print("\nMean normalized intrinsics across {} sequences: fx={:.6f} fy={:.6f} cx={:.6f} cy={:.6f}".format(
        len(results), *mean_K))


if __name__ == "__main__":
    main()
