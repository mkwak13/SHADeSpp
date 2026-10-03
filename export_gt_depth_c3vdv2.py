"""Same convention as export_gt_depth_c3vd.py (identical uint16 encoding, same
/1000 mm->m scale, same [MIN_DEPTH, MAX_DEPTH] filter -- verified by comparing
raw depth.tiff percentiles between the two datasets), adapted for C3VDv2's
different directory layout: rgb/0000.png -> depth/0000_depth.tiff, instead of
C3VD's 0000_color.png -> 0000_depth.tiff in the same folder.
"""
import os
import cv2
import numpy as np
from utils import readlines

SPLIT_FILE = "splits/c3vdv2/test_files.txt"
OUTPUT_PATH = "splits/c3vdv2/gt_depths.npz"

MIN_DEPTH = 0.01
MAX_DEPTH = 10.0

filenames = readlines(SPLIT_FILE)

gt_depths = []

for line in filenames:
    color_path = line.strip()

    seq_dir = os.path.dirname(os.path.dirname(color_path))  # .../<seq>/rgb -> .../<seq>
    stem = os.path.splitext(os.path.basename(color_path))[0]  # "0000"
    depth_path = os.path.join(seq_dir, "depth", "{}_depth.tiff".format(stem))

    if not os.path.exists(depth_path):
        print("Missing:", depth_path)
        continue

    depth = cv2.imread(depth_path, -1)

    if depth is None:
        print("Failed:", depth_path)
        continue

    depth = depth.astype(np.float32)

    # mm -> meter
    depth /= 1000.0

    depth[(depth < MIN_DEPTH) | (depth > MAX_DEPTH)] = 0

    gt_depths.append(depth)

gt_depths = np.array(gt_depths)

print("GT shape:", gt_depths.shape)

np.savez_compressed(OUTPUT_PATH, data=gt_depths)

print("Saved to:", OUTPUT_PATH)
