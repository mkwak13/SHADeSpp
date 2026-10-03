"""Reproduces the exact original BBPS-2-3 (Hyper-Kvasir) preprocessing
pipeline documented in prepare_data/undistort.ipynb, combined with
video2frames.py + resizeFrames288.py + cap_frame_926.sh into one script
operating directly on the downloaded .avi files (skips the .jpg intermediate
stage and writes straight to the final .png output HKDataset expects).

Per video:
  1. Extract every frame (video2frames.py's logic).
  2. Center-crop to square + resize to 288x288 (resizeFrames288.py's logic).
  3. Cap to the first 926 frames (cap_frame_926.sh's logic).
  4. Undistort with the exact K_old/K_new/distortion coefficients from
     undistort.ipynb's "Finally we used C3VD intrinsics and distortions"
     cell -- the same calibration already hardcoded as HKInitDataset's
     distorted=False normalized intrinsics, confirmed by cross-checking the
     normalized values match exactly.

Output: datasets/BBPS-2-3Frames/Undistorted/Frames/<video-uuid>/00000.png, ...
-- the "BBPS-2-3Frames" substring in the path is load-bearing: both
HKDataset.get_image_path and trainer.py's process_batch hardcode a check
for that exact string to pick the BBPS frame-naming/inpaint-dir convention.
"""
import argparse
import glob
import os

import cv2
import numpy as np

OLD_INTRINSICS = [213.95173333333332, 213.83599999999998, 142.2096, 146.06213333333332]
NEW_INTRINSICS = [184.98585502685236, 184.35666485498837, 138.93697343164294, 152.59259458599243]
DISTORTIONS = [-0.42234, 0.10654, 0, 0]

K_OLD = np.array([[OLD_INTRINSICS[0], 0, OLD_INTRINSICS[2]],
                   [0, OLD_INTRINSICS[1], OLD_INTRINSICS[3]],
                   [0, 0, 1]])
K_NEW = np.array([[NEW_INTRINSICS[0], 0, NEW_INTRINSICS[2]],
                   [0, NEW_INTRINSICS[1], NEW_INTRINSICS[3]],
                   [0, 0, 1]])
DIST_COEFFS = np.array([DISTORTIONS])


def center_crop_resize(img, size=288):
    height, width = img.shape[:2]
    side = min(height, width)
    hrem, wrem = height - side, width - side
    cropped = img[hrem // 2: height - hrem // 2, wrem // 2: width - wrem // 2]
    return cv2.resize(cropped, (size, size))


def process_video(video_path, out_root, max_frames=926):
    uuid = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(out_root, uuid)
    os.makedirs(out_dir, exist_ok=True)

    cam = cv2.VideoCapture(video_path)
    i = 0
    n_written = 0
    while i < max_frames:
        ret, frame = cam.read()
        if not ret:
            break
        resized = center_crop_resize(frame, 288)
        undistorted = cv2.undistort(resized, K_OLD, DIST_COEFFS, None, K_NEW)
        out_path = os.path.join(out_dir, "{:05d}.png".format(i))
        cv2.imwrite(out_path, undistorted)
        i += 1
        n_written += 1
    cam.release()
    return uuid, n_written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos_dir", type=str, default="datasets/BBPS-2-3/raw_videos")
    parser.add_argument("--out_root", type=str, default="datasets/BBPS-2-3Frames/Undistorted/Frames")
    parser.add_argument("--max_frames", type=int, default=926)
    args = parser.parse_args()

    videos = sorted(glob.glob(os.path.join(args.videos_dir, "*.avi")))
    print("Found {} videos".format(len(videos)))
    total = 0
    for v in videos:
        uuid, n = process_video(v, args.out_root, args.max_frames)
        total += n
        print("{}: {} frames".format(uuid, n))
    print("TOTAL_FRAMES: {}".format(total))


if __name__ == "__main__":
    main()
