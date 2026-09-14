"""Generate a synthetic COLMAP-format sequence for testing the EndoMapper
evaluation pipeline (datasets/colmap_utils.py, datasets/EndoMapper_dataset.py,
evaluate_endomapper.py, visualize_endomapper.py) without needing real
EndoMapper data.

Writes:
    <out_dir>/<seq_name>/frames/000000.png, 000001.png, ...
    <out_dir>/<seq_name>/meta-data/colmap/cameras.txt
    <out_dir>/<seq_name>/meta-data/colmap/images.txt
    <out_dir>/<seq_name>/meta-data/colmap/points3D.txt

The camera moves along +Z with identity rotation; 3D points are randomly
scattered in front of it, so reprojections are geometrically consistent
(this exercises the real projection math in colmap_utils, not just file
parsing).
"""
from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
from PIL import Image


def generate(out_dir, seq_name="Seq_TEST", n_frames=8, n_points=400,
             width=288, height=288, seed=0):
    rng = np.random.RandomState(seed)

    fx = fy = 300.0
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Points scattered in a box in front of the (identity-rotation) starting camera.
    points_world = np.stack([
        rng.uniform(-2.0, 2.0, n_points),
        rng.uniform(-2.0, 2.0, n_points),
        rng.uniform(3.0, 8.0, n_points),
    ], axis=1)
    point_colors = rng.randint(0, 255, size=(n_points, 3))

    seq_dir = os.path.join(out_dir, seq_name)
    frames_dir = os.path.join(seq_dir, "frames")
    colmap_dir = os.path.join(seq_dir, "meta-data", "colmap")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(colmap_dir, exist_ok=True)

    # Camera translates slowly along +Z (i.e. moves toward the points); identity rotation.
    translations = [np.array([0.0, 0.0, -0.05 * i]) for i in range(n_frames)]

    per_image_obs = []  # list of (image_id, [(u, v, point3D_id), ...])
    point_tracks = {pid: [] for pid in range(n_points)}  # pid -> [(image_id, point2D_idx), ...]

    for i, t in enumerate(translations):
        image_id = i + 1
        xyz_cam = points_world + t[None, :]  # R = I
        depth = xyz_cam[:, 2]
        in_front = depth > 0.1
        uvw = (K @ xyz_cam.T).T
        uv = uvw[:, :2] / uvw[:, 2:3]
        in_bounds = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        visible = in_front & in_bounds

        obs = []
        for pid in np.where(visible)[0]:
            point2D_idx = len(obs)
            obs.append((uv[pid, 0], uv[pid, 1], pid))
            point_tracks[pid].append((image_id, point2D_idx))
        per_image_obs.append(obs)

        # Fake RGB frame (content is irrelevant for the smoke test -- only
        # used to exercise real model inference on a real-sized image).
        img = rng.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(frames_dir, "{:06d}.png".format(i)))

    # --- cameras.txt ---
    with open(os.path.join(colmap_dir, "cameras.txt"), "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[fx,fy,cx,cy]\n")
        f.write("1 PINHOLE {} {} {} {} {} {}\n".format(width, height, fx, fy, cx, cy))

    # --- images.txt ---
    with open(os.path.join(colmap_dir, "images.txt"), "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, (t, obs) in enumerate(zip(translations, per_image_obs)):
            image_id = i + 1
            name = "frames/{:06d}.png".format(i)
            f.write("{} 1.0 0.0 0.0 0.0 {} {} {} 1 {}\n".format(
                image_id, t[0], t[1], t[2], name))
            f.write(" ".join("{} {} {}".format(u, v, pid) for u, v, pid in obs) + "\n")

    # --- points3D.txt ---
    with open(os.path.join(colmap_dir, "points3D.txt"), "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid in range(n_points):
            track = point_tracks[pid]
            if len(track) == 0:
                continue  # COLMAP never writes untracked points; keep the fixture realistic
            x, y, z = points_world[pid]
            r, g, b = point_colors[pid]
            track_str = " ".join("{} {}".format(img_id, idx) for img_id, idx in track)
            f.write("{} {} {} {} {} {} {} {} {}\n".format(pid, x, y, z, r, g, b, 0.5, track_str))

    return seq_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seq_name", type=str, default="Seq_TEST")
    parser.add_argument("--n_frames", type=int, default=8)
    parser.add_argument("--n_points", type=int, default=400)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()

    seq_dir = generate(args.out_dir, args.seq_name, args.n_frames, args.n_points,
                        args.width, args.height)
    print("Wrote fake sequence to", seq_dir)


if __name__ == "__main__":
    main()
