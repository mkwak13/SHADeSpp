"""Per-frame visualizations for the EndoMapper/COLMAP evaluation pipeline,
and a quick sanity-check summary of a COLMAP reconstruction's quality.

For each (sampled) frame, saves one composite PNG per model with:
    [ input image | predicted depth (colormap) | predicted depth + reprojected COLMAP points overlaid ]
stacking one such row per model (e.g. SHADeS on top, SHADeS++ below) so the
two are easy to compare side by side.

Also prints a per-sequence COLMAP health summary (registered-frame coverage,
sparse point counts per frame, mean reprojection error) *before* running any
model, purely from the reconstruction itself -- use this to screen out
sequences that are too sparse/noisy to bother evaluating on.

Usage:
    # Just check reconstruction quality, no images written:
    python visualize_endomapper.py --config configs/endomapper_example.json --health_only

    # Full visualization for one sequence (patient 00033, sub-sequence 13), one frame in 10:
    python visualize_endomapper.py --config configs/endomapper_example.json \
        --seq 00033/13 --frame_stride 10 --max_frames 50
"""
from __future__ import absolute_import, division, print_function

import argparse
import os

import cv2
import numpy as np
import torch

from datasets.EndoMapper_dataset import EndoMapperSequence, list_sequences
from evaluate_endomapper import evaluate_frame, load_config
from shades_inference import load_shades_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seq", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--frame_stride", type=int, default=1,
                         help="only visualize every Nth registered frame")
    parser.add_argument("--max_frames", type=int, default=None,
                         help="cap the number of visualized frames per sequence")
    parser.add_argument("--min_points_per_frame", type=int, default=None)
    parser.add_argument("--health_only", action="store_true",
                         help="only print the COLMAP reconstruction health summary; skip inference/images")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def summarize_colmap_health(seq, min_points_per_frame):
    """Model-independent sanity check of a COLMAP reconstruction: is it dense
    enough / clean enough to bother evaluating depth against?
    """
    n_registered = len(seq.images)
    n_with_frame = len(seq)
    counts = []
    for image_id in seq.image_ids:
        pixels, depths, _ = seq.get_sparse_depth(image_id)
        counts.append(pixels.shape[0])
    counts = np.array(counts) if len(counts) > 0 else np.zeros((0,))

    errors = np.array([p.error for p in seq.points3D.values()]) if len(seq.points3D) > 0 else np.zeros((0,))

    print("  registered images in COLMAP model : {}".format(n_registered))
    print("  registered images found on disk   : {}".format(n_with_frame))
    print("  total 3D points in model          : {}".format(len(seq.points3D)))
    if counts.size > 0:
        print("  sparse points per frame           : min={:.0f} mean={:.1f} max={:.0f}".format(
            counts.min(), counts.mean(), counts.max()))
        n_bad = int((counts < min_points_per_frame).sum())
        print("  frames below min_points_per_frame ({}) : {} / {}".format(
            min_points_per_frame, n_bad, counts.size))
    else:
        print("  sparse points per frame           : N/A (no frames found on disk)")
    if errors.size > 0:
        print("  COLMAP mean reprojection error     : {:.3f} px".format(errors.mean()))

    degenerate = counts.size == 0 or np.median(counts) < min_points_per_frame
    if degenerate:
        print("  ** WARNING: this reconstruction looks sparse/degenerate -- inspect the "
              "overlay images before trusting metrics from it. **")
    return {
        "n_registered": n_registered,
        "n_with_frame": n_with_frame,
        "counts": counts,
        "errors": errors,
        "degenerate": degenerate,
    }


def colorize_depth(depth, valid_mask=None):
    d = depth.copy()
    if valid_mask is not None:
        finite = valid_mask & np.isfinite(d) & (d > 0)
    else:
        finite = np.isfinite(d) & (d > 0)
    if finite.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(d[finite], [1, 99])
    norm = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    norm[~finite] = 0
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    return colored


def draw_points_overlay(base_bgr, pixels, values, in_bounds, radius=3):
    img = base_bgr.copy()
    if pixels.shape[0] == 0:
        return img
    finite = np.isfinite(values) & (values > 0)
    mask = in_bounds & finite
    if mask.sum() == 0:
        return img
    lo, hi = np.percentile(values[mask], [1, 99])
    norm = np.clip((values[mask] - lo) / max(hi - lo, 1e-6), 0, 1)
    colors = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_MAGMA).reshape(-1, 3)
    pts = pixels[mask]
    for (u, v), color in zip(pts, colors):
        cv2.circle(img, (int(round(u)), int(round(v))), radius, tuple(int(c) for c in color), -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(img, (int(round(u)), int(round(v))), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return img


def make_frame_row(model_name, frame_path, out):
    """[input | depth colormap | depth colormap + COLMAP overlay] row for one model."""
    input_bgr = cv2.imread(frame_path)
    h, w = out["pred_depth"].shape[:2]
    if input_bgr.shape[:2] != (h, w):
        input_bgr = cv2.resize(input_bgr, (w, h))

    depth_vis = colorize_depth(out["pred_depth"])
    overlay_vis = draw_points_overlay(depth_vis, out["pixels"], out["gt_depths"], out["valid"])

    label_h = 24
    row = np.hstack([input_bgr, depth_vis, overlay_vis])
    labeled = np.zeros((label_h + row.shape[0], row.shape[1], 3), dtype=np.uint8)
    labeled[label_h:] = row
    cv2.putText(labeled, "{} | input | pred depth | pred depth + COLMAP pts (n={})".format(
        model_name, out["n_points"]), (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        cv2.LINE_AA)
    return labeled


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir or os.path.join(cfg["output_dir"], "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    min_points = args.min_points_per_frame or cfg["min_points_per_frame"]

    data_root = cfg["data_root"]
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            "data_root '{}' does not exist -- point it at the real EndoMapper directory once "
            "downloaded.".format(data_root))

    seq_names = args.seq or cfg["sequences"] or list_sequences(
        data_root, cfg["colmap_root"], cfg["frames_root"], cfg["colmap_model_subdir"])
    print("-> Found {} sequence(s): {}".format(len(seq_names), seq_names))

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    loaded_models = {}
    if not args.health_only:
        for m in cfg["models"]:
            print("-> loading '{}' from {}".format(m["name"], m["load_weights_folder"]))
            loaded_models[m["name"]] = load_shades_model(
                m["load_weights_folder"],
                num_layers=m.get("num_layers", cfg["num_layers"]),
                device=device,
                is_shadespp=m.get("is_shadespp", None))

    for seq_name in seq_names:
        print("\n===== Sequence: {} =====".format(seq_name))
        try:
            seq = EndoMapperSequence(
                data_root, seq_name,
                colmap_root=cfg["colmap_root"],
                frames_root=cfg["frames_root"],
                colmap_model_subdir=cfg["colmap_model_subdir"])
        except FileNotFoundError as e:
            print("  SKIPPING sequence: {}".format(e))
            continue

        summarize_colmap_health(seq, min_points)
        if args.health_only:
            continue

        seq_out_dir = os.path.join(output_dir, seq_name)
        os.makedirs(seq_out_dir, exist_ok=True)

        image_ids = seq.image_ids[::args.frame_stride]
        if args.max_frames is not None:
            image_ids = image_ids[:args.max_frames]

        for image_id in image_ids:
            rows = []
            for model_name, model in loaded_models.items():
                out = evaluate_frame(model, seq, image_id, cfg["min_depth"], cfg["max_depth"], min_points=0,
                                      min_gt_depth_ratio=cfg["min_gt_depth_ratio"])
                if out is None:
                    continue
                if out.get("skipped"):
                    # still visualize (n_points just low), pull a plain inference for the image
                    continue
                rows.append(make_frame_row(model_name, out["frame_path"], out))

            if len(rows) == 0:
                continue

            max_w = max(r.shape[1] for r in rows)
            rows = [cv2.copyMakeBorder(r, 0, 0, 0, max_w - r.shape[1], cv2.BORDER_CONSTANT, value=0)
                    for r in rows]
            composite = np.vstack(rows)

            frame_name = os.path.splitext(os.path.basename(seq.frame_path(image_id)))[0]
            out_path = os.path.join(seq_out_dir, "{}.png".format(frame_name))
            cv2.imwrite(out_path, composite)

        print("  wrote visualizations for {} frame(s) to {}".format(len(image_ids), seq_out_dir))


if __name__ == "__main__":
    main()
