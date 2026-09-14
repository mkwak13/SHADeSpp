"""Evaluate SHADeS / SHADeS++ depth predictions against sparse COLMAP
reconstructions from EndoMapper sequences.

For every frame COLMAP registered in a sequence:
  1. run monocular depth inference (shades_inference.run_inference)
  2. reproject that frame's sparse COLMAP points to get (pixel, depth) pairs
     (datasets/EndoMapper_dataset.py + datasets/colmap_utils.py)
  3. sample the predicted depth map at those sparse pixel locations
  4. per-frame median-ratio scale alignment, exactly as `evaluate_depth.py`
     does for C3VD/C3VDv2 (ratio = median(gt) / median(pred) over the valid
     sparse points in that frame)
  5. compute AbsRel/RMSE (and the other `compute_errors` metrics) at the
     sparse point locations only

Usage:
    python evaluate_endomapper.py --config configs/endomapper_example.json
    python evaluate_endomapper.py --config configs/endomapper_example.json --seq Seq_001 Seq_002

NOTE: COLMAP reconstructions are only defined up to an arbitrary global
scale (and, for EndoMapper, that scale is whatever the SfM solve happened to
converge to -- it is *not* metric unless the reconstruction was
georeferenced/scaled externally). Comparing AbsRel is scale-invariant and
meaningful; RMSE is only meaningful in the units of that COLMAP
reconstruction's scale, not millimetres, unless you know it's been scaled.
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import datetime
import json
import os

import numpy as np
import torch

from datasets.EndoMapper_dataset import EndoMapperSequence, list_sequences
from evaluate_depth import compute_errors
from shades_inference import load_shades_model, run_inference

METRIC_COLS = ["abs_diff", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True,
                         help="path to an endomapper eval config json (see configs/endomapper_example.json)")
    parser.add_argument("--seq", type=str, nargs="+", default=None,
                         help="restrict evaluation to these sequence names (default: config's `sequences`, "
                              "or auto-discover every sequence under data_root)")
    parser.add_argument("--output_dir", type=str, default=None,
                         help="override the config's output_dir")
    parser.add_argument("--min_points_per_frame", type=int, default=None,
                         help="override the config's min_points_per_frame (frames with fewer valid "
                              "sparse points are skipped and reported separately)")
    parser.add_argument("--no_cuda", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, "r") as f:
        cfg = json.load(f)
    cfg.setdefault("colmap_subdir", "meta-data/colmap")
    cfg.setdefault("frames_subdir", None)
    cfg.setdefault("img_ext", None)
    cfg.setdefault("min_depth", 0.1)
    cfg.setdefault("max_depth", 150.0)
    cfg.setdefault("num_layers", 18)
    cfg.setdefault("min_points_per_frame", 20)
    cfg.setdefault("output_dir", "outputs/endomapper_eval")
    cfg.setdefault("sequences", None)
    return cfg


def sample_pred_at_points(pred_depth, pixels):
    """Nearest-pixel sample of a dense predicted depth map at sparse (u, v) points.

    Returns (sampled_depths, in_bounds_mask).
    """
    h, w = pred_depth.shape[:2]
    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)
    in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sampled = np.zeros((pixels.shape[0],), dtype=np.float32)
    sampled[in_bounds] = pred_depth[v[in_bounds], u[in_bounds]]
    return sampled, in_bounds


def evaluate_frame(model, seq, image_id, min_depth, max_depth, min_points):
    """Run inference + sparse-depth comparison for a single frame.

    Returns a dict of results, or None if the frame doesn't have enough
    valid sparse points to evaluate.
    """
    frame_path = seq.frame_path(image_id)
    result = run_inference(model, frame_path, min_depth=min_depth, max_depth=max_depth)
    pred_depth = result["pred_depth"]

    pixels, gt_depths, _ = seq.get_sparse_depth(image_id)
    if pixels.shape[0] == 0:
        return None

    pred_at_pts, in_bounds = sample_pred_at_points(pred_depth, pixels)
    valid = in_bounds & (gt_depths > 0) & np.isfinite(pred_at_pts) & (pred_at_pts > 0)

    n_valid = int(valid.sum())
    if n_valid < min_points:
        return {"skipped": True, "n_points": n_valid, "frame_path": frame_path}

    gt_valid = gt_depths[valid]
    pred_valid = pred_at_pts[valid].astype(np.float64)

    ratio = float(np.median(gt_valid) / np.median(pred_valid))
    pred_scaled = pred_valid * ratio

    errors = compute_errors(gt_valid, pred_scaled)

    return {
        "skipped": False,
        "n_points": n_valid,
        "frame_path": frame_path,
        "ratio": ratio,
        "errors": errors,
        "pred_depth": pred_depth,
        "pixels": pixels,
        "gt_depths": gt_depths,
        "valid": valid,
        "spec_mask": result["spec_mask"],
    }


def evaluate_sequence(model_name, model, seq, min_depth, max_depth, min_points, rows):
    ratios = []
    per_model_errors = []
    n_skipped = 0

    for image_id in seq.image_ids:
        out = evaluate_frame(model, seq, image_id, min_depth, max_depth, min_points)
        if out is None:
            continue
        if out["skipped"]:
            n_skipped += 1
            rows.append([seq.seq_name, model_name, os.path.basename(out["frame_path"]),
                         out["n_points"], ""] + [""] * len(METRIC_COLS))
            continue

        ratios.append(out["ratio"])
        per_model_errors.append(out["errors"])
        rows.append([seq.seq_name, model_name, os.path.basename(out["frame_path"]),
                     out["n_points"], out["ratio"]] + list(out["errors"]))

    n_frames = len(seq)
    print("  [{}] {} frames total, {} evaluated, {} skipped (< {} sparse points)".format(
        model_name, n_frames, len(per_model_errors), n_skipped, min_points))

    if len(ratios) > 0:
        ratios_arr = np.array(ratios)
        med = np.median(ratios_arr)
        print("    scaling ratio | med: {:0.3f} | std: {:0.3f}".format(
            med, np.std(ratios_arr / med) if med != 0 else float("nan")))

    if len(per_model_errors) > 0:
        mean_errors = np.array(per_model_errors).mean(0)
        print("    " + ("&{: 8.3f}  " * len(METRIC_COLS)).format(*mean_errors.tolist()) + "\\\\")
        return mean_errors
    else:
        print("    No frames with enough valid sparse points -- reconstruction may be degenerate; "
              "see visualize_endomapper.py to sanity check it.")
        return None


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = args.output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")
    min_points = args.min_points_per_frame or cfg["min_points_per_frame"]

    data_root = cfg["data_root"]
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            "data_root '{}' does not exist -- this is expected until the EndoMapper download "
            "finishes; point `data_path`/config['data_root'] at the real directory once you have "
            "it.".format(data_root))

    seq_names = args.seq or cfg["sequences"] or list_sequences(data_root, cfg["colmap_subdir"])
    if len(seq_names) == 0:
        raise RuntimeError(
            "No sequences found under {} (looked for a '{}' subdirectory in each). Check "
            "data_root / colmap_subdir in the config.".format(data_root, cfg["colmap_subdir"]))
    print("-> Found {} sequence(s): {}".format(len(seq_names), seq_names))

    print("-> Loading {} model(s)".format(len(cfg["models"])))
    loaded_models = {}
    for m in cfg["models"]:
        print("  loading '{}' from {}".format(m["name"], m["load_weights_folder"]))
        loaded_models[m["name"]] = load_shades_model(
            m["load_weights_folder"],
            num_layers=m.get("num_layers", cfg["num_layers"]),
            device=device,
            is_shadespp=m.get("is_shadespp", None))

    rows = [["sequence", "model", "frame", "n_points", "scale_ratio"] + METRIC_COLS]
    summary_rows = [["sequence", "model", "n_frames_evaluated"] + METRIC_COLS]

    for seq_name in seq_names:
        print("\n===== Sequence: {} =====".format(seq_name))
        try:
            seq = EndoMapperSequence(
                data_root, seq_name,
                colmap_subdir=cfg["colmap_subdir"],
                frames_subdir=cfg["frames_subdir"],
                img_ext=cfg["img_ext"])
        except FileNotFoundError as e:
            print("  SKIPPING sequence: {}".format(e))
            continue

        for model_name, model in loaded_models.items():
            mean_errors = evaluate_sequence(
                model_name, model, seq, cfg["min_depth"], cfg["max_depth"], min_points, rows)
            if mean_errors is not None:
                n_eval = sum(1 for r in rows if r[0] == seq_name and r[1] == model_name and r[4] != "")
                summary_rows.append([seq_name, model_name, n_eval] + list(mean_errors))

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    per_frame_csv = os.path.join(output_dir, "endomapper_per_frame_results_{}.csv".format(date))
    summary_csv = os.path.join(output_dir, "endomapper_summary_results_{}.csv".format(date))

    with open(per_frame_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    with open(summary_csv, "w", newline="") as f:
        csv.writer(f).writerows(summary_rows)

    print("\n-> Wrote per-frame results to {}".format(per_frame_csv))
    print("-> Wrote summary results to {}".format(summary_csv))


if __name__ == "__main__":
    main()
