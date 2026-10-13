"""Monocular camera-pose evaluation against EndoMapper's COLMAP pseudo-GT.

Complements evaluate_endomapper.py (depth-only) with a pose-accuracy check:
for each sequence, finds pairs of temporally CONSECUTIVE registered frames
(frame-number difference exactly 1, so no dropped/unregistered frame sits
between them), computes the GT relative camera pose from COLMAP's qvec/tvec
and the model's PREDICTED relative pose from its own trained pose_encoder +
pose decoder, and reports:

  - rotation error, in degrees (scale-invariant, no alignment needed)
  - translation error, after a single per-sequence scale alignment (monocular
    pose has no absolute scale, and neither does COLMAP's uncalibrated SfM
    reconstruction -- so only relative/scaled translation accuracy is
    meaningful, same reasoning evaluate_endomapper.py already applies to
    depth via per-frame median scaling)

Undistorting the input frames before feeding the pose network does not
invalidate COLMAP's qvec/tvec as ground truth: undistortion only reprojects
pixel coordinates (intrinsics), it does not move the camera (extrinsics) --
exactly the same reasoning evaluate_endomapper.py already relies on for
sparse depth.

Usage:
    python evaluate_endomapper_pose.py --config configs/endomapper_example.json \\
        --seq 00033/13 00033/14 --output_dir outputs/endomapper_pose_eval
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import re

import numpy as np
import torch
from PIL import Image as pil
from torchvision import transforms

import networks
from layers import transformation_from_parameters
from datasets import colmap_utils
from datasets.EndoMapper_dataset import EndoMapperSequence, list_sequences
from evaluate_endomapper import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seq", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--frame_gap", type=int, default=1,
                         help="Evaluate pairs of frames this many frame-numbers apart "
                              "instead of strictly-consecutive (gap=1) frames. A larger "
                              "gap gives non-trivial ground-truth motion to estimate, "
                              "avoiding the near-zero-rotation/unstable-translation-direction "
                              "regime that consecutive EndoMapper frames sit in.")
    return parser.parse_args()


def load_pose_model(load_weights_folder, num_layers=18, device=None):
    """Load a checkpoint's own trained pose_encoder.pth/pose.pth -- same
    architecture trainer.py constructs (ResnetEncoder with num_input_images=2,
    PoseDecoder with num_frames_to_predict_for=2, only index 0 used -- the
    second slot is unused boilerplate inherited from upstream monodepth2's
    generic multi-frame interface)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_weights_folder = os.path.expanduser(load_weights_folder)

    encoder_path = os.path.join(load_weights_folder, "pose_encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "pose.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict.get("height", 288)
    feed_width = encoder_dict.get("width", 288)

    pose_encoder = networks.ResnetEncoder(num_layers, False, num_input_images=2)
    model_dict = pose_encoder.state_dict()
    pose_encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    pose_decoder = networks.PoseDecoder(
        pose_encoder.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
    pose_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    pose_encoder.to(device).eval()
    pose_decoder.to(device).eval()
    return pose_encoder, pose_decoder, feed_height, feed_width, device


def preprocess(img, feed_height, feed_width):
    resized = img.resize((feed_width, feed_height), pil.LANCZOS)
    return transforms.ToTensor()(resized).unsqueeze(0)


@torch.no_grad()
def predict_relative_pose(pose_encoder, pose_decoder, img0_t, img1_t, device):
    """Predicted transform taking a point in cam0 coords to cam1 coords --
    matches trainer.py's cam_T_cam for a positive frame_id (source=frame 0,
    target=frame f_i>0, invert=False)."""
    inp = torch.cat([img0_t, img1_t], dim=1).to(device)
    features = [pose_encoder(inp)]
    axisangle, translation = pose_decoder(features)
    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=False)
    return T[0].cpu().numpy()


def gt_relative_pose(qvec0, tvec0, qvec1, tvec1):
    """GT transform taking a point in cam0 coords to cam1 coords, from
    COLMAP's world-to-camera poses (X_cam = R @ X_world + t)."""
    R0, t0 = colmap_utils.qvec2rotmat(qvec0), np.asarray(tvec0).reshape(3, 1)
    R1, t1 = colmap_utils.qvec2rotmat(qvec1), np.asarray(tvec1).reshape(3, 1)
    T0 = np.eye(4); T0[:3, :3] = R0; T0[:3, 3:4] = t0
    T1 = np.eye(4); T1[:3, :3] = R1; T1[:3, 3:4] = t1
    # T0: world->cam0, T1: world->cam1. inv(T0): cam0->world. T1 @ inv(T0): cam0->cam1.
    return T1 @ np.linalg.inv(T0)


def rotation_error_deg(R_pred, R_gt):
    R_diff = R_pred.T @ R_gt
    cos_theta = np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def translation_angle_error_deg(t_pred, t_gt):
    """Angle between predicted and GT translation direction vectors --
    scale-free (no per-sequence scale fit needed, since normalizing each
    vector to unit length cancels any scalar scale), so it isolates whether
    the model points the camera motion the right way, independent of the
    magnitude issues trans_error_rel is prone to (near-zero GT translation
    blows up a relative-magnitude error; the angle stays well-defined as
    long as both vectors are non-degenerate)."""
    norm_pred = np.linalg.norm(t_pred)
    norm_gt = np.linalg.norm(t_gt)
    if norm_pred < 1e-9 or norm_gt < 1e-9:
        return float("nan")
    cos_theta = np.clip(np.dot(t_pred, t_gt) / (norm_pred * norm_gt), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def frame_number(name):
    m = re.search(r"(\d+)\.\w+$", name)
    return int(m.group(1)) if m else None


def frame_gap_pairs(seq, gap=1):
    """(image_id_a, image_id_b) for every pair of registered frames whose
    frame numbers differ by exactly `gap` -- genuinely gap-separated video
    frames, not just adjacent in the (possibly gappy) registered-frame list.
    gap=1 reproduces the original consecutive-frame pairing."""
    entries = []
    for image_id in seq.image_ids:
        n = frame_number(seq.images[image_id].name)
        if n is not None:
            entries.append((n, image_id))
    entries.sort(key=lambda e: e[0])

    by_number = {n: image_id for n, image_id in entries}
    pairs = []
    for n, image_id in entries:
        if (n + gap) in by_number:
            pairs.append((image_id, by_number[n + gap]))
    return pairs


def consecutive_pairs(seq):
    return frame_gap_pairs(seq, gap=1)


def evaluate_sequence(model_name, pose_encoder, pose_decoder, feed_height, feed_width, device,
                       seq, undistort, undistort_balance, frame_gap=1):
    pairs = frame_gap_pairs(seq, gap=frame_gap)
    if len(pairs) == 0:
        return None

    pred_translations = []
    gt_translations = []
    rows = []  # (image_id0, image_id1, R_pred, t_pred, R_gt, t_gt)

    for image_id0, image_id1 in pairs:
        if undistort:
            img0, _ = seq.load_image_undistorted(image_id0, balance=undistort_balance)
            img1, _ = seq.load_image_undistorted(image_id1, balance=undistort_balance)
        else:
            img0 = seq.load_image(image_id0)
            img1 = seq.load_image(image_id1)

        img0_t = preprocess(img0, feed_height, feed_width)
        img1_t = preprocess(img1, feed_height, feed_width)
        T_pred = predict_relative_pose(pose_encoder, pose_decoder, img0_t, img1_t, device)

        image0 = seq.images[image_id0]
        image1 = seq.images[image_id1]
        T_gt = gt_relative_pose(image0.qvec, image0.tvec, image1.qvec, image1.tvec)

        R_pred, t_pred = T_pred[:3, :3], T_pred[:3, 3]
        R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

        pred_translations.append(t_pred)
        gt_translations.append(t_gt)
        rows.append((image_id0, image_id1, R_pred, t_pred, R_gt, t_gt))

    pred_translations = np.stack(pred_translations)
    gt_translations = np.stack(gt_translations)

    # single per-sequence scale (monocular pose + uncalibrated COLMAP SfM are
    # both scale-ambiguous): least-squares scalar fit minimizing
    # sum ||gt - scale*pred||^2, i.e. scale = (gt . pred) / (pred . pred)
    denom = float(np.sum(pred_translations * pred_translations))
    scale = float(np.sum(gt_translations * pred_translations) / denom) if denom > 1e-12 else 0.0

    rot_errors = []
    trans_errors_rel = []
    trans_angle_errors = []
    for image_id0, image_id1, R_pred, t_pred, R_gt, t_gt in rows:
        rot_errors.append(rotation_error_deg(R_pred, R_gt))
        gt_norm = np.linalg.norm(t_gt)
        if gt_norm > 1e-9:
            trans_errors_rel.append(np.linalg.norm(t_gt - scale * t_pred) / gt_norm)
        angle_err = translation_angle_error_deg(t_pred, t_gt)
        if not np.isnan(angle_err):
            trans_angle_errors.append(angle_err)

    return {
        "n_pairs": len(pairs),
        "scale": scale,
        "rot_error_deg_mean": float(np.mean(rot_errors)),
        "rot_error_deg_median": float(np.median(rot_errors)),
        "trans_error_rel_mean": float(np.mean(trans_errors_rel)) if trans_errors_rel else float("nan"),
        "trans_error_rel_median": float(np.median(trans_errors_rel)) if trans_errors_rel else float("nan"),
        "trans_angle_deg_mean": float(np.mean(trans_angle_errors)) if trans_angle_errors else float("nan"),
        "trans_angle_deg_median": float(np.median(trans_angle_errors)) if trans_angle_errors else float("nan"),
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = args.output_dir or os.path.join(cfg["output_dir"], "pose_eval")
    os.makedirs(output_dir, exist_ok=True)

    data_root = cfg["data_root"]
    seq_names = args.seq or cfg["sequences"] or list_sequences(
        data_root, cfg["colmap_root"], cfg["frames_root"], cfg["colmap_model_subdir"])
    print("-> Found {} sequence(s)".format(len(seq_names)))

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu")

    loaded_models = {}
    for m in cfg["models"]:
        print("-> loading pose network for '{}' from {}".format(m["name"], m["load_weights_folder"]))
        loaded_models[m["name"]] = load_pose_model(
            m["load_weights_folder"], num_layers=m.get("num_layers", args.num_layers), device=device)

    summary_rows = []
    for seq_name in seq_names:
        print("\n===== Sequence: {} =====".format(seq_name))
        try:
            seq = EndoMapperSequence(
                data_root, seq_name, colmap_root=cfg["colmap_root"],
                frames_root=cfg["frames_root"], colmap_model_subdir=cfg["colmap_model_subdir"])
        except FileNotFoundError as e:
            print("  SKIPPING sequence: {}".format(e))
            continue

        for model_name, (pose_encoder, pose_decoder, feed_height, feed_width, dev) in loaded_models.items():
            result = evaluate_sequence(
                model_name, pose_encoder, pose_decoder, feed_height, feed_width, dev,
                seq, cfg["undistort"], cfg["undistort_balance"], frame_gap=args.frame_gap)
            if result is None:
                print("  [{}] no frame-gap-{} registered pairs, skipping".format(model_name, args.frame_gap))
                continue
            print("  [{}] n_pairs={}  rot_err_deg(mean/median)={:.3f}/{:.3f}  "
                  "trans_err_rel(mean/median)={:.4f}/{:.4f}  "
                  "trans_angle_deg(mean/median)={:.3f}/{:.3f}  scale={:.3f}".format(
                      model_name, result["n_pairs"],
                      result["rot_error_deg_mean"], result["rot_error_deg_median"],
                      result["trans_error_rel_mean"], result["trans_error_rel_median"],
                      result["trans_angle_deg_mean"], result["trans_angle_deg_median"],
                      result["scale"]))
            summary_rows.append({"sequence": seq_name, "model": model_name, **result})

    import csv
    import datetime
    date_str = datetime.date.today().isoformat()
    summary_path = os.path.join(output_dir, "endomapper_pose_summary_{}.csv".format(date_str))
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["sequence", "model", "n_pairs", "scale",
                      "rot_error_deg_mean", "rot_error_deg_median",
                      "trans_error_rel_mean", "trans_error_rel_median",
                      "trans_angle_deg_mean", "trans_angle_deg_median"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print("\n-> Wrote per-sequence summary to {}".format(summary_path))

    # frame-pair-weighted aggregate per model (weight by n_pairs, same spirit
    # as evaluate_endomapper.py's frame-weighted depth aggregation)
    print("\n===== Overall (pair-weighted) =====")
    for model_name in loaded_models:
        rows = [r for r in summary_rows if r["model"] == model_name]
        if not rows:
            continue
        total_pairs = sum(r["n_pairs"] for r in rows)
        rot_mean = sum(r["rot_error_deg_mean"] * r["n_pairs"] for r in rows) / total_pairs
        trans_mean = sum(r["trans_error_rel_mean"] * r["n_pairs"] for r in rows
                          if not np.isnan(r["trans_error_rel_mean"])) / total_pairs
        trans_angle_mean = sum(r["trans_angle_deg_mean"] * r["n_pairs"] for r in rows
                                if not np.isnan(r["trans_angle_deg_mean"])) / total_pairs
        print("  {}: n_sequences={} n_pairs={}  rot_err_deg={:.3f}  trans_err_rel={:.4f}  "
              "trans_angle_deg={:.3f}".format(
                  model_name, len(rows), total_pairs, rot_mean, trans_mean, trans_angle_mean))


if __name__ == "__main__":
    main()
