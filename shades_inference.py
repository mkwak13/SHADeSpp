"""Shared SHADeS / SHADeS++ model loading + single-frame inference helpers.

This factors out the model-loading and forward-pass logic that
`evaluate_depth.py` uses for the C3VD/C3VDv2 benchmarks so it can be reused
by `evaluate_endomapper.py` and `visualize_endomapper.py` without duplicating
it. Behaviour is kept identical to `evaluate_depth.py`:

- A checkpoint is treated as SHADeS++ iff "shadespp" appears in its weights
  folder path (case-insensitive) -- same heuristic `evaluate_depth.py` uses
  -- unless `is_shadespp` is passed explicitly.
- SHADeS++ checkpoints feed [filtered, reflectance, specular-mask] (7
  channels) into the depth encoder; plain SHADeS feeds the 3-channel image
  only. Both use the same `networks.decompose_decoder` architecture to
  produce the reflectance/light/mask outputs (SHADeS just doesn't route them
  back into the depth encoder).
- `filtered` is the specular-suppressed image `compute_specular_filtered_image`
  builds below -- the raw image with masked (specular) pixels replaced by a
  neighborhood-averaged reflectance estimate. This matches exactly what
  `trainer.py.process_batch` feeds the depth encoder during training
  (trainer.py lines ~379-411); using the raw image instead, as earlier
  versions of this file and `evaluate_depth.py` did, is a train/inference
  mismatch -- the depth encoder was trained to expect specular suppression
  in its first 3 input channels and never learns to cope with their absence.
"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import PIL.Image as pil
import torch
import torch.nn.functional as F
from torchvision import transforms

import networks
from layers import disp_to_depth


class ShadesModel(object):
    """Loaded SHADeS or SHADeS++ depth model, ready for single-frame inference."""

    def __init__(self, encoder, depth_decoder, decompose_encoder, decompose_decoder,
                 feed_height, feed_width, is_shadespp, device):
        self.encoder = encoder
        self.depth_decoder = depth_decoder
        self.decompose_encoder = decompose_encoder
        self.decompose_decoder = decompose_decoder
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.is_shadespp = is_shadespp
        self.device = device


def load_shades_model(load_weights_folder, num_layers=18, device=None, is_shadespp=None):
    """Load a SHADeS/SHADeS++ checkpoint folder (containing encoder.pth, depth.pth,
    and -- for SHADeS++ -- decompose_encoder.pth/decompose.pth).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_weights_folder = os.path.expanduser(load_weights_folder)
    assert os.path.isdir(load_weights_folder), \
        "Cannot find a folder at {}".format(load_weights_folder)

    if is_shadespp is None:
        is_shadespp = "shadespp" in load_weights_folder.lower()

    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=device)
    feed_height = encoder_dict["height"]
    feed_width = encoder_dict["width"]

    num_in = 2 if is_shadespp else 1
    encoder = networks.ResnetEncoder(num_layers, False, num_input_images=num_in)
    if is_shadespp:
        # image(3) + reflectance(3) + specular-mask(1) = 7 input channels
        encoder.encoder.conv1 = torch.nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.to(device).eval()
    depth_decoder.to(device).eval()

    decompose_encoder = None
    decompose_decoder = None
    if is_shadespp:
        decompose_encoder = networks.ResnetEncoder(num_layers, False)
        decompose_decoder = networks.decompose_decoder(decompose_encoder.num_ch_enc, scales=range(4))
        decompose_encoder.load_state_dict(
            torch.load(os.path.join(load_weights_folder, "decompose_encoder.pth"), map_location=device))
        decompose_decoder.load_state_dict(
            torch.load(os.path.join(load_weights_folder, "decompose.pth"), map_location=device))
        decompose_encoder.to(device).eval()
        decompose_decoder.to(device).eval()

    return ShadesModel(
        encoder=encoder, depth_decoder=depth_decoder,
        decompose_encoder=decompose_encoder, decompose_decoder=decompose_decoder,
        feed_height=feed_height, feed_width=feed_width, is_shadespp=is_shadespp, device=device)


def diffusion_fill(source, known, iters, kernel=5):
    """Fill pixels where `known` is 0 by repeatedly averaging with already-
    known neighbors, letting newly-filled pixels count as known for the
    next pass. Ported 1:1 from `trainer.py` -- see its docstring. Needed
    because a specular region wider than kernel_filtered/kernel_reflectance
    leaves pixels with no valid unmasked neighbor at all, which the
    single-pass neighbor average below falls back to filling with
    near-black / a stale contaminated value.
    """
    pad = kernel // 2
    k2 = kernel ** 2
    value = source * known
    weight = known
    for _ in range(iters):
        sum_v = F.avg_pool2d(value, kernel, stride=1, padding=pad) * k2
        sum_w = F.avg_pool2d(weight, kernel, stride=1, padding=pad) * k2
        avg = sum_v / (sum_w + 1e-6)
        newly_known = (sum_w > 0.5).float() * (1 - weight)
        value = torch.where(weight > 0.5, value, torch.where(newly_known > 0.5, avg, value))
        weight = torch.clamp(weight + newly_known, max=1.0)
    return value


def compute_specular_filtered_image(input_color, reflectance, mask, kernel_filtered=7, kernel_reflectance=21,
                                      fill_iters=40, fov_black_threshold=0.03):
    """Replace masked (specular) pixels with a neighborhood-averaged
    reflectance estimate, color-matched to the surrounding non-specular
    luminance. Ported 1:1 from `trainer.py.process_batch`, which is what the
    SHADeS++ depth encoder actually saw as its first 6 input channels during
    training -- the depth encoder was never trained on specular-contaminated
    input, so inference must reproduce this exact transform rather than
    feeding the raw image.

    Returns (filtered, reflectance_filtered). `filtered` gets the
    input-luminance-matched replacement (`neigh_refl`), since it has to
    blend into a real photographed image -- its neighbor average uses the
    small `kernel_filtered` window to preserve real image resolution/texture
    outside the masked pixels. `reflectance_filtered` gets a separate,
    un-rescaled reflectance-space neighbor average (computed over the larger
    `kernel_reflectance` window) instead of reusing that same value --
    `reflectance` is a distinct albedo channel with its own brightness
    statistics, so rescaling its replacement to match `input_color`'s
    luminance just mismatches it against its own neighborhood (visible as
    dark dots at the replaced pixels rather than a smooth blend), and since
    only the representative local color matters there (not spatial
    sharpness), a wider window gives a smoother fill.
    """
    # Exclude pixels outside the endoscope's visible FOV (pure black
    # background/vignette corners) from counting as valid neighbors -- see
    # trainer.py's matching comment. Otherwise a masked pixel near the
    # frame's black border averages in mostly-black "neighbors" and its fill
    # collapses toward black instead of the true local tissue color.
    valid_fov_avg = (input_color.mean(1, keepdim=True) > fov_black_threshold).float()

    pad_filt = kernel_filtered // 2

    valid_filt_px = (1 - mask) * valid_fov_avg
    non_spec = input_color * valid_filt_px
    sum_non = F.avg_pool2d(non_spec, kernel_filtered, stride=1, padding=pad_filt) * (kernel_filtered ** 2)
    count_filt = F.avg_pool2d(valid_filt_px, kernel_filtered, stride=1, padding=pad_filt) * (kernel_filtered ** 2)

    refl_non_spec_filt = reflectance * valid_filt_px
    sum_refl_filt = F.avg_pool2d(refl_non_spec_filt, kernel_filtered, stride=1, padding=pad_filt) * (kernel_filtered ** 2)
    neigh_refl_raw_filt = sum_refl_filt / (count_filt + 1e-6)

    neigh_color = sum_non / (count_filt + 1e-6)
    luma_weights = torch.tensor([0.299, 0.587, 0.114], device=input_color.device).view(1, 3, 1, 1)
    luma_color = (neigh_color * luma_weights).sum(dim=1, keepdim=True)
    luma_refl = (neigh_refl_raw_filt * luma_weights).sum(dim=1, keepdim=True)
    ratio = luma_color / (luma_refl + 1e-6)
    ratio = torch.clamp(ratio, 0.0, 2.0)

    valid_filt = count_filt > 0.5
    neigh_refl = torch.where(valid_filt, neigh_refl_raw_filt * ratio, neigh_color)
    neigh_refl = torch.clamp(neigh_refl, 0.0, 1.0)

    pad_refl = kernel_reflectance // 2
    valid_refl_px = (1 - mask) * valid_fov_avg
    refl_non_spec_refl = reflectance * valid_refl_px
    sum_refl_refl = F.avg_pool2d(refl_non_spec_refl, kernel_reflectance, stride=1, padding=pad_refl) * (kernel_reflectance ** 2)
    count_refl = F.avg_pool2d(valid_refl_px, kernel_reflectance, stride=1, padding=pad_refl) * (kernel_reflectance ** 2)
    neigh_refl_raw_refl = sum_refl_refl / (count_refl + 1e-6)
    valid_refl = count_refl > 0.5
    neigh_refl_for_reflectance = torch.where(valid_refl, neigh_refl_raw_refl, reflectance)
    neigh_refl_for_reflectance = torch.clamp(neigh_refl_for_reflectance, 0.0, 1.0)

    # known/source set for diffusion excludes black background for the same
    # reason as the kernel average above; the final output still only takes
    # the diffused value where the pixel was actually masked, so black
    # background itself is never altered even though diffusion may compute
    # some (discarded) value for it internally.
    filtered_partial = input_color * (1 - mask) + neigh_refl * mask
    known_filt = torch.clamp(valid_filt_px + valid_filt.float() * mask, max=1.0)
    filtered_diffused = diffusion_fill(filtered_partial, known_filt, fill_iters)
    filtered = torch.where(mask > 0.5, filtered_diffused, filtered_partial)

    reflectance_filtered_partial = reflectance * (1 - mask) + neigh_refl_for_reflectance * mask
    known_refl = torch.clamp(valid_refl_px + valid_refl.float() * mask, max=1.0)
    reflectance_filtered_diffused = diffusion_fill(reflectance_filtered_partial, known_refl, fill_iters)
    reflectance_filtered = torch.where(mask > 0.5, reflectance_filtered_diffused, reflectance_filtered_partial)
    return filtered, reflectance_filtered


def preprocess_image(input_image, feed_height, feed_width):
    """Resize an already-loaded RGB PIL image to the model's input resolution.

    Returns:
        input_tensor: (1, 3, feed_height, feed_width) tensor in [0, 1]
        original_size: (width, height) of the source image, for resizing the
            prediction back to native resolution.
    """
    original_size = input_image.size  # (W, H)
    resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(resized).unsqueeze(0)
    return input_tensor, original_size


def load_input_image(image_path, feed_height, feed_width):
    """Load an image from disk and resize it to the model's input resolution.
    See `preprocess_image` for the return value -- use that directly if you
    already have a PIL image in memory (e.g. an undistorted frame).
    """
    input_image = pil.open(image_path).convert("RGB")
    return preprocess_image(input_image, feed_height, feed_width)


@torch.no_grad()
def infer_depth(model, input_tensor, min_depth=0.1, max_depth=150.0, original_size=None):
    """Run one forward pass. `input_tensor` is (1, 3, feed_H, feed_W) in [0, 1].

    Returns a dict with:
        pred_depth: (H, W) numpy array (H, W = original_size if given, else
            the model's feed resolution)
        spec_mask: (H, W) numpy array in [0, 1], the soft specular mask
            (all-zero for plain SHADeS, which has no decompose->depth path)
        reflectance / light: (H, W, 3) / (H, W) numpy arrays in [0, 1] if the
            model has a decompose branch, else None
    """
    input_tensor = input_tensor.to(model.device)

    reflectance = light = None
    if model.is_shadespp:
        decompose_feat = model.decompose_encoder(input_tensor)
        reflectance_t, light_t, mask_soft = model.decompose_decoder(decompose_feat)
        filtered, reflectance_filtered_t = compute_specular_filtered_image(input_tensor, reflectance_t, mask_soft)
        depth_input = torch.cat([filtered, reflectance_filtered_t, mask_soft], dim=1)
        # the filtered reflectance -- what the encoder actually sees -- not the raw
        # decoder output, so this matches `filtered` and `pred_depth` below.
        reflectance = reflectance_filtered_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        light = light_t.squeeze(0).squeeze(0).cpu().numpy()
    else:
        mask_soft = torch.zeros_like(input_tensor[:, :1])
        depth_input = input_tensor

    outputs = model.depth_decoder(model.encoder(depth_input))
    pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)

    if original_size is not None:
        out_w, out_h = original_size
        pred_disp = torch.nn.functional.interpolate(
            pred_disp, (out_h, out_w), mode="bilinear", align_corners=False)
        mask_soft = torch.nn.functional.interpolate(
            mask_soft, (out_h, out_w), mode="bilinear", align_corners=False)

    pred_disp_np = pred_disp.squeeze().cpu().numpy()
    pred_depth = 1.0 / pred_disp_np
    spec_mask = mask_soft.squeeze().cpu().numpy()

    return {
        "pred_depth": pred_depth,
        "spec_mask": spec_mask,
        "reflectance": reflectance,
        "light": light,
    }


def run_inference(model, image_path, min_depth=0.1, max_depth=150.0, native_resolution=True):
    """Convenience: load an image from disk and run `infer_depth` on it."""
    input_tensor, original_size = load_input_image(image_path, model.feed_height, model.feed_width)
    return infer_depth(
        model, input_tensor, min_depth=min_depth, max_depth=max_depth,
        original_size=original_size if native_resolution else None)
