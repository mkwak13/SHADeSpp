"""Shared SHADeS / SHADeS++ model loading + single-frame inference helpers.

This factors out the model-loading and forward-pass logic that
`evaluate_depth.py` uses for the C3VD/C3VDv2 benchmarks so it can be reused
by `evaluate_endomapper.py` and `visualize_endomapper.py` without duplicating
it. Behaviour is kept identical to `evaluate_depth.py`:

- A checkpoint is treated as SHADeS++ iff "shadespp" appears in its weights
  folder path (case-insensitive) -- same heuristic `evaluate_depth.py` uses
  -- unless `is_shadespp` is passed explicitly.
- SHADeS++ checkpoints feed [image, reflectance, specular-mask] (7 channels)
  into the depth encoder; plain SHADeS feeds the 3-channel image only. Both
  use the same `networks.decompose_decoder` architecture to produce the
  reflectance/light/mask outputs (SHADeS just doesn't route them back into
  the depth encoder).
"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
import PIL.Image as pil
import torch
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


def load_input_image(image_path, feed_height, feed_width):
    """Load an image and resize it to the model's input resolution.

    Returns:
        input_tensor: (1, 3, feed_height, feed_width) tensor in [0, 1]
        original_size: (width, height) of the source image, for resizing the
            prediction back to native resolution.
    """
    input_image = pil.open(image_path).convert("RGB")
    original_size = input_image.size  # (W, H)
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0)
    return input_tensor, original_size


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
        depth_input = torch.cat([input_tensor, reflectance_t, mask_soft], dim=1)
        reflectance = reflectance_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
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
