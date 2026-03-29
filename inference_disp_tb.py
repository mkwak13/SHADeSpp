from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import networks

cv2.setNumThreads(0)


def inference(opt):
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        f"Cannot find folder {opt.load_weights_folder}"

    print("-> Loading weights from", opt.load_weights_folder)

    filenames = readlines(os.path.join("splits", opt.eval_split, "test_files.txt"))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    # ===== model =====
    if "shadespp" in opt.load_weights_folder.lower():
        num_in = 2
    else:
        num_in = 1

    encoder = networks.ResnetEncoder(opt.num_layers, False, num_input_images=num_in)

    if num_in == 2:
        encoder.encoder.conv1 = torch.nn.Conv2d(
            7, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

    # ===== FIX (state_dict filter) =====
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda().eval()
    depth_decoder.cuda().eval()

    # ===== decompose (SHADeS++) =====
    if num_in == 2:
        decompose_encoder = networks.ResnetEncoder(opt.num_layers, False)
        decompose_decoder = networks.decompose_decoder(
            decompose_encoder.num_ch_enc, scales=range(4)
        )

        decompose_encoder.load_state_dict(
            torch.load(os.path.join(opt.load_weights_folder, "decompose_encoder.pth"))
        )
        decompose_decoder.load_state_dict(
            torch.load(os.path.join(opt.load_weights_folder, "decompose.pth"))
        )

        decompose_encoder.cuda().eval()
        decompose_decoder.cuda().eval()

    # ===== TensorBoard =====
    writer = SummaryWriter(os.path.join(opt.load_weights_folder, "tb_logs_disp"))

    print("-> Running inference...")

    with torch.no_grad():
        for idx, img_path in enumerate(filenames):

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (encoder_dict['width'], encoder_dict['height']))
            img = img.astype(np.float32) / 255.0

            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

            if num_in == 2:
                feat = decompose_encoder(img)
                reflectance, light, mask_soft = decompose_decoder(feat)
                depth_input = torch.cat([img, reflectance, mask_soft], dim=1)
            else:
                depth_input = img

            output = depth_decoder(encoder(depth_input))
            disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

            disp_np = disp[0, 0].cpu().numpy()

            # normalize for visualization
            disp_norm = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)

            disp_color = (disp_norm * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_color, cv2.COLORMAP_MAGMA)

            disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB)

            writer.add_image(
                "disp",
                disp_color,
                global_step=idx,
                dataformats='HWC'
            )

    writer.close()
    print("-> Done (TensorBoard)")


if __name__ == "__main__":
    options = MonodepthOptions()
    inference(options.parse())