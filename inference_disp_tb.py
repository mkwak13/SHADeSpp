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

    assert os.path.isdir(opt.load_weights_folder)

    print("-> Loading weights from", opt.load_weights_folder)

    filenames = readlines(os.path.join("splits", opt.eval_split, "test_files.txt"))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    # ===== model =====
    num_in = 2 if "shadespp" in opt.load_weights_folder.lower() else 1

    encoder = networks.ResnetEncoder(opt.num_layers, False, num_input_images=num_in)

    if num_in == 2:
        encoder.encoder.conv1 = torch.nn.Conv2d(7, 64, 7, 2, 3, bias=False)

    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda().eval()
    depth_decoder.cuda().eval()

    # ===== decompose =====
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

    writer = SummaryWriter(os.path.join(opt.load_weights_folder, "tb_logs_disp"))

    print("-> Running inference...")

    with torch.no_grad():
        step = 0

        for img_path in filenames:

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (encoder_dict['width'], encoder_dict['height']))
            img = img.astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

            if num_in == 2:
                feat = decompose_encoder(img_tensor)
                reflectance, light, mask_soft = decompose_decoder(feat)
                depth_input = torch.cat([img_tensor, reflectance, mask_soft], dim=1)
            else:
                depth_input = img_tensor

            output = depth_decoder(encoder(depth_input))
            disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

            disp_np = disp[0, 0].cpu().numpy()

            # normalize
            disp_norm = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)

            # colormap
            disp_color = (disp_norm * 255).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_color, cv2.COLORMAP_MAGMA)
            disp_color = cv2.cvtColor(disp_color, cv2.COLOR_BGR2RGB) / 255.0

            # ===== combine input + disp =====
            img_resized = cv2.resize(img, (disp_color.shape[1], disp_color.shape[0]))

            combined = np.concatenate([img_resized, disp_color], axis=1)

            writer.add_image(
                "input_disp",
                combined,
                global_step=step,
                dataformats='HWC'
            )

            step += 1

    writer.close()
    print("-> Done")


if __name__ == "__main__":
    options = MonodepthOptions()
    inference(options.parse())