from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
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

    dataset = datasets.C3VDDataset(
        opt.data_path,
        filenames,
        encoder_dict['height'],
        encoder_dict['width'],
        [0], 4,
        is_train=False,
        img_ext=".png"
    )

    dataloader = DataLoader(
        dataset, 1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

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

    encoder.load_state_dict(torch.load(encoder_path))
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
        for idx, data in enumerate(dataloader):
            input_color = data[("color", 0, 0)].cuda()

            if num_in == 2:
                feat = decompose_encoder(input_color)
                reflectance, light, mask_soft = decompose_decoder(feat)
                depth_input = torch.cat([input_color, reflectance, mask_soft], dim=1)
            else:
                depth_input = input_color

            output = depth_decoder(encoder(depth_input))
            disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)

            disp_np = disp[0, 0].cpu().numpy()

            # normalize
            disp_norm = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)

            # ===== TensorBoard logging =====
            writer.add_image(
                "disp",
                disp_norm,
                global_step=idx,
                dataformats='HW'
            )

    writer.close()
    print("-> Done (TensorBoard)")


if __name__ == "__main__":
    options = MonodepthOptions()
    inference(options.parse())