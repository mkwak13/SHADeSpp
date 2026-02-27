from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    abs_diff=np.mean(np.abs(gt - pred))
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_diff,abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150
    print("DEBUG data_path:", opt.data_path)

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)
        if opt.eval_split=="endovis":
            dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                            encoder_dict['height'], encoder_dict['width'],
                                            [0], 4, is_train=False)
            dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)
        elif opt.eval_split=="c3vd":
            dataset = datasets.C3VDDataset(
                opt.data_path if isinstance(opt.data_path, str) else opt.data_path[0],
                filenames,
                encoder_dict['height'],
                encoder_dict['width'],
                [0], 4,
                is_train=False,
                img_ext=".png"
            )
            dataloader = DataLoader(
                dataset, 16,
                shuffle=False,
                num_workers=opt.num_workers,
                pin_memory=True,
                drop_last=False
            )


        # baseline?? 1, SHADeS++?? 2
        if "shadespp" in opt.load_weights_folder.lower():
            num_in = 2
        else:
            num_in = 1

        encoder = networks.ResnetEncoder(
            opt.num_layers,
            False,
            num_input_images=num_in
        )
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        if num_in == 2:
            decompose_encoder = networks.ResnetEncoder(
                opt.num_layers,
                False
            )
            decompose_decoder = networks.decompose_decoder(
                decompose_encoder.num_ch_enc,
                scales=range(4)
            )

            decompose_encoder.load_state_dict(
                torch.load(os.path.join(opt.load_weights_folder, "decompose_encoder.pth"))
            )
            decompose_decoder.load_state_dict(
                torch.load(os.path.join(opt.load_weights_folder, "decompose.pth"))
            )

            decompose_encoder.cuda()
            decompose_encoder.eval()
            decompose_decoder.cuda()
            decompose_decoder.eval()

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        pred_disps = []
        pred_masks = []

        print("-> Computing predictions with size {}x{}".format(
            encoder_dict['width'], encoder_dict['height']))

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                # ----- decompose forward -----
                if num_in == 2:
                    decompose_feat = decompose_encoder(input_color)
                    reflectance, light, mask_soft = decompose_decoder(decompose_feat)
                    depth_input = torch.cat([input_color, reflectance], dim=1)
                else:
                    mask_soft = torch.zeros_like(input_color[:, :1])
                    depth_input = input_color

                output = depth_decoder(encoder(depth_input))
                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                mask_np = mask_soft.cpu()[:, 0].numpy()
                pred_masks.append(mask_np)

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
        pred_masks = np.concatenate(pred_masks)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Mono evaluation - using median scaling")

    errors_all = []
    errors_spec = []
    errors_nonspec = []
    ratios = []

    save_dir = os.path.join(opt.load_weights_folder, "depth_predictions")
    print("-> Saving out benchmark predictions to {}".format(save_dir))
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1/pred_disp

        valid_mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        spec_mask = pred_masks[i]
        spec_mask = cv2.resize(spec_mask, (gt_width, gt_height))
        spec_mask = (spec_mask > 0.5)

        spec_valid = np.logical_and(valid_mask, spec_mask)
        nonspec_valid = np.logical_and(valid_mask, np.logical_not(spec_mask))

        pred_depth_my = pred_depth.copy()

        pred_depth *= opt.pred_depth_scale_factor

        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth[valid_mask]) / np.median(pred_depth[valid_mask])
            ratios.append(ratio)
            pred_depth *= ratio
            pred_depth_my *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        pred_depth_my[pred_depth_my < MIN_DEPTH] = MIN_DEPTH
        pred_depth_my[pred_depth_my > MAX_DEPTH] = MAX_DEPTH

        # Overall
        errors_all.append(
            compute_errors(
                gt_depth[valid_mask],
                pred_depth_my[valid_mask]
            )
        )

        # Specular
        if spec_valid.sum() > 50:
            errors_spec.append(
                compute_errors(
                    gt_depth[spec_valid],
                    pred_depth_my[spec_valid]
                )
            )

        # Non-specular
        if nonspec_valid.sum() > 50:
            errors_nonspec.append(
                compute_errors(
                    gt_depth[nonspec_valid],
                    pred_depth_my[nonspec_valid]
                )
            )

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_all = np.array(errors_all).mean(0)
    mean_spec = np.array(errors_spec).mean(0)
    mean_nonspec = np.array(errors_nonspec).mean(0)

    print("\n===== Overall =====")
    print(("&{: 8.3f}  " * 8).format(*mean_all.tolist()) + "\\\\")

    print("\n===== Specular =====")
    print(("&{: 8.3f}  " * 8).format(*mean_spec.tolist()) + "\\\\")

    print("\n===== Non-Specular =====")
    print(("&{: 8.3f}  " * 8).format(*mean_nonspec.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
