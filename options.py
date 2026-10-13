from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import json

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
     def __init__(self):
          self.parser = argparse.ArgumentParser(description="IID_SFM options")

          # PATHS
          self.parser.add_argument("--data_path",
                                   type=str,
                                   help="path to the training data")
          self.parser.add_argument("--log_dir",
                                   type=str,
                                   help="log directory")

          # TRAINING options
          self.parser.add_argument("--model_name",
                                   type=str,
                                   help="the name of the folder to save the model in",
                                   default=time.strftime('%Y-%m-%d-%H-%M-%S'))
          self.parser.add_argument("--split",
                                   type=str,
                                   help="which training split to use",
                                   choices=["endovis", "hamlyn", "hk", "c3vd", "endomapper"],
                                   default="hk")
          self.parser.add_argument("--num_layers",
                                   type=int,
                                   help="number of resnet layers",
                                   default=18,
                                   choices=[18, 34, 50, 101, 152])
          self.parser.add_argument("--dataset",
                                   type=str,
                                   help="dataset to train on",
                                   default="hk",
                                   choices=["endovis", "hamlyn", "hk", "c3vd", "endomapper"])
          self.parser.add_argument("--height",
                                   type=int,
                                   help="input image height",
                                   default=256)
          self.parser.add_argument("--width",
                                   type=int,
                                   help="input image width",
                                   default=320)
          self.parser.add_argument("--disparity_smoothness", # Les
                                   type=float,
                                   help="disparity smoothness weight",
                                   default=0.01)
          self.parser.add_argument("--reconstruction_constraint", # Lds
                                   type=float,
                                   help="consistency constraint weight",
                                   default=0.2)
          self.parser.add_argument("--reflec_constraint", # La
                                   type=float,
                                   help="epipolar constraint weight",
                                   default=0.2)
          self.parser.add_argument("--reprojection_constraint", #Lms
                                   type=float,
                                   help="geometry constraint weight",
                                   default=1)
          self.parser.add_argument("--scales",
                                   nargs="+",
                                   type=int,
                                   help="scales used in the loss",
                                   default=[0, 1, 2, 3])
          self.parser.add_argument("--min_depth",
                                   type=float,
                                   help="minimum depth",
                                   default=0.1)
          self.parser.add_argument("--max_depth",
                                   type=float,
                                   help="maximum depth",
                                   default=150.0)
          self.parser.add_argument("--frame_ids",
                                   nargs="+",
                                   type=int,
                                   help="frames to load",
                                   default=[0, -1, 1])
          # added by rema for training

          self.parser.add_argument("--inpaint_pseudo_gt_dir", type=str,
                                   help='path to the pseudo gt directory',
                                   default=None)
          self.parser.add_argument("--flipping", help="if set, uses flipping",
                                   action="store_true")
          self.parser.add_argument("--rotating", help="if set, uses rotating",
                                   action="store_true")
          self.parser.add_argument("--light_in_depth", help="if set, uses light in depth",
                                   action="store_true")
          self.parser.add_argument("--input_mask_path", type=str,
                                   help='path to the input mask',
                                   default=None)
          self.parser.add_argument("--distorted", help="if set, uses distorted intrinsics",
                                   action="store_true")
          self.parser.add_argument("--config", type=str,
                                   help='path to the config file',
                                   default=None)
          self.parser.add_argument("--aug_type", type=str,
                                   help='type of data augmentation',
                                   default='',
                                   choices=['', 'add', 'rem', 'addrem'])
          self.parser.add_argument("--automasking",
                                   help="if set, does auto-masking",
                                   action="store_true")
          self.parser.add_argument("--png",
                                   help="if set, trains from raw KITTI png files (instead of jpgs)",
                                   action="store_true")
          self.parser.add_argument("--noadjust", help="if set, does not adjust the shading",
                                   action="store_true")
          self.parser.add_argument("--disparity_spatial_constraint",
                                   help="disparity spatial constraint weight",
                                   type=float,
                                   default=0.0)
          self.parser.add_argument("--specular_grow_tolerance",
                                   help="max brightness difference (in [0,1] units) for a pixel "
                                        "to be folded into the specular pseudo-label mask, "
                                        "compared against the ACTUAL brightness of the nearest "
                                        "confirmed (97th percentile core) specular pixel -- never "
                                        "a global/whole-image threshold and never another "
                                        "already-grown halo pixel, which is what keeps growth "
                                        "bounded even on smoothly-lit domains",
                                   type=float,
                                   default=0.01)
          self.parser.add_argument("--specular_grow_iters",
                                   help="number of 3x3 dilation hops growth can take (bounds how "
                                        "far, in pixels, growth can reach from a confirmed "
                                        "specular pixel) -- the loop also breaks early if a hop "
                                        "adds zero new pixels. Kept deliberately small: raising it "
                                        "to let growth reach all the way into a large uniform "
                                        "highlight's flat interior also lets it reach further out "
                                        "along real but unrelated nearby structure (verified "
                                        "directly: it produced a visible new streak extending from "
                                        "an otherwise correct blob). A large, solid highlight's "
                                        "un-reached interior is instead handled by the "
                                        "always-on enclosed-hole fill after growth, which can only "
                                        "ever patch pixels fully surrounded by the mask -- never "
                                        "extend it outward -- so it fixes exactly this case "
                                        "without that side effect.",
                                   type=int,
                                   default=8)
          self.parser.add_argument("--specular_grow_max_coverage",
                                   help="hard per-frame ceiling on grown_mask coverage as a "
                                        "fraction of the image (e.g. 0.06 = 6%%), applied by "
                                        "keeping only the lowest-hop-distance (closest-to-core) "
                                        "grown pixels when natural growth would exceed it, leaving "
                                        "frames that stay under the cap untouched. Isolates "
                                        "growth's pixel-SELECTION benefit from the side effect of "
                                        "an unpredictable, content-dependent mask size: with no "
                                        "cap (1.0), growth coverage ranges from ~3%% (no real "
                                        "specularity) to 15%%+ (heavily specular frames) with "
                                        "growth on, vs an exact, zero-variance 3%% for growth off "
                                        "-- a large swing in per-frame training-time mask size that "
                                        "a fixed cap removes without discarding growth's benefit "
                                        "on frames that actually have real halos to close.",
                                   type=float,
                                   default=1.0)
          self.parser.add_argument("--specular_kernel_filtered",
                                   help="neighbor-averaging kernel size used to reconstruct "
                                        "`filtered` (the specular-suppressed real image fed to "
                                        "the depth encoder) -- kept small to preserve real image "
                                        "resolution/texture outside the masked pixels",
                                   type=int,
                                   default=7)
          self.parser.add_argument("--specular_kernel_reflectance",
                                   help="neighbor-averaging kernel size used to reconstruct the "
                                        "`reflectance` channel -- can be larger than "
                                        "specular_kernel_filtered since what matters there is a "
                                        "representative local color/albedo value, not spatial "
                                        "sharpness, and a wider window blends more smoothly",
                                   type=int,
                                   default=21)
          self.parser.add_argument("--specular_border_frac",
                                   help="fraction of the frame margin excluded from the specular "
                                        "heuristic (core threshold + region growing), to avoid "
                                        "the endoscope vignette edge being mistaken for a "
                                        "specular highlight. 0 disables border exclusion entirely.",
                                   type=float,
                                   default=0.05)
          self.parser.add_argument("--specular_core_brightness_floor",
                                   help="minimum absolute brightness a pixel must have (in "
                                        "addition to being in the top 97th-percentile of local "
                                        "contrast) to join the core/seed set. The 97th-percentile "
                                        "cutoff alone is purely per-frame-relative, so it always "
                                        "forces exactly ~3%% of every frame into the core set even "
                                        "on frames with little or no genuine specular highlight -- "
                                        "this floor turns that into a cap instead: a frame with no "
                                        "pixels this bright ends up with a smaller (or empty) core, "
                                        "not an artificially padded one.",
                                   type=float,
                                   default=0.75)
          self.parser.add_argument("--specular_core_closing_kernel",
                                   help="morphological closing (dilate then erode, same kernel "
                                        "size) applied to core_mask before growth/hole-fill. The "
                                        "97th-percentile local-contrast threshold selects a sparse, "
                                        "porous scatter of pixels even inside a large, obviously "
                                        "saturated highlight (contrast fluctuates pixel-to-pixel "
                                        "even on a flat bright area) -- verified directly: a 40x40 "
                                        "box around a real specular peak had only 5-14%% core "
                                        "coverage, scattered with gaps, and the gaps stayed "
                                        "connected to the image border through that porosity, so "
                                        "hole-fill (which only patches FULLY enclosed regions) added "
                                        "zero pixels. Closing seals small gaps first so a genuinely "
                                        "large cluster becomes one sealed blob hole-fill can then "
                                        "solidify. 1 (or any value <=1) disables it.",
                                   type=int,
                                   default=1)
          self.parser.add_argument("--specular_fov_black_threshold",
                                   help="brightness below which a pixel is considered outside the "
                                        "endoscope's visible field of view (pure black background/"
                                        "vignette corners, e.g. C3VD/C3VDv2's octagonal crop) rather "
                                        "than real tissue -- used to detect the TRUE shape of the "
                                        "visible FOV directly from the image, instead of assuming "
                                        "a rectangular inset that misses diagonal vignette edges",
                                   type=float,
                                   default=0.03)
          self.parser.add_argument("--specular_fov_margin",
                                   help="pixels the detected FOV boundary is eroded inward by, so "
                                        "the blur(,9)-radius halo around that boundary (which always "
                                        "reads as locally brighter than its neighborhood once "
                                        "outside-FOV black pixels enter the 9x9 blur window) is "
                                        "excluded from the specular heuristic and from growth",
                                   type=int,
                                   default=6)
          self.parser.add_argument("--specular_fill_iters",
                                   help="number of 3x3-radius diffusion passes used to fill "
                                        "pixels that have no valid unmasked neighbor within "
                                        "specular_kernel_filtered/_reflectance (e.g. the interior "
                                        "of a specular region wider than the kernel) -- "
                                        "propagates already-filled boundary color further inward "
                                        "pass by pass instead of leaving a near-black hole",
                                   type=int,
                                   default=40)
          self.parser.add_argument("--specular_white_threshold",
                                   help="pixels whose darkest RGB channel exceeds this value are "
                                        "treated as confirmed specular pixels outright (folded "
                                        "into the core/seed set used for region growing), "
                                        "regardless of the 97th-percentile core_mask threshold. "
                                        "Default of 1.01 is above the valid [0,1] pixel range, so "
                                        "this is a no-op unless explicitly lowered.",
                                   type=float,
                                   default=1.01)

          # OPTIMIZATION options
          self.parser.add_argument("--batch_size",
                                   type=int,
                                   help="batch size",
                                   default=8)
          self.parser.add_argument("--learning_rate",
                                   type=float,
                                   help="learning rate",
                                   default=5e-5)
          self.parser.add_argument("--num_epochs",
                                   type=int,
                                   help="number of epochs",
                                   default=20)
          self.parser.add_argument("--scheduler_step_size",
                                   type=int,
                                   help="step size of the scheduler",
                                   default=10)

          # ABLATION options
          self.parser.add_argument("--weights_init",
                                   type=str,
                                   help="pretrained or scratch",
                                   default="pretrained",
                                   choices=["pretrained", "scratch"])
          self.parser.add_argument("--pose_model_input",
                                   type=str,
                                   help="how many images the pose network gets",
                                   default="pairs",
                                   choices=["pairs", "all"])

          # SYSTEM options
          self.parser.add_argument("--no_cuda",
                                   help="if set disables CUDA",
                                   action="store_true")
          self.parser.add_argument("--num_workers",
                                   type=int,
                                   help="number of dataloader workers",
                                   default=12)

          # LOADING options
          self.parser.add_argument("--load_weights_folder",
                                   type=str,
                                   help="name of model to load"
                                   )
          self.parser.add_argument("--models_to_load",
                                   nargs="+",
                                   type=str,
                                   help="models to load")

          # LOGGING options
          self.parser.add_argument("--log_frequency",
                                   type=int,
                                   help="number of batches between each tensorboard log",
                                   default=200)
          self.parser.add_argument("--save_frequency",
                                   type=int,
                                   help="number of epochs between each save",
                                   default=10)

          # EVALUATION options
          self.parser.add_argument("--eval_stereo",
                                   help="if set evaluates in stereo mode",
                                   action="store_true")
          self.parser.add_argument("--eval_mono",
                                   help="if set evaluates in mono mode",
                                   action="store_true",
                                   default=True)
          self.parser.add_argument("--disable_median_scaling",
                                   help="if set disables median scaling in evaluation",
                                   action="store_true")
          self.parser.add_argument("--pred_depth_scale_factor",
                                   help="if set multiplies predictions by this number",
                                   type=float,
                                   default=1)
          self.parser.add_argument("--ext_disp_to_eval",
                                   type=str,
                                   help="optional path to a .npy disparities file to evaluate")
          self.parser.add_argument("--eval_split",
                                   type=str,
                                   default="endovis",
                                   choices=["endovis","hamlyn", "c3vd", "c3vdv2"],
                                   help="which split to run eval on")
          self.parser.add_argument("--save_pred_disps",
                                   help="if set saves predicted disparities",
                                   action="store_true")
          self.parser.add_argument("--no_eval",
                                   help="if set disables evaluation",
                                   action="store_true")
          self.parser.add_argument("--eval_eigen_to_benchmark",
                                   help="if set assume we are loading eigen results from npy but "
                                        "we want to evaluate using the new benchmark.",
                                   action="store_true")
          self.parser.add_argument("--eval_out_dir",
                                   help="if set will output the disparities to this folder",
                                   type=str)
          self.parser.add_argument("--post_process",
                                   help="if set will perform the flipping post processing "
                                        "from the original monodepth paper",
                                   action="store_true")

        
        
     def parse(self):
          self.options = self.parser.parse_args()
          
          if self.options.config is not None:
               # Load options from the configuration file
               with open(self.options.config, 'r') as f:
                    config = json.load(f)
               
               # Update default options with options from the configuration file
               for key, value in config.items():
                    setattr(self.options, key, value)
          return self.options
