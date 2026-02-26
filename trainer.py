from __future__ import absolute_import, division, print_function

import numpy as np
from itertools import chain
from PIL import Image, ImageFilter

import glob
from sklearn.model_selection import train_test_split

import time
import json
import datasets
import networks
import torch.optim as optim
from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms
import random
random.seed(42)

class Trainer:
    def __init__(self, options):


        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # normalize data_path once
        if isinstance(self.opt.data_path, str):
            self.opt.data_path = [os.path.abspath(self.opt.data_path)]
        else:
            self.opt.data_path = [os.path.abspath(p) for p in self.opt.data_path]

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}  # ??
        self.parameters_to_train = []  # ??
        self.parameters_to_train_1 = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)  # 4
        self.num_input_frames = len(self.opt.frame_ids)  # 3
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames  # 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")  # 18
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["decompose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")  # 18
        self.models["decompose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["decompose_encoder"].parameters())

        self.models["decompose"] = networks.decompose_decoder(
            self.models["decompose_encoder"].num_ch_enc, self.opt.scales)
        self.models["decompose"].to(self.device)
        self.parameters_to_train += list(self.models["decompose"].parameters())

        if not self.opt.noadjust:
            self.models["adjust_net"]=networks.adjust_net()
            self.models["adjust_net"].to(self.device)
            self.parameters_to_train += list(self.models["adjust_net"].parameters())

        self.models["pose_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers,
            self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["pose_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["pose_encoder"].parameters())

        self.models["pose"] = networks.PoseDecoder(
            self.models["pose_encoder"].num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train,self.opt.learning_rate)
        sched = self.opt.scheduler_step_size if type(self.opt.scheduler_step_size) == list else [self.opt.scheduler_step_size]
        self.model_lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.model_optimizer, sched, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset,
                         "hk": datasets.HKDataset,
                         "c3vd": datasets.C3VDDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        if not isinstance(self.opt.split, list):
            self.opt.split = [self.opt.split]

        fpath = [os.path.join(os.path.dirname(__file__), "splits", split, "{}_files.txt") for split in self.opt.split]        
        img_ext = '.png' if self.opt.png else '.jpg'

        val_filenames =[]
        train_filenames = []
        for i, split in enumerate(self.opt.split):
            #data_path = os.path.abspath(self.opt.data_path)
            if split == "hk" or split == "c3vd":
                aug = self.opt.aug_type
                if not os.path.exists(fpath[i].format(f"train{aug}")):

                    data_path = self.opt.data_path
                    train_filenames, test_filenames, val_filenames = self.generate_train_test_val(split, data_path, img_ext)

                    # Extract the directory from the file path pattern
                    directory = os.path.dirname(fpath[i])

                    # Ensure the directory exists
                    os.makedirs(directory, exist_ok=True)

                    # Save train_filenames to a text file
                    with open(fpath[i].format(f"train{aug}"), 'w') as f:
                        for filename in train_filenames:
                            f.write("%s\n" % filename)

                    # Save test_filenames to a text file
                    with open(fpath[i].format(f"test{aug}"), 'w') as f:
                        for filename in test_filenames:
                            f.write("%s\n" % filename)

                    # Save val_filenames to a text file
                    with open(fpath[i].format(f"val{aug}"), 'w') as f:
                        for filename in val_filenames:
                            f.write("%s\n" % filename)
                else:
                    train_filenames.extend(readlines(fpath[i].format(f"train{aug}")))
                    val_filenames.extend(readlines(fpath[i].format(f"val{aug}")))

            else:
                train_filenames.extend(readlines(fpath[i].format("train")))
                val_filenames.extend(readlines(fpath[i].format("val")))


        print("train files:", len(train_filenames))
        print("val files:", len(val_filenames))   

        if self.opt.input_mask_path is not None:
            input_mask_pil = Image.open(self.opt.input_mask_path).convert('1').filter(ImageFilter.MinFilter(size=5))
            self.input_mask = transforms.ToTensor()(input_mask_pil).to(self.device).unsqueeze(dim=0)
        else:
            self.input_mask = None

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        if self.opt.dataset == "hk" or self.opt.dataset == "c3vd":
            train_dataset = self.dataset(
                self.opt.data_path[0], train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, 
                flipping=self.opt.flipping, rotating=self.opt.rotating,
                distorted = self.opt.distorted, inpaint_pseudo_gt_dir = self.opt.inpaint_pseudo_gt_dir)
        else:
            train_dataset = self.dataset(
                self.opt.data_path[0], train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        if self.opt.dataset == "hk" or self.opt.dataset == "c3vd":
            val_dataset = self.dataset(
            self.opt.data_path[0], val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext,
            flipping=self.opt.flipping, rotating=self.opt.rotating,
            distorted = self.opt.distorted, inpaint_pseudo_gt_dir = self.opt.inpaint_pseudo_gt_dir)
        else:
            val_dataset = self.dataset(
                self.opt.data_path[0], val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))


        self.ssim = SSIM()
        self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}

        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()


    def generate_train_test_val(self, split, data_path, img_ext):
        if split == "hk":
            traintestval = []
            for data_path in self.opt.data_path:
                contents_lists = glob.glob(os.path.join(data_path, "*"))
                sub_files =[]
                for subdir in contents_lists:
                    sub_files.append(sorted(glob.glob(os.path.join(subdir,f"*{img_ext}")))[1:-1])
                all_files = list(chain.from_iterable(sub_files))
                train_temp_f, test_f = train_test_split(all_files, test_size=0.04, shuffle=False)
                train_f, val_f = train_test_split(train_temp_f, test_size=0.1, shuffle=False)
                traintestval.append([train_f, test_f, val_f])
            train_filenames = list(chain.from_iterable([traintestval[i][0] for i in range(len(traintestval))]))
            test_filenames = list(chain.from_iterable([traintestval[i][1] for i in range(len(traintestval))]))
            val_filenames = list(chain.from_iterable([traintestval[i][2] for i in range(len(traintestval))]))
            return train_filenames, test_filenames, val_filenames
        elif split == "c3vd":
            # Initialize empty lists to accumulate filenames
            train_filenames = []
            test_filenames = []
            val_filenames = []

            for dp in data_path:
                test_seq = ["cecum_t2_b", "trans_t4_a", "sigmoid_t3_a"]
                train_seq = ["cecum_t1_a", "cecum_t1_b", "cecum_t2_a", "cecum_t2_c",
                            "cecum_t4_a", "cecum_t4_b", "desc_t4_a",
                            "sigmoid_t1_a", "sigmoid_t3_b", "trans_t1_a",
                            "trans_t1_b", "trans_t2_a", "trans_t2_b", "trans_t2_c",
                            "trans_t3_a", "trans_t3_b", "trans_t4_b"]
                val_seq = [
                    "cecum_t3_a",
                    "sigmoid_t2_a"
                ]

                # Extend the lists with filenames from the current data_path
                train_filenames.extend(list(chain.from_iterable([sorted(glob.glob(os.path.join(dp, seq, f"*color{img_ext}")))[1:-1] for seq in train_seq])))
                test_filenames.extend(list(chain.from_iterable([sorted(glob.glob(os.path.join(dp, seq, f"*color{img_ext}")))[1:-1] for seq in test_seq])))
                val_filenames.extend(list(chain.from_iterable([sorted(glob.glob(os.path.join(dp, seq, f"*color{img_ext}")))[1:-1] for seq in val_seq])))

            return train_filenames, test_filenames, val_filenames


    def set_train(self):
        """Convert all models to training mode
        """
        for param in self.models["encoder"].parameters():
            param.requires_grad = True
        for param in self.models["depth"].parameters():
            param.requires_grad = True
        for param in self.models["pose_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["pose"].parameters():
            param.requires_grad = True
        for param in self.models["decompose_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["decompose"].parameters():
            param.requires_grad = True
        if not self.opt.noadjust:
            for param in self.models["adjust_net"].parameters():
                param.requires_grad = True

        self.models["encoder"].train()
        self.models["depth"].train()
        self.models["pose_encoder"].train()
        self.models["pose"].train()
        self.models["decompose_encoder"].train()
        self.models["decompose"].train()
        if not self.opt.noadjust:
            self.models["adjust_net"].train()


    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        self.models["encoder"].eval()
        self.models["depth"].eval()
        self.models["pose_encoder"].eval()
        self.models["pose"].eval()
        self.models["decompose_encoder"].eval()
        self.models["decompose"].eval()
        if not self.opt.noadjust:
            self.models["adjust_net"].eval()


    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """

        print("Training")
        print(self.model_optimizer.param_groups[0]['lr'])
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            # depth, pose, decompose
            self.set_train()
            outputs, losses = self.process_batch(inputs)
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            phase = batch_idx % self.opt.log_frequency == 0

            if phase:

                self.log_time(batch_idx, duration, losses["loss"].cpu().data)
                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

        self.model_lr_scheduler.step()


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # decompose
        outputs ={}
        self.decompose(inputs,outputs)

        # we only feed the image with frame_id 0 through the depth encoder
        if self.opt.light_in_depth:
            features = self.models["encoder"](outputs["light", 0, 0].repeat_interleave(3, dim=1))
        else:
            features = self.models["encoder"](inputs["color_aug", 0, 0])
        outputs.update(self.models["depth"](features))

        # pose
        outputs.update(self.predict_poses(inputs))

        # decompose
        self.decompose_postprocess(inputs,outputs)

        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:

            pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:

                if f_i != "s":
                    if f_i < 0:
                        inputs_all = [pose_feats[f_i], pose_feats[0]]
                    else:
                        inputs_all = [pose_feats[0], pose_feats[f_i]]                                                                    

                    # pose
                    pose_inputs = [self.models["pose_encoder"](torch.cat(inputs_all, 1))]
                    axisangle, translation = self.models["pose"](pose_inputs)

                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0],invert=(f_i < 0))

        return outputs

    def decompose(self, inputs, outputs):
        for f_i in self.opt.frame_ids:

            decompose_features = self.models["decompose_encoder"](
                inputs[("color_aug", f_i, 0)]
            )

            reflectance, light, mask = self.models["decompose"](decompose_features)

            outputs[("reflectance", 0, f_i)] = reflectance
            outputs[("light", 0, f_i)] = light
            outputs[("mask", 0, f_i)] = mask

            outputs[("reprojection_color", 0, f_i)] = reflectance * light

    def decompose_postprocess(self,inputs,outputs):
        disp = outputs[("disp", 0)]
        disp = F.interpolate(disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=True)
        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            T = outputs[("cam_T_cam", 0, frame_id)]
            cam_points = self.backproject_depth[0](
                depth, inputs[("inv_K", 0)])
            pix_coords = self.project_3d[0](
                cam_points, inputs[("K", 0)], T)

            outputs[("warp", 0, frame_id)] = pix_coords

            outputs[("reflectance_warp", 0, frame_id)] = F.grid_sample(
                outputs[("reflectance", 0, frame_id)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border",align_corners=True)

            outputs[("light_warp", 0, frame_id)] = F.grid_sample(
                outputs[("light", 0, frame_id)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border",align_corners=True)

            outputs[("color_warp",0,frame_id)] = F.grid_sample(
                inputs[("color_aug", frame_id, 0)],
                outputs[("warp", 0, frame_id)],
                padding_mode="border",align_corners=True)
             # masking zero values
            mask_ones = torch.ones_like(inputs[("color_aug", frame_id, 0)])
            mask_warp = F.grid_sample(
                mask_ones,
                outputs[("warp", 0, frame_id)],
                padding_mode="zeros",align_corners=True)
            valid_mask = (mask_warp.abs().mean(dim=1, keepdim=True) > 0.0).float()
            outputs[("valid_mask", 0, frame_id)] = valid_mask

            if not self.opt.noadjust:
                outputs[("warp_diff_color", 0, frame_id)] = torch.abs(inputs[("color_aug",0,0)]-outputs[("color_warp",0,frame_id)])*valid_mask
                outputs[("transform", 0, frame_id)] = self.models["adjust_net"](outputs[("warp_diff_color", 0, frame_id)])
                outputs[("light_adjust_warp",0,frame_id)] = outputs[("transform", 0, frame_id)] + outputs[("light_warp",0,frame_id)] 
                outputs[("light_adjust_warp",0,frame_id)] = torch.clamp(outputs[("light_adjust_warp",0,frame_id)], min=0.0, max=1.0)
                outputs[("reprojection_color_warp", 0, frame_id)] = outputs[("reflectance_warp", 0, frame_id)]*outputs[("light_adjust_warp", 0, frame_id)]
            else:
                outputs[("reprojection_color_warp", 0, frame_id)] = outputs[("reflectance_warp", 0, frame_id)]*outputs[("light_warp", 0, frame_id)]




    def proportional_loss_with_threshold(self, disp_map, threshold):
        """
        Computes the loss that enforces proportionality between depth differences and spatial differences
        only when depth differences are large.

        Args:
        - disp_map (torch.Tensor): Tensor containing disp values of shape (B, 1, H, W).
        - threshold (float): The depth difference threshold for enforcing proportionality.

        Returns:
        - loss (torch.Tensor): The computed loss value.
        """


        batch_size, _, rows, cols = disp_map.shape

        # Normalize spatial distances by the maximum possible distance
        max_dist = torch.sqrt(torch.tensor(rows**2 + cols**2, dtype=torch.float32))


        # Initialize DD_map and SD_map on the same device as disp_map
        DD_map = torch.zeros((batch_size, rows, cols), dtype=disp_map.dtype, device=disp_map.device)
        SD_map = torch.zeros((batch_size, rows, cols), dtype=disp_map.dtype, device=disp_map.device)

        # Generate random indices for the entire map using PyTorch
        rand_x = torch.randint(0, rows, (batch_size, rows, cols), device=disp_map.device)
        rand_y = torch.randint(0, cols, (batch_size, rows, cols), device=disp_map.device)

        # Compute DD_map and SD_map using vectorized operations
        i_indices = torch.arange(rows, device=disp_map.device).view(1, -1, 1).expand(batch_size, rows, cols)
        j_indices = torch.arange(cols, device=disp_map.device).view(1, 1, -1).expand(batch_size, rows, cols)

        DD_map = disp_map.squeeze(1) - disp_map.squeeze(1)[np.arange(batch_size)[:, None, None], rand_x, rand_y]
        SD_map = torch.sqrt((i_indices - rand_x).float() ** 2 + (j_indices - rand_y).float() ** 2)/max_dist

        # Compute the loss
        mask = DD_map < threshold
        loss = torch.mean(mask.float() * (SD_map - DD_map) ** 2)


        return loss

    def compute_reprojection_loss(self, pred, target):

        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    #updated method for SHADeS++
    def compute_losses(self, inputs, outputs):

        losses = {}
        total_loss = 0
        loss_reflec = 0
        loss_reprojection = 0
        loss_disp_smooth = 0
        loss_decomp_recon = 0
        tau = self.opt.tau
        inpaint = "inpaint_" if self.opt.inpaint_pseudo_gt_dir is not None else ""

        # new reconstruction loss
        for frame_id in self.opt.frame_ids:

            raw = inputs[("color_aug", frame_id, 0)]
            pred = outputs[("reprojection_color", 0, frame_id)]

            recon = (
                outputs[("reflectance", 0, frame_id)] *
                outputs[("light", 0, frame_id)]
            )

            loss_decomp_recon += torch.abs(raw - recon).mean()



        for frame_id in self.opt.frame_ids[1:]: 
            mask = outputs[("valid_mask", 0, frame_id)]
            mask_comb = mask.clone()
            reflec_loss_item = torch.abs(outputs[("reflectance",0,0)] - outputs[("reflectance_warp", 0, frame_id)]).mean(1, True)

            raw = inputs[("color_aug", 0, 0)]
            pred = outputs[("reprojection_color_warp", 0, frame_id)]

            photo = self.compute_reprojection_loss(raw, pred)

            if self.opt.automasking:
                identity_reprojection_loss_item = self.compute_reprojection_loss(
                    inputs[("color", frame_id, 0)],
                    inputs[("color", 0, 0)]
                )
                identity_reprojection_loss_item += torch.randn_like(identity_reprojection_loss_item) * 1e-5

                mask_idt = (photo < identity_reprojection_loss_item).float()
                mask_comb = mask * mask_idt
                outputs["identity_selection"] = mask_comb.clone()

            reprojection_loss_item = photo

            M_soft = outputs[("mask", 0, 0)]

            loss_reflec += (
                reflec_loss_item * mask_comb * M_soft
            ).mean()

            loss_reprojection += (
                reprojection_loss_item * mask_comb
            ).mean()

        disp = outputs[("disp", 0)]
        color = inputs[("color_aug", 0, 0)]
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        loss_disp_smooth = get_smooth_loss(norm_disp, color)


        if self.opt.disparity_spatial_constraint > 0:
            loss_disp_spatial = self.proportional_loss_with_threshold(disp, 0.5)
        else:
            loss_disp_spatial = 0

        total_loss = (self.opt.reprojection_constraint*loss_reprojection / 2.0 + 
                      self.opt.reflec_constraint*(loss_reflec / 2.0) + 
                      self.opt.disparity_smoothness*loss_disp_smooth + 
                      self.opt.disparity_spatial_constraint*loss_disp_spatial)
        
        total_loss += self.opt.decomp_recon_weight * (loss_decomp_recon / len(self.opt.frame_ids))

        loss_light_smooth = get_smooth_loss(
            outputs[("light", 0, 0)],
            inputs[("color_aug", 0, 0)]
        )
        total_loss += 0.05 * loss_light_smooth

        M0 = outputs[("mask", 0, 0)]

        loss_mask_reg = (M0 ** 2).mean()

        loss_mask_tv = (
            torch.abs(M0[:, :, :, :-1] - M0[:, :, :, 1:]).mean() +
            torch.abs(M0[:, :, :-1, :] - M0[:, :, 1:, :]).mean()
        )

        total_loss += 0.001 * loss_mask_reg + 0.01 * loss_mask_tv
        losses["loss"] = total_loss

        return losses

    def val(self):
        """Validate the model on a single minibatch
        """
        if len(self.val_loader) != 0:
            self.set_eval()
            try:
                inputs = self.val_iter.next()
            except StopIteration:
                self.val_iter = iter(self.val_loader)
                inputs = self.val_iter.next()

            with torch.no_grad():
                outputs, losses = self.process_batch(inputs)
                self.log("val", inputs, outputs, losses)
                del inputs, outputs, losses

            self.set_train()


    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        inpaint = "inpaint_" if self.opt.inpaint_pseudo_gt_dir is not None else ""
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
                writer.add_image(
                        "disp/{}".format(j),
                        visualize_depth(outputs[("disp", 0)][j]), self.step)
                writer.add_image(
                        "input/{}".format(j),
                        inputs[("color", 0, 0)][j].data, self.step)
                writer.add_image(
                        "input_processed/{}".format(j),
                        inputs[(inpaint+"color_aug", 0, 0)][j].data, self.step)
                writer.add_image(
                        "reflectance/{}".format(j),
                        outputs[("reflectance", 0, 0)][j].data, self.step)
                writer.add_image(
                        "light/{}".format(j),
                        outputs[("light", 0, 0)][j].data, self.step)
                writer.add_image(
                        "AS_reprojection/{}".format(j),
                        outputs[("reprojection_color", 0, 0)][j].data, self.step)
                writer.add_image(
                    "mask/{}".format(j),
                    outputs[("mask", 0, 0)][j].data, self.step)
                writer.add_image(
                        "input_1/{}".format(j),
                        inputs[("color", 1, 0)][j].data, self.step)
                if not self.opt.noadjust:
                    writer.add_image(
                            "transform/{}".format(j),
                            outputs[("transform", 0, 1)][j].data, self.step)
                writer.add_image(
                        "light_warped/{}".format(j),
                        outputs[("light_warp", 0, 1)][j].data, self.step)
                if not self.opt.noadjust:
                    writer.add_image(
                            "light_adjust_warped/{}".format(j),
                            outputs[("light_adjust_warp", 0, 1)][j].data, self.step)
                if self.opt.automasking:
                    writer.add_image(
                            "automask/{}".format(j),
                            outputs["identity_selection"][j].data, self.step)


    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)