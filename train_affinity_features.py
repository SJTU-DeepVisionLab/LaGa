#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Modified by Jiazhong Cen, 2025

import torch
from random import randint
from gaussian_renderer import network_gui, render_contrastive_feature
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np

from utils.sh_utils import SH2RGB

def training(dataset, opt, pipe, iteration, downsample = False):

    dataset.need_features = False
    dataset.need_masks = True

    gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 0.2 if 'Replica' in dataset.source_path else 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    print('Number of 3D Gaussians:', gaussians.get_xyz.shape[0])
    
    point_colors = gaussians._features_dc.detach().clone()
    point_colors = SH2RGB(point_colors.squeeze())
    point_colors = torch.clip(point_colors, 0, 1)
    
    num_lvl = 1 if 'scannet' in dataset.source_path or '3dovs' in dataset.source_path else 3
    ADJUSTABLE = 'scannet' in dataset.source_path
    print('Gaussian adjustable:', ADJUSTABLE)
    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False, adjustable= ADJUSTABLE, opacity_adjustable=True)

    MULTI_LVL_DIM = [16, 8, 8] if num_lvl == 3 else [32]

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()
    # feature_gaussians.training_setup(opt)

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if iteration < -1:
            viewpoint_cam = viewpoint_stack[0]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        with torch.no_grad():
            # gather the sam masks
            # N_mask, H, W
            sam_masks = viewpoint_cam.original_masks.cuda().float()

            # for garden, disable it in other scenes
            # sam_masks = torch.nn.functional.interpolate(sam_masks.unsqueeze(1), (sam_masks.shape[-2] //4,sam_masks.shape[-1] //4), mode='nearest').squeeze()
        
            multi_lvl_masks = []
            multi_lvl_masks_before_filt = []
            MASK_SIZE_THRESH = 400

            sam_masks[sam_masks == -1] = -1000
            for lvl in range(1,num_lvl+1) if num_lvl > 1 else [0]:
                tmp_masks = sam_masks[lvl].clone()
                smallest_index = sam_masks[lvl-1].int().max()+1 if lvl > 0 else 0 
                tmp_masks -= smallest_index-1
                tmp_masks[tmp_masks < 0] = 0
                tmp_lvl_masks = torch.nn.functional.one_hot(tmp_masks.long(), num_classes=tmp_masks.max().int().item()+1)[:,:,1:].float()
                multi_lvl_masks_before_filt.append(tmp_lvl_masks)
            
            for lvl in range(num_lvl):
                tmp_mask = multi_lvl_masks_before_filt[lvl]
                mask_non_zero_count = tmp_mask.sum(dim = (0,1))
                fi = mask_non_zero_count > MASK_SIZE_THRESH

                for sub_lvl in range(0, lvl):
                    tmp_mask2 = multi_lvl_masks_before_filt[sub_lvl]

                    intersection = torch.einsum('hwc,hwf->cf', tmp_mask, tmp_mask2)
                    union = tmp_mask.sum(dim = 0).sum(dim = 0).unsqueeze(-1) + tmp_mask2.sum(dim = 0).sum(dim = 0) - intersection
                    inter_over_union = intersection / union
                    t_fi = torch.logical_and(fi, inter_over_union.max(dim = 1)[0] < 0.8)
                    fi = fi if t_fi.count_nonzero() == 0 else t_fi

                # for garden, disable it in other scenes
                # if fi.count_nonzero() > 100:
                #     # random select 100
                #     fi = torch.logical_and(fi, torch.rand_like(fi, dtype=torch.float) < 100 / fi.count_nonzero())
                tmp_mask = tmp_mask[:,:,fi]
                multi_lvl_masks.append(tmp_mask)

            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

        torch.cuda.empty_cache()

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, multi_lvl_norm=True, multi_lvl_dim=MULTI_LVL_DIM, smooth_type = None, smooth_weights=None)
        # for scannet
        # render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, multi_lvl_norm=True, multi_lvl_dim=MULTI_LVL_DIM, smooth_type = 'traditional', smooth_weights=None)

        rendered_features = render_pkg_feat["render"]

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), viewpoint_cam.original_masks.shape[-2:], mode='bilinear').squeeze(0)

        if downsample:
            rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0),(rendered_features.shape[-2]//args.downsample_scale, rendered_features.shape[-1]//args.downsample_scale), mode='bilinear').squeeze(0)
            multi_lvl_masks = [torch.nn.functional.interpolate(mlm.unsqueeze(0).permute([0,3,1,2]), rendered_features.shape[-2:], mode='nearest').squeeze(0).permute([1,2,0]) for mlm in multi_lvl_masks]
        

        multi_lvl_rendered_feature_norm = 0
        pre_dim = 0
        for dim in MULTI_LVL_DIM:
            multi_lvl_rendered_feature_norm += rendered_features[pre_dim:pre_dim+dim, :,:].norm(dim = 0).mean()
            pre_dim += dim

        rendered_feature_norm = multi_lvl_rendered_feature_norm / len(MULTI_LVL_DIM)
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2

        rendered_features = rendered_features.permute([1,2,0])
        multi_lvl_features = []
        pre_dim = 0
        for dim in MULTI_LVL_DIM:
            # NOTE: FINEST -> MEDIAN -> COARSEST
            # a. naive concatenation
            cur_lvl_features = rendered_features[:,:,0:pre_dim+dim]

            multi_lvl_features.insert(0, cur_lvl_features)
            pre_dim += dim

        multi_lvl_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim = -1, p = 2) for mlf, mlm in zip(multi_lvl_features, multi_lvl_masks)]
        multi_lvl_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim = -1, p = 2)) for mlp, mlf in zip(multi_lvl_prototypes, multi_lvl_features)]

        # use finer level masks to segment on the coarser level features, if two features belong to the same target at finer level, they should also belong to the same target at coarser level
        # On the contrary, if two features belong to different targets at coarser level, they should also belong to different targets at finer level

        fine_mask_on_coarse_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim = -1, p = 2) for mlf, mlm in zip(multi_lvl_features[1:], multi_lvl_masks[:-1])]
        fine_mask_on_coarse_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim = -1, p = 2)) for mlp, mlf in zip(fine_mask_on_coarse_prototypes, multi_lvl_features[1:])]

        coarse_mask_on_fine_prototypes = [torch.nn.functional.normalize(torch.einsum('hwc,hwf->fc', mlf, mlm), dim = -1, p = 2) for mlf, mlm in zip(multi_lvl_features[:-1], multi_lvl_masks[1:])]
        coarse_mask_on_fine_cos_seg = [torch.einsum('fc,hwc->hwf', mlp, torch.nn.functional.normalize(mlf, dim = -1, p = 2)) for mlp, mlf in zip(coarse_mask_on_fine_prototypes, multi_lvl_features[:-1])]

        multi_lvl_loss = 0
        if num_lvl > 1:
            for lvl in range(num_lvl):
                pos_mask = multi_lvl_masks[lvl] == 1
                neg_mask = multi_lvl_masks[lvl] == 0

                neg_sample_rate = pos_mask.sum(dim = (0,1)) / neg_mask.sum(dim = (0,1))
                random_num = torch.rand_like(multi_lvl_cos_seg[lvl])
                sampled_neg_mask = torch.logical_and(neg_mask, random_num < neg_sample_rate)
                sampled_neg_mask = torch.logical_or(sampled_neg_mask, torch.logical_and(multi_lvl_cos_seg[lvl] > 0.5, neg_mask))

                if lvl == 0:
                    multi_lvl_loss += -torch.min(multi_lvl_cos_seg[lvl], fine_mask_on_coarse_cos_seg[lvl])[pos_mask].mean() + torch.relu((multi_lvl_cos_seg[lvl])[sampled_neg_mask]).mean()

                elif lvl == num_lvl-1:
                    multi_lvl_loss += -(multi_lvl_cos_seg[lvl])[pos_mask].mean() + torch.relu(torch.max(multi_lvl_cos_seg[lvl], coarse_mask_on_fine_cos_seg[lvl-1])[sampled_neg_mask]).mean()

                else:
                    multi_lvl_loss += -torch.min(multi_lvl_cos_seg[lvl], fine_mask_on_coarse_cos_seg[lvl])[pos_mask].mean() + torch.relu(torch.max(multi_lvl_cos_seg[lvl], coarse_mask_on_fine_cos_seg[lvl-1])[sampled_neg_mask]).mean()

        else:
            pos_mask = multi_lvl_masks[0] == 1
            neg_mask = multi_lvl_masks[0] == 0

            neg_sample_rate = pos_mask.sum(dim = (0,1)) / neg_mask.sum(dim = (0,1))
            random_num = torch.rand_like(multi_lvl_cos_seg[0])
            sampled_neg_mask = torch.logical_and(neg_mask, random_num < neg_sample_rate)
            sampled_neg_mask = torch.logical_or(sampled_neg_mask, torch.logical_and(multi_lvl_cos_seg[0] > 0.5, neg_mask))

            multi_lvl_loss += -(multi_lvl_cos_seg[0])[pos_mask].mean() + torch.relu((multi_lvl_cos_seg[0])[sampled_neg_mask]).mean()

        loss = multi_lvl_loss + rendered_feature_norm_reg

        with torch.no_grad():
            multi_lvl_cos_seg = torch.cat(multi_lvl_cos_seg, dim = -1)
            multi_lvl_lbl = torch.cat(multi_lvl_masks, dim = -1)
            cosine_pos = multi_lvl_cos_seg[multi_lvl_lbl == 1].mean()
            cosine_neg = multi_lvl_cos_seg[multi_lvl_lbl == 0].mean()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Multi Lvl Loss": f"{multi_lvl_loss.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}",
            })
            progress_bar.update(10)

        if iteration in args.save_iterations:
            scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = None)

    scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = None)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature'])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--downsample_scale",default=2, type=int)

    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.iteration, args.downsample)

    # All done
    print("\nTraining complete.")