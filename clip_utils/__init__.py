

# import sys
# sys.path.append('/home/cenjiazhong/saga2/clip_utils')
from .clip_utils import OpenCLIPNetwork

from tqdm import tqdm
import torch, torchvision
import numpy as np

import cv2
import matplotlib.pyplot as plt
import os

default_template = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

def apply_gaussian_blur(image):
    image = image[0]
    blurred_image = cv2.GaussianBlur(image.numpy(), (15, 15), 0)
    blurred_image = cv2.GaussianBlur(blurred_image, (15, 15), 0)
    return torch.from_numpy(np.expand_dims(blurred_image, axis=0))


def get_mixed_clip_features(clip_feature_list):
    stacked_tensors = torch.stack(clip_feature_list).squeeze()
    norm = torch.norm(stacked_tensors, p=2, dim=1, keepdim=True)
    normalized_tensors = stacked_tensors / norm
    cos_matrix = torch.mm(normalized_tensors, normalized_tensors.T)
    cos_matrix*=(1-torch.eye(cos_matrix.shape[0]).cuda())
    weight = torch.nn.functional.softmax(cos_matrix.sum(dim = -1),dim=0)
    return cos_matrix, (stacked_tensors * weight[:,None]).sum(dim=0).unsqueeze(0).cpu()


def pad_to_square(tensor):
    if tensor.ndimension() == 3 and tensor.shape[-1] in [1, 3]:
        tensor = tensor.permute(2, 0, 1)

    _, h, w = tensor.shape
    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left
    padded_tensor = torch.nn.functional.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_tensor.permute([1,2,0])

# langsplat version
# def pad_img(img):
#     h, w, _ = img.shape
#     l = max(w,h)
#     pad = np.zeros((l,l,3), dtype=np.uint8)
#     if h > w:
#         pad[:,(h-w)//2:(h-w)//2 + w, :] = img
#     else:
#         pad[(w-h)//2:(w-h)//2 + h, :, :] = img
#     return pad

def expand_bbox_with_padding(masked_images, mask, scale): 
    #bbox（xmin,ymin,xmax,ymax）中的x是mask.shape[1],y是mask.shape[0] 
    bbox = torchvision.ops.masks_to_boxes(mask)     
    x_center = (bbox[:,0] + bbox[:,2]) // 2  
    y_center = (bbox[:,1] + bbox[:,3]) // 2  
    width = (bbox[:,2] - bbox[:,0]) * scale  
    height = (bbox[:,3] - bbox[:,1]) * scale  
    padd_width = int(mask.shape[2] * scale // 2)
    padd_height = int(mask.shape[1] * scale // 2) 
    # 计算扩展后bbox的坐标  
    x_min_new = (padd_width + x_center - width // 2).to(torch.int64)  
    x_max_new = (padd_width + x_center + width // 2).to(torch.int64)  
    y_min_new = (padd_height + y_center - height // 2).to(torch.int64)  
    y_max_new = (padd_height + y_center + height // 2).to(torch.int64)  
    padd = (padd_width, padd_width, padd_height, padd_height)
    # masked_image_padd = torch.zeros(masked_images.shape[0],masked_images.shape[1]+2*padd_height,masked_images.shape[2]+2*padd_width,3)
    masked_image_padd = torch.nn.functional.pad(masked_images.permute(0,3,1,2),padd, mode='constant', value=0).permute(0,2,3,1)    
    return torch.stack((x_min_new, y_min_new, x_max_new, y_max_new),dim=1), masked_image_padd

@torch.no_grad()
def get_features_from_image_and_masks(args, clip_model, image: np.array, masks: torch.tensor, background = 1.):

    image_shape = image.shape[:2]
    masks = torch.nn.functional.interpolate(masks.unsqueeze(0).float(), image_shape, mode='bilinear').squeeze(0)
    masks[masks > 0.5] = 1
    masks[masks != 1] = 0
    masks = masks.cpu()
    mask_size = masks.shape[1]*masks.shape[2]

    original_image = torch.from_numpy(image)[None]
    background = apply_gaussian_blur(original_image)
    masked_images = masks[:,:,:,None] * original_image + (1 - masks[:,:,:,None]) * background
    black_maksed_images = masks[:,:,:,None] * original_image + (1 - masks[:,:,:,None]) * 0.0

    bboxes_s1 = torchvision.ops.masks_to_boxes(masks)
    if args.multi_scale:
        extra_bbox_list=[]
        extra_masked_image_list=[]
        for index in range(len(args.extra_scales)):
            tmp_bboxes, tmp_images=expand_bbox_with_padding(masked_images, masks, scale=args.extra_scales[index])
            extra_bbox_list.append(tmp_bboxes)
            extra_masked_image_list.append(tmp_images)
    bboxes_s1 = bboxes_s1.int().tolist()
    cropped_seg_image_features1x = []
    masked_features=[]

    for seg_idx in range(len(bboxes_s1)):
        with torch.no_grad():
            clip_feature_list = []
            tmp_image_s1 = masked_images[seg_idx][bboxes_s1[seg_idx][1]:bboxes_s1[seg_idx][3], bboxes_s1[seg_idx][0]:bboxes_s1[seg_idx][2], :]
            tmp_image_s1=pad_to_square(tmp_image_s1)
            masked_image_clip_features_s1 = clip_model.encode_image(tmp_image_s1[None,...].cuda().permute([0,3,1,2]) / 255.0)
            masked_image_clip_features = masked_image_clip_features_s1.cpu()

            black_maksed_image = black_maksed_images[seg_idx][bboxes_s1[seg_idx][1]:bboxes_s1[seg_idx][3], bboxes_s1[seg_idx][0]:bboxes_s1[seg_idx][2], :]
            black_maksed_image = pad_to_square(black_maksed_image)
            masked_clip_feature = clip_model.encode_image(black_maksed_image[None,...].cuda().permute([0,3,1,2]) / 255.0)
            masked_features.append(masked_clip_feature)
            # vis_root = "./vis_check/waldo_kitchen"
            # os.makedirs(vis_root, exist_ok=True)
            # plt.imsave(os.path.join(vis_root,"crop"+str(torch.sum(masks[seg_idx]==1).item()/mask_size)+".jpg"), black_maksed_image.cpu().numpy() / 255.0)

            clip_feature_list.append(masked_image_clip_features_s1) # masked_image_clip_features_s1 (1,512)

            if args.multi_scale and torch.sum(masks[seg_idx]==1)/mask_size<0.016:
                for i in range(len(args.extra_scales)):
                    extra_masked_images = extra_masked_image_list[i] #extra_masked_images(n,h,w,3)
                    extra_bboxes = extra_bbox_list[i]
                    temp_image = extra_masked_images[seg_idx][extra_bboxes[seg_idx][1]:extra_bboxes[seg_idx][3], extra_bboxes[seg_idx][0]:extra_bboxes[seg_idx][2], :]
                    temp_image = pad_to_square(temp_image)
                    extra_masked_image_clip_features = clip_model.encode_image(temp_image[None,...].cuda().permute([0,3,1,2]) / 255.0)
                    clip_feature_list.append(extra_masked_image_clip_features)

                cos_matrix, mixed_clip_features = get_mixed_clip_features(clip_feature_list)
                cropped_seg_image_features1x.append(mixed_clip_features)
            else:
                cropped_seg_image_features1x.append(masked_clip_feature.cpu())

    cropped_seg_image_features1x = torch.cat(cropped_seg_image_features1x, dim=0)
    masked_features = torch.cat(masked_features, dim=0)
    return cropped_seg_image_features1x,  masked_features



@torch.no_grad()
def get_features_from_image_and_masks_langsplat(clip_model, image: np.array, masks: torch.tensor):

    image_shape = image.shape[:2]
    masks = torch.nn.functional.interpolate(masks.unsqueeze(0).float(), image_shape, mode='bilinear').squeeze(0)
    masks[masks > 0.5] = 1
    masks[masks != 1] = 0
    masks = masks.cpu()
    mask_size = masks.shape[1]*masks.shape[2]

    original_image = torch.from_numpy(image)[None]
    # background = apply_gaussian_blur(original_image)
    # masked_images = masks[:,:,:,None] * original_image + (1 - masks[:,:,:,None]) * background
    black_masked_images = masks[:,:,:,None] * original_image + (1 - masks[:,:,:,None]) * 0.0

    bboxes_s1 = torchvision.ops.masks_to_boxes(masks)
    # if args.multi_scale:
    #     extra_bbox_list=[]
    #     extra_masked_image_list=[]
    #     for index in range(len(args.extra_scales)):
    #         tmp_bboxes, tmp_images=expand_bbox_with_padding(black_masked_images, masks, scale=args.extra_scales[index])
    #         extra_bbox_list.append(tmp_bboxes)
    #         extra_masked_image_list.append(tmp_images)
    bboxes_s1 = bboxes_s1.int().tolist()
    cropped_seg_image_features1x = []
    # masked_features=[]

    clip_feature_list = []
    for seg_idx in range(len(bboxes_s1)):
        with torch.no_grad():
            tmp_image_s1 = black_masked_images[seg_idx][bboxes_s1[seg_idx][1]:bboxes_s1[seg_idx][3], bboxes_s1[seg_idx][0]:bboxes_s1[seg_idx][2], :]
            tmp_image_s1=pad_to_square(tmp_image_s1)
            masked_image_clip_features_s1 = clip_model.encode_image(tmp_image_s1[None,...].cuda().permute([0,3,1,2]) / 255.0)
            # masked_image_clip_features = masked_image_clip_features_s1.cpu()

            # black_maksed_image = black_maksed_images[seg_idx][bboxes_s1[seg_idx][1]:bboxes_s1[seg_idx][3], bboxes_s1[seg_idx][0]:bboxes_s1[seg_idx][2], :]
            # black_maksed_image = pad_to_square(black_maksed_image)
            # masked_clip_feature = clip_model.encode_image(black_maksed_image[None,...].cuda().permute([0,3,1,2]) / 255.0)
            # masked_features.append(masked_clip_feature)
            # vis_root = "./vis_check/waldo_kitchen"
            # os.makedirs(vis_root, exist_ok=True)
            # plt.imsave(os.path.join(vis_root,"crop"+str(torch.sum(masks[seg_idx]==1).item()/mask_size)+".jpg"), black_maksed_image.cpu().numpy() / 255.0)

            clip_feature_list.append(masked_image_clip_features_s1) # masked_image_clip_features_s1 (1,512)

            # if args.multi_scale and torch.sum(masks[seg_idx]==1)/mask_size<0.016:
            #     for i in range(len(args.extra_scales)):
            #         extra_masked_images = extra_masked_image_list[i] #extra_masked_images(n,h,w,3)
            #         extra_bboxes = extra_bbox_list[i]
            #         temp_image = extra_masked_images[seg_idx][extra_bboxes[seg_idx][1]:extra_bboxes[seg_idx][3], extra_bboxes[seg_idx][0]:extra_bboxes[seg_idx][2], :]
            #         temp_image = pad_to_square(temp_image)
            #         extra_masked_image_clip_features = clip_model.encode_image(temp_image[None,...].cuda().permute([0,3,1,2]) / 255.0)
            #         clip_feature_list.append(extra_masked_image_clip_features)

            #     cos_matrix, mixed_clip_features = get_mixed_clip_features(clip_feature_list)
            #     cropped_seg_image_features1x.append(mixed_clip_features)
            # else:
            #     cropped_seg_image_features1x.append(masked_clip_feature.cpu())

    cropped_seg_image_features1x = torch.cat(clip_feature_list, dim=0)

    return cropped_seg_image_features1x


def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def mask2segmap(masks, image):
    seg_img_list = []
    seg_map = -np.ones(image.shape[:2], dtype=np.int32)
    for i in range(len(masks)):
        mask = masks[i]
        seg_img = get_seg_img(mask, image)
        pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
        seg_img_list.append(pad_seg_img)

        seg_map[masks[i]['segmentation']] = i
    seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
    seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

    return seg_imgs, seg_map

def get_scores(clip_model, images_features, prompt):
    with torch.no_grad():
        clip_model.set_positives([prompt])
        r_scores = []
        images_features = images_features.cuda()
        relevancy_score = clip_model.get_relevancy(images_features, 0)
        
        r_score = relevancy_score[:,0]
    return r_score

def get_scores_with_template(clip_model, images_features, prompt, template = default_template):
    with torch.no_grad():
        clip_model.set_positives([t.format(prompt) for t in template])
        r_scores = []

        for i,f in enumerate(images_features):

            # N_image_features x N_pos x 2
            relevancy_scores = clip_model.get_relevancy_with_template(f)
            # N_image_features x N_pos
            r_score = relevancy_scores[...,0]
            r_scores.append(r_score)

    return r_score

def get_scores_with_template(clip_model, images_features, prompt, template = default_template):
    with torch.no_grad():
        # clip_model.set_positives([t.format(prompt) for t in template])
        from time import time
        start_time = time()
        clip_model.set_positive_with_template(prompt, template)
        # print('set_positive_with_template', time() - start_time)
        # N_image_features x N_pos x 2
        start_time = time()
        relevancy_scores = clip_model.get_relevancy_with_template(images_features)
        # print('get_relevancy_with_template', time() - start_time)
        # N_image_features x N_pos
        r_score = relevancy_scores[...,0]

    return r_score


def get_relevancy_cosine(clip_model, images_features, prompt):
    with torch.no_grad():
        clip_model.set_positives([prompt])
        images_features = images_features.cuda()
        return clip_model.get_relevancy_cosine(images_features, 0)

def get_relevancy_cosine_with_template(clip_model, images_features, prompt, template = default_template):
    with torch.no_grad():
        clip_model.set_positive_with_template(prompt, template)
        images_features = images_features.cuda()
        return clip_model.get_relevancy_cosine(images_features, 0)


def get_segmentation(clip_model, images_features, images_masks, prompts = []):
    with torch.no_grad():
        clip_model.set_positives(prompts)
    images_scores = []
    for i,f in tqdm(enumerate(images_features)):
        with torch.no_grad():
            # f = torch.nn.functional.normalize(f, dim = -1)
            # k, n_p
            segmentation_score = clip_model.get_segmentation(f)
            # n_p, h, w
            image_score = torch.einsum('kp,khw->kphw', segmentation_score, images_masks[i]).sum(dim = 0) / (images_masks[i].sum(dim = 0, keepdim = True)+1e-9)
        images_scores.append(image_score)
    return images_scores


import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

def get_3d_mask(args, pipeline, scene_gaussians, cameras, image_names, images_scores, save_path = None, filtered_views = None):
    tmp_mask = torch.zeros_like(scene_gaussians.get_mask).float().detach().clone()
    tmp_mask.requires_grad = True

    for it, view in tqdm(enumerate(cameras)):
        if filtered_views is not None and it not in filtered_views:
            continue
        image_idx = None
        try:
            image_idx = image_names.index(view.image_name+'.jpg')
        except:
            continue

        background = torch.zeros(tmp_mask.shape[0], 3, device = 'cuda')
        rendered_mask_pkg = gaussian_renderer.render_mask(view, scene_gaussians, pipeline.extract(args), background, precomputed_mask=tmp_mask)

        gt_score = images_scores[image_idx]

        tmp_target_mask = torch.nn.functional.interpolate(gt_score.unsqueeze(0).unsqueeze(0).float(), size=rendered_mask_pkg['mask'].shape[-2:] , mode='bilinear').squeeze(0)

        loss = -(tmp_target_mask * rendered_mask_pkg['mask']).sum()

        loss.backward()
        grad_score = tmp_mask.grad.clone()

        tmp_mask.grad.detach_()
        tmp_mask.grad.zero_()
        with torch.no_grad():
            tmp_mask = tmp_mask - grad_score

        tmp_mask.requires_grad = True

    with torch.no_grad():
        tmp_mask[tmp_mask <= 0] = 0
        tmp_mask[tmp_mask != 0] = 1
    if save_path is not None:
        torch.save(tmp_mask.bool(), save_path)
    # torch.save(tmp_mask.bool(), './segmentation_res/final_mask.pt')
    # final_mask = tmp_mask
    return tmp_mask


def load_multi_lvl_features_and_masks(image_path = './data/3dovs/bed/images/', feature_path = './data/3dovs/bed/language_features/'):

    images_features = [[],[],[],[]]
    images_masks = [[],[],[],[]]

    for image_name in sorted(os.listdir(image_path)):
        name = image_name.split('.')[0]
        feature_name = name + '_f.npy'
        mask_name = name + '_s.npy'

        tmp_f = np.load('./data/3dovs/bed/language_features/'+feature_name)
        tmp_s = np.load('./data/3dovs/bed/language_features/'+mask_name)

        all_features = 0
        for lvl, mask in enumerate(tmp_s):
            idxes = list(np.unique(mask))
            try:
                idxes.remove(-1)
            except:
                pass
            num_masks_in_lvl = len(idxes)
            images_features[lvl].append(tmp_f[all_features:all_features+num_masks_in_lvl])
            all_features += num_masks_in_lvl

            this_lvl_masks = []
            for idx in sorted(idxes):
                idx = int(idx)
                this_lvl_masks.append((mask == idx).astype(np.float32))
            images_masks[lvl].append(this_lvl_masks)


def get_multi_lvl_scores(clip_model, images_features, images_masks, prompt):
    with torch.no_grad():
        clip_model.set_positives([prompt])
    images_scores = [[],[],[],[]]
    for lvl, fs_in_lvl in tqdm(enumerate(images_features)):
        if lvl == 0:
            continue
        for i,f in enumerate(fs_in_lvl):

            with torch.no_grad():
                relevancy_score = clip_model.get_relevancy(torch.from_numpy(f).cuda(), 0)
                r_score = relevancy_score[:,0]

                stacked_image_mask = torch.from_numpy(np.stack(images_masks[lvl][i])).cuda()
                image_score = (r_score[:,None, None] * stacked_image_mask).sum(dim = 0)
                
            images_scores[lvl].append(image_score)
    final_images_scores = []
    for i in range(len(images_scores[1])):
        final_images_scores.append(torch.stack([images_scores[1][i], images_scores[2][i], images_scores[3][i]], dim = 0))
    return final_images_scores