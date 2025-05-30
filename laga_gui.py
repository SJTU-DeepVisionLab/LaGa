# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
# from sklearn.decomposition import PCA

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal

from scipy.spatial.transform import Rotation as R

# from cuml.cluster.hdbscan import HDBSCAN
from hdbscan import HDBSCAN

import clip_utils
from clip_utils import get_relevancy_cosine
from clip_utils.clip_utils import load_clip, OpenCLIPNetworkConfig

def bilateral_filter_with_color(points, scores, colors, K=16, neighbor_map = None, 
                                spatial_sigma=0.1, range_score_sigma=0.1, 
                                range_color_sigma=0.1):

    N, _ = points.shape

    assert neighbor_map is not None
    knn_idx = neighbor_map

    neighbor_points = points[knn_idx]       # (N, K, 3)
    neighbor_scores = scores[knn_idx]       # (N, K)
    neighbor_colors = colors[knn_idx]       # (N, K, 3)

    points_expanded = points.unsqueeze(1).expand(-1, K, -1)   # (N, K, 3)
    scores_expanded = scores.unsqueeze(1).expand(-1, K)       # (N, K)
    colors_expanded = colors.unsqueeze(1).expand(-1, K, -1)   # (N, K, 3)

    spatial_distance = torch.norm(neighbor_points - points_expanded, dim=2)  # (N, K)

    spatial_weight = torch.exp(- (spatial_distance ** 2) / (2 * spatial_sigma ** 2))  # (N, K)

    score_difference = scores_expanded - neighbor_scores  # (N, K)

    range_score_weight = torch.exp(- (score_difference ** 2) / (2 * range_score_sigma ** 2))  # (N, K)

    color_difference = torch.norm(neighbor_colors - colors_expanded, dim=2)  # (N, K)

    range_color_weight = torch.exp(- (color_difference ** 2) / (2 * range_color_sigma ** 2))  # (N, K)

    range_weight = range_score_weight * range_color_weight  # (N, K)

    weights = spatial_weight * range_weight  # (N, K)

    weights_normalized = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-8)  # (N, K)

    filtered_scores = torch.sum(weights_normalized * neighbor_scores, dim=1)  # (N,)

    return filtered_scores

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 0.8   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = True

    FEATURE_DIM = 32
    MODEL_PATH = './output/lerfovs-figurines-minimal' # 30000

    FEATURE_GAUSSIAN_ITERATION = 31000
    SCENE_GAUSSIAN_ITERATION = 30000

    # SCALE_GATE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/scale_gate.pt')

    FEATURE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/contrastive_feature_point_cloud.ply')
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')
    
    DESCRIPTOR_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/multi_lvl_cluster_features.pth')
    DESCRIPTOR_WEIGHTS_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/multi_lvl_cluster_feature_weights.pth')
    SEG_SCORE_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(FEATURE_GAUSSIAN_ITERATION)}/multi_lvl_seg_scores.pth')
    
    K_MAX = 20
    NUM_LVL = 3


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, descriptors, descriptor_weights, seg_scores, clip_model) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'descriptors': descriptors,
            'descriptor_weights': descriptor_weights,
            'seg_scores': seg_scores,
            'clip_model': clip_model,
        }

        self.cluster_point_colors = None
        self.label_to_color = np.random.rand(1000, 3)
        self.seg_score = None

        self.proj_mat = None

        self.load_model = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
        self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
        # self.engine['scale_gate'].load_state_dict(torch.load(self.opt.SCALE_GATE_PATH))
        # self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        # self.clear_edit = False                 # clear all the click prompts
        self.roll_back = False
        self.preview = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)
        self.cluster_in_3D_flag = False

        self.render_mode_rgb = False
        self.render_mode_similarity = False
        self.render_mode_cluster = False
        
        self.gaussian_colors, self.neighbor_map = None, None # post-processing

        self.save_flag = False
        
        self.cosine_filter, self.postprocess = True, True
        self.do_infer = False
        self.relevance, self.cosines, self.filtered_relevance = None, None, None
        self.selected_layer = 0
        
        self.mask_3d = None
        
    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)
    
    def grayscale_to_colormap(self, gray):
        """Convert a grayscale value to Jet colormap RGB values."""
        # Ensure the grayscale values are in the range [0, 1]
        # gray = np.clip(gray, 0, 1)

        # Jet colormap ranges (these are normalized to [0, 1])
        jet_colormap = np.array([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 0.5],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0, 0],
            [0.5, 0, 0]
        ])

        # Corresponding positions for the colors in the colormap
        positions = np.linspace(0, 1, jet_colormap.shape[0])

        # Interpolate the RGB values based on the grayscale value
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])

        return np.stack((r, g, b), axis=-1)

    def register_dpg(self):
        
        with dpg.font_registry():
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # ← 改成实际存在的路径

            big_font = dpg.add_font(font_path, 36, tag="__big_font")
        dpg.bind_font("__big_font")

        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width * 1.5, height=self.window_height):
            dpg.add_image("_texture")   # add the texture

        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        # def clickmode_callback(sender):
        #     self.clickmode_button = 1 - self.clickmode_button
        # def clickmode_multi_callback(sender):
        #     self.clickmode_multi_button = dpg.get_value(sender)
        #     print("clickmode_multi_button = ", self.clickmode_multi_button)
            # print("binary_threshold_button = ", self.binary_threshold_button)
        # def clear_edit():
        #     self.clear_edit = True
        def roll_back():
            self.roll_back = True
        def do_infer():
            self.do_infer = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_reload():
            self.reload_flag = True

        def callback_reshuffle_color():
            self.label_to_color = np.random.rand(1000, 3)
            # try:
            #     self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
            #     self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)
            # except:
            #     pass

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = not self.render_mode_rgb
        def render_mode_similarity_callback(sender):
            self.render_mode_similarity = not self.render_mode_similarity
        def render_mode_cluster_callback(sender):
            self.render_mode_cluster = not self.render_mode_cluster
        
        def layer_select_callback(sender, app_data):
            self.selected_layer = int(app_data[-1])
            
        def preview_callback(sender):
            self.preview = not self.preview
            
        def cosine_filter_callback(sender):
            self.cosine_filter = not self.cosine_filter
            
        def postprocess_callback(sender):
            self.postprocess = not self.postprocess
            
        # control window
        with dpg.window(label="Control", tag="_control_window", width=int(self.window_width*0.3), height=self.window_height, pos=[self.window_width+10, 0]):

            dpg.add_text("Mouse position: click anywhere to start. ", tag="pos_item")
            dpg.add_text(" ", tag="NONE")
            
            
            dpg.add_text("Text Prompt:", tag="NONE2")
            dpg.add_input_text(label="", default_value="", tag="prompt")
            dpg.add_checkbox(label="Postprocess", callback=postprocess_callback, user_data="Some Data", default_value=True)
            dpg.add_button(label="Do Inference", callback=do_infer, user_data="Some Data")
            
            dpg.add_text(" ", tag="NONE3")
            
            
            dpg.add_text("ScoreThres:", tag="NONE4")
            dpg.add_slider_float(label="", default_value=0.6,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")
            
            dpg.add_checkbox(label="Cosine Filter", callback=cosine_filter_callback, user_data="Some Data", default_value=True)

            dpg.add_text("\nRender option: ", tag="render")
            dpg.add_checkbox(label="RGB", callback=render_mode_rgb_callback, user_data="Some Data")

            dpg.add_checkbox(label="RELEVANCE", callback=render_mode_similarity_callback, user_data="Some Data")
            dpg.add_checkbox(label="DECOMPOSITION", callback=render_mode_cluster_callback, user_data="Some Data")
            with dpg.group(indent=50):
                dpg.add_radio_button(
                        items=[f"Layer {i}" for i in range(self.opt.NUM_LVL)],
                        default_value="Layer 0",
                        callback=layer_select_callback,
                        tag="radio_selector"
                    )
            dpg.add_button(label="reshuffle_cluster_color", callback=callback_reshuffle_color, user_data="Some Data")
            
            # dpg.add_text("\nSegment option: ", tag="seg")
            # dpg.add_checkbox(label="clickmode", callback=clickmode_callback, user_data="Some Data")
            # dpg.add_checkbox(label="multi-clickmode", callback=clickmode_multi_callback, user_data="Some Data")
            dpg.add_checkbox(label="preview_segmentation_in_2d", callback=preview_callback, user_data="Some Data")
            
            dpg.add_text("\n")
            dpg.add_button(label="segment3d", callback=callback_segment3d, user_data="Some Data")
            dpg.add_button(label="roll_back", callback=roll_back, user_data="Some Data")
            # dpg.add_button(label="clear", callback=clear_edit, user_data="Some Data")
            dpg.add_button(label="save as", callback=callback_save, user_data="Some Data")
            dpg.add_input_text(label="", default_value="precomputed_mask", tag="save_name")
            dpg.add_text("\n")

            # dpg.add_button(label="cluster3d", callback=callback_cluster, user_data="Some Data")
            # dpg.add_button(label="reshuffle_cluster_color", callback=callback_reshuffle_color, user_data="Some Data")
            dpg.add_button(label="reload_data", callback=callback_reload, user_data="Some Data")

            def callback(sender, app_data, user_data):
                self.load_model = False
                file_data = app_data["selections"]
                file_names = []
                for key in file_data.keys():
                    file_names.append(key)

                self.opt.ply_file = file_data[file_names[0]]

                # if not self.load_model:
                print("loading model file...")
                self.engine.load_ply(self.opt.ply_file)
                # self.do_pca()   # calculate new self.proj_mat after loading new .ply file
                print("loading model file done.")
                self.load_model = True

        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def toggle_moving_left():
            self.moving = not self.moving

        def toggle_moving_middle():
            self.moving_middle = not self.moving_middle

        def move_handler(sender, pos, user):
            if self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def change_pos(sender, app_data):
            # if not dpg.is_item_focused("_primary_window"):
            #     return
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            if self.clickmode_button and app_data == 1:     # in the click mode and right click
                print(xy)
                self.new_click_xy = np.array(xy)
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda:toggle_moving_left())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda:toggle_moving_middle())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            
            dpg.add_mouse_click_handler(callback=change_pos)
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=int(self.window_width * 1.35), height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam

    @torch.no_grad()
    def fetch_data(self, view_camera):
        
        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)

        self.rendered_cluster = None
        
        if self.cluster_point_colors is not None and self.engine['scene']._xyz.shape[0] == self.cluster_point_colors.shape[0]:
            self.rendered_cluster = render(view_camera, self.engine['scene'], self.opt, self.bg_color, override_color=torch.from_numpy(self.cluster_point_colors).cuda().float())["render"].permute(1, 2, 0)

        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0)  #

        rgb_score = img.clone()
        # depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        # sems = feature_outputs["render"].permute(1, 2, 0)
        # H, W, C = sems.shape
        # sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        # sem_transed = sems @ self.proj_mat
        # sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        # scale = dpg.get_value('_Scale')
        # self.gates = self.engine['scale_gate'](torch.tensor([scale]).cuda())
        # scale_gated_feat = sems * self.gates.unsqueeze(0).unsqueeze(0)
        # scale_gated_feat = torch.nn.functional.normalize(scale_gated_feat, dim = -1, p = 2)
        
        # if self.clear_edit:
        #     # self.new_click_xy = []
        #     self.clear_edit = False
        #     self.prompt_num = 0
        #     try:
        #         self.engine['scene'].clear_segment()
        #         self.engine['feature'].clear_segment()
        #     except:
        #         pass

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            # try:
            self.engine['scene'].roll_back()
            # self.engine['feature'].roll_back()
            # except:
                # pass
        
        if self.reload_flag:
            self.reload_flag = False
            print("loading model file...")
            self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
            self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
            self.load_model = True

        if self.do_infer:
            self.do_infer = False
            self.relevance, self.cosines = self.do_inference(dpg.get_value('prompt'), postprocess=self.postprocess)
            
        if self.cosine_filter:
            if self.relevance is not None and self.cosines is not None:
                self.filtered_relevance = self.relevance.copy()
                self.filtered_relevance[(self.cosines < 0.23).squeeze().cpu().numpy(), :] = 0
        
        score_map = None

        if self.segment3d_flag:
            self.segment3d_flag = False
            scores_3d = None
            
            if self.cosine_filter and self.filtered_relevance is not None:
                scores_3d = self.filtered_relevance
            elif self.relevance is not None:
                scores_3d = self.relevance
            else:
                pass
            
            if scores_3d is not None:
                self.mask_3d = torch.from_numpy(scores_3d[:,0] > dpg.get_value('_ScoreThres')).cuda()
                self.engine['scene'].roll_back()
                # self.engine['feature'].roll_back()
                self.engine['scene'].segment(self.mask_3d)
                # self.engine['feature'].segment(self.mask_3d)

        self.render_buffer = None
        render_num = 0
        # if preview, we must render the rgb
        if self.render_mode_rgb or (not self.render_mode_cluster and not self.render_mode_similarity) or self.preview:
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1
        
        if self.render_mode_cluster:
            
            selected_layer = self.selected_layer
            
            seg_score = self.engine['seg_scores'][selected_layer]
            
            self.cluster_point_colors = self.label_to_color[seg_score.argmax(dim = -1).cpu().numpy()]
            
            if self.rendered_cluster is None:
                self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)
            else:
                self.render_buffer = self.rendered_cluster.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + self.rendered_cluster.cpu().numpy().reshape(-1)
            
            render_num += 1
            
        if not self.preview and self.render_mode_similarity:
            scores_3d = None
            
            if self.cosine_filter and self.filtered_relevance is not None:
                scores_3d = self.filtered_relevance
            elif self.relevance is not None:
                scores_3d = self.relevance
            
            if scores_3d is not None:
                with torch.no_grad():
                    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
                    self.engine['scene'].roll_back()
                    
                    # scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)
                    
                    render_res = render(view_camera, self.engine['scene'], self.opt, background, override_color=torch.from_numpy(scores_3d).float().cuda(), override_mask=torch.from_numpy(scores_3d).float().cuda()[:,0:1])['render']
                    
                    render_res = render_res.permute(1, 2, 0)  # (H, W, C)
                    
                    score_map = render_res[:, :, 0]  # (H, W)

                if score_map is not None:
                    self.render_buffer = self.grayscale_to_colormap(score_map.cpu().numpy()).reshape(-1).astype(np.float32) if self.render_buffer is None else self.render_buffer + self.grayscale_to_colormap(score_map.cpu().numpy()).reshape(-1).astype(np.float32)
                else:
                    self.render_buffer = rgb_score.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + rgb_score.cpu().numpy().reshape(-1)

                render_num += 1
            
            
        if self.preview:
            scores_3d = None
            if self.cosine_filter and self.filtered_relevance is not None:
                scores_3d = self.filtered_relevance
            elif self.relevance is not None:
                scores_3d = self.relevance
            
            if scores_3d is not None:
                
                with torch.no_grad():
                    background = torch.tensor([0,0,0], dtype=torch.float32, device="cuda")
                    self.engine['scene'].roll_back()
                    
                    # scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color)
                    
                    render_res = render(view_camera, self.engine['scene'], self.opt, background, override_color=torch.from_numpy(scores_3d).float().cuda(), override_mask=torch.from_numpy(scores_3d).float().cuda()[:,0:1])['render']
                    
                    render_res = render_res.permute(1, 2, 0)  # (H, W, C)
                    
                    score_map = render_res  # (H, W, 3)
                
                self.render_buffer = self.render_buffer * (score_map > 0.2).cpu().numpy().reshape(-1).astype(np.float32)
                render_num = 1
                
        self.render_buffer /= render_num

        dpg.set_value("_texture", self.render_buffer)
        
        if self.save_flag:
            print("Saving ...")
            self.save_flag = False

            if self.mask_3d is None:
                with dpg.window(label="Tips"):
                    dpg.add_text('You should segment the 3D object before save it (click segment3d first).')
            else:
                os.makedirs("./segmentation_res", exist_ok=True)
                torch.save(self.mask_3d, f"./segmentation_res/{dpg.get_value('save_name')}.pt")

    @torch.no_grad()
    def do_inference(self, prompt, postprocess=True):
        # Not all scene can apply this. Some scenes are too large.
        self.engine['scene'].roll_back()
        if postprocess:
            if self.gaussian_colors is None:
                from utils.sh_utils import SH2RGB
                
                gaussian_colors = self.engine['scene']._features_dc
                gaussian_colors = SH2RGB(gaussian_colors.squeeze())
                self.gaussian_colors = torch.clip(gaussian_colors, 0, 1)

            if self.neighbor_map is None:
                K = 16
                from pytorch3d.ops import knn_points
                points = self.engine['scene'].get_xyz
                knn = knn_points(points[None, ...], points[None, ...], K=K, return_nn=False)
                self.neighbor_map = knn.idx[0]  # (N, K)

        
        NUM_LVL = self.opt.NUM_LVL
        num_per_cluster_features = self.opt.K_MAX
        multi_lvl_cluster_features, multi_lvl_cluster_feature_weights, multi_lvl_seg_scores = self.engine['descriptors'], self.engine['descriptor_weights'], self.engine['seg_scores']
        point_colors = None
        stack_of_cosine = []
        multi_lvl_cluster_scores = []
        for lvl in range(0,NUM_LVL):

            cluster_features = multi_lvl_cluster_features[lvl]
            cluster_weights = multi_lvl_cluster_feature_weights[lvl].clone()
            seg_score = multi_lvl_seg_scores[lvl]
            rel, pos, neg = get_relevancy_cosine(clip_model, torch.nn.functional.normalize(cluster_features.cuda(), dim = -1, p = 2), prompt)
            cluster_scores = (rel * cluster_weights).reshape([-1, num_per_cluster_features])

            cluster_scores, index = cluster_scores.max(dim = 1)[0], cluster_scores.max(dim = 1)[1]

            multi_lvl_cluster_scores.append(cluster_scores)

            pos = pos.reshape([-1, num_per_cluster_features])
            batch_indices = torch.arange(pos.shape[0]).to(pos.device)  # [batch_size]

            selected_pos = pos[batch_indices, index]  # [batch_size]


            stack_of_cosine.append(selected_pos[seg_score.argmax(dim = -1).cpu().numpy()])


        for lvl in [0,1,2]:
            seg_score = multi_lvl_seg_scores[lvl]
            cluster_scores = multi_lvl_cluster_scores[lvl]

            cluster_colors = np.array(cluster_scores.cpu())

            cluster_colors[cluster_colors < 0] = 0

            cluster_colors = np.expand_dims(cluster_colors, axis=1)
            cur_lvl_point_colors = cluster_colors[seg_score.argmax(dim = -1).cpu().numpy()]
            point_colors = cur_lvl_point_colors if point_colors is None else point_colors + cur_lvl_point_colors
        
        stack_of_cosine = torch.stack(stack_of_cosine, 0)
        stack_of_cosine = stack_of_cosine.max(dim = 0)[0]

        point_colors /= NUM_LVL
        # remove too low scores before min-max normalization for stability
        point_colors = np.clip(point_colors, np.quantile(point_colors, 0.25), 1e9)
        point_colors = point_colors - point_colors.min()
        point_colors = point_colors / point_colors.max()
        # point_colors[(stack_of_cosine < 0.23).cpu().numpy()] = 0

        # point_colors[point_colors < 0.6] = 0
        # point_colors[point_colors != 0] = 1

        if not postprocess:
            point_colors = point_colors.repeat(3, axis=1)
        else:
            point_colors = bilateral_filter_with_color(self.engine['scene'].get_xyz, torch.from_numpy(point_colors).squeeze().cuda(), self.gaussian_colors, spatial_sigma=0.5, range_score_sigma = 500, range_color_sigma = 500, neighbor_map=self.neighbor_map)
            point_colors = point_colors.cpu().unsqueeze(-1).numpy().repeat(3, axis=1)

        return point_colors, stack_of_cosine

if __name__ == "__main__":
    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/lerfovs-figurines-minimal")
    parser.add_argument('-f', '--feature_iteration', type=int, default=31000)
    parser.add_argument('-s', '--scene_iteration', type=int, default=30000)
    parser.add_argument('-k', '--k_max', type=int, default=20)
    parser.add_argument('-c', '--clip_path', type=str, default="./clip_ckpt/ViT-B-16-laion2b_s34b_b88k.bin")
    
    args = parser.parse_args()

    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration

    opt.DESCRIPTOR_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/multi_lvl_cluster_features.pth')
    opt.DESCRIPTOR_WEIGHTS_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/multi_lvl_cluster_feature_weights.pth')
    opt.SEG_SCORE_PATH = os.path.join(opt.MODEL_PATH, f'point_cloud/iteration_{str(opt.FEATURE_GAUSSIAN_ITERATION)}/multi_lvl_seg_scores.pth')
    
    opt.CLIP_PATH = args.clip_path
    
    opt.K_MAX = args.k_max

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.FEATURE_DIM)
    
    descriptors = torch.load(opt.DESCRIPTOR_PATH)
    descriptor_weights = torch.load(opt.DESCRIPTOR_WEIGHTS_PATH)
    seg_scores = torch.load(opt.SEG_SCORE_PATH)
    
    print("Loading CLIP ...")
    clip_config = OpenCLIPNetworkConfig()
    clip_config.clip_model_pretrained = opt.CLIP_PATH

    clip_model = load_clip(clip_config)
    clip_model.eval()
    print("CLIP model loaded.")
    
    gui = GaussianSplattingGUI(opt, gs_model, feat_gs_model, descriptors, descriptor_weights, seg_scores, clip_model)

    gui.render()