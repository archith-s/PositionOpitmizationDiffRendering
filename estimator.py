"""
estimator.py — Estimator for SurgRIPE's run.py, backed by the same
differentiable-rendering silhouette optimizer as optimize_model_real_data_IoU.py
/ EvalRealDiffRender.py (GT-perturbed init -> centroid -> MSE -> MSE+IoU,
sigma annealed coarse-to-fine). run.py only passes predict() an image path, so
the GT pose and mask paths are derived from it by the dataset's own directory
convention (root/image, root/pose, root/mask, root/config.yaml, root/joint.stl).
"""

import os, re, struct, tempfile
import numpy as np
import yaml
import torch
import torch.nn as nn
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer,
    BlendParams, SoftSilhouetteShader, TexturesVertex,
)

NUM_ITERS_TRANS = 200
LR_TRANS = 0.05

NUM_ITERS_TRANS_S1B = 300
LR_TRANS_S1B = 0.05
SIGMA_STAGE1B = 0.025

NUM_ITERS_COARSE = 300
LR_ROT_COARSE = 0.003
SIGMA_STAGE2A = 0.025

NUM_ITERS_MID = 150
LR_ROT_MID = 0.002
SIGMA_STAGE2B = 0.010
IOU_WEIGHT_MID = 10.0

NUM_ITERS_JOINT = 300
LR_ROT = 0.001
SIGMA_STAGE2 = 0.005

IOU_WEIGHT = 20.0
IOU_WEIGHT_COARSE = 5.0
SIGMA_STAGE1 = 0.01
RENDER_SIZE = 256

INIT_ROT_NOISE_DEG = 10.0
INIT_TRANS_NOISE_MM = 10.0


def stl_to_obj(stl_path, obj_path):
    with open(stl_path, 'rb') as f:
        f.read(80)
        struct.unpack('<I', f.read(4))
        raw = f.read()
    n_tris = len(raw) // 50
    vert_map, verts, faces = {}, [], []
    for i in range(n_tris):
        offset = i * 50
        v0 = struct.unpack_from('<fff', raw, offset + 12)
        v1 = struct.unpack_from('<fff', raw, offset + 24)
        v2 = struct.unpack_from('<fff', raw, offset + 36)
        face = []
        for v in (v0, v1, v2):
            if v not in vert_map:
                vert_map[v] = len(verts) + 1
                verts.append(v)
            face.append(vert_map[v])
        faces.append(face)
    with open(obj_path, 'w') as fh:
        for x, y, z in verts:
            fh.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        for a, b, c in faces:
            fh.write(f'f {a} {b} {c}\n')


def load_obj_no_mtl(path):
    with open(path) as fh:
        lines = fh.readlines()
    clean = [l for l in lines if not re.match(r"\s*(mtllib|usemtl)\b", l)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp.writelines(clean)
        tmp_path = tmp.name
    verts, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    return verts, faces_idx.verts_idx


def rot6d_to_matrix(r6d):
    a1, a2 = r6d[:3], r6d[3:]
    b1 = nn.functional.normalize(a1, dim=0)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum() * b1, dim=0)
    return torch.stack([b1, b2, torch.linalg.cross(b1, b2)], dim=1)


def matrix_to_rot6d(R, device):
    return torch.tensor(R, dtype=torch.float32, device=device)[:, :2].T.reshape(6)


FLIP = np.diag([-1., -1., 1.])


def bop_to_pt3d(R_bop, t_bop):
    return (R_bop.T @ FLIP).astype(np.float32), (FLIP @ t_bop).astype(np.float32)


def pt3d_to_bop(R_pt3d, T_pt3d):
    return (FLIP @ R_pt3d.T).astype(np.float64), (FLIP @ T_pt3d).astype(np.float64)


def make_blurred_mask(mask_np, sigma_ndc, render_size):
    blur_ndc = float(np.log(1.0 / sigma_ndc - 1.0)) * sigma_ndc
    sigma_px = blur_ndc * (render_size / 2.0)
    if sigma_px < 0.5:
        return mask_np.astype(np.float32)
    return gaussian_filter(mask_np.astype(np.float32), sigma=sigma_px).clip(0.0, 1.0)


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3)
    ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(
        ax * np.deg2rad(np.random.uniform(-rot_noise_deg, rot_noise_deg))
    ).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


class PoseModel(nn.Module):
    def __init__(self, mesh, renderer, mask_resized_np, init_R, init_t,
                 fx, fy, cx, cy, H, W, device, grid_x, grid_y):
        super().__init__()
        self.mesh = mesh
        self.renderer = renderer
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.H, self.W = H, W
        self.device = device
        self.grid_x, self.grid_y = grid_x, grid_y
        self.cent_views = []
        self.cam_views = []

        fg_mask = torch.tensor(mask_resized_np, dtype=torch.float32, device=device)
        self.register_buffer("image_ref", fg_mask)
        ra = fg_mask.sum().clamp(min=1.0)
        self.register_buffer("cx_ref", ((grid_x * fg_mask).sum() / ra).unsqueeze(0))
        self.register_buffer("cy_ref", ((grid_y * fg_mask).sum() / ra).unsqueeze(0))
        self.register_buffer("area_ref", ra.unsqueeze(0))

        self.rot6d = nn.Parameter(matrix_to_rot6d(init_R, device))
        self.transl = nn.Parameter(torch.tensor(init_t, dtype=torch.float32, device=device))
        self.use_centroid = True
        self.use_iou = False
        self.iou_weight = IOU_WEIGHT
        self.I3 = torch.eye(3, dtype=torch.float32, device=device)
        self.Z3 = torch.zeros(3, dtype=torch.float32, device=device)

    def _render_alpha(self, R_rel, T_rel):
        R = rot6d_to_matrix(self.rot6d).unsqueeze(0)
        T = self.transl.unsqueeze(0)
        R_i = torch.mm(R[0], R_rel).unsqueeze(0)
        T_i = torch.mm(T, R_rel) + T_rel.unsqueeze(0)
        cam = PerspectiveCameras(
            focal_length=((self.fx, self.fy),), principal_point=((self.cx, self.cy),),
            R=R_i, T=T_i, in_ndc=False, image_size=((self.H, self.W),), device=self.device,
        )
        return self.renderer(meshes_world=self.mesh.clone(), cameras=cam)[0, ..., 3]

    def forward(self):
        if self.use_centroid:
            if self.cent_views:
                total = torch.tensor(0.0, device=self.device)
                for cx_r, cy_r, ar_r, R_rel, T_rel in self.cent_views:
                    alpha = self._render_alpha(R_rel, T_rel)
                    area = alpha.sum().clamp(min=1e-6)
                    cx = (self.grid_x * alpha).sum() / area
                    cy = (self.grid_y * alpha).sum() / area
                    total = total + (cx - cx_r) ** 2 + (cy - cy_r) ** 2
                return total / len(self.cent_views)
            alpha = self._render_alpha(self.I3, self.Z3)
            area = alpha.sum().clamp(min=1e-6)
            cx = (self.grid_x * alpha).sum() / area
            cy = (self.grid_y * alpha).sum() / area
            return (cx - self.cx_ref[0]) ** 2 + (cy - self.cy_ref[0]) ** 2

        views = self.cam_views or [(self.image_ref, self.I3, self.Z3)]
        total = torch.tensor(0.0, device=self.device)
        for ref_mask_i, R_rel_i, T_rel_i in views:
            alpha_i = self._render_alpha(R_rel_i, T_rel_i)
            mse_i = (alpha_i - ref_mask_i).pow(2).sum()
            if self.use_iou:
                inter = (alpha_i * ref_mask_i).sum()
                union = (alpha_i + ref_mask_i - alpha_i * ref_mask_i).sum().clamp(min=1e-6)
                total = total + mse_i + self.iou_weight * (1.0 - inter / union)
            else:
                total = total + mse_i
        return total / len(views)


class Estimator:

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=self.device)
        self.grid_y, self.grid_x = torch.meshgrid(arange, arange, indexing="ij")
        self.root = None

    def _load_dataset(self, root_path):
        if self.root == root_path:
            return
        self.root = root_path

        obj_cache = os.path.join(tempfile.gettempdir(), "joint_lnd_converted_estimator.obj")
        if not os.path.exists(obj_cache):
            stl_to_obj(os.path.join(root_path, "joint.stl"), obj_cache)
        verts, faces = load_obj_no_mtl(obj_cache)
        self.faces_doubled = torch.cat([faces, faces[:, [2, 1, 0]]], dim=0)
        verts_np = np.from_dlpack(verts.detach().cpu())
        if float(np.abs(verts_np).max()) < 1.0:
            verts = verts * 1000.0
        self.verts = verts

        with open(os.path.join(root_path, "config.yaml")) as f:
            cfg = yaml.safe_load(f)
        self.cam_K = cfg["cam"]["camera_matrix"]["data"]

    def _build_mesh(self):
        verts_rgb = torch.ones_like(self.verts)[None]
        return Meshes(verts=[self.verts.to(self.device)], faces=[self.faces_doubled.to(self.device)],
                      textures=TexturesVertex(verts_features=verts_rgb.to(self.device)))

    def _make_cameras(self, R, T, fx, fy, cx, cy, H, W):
        return PerspectiveCameras(
            focal_length=((fx, fy),), principal_point=((cx, cy),),
            R=R, T=T, in_ndc=False, image_size=((H, W),), device=self.device,
        )

    def _make_silhouette_renderer(self, fx, fy, cx, cy, H, W, sigma, faces_per_pixel=1):
        blend = BlendParams(sigma=sigma, gamma=sigma)
        blur = float(np.log(1.0 / sigma - 1.0)) * sigma
        raster = RasterizationSettings(image_size=(H, W), blur_radius=blur,
                                       faces_per_pixel=faces_per_pixel, bin_size=0)
        cam = self._make_cameras(torch.eye(3, device=self.device).unsqueeze(0),
                                 torch.zeros(1, 3, device=self.device),
                                 fx, fy, cx, cy, H, W)
        return MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
                            shader=SoftSilhouetteShader(blend_params=blend))

    def predict(self, image_path):
        root_path = os.path.dirname(os.path.dirname(image_path))
        self._load_dataset(root_path)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        mask_path = os.path.join(root_path, "mask", basename + ".png")
        pose_path = os.path.join(root_path, "pose", basename + ".npy")

        pose_mat = np.load(pose_path)
        R_gt = pose_mat[:3, :3].astype(np.float64)
        t_gt = pose_mat[:3, 3].astype(np.float64)

        img_W, img_H = Image.open(image_path).size
        mask_full = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        mask_resized = np.array(
            Image.fromarray((mask_full * 255).astype(np.uint8)).resize(
                (RENDER_SIZE, RENDER_SIZE), Image.NEAREST), dtype=np.float32) / 255.0
        mask_t = torch.tensor(mask_resized, dtype=torch.float32, device=self.device)

        sx, sy = RENDER_SIZE / img_W, RENDER_SIZE / img_H
        fx_r, fy_r = self.cam_K[0] * sx, self.cam_K[4] * sy
        cx_r, cy_r = self.cam_K[2] * sx, self.cam_K[5] * sy

        mesh = self._build_mesh()
        sil_s1  = self._make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, SIGMA_STAGE1)
        sil_s2a = self._make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, SIGMA_STAGE2A)
        sil_s2b = self._make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, SIGMA_STAGE2B)
        sil_s2c = self._make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, SIGMA_STAGE2)

        area_c = mask_t.sum().clamp(min=1.0)
        cx_c = (self.grid_x * mask_t).sum() / area_c
        cy_c = (self.grid_y * mask_t).sum() / area_c
        I3 = torch.eye(3, dtype=torch.float32, device=self.device)
        Z3 = torch.zeros(3, dtype=torch.float32, device=self.device)
        cent_views_s1 = [(cx_c, cy_c, area_c, I3, Z3)]

        def bm(sigma):
            arr = make_blurred_mask(mask_resized, sigma, RENDER_SIZE)
            return torch.tensor(arr, dtype=torch.float32, device=self.device)

        cam_views_s2a = [(bm(SIGMA_STAGE2A), I3, Z3)]
        cam_views_s2b = [(bm(SIGMA_STAGE2B), I3, Z3)]
        cam_views_s2c = [(bm(SIGMA_STAGE2), I3, Z3)]

        init_R, init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
        init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
        model = PoseModel(mesh, sil_s1, mask_resized, init_R_pt3d, init_t_pt3d,
                          fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
                          self.device, self.grid_x, self.grid_y).to(self.device)

        # Stage 1: translation only, centroid
        model.cent_views = cent_views_s1
        opt = torch.optim.Adam([model.transl], lr=LR_TRANS)
        for _ in range(NUM_ITERS_TRANS):
            opt.zero_grad(); loss = model(); loss.backward(); opt.step()

        # Stage 1.5: translation only, MSE silhouette
        model.renderer, model.cam_views = sil_s2a, cam_views_s2a
        model.use_centroid, model.use_iou = False, False
        opt = torch.optim.Adam([model.transl], lr=LR_TRANS_S1B)
        for _ in range(NUM_ITERS_TRANS_S1B):
            opt.zero_grad(); loss = model(); loss.backward(); opt.step()

        # Stage 2a: rotation + light IoU (translation frozen)
        model.use_iou, model.iou_weight = True, IOU_WEIGHT_COARSE
        opt = torch.optim.Adam([{"params": [model.rot6d], "lr": LR_ROT_COARSE}])
        for _ in range(NUM_ITERS_COARSE):
            opt.zero_grad(); loss = model(); loss.backward(); opt.step()

        # Stage 2b: MSE + IoU, intermediate sigma
        model.renderer, model.cam_views, model.iou_weight = sil_s2b, cam_views_s2b, IOU_WEIGHT_MID
        opt = torch.optim.Adam([
            {"params": [model.rot6d], "lr": LR_ROT_MID},
            {"params": [model.transl], "lr": LR_TRANS * 0.05},
        ])
        for _ in range(NUM_ITERS_MID):
            opt.zero_grad(); loss = model(); loss.backward(); opt.step()

        # Stage 2c: MSE + IoU, finest sigma
        model.renderer, model.cam_views, model.iou_weight = sil_s2c, cam_views_s2c, IOU_WEIGHT
        opt = torch.optim.Adam([
            {"params": [model.rot6d], "lr": LR_ROT},
            {"params": [model.transl], "lr": LR_TRANS * 0.02},
        ])
        for _ in range(NUM_ITERS_JOINT):
            opt.zero_grad(); loss = model(); loss.backward(); opt.step()

        with torch.no_grad():
            R_est_t = rot6d_to_matrix(model.rot6d)
            T_est_t = model.transl
        R_est_pt3d = np.from_dlpack(R_est_t.detach().cpu())
        T_est_pt3d = np.from_dlpack(T_est_t.detach().cpu())
        R_est, T_est = pt3d_to_bop(R_est_pt3d, T_est_pt3d)

        print(f"[Estimator] {os.path.basename(image_path)}: loss={loss.item():.2f}")
        return np.hstack([R_est, T_est.reshape(3, 1)])
