"""
optimize_model_real_new_param.py  (single-view, Real_SR_Data/LND_TEST)
Based on optimize_model_real_data.py with tuned hyperparameters for lowest
possible error on real data (single-frame mode by default).

Key changes vs optimize_model_real_data.py
──────────────────────────────────────────
* IoU loss added from Stage 2a (was only from 2b) — more discriminative,
  robust to mesh/object mismatch and rendering halo.
* IoU computed vs BINARY mask (not blurred) — hard-IoU is the right metric.
* IoU weight raised to 50 (was 20) so IoU dominates over MSE halo penalty.
* faces_per_pixel=2 (was 1) — better coverage on hollow/shell mesh.
* 5-stage sigma schedule: 0.050 → 0.025 → 0.010 → 0.005 → 0.003
  (extra coarse stage + extra fine stage vs original 4 stages).
* Multi-restart: runs full pipeline NUM_RESTARTS times, keeps pose with best
  final IoU vs binary mask.
* Conservative rotation LRs to prevent the divergence seen in baseline.
* Default: first frame only (pass --all or --frames N for more).
"""

import os, re, csv, struct, tempfile
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)


def save_tensor_as_png(tensor: torch.Tensor, path: str):
    arr = np.from_dlpack(tensor.detach().cpu().clamp(0.0, 1.0))
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


# ─────────────────────────── STL → OBJ ────────────────────────────────────────
def stl_to_obj(stl_path: str, obj_path: str):
    with open(stl_path, 'rb') as f:
        f.read(80)
        n_tris_header = struct.unpack('<I', f.read(4))[0]
        raw = f.read()

    n_tris = len(raw) // 50
    if n_tris != n_tris_header:
        print(f"  STL header says {n_tris_header} tris but file contains {n_tris}")

    vert_map: dict = {}
    verts: list = []
    faces: list = []
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

    print(f"STL→OBJ: {len(verts)} verts, {len(faces)} faces → {obj_path}")


# ─────────────────────────── CONFIG ───────────────────────────────────────────
DATASET_ROOT = "../Real_SR_Data/LND_TEST"
MESH_STL     = os.path.join(DATASET_ROOT, "joint.stl")
MODEL_PTS    = os.path.join(DATASET_ROOT, "joint.npy")

OUTPUT_DIR = "../Diff_Render_CSV_Visuals/Real_Data_New_Params"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pose_estimation_results.csv")
VIZ_DIR    = OUTPUT_DIR
DEBUG_DIR  = "../debug_renders"

RENDER_SIZE     = 256
FACES_PER_PIXEL = 2       # was 1; better hollow-mesh silhouette coverage
FG_THRESHOLD    = 0.5

# Multi-restart: run full pipeline N times per frame, keep best final IoU
NUM_RESTARTS        = 3
INIT_ROT_NOISE_DEG  = 15.0  # slightly wider to explore more of rotation space
INIT_TRANS_NOISE_MM = 10.0

USE_GT_INITIALIZATION = True
USE_GT_INIT_ONLY      = False

# Stage 1: translation only (centroid)
NUM_ITERS_TRANS = 150     # fewer to avoid over-shooting
LR_TRANS        = 0.05

# Stage 1.5: translation, MSE vs blurred binary mask
NUM_ITERS_S1B   = 200
LR_S1B          = 0.03
SIGMA_S1B       = 0.050   # coarser than original 0.025 → gentler gradient

# Stage 2a: R+T, IoU + MSE, coarsest sigma
NUM_ITERS_2A  = 300
LR_ROT_2A     = 0.001     # was 0.003 — conservative to prevent divergence
LR_T_2A       = 0.005
SIGMA_2A      = 0.050
IOU_WEIGHT_2A = 50.0

# Stage 2b: R+T, IoU + MSE, medium-coarse sigma
NUM_ITERS_2B  = 200
LR_ROT_2B     = 0.0008
LR_T_2B       = 0.002
SIGMA_2B      = 0.025
IOU_WEIGHT_2B = 50.0

# Stage 2c: R+T, IoU + MSE, medium sigma
NUM_ITERS_2C  = 200
LR_ROT_2C     = 0.0005
LR_T_2C       = 0.001
SIGMA_2C      = 0.010
IOU_WEIGHT_2C = 60.0

# Stage 2d: R+T, IoU + MSE, fine sigma
NUM_ITERS_2D  = 250
LR_ROT_2D     = 0.0003
LR_T_2D       = 0.0005
SIGMA_2D      = 0.005
IOU_WEIGHT_2D = 70.0

# Stage 2e: R+T, IoU + MSE, finest sigma
NUM_ITERS_2E  = 200
LR_ROT_2E     = 0.0001
LR_T_2E       = 0.0002
SIGMA_2E      = 0.003
IOU_WEIGHT_2E = 80.0

SIGMA_STAGE1  = 0.01  # for centroid renderer
# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--all",    action="store_true", help="Run all frames")
parser.add_argument("--frames", type=int, default=None,
                    help="Run only the first N frames (default: 1)")
args = parser.parse_args()

os.makedirs(DEBUG_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Debug → {DEBUG_DIR}   Viz → {VIZ_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=device)
GRID_Y, GRID_X = torch.meshgrid(_arange, _arange, indexing="ij")


# ── Camera intrinsics from config.yaml ────────────────────────────────────────
with open(os.path.join(DATASET_ROOT, "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

cam_K = _cfg["cam"]["camera_matrix"]["data"]
print(f"Camera K: fx={cam_K[0]:.2f}  fy={cam_K[4]:.2f}  cx={cam_K[2]:.2f}  cy={cam_K[5]:.2f}")


# ── Build frame list ───────────────────────────────────────────────────────────
_pose_dir  = os.path.join(DATASET_ROOT, "pose")
_image_dir = os.path.join(DATASET_ROOT, "image")
_mask_dir  = os.path.join(DATASET_ROOT, "mask")

_pose_ids  = {int(fn.split('.')[0]) for fn in os.listdir(_pose_dir)  if fn.endswith('.npy')}
_image_ids = {int(fn.split('.')[0]) for fn in os.listdir(_image_dir) if fn.lower().endswith('.png')}
_mask_ids  = {int(fn.split('.')[0]) for fn in os.listdir(_mask_dir)  if fn.lower().endswith('.png')}

valid_frame_ids = sorted(_pose_ids & _image_ids & _mask_ids)
print(f"Valid frames: {len(valid_frame_ids)}")

if args.frames is not None:
    valid_frame_ids = valid_frame_ids[:args.frames]
    print(f"Running {len(valid_frame_ids)} frame(s) (--frames {args.frames})")
elif args.all:
    print(f"Running all {len(valid_frame_ids)} frames")
else:
    valid_frame_ids = valid_frame_ids[:1]
    print(f"Running first 1 frame (pass --all or --frames N for more)")


# ── Image dimensions ───────────────────────────────────────────────────────────
_sample_img = Image.open(os.path.join(_image_dir, f"{valid_frame_ids[0]}.png"))
img_W, img_H = _sample_img.size
print(f"Image size: {img_W}×{img_H}")


# ── Convert joint.stl → OBJ and load mesh ────────────────────────────────────
_obj_cache = os.path.join(tempfile.gettempdir(), "joint_lnd_converted.obj")
if not os.path.exists(_obj_cache):
    print("Converting joint.stl → OBJ (first run only) ...")
    stl_to_obj(MESH_STL, _obj_cache)
else:
    print(f"Reusing cached OBJ: {_obj_cache}")


def load_obj_no_mtl(path: str):
    with open(path) as fh:
        lines = fh.readlines()
    clean = [l for l in lines if not re.match(r"\s*(mtllib|usemtl)\b", l)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp.writelines(clean)
        tmp_path = tmp.name
    verts, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    return verts, faces_idx.verts_idx


verts, faces = load_obj_no_mtl(_obj_cache)
faces_doubled = torch.cat([faces, faces[:, [2, 1, 0]]], dim=0)
verts_np = np.from_dlpack(verts.detach().cpu())
_vmax = float(verts_np.__abs__().max())
print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces | max|v|={_vmax:.4f}")
if _vmax < 1.0:
    print("  Auto-scaling ×1000 (m→mm)")
    verts = verts * 1000.0
    verts_np = verts_np * 1000.0
else:
    print("  Vertices in mm.")


# ── Load model point cloud for ADD metric ─────────────────────────────────────
model_pts_np = np.load(MODEL_PTS).astype(np.float64)
_pts_vmax = float(np.abs(model_pts_np).max())
if _pts_vmax < 1.0:
    model_pts_np = model_pts_np * 1000.0
print(f"Model pts: {model_pts_np.shape[0]} points | max|v|={_pts_vmax:.4f}")


# ── Fallback translation ───────────────────────────────────────────────────────
_all_gt_t = [np.load(os.path.join(_pose_dir, f"{fid}.npy"))[:3, 3]
             for fid in valid_frame_ids]
mean_t_fallback = np.mean(_all_gt_t, axis=0)
print(f"Mean GT translation: {mean_t_fallback.round(2)}")


# ── Camera / renderer helpers ──────────────────────────────────────────────────
# Screen-space (in_ndc=False) cameras with an explicit image_size, NOT
# PyTorch3D's NDC convention -- NDC silently mishandles non-square aspect
# ratios, which produced a confirmed, measured pixel offset against a
# manual-rasterizer render of identical pose data (see optimize_model_real_data_IoU.py,
# sim_iter_images.py, fixed 2026-07-10).
def make_cameras(R, T, fx, fy, cx, cy, H, W) -> PerspectiveCameras:
    return PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        R=R, T=T, in_ndc=False, image_size=((H, W),), device=device,
    )


def scale_intrinsics(K_flat: list, W_from: int, H_from: int, W_to: int, H_to: int):
    """Rescale pixel-space intrinsics for rendering at a different resolution
    (the optimizer renders at RENDER_SIZE, not the native capture resolution)."""
    sx, sy = W_to / W_from, H_to / H_from
    return K_flat[0] * sx, K_flat[4] * sy, K_flat[2] * sx, K_flat[5] * sy


def make_silhouette_renderer(fx, fy, cx, cy, H, W,
                              sigma=1e-4, image_size=None,
                              faces_per_pixel=FACES_PER_PIXEL):
    if image_size is None:
        image_size = (H, W)
    blend  = BlendParams(sigma=sigma, gamma=sigma)
    blur   = float(np.log(1.0 / sigma - 1.0)) * sigma
    raster = RasterizationSettings(
        image_size=image_size, blur_radius=blur,
        faces_per_pixel=faces_per_pixel, bin_size=0,
    )
    cam = make_cameras(torch.eye(3, device=device).unsqueeze(0),
                       torch.zeros(1, 3, device=device),
                       fx, fy, cx, cy, H, W)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=blend),
    )


def build_mesh(solid: bool = False) -> Meshes:
    f = faces_doubled if solid else faces
    verts_rgb = torch.ones_like(verts)[None]
    return Meshes(verts=[verts.to(device)], faces=[f.to(device)],
                  textures=TexturesVertex(verts_features=verts_rgb.to(device)))


def make_blurred_mask(mask_np: np.ndarray, sigma_ndc: float) -> np.ndarray:
    """Gaussian-blur binary mask to match soft-renderer bandwidth for MSE reference."""
    blur_ndc = float(np.log(1.0 / sigma_ndc - 1.0)) * sigma_ndc
    sigma_px = blur_ndc * (RENDER_SIZE / 2.0)
    if sigma_px < 0.5:
        return mask_np.astype(np.float32)
    return gaussian_filter(mask_np.astype(np.float32), sigma=sigma_px).clip(0.0, 1.0)


# ── 6-D ↔ SO(3) ───────────────────────────────────────────────────────────────
def rot6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    a1, a2 = r6d[:3], r6d[3:]
    b1 = nn.functional.normalize(a1, dim=0)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum() * b1, dim=0)
    return torch.stack([b1, b2, torch.linalg.cross(b1, b2)], dim=1)


def matrix_to_rot6d(R: np.ndarray) -> torch.Tensor:
    return torch.tensor(R, dtype=torch.float32, device=device)[:, :2].T.reshape(6)


# ── BOP ↔ PyTorch3D ───────────────────────────────────────────────────────────
_FLIP = np.diag([-1., -1., 1.])


def bop_to_pt3d(R_bop, t_bop):
    return (R_bop.T @ _FLIP).astype(np.float32), (_FLIP @ t_bop).astype(np.float32)


def pt3d_to_bop(R_pt3d, T_pt3d):
    return (_FLIP @ R_pt3d.T).astype(np.float64), (_FLIP @ T_pt3d).astype(np.float64)


_I3 = torch.eye(3,  dtype=torch.float32, device=device)
_Z3 = torch.zeros(3, dtype=torch.float32, device=device)


def perturb_pose(R_gt, t_gt, rot_noise_deg=15.0, trans_noise_mm=10.0):
    ax = np.random.randn(3)
    ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(
        ax * np.deg2rad(np.random.uniform(-rot_noise_deg, rot_noise_deg))
    ).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Pose model ────────────────────────────────────────────────────────────────
class PoseModel(nn.Module):
    """
    Stage 1 (use_centroid=True): centroid loss from binary mask.
    Stage 2+ (use_centroid=False): MSE vs blurred reference + IoU vs binary mask.
      cam_views  = [(ref_mse, ref_iou, R_rel, T_rel), ...]
      ref_mse    = blurred binary mask (for MSE)
      ref_iou    = binary mask directly (for hard-IoU, more robust)
    """

    def __init__(self, mesh, renderer, mask_resized_np, init_R, init_t,
                 fx, fy, cx, cy, H, W):
        super().__init__()
        self.mesh = mesh
        self.renderer = renderer
        self.fx = fx;  self.fy = fy
        self.cx = cx;  self.cy = cy
        self.H  = H;   self.W  = W
        self.cent_views = []
        self.cam_views  = []   # list of (ref_mse, ref_iou, R_rel, T_rel)

        fg_mask = torch.tensor(mask_resized_np, dtype=torch.float32, device=device)
        self.register_buffer("image_ref", fg_mask)  # binary mask for centroid
        self.register_buffer("binary_mask", fg_mask.clone())

        ra = fg_mask.sum().clamp(min=1.0)
        self.register_buffer("cx_ref",   ((GRID_X * fg_mask).sum() / ra).unsqueeze(0))
        self.register_buffer("cy_ref",   ((GRID_Y * fg_mask).sum() / ra).unsqueeze(0))
        self.register_buffer("area_ref", ra.unsqueeze(0))

        self.rot6d  = nn.Parameter(matrix_to_rot6d(init_R))
        self.transl = nn.Parameter(torch.tensor(init_t, dtype=torch.float32, device=device))
        self.use_centroid = True
        self.use_iou      = False
        self.iou_weight   = 50.0

    def _render_alpha(self, R_rel, T_rel):
        R = rot6d_to_matrix(self.rot6d).unsqueeze(0)
        T = self.transl.unsqueeze(0)
        R_i = torch.mm(R[0], R_rel).unsqueeze(0)
        T_i = torch.mm(T, R_rel) + T_rel.unsqueeze(0)
        cam = make_cameras(R_i, T_i, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
        return self.renderer(meshes_world=self.mesh.clone(), cameras=cam)[0, ..., 3]

    def forward(self):
        if self.use_centroid:
            if self.cent_views:
                total = torch.tensor(0.0, device=device)
                for cx_r, cy_r, ar_r, R_rel, T_rel in self.cent_views:
                    alpha = self._render_alpha(R_rel, T_rel)
                    area  = alpha.sum().clamp(min=1e-6)
                    cx    = (GRID_X * alpha).sum() / area
                    cy    = (GRID_Y * alpha).sum() / area
                    total = total + (cx - cx_r)**2 + (cy - cy_r)**2
                loss = total / len(self.cent_views)
            else:
                alpha = self._render_alpha(_I3, _Z3)
                area  = alpha.sum().clamp(min=1e-6)
                cx    = (GRID_X * alpha).sum() / area
                cy    = (GRID_Y * alpha).sum() / area
                loss  = (cx - self.cx_ref[0])**2 + (cy - self.cy_ref[0])**2
        else:
            views = self.cam_views or [(self.binary_mask, self.binary_mask, _I3, _Z3)]
            total = torch.tensor(0.0, device=device)
            for ref_mse_i, ref_iou_i, R_rel_i, T_rel_i in views:
                alpha_i = self._render_alpha(R_rel_i, T_rel_i)
                mse_i   = (alpha_i - ref_mse_i).pow(2).sum()
                if self.use_iou:
                    inter = (alpha_i * ref_iou_i).sum()
                    union = (alpha_i + ref_iou_i - alpha_i * ref_iou_i).sum().clamp(min=1e-6)
                    total = total + mse_i + self.iou_weight * (1.0 - inter / union)
                else:
                    total = total + mse_i
            loss = total / len(views)

        return loss, rot6d_to_matrix(self.rot6d).unsqueeze(0), self.transl.unsqueeze(0)

    def final_iou(self, renderer, binary_mask_t: torch.Tensor) -> float:
        """Compute hard-ish IoU vs binary mask for restart comparison."""
        with torch.no_grad():
            alpha = self._render_alpha(_I3, _Z3)
            inter = (alpha * binary_mask_t).sum()
            union = (alpha + binary_mask_t - alpha * binary_mask_t).sum().clamp(min=1e-6)
        return float(inter / union)


# ── Error metrics ──────────────────────────────────────────────────────────────
def rotation_error_deg(R_est, R_gt):
    cos_a = float(np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def translation_error_mm(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


def add_metric(pts_np, R_est, t_est, R_gt, t_gt):
    return float(np.mean(np.linalg.norm(
        ((R_est @ pts_np.T).T + t_est) - ((R_gt @ pts_np.T).T + t_gt), axis=1)))


# ── Visualization renderer (full resolution, created once) ────────────────────
_viz_renderer = make_silhouette_renderer(cam_K[0], cam_K[4], cam_K[2], cam_K[5],
                                         img_H, img_W, sigma=0.005,
                                         faces_per_pixel=100)
_mesh_viz = build_mesh()


def draw_pose_overlay(rgb_pil, R_bop, t_bop, color_sil=(0, 220, 0)) -> Image.Image:
    W, H = rgb_pil.size
    R_pt3d, T_pt3d = bop_to_pt3d(np.array(R_bop), np.array(t_bop))
    R_t = torch.tensor(R_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    T_t = torch.tensor(T_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    cam = make_cameras(R_t, T_t, cam_K[0], cam_K[4], cam_K[2], cam_K[5], H, W)
    with torch.no_grad():
        alpha = np.from_dlpack(
            _viz_renderer(meshes_world=_mesh_viz.clone(), cameras=cam)[0, ..., 3].detach().cpu())

    overlay = np.zeros((*alpha.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    rgb_arr = np.array(rgb_pil, dtype=np.float32)
    blended = np.clip(0.5 * alpha[..., None] * overlay + rgb_arr, 0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)

    draw = ImageDraw.Draw(result)
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]

    def proj(p):
        pc = np.array(R_bop) @ np.array(p) + np.array(t_bop)
        return (int(round(fx * pc[0] / pc[2] + cx)),
                int(round(fy * pc[1] / pc[2] + cy))) if pc[2] > 0 else None

    L   = 50.0
    cen = proj([0, 0, 0])
    for end, col in zip([proj([L, 0, 0]), proj([0, L, 0]), proj([0, 0, L])],
                        [(220, 50, 50), (50, 220, 50), (50, 50, 220)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


def run_one_restart(frame_id, mesh, mask_resized, mask_t,
                    R_gt, t_gt, fx_r, fy_r, cx_r, cy_r, restart_idx):
    """Full 5-stage optimization pipeline for one restart. Returns (R_est_np, T_est_np, iou)."""

    sil_s1   = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE1)
    sil_s1b  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_S1B)
    sil_s2a  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_2A)
    sil_s2b  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_2B)
    sil_s2c  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_2C)
    sil_s2d  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_2D)
    sil_s2e  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_2E)

    # Blurred masks for MSE reference at each sigma
    def _bm(sigma):
        arr = make_blurred_mask(mask_resized, sigma)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    bm_s1b = _bm(SIGMA_S1B)
    bm_s2a = _bm(SIGMA_2A)
    bm_s2b = _bm(SIGMA_2B)
    bm_s2c = _bm(SIGMA_2C)
    bm_s2d = _bm(SIGMA_2D)
    bm_s2e = _bm(SIGMA_2E)
    # IoU reference: always the raw binary mask
    bin_ref = mask_t

    if USE_GT_INITIALIZATION:
        init_R, init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
    else:
        init_R = np.eye(3, dtype=np.float64)
        init_t = mean_t_fallback.copy()

    r_tag = f"r{restart_idx} f{frame_id:04d}"

    # Centroid
    area_c = mask_t.sum().clamp(min=1.0)
    cx_c   = (GRID_X * mask_t).sum() / area_c
    cy_c   = (GRID_Y * mask_t).sum() / area_c

    init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
    model = PoseModel(mesh, sil_s1, mask_resized,
                      init_R_pt3d, init_t_pt3d,
                      fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE).to(device)

    # ── Stage 1: centroid, translation only ────────────────────────────────────
    model.use_centroid = True
    model.cent_views   = [(cx_c, cy_c, area_c, _I3, _Z3)]
    opt = torch.optim.Adam([model.transl], lr=LR_TRANS)
    loop = tqdm(range(NUM_ITERS_TRANS), desc=f"  s1  {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s1  done: loss={loss.item():.4f}")

    # ── Stage 1.5: translation, MSE vs blurred mask ────────────────────────────
    model.renderer     = sil_s1b
    model.cam_views    = [(bm_s1b, bin_ref, _I3, _Z3)]
    model.use_centroid = False
    model.use_iou      = False
    opt = torch.optim.Adam([model.transl], lr=LR_S1B)
    loop = tqdm(range(NUM_ITERS_S1B), desc=f"  s1b {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s1b done: loss={loss.item():.4f}  (σ={SIGMA_S1B})")

    # ── Stage 2a: R+T, IoU + MSE, coarsest sigma ──────────────────────────────
    model.renderer   = sil_s2a
    model.cam_views  = [(bm_s2a, bin_ref, _I3, _Z3)]
    model.use_iou    = True
    model.iou_weight = IOU_WEIGHT_2A
    opt = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_2A},
        {"params": [model.transl], "lr": LR_T_2A},
    ])
    loop = tqdm(range(NUM_ITERS_2A), desc=f"  s2a {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s2a done: loss={loss.item():.4f}  (σ={SIGMA_2A})")

    # ── Stage 2b: R+T, IoU + MSE ──────────────────────────────────────────────
    model.renderer   = sil_s2b
    model.cam_views  = [(bm_s2b, bin_ref, _I3, _Z3)]
    model.iou_weight = IOU_WEIGHT_2B
    opt = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_2B},
        {"params": [model.transl], "lr": LR_T_2B},
    ])
    loop = tqdm(range(NUM_ITERS_2B), desc=f"  s2b {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s2b done: loss={loss.item():.4f}  (σ={SIGMA_2B})")

    # ── Stage 2c: R+T, IoU + MSE ──────────────────────────────────────────────
    model.renderer   = sil_s2c
    model.cam_views  = [(bm_s2c, bin_ref, _I3, _Z3)]
    model.iou_weight = IOU_WEIGHT_2C
    opt = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_2C},
        {"params": [model.transl], "lr": LR_T_2C},
    ])
    loop = tqdm(range(NUM_ITERS_2C), desc=f"  s2c {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s2c done: loss={loss.item():.4f}  (σ={SIGMA_2C})")

    # ── Stage 2d: R+T, IoU + MSE, fine sigma ──────────────────────────────────
    model.renderer   = sil_s2d
    model.cam_views  = [(bm_s2d, bin_ref, _I3, _Z3)]
    model.iou_weight = IOU_WEIGHT_2D
    opt = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_2D},
        {"params": [model.transl], "lr": LR_T_2D},
    ])
    loop = tqdm(range(NUM_ITERS_2D), desc=f"  s2d {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s2d done: loss={loss.item():.4f}  (σ={SIGMA_2D})")

    # ── Stage 2e: R+T, IoU + MSE, finest sigma ────────────────────────────────
    model.renderer   = sil_s2e
    model.cam_views  = [(bm_s2e, bin_ref, _I3, _Z3)]
    model.iou_weight = IOU_WEIGHT_2E
    opt = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_2E},
        {"params": [model.transl], "lr": LR_T_2E},
    ])
    loop = tqdm(range(NUM_ITERS_2E), desc=f"  s2e {r_tag}", leave=False)
    for _ in loop:
        opt.zero_grad(); loss, _, _ = model(); loss.backward(); opt.step()
    print(f"  [r{restart_idx}] s2e done: loss={loss.item():.4f}  (σ={SIGMA_2E})")

    iou_final = model.final_iou(sil_s2e, bin_ref)

    with torch.no_grad():
        R_est_t = rot6d_to_matrix(model.rot6d)
        T_est_t = model.transl
    R_est_pt3d = np.from_dlpack(R_est_t.detach().cpu())
    T_est_pt3d = np.from_dlpack(T_est_t.detach().cpu())
    R_est_np, T_est_np = pt3d_to_bop(R_est_pt3d, T_est_pt3d)

    return R_est_np, T_est_np, iou_final


# ── Main loop ──────────────────────────────────────────────────────────────────
csv_rows = []
fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)

for frame_id in valid_frame_ids:
    rgb_path  = os.path.join(_image_dir, f"{frame_id}.png")
    mask_path = os.path.join(_mask_dir,  f"{frame_id}.png")
    pose_path = os.path.join(_pose_dir,  f"{frame_id}.npy")

    pose_mat = np.load(pose_path)
    R_gt     = pose_mat[:3, :3].copy().astype(np.float64)
    t_gt     = pose_mat[:3,  3].copy().astype(np.float64)

    try:
        rgb_orig_pil = Image.open(rgb_path).convert("RGB")
        mask_full    = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    except OSError as e:
        print(f"\n  Frame {frame_id:04d} skipped — image load error: {e}")
        continue

    mask_resized = np.array(
        Image.fromarray((mask_full * 255).astype(np.uint8)).resize(
            (RENDER_SIZE, RENDER_SIZE), Image.NEAREST),
        dtype=np.float32) / 255.0
    mask_t = torch.tensor(mask_resized, dtype=torch.float32, device=device)

    print(f"\n══ Frame {frame_id:04d} ══  mask fg px: {int(mask_full.sum())}")
    print(f"  GT t: {t_gt.round(2)}")

    mesh = build_mesh(solid=True)

    best_R, best_T, best_iou = None, None, -1.0
    best_restart_idx = -1

    for restart_idx in range(NUM_RESTARTS):
        print(f"\n  ── Restart {restart_idx + 1}/{NUM_RESTARTS} ──")
        R_est_np, T_est_np, iou_val = run_one_restart(
            frame_id, mesh, mask_resized, mask_t,
            R_gt, t_gt, fx_r, fy_r, cx_r, cy_r, restart_idx)

        rot_r   = rotation_error_deg(R_est_np, R_gt)
        trans_r = translation_error_mm(T_est_np, t_gt)
        add_r   = add_metric(model_pts_np, R_est_np, T_est_np, R_gt, t_gt)
        print(f"  [r{restart_idx}] IoU={iou_val:.4f}  rot={rot_r:.2f}°  "
              f"trans={trans_r:.2f}mm  ADD={add_r:.2f}mm")

        if iou_val > best_iou:
            best_iou = iou_val
            best_R   = R_est_np.copy()
            best_T   = T_est_np.copy()
            best_restart_idx = restart_idx

    R_est_np = best_R
    T_est_np = best_T
    print(f"\n  Best restart: {best_restart_idx} (IoU={best_iou:.4f})")

    # ── Compute errors on best result ─────────────────────────────────────────
    rot_err   = rotation_error_deg(R_est_np, R_gt)
    trans_err = translation_error_mm(T_est_np, t_gt)
    add_val   = add_metric(model_pts_np, R_est_np, T_est_np, R_gt, t_gt)
    print(f"  FINAL: rot={rot_err:.2f}°  trans={trans_err:.2f}mm  ADD={add_val:.2f}mm")

    # ── Visualization ─────────────────────────────────────────────────────────
    try:
        gt_ov  = draw_pose_overlay(rgb_orig_pil, R_gt,     t_gt,     (0, 220, 0))
        est_ov = draw_pose_overlay(rgb_orig_pil, R_est_np, T_est_np, (220, 80, 0))
        LH = 28
        combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
        combined.paste(gt_ov,  (0,     LH))
        combined.paste(est_ov, (img_W, LH))
        d = ImageDraw.Draw(combined)
        d.text((8,         5), f"GT  frame={frame_id:04d}", fill=(100, 255, 100))
        d.text((img_W + 8, 5),
               f"EST  rot={rot_err:.1f}°  trans={trans_err:.1f}mm  ADD={add_val:.1f}mm",
               fill=(255, 180, 80))
        out = os.path.join(VIZ_DIR, f"viz_frame{frame_id:04d}_new_param.png")
        combined.save(out)
        print(f"  viz → {out}")
    except Exception as e:
        print(f"  viz skipped: {e}")

    # Compute init errors from first restart for logging
    _init_R, _init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
    init_rot_err   = rotation_error_deg(_init_R, R_gt)
    init_trans_err = translation_error_mm(_init_t, t_gt)

    csv_rows.append({
        "frame_id":   frame_id,
        "est_R_00": R_est_np[0, 0], "est_R_01": R_est_np[0, 1], "est_R_02": R_est_np[0, 2],
        "est_R_10": R_est_np[1, 0], "est_R_11": R_est_np[1, 1], "est_R_12": R_est_np[1, 2],
        "est_R_20": R_est_np[2, 0], "est_R_21": R_est_np[2, 1], "est_R_22": R_est_np[2, 2],
        "est_tx_mm": T_est_np[0],   "est_ty_mm": T_est_np[1],   "est_tz_mm": T_est_np[2],
        "gt_R_00": R_gt[0, 0], "gt_R_01": R_gt[0, 1], "gt_R_02": R_gt[0, 2],
        "gt_R_10": R_gt[1, 0], "gt_R_11": R_gt[1, 1], "gt_R_12": R_gt[1, 2],
        "gt_R_20": R_gt[2, 0], "gt_R_21": R_gt[2, 1], "gt_R_22": R_gt[2, 2],
        "gt_tx_mm": t_gt[0],   "gt_ty_mm": t_gt[1],   "gt_tz_mm": t_gt[2],
        "init_rotation_error_deg":   init_rot_err,
        "init_translation_error_mm": init_trans_err,
        "rotation_error_deg":   rot_err,
        "translation_error_mm": trans_err,
        "ADD_mm":               add_val,
        "best_iou":             best_iou,
        "num_restarts":         NUM_RESTARTS,
    })


# ── Write CSV ──────────────────────────────────────────────────────────────────
if csv_rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV → {OUTPUT_CSV}  ({len(csv_rows)} rows)")

if csv_rows:
    re_ = [r["rotation_error_deg"]   for r in csv_rows]
    te_ = [r["translation_error_mm"] for r in csv_rows]
    ae_ = [r["ADD_mm"]               for r in csv_rows]
    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Rows             : {len(csv_rows)}")
    print(f"  Mean rotation    : {np.mean(re_):.4f}°  (std {np.std(re_):.4f})")
    print(f"  Mean translation : {np.mean(te_):.4f} mm (std {np.std(te_):.4f})")
    print(f"  Mean ADD         : {np.mean(ae_):.4f} mm (std {np.std(ae_):.4f})")
    print("────────────────────────────────────────────────────────")
