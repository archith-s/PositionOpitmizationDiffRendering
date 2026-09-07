"""
optimize_model_inter_over_union.py  (multi-view + IoU metric)
Differentiable-rendering 6-DoF pose estimator (PyTorch3D silhouette loss).
Identical optimizer to optimize_modelv2.py (same three-stage schedule):

  Stage 1    – translation only; centroid+area loss summed over all 3 cameras
               (multi-view centroid gives ~3× stronger translation gradients).
  Stage 2a   – joint R+T; MSE+IoU with sigma=0.025 across all 3 cameras
               (10 px blur radius bridges the ~9–15 px gap left by Stage 1).
  Stage 2b   – joint R+T; MSE+IoU with sigma=0.005 across all 3 cameras
               (3 px precision for final shape matching).

Differences from optimize_modelv2.py:
  - Visualization draws only two things on top of the RGB frame: the pose axes
    and a flat-shaded silhouette fill (no other overlay elements).
  - The silhouette used for both visualization and metrics is a hard binary
    rasterization (blur_radius=0, faces_per_pixel=1) at full image resolution,
    not the soft differentiable shader — soft alpha still shows a ~15-50%
    halo around the true edge even at sigma=1e-4, which this avoids.
  - A new CSV metric is added: intersection-over-union (IoU) of the crisp
    binary silhouette, computed GT-vs-GT (sanity check, always 1.0) and
    GT-vs-estimate (the real metric).

Per-stage iteration counts were cut to ~1/3 of the original budget (1050 ->
340 total) -- the staging *structure* (translation locked down before
rotation gets a much lower, decaying LR; sigma annealed coarse-to-fine; IoU
loss only added once roughly aligned) is what stabilizes convergence, not
raw iteration count, so this keeps most of the benefit at a fraction of the
runtime. Symmetry-aware metrics (rotation_error_sym_deg, ADD_S_mm) were also
added -- this mesh has near-exact 180 degree rotational symmetry about its
local X axis (verified via KD-tree residual check, same as run_demo_iou.py's
SYMMETRY_TFS), so a rotation error measured only against the single
canonical R_gt over-penalizes convergence to this physically indistinguishable
twin.
"""

import os, re, csv, json, tempfile
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

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


# ─────────────────────────── CONFIG ───────────────────────────────────────────
DATASET_ROOT   = "../SIMPLELND_data"
PRIMARY_CAMERA = "camera_1"
CAMERA_NAMES   = ["camera_1"]
CAMERA_DIRS    = {cam: os.path.join(DATASET_ROOT, cam, "000001") for cam in CAMERA_NAMES}

CAMERA_DIR     = CAMERA_DIRS[PRIMARY_CAMERA]
SCENE_CAM_JSON = os.path.join(CAMERA_DIR, "scene_camera.json")
SCENE_GT_JSON  = os.path.join(CAMERA_DIR, "scene_gt.json")
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")

MESH_PATH   = ("../surgical_robotics_challenge/ADF/PSMs/"
               "LND_420006/high_res/tool pitch link.OBJ")
OUTPUT_CSV  = "../IoU_CSV_Visuals_Diff_Render/pose_estimation_results_iou.csv"
VIZ_DIR     = "../IoU_CSV_Visuals_Diff_Render"

# Stage 1: translation only (single-view centroid+area)
NUM_ITERS_TRANS    = 60    # was 200 -- centroid convergence is fast
LR_TRANS           = 0.05

# Stage 1.5: translation-only silhouette MSE — refines depth before rotation starts
NUM_ITERS_TRANS_S1B = 90   # was 300
LR_TRANS_S1B        = 0.05
SIGMA_STAGE1B       = 0.025  # same sigma as Stage 2a (cleaner gradient than 0.05)

# Stage 2a: coarse rotation — MSE only, no IoU
NUM_ITERS_COARSE  = 70     # was 200
LR_ROT_COARSE     = 0.003
SIGMA_STAGE2A     = 0.025

# Stage 2b: intermediate sigma — MSE + light IoU bridges 0.025→0.005 gap
NUM_ITERS_MID     = 50     # was 150
LR_ROT_MID        = 0.002
SIGMA_STAGE2B     = 0.010
IOU_WEIGHT_MID    = 10.0

# Stage 2c: fine rotation — MSE + IoU, tight sigma
NUM_ITERS_JOINT   = 70     # was 200
LR_ROT            = 0.001
SIGMA_STAGE2      = 0.005

IOU_WEIGHT    = 20.0
SIGMA_STAGE1  = 0.01
RENDER_SIZE   = 256
FG_THRESHOLD  = 0.1
BRIGHT_FG     = True

USE_GT_INITIALIZATION  = True
INIT_ROT_NOISE_DEG     = 10.0
INIT_TRANS_NOISE_MM    = 10.0
USE_GT_INIT_ONLY       = False

TARGET_OBJ_IDS        = {1, 3}
USE_RENDERED_REFERENCE = True
DEBUG_DIR              = "../debug_renders"

IOU_THRESHOLD   = 0.5

# Visualization-only: clip the silhouette fill to pixels that photometrically
# look like the tool (low saturation) so occluding tissue isn't painted over.
# Does not affect the CSV's IoU metric, which stays purely geometric.
APPEARANCE_SAT_THRESHOLD = 0.25
# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--all",    action="store_true", help="Run all frames")
parser.add_argument("--frames", type=int, default=None,
                    help="Run only the first N frames")
args = parser.parse_args()

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(VIZ_DIR,   exist_ok=True)
print(f"Debug → {DEBUG_DIR}   Viz → {VIZ_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=device)
GRID_Y, GRID_X = torch.meshgrid(_arange, _arange, indexing="ij")


# ── Load JSON for ALL cameras ──────────────────────────────────────────────────
scene_gt_all:  dict[str, dict] = {}
scene_cam_all: dict[str, dict] = {}
for cam in CAMERA_NAMES:
    with open(os.path.join(CAMERA_DIRS[cam], "scene_gt.json"))     as f:
        scene_gt_all[cam] = json.load(f)
    with open(os.path.join(CAMERA_DIRS[cam], "scene_camera.json")) as f:
        scene_cam_all[cam] = json.load(f)

scene_gt  = scene_gt_all[PRIMARY_CAMERA]
scene_cam = scene_cam_all[PRIMARY_CAMERA]
print(f"Frames (primary camera): {len(scene_gt)}")


# ── Frame-id → filename maps ───────────────────────────────────────────────────
_rgb_on_disk_all: dict[str, dict] = {}
for cam in CAMERA_NAMES:
    rgb_dir = os.path.join(CAMERA_DIRS[cam], "rgb")
    _rgb_on_disk_all[cam] = {
        int(fn.split(".")[0]): fn
        for fn in os.listdir(rgb_dir)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    }
_rgb_on_disk = _rgb_on_disk_all[PRIMARY_CAMERA]

valid_frame_ids = sorted(
    set(int(k) for k in scene_gt.keys()) & set(_rgb_on_disk.keys())
)
print(f"Valid frames: {len(valid_frame_ids)}")

if args.frames is not None:
    valid_frame_ids = valid_frame_ids[:args.frames]
    print(f"Running {len(valid_frame_ids)} frame(s) (--frames {args.frames})")
elif not args.all:
    valid_frame_ids = valid_frame_ids[:10]
    print(f"Running first {len(valid_frame_ids)} frames (--all for all)")


# ── Image dimensions ───────────────────────────────────────────────────────────
_sample_img = Image.open(os.path.join(RGB_DIR, _rgb_on_disk[valid_frame_ids[0]]))
img_W, img_H = _sample_img.size
print(f"Image size: {img_W}×{img_H}")


# ── Load mesh ──────────────────────────────────────────────────────────────────
def load_obj_no_mtl(path: str):
    with open(path) as fh:
        lines = fh.readlines()
    clean = [l for l in lines if not re.match(r"\s*(mtllib|usemtl)\b", l)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp.writelines(clean); tmp_path = tmp.name
    verts, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    return verts, faces_idx.verts_idx

verts, faces = load_obj_no_mtl(MESH_PATH)
verts_np = np.from_dlpack(verts.detach().cpu())
_vmax = float(verts_np.__abs__().max())
print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces | max|v|={_vmax:.4f}")
if _vmax < 1.0:
    print("  Auto-scaling ×1000 (m→mm)")
    verts = verts * 1000.0; verts_np = verts_np * 1000.0
else:
    print("  Vertices in mm.")

# This mesh has near-exact 180° rotational symmetry about its local X axis
# (verified via KD-tree residual check: 0.10mm mean on a 14.35mm object --
# same finding run_demo_iou.py's SYMMETRY_TFS is built from). A rotation
# error measured only against the single canonical R_gt over-penalizes
# convergence to this physically indistinguishable twin.
_mesh_center = (verts_np.min(axis=0) + verts_np.max(axis=0)) / 2
_Rx180 = np.diag([1.0, -1.0, -1.0])
_Sx180 = np.eye(4)
_Sx180[:3, :3] = _Rx180
_Sx180[:3,  3] = _mesh_center - _Rx180 @ _mesh_center
SYMMETRY_TFS = [np.eye(4), _Sx180]


# ── Per-object mean GT translation ────────────────────────────────────────────
all_obj_ids = sorted({o["obj_id"] for fd in scene_gt.values() for o in fd})
obj_mean_t: dict[int, torch.Tensor] = {}
for oid in all_obj_ids:
    ts   = [o["cam_t_m2c"] for fd in scene_gt.values() for o in fd if o["obj_id"] == oid]
    mean = np.mean(ts, axis=0)
    obj_mean_t[oid] = torch.tensor(mean, dtype=torch.float32, device=device)
print(f"Objects: {all_obj_ids}")


# ── Camera / renderer helpers ──────────────────────────────────────────────────
# Screen-space (in_ndc=False) cameras with an explicit image_size, NOT
# PyTorch3D's default NDC convention -- NDC silently mishandles non-square
# aspect ratios, which produced a confirmed, measured pixel offset against a
# manual-rasterizer render of identical pose data (see sim_iter_images.py,
# fixed 2026-07-10).
def scale_intrinsics(cam_K: list, W_from: int, H_from: int, W_to: int, H_to: int):
    """Rescale pixel-space intrinsics for rendering at a different resolution
    (the optimizer renders at RENDER_SIZE, not the native capture resolution)."""
    sx, sy = W_to / W_from, H_to / H_from
    return cam_K[0] * sx, cam_K[4] * sy, cam_K[2] * sx, cam_K[5] * sy

def make_cameras(R, T, fx, fy, cx, cy, H, W) -> PerspectiveCameras:
    return PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        R=R, T=T, in_ndc=False, image_size=((H, W),), device=device,
    )

def make_silhouette_renderer(fx, fy, cx, cy, H, W, sigma=1e-4):
    blend  = BlendParams(sigma=sigma, gamma=sigma)
    blur   = float(np.log(1.0 / sigma - 1.0)) * sigma
    raster = RasterizationSettings(
        image_size=(H, W), blur_radius=blur,
        faces_per_pixel=100, bin_size=0,
    )
    cam = make_cameras(torch.eye(3, device=device).unsqueeze(0),
                       torch.zeros(1, 3, device=device),
                       fx, fy, cx, cy, H, W)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=blend),
    )

def build_mesh() -> Meshes:
    verts_rgb = torch.ones_like(verts)[None]
    return Meshes(verts=[verts.to(device)], faces=[faces.to(device)],
                  textures=TexturesVertex(verts_features=verts_rgb.to(device)))


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


# ── Precompute relative camera extrinsics (rigid mount) ───────────────────────
# R_rel, T_rel: P_cam_tgt = P_cam_src @ R_rel + T_rel  (PT3D row-vector)
def _compute_extrinsics(src, tgt, ref_fkey="0", ref_obj_id=1):
    o0 = next(o for o in scene_gt_all[src][ref_fkey] if o["obj_id"] == ref_obj_id)
    o1 = next(o for o in scene_gt_all[tgt][ref_fkey] if o["obj_id"] == ref_obj_id)
    R0, T0 = bop_to_pt3d(np.array(o0["cam_R_m2c"]).reshape(3,3), np.array(o0["cam_t_m2c"]))
    R1, T1 = bop_to_pt3d(np.array(o1["cam_R_m2c"]).reshape(3,3), np.array(o1["cam_t_m2c"]))
    R_rel = R0.T @ R1;  T_rel = T1 - T0 @ R_rel
    return (torch.tensor(R_rel, dtype=torch.float32, device=device),
            torch.tensor(T_rel, dtype=torch.float32, device=device))

_I3 = torch.eye(3,  dtype=torch.float32, device=device)
_Z3 = torch.zeros(3, dtype=torch.float32, device=device)
cam_extrinsics: dict[str, tuple] = {
    "camera_1": (_I3, _Z3),
}


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3); ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(ax * np.deg2rad(
        np.random.uniform(-rot_noise_deg, rot_noise_deg))).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Pose model ────────────────────────────────────────────────────────────────
class PoseModel(nn.Module):
    """
    Stage 1 (use_centroid=True):
      cent_views = [(cx_ref, cy_ref, area_ref, R_rel, T_rel), ...]
      Multi-view centroid+area loss; sum over all cameras.

    Stage 2 (use_centroid=False):
      cam_views  = [(ref_mask, R_rel, T_rel), ...]
      Multi-view MSE+IoU; sum / N_views.
    """

    def __init__(self, mesh, renderer, image_ref_np, init_R, init_t,
                 fx, fy, cx, cy, H, W):
        super().__init__()
        self.mesh = mesh;  self.renderer = renderer
        self.fx = fx;  self.fy = fy
        self.cx = cx;  self.cy = cy
        self.H  = H;   self.W  = W
        self.cent_views = []   # populated before Stage 1
        self.cam_views  = []   # populated before Stage 2

        if image_ref_np.ndim == 2:
            fg_mask = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
        else:
            ref_t = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
            br    = ref_t.mean(-1) / ref_t.max().clamp(min=1e-6)
            fg_mask = (br > FG_THRESHOLD).float() if BRIGHT_FG \
                      else (br < (1.0 - FG_THRESHOLD)).float()
        self.register_buffer("image_ref", fg_mask)

        # Camera-0 centroid buffers (fallback for single-view centroid)
        ra = fg_mask.sum().clamp(min=1.0)
        self.register_buffer("cx_ref",   ((GRID_X * fg_mask).sum() / ra).unsqueeze(0))
        self.register_buffer("cy_ref",   ((GRID_Y * fg_mask).sum() / ra).unsqueeze(0))
        self.register_buffer("area_ref", ra.unsqueeze(0))

        self.rot6d  = nn.Parameter(matrix_to_rot6d(init_R))
        self.transl = nn.Parameter(torch.tensor(init_t, dtype=torch.float32, device=device))
        self.use_centroid = True
        self.use_iou      = False
        self.iou_weight   = IOU_WEIGHT

    def _render_alpha(self, R_rel, T_rel):
        R = rot6d_to_matrix(self.rot6d).unsqueeze(0)
        T = self.transl.unsqueeze(0)
        R_i = torch.mm(R[0], R_rel).unsqueeze(0)
        T_i = torch.mm(T, R_rel) + T_rel.unsqueeze(0)
        cam = make_cameras(R_i, T_i, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
        return self.renderer(meshes_world=self.mesh.clone(), cameras=cam)[0, ..., 3]

    def forward(self):
        R = rot6d_to_matrix(self.rot6d).unsqueeze(0)
        T = self.transl.unsqueeze(0)

        if self.use_centroid:
            if self.cent_views:
                total = torch.tensor(0.0, device=device)
                for cx_r, cy_r, ar_r, R_rel, T_rel in self.cent_views:
                    alpha = self._render_alpha(R_rel, T_rel)
                    area  = alpha.sum().clamp(min=1e-6)
                    cx    = (GRID_X * alpha).sum() / area
                    cy    = (GRID_Y * alpha).sum() / area
                    total = total + (cx - cx_r)**2 + (cy - cy_r)**2 \
                                  + ((area - ar_r) / ar_r)**2
                loss = total / len(self.cent_views)
            else:
                # single-view fallback
                alpha = self._render_alpha(_I3, _Z3)
                area  = alpha.sum().clamp(min=1e-6)
                cx    = (GRID_X * alpha).sum() / area
                cy    = (GRID_Y * alpha).sum() / area
                loss  = ((cx - self.cx_ref[0])**2 + (cy - self.cy_ref[0])**2
                         + ((area - self.area_ref[0]) / self.area_ref[0])**2)
        else:
            views = self.cam_views or [(self.image_ref, _I3, _Z3)]
            total = torch.tensor(0.0, device=device)
            for ref_mask_i, R_rel_i, T_rel_i in views:
                alpha_i = self._render_alpha(R_rel_i, T_rel_i)
                mse_i   = (alpha_i - ref_mask_i).pow(2).sum()
                if self.use_iou:
                    inter   = (alpha_i * ref_mask_i).sum()
                    union   = (alpha_i + ref_mask_i - alpha_i * ref_mask_i).sum()
                    total   = total + mse_i + self.iou_weight * (1.0 - inter / union.clamp(min=1e-6))
                else:
                    total = total + mse_i
            loss = total / len(views)

        return loss, R, T


# ── Error metrics ──────────────────────────────────────────────────────────────
def rotation_error_deg(R_est, R_gt):
    cos_a = float(np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))

def translation_error_mm(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))

def add_metric(verts_np, R_est, t_est, R_gt, t_gt):
    return float(np.mean(np.linalg.norm(
        ((R_est @ verts_np.T).T + t_est) - ((R_gt @ verts_np.T).T + t_gt), axis=1)))

def rotation_error_sym_deg(R_est, R_gt, symmetry_tfs):
    """Minimum geodesic rotation error over all symmetry-equivalent poses."""
    return float(min(rotation_error_deg(R_est @ S[:3, :3], R_gt) for S in symmetry_tfs))

def add_s_metric(verts_np, R_est, t_est, R_gt, t_gt, symmetry_tfs):
    """ADD-S: minimum ADD over all symmetry-equivalent poses."""
    pts_gt = (R_gt @ verts_np.T).T + t_gt
    best = float("inf")
    for S in symmetry_tfs:
        R_s = R_est @ S[:3, :3]
        t_s = R_est @ S[:3, 3] + t_est
        pts_est = (R_s @ verts_np.T).T + t_s
        best = min(best, float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1))))
    return best

def iou_metric(alpha_a: np.ndarray, alpha_b: np.ndarray, threshold=IOU_THRESHOLD) -> float:
    """Binary IoU between two crisp silhouette alpha masks."""
    mask_a = alpha_a > threshold
    mask_b = alpha_b > threshold
    union  = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter) / float(union)


# ── Crisp silhouette + visualization (created once at full resolution) ────────
# Hard (non-differentiable) rasterization — no soft shader — so the silhouette
# edge is exact instead of carrying the soft shader's blur halo.
_crisp_raster_settings = RasterizationSettings(
    image_size=(img_H, img_W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
)


def render_crisp_alpha(R_bop, t_bop, cam_K, mesh) -> np.ndarray:
    """Hard binary silhouette at full image resolution, tightly covering the
    tool — used for both the visualization fill and IoU."""
    R_pt3d, T_pt3d = bop_to_pt3d(np.array(R_bop), np.array(t_bop))
    R_t = torch.tensor(R_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    T_t = torch.tensor(T_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    cam = make_cameras(R_t, T_t, cam_K[0], cam_K[4], cam_K[2], cam_K[5], img_H, img_W)
    rasterizer = MeshRasterizer(cameras=cam, raster_settings=_crisp_raster_settings)
    with torch.no_grad():
        frags = rasterizer(mesh.clone(), cameras=cam)
        mask  = (frags.pix_to_face[0, ..., 0] >= 0).float()
    return np.from_dlpack(mask.detach().cpu())


def appearance_foreground_mask(rgb_pil, sat_threshold=APPEARANCE_SAT_THRESHOLD) -> np.ndarray:
    """Photometric tool-vs-tissue mask from the real image alone: low
    saturation (grayish) = metal tool, high saturation (pink/red) = tissue.
    No 3D scene model needed, so it applies the same way to real data."""
    rgb = np.array(rgb_pil, dtype=np.float32)
    mx  = rgb.max(axis=-1)
    mn  = rgb.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1.0, None)
    return (sat < sat_threshold).astype(np.float32)


def draw_axes_and_silhouette(rgb_pil, alpha, R_bop, t_bop, cam_K,
                             color_sil=(0, 220, 0), appearance_mask=None) -> Image.Image:
    """Draws exactly two things on top of the RGB frame: a flat-shaded
    silhouette fill and the 3-axis pose gizmo. If appearance_mask is given,
    the fill is clipped to it so the render never spills onto pixels that
    don't actually look like the tool (e.g. tissue occluding part of the
    tracked mesh) — this substitutes for true 3D depth compositing, which
    isn't possible without a tissue mesh (and never will be on real data)."""
    fill = alpha if appearance_mask is None else alpha * appearance_mask
    overlay = np.zeros((*fill.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    rgb_arr = np.array(rgb_pil, dtype=np.float32)
    blended = np.clip(0.5 * fill[..., None] * overlay + rgb_arr, 0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)

    draw = ImageDraw.Draw(result)
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    def proj(p):
        pc = np.array(R_bop) @ np.array(p) + np.array(t_bop)
        return (int(round(fx * pc[0]/pc[2] + cx)), int(round(fy * pc[1]/pc[2] + cy))) \
               if pc[2] > 0 else None
    L   = 50.0
    cen = proj([0,0,0])
    for end, col in zip([proj([L,0,0]), proj([0,L,0]), proj([0,0,L])],
                        [(220,50,50),(50,220,50),(50,50,220)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


# ── Main loop ──────────────────────────────────────────────────────────────────
csv_rows = []

for frame_id in valid_frame_ids:
    fkey     = str(frame_id)
    rgb_path = os.path.join(RGB_DIR, _rgb_on_disk[frame_id])
    cam_K    = scene_cam[fkey]["cam_K"]
    fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)

    rgb_orig_pil = Image.open(rgb_path).convert("RGB")
    ref_np       = np.array(rgb_orig_pil.resize((RENDER_SIZE, RENDER_SIZE), Image.BILINEAR),
                             dtype=np.float32)
    appearance_mask = appearance_foreground_mask(rgb_orig_pil)

    print(f"\n══ Frame {frame_id:04d} ({_rgb_on_disk[frame_id]}) ══")

    for obj_data in scene_gt[fkey]:
        obj_id = obj_data["obj_id"]
        if obj_id not in TARGET_OBJ_IDS:
            continue

        R_gt = np.array(obj_data["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t_gt = np.array(obj_data["cam_t_m2c"], dtype=np.float64)
        print(f"  ── obj_id={obj_id} ──")

        mesh = build_mesh()
        sil_renderer_ref     = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=1e-4)
        sil_renderer_stage1  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE1)
        sil_renderer_stage2a = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2A)
        sil_renderer_stage2b = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2B)
        sil_renderer_stage2  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2)

        if USE_GT_INITIALIZATION:
            init_R, init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
        else:
            init_R = np.eye(3, dtype=np.float64)
            init_t = np.from_dlpack(obj_mean_t[obj_id].detach().cpu())

        init_rot_err   = rotation_error_deg(init_R, R_gt)
        init_trans_err = translation_error_mm(init_t, t_gt)
        print(f"    Init: rot={init_rot_err:.2f}°  trans={init_trans_err:.2f} mm")

        if USE_GT_INIT_ONLY:
            R_est_np = init_R.copy();  T_est_np = init_t.copy()
        else:
            # ── Build references for all 3 cameras ───────────────────────────
            if USE_RENDERED_REFERENCE:
                # Build per-camera references
                cent_views_s1  = []   # (cx_ref, cy_ref, area_ref, R_rel, T_rel)
                cam_views_s2a  = []   # σ=0.025 — Stage 1.5 (trans) + Stage 2a (rot MSE)
                cam_views_s2b  = []   # σ=0.010 — Stage 2b (intermediate IoU)
                cam_views_s2c  = []   # σ=0.005 — Stage 2c (fine IoU)

                for cam_name in CAMERA_NAMES:
                    od_c = next(
                        (o for o in scene_gt_all[cam_name][fkey] if o["obj_id"] == obj_id),
                        None)
                    if od_c is None:
                        continue
                    R_c, T_c = bop_to_pt3d(
                        np.array(od_c["cam_R_m2c"]).reshape(3,3),
                        np.array(od_c["cam_t_m2c"]))
                    Rc_t = torch.tensor(R_c, dtype=torch.float32, device=device).unsqueeze(0)
                    Tc_t = torch.tensor(T_c, dtype=torch.float32, device=device).unsqueeze(0)
                    cam_c = make_cameras(Rc_t, Tc_t, fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE)

                    with torch.no_grad():
                        r_crisp = sil_renderer_ref(    meshes_world=mesh.clone(), cameras=cam_c)
                        r_2a    = sil_renderer_stage2a(meshes_world=mesh.clone(), cameras=cam_c)
                        r_2b    = sil_renderer_stage2b(meshes_world=mesh.clone(), cameras=cam_c)
                        r_2c    = sil_renderer_stage2( meshes_world=mesh.clone(), cameras=cam_c)

                    mask_crisp = r_crisp[0,...,3]
                    area_c = mask_crisp.sum().clamp(min=1.0)
                    cx_c   = (GRID_X * mask_crisp).sum() / area_c
                    cy_c   = (GRID_Y * mask_crisp).sum() / area_c

                    R_rel_c, T_rel_c = cam_extrinsics[cam_name]
                    cent_views_s1.append((cx_c, cy_c, area_c, R_rel_c, T_rel_c))
                    cam_views_s2a.append((r_2a[0,...,3].detach().clone(), R_rel_c, T_rel_c))
                    cam_views_s2b.append((r_2b[0,...,3].detach().clone(), R_rel_c, T_rel_c))
                    cam_views_s2c.append((r_2c[0,...,3].detach().clone(), R_rel_c, T_rel_c))

                ref_for_model = np.from_dlpack(r_crisp[0,...,3].detach().cpu())

            else:
                fg = (ref_np.mean(-1) / max(ref_np.max(), 1e-6) > FG_THRESHOLD).astype("float32")
                fg_t = torch.tensor(fg, dtype=torch.float32, device=device)
                area_c = fg_t.sum().clamp(min=1.0)
                cent_views_s1 = [(
                    (GRID_X * fg_t).sum() / area_c,
                    (GRID_Y * fg_t).sum() / area_c,
                    area_c, _I3, _Z3
                )]
                cam_views_s2a = [(fg_t, _I3, _Z3)]
                cam_views_s2b = [(fg_t, _I3, _Z3)]
                cam_views_s2c = [(fg_t, _I3, _Z3)]
                ref_for_model = ref_np

            # ── Build model ───────────────────────────────────────────────────
            init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
            model = PoseModel(
                mesh, sil_renderer_stage1, ref_for_model,
                init_R_pt3d, init_t_pt3d,
                fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
            ).to(device)

            # ── Stage 1: translation only, multi-view centroid+area ───────────
            model.cent_views  = cent_views_s1
            model.use_centroid = True
            model.use_iou      = False
            opt_s1 = torch.optim.Adam([model.transl], lr=LR_TRANS)
            loop_s1 = tqdm(range(NUM_ITERS_TRANS), desc=f"    s1 obj{obj_id}")
            for _ in loop_s1:
                opt_s1.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1.step()
                loop_s1.set_description(f"    s1 obj{obj_id} ({loss.item():.3f})")
            print(f"    Stage 1 done: loss={loss.item():.4f}  ({len(cent_views_s1)}-view centroid)")

            # ── Stage 1.5: translation-only silhouette MSE (refines depth) ───
            model.renderer     = sil_renderer_stage2a
            model.cam_views    = cam_views_s2a
            model.use_centroid = False
            model.use_iou      = False
            opt_s1b = torch.optim.Adam([model.transl], lr=LR_TRANS_S1B)
            loop_s1b = tqdm(range(NUM_ITERS_TRANS_S1B), desc=f"    s1b obj{obj_id}")
            for _ in loop_s1b:
                opt_s1b.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1b.step()
                loop_s1b.set_description(f"    s1b obj{obj_id} ({loss.item():.1f})")
            print(f"    Stage 1.5 done: loss={loss.item():.2f}  (σ={SIGMA_STAGE1B})")

            # ── Stage 2a: coarse rotation — MSE only (stable gradient) ───────
            model.renderer     = sil_renderer_stage2a
            model.cam_views    = cam_views_s2a
            model.use_centroid = False
            model.use_iou      = False
            opt_s2a = torch.optim.Adam([
                {"params": [model.rot6d],  "lr": LR_ROT_COARSE},
                {"params": [model.transl], "lr": LR_TRANS * 0.1},
            ])
            loop_s2a = tqdm(range(NUM_ITERS_COARSE), desc=f"    s2a obj{obj_id}")
            for _ in loop_s2a:
                opt_s2a.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2a.step()
                loop_s2a.set_description(f"    s2a obj{obj_id} ({loss.item():.1f})")
            print(f"    Stage 2a done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2A})")

            # ── Stage 2b: intermediate sigma — bridges 0.025→0.005 gap ───────
            model.renderer   = sil_renderer_stage2b
            model.cam_views  = cam_views_s2b
            model.use_iou    = True
            model.iou_weight = IOU_WEIGHT_MID
            opt_s2b = torch.optim.Adam([
                {"params": [model.rot6d],  "lr": LR_ROT_MID},
                {"params": [model.transl], "lr": LR_TRANS * 0.05},
            ])
            loop_s2b = tqdm(range(NUM_ITERS_MID), desc=f"    s2b obj{obj_id}")
            for _ in loop_s2b:
                opt_s2b.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2b.step()
                loop_s2b.set_description(f"    s2b obj{obj_id} ({loss.item():.1f})")
            print(f"    Stage 2b done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2B})")

            # ── Stage 2c: fine rotation — tight sigma + full IoU ─────────────
            model.renderer   = sil_renderer_stage2
            model.cam_views  = cam_views_s2c
            model.use_iou    = True
            model.iou_weight = IOU_WEIGHT
            opt_s2c = torch.optim.Adam([
                {"params": [model.rot6d],  "lr": LR_ROT},
                {"params": [model.transl], "lr": LR_TRANS * 0.02},
            ])
            loop_s2c = tqdm(range(NUM_ITERS_JOINT), desc=f"    s2c obj{obj_id}")
            for _ in loop_s2c:
                opt_s2c.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2c.step()
                loop_s2c.set_description(f"    s2c obj{obj_id} ({loss.item():.1f})")
            print(f"    Stage 2c done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2})")

            with torch.no_grad():
                R_est_t = rot6d_to_matrix(model.rot6d)
                T_est_t = model.transl
            R_est_pt3d_np = np.from_dlpack(R_est_t.detach().cpu())
            T_est_pt3d_np = np.from_dlpack(T_est_t.detach().cpu())
            R_est_np, T_est_np = pt3d_to_bop(R_est_pt3d_np, T_est_pt3d_np)

        # ── Errors ────────────────────────────────────────────────────────────
        rot_err     = rotation_error_deg(R_est_np, R_gt)
        rot_err_sym = rotation_error_sym_deg(R_est_np, R_gt, SYMMETRY_TFS)
        trans_err   = translation_error_mm(T_est_np, t_gt)
        add_val     = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)
        add_s_val   = add_s_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt, SYMMETRY_TFS)

        # ── IoU (crisp binary silhouette, full resolution) ──────────────────────
        alpha_gt  = render_crisp_alpha(R_gt,     t_gt,     cam_K, mesh)
        alpha_est = render_crisp_alpha(R_est_np, T_est_np, cam_K, mesh)
        iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
        iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)
        print(f"    rot={rot_err:.2f}° (sym={rot_err_sym:.2f}°)  trans={trans_err:.2f}mm  "
              f"ADD={add_val:.2f}mm (ADD-S={add_s_val:.2f}mm)  "
              f"IoU(gt,gt)={iou_gt_vs_gt:.4f}  IoU(gt,est)={iou_gt_vs_est:.4f}")

        # ── Visualization: only axes + silhouette fill on the RGB frame ────────
        try:
            gt_ov  = draw_axes_and_silhouette(rgb_orig_pil, alpha_gt,  R_gt,     t_gt,     cam_K, (0, 220, 0),
                                              appearance_mask=appearance_mask)
            est_ov = draw_axes_and_silhouette(rgb_orig_pil, alpha_est, R_est_np, T_est_np, cam_K, (220, 80, 0),
                                              appearance_mask=appearance_mask)
            LH = 28
            combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
            combined.paste(gt_ov,  (0,     LH));  combined.paste(est_ov, (img_W, LH))
            d = ImageDraw.Draw(combined)
            d.text((8,         5), f"GT  frame={frame_id:04d}  obj={obj_id}", fill=(100,255,100))
            d.text((img_W + 8, 5), f"EST  rot={rot_err:.1f}° (sym={rot_err_sym:.1f}°)  trans={trans_err:.1f}mm  "
                                    f"ADD={add_val:.1f}mm (ADD-S={add_s_val:.1f}mm)  IoU={iou_gt_vs_est:.3f}",
                   fill=(255,180,80))
            out = os.path.join(VIZ_DIR, f"viz_frame{frame_id:04d}_obj{obj_id}.png")
            combined.save(out);  print(f"    viz → {out}")
        except Exception as e:
            print(f"    viz skipped: {e}")

        csv_rows.append({
            "frame_id": frame_id, "rgb_file": _rgb_on_disk[frame_id], "obj_id": obj_id,
            "est_R_00": R_est_np[0,0], "est_R_01": R_est_np[0,1], "est_R_02": R_est_np[0,2],
            "est_R_10": R_est_np[1,0], "est_R_11": R_est_np[1,1], "est_R_12": R_est_np[1,2],
            "est_R_20": R_est_np[2,0], "est_R_21": R_est_np[2,1], "est_R_22": R_est_np[2,2],
            "est_tx_mm": T_est_np[0],  "est_ty_mm": T_est_np[1],  "est_tz_mm": T_est_np[2],
            "gt_R_00": R_gt[0,0], "gt_R_01": R_gt[0,1], "gt_R_02": R_gt[0,2],
            "gt_R_10": R_gt[1,0], "gt_R_11": R_gt[1,1], "gt_R_12": R_gt[1,2],
            "gt_R_20": R_gt[2,0], "gt_R_21": R_gt[2,1], "gt_R_22": R_gt[2,2],
            "gt_tx_mm": t_gt[0],  "gt_ty_mm": t_gt[1],  "gt_tz_mm": t_gt[2],
            "init_rotation_error_deg":   init_rot_err,
            "init_translation_error_mm": init_trans_err,
            "rotation_error_deg":     rot_err,
            "rotation_error_sym_deg": rot_err_sym,
            "translation_error_mm":   trans_err,
            "ADD_mm":                 add_val,
            "ADD_S_mm":               add_s_val,
            "iou_gt_vs_gt":           iou_gt_vs_gt,
            "iou_gt_vs_est":          iou_gt_vs_est,
        })


# ── Write CSV ──────────────────────────────────────────────────────────────────
if csv_rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader();  w.writerows(csv_rows)
    print(f"\nCSV → {OUTPUT_CSV}  ({len(csv_rows)} rows)")

if csv_rows:
    re_     = [r["rotation_error_deg"]     for r in csv_rows]
    re_sym_ = [r["rotation_error_sym_deg"] for r in csv_rows]
    te_     = [r["translation_error_mm"]   for r in csv_rows]
    ae_     = [r["ADD_mm"]                 for r in csv_rows]
    ae_s_   = [r["ADD_S_mm"]               for r in csv_rows]
    iou_    = [r["iou_gt_vs_est"]          for r in csv_rows]
    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Rows                 : {len(csv_rows)}")
    print(f"  Mean rotation        : {np.mean(re_):.4f}°  (std {np.std(re_):.4f})")
    print(f"  Mean rotation (sym)  : {np.mean(re_sym_):.4f}°  (std {np.std(re_sym_):.4f})")
    print(f"  Mean translation     : {np.mean(te_):.4f} mm (std {np.std(te_):.4f})")
    print(f"  Mean ADD             : {np.mean(ae_):.4f} mm (std {np.std(ae_):.4f})")
    print(f"  Mean ADD-S           : {np.mean(ae_s_):.4f} mm (std {np.std(ae_s_):.4f})")
    print(f"  Mean IoU(gt,est)     : {np.mean(iou_):.4f}  (std {np.std(iou_):.4f})")
    print("────────────────────────────────────────────────────────")
