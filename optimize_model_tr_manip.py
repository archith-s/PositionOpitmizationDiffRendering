"""
optimize_model_tr_manip.py
Ablation study: sweep translation and rotation initialization noise on the first frame.

8 configurations:
  Translation sweep (rotation fixed at 10°): 10, 15, 20, 25 mm
  Rotation sweep    (translation fixed at 10mm): 15, 20, 25 deg  [10deg/10mm shared above]
  No-anchor: mean object translation as start (no GT proximity knowledge)

Outputs:
  CSV  — one row per config with estimate, GT, and all error metrics
  PNGs — 8 side-by-side GT vs estimated overlay images

Pipeline matches optimize_modelv2.py (5-stage optimizer with sigma annealing).
"""

import os, re, csv, json, tempfile
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
PRIMARY_CAMERA = "camera_l"
CAMERA_NAMES   = ["camera_l"]
CAMERA_DIRS    = {cam: os.path.join(DATASET_ROOT, cam, "000001") for cam in CAMERA_NAMES}

CAMERA_DIR     = CAMERA_DIRS[PRIMARY_CAMERA]
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")

MESH_PATH   = ("../surgical_robotics_challenge/ADF/PSMs/"
               "LND_420006/high_res/tool pitch link.OBJ")
OUTPUT_CSV  = "../Diff_Render_CSV_Visuals/pose_estimation_tr_manip.csv"
VIZ_DIR     = "../Diff_Render_CSV_Visuals/tr_manip"

TARGET_OBJ_ID = 1

# Stage 1: translation only (single-view centroid+area)
NUM_ITERS_TRANS    = 200
LR_TRANS           = 0.05

# Stage 1.5: translation-only silhouette MSE — refines depth before rotation starts
NUM_ITERS_TRANS_S1B = 300
LR_TRANS_S1B        = 0.05
SIGMA_STAGE1B       = 0.025

# Stage 2a: coarse rotation — MSE only, no IoU
NUM_ITERS_COARSE  = 200
LR_ROT_COARSE     = 0.003
SIGMA_STAGE2A     = 0.025

# Stage 2b: intermediate sigma — MSE + light IoU bridges 0.025→0.005 gap
NUM_ITERS_MID     = 150
LR_ROT_MID        = 0.002
SIGMA_STAGE2B     = 0.010
IOU_WEIGHT_MID    = 10.0

# Stage 2c: fine rotation — MSE + IoU, tight sigma
NUM_ITERS_JOINT   = 200
LR_ROT            = 0.001
SIGMA_STAGE2      = 0.005

IOU_WEIGHT    = 20.0
SIGMA_STAGE1  = 0.01
RENDER_SIZE   = 256
FG_THRESHOLD  = 0.1
BRIGHT_FG     = True
USE_RENDERED_REFERENCE = True
DEBUG_DIR              = "../debug_renders"

# Fixed seed ensures identical perturbation direction across configs;
# only the noise magnitude varies, keeping the ablation clean.
FIXED_SEED = 42
# ──────────────────────────────────────────────────────────────────────────────

# 8 ablation configurations.
# use_gt_anchor=True  → perturb GT pose by (rot_deg, trans_mm)
# use_gt_anchor=False → fall back to mean object translation (no GT proximity info)
CONFIGS = [
    {"label": "T10mm_R10deg",  "rot_deg": 10.0, "trans_mm": 10.0, "use_gt_anchor": True},
    {"label": "T15mm_R10deg",  "rot_deg": 10.0, "trans_mm": 15.0, "use_gt_anchor": True},
    {"label": "T20mm_R10deg",  "rot_deg": 10.0, "trans_mm": 20.0, "use_gt_anchor": True},
    {"label": "T25mm_R10deg",  "rot_deg": 10.0, "trans_mm": 25.0, "use_gt_anchor": True},
    {"label": "T10mm_R15deg",  "rot_deg": 15.0, "trans_mm": 10.0, "use_gt_anchor": True},
    {"label": "T10mm_R20deg",  "rot_deg": 20.0, "trans_mm": 10.0, "use_gt_anchor": True},
    {"label": "T10mm_R25deg",  "rot_deg": 25.0, "trans_mm": 10.0, "use_gt_anchor": True},
    {"label": "no_gt_anchor",  "rot_deg":  0.0, "trans_mm":  0.0, "use_gt_anchor": False},
]

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

# Single frame: first in dataset
frame_id = valid_frame_ids[0]
print(f"Running on frame {frame_id}")


# ── Image dimensions ───────────────────────────────────────────────────────────
_sample_img = Image.open(os.path.join(RGB_DIR, _rgb_on_disk[frame_id]))
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


# ── Precompute relative camera extrinsics ─────────────────────────────────────
_I3 = torch.eye(3,  dtype=torch.float32, device=device)
_Z3 = torch.zeros(3, dtype=torch.float32, device=device)
cam_extrinsics: dict[str, tuple] = {
    "camera_l": (_I3, _Z3),
}


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3); ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(ax * np.deg2rad(
        np.random.uniform(-rot_noise_deg, rot_noise_deg))).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Pose model ────────────────────────────────────────────────────────────────
class PoseModel(nn.Module):
    def __init__(self, mesh, renderer, image_ref_np, init_R, init_t,
                 fx, fy, cx, cy, H, W):
        super().__init__()
        self.mesh = mesh;  self.renderer = renderer
        self.fx = fx;  self.fy = fy
        self.cx = cx;  self.cy = cy
        self.H  = H;   self.W  = W
        self.cent_views = []
        self.cam_views  = []

        if image_ref_np.ndim == 2:
            fg_mask = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
        else:
            ref_t = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
            br    = ref_t.mean(-1) / ref_t.max().clamp(min=1e-6)
            fg_mask = (br > FG_THRESHOLD).float() if BRIGHT_FG \
                      else (br < (1.0 - FG_THRESHOLD)).float()
        self.register_buffer("image_ref", fg_mask)

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


# ── Visualization renderer (full resolution, created once) ────────────────────
fkey  = str(frame_id)
cam_K = scene_cam[fkey]["cam_K"]
_viz_renderer = make_silhouette_renderer(cam_K[0], cam_K[4], cam_K[2], cam_K[5],
                                         img_H, img_W, sigma=0.005)


def draw_pose_overlay(rgb_pil, R_bop, t_bop, cam_K, mesh,
                      color_sil=(0, 220, 0)) -> Image.Image:
    W, H = rgb_pil.size
    R_pt3d, T_pt3d = bop_to_pt3d(np.array(R_bop), np.array(t_bop))
    R_t = torch.tensor(R_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    T_t = torch.tensor(T_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    cam = make_cameras(R_t, T_t, cam_K[0], cam_K[4], cam_K[2], cam_K[5], H, W)
    with torch.no_grad():
        alpha = np.from_dlpack(
            _viz_renderer(meshes_world=mesh.clone(), cameras=cam)[0, ..., 3].detach().cpu())

    overlay = np.zeros((*alpha.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    rgb_arr = np.array(rgb_pil, dtype=np.float32)
    blended = np.clip(0.5 * alpha[..., None] * overlay + rgb_arr, 0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)

    draw = ImageDraw.Draw(result)
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    def proj(p):
        pc = np.array(R_bop) @ np.array(p) + np.array(t_bop)
        return (int(round(fx * pc[0]/pc[2] + cx)), int(round(fy * pc[1]/pc[2] + cy))) \
               if pc[2] > 0 else None
    L   = 50.0
    cen = proj([0, 0, 0])
    for end, col in zip([proj([L, 0, 0]), proj([0, L, 0]), proj([0, 0, L])],
                        [(220, 50, 50), (50, 220, 50), (50, 50, 220)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


# ── Load frame data ────────────────────────────────────────────────────────────
rgb_path     = os.path.join(RGB_DIR, _rgb_on_disk[frame_id])
rgb_orig_pil = Image.open(rgb_path).convert("RGB")
ref_np       = np.array(rgb_orig_pil.resize((RENDER_SIZE, RENDER_SIZE), Image.BILINEAR),
                         dtype=np.float32)

obj_data = next((o for o in scene_gt[fkey] if o["obj_id"] == TARGET_OBJ_ID), None)
if obj_data is None:
    raise RuntimeError(f"obj_id={TARGET_OBJ_ID} not found in frame {frame_id}")

R_gt = np.array(obj_data["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
t_gt = np.array(obj_data["cam_t_m2c"], dtype=np.float64)
print(f"\nFrame {frame_id}, obj_id={TARGET_OBJ_ID}")

fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)

mesh = build_mesh()
sil_renderer_ref     = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=1e-4)
sil_renderer_stage1  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE1)
sil_renderer_stage2a = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2A)
sil_renderer_stage2b = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2B)
sil_renderer_stage2  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2)


# ── Build GT references once (shared across all 8 configs) ────────────────────
cent_views_s1 = []
cam_views_s2a = []   # σ=0.025 — Stage 1.5 (trans) + Stage 2a (rot MSE)
cam_views_s2b = []   # σ=0.010 — Stage 2b (intermediate IoU)
cam_views_s2c = []   # σ=0.005 — Stage 2c (fine IoU)

if USE_RENDERED_REFERENCE:
    for cam_name in CAMERA_NAMES:
        od_c = next(
            (o for o in scene_gt_all[cam_name][fkey] if o["obj_id"] == TARGET_OBJ_ID),
            None)
        if od_c is None:
            continue
        R_c, T_c = bop_to_pt3d(
            np.array(od_c["cam_R_m2c"]).reshape(3, 3),
            np.array(od_c["cam_t_m2c"]))
        Rc_t = torch.tensor(R_c, dtype=torch.float32, device=device).unsqueeze(0)
        Tc_t = torch.tensor(T_c, dtype=torch.float32, device=device).unsqueeze(0)
        cam_c = make_cameras(Rc_t, Tc_t, fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE)

        with torch.no_grad():
            r_crisp = sil_renderer_ref(    meshes_world=mesh.clone(), cameras=cam_c)
            r_2a    = sil_renderer_stage2a(meshes_world=mesh.clone(), cameras=cam_c)
            r_2b    = sil_renderer_stage2b(meshes_world=mesh.clone(), cameras=cam_c)
            r_2c    = sil_renderer_stage2( meshes_world=mesh.clone(), cameras=cam_c)

        mask_crisp = r_crisp[0, ..., 3]
        area_c = mask_crisp.sum().clamp(min=1.0)
        cx_c   = (GRID_X * mask_crisp).sum() / area_c
        cy_c   = (GRID_Y * mask_crisp).sum() / area_c

        R_rel_c, T_rel_c = cam_extrinsics[cam_name]
        cent_views_s1.append((cx_c, cy_c, area_c, R_rel_c, T_rel_c))
        cam_views_s2a.append((r_2a[0, ..., 3].detach().clone(), R_rel_c, T_rel_c))
        cam_views_s2b.append((r_2b[0, ..., 3].detach().clone(), R_rel_c, T_rel_c))
        cam_views_s2c.append((r_2c[0, ..., 3].detach().clone(), R_rel_c, T_rel_c))

    ref_for_model = np.from_dlpack(r_crisp[0, ..., 3].detach().cpu())
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


# ── Ablation loop ──────────────────────────────────────────────────────────────
csv_rows = []

for cfg in CONFIGS:
    label         = cfg["label"]
    rot_deg       = cfg["rot_deg"]
    trans_mm      = cfg["trans_mm"]
    use_gt_anchor = cfg["use_gt_anchor"]

    print(f"\n══ Config: {label}  (rot_noise={rot_deg}°, trans_noise={trans_mm}mm) ══")

    np.random.seed(FIXED_SEED)
    if use_gt_anchor:
        init_R, init_t = perturb_pose(R_gt, t_gt, rot_deg, trans_mm)
    else:
        init_R = np.eye(3, dtype=np.float64)
        init_t = np.from_dlpack(obj_mean_t[TARGET_OBJ_ID].detach().cpu())

    init_rot_err   = rotation_error_deg(init_R, R_gt)
    init_trans_err = translation_error_mm(init_t, t_gt)
    print(f"  Init: rot={init_rot_err:.2f}°  trans={init_trans_err:.2f}mm")

    # ── Build model ────────────────────────────────────────────────────────────
    init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
    model = PoseModel(
        mesh, sil_renderer_stage1, ref_for_model,
        init_R_pt3d, init_t_pt3d,
        fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
    ).to(device)

    # ── Stage 1: translation only, centroid+area ───────────────────────────────
    model.cent_views   = cent_views_s1
    model.use_centroid = True
    model.use_iou      = False
    opt_s1 = torch.optim.Adam([model.transl], lr=LR_TRANS)
    loop_s1 = tqdm(range(NUM_ITERS_TRANS), desc=f"  s1 {label}")
    for _ in loop_s1:
        opt_s1.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1.step()
        loop_s1.set_description(f"  s1 {label} ({loss.item():.3f})")
    print(f"  Stage 1 done: loss={loss.item():.4f}")

    # ── Stage 1.5: translation-only silhouette MSE (refines depth) ────────────
    model.renderer     = sil_renderer_stage2a
    model.cam_views    = cam_views_s2a
    model.use_centroid = False
    model.use_iou      = False
    opt_s1b = torch.optim.Adam([model.transl], lr=LR_TRANS_S1B)
    loop_s1b = tqdm(range(NUM_ITERS_TRANS_S1B), desc=f"  s1b {label}")
    for _ in loop_s1b:
        opt_s1b.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1b.step()
        loop_s1b.set_description(f"  s1b {label} ({loss.item():.1f})")
    print(f"  Stage 1.5 done: loss={loss.item():.2f}  (σ={SIGMA_STAGE1B})")

    # ── Stage 2a: coarse rotation — MSE only (stable gradient) ────────────────
    model.renderer     = sil_renderer_stage2a
    model.cam_views    = cam_views_s2a
    model.use_centroid = False
    model.use_iou      = False
    opt_s2a = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_COARSE},
        {"params": [model.transl], "lr": LR_TRANS * 0.1},
    ])
    loop_s2a = tqdm(range(NUM_ITERS_COARSE), desc=f"  s2a {label}")
    for _ in loop_s2a:
        opt_s2a.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2a.step()
        loop_s2a.set_description(f"  s2a {label} ({loss.item():.1f})")
    print(f"  Stage 2a done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2A})")

    # ── Stage 2b: intermediate sigma — bridges 0.025→0.005 gap ────────────────
    model.renderer   = sil_renderer_stage2b
    model.cam_views  = cam_views_s2b
    model.use_iou    = True
    model.iou_weight = IOU_WEIGHT_MID
    opt_s2b = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT_MID},
        {"params": [model.transl], "lr": LR_TRANS * 0.05},
    ])
    loop_s2b = tqdm(range(NUM_ITERS_MID), desc=f"  s2b {label}")
    for _ in loop_s2b:
        opt_s2b.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2b.step()
        loop_s2b.set_description(f"  s2b {label} ({loss.item():.1f})")
    print(f"  Stage 2b done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2B})")

    # ── Stage 2c: fine rotation — tight sigma + full IoU ──────────────────────
    model.renderer   = sil_renderer_stage2
    model.cam_views  = cam_views_s2c
    model.use_iou    = True
    model.iou_weight = IOU_WEIGHT
    opt_s2c = torch.optim.Adam([
        {"params": [model.rot6d],  "lr": LR_ROT},
        {"params": [model.transl], "lr": LR_TRANS * 0.02},
    ])
    loop_s2c = tqdm(range(NUM_ITERS_JOINT), desc=f"  s2c {label}")
    for _ in loop_s2c:
        opt_s2c.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2c.step()
        loop_s2c.set_description(f"  s2c {label} ({loss.item():.1f})")
    print(f"  Stage 2c done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2})")

    with torch.no_grad():
        R_est_t = rot6d_to_matrix(model.rot6d)
        T_est_t = model.transl
    R_est_pt3d_np = np.from_dlpack(R_est_t.detach().cpu())
    T_est_pt3d_np = np.from_dlpack(T_est_t.detach().cpu())
    R_est_np, T_est_np = pt3d_to_bop(R_est_pt3d_np, T_est_pt3d_np)

    rot_err   = rotation_error_deg(R_est_np, R_gt)
    trans_err = translation_error_mm(T_est_np, t_gt)
    add_val   = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)
    print(f"  rot={rot_err:.2f}°  trans={trans_err:.2f}mm  ADD={add_val:.2f}mm")

    # ── Visualization: GT (green) | estimate (orange) side by side ────────────
    try:
        gt_ov  = draw_pose_overlay(rgb_orig_pil, R_gt,     t_gt,     cam_K, mesh, (0, 220, 0))
        est_ov = draw_pose_overlay(rgb_orig_pil, R_est_np, T_est_np, cam_K, mesh, (220, 80, 0))
        LH = 28
        combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
        combined.paste(gt_ov,  (0,     LH))
        combined.paste(est_ov, (img_W, LH))
        d = ImageDraw.Draw(combined)
        d.text((8,         5), f"GT  frame={frame_id:04d}  obj={TARGET_OBJ_ID}  [{label}]",
               fill=(100, 255, 100))
        d.text((img_W + 8, 5),
               f"EST  rot={rot_err:.1f}°  trans={trans_err:.1f}mm  ADD={add_val:.1f}mm",
               fill=(255, 180, 80))
        out = os.path.join(VIZ_DIR, f"viz_{label}.png")
        combined.save(out)
        print(f"  viz → {out}")
    except Exception as e:
        print(f"  viz skipped: {e}")

    csv_rows.append({
        "config":                label,
        "rot_noise_deg":         rot_deg,
        "trans_noise_mm":        trans_mm,
        "init_rot_error_deg":    init_rot_err,
        "init_trans_error_mm":   init_trans_err,
        "est_R_00": R_est_np[0, 0], "est_R_01": R_est_np[0, 1], "est_R_02": R_est_np[0, 2],
        "est_R_10": R_est_np[1, 0], "est_R_11": R_est_np[1, 1], "est_R_12": R_est_np[1, 2],
        "est_R_20": R_est_np[2, 0], "est_R_21": R_est_np[2, 1], "est_R_22": R_est_np[2, 2],
        "est_tx_mm": T_est_np[0],   "est_ty_mm": T_est_np[1],   "est_tz_mm": T_est_np[2],
        "gt_R_00": R_gt[0, 0], "gt_R_01": R_gt[0, 1], "gt_R_02": R_gt[0, 2],
        "gt_R_10": R_gt[1, 0], "gt_R_11": R_gt[1, 1], "gt_R_12": R_gt[1, 2],
        "gt_R_20": R_gt[2, 0], "gt_R_21": R_gt[2, 1], "gt_R_22": R_gt[2, 2],
        "gt_tx_mm": t_gt[0],   "gt_ty_mm": t_gt[1],   "gt_tz_mm": t_gt[2],
        "rot_error_deg":   rot_err,
        "trans_error_mm":  trans_err,
        "ADD_mm":          add_val,
    })


# ── Write CSV ──────────────────────────────────────────────────────────────────
if csv_rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader();  w.writerows(csv_rows)
    print(f"\nCSV → {OUTPUT_CSV}  ({len(csv_rows)} rows)")

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  {'config':<25s}  {'init_rot':>8s}  {'init_trans':>10s}  "
          f"{'rot_err':>8s}  {'trans_err':>10s}  {'ADD':>8s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for r in csv_rows:
        print(f"  {r['config']:<25s}  "
              f"{r['init_rot_error_deg']:>7.2f}°  "
              f"{r['init_trans_error_mm']:>9.2f}mm  "
              f"{r['rot_error_deg']:>7.2f}°  "
              f"{r['trans_error_mm']:>9.2f}mm  "
              f"{r['ADD_mm']:>7.2f}mm")
    print("─────────────────────────────────────────────────────────────────────")
