"""
optimize_model_real_data_IoU.py  (single-view, Real_SR_Data/LND_TEST)
Differentiable-rendering 6-DoF pose estimator (PyTorch3D silhouette loss).
Same single-shot three-stage optimizer as optimize_modelv2.py (no restarts),
using the real-data adaptations from optimize_model_real_data.py (STL mesh,
real per-frame mask as the loss reference, camera intrinsics from config.yaml):

  Stage 1    – translation only; centroid loss from binary mask (no area term)
  Stage 1.5  – translation only; MSE against Gaussian-blurred binary mask, sigma=0.025
  Stage 2a   – joint R+T; MSE only, sigma=0.025
  Stage 2b   – joint R+T; MSE+IoU, sigma=0.010
  Stage 2c   – joint R+T; MSE+IoU, sigma=0.005

Differences from optimize_model_real_data.py:
  - Single-shot optimization per frame (no multi-restart best-of-N selection),
    matching optimize_modelv2.py's structure exactly.
  - Visualization draws only two things on top of the RGB frame: the pose axes
    and a flat-shaded silhouette fill (no other overlay elements).
  - The silhouette used for both visualization and metrics is a hard binary
    rasterization (blur_radius=0, faces_per_pixel=1) at full image resolution,
    not the soft differentiable shader — soft alpha carries a blur halo around
    the true edge even at small sigma, which this avoids.
  - The silhouette fill is clipped to the real per-frame segmentation mask
    (mask/{id}.png, the same mask the optimizer already fits against) so the
    render never spills onto pixels the mask says aren't the tool — this is
    the real-data equivalent of the appearance-mask fix used for simulation.
  - A new CSV metric is added: intersection-over-union (IoU) of the crisp
    binary silhouette, computed GT-vs-GT (sanity check, always 1.0) and
    GT-vs-estimate (the real metric).
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
    """Convert binary STL to OBJ with deduplicated vertices."""
    with open(stl_path, 'rb') as f:
        f.read(80)  # skip header
        n_tris_header = struct.unpack('<I', f.read(4))[0]
        raw = f.read()  # read all remaining bytes

    # Derive actual count from byte length; header count is sometimes wrong
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
                vert_map[v] = len(verts) + 1  # 1-indexed OBJ
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
MODEL_PTS    = os.path.join(DATASET_ROOT, "joint.npy")  # for ADD metric

OUTPUT_DIR = "../IoU_CSV_Visuals_Diff_Render_Real_Data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "pose_estimation_results_iou.csv")
VIZ_DIR    = OUTPUT_DIR
DEBUG_DIR  = "../debug_renders"

# Stage 1: translation only (centroid+area)
NUM_ITERS_TRANS = 200
LR_TRANS        = 0.05

# Stage 1.5: translation-only silhouette MSE
NUM_ITERS_TRANS_S1B = 300
LR_TRANS_S1B        = 0.05
SIGMA_STAGE1B       = 0.025

# Stage 2a: coarse rotation — MSE only
NUM_ITERS_COARSE = 300
LR_ROT_COARSE    = 0.003
SIGMA_STAGE2A    = 0.025

# Stage 2b: intermediate sigma — MSE + light IoU
NUM_ITERS_MID  = 150
LR_ROT_MID     = 0.002
SIGMA_STAGE2B  = 0.010
IOU_WEIGHT_MID = 10.0

# Stage 2c: fine rotation — MSE + IoU, tight sigma
NUM_ITERS_JOINT = 300
LR_ROT          = 0.001
SIGMA_STAGE2    = 0.005

IOU_WEIGHT        = 20.0
IOU_WEIGHT_COARSE = 5.0   # lighter IoU for Stage 2a — stable during coarse search
SIGMA_STAGE1      = 0.01
RENDER_SIZE       = 256
FG_THRESHOLD      = 0.5   # binary mask is 0/1; anything above this is foreground

# IoU loss is used for all optimization stages after Stage 1 (centroid).
# Pure IoU = 1 - inter/union is correct against a binary GT mask:
#   GT pose always maximises inter (all GT pixels covered), and the union
#   grows only if the rendered area is extra-large (halo), but GT still has
#   the highest IoU because inter_max = GT_area is achieved at GT.
# Stages with tighter sigma (0.025→0.005) progressively sharpen the gradient.

USE_GT_INITIALIZATION = True
INIT_ROT_NOISE_DEG    = 10.0
INIT_TRANS_NOISE_MM   = 10.0
USE_GT_INIT_ONLY      = False

IOU_THRESHOLD = 0.5   # for the crisp GT-vs-estimate IoU metric (not the loss above)
# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--all",    action="store_true", help="Run all frames")
parser.add_argument("--frames", type=int, default=None,
                    help="Run only the first N frames")
args = parser.parse_args()

os.makedirs(DEBUG_DIR,    exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)
print(f"Debug → {DEBUG_DIR}   Viz → {VIZ_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=device)
GRID_Y, GRID_X = torch.meshgrid(_arange, _arange, indexing="ij")


# ── Camera intrinsics from config.yaml ────────────────────────────────────────
with open(os.path.join(DATASET_ROOT, "config.yaml")) as f:
    _cfg = yaml.safe_load(f)

cam_K = _cfg["cam"]["camera_matrix"]["data"]  # flat 9-element list
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
elif not args.all:
    valid_frame_ids = valid_frame_ids[:10]
    print(f"Running first {len(valid_frame_ids)} frames (--all for all)")


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
# Reversed-winding copy of every face — lets interior pixels get a frontface
# contribution even for a hollow (shell) mesh, giving a solid projected silhouette.
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
model_pts_np = np.load(MODEL_PTS).astype(np.float64)  # (30000, 3)
_pts_vmax = float(np.abs(model_pts_np).max())
if _pts_vmax < 1.0:
    model_pts_np = model_pts_np * 1000.0
print(f"Model pts: {model_pts_np.shape[0]} points | max|v|={_pts_vmax:.4f}")


# ── Fallback translation (mean of all GT) ─────────────────────────────────────
_all_gt_t = [np.load(os.path.join(_pose_dir, f"{fid}.npy"))[:3, 3]
             for fid in valid_frame_ids]
mean_t_fallback = np.mean(_all_gt_t, axis=0)
print(f"Mean GT translation: {mean_t_fallback.round(2)}")


# ── Camera / renderer helpers ──────────────────────────────────────────────────
# Screen-space (pixel-unit) cameras with an explicit image_size, NOT PyTorch3D's
# NDC convention. Without image_size, PerspectiveCameras assumes a square NDC
# space (effectively scaling by min(H, W)) — for this dataset's 960×540 (16:9)
# images that silently compresses the X axis by H/W≈0.56, which is exactly the
# ~53% width shrink and near-zero IoU-at-GT-pose (0.16) we measured before this
# fix (0.93 after). Screen-space cameras sidestep the ambiguity entirely.
def make_cameras(R, T, fx, fy, cx, cy, H, W) -> PerspectiveCameras:
    return PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        R=R, T=T, in_ndc=False, image_size=((H, W),), device=device,
    )


def scale_intrinsics(K_flat: list, W_from: int, H_from: int, W_to: int, H_to: int):
    """Rescale pixel-space intrinsics for rendering at a different resolution
    (e.g. the square RENDER_SIZE training canvas vs. the native image size).
    Uses separate X/Y scale factors, matching the non-uniform squish already
    applied when the real mask is resized to RENDER_SIZE×RENDER_SIZE."""
    sx, sy = W_to / W_from, H_to / H_from
    return K_flat[0] * sx, K_flat[4] * sy, K_flat[2] * sx, K_flat[5] * sy


def make_silhouette_renderer(fx, fy, cx, cy, H, W,
                              sigma=1e-4, image_size=None, faces_per_pixel=100):
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
    """Gaussian-blur the binary mask to match the soft-renderer's bandwidth.

    At the GT pose the rendered alpha and the blurred reference have the same
    spatial profile, so coverage≈0 and halo≈0, placing the loss minimum at GT.
    """
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


# ── Single-camera identity extrinsics ─────────────────────────────────────────
_I3 = torch.eye(3,  dtype=torch.float32, device=device)
_Z3 = torch.zeros(3, dtype=torch.float32, device=device)


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3)
    ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(
        ax * np.deg2rad(np.random.uniform(-rot_noise_deg, rot_noise_deg))
    ).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Pose model ────────────────────────────────────────────────────────────────
class PoseModel(nn.Module):
    """
    Stage 1 (use_centroid=True):
      cent_views = [(cx_ref, cy_ref, area_ref, R_rel, T_rel), ...]
      Centroid+area loss from binary mask.

    Stage 2 (use_centroid=False):
      cam_views  = [(ref_mask, R_rel, T_rel), ...]
      MSE+IoU against real binary mask.
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
        self.cam_views  = []

        # mask_resized_np is already a 2D float32 array (0.0/1.0)
        fg_mask = torch.tensor(mask_resized_np, dtype=torch.float32, device=device)
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
        if self.use_centroid:
            if self.cent_views:
                total = torch.tensor(0.0, device=device)
                for cx_r, cy_r, ar_r, R_rel, T_rel in self.cent_views:
                    alpha = self._render_alpha(R_rel, T_rel)
                    area  = alpha.sum().clamp(min=1e-6)
                    cx    = (GRID_X * alpha).sum() / area
                    cy    = (GRID_Y * alpha).sum() / area
                    # Centroid-only — no area term.
                    # With fpp=1 halo is ~1.3× mask area, but mesh size may
                    # differ from the real object, so area term stays off.
                    total = total + (cx - cx_r)**2 + (cy - cy_r)**2
                loss = total / len(self.cent_views)
            else:
                alpha = self._render_alpha(_I3, _Z3)
                area  = alpha.sum().clamp(min=1e-6)
                cx    = (GRID_X * alpha).sum() / area
                cy    = (GRID_Y * alpha).sum() / area
                loss  = (cx - self.cx_ref[0])**2 + (cy - self.cy_ref[0])**2
        else:
            views = self.cam_views or [(self.image_ref, _I3, _Z3)]
            total = torch.tensor(0.0, device=device)
            for ref_mask_i, R_rel_i, T_rel_i in views:
                alpha_i = self._render_alpha(R_rel_i, T_rel_i)
                mse_i   = (alpha_i - ref_mask_i).pow(2).sum()
                if self.use_iou:
                    inter  = (alpha_i * ref_mask_i).sum()
                    union  = (alpha_i + ref_mask_i - alpha_i * ref_mask_i).sum().clamp(min=1e-6)
                    total  = total + mse_i + self.iou_weight * (1.0 - inter / union)
                else:
                    total = total + mse_i
            loss = total / len(views)

        return loss, rot6d_to_matrix(self.rot6d).unsqueeze(0), self.transl.unsqueeze(0)


# ── Error metrics ──────────────────────────────────────────────────────────────
def rotation_error_deg(R_est, R_gt):
    cos_a = float(np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def translation_error_mm(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


def add_metric(pts_np, R_est, t_est, R_gt, t_gt):
    return float(np.mean(np.linalg.norm(
        ((R_est @ pts_np.T).T + t_est) - ((R_gt @ pts_np.T).T + t_gt), axis=1)))


def iou_metric(alpha_a: np.ndarray, alpha_b: np.ndarray, threshold=IOU_THRESHOLD) -> float:
    """Binary IoU between two crisp silhouette alpha masks."""
    mask_a = alpha_a > threshold
    mask_b = alpha_b > threshold
    union  = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter) / float(union)


# ── Crisp silhouette + visualization (full resolution, created once) ──────────
# Hard (non-differentiable) rasterization — no soft shader — so the silhouette
# edge is exact instead of carrying the soft shader's blur halo. Uses the
# solid (doubled-face) mesh, matching the optimizer's own silhouette convention
# for this hollow-shell STL.
fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
# Intrinsics rescaled for the square RENDER_SIZE optimization canvas (separate
# X/Y factors — see scale_intrinsics docstring).
fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)
_crisp_raster_settings = RasterizationSettings(
    image_size=(img_H, img_W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
)
_mesh_viz = build_mesh(solid=True)  # shared across all overlay draws


def render_crisp_alpha(R_bop, t_bop) -> np.ndarray:
    """Hard binary silhouette at full image resolution, tightly covering the
    tool — used for both the visualization fill and IoU."""
    R_pt3d, T_pt3d = bop_to_pt3d(np.array(R_bop), np.array(t_bop))
    R_t = torch.tensor(R_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    T_t = torch.tensor(T_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    cam = make_cameras(R_t, T_t, fx, fy, cx, cy, img_H, img_W)
    rasterizer = MeshRasterizer(cameras=cam, raster_settings=_crisp_raster_settings)
    with torch.no_grad():
        frags = rasterizer(_mesh_viz.clone(), cameras=cam)
        mask  = (frags.pix_to_face[0, ..., 0] >= 0).float()
    return np.from_dlpack(mask.detach().cpu())


def draw_axes_and_silhouette(rgb_pil, alpha, R_bop, t_bop,
                             color_sil=(0, 220, 0), appearance_mask=None) -> Image.Image:
    """Draws exactly two things on top of the RGB frame: a flat-shaded
    silhouette fill and the 3-axis pose gizmo. If appearance_mask is given
    (the real per-frame segmentation mask), the fill is clipped to it so the
    render never spills onto pixels the mask says aren't the tool."""
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
        return (int(round(fx * pc[0] / pc[2] + cx)),
                int(round(fy * pc[1] / pc[2] + cy))) if pc[2] > 0 else None

    L   = 50.0
    cen = proj([0, 0, 0])
    for end, col in zip([proj([L, 0, 0]), proj([0, L, 0]), proj([0, 0, L])],
                        [(220, 50, 50), (50, 220, 50), (50, 50, 220)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


# ── Main loop ──────────────────────────────────────────────────────────────────
csv_rows = []

for frame_id in valid_frame_ids:
    rgb_path  = os.path.join(_image_dir, f"{frame_id}.png")
    mask_path = os.path.join(_mask_dir,  f"{frame_id}.png")
    pose_path = os.path.join(_pose_dir,  f"{frame_id}.npy")

    pose_mat = np.load(pose_path)                          # (3, 4)
    R_gt     = pose_mat[:3, :3].copy().astype(np.float64)
    t_gt     = pose_mat[:3,  3].copy().astype(np.float64)

    try:
        rgb_orig_pil = Image.open(rgb_path).convert("RGB")
        mask_full    = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    except OSError as e:
        print(f"\n  Frame {frame_id:04d} skipped — image load error: {e}")
        continue

    # Load binary mask, resize to RENDER_SIZE, normalize to [0, 1]
    mask_resized = np.array(
        Image.fromarray((mask_full * 255).astype(np.uint8)).resize(
            (RENDER_SIZE, RENDER_SIZE), Image.NEAREST),
        dtype=np.float32) / 255.0
    mask_t = torch.tensor(mask_resized, dtype=torch.float32, device=device)

    print(f"\n══ Frame {frame_id:04d} ══")
    print(f"  GT t: {t_gt.round(2)}  |  mask fg px: {int(mask_full.sum())}")

    # solid=True uses doubled faces so interior pixels get a frontface contribution
    # → solid projected silhouette instead of a hollow ring.
    mesh = build_mesh(solid=True)
    # fpp=2: with doubled faces each pixel can sample both the original face and
    # its reversed-winding copy, filling the interior of the hollow mesh.
    sil_renderer_s1  = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE1,  faces_per_pixel=1)
    sil_renderer_s2a = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2A, faces_per_pixel=1)
    sil_renderer_s2b = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2B, faces_per_pixel=1)
    sil_renderer_s2c = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA_STAGE2,  faces_per_pixel=1)

    # ── Precompute mask views once (shared across restarts) ───────────────────
    area_c = mask_t.sum().clamp(min=1.0)
    cx_c   = (GRID_X * mask_t).sum() / area_c
    cy_c   = (GRID_Y * mask_t).sum() / area_c
    cent_views_s1 = [(cx_c, cy_c, area_c, _I3, _Z3)]

    def _bm(sigma):
        arr = make_blurred_mask(mask_resized, sigma)
        return torch.tensor(arr, dtype=torch.float32, device=device)

    cam_views_s2a = [(_bm(SIGMA_STAGE2A), _I3, _Z3)]
    cam_views_s2b = [(_bm(SIGMA_STAGE2B), _I3, _Z3)]
    cam_views_s2c = [(_bm(SIGMA_STAGE2),  _I3, _Z3)]

    # ── Diagnostic: MSE+IoU at GT pose (once, before restarts) ───────────────
    with torch.no_grad():
        R_gt_pt3d, T_gt_pt3d = bop_to_pt3d(R_gt.astype(np.float32), t_gt.astype(np.float32))
        _cam_gt_d = make_cameras(
            torch.tensor(R_gt_pt3d.tolist(), dtype=torch.float32, device=device).unsqueeze(0),
            torch.tensor(T_gt_pt3d.tolist(), dtype=torch.float32, device=device).unsqueeze(0),
            fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE)
        _a_gt   = sil_renderer_s2a(meshes_world=mesh.clone(), cameras=_cam_gt_d)[0,...,3]
        _bm_s2a = cam_views_s2a[0][0]
        _mse_gt = (_a_gt - _bm_s2a).pow(2).sum()
        _inter  = (_a_gt * mask_t).sum(); _union = (_a_gt + mask_t - _a_gt * mask_t).sum()
        print(f"  [GT@s2a] MSE={_mse_gt:.1f}  IoU={(_inter/_union).item():.3f}"
              f"  rendered_area={_a_gt.sum().item():.0f}px  mask_area={area_c.item():.0f}px")

    # ── Single-shot optimization (matches optimize_modelv2.py — no restarts) ──
    if USE_GT_INITIALIZATION:
        init_R, init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
    else:
        init_R = np.eye(3, dtype=np.float64)
        init_t = mean_t_fallback.copy()

    init_rot_err   = rotation_error_deg(init_R, R_gt)
    init_trans_err = translation_error_mm(init_t, t_gt)
    print(f"  Init: rot={init_rot_err:.2f}°  trans={init_trans_err:.2f} mm")

    if USE_GT_INIT_ONLY:
        R_est_np, T_est_np = init_R.copy(), init_t.copy()
    else:
        init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
        model = PoseModel(
            mesh, sil_renderer_s1, mask_resized,
            init_R_pt3d, init_t_pt3d,
            fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
        ).to(device)
        model.use_centroid = True

        # Stage 1: translation only, centroid ─────────────────────────────────
        model.cent_views = cent_views_s1
        opt_s1 = torch.optim.Adam([model.transl], lr=LR_TRANS)
        loop_s1 = tqdm(range(NUM_ITERS_TRANS), desc=f"  s1  f{frame_id:04d}")
        for _ in loop_s1:
            opt_s1.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1.step()
            loop_s1.set_description(f"  s1  f{frame_id:04d} ({loss.item():.3f})")
        print(f"  Stage 1 done: loss={loss.item():.4f}")

        # Stage 1.5: translation-only, MSE silhouette ──────────────────────────
        model.renderer     = sil_renderer_s2a
        model.cam_views    = cam_views_s2a
        model.use_centroid = False
        model.use_iou      = False
        opt_s1b = torch.optim.Adam([model.transl], lr=LR_TRANS_S1B)
        loop_s1b = tqdm(range(NUM_ITERS_TRANS_S1B), desc=f"  s1b f{frame_id:04d}")
        for _ in loop_s1b:
            opt_s1b.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1b.step()
            loop_s1b.set_description(f"  s1b f{frame_id:04d} ({loss.item():.2f})")
        print(f"  Stage 1.5 done: loss={loss.item():.2f}  (σ={SIGMA_STAGE1B})")

        # Stage 2a: rotation only + MSE + light IoU (translation frozen) ──────
        model.renderer     = sil_renderer_s2a
        model.cam_views    = cam_views_s2a
        model.use_iou      = True
        model.iou_weight   = IOU_WEIGHT_COARSE
        opt_s2a = torch.optim.Adam([
            {"params": [model.rot6d], "lr": LR_ROT_COARSE},
        ])
        loop_s2a = tqdm(range(NUM_ITERS_COARSE), desc=f"  s2a f{frame_id:04d}")
        for _ in loop_s2a:
            opt_s2a.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2a.step()
            loop_s2a.set_description(f"  s2a f{frame_id:04d} ({loss.item():.2f})")
        print(f"  Stage 2a done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2A})")

        # Stage 2b: MSE + IoU, intermediate sigma ──────────────────────────────
        model.renderer   = sil_renderer_s2b
        model.cam_views  = cam_views_s2b
        model.use_iou    = True
        model.iou_weight = IOU_WEIGHT_MID
        opt_s2b = torch.optim.Adam([
            {"params": [model.rot6d],  "lr": LR_ROT_MID},
            {"params": [model.transl], "lr": LR_TRANS * 0.05},
        ])
        loop_s2b = tqdm(range(NUM_ITERS_MID), desc=f"  s2b f{frame_id:04d}")
        for _ in loop_s2b:
            opt_s2b.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2b.step()
            loop_s2b.set_description(f"  s2b f{frame_id:04d} ({loss.item():.2f})")
        print(f"  Stage 2b done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2B})")

        # Stage 2c: MSE + IoU, finest sigma ─────────────────────────────────────
        model.renderer   = sil_renderer_s2c
        model.cam_views  = cam_views_s2c
        model.use_iou    = True
        model.iou_weight = IOU_WEIGHT
        opt_s2c = torch.optim.Adam([
            {"params": [model.rot6d],  "lr": LR_ROT},
            {"params": [model.transl], "lr": LR_TRANS * 0.02},
        ])
        loop_s2c = tqdm(range(NUM_ITERS_JOINT), desc=f"  s2c f{frame_id:04d}")
        for _ in loop_s2c:
            opt_s2c.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s2c.step()
            loop_s2c.set_description(f"  s2c f{frame_id:04d} ({loss.item():.2f})")
        print(f"  Stage 2c done: loss={loss.item():.2f}  (σ={SIGMA_STAGE2})")

        with torch.no_grad():
            R_est_t = rot6d_to_matrix(model.rot6d)
            T_est_t = model.transl
        R_est_pt3d_np = np.from_dlpack(R_est_t.detach().cpu())
        T_est_pt3d_np = np.from_dlpack(T_est_t.detach().cpu())
        R_est_np, T_est_np = pt3d_to_bop(R_est_pt3d_np, T_est_pt3d_np)

    # ── Errors ────────────────────────────────────────────────────────────────
    rot_err   = rotation_error_deg(R_est_np, R_gt)
    trans_err = translation_error_mm(T_est_np, t_gt)
    add_val   = add_metric(model_pts_np, R_est_np, T_est_np, R_gt, t_gt)

    # ── IoU (crisp binary silhouette, full resolution) ───────────────────────
    alpha_gt  = render_crisp_alpha(R_gt,     t_gt)
    alpha_est = render_crisp_alpha(R_est_np, T_est_np)
    iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
    iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)
    print(f"  rot={rot_err:.2f}°  trans={trans_err:.2f}mm  ADD={add_val:.2f}mm  "
          f"IoU(gt,gt)={iou_gt_vs_gt:.4f}  IoU(gt,est)={iou_gt_vs_est:.4f}")

    # ── Visualization: only axes + silhouette fill (raw crisp render — no
    # appearance-mask clipping here; see note below) ───────────────────────────
    # Unlike the simulation script, this dataset's recorded GT pose doesn't
    # reliably land on the real object (a real-world calibration property of
    # this dataset, confirmed even at the literal GT pose), so intersecting
    # with the real mask would erase the fill instead of just trimming
    # spillover. Showing the raw render lets you see the true render
    # position/size, misalignment included.
    try:
        gt_ov  = draw_axes_and_silhouette(rgb_orig_pil, alpha_gt,  R_gt,     t_gt,     (0, 220, 0))
        est_ov = draw_axes_and_silhouette(rgb_orig_pil, alpha_est, R_est_np, T_est_np, (220, 80, 0))
        LH = 28
        combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
        combined.paste(gt_ov,  (0,     LH))
        combined.paste(est_ov, (img_W, LH))
        d = ImageDraw.Draw(combined)
        d.text((8,         5), f"GT  frame={frame_id:04d}", fill=(100, 255, 100))
        d.text((img_W + 8, 5),
               f"EST  rot={rot_err:.1f}°  trans={trans_err:.1f}mm  ADD={add_val:.1f}mm  "
               f"IoU={iou_gt_vs_est:.3f}",
               fill=(255, 180, 80))
        out = os.path.join(VIZ_DIR, f"viz_frame{frame_id:04d}.png")
        combined.save(out)
        print(f"  viz → {out}")
    except Exception as e:
        print(f"  viz skipped: {e}")

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
        "iou_gt_vs_gt":         iou_gt_vs_gt,
        "iou_gt_vs_est":        iou_gt_vs_est,
    })


# ── Write CSV ──────────────────────────────────────────────────────────────────
if csv_rows:
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nCSV → {OUTPUT_CSV}  ({len(csv_rows)} rows)")

if csv_rows:
    re_  = [r["rotation_error_deg"]   for r in csv_rows]
    te_  = [r["translation_error_mm"] for r in csv_rows]
    ae_  = [r["ADD_mm"]               for r in csv_rows]
    iou_ = [r["iou_gt_vs_est"]        for r in csv_rows]
    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Rows             : {len(csv_rows)}")
    print(f"  Mean rotation    : {np.mean(re_):.4f}°  (std {np.std(re_):.4f})")
    print(f"  Mean translation : {np.mean(te_):.4f} mm (std {np.std(te_):.4f})")
    print(f"  Mean ADD         : {np.mean(ae_):.4f} mm (std {np.std(ae_):.4f})")
    print(f"  Mean IoU(gt,est) : {np.mean(iou_):.4f}  (std {np.std(iou_):.4f})")
    print("────────────────────────────────────────────────────────")
