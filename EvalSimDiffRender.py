"""
EvalSimDiffRender.py — master evaluation entry point for the differentiable-
rendering pose optimizer on simulated data
(SIMPLE_LND_Multi_Dist_Angle/dist_0.04).

Each distance folder holds ONE scene instant captured by 5 cameras
(camera_0..camera_4) — scene_gt.json / scene_camera.json are keyed by camera
name, not frame index. This script runs a full single-view optimization
(three-stage translation/rotation schedule, PyTorch3D silhouette loss —
identical per-item pipeline to optimize_model_inter_over_union.py) once per
camera, treating each of the 5 camera images as its own independent
single-view pose estimate (not fused across views).

Does not write the per-item CSV or the side-by-side GT/estimate PNGs.
Instead, every camera/object's (pose_pred, pose_gt) pair is fed into
evaluate.py's Evaluator, and the run ends with Evaluator.summarize() — proj2d
/ ADD / ADD-S / 5mm-5deg / mean rotation & translation error — printed to the
console and saved to OUTPUT_SUMMARY as an .npy file.

evaluate.py's Evaluator expects a joint.npy (mm-scale model point cloud) on
disk and uses a fixed camera matrix for instrument_type='LND' (not this
dataset's actual per-camera scene_camera.json K). The mesh here is already
mm-scaled by the loader below, so a copy is written to a scratch directory
at startup.
"""

import os, re, json, tempfile, textwrap, datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)

from evaluate import Evaluator, model_diameter


# ─────────────────────────── CONFIG ───────────────────────────────────────────
DATASET_ROOT = "../SIMPLE_LND_Multi_Dist_Angle"
DIST         = "dist_0.14"   # hardcoded for now — edit to point at another distance folder
SCENE_DIR    = os.path.join(DATASET_ROOT, DIST, "000001")
RGB_DIR      = os.path.join(SCENE_DIR, "rgb")

MESH_PATH      = ("../surgical_robotics_challenge/ADF/PSMs/"
                   "LND_420006/high_res/tool pitch link.OBJ")
OUTPUT_SUMMARY = f"../Eval_Results_DiffRender_Sim/eval_summary_sim_dr_{DIST}.npy"
INSTRUMENT_TYPE = 'LND'

# Stage 1: translation only (single-view centroid+area)
NUM_ITERS_TRANS    = 60
LR_TRANS           = 0.05

# Stage 1.5: translation-only silhouette MSE — refines depth before rotation starts
NUM_ITERS_TRANS_S1B = 90
LR_TRANS_S1B        = 0.05
SIGMA_STAGE1B       = 0.025

# Stage 2a: coarse rotation — MSE only, no IoU
NUM_ITERS_COARSE  = 70
LR_ROT_COARSE     = 0.003
SIGMA_STAGE2A     = 0.025

# Stage 2b: intermediate sigma — MSE + light IoU bridges 0.025→0.005 gap
NUM_ITERS_MID     = 50
LR_ROT_MID        = 0.002
SIGMA_STAGE2B     = 0.010
IOU_WEIGHT_MID    = 10.0

# Stage 2c: fine rotation — MSE + IoU, tight sigma
NUM_ITERS_JOINT   = 70
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
# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=int, default=None,
                    help="Run only the first N camera images (default: all 5).")
parser.add_argument("--output_summary", type=str, default=OUTPUT_SUMMARY,
                    help="Where to save evaluate.py Evaluator.summarize() results.")
args = parser.parse_args()

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args.output_summary)), exist_ok=True)
print(f"Scene → {SCENE_DIR}   Eval summary → {args.output_summary}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

_arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=device)
GRID_Y, GRID_X = torch.meshgrid(_arange, _arange, indexing="ij")


# ── Load JSON (keyed by camera name — one scene instant, 5 camera views) ──────
with open(os.path.join(SCENE_DIR, "scene_gt.json")) as f:
    scene_gt = json.load(f)
with open(os.path.join(SCENE_DIR, "scene_camera.json")) as f:
    scene_cam = json.load(f)

CAMERA_NAMES = sorted(scene_gt.keys(), key=lambda c: int(c.split("_")[1]))
print(f"Cameras: {CAMERA_NAMES}")

if args.cameras is not None:
    CAMERA_NAMES = CAMERA_NAMES[:args.cameras]
    print(f"Running {len(CAMERA_NAMES)} camera(s) (--cameras {args.cameras})")
else:
    print(f"Running all {len(CAMERA_NAMES)} cameras")


# ── Image dimensions ───────────────────────────────────────────────────────────
_sample_img = Image.open(os.path.join(RGB_DIR, f"{CAMERA_NAMES[0]}.png"))
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
# (verified via KD-tree residual check: 0.10mm mean on a 14.35mm object).
# A rotation error measured only against the single canonical R_gt over-
# penalizes convergence to this physically indistinguishable twin.
_mesh_center = (verts_np.min(axis=0) + verts_np.max(axis=0)) / 2
_Rx180 = np.diag([1.0, -1.0, -1.0])
_Sx180 = np.eye(4)
_Sx180[:3, :3] = _Rx180
_Sx180[:3,  3] = _mesh_center - _Rx180 @ _mesh_center
SYMMETRY_TFS = [np.eye(4), _Sx180]


# ── Per-object mean GT translation (fallback init, averaged across cameras) ───
all_obj_ids = sorted({o["obj_id"] for cd in scene_gt.values() for o in cd})
obj_mean_t: dict[int, torch.Tensor] = {}
for oid in all_obj_ids:
    ts   = [o["cam_t_m2c"] for cd in scene_gt.values() for o in cd if o["obj_id"] == oid]
    mean = np.mean(ts, axis=0)
    obj_mean_t[oid] = torch.tensor(mean, dtype=torch.float32, device=device)
print(f"Objects: {all_obj_ids}")


# ── Camera / renderer helpers ──────────────────────────────────────────────────
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


# ── Single-view identity extrinsics — each camera is evaluated independently,
# so it is always its own reference frame (no cross-camera fusion here). ──────
_I3 = torch.eye(3,  dtype=torch.float32, device=device)
_Z3 = torch.zeros(3, dtype=torch.float32, device=device)


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3); ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(ax * np.deg2rad(
        np.random.uniform(-rot_noise_deg, rot_noise_deg))).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Pose model ────────────────────────────────────────────────────────────────
class PoseModel(nn.Module):
    """
    Stage 1 (use_centroid=True):
      cent_views = [(cx_ref, cy_ref, area_ref, R_rel, T_rel)]
      Centroid+area loss (single view — always one entry here).

    Stage 2 (use_centroid=False):
      cam_views  = [(ref_mask, R_rel, T_rel)]
      MSE+IoU (single view — always one entry here).
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


def build_metric_descriptions(instrument_type: str) -> dict:
    """One-line explanation of each key evaluate.py's Evaluator.summarize()
    writes, so the exported JSON is self-explanatory without reading
    evaluate.py. Values/thresholds are evaluate.py's own — nothing here is
    computed independently."""
    diam = model_diameter(instrument_type)
    thresh_mm = diam * 0.1
    return {
        'proj2d': (
            "Fraction of items where the mean 2D reprojection error (GT vs. "
            "predicted model points, projected using evaluate.py's fixed "
            f"'{instrument_type}' camera intrinsics — not this dataset's actual "
            "per-item camera K) is below 5 pixels."
        ),
        'add': (
            "Fraction of items where ADD (mean point-to-corresponding-point "
            f"distance between GT and predicted model points) is below 10% of "
            f"the {instrument_type} object diameter ({thresh_mm:.2f} mm, "
            f"diameter={diam:.2f} mm)."
        ),
        'add-s': (
            "Fraction of items where ADD-S (mean distance between GT and "
            "predicted model points, nearest-neighbor matched instead of "
            f"corresponding — tolerant to symmetry) is below 10% of the "
            f"{instrument_type} object diameter ({thresh_mm:.2f} mm)."
        ),
        'ADD-distance': "Mean ADD distance across all evaluated items, in mm.",
        'ADD-distance list': (
            "Per-item ADD distance in mm, one entry per evaluated item, in "
            "the same order as pred_list/gt_list."
        ),
        'ADDS-distance': "Mean ADD-S distance across all evaluated items, in mm.",
        'cmd5': (
            "Fraction of items within BOTH 5mm translation error AND 5 "
            "degrees rotation error simultaneously."
        ),
        'trans_error': (
            "Mean translation error (Euclidean distance between GT and "
            "predicted translation) across all evaluated items, in mm."
        ),
        'rot_error': (
            "Mean rotation error (geodesic angle between GT and predicted "
            "rotation matrices) across all evaluated items, in degrees."
        ),
        'pred_list': (
            "Predicted pose as a 3x4 [R|t] matrix (t in mm) for each "
            "evaluated item, in evaluation order — parallel to gt_list."
        ),
        'gt_list': (
            "Ground-truth pose as a 3x4 [R|t] matrix (t in mm) for each "
            "evaluated item, in evaluation order — parallel to pred_list."
        ),
    }


def build_report_lines(npy_path: str, descriptions: dict,
                       item_records: list, dataset_dir: str,
                       instrument_type: str, target_obj_ids: set) -> list:
    """Human-readable master report combining evaluate.py's Evaluator metrics
    (same info SurgRIPE's run.py produces) with the per-item domain metrics
    optimize_model_inter_over_union.py prints/writes to CSV (rot/trans error,
    ADD, ADD-S, IoU) — everything one run of this script measured, labeled.
    Returns the report as a list of lines (rendered only into the PDF —
    there's no separate .txt output)."""
    data = np.load(npy_path, allow_pickle=True).item()
    W = 90
    L = []
    L.append("=" * W)
    L.append("MASTER EVALUATION REPORT — EvalSimDiffRender.py".center(W))
    L.append("(differentiable-rendering pose optimizer, simulation)".center(W))
    L.append("=" * W)
    L.append(f"Dataset          : {dataset_dir}")
    L.append(f"Instrument type  : {instrument_type}")
    L.append(f"Target objects   : {sorted(target_obj_ids)}")
    L.append(f"Items evaluated  : {len(item_records)}  (camera x object pairs)")
    L.append(f"Generated        : {datetime.datetime.now().isoformat(timespec='seconds')}")
    L.append("")

    L.append("-" * W)
    L.append("SECTION 1 — evaluate.py Evaluator metrics")
    L.append("(same metric set / same evaluate.py code SurgRIPE's run.py produces)")
    L.append("-" * W)
    for key in ['proj2d', 'add', 'add-s', 'ADD-distance', 'ADDS-distance',
                'cmd5', 'trans_error', 'rot_error']:
        if key not in data:
            continue
        L.append(f"{key:<16} = {float(data[key]):.4f}")
        for wline in textwrap.wrap(descriptions.get(key, ''), width=W - 4):
            L.append("    " + wline)
        L.append("")

    L.append("-" * W)
    L.append("SECTION 2 — Per-item domain metrics, aggregate")
    L.append("(matches the mean/std block optimize_model_inter_over_union.py prints to")
    L.append(" console, and the columns it writes per-row to CSV)")
    L.append("-" * W)

    def agg(key, label, unit):
        vals = np.array([r[key] for r in item_records if r.get(key) is not None], dtype=float)
        if len(vals) == 0:
            return
        L.append(f"{label:<34} mean={vals.mean():9.3f}{unit}   std={vals.std():9.3f}{unit}")

    agg('init_rot_err',   'Init rotation error',              ' deg')
    agg('init_trans_err', 'Init translation error',           ' mm')
    agg('rot_err',        'Rotation error',                   ' deg')
    agg('rot_err_sym',    'Rotation error (symmetry-aware)',  ' deg')
    agg('trans_err',      'Translation error',                ' mm')
    agg('add_val',        'ADD',                              ' mm')
    agg('add_s_val',      'ADD-S',                            ' mm')
    agg('iou_gt_vs_gt',   'IoU(gt,gt) [sanity check, ~1.0]',  '')
    agg('iou_gt_vs_est',  'IoU(gt,est)',                       '')
    L.append("")

    L.append("-" * W)
    L.append("SECTION 3 — Per-item breakdown")
    L.append("-" * W)
    header = (f"{'camera':<10}{'obj':>4}  {'init_rot':>9}{'init_trans':>11}  "
              f"{'rot_err':>8}{'rot_sym':>8}{'trans_err':>10}  "
              f"{'ADD':>7}{'ADD-S':>7}  {'IoU(gt,gt)':>11}{'IoU(gt,est)':>12}")
    L.append(header)
    L.append("-" * len(header))
    for r in item_records:
        L.append(
            f"{r['cam_name']:<10}{r['obj_id']:>4}  "
            f"{r['init_rot_err']:>9.2f}{r['init_trans_err']:>11.2f}  "
            f"{r['rot_err']:>8.2f}{r['rot_err_sym']:>8.2f}{r['trans_err']:>10.2f}  "
            f"{r['add_val']:>7.2f}{r['add_s_val']:>7.2f}  "
            f"{r['iou_gt_vs_gt']:>11.4f}{r['iou_gt_vs_est']:>12.4f}"
        )
    L.append("")

    diam = model_diameter(instrument_type)
    add_vals   = [r['add_val']   for r in item_records]
    add_s_vals = [r['add_s_val'] for r in item_records]
    thr, add_acc,   add_auc   = compute_threshold_curve(add_vals,   diam)
    _,   add_s_acc, add_s_auc = compute_threshold_curve(add_s_vals, diam)
    L.append("-" * W)
    L.append("SECTION 4 — ADD / ADD-S accuracy-vs-threshold curve")
    L.append(f"(fraction of items with ADD[-S] <= threshold, swept 0 to the object diameter "
             f"{diam:.2f} mm; AUC normalized to [0,1] over that range)")
    L.append("-" * W)
    L.append(f"ADD   AUC = {add_auc:.4f}")
    L.append(f"ADD-S AUC = {add_s_auc:.4f}")
    L.append("")
    L.append(f"{'threshold(mm)':>16}{'ADD acc':>12}{'ADD-S acc':>12}")
    for t, a, s in zip(thr, add_acc, add_s_acc):
        L.append(f"{t:>16.2f}{a:>12.4f}{s:>12.4f}")
    L.append("=" * W)
    return L


def write_pdf_report(pdf_path: str, npy_path: str, descriptions: dict,
                     item_records: list, dataset_dir: str,
                     instrument_type: str, target_obj_ids: set):
    """Single self-contained PDF (the only output meant for human review):
    the labeled metrics report, the ADD/ADD-S threshold curve plot, and one
    page per item with the GT (green) vs estimated (orange) silhouette+axes
    overlay."""
    lines = build_report_lines(npy_path, descriptions, item_records,
                               dataset_dir, instrument_type, target_obj_ids)

    diam = model_diameter(instrument_type)
    add_vals   = [r['add_val']   for r in item_records]
    add_s_vals = [r['add_s_val'] for r in item_records]
    thr, add_acc,   add_auc   = compute_threshold_curve(add_vals,   diam)
    _,   add_s_acc, add_s_auc = compute_threshold_curve(add_s_vals, diam)

    with PdfPages(pdf_path) as pdf:
        # Text pages — the labeled metrics report, paginated.
        lines_per_page = 62
        for i in range(0, len(lines), lines_per_page):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.02, 0.98, "\n".join(lines[i:i + lines_per_page]),
                     family='monospace', fontsize=7, va='top', ha='left')
            pdf.savefig(fig)
            plt.close(fig)

        # ADD / ADD-S threshold curve plot
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.plot(thr, add_acc,   label=f"ADD   (AUC={add_auc:.3f})",   color='tab:blue')
        ax.plot(thr, add_s_acc, label=f"ADD-S (AUC={add_s_auc:.3f})", color='tab:orange')
        ax.set_xlabel("Threshold (mm)")
        ax.set_ylabel("Fraction of items with distance <= threshold")
        ax.set_title(f"ADD / ADD-S accuracy vs threshold — {instrument_type} "
                     f"(diameter={diam:.2f} mm)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # One page per item — GT vs estimated overlay image.
        for r in item_records:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.imshow(np.array(r['viz_image']))
            ax.set_title(r['label'])
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)


# ── Crisp silhouette (full resolution, for the console-only IoU metric) ───────
_crisp_raster_settings = RasterizationSettings(
    image_size=(img_H, img_W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
)


def render_crisp_alpha(R_bop, t_bop, cam_K, mesh) -> np.ndarray:
    """Hard binary silhouette at full image resolution, tightly covering the
    tool — used for the GT-vs-estimate IoU console metric."""
    R_pt3d, T_pt3d = bop_to_pt3d(np.array(R_bop), np.array(t_bop))
    R_t = torch.tensor(R_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    T_t = torch.tensor(T_pt3d, dtype=torch.float32, device=device).unsqueeze(0)
    cam = make_cameras(R_t, T_t, cam_K[0], cam_K[4], cam_K[2], cam_K[5], img_H, img_W)
    rasterizer = MeshRasterizer(cameras=cam, raster_settings=_crisp_raster_settings)
    with torch.no_grad():
        frags = rasterizer(mesh.clone(), cameras=cam)
        mask  = (frags.pix_to_face[0, ..., 0] >= 0).float()
    return np.from_dlpack(mask.detach().cpu())


def draw_axes_and_silhouette(rgb_pil, alpha, R_bop, t_bop, cam_K,
                             color_sil=(0, 220, 0)) -> Image.Image:
    """Flat-shaded silhouette fill + 3-axis pose gizmo on top of the RGB
    frame (same style optimize_model_inter_over_union.py used before the
    CSV/image output was replaced with evaluate.py's Evaluator)."""
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
    cen = proj([0,0,0])
    for end, col in zip([proj([L,0,0]), proj([0,L,0]), proj([0,0,L])],
                        [(220,50,50),(50,220,50),(50,50,220)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


def make_side_by_side(rgb_pil, alpha_gt, alpha_est, R_gt, t_gt, R_est, t_est,
                      cam_K, left_label, right_label) -> Image.Image:
    """GT (green, left) vs estimated (orange, right) overlay, side by side."""
    img_gt  = draw_axes_and_silhouette(rgb_pil, alpha_gt,  R_gt,  t_gt,  cam_K, (0, 220, 0))
    img_est = draw_axes_and_silhouette(rgb_pil, alpha_est, R_est, t_est, cam_K, (220, 80, 0))
    W, H = rgb_pil.size
    LH = 24
    combined = Image.new("RGB", (W * 2, H + LH), (40, 40, 40))
    combined.paste(img_gt,  (0, LH))
    combined.paste(img_est, (W, LH))
    d = ImageDraw.Draw(combined)
    d.text((8,     4), left_label,  fill=(100, 255, 100))
    d.text((W + 8, 4), right_label, fill=(255, 180, 80))
    return combined


def compute_threshold_curve(values: list, max_threshold: float, n_points: int = 21):
    """Accuracy-vs-threshold curve: fraction of items with value <= t, for t
    swept from 0 to max_threshold. Returns (thresholds, accuracy, auc), AUC
    normalized to [0, 1] over the swept range."""
    vals = np.array(values, dtype=float)
    thresholds = np.linspace(0.0, max_threshold, n_points)
    accuracy = np.array([(vals <= t).mean() for t in thresholds])
    auc = float(np.trapezoid(accuracy, thresholds) / max_threshold) if max_threshold > 0 else 0.0
    return thresholds, accuracy, auc


# ── evaluate.py Evaluator setup ─────────────────────────────────────────────────
# Evaluator.__init__ loads root_path/joint.npy (mm-scale model points).
# verts_np is already mm-scaled above, so write it straight to a scratch dir.
eval_tmp_dir = tempfile.mkdtemp(prefix='eval_sim_dr_')
np.save(os.path.join(eval_tmp_dir, 'joint.npy'), verts_np)
evaluator = Evaluator(root_path=eval_tmp_dir, instrument_type=INSTRUMENT_TYPE)
metric_descriptions = build_metric_descriptions(INSTRUMENT_TYPE)
# Per-item domain metrics (rot/trans error, ADD, ADD-S, IoU) — the same
# quantities optimize_model_inter_over_union.py prints in its console
# "Summary" block and writes per-row to CSV. evaluate.py's Evaluator doesn't
# track these, so they're collected here for the master report.
item_records = []
print(f"Evaluator ready (instrument_type={INSTRUMENT_TYPE})")


# ── Main loop: one item = one camera's single-view pose estimate ──────────────
for cam_name in CAMERA_NAMES:
    rgb_path = os.path.join(RGB_DIR, f"{cam_name}.png")
    cam_K    = scene_cam[cam_name]["cam_K"]
    fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)

    rgb_orig_pil = Image.open(rgb_path).convert("RGB")
    ref_np       = np.array(rgb_orig_pil.resize((RENDER_SIZE, RENDER_SIZE), Image.BILINEAR),
                             dtype=np.float32)

    print(f"\n══ {cam_name} ══")

    for obj_data in scene_gt[cam_name]:
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
            # ── Build single-view references for this camera ──────────────────
            if USE_RENDERED_REFERENCE:
                R_c, T_c = bop_to_pt3d(R_gt, t_gt)
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

                cent_views_s1 = [(cx_c, cy_c, area_c, _I3, _Z3)]
                cam_views_s2a = [(r_2a[0,...,3].detach().clone(), _I3, _Z3)]
                cam_views_s2b = [(r_2b[0,...,3].detach().clone(), _I3, _Z3)]
                cam_views_s2c = [(r_2c[0,...,3].detach().clone(), _I3, _Z3)]
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

            # ── Stage 1: translation only, centroid+area ───────────────────────
            model.cent_views  = cent_views_s1
            model.use_centroid = True
            model.use_iou      = False
            opt_s1 = torch.optim.Adam([model.transl], lr=LR_TRANS)
            loop_s1 = tqdm(range(NUM_ITERS_TRANS), desc=f"    s1 obj{obj_id}")
            for _ in loop_s1:
                opt_s1.zero_grad(); loss, _, _ = model(); loss.backward(); opt_s1.step()
                loop_s1.set_description(f"    s1 obj{obj_id} ({loss.item():.3f})")
            print(f"    Stage 1 done: loss={loss.item():.4f}")

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

        # ── Console-only errors ──────────────────────────────────────────────
        rot_err     = rotation_error_deg(R_est_np, R_gt)
        rot_err_sym = rotation_error_sym_deg(R_est_np, R_gt, SYMMETRY_TFS)
        trans_err   = translation_error_mm(T_est_np, t_gt)
        add_val     = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)
        add_s_val   = add_s_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt, SYMMETRY_TFS)

        alpha_gt  = render_crisp_alpha(R_gt,     t_gt,     cam_K, mesh)
        alpha_est = render_crisp_alpha(R_est_np, T_est_np, cam_K, mesh)
        iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
        iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)
        print(f"    rot={rot_err:.2f}° (sym={rot_err_sym:.2f}°)  trans={trans_err:.2f}mm  "
              f"ADD={add_val:.2f}mm (ADD-S={add_s_val:.2f}mm)  "
              f"IoU(gt,gt)={iou_gt_vs_gt:.4f}  IoU(gt,est)={iou_gt_vs_est:.4f}")

        viz_image = make_side_by_side(
            rgb_orig_pil, alpha_gt, alpha_est, R_gt, t_gt, R_est_np, T_est_np, cam_K,
            left_label=f"GT  {cam_name} obj={obj_id}",
            right_label=f"EST  rot={rot_err:.1f}deg trans={trans_err:.1f}mm "
                        f"ADD={add_val:.1f}mm IoU={iou_gt_vs_est:.3f}",
        )

        item_records.append({
            'cam_name': cam_name, 'obj_id': obj_id,
            'init_rot_err': init_rot_err, 'init_trans_err': init_trans_err,
            'rot_err': rot_err, 'rot_err_sym': rot_err_sym, 'trans_err': trans_err,
            'add_val': add_val, 'add_s_val': add_s_val,
            'iou_gt_vs_gt': iou_gt_vs_gt, 'iou_gt_vs_est': iou_gt_vs_est,
            'viz_image': viz_image, 'label': f"{cam_name}_obj{obj_id}",
        })

        # ── Feed evaluate.py's Evaluator (mm-scale 3×4 poses) ────────────────
        pose_pred_34 = np.hstack([R_est_np, T_est_np.reshape(3, 1)])
        pose_gt_34   = np.hstack([R_gt,     t_gt.reshape(3, 1)])
        evaluator.evaluate(pose_gt=pose_gt_34, pose_pred=pose_pred_34)


print("\nDone running cameras — summarizing with evaluate.py's Evaluator")
evaluator.summarize(save_path=args.output_summary)
print(f"Eval summary saved to: {args.output_summary}")

pdf_summary_path = os.path.splitext(args.output_summary)[0] + '.pdf'
write_pdf_report(pdf_summary_path, args.output_summary, metric_descriptions,
                 item_records, SCENE_DIR, INSTRUMENT_TYPE, TARGET_OBJ_IDS)
print(f"Master report (pdf, with GT/est images + ADD threshold curve) saved to: {pdf_summary_path}")
