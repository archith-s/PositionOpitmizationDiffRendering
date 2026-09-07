"""
sim_iter_images.py  (simulation, per-iteration debug visualizer)

Debug tool: runs the *same* single-stage differentiable-rendering optimizer as
optimize_model_orig_sim.py (one silhouette renderer, fixed sigma=1e-4, one
Adam optimizer, plain MSE silhouette loss — mirrors the original PyTorch3D
camera-position tutorial extended to full 6-DOF) on ONE frame / ONE object,
but instead of only reporting the final result, it snapshots the pose
estimate at several points across the 200 iterations so you can see the
"wiggle from a 10mm/10°-perturbed start to convergence" process directly:

  - Draws a GT-vs-estimate overlay image at each snapshot iteration.
  - Logs a CSV row per snapshot with the same metrics columns used by the
    IoU scripts (rotation/translation error, ADD, IoU) plus `iteration` and
    `loss`.

Uses camera_0 (the side-facing camera), not camera_l, since camera_0's
viewing angle is closest to the real dataset's camera — this makes the sim
and real debug runs visually comparable.
"""

import os, re, csv, json, tempfile
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from scipy import ndimage

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)


# ─────────────────────────── CONFIG ───────────────────────────────────────────
DATASET_ROOT   = "../SIMPLELND_TOPDOWNdataset"
PRIMARY_CAMERA = "camera_0"   # side-facing camera — closest match to real data's view
CAMERA_NAMES   = ["camera_0"]
CAMERA_DIRS    = {cam: os.path.join(DATASET_ROOT, cam, "000001") for cam in CAMERA_NAMES}

CAMERA_DIR     = CAMERA_DIRS[PRIMARY_CAMERA]
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")

MESH_PATH  = ("../surgical_robotics_challenge/ADF/PSMs/"
              "LND_420006/high_res/tool pitch link.OBJ")
OUTPUT_DIR = "../Sim_Iter_Debug_Visuals"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "sim_iter_metrics.csv")

# Single-stage optimizer — identical hyperparameters to optimize_model_orig_sim.py,
# except sigma is annealed coarse-to-fine (see sigma_schedule) instead of fixed.
SIGMA        = 1e-4   # tutorial's blend_params.sigma (fine, end-of-run value)
SIGMA_COARSE = 1e-2   # coarse, start-of-run value -- wide capture basin
NUM_ITERS = 200    # tutorial's range(200)
LR        = 0.05   # tutorial's single Adam lr

RENDER_SIZE  = 256
FG_THRESHOLD = 0.1
BRIGHT_FG    = True

INIT_ROT_NOISE_DEG  = 10.0
INIT_TRANS_NOISE_MM = 10.0
NUM_INIT_HYPOTHESES = 8   # candidate initial poses scored (coarse render, no
                          # gradient steps) before picking one to refine

TARGET_OBJ_IDS = {1, 3}
IOU_THRESHOLD  = 0.5

APPEARANCE_SAT_THRESHOLD = 0.25
# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--frame-index", type=int, default=0,
                     help="Index into the sorted list of valid frames (default: first frame)")
parser.add_argument("--obj-id", type=int, default=None,
                     help="Object id to debug (default: first of TARGET_OBJ_IDS present in the frame)")
parser.add_argument("--num-snapshots", type=int, default=5,
                     help="Number of iteration snapshots to draw/log, evenly spaced 0..NUM_ITERS (default: 5)")
parser.add_argument("--seed", type=int, default=None,
                     help="Random seed for the init-pose perturbation (default: unseeded)")
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output → {OUTPUT_DIR}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── Load JSON for the (single) camera ──────────────────────────────────────────
with open(os.path.join(CAMERA_DIR, "scene_gt.json"))     as f: scene_gt  = json.load(f)
with open(os.path.join(CAMERA_DIR, "scene_camera.json")) as f: scene_cam = json.load(f)

_rgb_on_disk = {
    int(fn.split(".")[0]): fn
    for fn in os.listdir(RGB_DIR)
    if fn.lower().endswith((".png", ".jpg", ".jpeg"))
}
valid_frame_ids = sorted(set(int(k) for k in scene_gt.keys()) & set(_rgb_on_disk.keys()))
print(f"Valid frames: {len(valid_frame_ids)}")

if not (0 <= args.frame_index < len(valid_frame_ids)):
    raise SystemExit(f"--frame-index {args.frame_index} out of range (0..{len(valid_frame_ids)-1})")
frame_id = valid_frame_ids[args.frame_index]
fkey     = str(frame_id)

present_target_objs = [o["obj_id"] for o in scene_gt[fkey] if o["obj_id"] in TARGET_OBJ_IDS]
if not present_target_objs:
    raise SystemExit(f"No TARGET_OBJ_IDS {TARGET_OBJ_IDS} present in frame {frame_id}")
obj_id = args.obj_id if args.obj_id is not None else present_target_objs[0]
if obj_id not in present_target_objs:
    raise SystemExit(f"--obj-id {obj_id} not present in frame {frame_id} (available: {present_target_objs})")

print(f"Debugging frame {frame_id:04d} ({_rgb_on_disk[frame_id]}), obj_id={obj_id}, camera={PRIMARY_CAMERA}")


# ── Image dimensions ───────────────────────────────────────────────────────────
rgb_path     = os.path.join(RGB_DIR, _rgb_on_disk[frame_id])
rgb_orig_pil = Image.open(rgb_path).convert("RGB")
img_W, img_H = rgb_orig_pil.size
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


# ── Camera / renderer helpers ──────────────────────────────────────────────────
# Screen-space (in_ndc=False) cameras with an explicit image_size throughout,
# NOT PyTorch3D's default NDC convention -- NDC silently mishandles non-square
# aspect ratios (this dataset is 640x480, not 1:1), which produced a
# confirmed, measured ~15px offset against check_gt_alignment.py's
# plain-pixel-space rendering of identical pose data (verified 2026-07-10).
# This is the same class of bug already fixed for the real-data scripts
# (960x540) -- see optimize_model_real_data_IoU.py -- just never applied to
# the sim side, in either the crisp GT/EST render (fixed above) or here in
# the actual differentiable-rendering optimizer.
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


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3); ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(ax * np.deg2rad(
        np.random.uniform(-rot_noise_deg, rot_noise_deg))).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


def sigma_schedule(step: int, num_iters: int, sigma_coarse: float, sigma_fine: float) -> float:
    """Geometric coarse-to-fine anneal: wide blur (big world-space capture
    basin) early, full precision by the end. A fixed sigma's capture basin
    shrinks in world-space as the object gets closer to the camera
    (mm_tolerance ~ blur_px * depth / focal_length), which silently starved
    the optimizer of any silhouette overlap to descend from on close-up views
    (e.g. IoU=0 for the entire run on the SIMPLELND_TOPDOWNdataset). Annealing
    keeps a usable capture basin regardless of view scale, instead of relying
    on picking the right fixed sigma per dataset."""
    t = step / num_iters
    return sigma_coarse * (sigma_fine / sigma_coarse) ** t


# ── Pose model — mirrors optimize_model_orig_sim.py's PoseModel exactly ────────
class PoseModel(nn.Module):
    def __init__(self, mesh, renderer, image_ref_np, init_R, init_t,
                 fx, fy, cx, cy, H, W):
        super().__init__()
        self.mesh = mesh;  self.renderer = renderer
        self.fx = fx;  self.fy = fy
        self.cx = cx;  self.cy = cy
        self.H  = H;   self.W  = W

        if image_ref_np.ndim == 2:
            fg_mask = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
        else:
            ref_t = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
            br    = ref_t.mean(-1) / ref_t.max().clamp(min=1e-6)
            fg_mask = (br > FG_THRESHOLD).float() if BRIGHT_FG \
                      else (br < (1.0 - FG_THRESHOLD)).float()
        self.register_buffer("image_ref", fg_mask)

        self.rot6d  = nn.Parameter(matrix_to_rot6d(init_R))
        self.transl = nn.Parameter(torch.tensor(init_t, dtype=torch.float32, device=device))

    def forward(self):
        R = rot6d_to_matrix(self.rot6d).unsqueeze(0)
        T = self.transl.unsqueeze(0)
        cam = make_cameras(R, T, self.fx, self.fy, self.cx, self.cy, self.H, self.W)
        alpha = self.renderer(meshes_world=self.mesh.clone(), cameras=cam)[0, ..., 3]
        loss = torch.sum((alpha - self.image_ref) ** 2)
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
    mask_a = alpha_a > threshold
    mask_b = alpha_b > threshold
    union  = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter) / float(union)


# ── Crisp silhouette + visualization ───────────────────────────────────────────
cam_K = scene_cam[fkey]["cam_K"]
_crisp_raster_settings = RasterizationSettings(
    image_size=(img_H, img_W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
)

def render_crisp_alpha(R_bop, t_bop, cam_K, mesh) -> np.ndarray:
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
    rgb = np.array(rgb_pil, dtype=np.float32)
    mx  = rgb.max(axis=-1)
    mn  = rgb.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1.0, None)
    return (sat < sat_threshold).astype(np.float32)

def draw_axes_and_silhouette(rgb_pil, alpha, R_bop, t_bop, cam_K,
                             color_sil=(255, 0, 255), appearance_mask=None) -> Image.Image:
    # Thin contour outline, not a translucent fill: a filled blob tints the
    # pixels underneath (hiding the actual tool edge you're trying to verify
    # against) and a fill color from the same hue family as the axis-gizmo's
    # fixed R/G/B lines (the old default (0,220,0) green fill vs the Y-axis's
    # (50,220,50) green line) makes the two visually blend together, which is
    # exactly what caused "is the silhouette off?" to be hard to judge by eye
    # here. Matches check_gt_alignment.py's contour style, and its default
    # magenta avoids the R/G/B axis colors entirely.
    fg = alpha if appearance_mask is None else alpha * appearance_mask
    fg_mask = fg > 0.5
    eroded = ndimage.binary_erosion(fg_mask, iterations=1)
    contour = ndimage.binary_dilation(fg_mask & ~eroded, iterations=1)
    rgb_arr = np.array(rgb_pil, dtype=np.uint8).copy()
    rgb_arr[contour] = color_sil
    result = Image.fromarray(rgb_arr)

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


# ── Setup for this single frame/object ─────────────────────────────────────────
ref_np          = np.array(rgb_orig_pil.resize((RENDER_SIZE, RENDER_SIZE), Image.BILINEAR),
                            dtype=np.float32)
appearance_mask = appearance_foreground_mask(rgb_orig_pil)
# The optimizer renders at RENDER_SIZE (256x256), not the native capture
# resolution, so intrinsics must be rescaled accordingly -- same pattern as
# optimize_model_orig_real.py's scale_intrinsics usage.
fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)

obj_data = next(o for o in scene_gt[fkey] if o["obj_id"] == obj_id)
R_gt = np.array(obj_data["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
t_gt = np.array(obj_data["cam_t_m2c"], dtype=np.float64)

mesh         = build_mesh()
sil_renderer = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA)

# Reference silhouette rendered at the GT pose (same as optimize_model_orig_sim.py)
R_gt_pt3d, T_gt_pt3d = bop_to_pt3d(R_gt, t_gt)
cam_gt = make_cameras(
    torch.tensor(R_gt_pt3d, dtype=torch.float32, device=device).unsqueeze(0),
    torch.tensor(T_gt_pt3d, dtype=torch.float32, device=device).unsqueeze(0),
    fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE)
with torch.no_grad():
    r_gt = sil_renderer(meshes_world=mesh.clone(), cameras=cam_gt)
ref_for_model = np.from_dlpack(r_gt[0, ..., 3].detach().cpu())

# Multi-hypothesis initialization: score several candidate perturbed poses
# with a single coarse (no-gradient) render each and keep the best-scoring
# one, instead of committing to one random draw and hoping 200 Adam steps
# find their way out of a bad basin. Cheap -- no backward pass, no 200-iter
# loop per candidate -- unlike a full multi-restart optimization sweep.
_scoring_renderer = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
                                             sigma=SIGMA_COARSE)
_ref_tensor = torch.tensor(ref_for_model, dtype=torch.float32, device=device)

def _score_candidate(R_bop, t_bop) -> float:
    R_pt3d, T_pt3d = bop_to_pt3d(R_bop, t_bop)
    cam = make_cameras(
        torch.tensor(R_pt3d, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(T_pt3d, dtype=torch.float32, device=device).unsqueeze(0),
        fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE)
    with torch.no_grad():
        alpha = _scoring_renderer(meshes_world=mesh.clone(), cameras=cam)[0, ..., 3]
        return float(torch.sum((alpha - _ref_tensor) ** 2).item())

print(f"Scoring {NUM_INIT_HYPOTHESES} candidate initial poses (coarse render, no gradient steps)...")
init_R, init_t, best_score = None, None, float("inf")
for h in range(NUM_INIT_HYPOTHESES):
    cand_R, cand_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
    score = _score_candidate(cand_R, cand_t)
    print(f"  hypothesis {h}: rot={rotation_error_deg(cand_R, R_gt):.1f}°  "
          f"trans={translation_error_mm(cand_t, t_gt):.1f}mm  score={score:.2f}")
    if score < best_score:
        best_score, init_R, init_t = score, cand_R, cand_t
print(f"Selected hypothesis: score={best_score:.2f}")

init_rot_err   = rotation_error_deg(init_R, R_gt)
init_trans_err = translation_error_mm(init_t, t_gt)
print(f"Init: rot={init_rot_err:.2f}°  trans={init_trans_err:.2f} mm  "
      f"(perturbation budget: ±{INIT_ROT_NOISE_DEG}°, ±{INIT_TRANS_NOISE_MM}mm)")

init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
model = PoseModel(
    mesh, sil_renderer, ref_for_model,
    init_R_pt3d, init_t_pt3d,
    fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

alpha_gt = render_crisp_alpha(R_gt, t_gt, cam_K, mesh)


# ── Snapshot schedule: --num-snapshots points evenly spaced across 0..NUM_ITERS ─
# 0 = the perturbed init pose before any optimizer step; NUM_ITERS = final result.
snapshot_steps = sorted(set(
    int(round(x)) for x in np.linspace(0, NUM_ITERS, args.num_snapshots)
))
print(f"Snapshotting at iterations: {snapshot_steps}")


def capture_snapshot(step: int, loss_val):
    with torch.no_grad():
        R_est_t = rot6d_to_matrix(model.rot6d)
        T_est_t = model.transl
    R_est_pt3d_np = np.from_dlpack(R_est_t.detach().cpu())
    T_est_pt3d_np = np.from_dlpack(T_est_t.detach().cpu())
    R_est_np, T_est_np = pt3d_to_bop(R_est_pt3d_np, T_est_pt3d_np)

    rot_err     = rotation_error_deg(R_est_np, R_gt)
    rot_err_sym = rotation_error_sym_deg(R_est_np, R_gt, SYMMETRY_TFS)
    trans_err   = translation_error_mm(T_est_np, t_gt)
    add_val     = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)
    add_s_val   = add_s_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt, SYMMETRY_TFS)

    alpha_est     = render_crisp_alpha(R_est_np, T_est_np, cam_K, mesh)
    iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
    iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)

    print(f"  [iter {step:>3d}/{NUM_ITERS}] rot={rot_err:.2f}° (sym={rot_err_sym:.2f}°)  trans={trans_err:.2f}mm  "
          f"ADD={add_val:.2f}mm (ADD-S={add_s_val:.2f}mm)  IoU(gt,est)={iou_gt_vs_est:.4f}  "
          f"loss={'n/a' if loss_val is None else f'{loss_val:.4f}'}")

    gt_ov  = draw_axes_and_silhouette(rgb_orig_pil, alpha_gt,  R_gt,     t_gt,     cam_K, (255, 0, 255),
                                      appearance_mask=appearance_mask)
    est_ov = draw_axes_and_silhouette(rgb_orig_pil, alpha_est, R_est_np, T_est_np, cam_K, (255, 255, 0),
                                      appearance_mask=appearance_mask)
    LH = 28
    combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
    combined.paste(gt_ov,  (0,     LH));  combined.paste(est_ov, (img_W, LH))
    d = ImageDraw.Draw(combined)
    d.text((8,         5), f"GT  frame={frame_id:04d}  obj={obj_id}", fill=(100,255,100))
    d.text((img_W + 8, 5), f"EST  iter={step}/{NUM_ITERS}  rot={rot_err:.1f}° (sym={rot_err_sym:.1f}°)  "
                            f"trans={trans_err:.1f}mm  ADD={add_val:.1f}mm (ADD-S={add_s_val:.1f}mm)  "
                            f"IoU={iou_gt_vs_est:.3f}",
           fill=(255,180,80))
    out = os.path.join(OUTPUT_DIR, f"sim_frame{frame_id:04d}_obj{obj_id}_iter{step:04d}.png")
    combined.save(out)

    return {
        "frame_id": frame_id, "obj_id": obj_id, "iteration": step,
        "loss": loss_val if loss_val is not None else "",
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
    }


# ── Optimization loop with snapshots ───────────────────────────────────────────
csv_rows = []
if 0 in snapshot_steps:
    csv_rows.append(capture_snapshot(0, None))

loss = None
for step in range(1, NUM_ITERS + 1):
    model.renderer = make_silhouette_renderer(
        fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
        sigma=sigma_schedule(step, NUM_ITERS, SIGMA_COARSE, SIGMA))
    optimizer.zero_grad()
    loss, _, _ = model()
    loss.backward()
    optimizer.step()
    if step in snapshot_steps:
        csv_rows.append(capture_snapshot(step, loss.item()))

print(f"Done: final loss={loss.item():.4f}")


# ── Write CSV ──────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader(); w.writerows(csv_rows)
print(f"\nCSV → {OUTPUT_CSV}  ({len(csv_rows)} rows)")
print(f"Images → {OUTPUT_DIR}  ({len(csv_rows)} snapshots)")
