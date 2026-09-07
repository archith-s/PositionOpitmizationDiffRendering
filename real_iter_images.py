"""
real_iter_images.py  (real data, per-iteration debug visualizer)

Debug tool: runs the *same* single-stage differentiable-rendering optimizer as
optimize_model_orig_real.py (one silhouette renderer, fixed sigma=1e-4, one
Adam optimizer, plain MSE silhouette loss against the real per-frame mask —
mirrors the original PyTorch3D camera-position tutorial extended to full
6-DOF) on ONE frame, but instead of only reporting the final result, it
snapshots the pose estimate at several points across the 200 iterations so
you can see the "wiggle from a 10mm/10°-perturbed start to convergence"
process directly:

  - Draws a GT-vs-estimate overlay image at each snapshot iteration.
  - Logs a CSV row per snapshot with the same metrics columns used by the
    IoU scripts (rotation/translation error, ADD, IoU) plus `iteration` and
    `loss`.

Mirrors sim_iter_images.py exactly except for the real-data-specific pieces
already present in optimize_model_orig_real.py: STL→OBJ mesh conversion, the
real per-frame mask as ground truth (instead of a GT-pose-rendered mask), and
screen-space (non-NDC) cameras with an explicit image_size (this dataset's
960×540 aspect ratio is silently mishandled by PyTorch3D's default square-NDC
assumption).
"""

import os, re, csv, struct, tempfile
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.spatial.transform import Rotation

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)


# ─────────────────────────── STL → OBJ ────────────────────────────────────────
def stl_to_obj(stl_path: str, obj_path: str):
    """Convert binary STL to OBJ with deduplicated vertices."""
    with open(stl_path, 'rb') as f:
        f.read(80)  # skip header
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
MODEL_PTS    = os.path.join(DATASET_ROOT, "joint.npy")  # for ADD metric

OUTPUT_DIR = "../Real_Iter_Debug_Visuals"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "real_iter_metrics.csv")

# Single-stage optimizer — identical hyperparameters to optimize_model_orig_real.py,
# except sigma is annealed coarse-to-fine (see sigma_schedule) instead of fixed.
SIGMA        = 1e-4   # tutorial's blend_params.sigma (fine, end-of-run value)
SIGMA_COARSE = 1e-2   # coarse, start-of-run value -- wide capture basin
NUM_ITERS = 200    # tutorial's range(200)
LR        = 0.05   # tutorial's single Adam lr

RENDER_SIZE  = 256
FG_THRESHOLD = 0.5

INIT_ROT_NOISE_DEG  = 10.0
INIT_TRANS_NOISE_MM = 10.0
NUM_INIT_HYPOTHESES = 8   # candidate initial poses scored (coarse render, no
                          # gradient steps) before picking one to refine

IOU_THRESHOLD = 0.5
# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--frame-index", type=int, default=0,
                     help="Index into the sorted list of valid frames (default: first frame)")
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

if not (0 <= args.frame_index < len(valid_frame_ids)):
    raise SystemExit(f"--frame-index {args.frame_index} out of range (0..{len(valid_frame_ids)-1})")
frame_id = valid_frame_ids[args.frame_index]
print(f"Debugging frame {frame_id}")


# ── Image dimensions ───────────────────────────────────────────────────────────
rgb_path  = os.path.join(_image_dir, f"{frame_id}.png")
mask_path = os.path.join(_mask_dir,  f"{frame_id}.png")
pose_path = os.path.join(_pose_dir,  f"{frame_id}.npy")

rgb_orig_pil = Image.open(rgb_path).convert("RGB")
img_W, img_H = rgb_orig_pil.size
print(f"Image size: {img_W}×{img_H}")

mask_full = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
mask_resized = np.array(
    Image.fromarray((mask_full * 255).astype(np.uint8)).resize(
        (RENDER_SIZE, RENDER_SIZE), Image.NEAREST),
    dtype=np.float32) / 255.0

pose_mat = np.load(pose_path)  # (3, 4)
R_gt = pose_mat[:3, :3].copy().astype(np.float64)
t_gt = pose_mat[:3,  3].copy().astype(np.float64)
print(f"GT t: {t_gt.round(2)}  |  mask fg px: {int(mask_full.sum())}")


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

# This mesh has near-exact 180° rotational symmetry about its local Y axis
# (verified via KD-tree residual check: 0.02mm mean on a 13.56mm object --
# note this is a *different* local axis than sim_iter_images.py's mesh
# (X there), since joint.stl and "tool pitch link.OBJ" use different local
# coordinate conventions for the same physical part, not a different real
# symmetry). A rotation error measured only against the single canonical
# R_gt over-penalizes convergence to this physically indistinguishable twin.
_mesh_center = (verts_np.min(axis=0) + verts_np.max(axis=0)) / 2
_Ry180 = np.diag([-1.0, 1.0, -1.0])
_Sy180 = np.eye(4)
_Sy180[:3, :3] = _Ry180
_Sy180[:3,  3] = _mesh_center - _Ry180 @ _mesh_center
SYMMETRY_TFS = [np.eye(4), _Sy180]


# ── Load model point cloud for ADD metric ─────────────────────────────────────
model_pts_np = np.load(MODEL_PTS).astype(np.float64)  # (30000, 3)
_pts_vmax = float(np.abs(model_pts_np).max())
if _pts_vmax < 1.0:
    model_pts_np = model_pts_np * 1000.0
print(f"Model pts: {model_pts_np.shape[0]} points | max|v|={_pts_vmax:.4f}")


# ── Camera / renderer helpers ──────────────────────────────────────────────────
def make_cameras(R, T, fx, fy, cx, cy, H, W) -> PerspectiveCameras:
    return PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        R=R, T=T, in_ndc=False, image_size=((H, W),), device=device,
    )


def scale_intrinsics(K_flat: list, W_from: int, H_from: int, W_to: int, H_to: int):
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
    ax = np.random.randn(3)
    ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(
        ax * np.deg2rad(np.random.uniform(-rot_noise_deg, rot_noise_deg))
    ).as_matrix() @ R_gt
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


# ── Pose model — mirrors optimize_model_orig_real.py's PoseModel exactly ───────
class PoseModel(nn.Module):
    def __init__(self, mesh, renderer, mask_resized_np, init_R, init_t,
                 fx, fy, cx, cy, H, W):
        super().__init__()
        self.mesh = mesh
        self.renderer = renderer
        self.fx = fx;  self.fy = fy
        self.cx = cx;  self.cy = cy
        self.H  = H;   self.W  = W

        fg_mask = torch.tensor(mask_resized_np, dtype=torch.float32, device=device)
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


def add_metric(pts_np, R_est, t_est, R_gt, t_gt):
    return float(np.mean(np.linalg.norm(
        ((R_est @ pts_np.T).T + t_est) - ((R_gt @ pts_np.T).T + t_gt), axis=1)))


def rotation_error_sym_deg(R_est, R_gt, symmetry_tfs):
    """Minimum geodesic rotation error over all symmetry-equivalent poses."""
    return float(min(rotation_error_deg(R_est @ S[:3, :3], R_gt) for S in symmetry_tfs))


def add_s_metric(pts_np, R_est, t_est, R_gt, t_gt, symmetry_tfs):
    """ADD-S: minimum ADD over all symmetry-equivalent poses."""
    pts_gt = (R_gt @ pts_np.T).T + t_gt
    best = float("inf")
    for S in symmetry_tfs:
        R_s = R_est @ S[:3, :3]
        t_s = R_est @ S[:3, 3] + t_est
        pts_est = (R_s @ pts_np.T).T + t_s
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
fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
fx_r, fy_r, cx_r, cy_r = scale_intrinsics(cam_K, img_W, img_H, RENDER_SIZE, RENDER_SIZE)
_crisp_raster_settings = RasterizationSettings(
    image_size=(img_H, img_W), blur_radius=0.0, faces_per_pixel=1, bin_size=0,
)
_mesh_viz = build_mesh(solid=True)


def render_crisp_alpha(R_bop, t_bop) -> np.ndarray:
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
                             color_sil=(0, 220, 0)) -> Image.Image:
    overlay = np.zeros((*alpha.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    rgb_arr = np.array(rgb_pil, dtype=np.float32)
    blended = np.clip(0.5 * alpha[..., None] * overlay + rgb_arr, 0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)

    draw = ImageDraw.Draw(result)
    fx_p, fy_p, cx_p, cy_p = cam_K[0], cam_K[4], cam_K[2], cam_K[5]

    def proj(p):
        pc = np.array(R_bop) @ np.array(p) + np.array(t_bop)
        return (int(round(fx_p * pc[0] / pc[2] + cx_p)),
                int(round(fy_p * pc[1] / pc[2] + cy_p))) if pc[2] > 0 else None

    L   = 50.0
    cen = proj([0, 0, 0])
    for end, col in zip([proj([L, 0, 0]), proj([0, L, 0]), proj([0, 0, L])],
                        [(220, 50, 50), (50, 220, 50), (50, 50, 220)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


# ── Setup for this single frame ────────────────────────────────────────────────
mesh         = build_mesh(solid=True)
sil_renderer = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
                                        sigma=SIGMA, faces_per_pixel=100)

# Multi-hypothesis initialization: score several candidate perturbed poses
# with a single coarse (no-gradient) render each and keep the best-scoring
# one, instead of committing to one random draw and hoping 200 Adam steps
# find their way out of a bad basin. Cheap -- no backward pass, no 200-iter
# loop per candidate -- unlike a full multi-restart optimization sweep.
_scoring_renderer = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
                                             sigma=SIGMA_COARSE, faces_per_pixel=100)
_ref_tensor = torch.tensor(mask_resized, dtype=torch.float32, device=device)

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
    mesh, sil_renderer, mask_resized,
    init_R_pt3d, init_t_pt3d,
    fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

alpha_gt = render_crisp_alpha(R_gt, t_gt)


# ── Snapshot schedule: --num-snapshots points evenly spaced across 0..NUM_ITERS ─
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
    add_val     = add_metric(model_pts_np, R_est_np, T_est_np, R_gt, t_gt)
    add_s_val   = add_s_metric(model_pts_np, R_est_np, T_est_np, R_gt, t_gt, SYMMETRY_TFS)

    alpha_est     = render_crisp_alpha(R_est_np, T_est_np)
    iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
    iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)

    print(f"  [iter {step:>3d}/{NUM_ITERS}] rot={rot_err:.2f}° (sym={rot_err_sym:.2f}°)  trans={trans_err:.2f}mm  "
          f"ADD={add_val:.2f}mm (ADD-S={add_s_val:.2f}mm)  IoU(gt,est)={iou_gt_vs_est:.4f}  "
          f"loss={'n/a' if loss_val is None else f'{loss_val:.4f}'}")

    gt_ov  = draw_axes_and_silhouette(rgb_orig_pil, alpha_gt,  R_gt,     t_gt,     (0, 220, 0))
    est_ov = draw_axes_and_silhouette(rgb_orig_pil, alpha_est, R_est_np, T_est_np, (220, 80, 0))
    LH = 28
    combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
    combined.paste(gt_ov,  (0,     LH))
    combined.paste(est_ov, (img_W, LH))
    d = ImageDraw.Draw(combined)
    d.text((8,         5), f"GT  frame={frame_id}", fill=(100, 255, 100))
    d.text((img_W + 8, 5),
           f"EST  iter={step}/{NUM_ITERS}  rot={rot_err:.1f}° (sym={rot_err_sym:.1f}°)  trans={trans_err:.1f}mm  "
           f"ADD={add_val:.1f}mm (ADD-S={add_s_val:.1f}mm)  IoU={iou_gt_vs_est:.3f}",
           fill=(255, 180, 80))
    out = os.path.join(OUTPUT_DIR, f"real_frame{frame_id}_iter{step:04d}.png")
    combined.save(out)

    return {
        "frame_id": frame_id, "iteration": step,
        "loss": loss_val if loss_val is not None else "",
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
        sigma=sigma_schedule(step, NUM_ITERS, SIGMA_COARSE, SIGMA), faces_per_pixel=100)
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
    w.writeheader()
    w.writerows(csv_rows)
print(f"\nCSV → {OUTPUT_CSV}  ({len(csv_rows)} rows)")
print(f"Images → {OUTPUT_DIR}  ({len(csv_rows)} snapshots)")
