"""
optimize_model_orig_sim.py  (simulation, single-stage "original tutorial" optimizer)
Differentiable-rendering 6-DoF pose estimator (PyTorch3D silhouette loss).

Unlike optimize_model_inter_over_union.py's 5-stage coarse-to-fine schedule,
this mirrors the *original* PyTorch3D camera-position tutorial's structure
(https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering,
see model.py in this folder) as directly as possible, extended from the
tutorial's 3-DOF camera-position-only problem to a full 6-DOF (rotation +
translation) pose:
  - One silhouette renderer, one fixed sigma (1e-4, matching the tutorial),
    faces_per_pixel=100 (matching the tutorial).
  - One Adam optimizer over all parameters jointly, lr=0.05 (matching the
    tutorial), for a single fixed run of iterations.
  - Plain MSE silhouette loss (matching the tutorial) — no centroid warm-up,
    no IoU term, no sigma annealing, no restarts.

Everything else (dataset/mesh loading, GT-pose initialization noise, crisp
full-resolution hard-mask rendering for visualization/IoU, the real-vs-tissue
appearance-mask clipping, error metrics) is unchanged from
optimize_model_inter_over_union.py — only the optimizer itself is simplified,
to compare speed/accuracy against the multi-stage version.
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
PRIMARY_CAMERA = "camera_l"
CAMERA_NAMES   = ["camera_l"]
CAMERA_DIRS    = {cam: os.path.join(DATASET_ROOT, cam, "000001") for cam in CAMERA_NAMES}

CAMERA_DIR     = CAMERA_DIRS[PRIMARY_CAMERA]
SCENE_CAM_JSON = os.path.join(CAMERA_DIR, "scene_camera.json")
SCENE_GT_JSON  = os.path.join(CAMERA_DIR, "scene_gt.json")
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")

MESH_PATH   = ("../surgical_robotics_challenge/ADF/PSMs/"
               "LND_420006/high_res/tool pitch link.OBJ")
OUTPUT_CSV  = "../Orig_Sim_CSV_Visuals/pose_estimation_results_orig.csv"
VIZ_DIR     = "../Orig_Sim_CSV_Visuals"

# Single-stage optimizer — mirrors model.py's tutorial code exactly.
SIGMA       = 1e-4   # tutorial's blend_params.sigma
NUM_ITERS   = 200    # tutorial's range(200)
LR          = 0.05   # tutorial's single Adam lr

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


# ── Load JSON for the (single) camera ──────────────────────────────────────────
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


# ── Camera / renderer helpers ──────────────────────────────────────────────────
# Screen-space (in_ndc=False) cameras with an explicit image_size, NOT
# PyTorch3D's default NDC convention -- NDC silently mishandles non-square
# aspect ratios (this dataset is not 1:1), which produced a confirmed,
# measured pixel offset against a manual-rasterizer render of identical pose
# data (see sim_iter_images.py, fixed 2026-07-10).
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


_I3 = torch.eye(3,  dtype=torch.float32, device=device)
_Z3 = torch.zeros(3, dtype=torch.float32, device=device)


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3); ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(ax * np.deg2rad(
        np.random.uniform(-rot_noise_deg, rot_noise_deg))).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Pose model — mirrors model.py's Model class exactly, extended to 6-DOF ────
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
        # Plain MSE — matches model.py's `torch.sum((image[..., 3] - self.image_ref) ** 2)`.
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
    saturation (grayish) = metal tool, high saturation (pink/red) = tissue."""
    rgb = np.array(rgb_pil, dtype=np.float32)
    mx  = rgb.max(axis=-1)
    mn  = rgb.min(axis=-1)
    sat = (mx - mn) / np.clip(mx, 1.0, None)
    return (sat < sat_threshold).astype(np.float32)


def draw_axes_and_silhouette(rgb_pil, alpha, R_bop, t_bop, cam_K,
                             color_sil=(0, 220, 0), appearance_mask=None) -> Image.Image:
    """Draws exactly two things on top of the RGB frame: a flat-shaded
    silhouette fill and the 3-axis pose gizmo."""
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
        sil_renderer = make_silhouette_renderer(fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE, sigma=SIGMA)

        if USE_GT_INITIALIZATION:
            init_R, init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
        else:
            init_R = np.eye(3, dtype=np.float64)
            init_t = t_gt.copy()

        init_rot_err   = rotation_error_deg(init_R, R_gt)
        init_trans_err = translation_error_mm(init_t, t_gt)
        print(f"    Init: rot={init_rot_err:.2f}°  trans={init_trans_err:.2f} mm")

        if USE_GT_INIT_ONLY:
            R_est_np = init_R.copy();  T_est_np = init_t.copy()
        else:
            # ── Build the one reference silhouette (rendered at GT pose) ──────
            if USE_RENDERED_REFERENCE:
                R_gt_pt3d, T_gt_pt3d = bop_to_pt3d(R_gt, t_gt)
                cam_gt = make_cameras(
                    torch.tensor(R_gt_pt3d, dtype=torch.float32, device=device).unsqueeze(0),
                    torch.tensor(T_gt_pt3d, dtype=torch.float32, device=device).unsqueeze(0),
                    fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE)
                with torch.no_grad():
                    r_gt = sil_renderer(meshes_world=mesh.clone(), cameras=cam_gt)
                ref_for_model = np.from_dlpack(r_gt[0,...,3].detach().cpu())
            else:
                ref_for_model = ref_np

            # ── Build model + single Adam optimizer over everything ───────────
            init_R_pt3d, init_t_pt3d = bop_to_pt3d(init_R, init_t)
            model = PoseModel(
                mesh, sil_renderer, ref_for_model,
                init_R_pt3d, init_t_pt3d,
                fx_r, fy_r, cx_r, cy_r, RENDER_SIZE, RENDER_SIZE,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            loop = tqdm(range(NUM_ITERS), desc=f"    obj{obj_id}")
            for _ in loop:
                optimizer.zero_grad(); loss, _, _ = model(); loss.backward(); optimizer.step()
                loop.set_description(f"    obj{obj_id} ({loss.item():.3f})")
            print(f"    Done: loss={loss.item():.4f}")

            with torch.no_grad():
                R_est_t = rot6d_to_matrix(model.rot6d)
                T_est_t = model.transl
            R_est_pt3d_np = np.from_dlpack(R_est_t.detach().cpu())
            T_est_pt3d_np = np.from_dlpack(T_est_t.detach().cpu())
            R_est_np, T_est_np = pt3d_to_bop(R_est_pt3d_np, T_est_pt3d_np)

        # ── Errors ────────────────────────────────────────────────────────────
        rot_err   = rotation_error_deg(R_est_np, R_gt)
        trans_err = translation_error_mm(T_est_np, t_gt)
        add_val   = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)

        # ── IoU (crisp binary silhouette, full resolution) ──────────────────────
        alpha_gt  = render_crisp_alpha(R_gt,     t_gt,     cam_K, mesh)
        alpha_est = render_crisp_alpha(R_est_np, T_est_np, cam_K, mesh)
        iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
        iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)
        print(f"    rot={rot_err:.2f}°  trans={trans_err:.2f}mm  ADD={add_val:.2f}mm  "
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
            d.text((img_W + 8, 5), f"EST  rot={rot_err:.1f}°  trans={trans_err:.1f}mm  "
                                    f"ADD={add_val:.1f}mm  IoU={iou_gt_vs_est:.3f}",
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
        w.writeheader();  w.writerows(csv_rows)
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
