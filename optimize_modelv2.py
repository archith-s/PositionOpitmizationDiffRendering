"""
pose_estimation_dr.py
Differentiable-rendering 6-DoF pose estimator (PyTorch3D silhouette loss).

Fixes vs. previous version
────────────────────────────
1.  Multiple objects per frame  – all obj_ids processed, not just objects[0].
2.  Frame ↔ image alignment     – explicit int→zero-padded-filename map instead
                                  of parallel-list indexing.
3.  Rotation parameterisation   – learnable 6-D vector + Gram-Schmidt →
                                  proper SO(3), replaces look_at_rotation which
                                  cannot represent arbitrary orientations.
4.  Per-object initialisation   – each obj_id is seeded with its own mean GT
                                  translation (computed once before the loop).
5.  Per-frame intrinsics        – camera matrix re-read for every frame from
                                  scene_camera.json (matters if it ever varies).
6.  numpy conversion            – np.from_dlpack() on detached CPU tensors
                                  (required; .numpy() unavailable in this env).
7.  Silhouette mask             – explicit foreground threshold replaces the
                                  "not white" heuristic.
"""

import os, sys, re, csv, json, tempfile
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation

from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, TexturesVertex,
)


def save_tensor_as_png(tensor: torch.Tensor, path: str):
    """Save a (H, W) float tensor in [0,1] as a greyscale PNG using PIL only
    (avoids torchvision / numpy incompatibility in this environment)."""
    arr = np.from_dlpack(tensor.detach().cpu().clamp(0.0, 1.0))
    arr = (arr * 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

# ─────────────────────────── CONFIG ───────────────────────────────────────────
DATASET_ROOT   = "../SIMPLELND_dataset"
CAMERA_DIR     = os.path.join(DATASET_ROOT, "camera_0", "000001")
SCENE_CAM_JSON = os.path.join(CAMERA_DIR, "scene_camera.json")
SCENE_GT_JSON  = os.path.join(CAMERA_DIR, "scene_gt.json")
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")
MESH_PATH      = ("../../surgical_robotics_challenge/ADF/PSMs/"
                  "LND_420006/high_res/tool pitch link.OBJ")
OUTPUT_CSV     = "../pose_estimation_results.csv"

NUM_ITERS      = 200
LR             = 0.05
LOSS_THRESHOLD = 200
RENDER_SIZE    = 256          # pixels (square) used for silhouette rendering
FG_THRESHOLD   = 0.1         # fraction of max brightness used as fg/bg boundary

# Set True  if the tool is BRIGHT against a dark background (AMBF simulator default)
# Set False if the tool is DARK  against a bright background (e.g. real laparoscope)
BRIGHT_FG = True

# Debug initialization settings
USE_GT_INITIALIZATION = True
INIT_ROT_NOISE_DEG    = 10.0
INIT_TRANS_NOISE_MM   = 10.0

# If True: skip the optimizer entirely and report the perturbed GT pose as the
# estimate.  Use this to verify the pipeline scores correctly when the initial
# pose is already within 10 deg / 10 mm of GT (your PI's sanity-check).
# If False: run the full differentiable-rendering optimisation loop.
USE_GT_INIT_ONLY = True

# Debug output directory – fg-mask and first-render PNGs are saved here
DEBUG_DIR = "../debug_renders"
# ──────────────────────────────────────────────────────────────────────────────


# ── Argument parsing ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--first10",
    action="store_true",
    help="Run only first 10 valid frames",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="Run all frames (default)",
)
args = parser.parse_args()

# Create debug output directory
os.makedirs(DEBUG_DIR, exist_ok=True)
print(f"Debug renders will be saved to: {DEBUG_DIR}")

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Load JSON files ────────────────────────────────────────────────────────────
with open(SCENE_CAM_JSON) as f:
    scene_cam: dict = json.load(f)

with open(SCENE_GT_JSON) as f:
    scene_gt: dict = json.load(f)

print(f"scene_gt  : {len(scene_gt)} frames")
print(f"scene_cam : {len(scene_cam)} frames")


# ── Build frame-id → image-filename map (BOP zero-padded names) ────────────────
# BOP saves rgb files as 000000.png, 000001.png …
# We build an explicit dict so frame 7 always maps to "000007.png" regardless
# of whether some frames are missing on disk.
def frame_id_to_filename(fid: int, ext: str = ".png") -> str:
    return f"{fid:06d}{ext}"

# Collect which frame ids actually have an image on disk
_rgb_on_disk = {
    int(fn.split(".")[0]): fn
    for fn in os.listdir(RGB_DIR)
    if fn.lower().endswith((".png", ".jpg", ".jpeg"))
}

# Frames present in BOTH scene_gt and rgb folder
valid_frame_ids = sorted(
    set(int(k) for k in scene_gt.keys()) & set(_rgb_on_disk.keys())
)
print(f"Valid frames (gt ∩ rgb): {len(valid_frame_ids)}")

if args.first10:
    valid_frame_ids = valid_frame_ids[:10]
    print(f"DEBUG MODE: Using first {len(valid_frame_ids)} frames only")


# ── Image dimensions (from the first image) ────────────────────────────────────
_sample_img = Image.open(os.path.join(RGB_DIR, _rgb_on_disk[valid_frame_ids[0]]))
img_W, img_H = _sample_img.size
print(f"Image size: {img_W}×{img_H}")


# ── Load mesh ──────────────────────────────────────────────────────────────────
def load_obj_no_mtl(path: str):
    """Strip mtllib/usemtl lines to avoid PyTorch3D / NumPy-2 crash."""
    with open(path) as fh:
        lines = fh.readlines()
    clean = [l for l in lines if not re.match(r"\s*(mtllib|usemtl)\b", l)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp.writelines(clean)
        tmp_path = tmp.name
    verts, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    return verts, faces_idx.verts_idx

verts, faces = load_obj_no_mtl(MESH_PATH)
verts_np = np.from_dlpack(verts.detach().cpu())

# ── Mesh scale sanity check ────────────────────────────────────────────────────
# GT translations are in mm (BOP standard: ~35–110 mm for this dataset).
# If the .OBJ was exported in meters the vertices will be ~0.001–0.1 range,
# causing a ~1000× scale mismatch and a sub-pixel silhouette → zero gradients.
_vmax = float(verts_np.__abs__().max())
print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces  |  max |vert| = {_vmax:.4f}")
if _vmax < 1.0:
    print(f"  ⚠  Vertices look like meters (max={_vmax:.4f}). "
          f"Auto-scaling ×1000 to match mm GT translations.")
    verts    = verts * 1000.0
    verts_np = verts_np * 1000.0
else:
    print(f"  ✓  Vertices look like mm (max={_vmax:.4f}). No scaling needed.")


# ── Per-object mean GT translation (used as initialisation) ────────────────────
# Collect all obj_ids that appear in the dataset
all_obj_ids = sorted({obj["obj_id"] for fdata in scene_gt.values() for obj in fdata})
print(f"Object IDs in dataset: {all_obj_ids}")

obj_mean_t: dict[int, torch.Tensor] = {}
for oid in all_obj_ids:
    ts = []
    for fdata in scene_gt.values():
        for obj in fdata:
            if obj["obj_id"] == oid:
                ts.append(obj["cam_t_m2c"])
    mean = np.mean(ts, axis=0)
    obj_mean_t[oid] = torch.tensor(mean, dtype=torch.float32, device=device)
    print(f"  obj {oid} mean t (mm): {mean}")


# ── Camera / renderer helpers ──────────────────────────────────────────────────
def intrinsics_to_ndc(cam_K: list, W: int, H: int):
    """Convert OpenCV-style K matrix to PyTorch3D NDC focal/principal."""
    fx, fy = cam_K[0], cam_K[4]
    cx, cy = cam_K[2], cam_K[5]
    fx_ndc = 2.0 * fx / W
    fy_ndc = 2.0 * fy / H
    px_ndc = (W / 2.0 - cx) / (W / 2.0)
    py_ndc = (H / 2.0 - cy) / (H / 2.0)
    return fx_ndc, fy_ndc, px_ndc, py_ndc


def make_cameras(R: torch.Tensor, T: torch.Tensor,
                 fx_ndc: float, fy_ndc: float,
                 px_ndc: float, py_ndc: float) -> PerspectiveCameras:
    return PerspectiveCameras(
        focal_length=((fx_ndc, fy_ndc),),
        principal_point=((px_ndc, py_ndc),),
        R=R, T=T,
        device=device,
    )


def make_silhouette_renderer(fx_ndc, fy_ndc, px_ndc, py_ndc):
    blend = BlendParams(sigma=1e-4, gamma=1e-4)
    raster = RasterizationSettings(
        image_size=RENDER_SIZE,
        blur_radius=float(np.log(1.0 / 1e-4 - 1.0)) * blend.sigma,
        faces_per_pixel=100,
    )
    cam = make_cameras(
        torch.eye(3, device=device).unsqueeze(0),
        torch.zeros(1, 3, device=device),
        fx_ndc, fy_ndc, px_ndc, py_ndc,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=blend),
    )


def build_mesh() -> Meshes:
    verts_rgb = torch.ones_like(verts)[None]
    return Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=TexturesVertex(verts_features=verts_rgb.to(device)),
    )


# ── 6-D → SO(3) (Gram-Schmidt, continuous representation) ─────────────────────
def rot6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """
    r6d : (6,)  first two columns of the rotation matrix, column-major.
    Returns a (3,3) rotation matrix.
    """
    a1 = r6d[:3]
    a2 = r6d[3:]
    b1 = nn.functional.normalize(a1, dim=0)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum() * b1, dim=0)
    b3 = torch.linalg.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=1)          # (3,3)


def matrix_to_rot6d(R: np.ndarray) -> torch.Tensor:
    """Convert a (3,3) numpy rotation matrix to the 6-D representation."""
    R_t = torch.tensor(R, dtype=torch.float32, device=device)
    return R_t[:, :2].T.reshape(6)                   # first two columns → (6,)


def perturb_pose(
    R_gt: np.ndarray,
    t_gt: np.ndarray,
    rot_noise_deg: float = 10.0,
    trans_noise_mm: float = 10.0,
):
    """Return a pose initialisation within rot_noise_deg / trans_noise_mm of GT."""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    angle_deg = np.random.uniform(-rot_noise_deg, rot_noise_deg)
    angle_rad = np.deg2rad(angle_deg)

    R_noise = Rotation.from_rotvec(axis * angle_rad).as_matrix()
    R_init  = R_noise @ R_gt

    t_init = t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, size=3)

    return R_init, t_init


# ── Pose optimisation model ────────────────────────────────────────────────────
class PoseModel(nn.Module):
    """
    Optimise R (via 6-D parameterisation) and T (translation vector) directly
    to minimise silhouette loss against a reference image.
    """

    def __init__(self, mesh: Meshes, renderer, image_ref_np: np.ndarray,
                 init_R: np.ndarray, init_t: np.ndarray,
                 fx_ndc, fy_ndc, px_ndc, py_ndc):
        super().__init__()
        self.mesh     = mesh
        self.renderer = renderer
        self.fx_ndc   = fx_ndc
        self.fy_ndc   = fy_ndc
        self.px_ndc   = px_ndc
        self.py_ndc   = py_ndc

        # Foreground mask from reference image
        ref_t = torch.tensor(image_ref_np, dtype=torch.float32, device=device)
        # Normalised brightness in [0, 1]
        brightness = ref_t.mean(-1) / ref_t.max().clamp(min=1e-6)
        if BRIGHT_FG:
            # AMBF / simulator: tool is bright, background is dark
            fg_mask = (brightness > FG_THRESHOLD).float()
        else:
            # Real laparoscope: tool is dark, background is bright
            fg_mask = (brightness < (1.0 - FG_THRESHOLD)).float()
        self.register_buffer("image_ref", fg_mask)

        # Learnable parameters
        self.rot6d = nn.Parameter(matrix_to_rot6d(init_R))
        self.transl = nn.Parameter(
            torch.tensor(init_t, dtype=torch.float32, device=device)
        )

    def forward(self):
        R = rot6d_to_matrix(self.rot6d).unsqueeze(0)   # (1,3,3)
        T = self.transl.unsqueeze(0)                    # (1,3)

        cameras = make_cameras(R, T,
                               self.fx_ndc, self.fy_ndc,
                               self.px_ndc, self.py_ndc)
        image = self.renderer(
            meshes_world=self.mesh.clone(),
            cameras=cameras,
        )                                               # (1, H, W, 4)
        loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, R, T


# ── Error metrics ──────────────────────────────────────────────────────────────
def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    cos_a = float(np.clip((np.trace(R_est @ R_gt.T) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))

def translation_error_mm(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_est - t_gt))

def add_metric(verts_np: np.ndarray,
               R_est: np.ndarray, t_est: np.ndarray,
               R_gt:  np.ndarray, t_gt:  np.ndarray) -> float:
    pts_est = (R_est @ verts_np.T).T + t_est
    pts_gt  = (R_gt  @ verts_np.T).T + t_gt
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


# ── Main loop ──────────────────────────────────────────────────────────────────
csv_rows = []

for frame_id in valid_frame_ids:
    fkey     = str(frame_id)
    rgb_path = os.path.join(RGB_DIR, _rgb_on_disk[frame_id])

    # Per-frame camera intrinsics
    cam_K = scene_cam[fkey]["cam_K"]
    fx_ndc, fy_ndc, px_ndc, py_ndc = intrinsics_to_ndc(cam_K, img_W, img_H)

    # Load & resize reference image
    ref_pil    = Image.open(rgb_path).convert("RGB").resize(
        (RENDER_SIZE, RENDER_SIZE), Image.BILINEAR)
    ref_np     = np.array(ref_pil, dtype=np.float32)          # (H, W, 3)

    print(f"\n══ Frame {frame_id:04d}  ({_rgb_on_disk[frame_id]}) ══")

    objects_in_frame = scene_gt[fkey]

    for obj_data in objects_in_frame:
        obj_id = obj_data["obj_id"]

        # Ground truth pose
        R_gt = np.array(obj_data["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
        t_gt = np.array(obj_data["cam_t_m2c"], dtype=np.float64)

        print(f"  ── obj_id={obj_id} ──")

        # Build per-iteration mesh + renderer (renderer holds camera ref)
        mesh         = build_mesh()
        sil_renderer = make_silhouette_renderer(fx_ndc, fy_ndc, px_ndc, py_ndc)

        # Initialise pose
        if USE_GT_INITIALIZATION:
            init_R, init_t = perturb_pose(
                R_gt, t_gt,
                rot_noise_deg=INIT_ROT_NOISE_DEG,
                trans_noise_mm=INIT_TRANS_NOISE_MM,
            )
        else:
            init_R = np.eye(3, dtype=np.float64)
            init_t = np.from_dlpack(obj_mean_t[obj_id].detach().cpu())

        init_rot_err   = rotation_error_deg(init_R, R_gt)
        init_trans_err = translation_error_mm(init_t, t_gt)
        print(
            f"    Initial pose error:"
            f" rot={init_rot_err:.2f} deg,"
            f" trans={init_trans_err:.2f} mm"
        )

        if USE_GT_INIT_ONLY:
            # ── Sanity-check mode: use the perturbed GT pose directly ──────────
            # No optimisation.  This confirms the error metrics and CSV are
            # correct when the initial pose is already within 10 deg / 10 mm.
            print(f"    [GT_INIT_ONLY] Skipping optimiser – using init pose as estimate.")
            R_est_np = init_R.copy()
            T_est_np = init_t.copy()
            # ──────────────────────────────────────────────────────────────────

        else:
            # ── Full differentiable-rendering optimisation ─────────────────────
            model = PoseModel(
                mesh, sil_renderer, ref_np,
                init_R, init_t,
                fx_ndc, fy_ndc, px_ndc, py_ndc,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # Debug: save fg-mask and initial render
            _debug_tag = f"frame{frame_id:04d}_obj{obj_id}"
            _mask_path = os.path.join(DEBUG_DIR, f"fgmask_{_debug_tag}.png")
            save_tensor_as_png(model.image_ref.float(), _mask_path)
            with torch.no_grad():
                _loss0, _R0, _T0 = model()
                _cam0 = make_cameras(_R0, _T0,
                                     model.fx_ndc, model.fy_ndc,
                                     model.px_ndc, model.py_ndc)
                _rendered0 = sil_renderer(meshes_world=mesh.clone(), cameras=_cam0)
                _render_path = os.path.join(DEBUG_DIR, f"render_init_{_debug_tag}.png")
                save_tensor_as_png(_rendered0[0, ..., 3].float(), _render_path)
            print(f"    Debug images → {_mask_path}")
            print(f"                 → {_render_path}")
            print(f"    Initial loss : {_loss0.item():.2f}")

            loop = tqdm(range(NUM_ITERS), desc=f"    optimising obj {obj_id}")
            for i in loop:
                optimizer.zero_grad()
                loss, _, _ = model()
                loss.backward()
                optimizer.step()
                loop.set_description(
                    f"    optimising obj {obj_id} (loss {loss.item():.2f})")
                if loss.item() < LOSS_THRESHOLD:
                    print(f"    Converged at iter {i}, loss={loss.item():.4f}")
                    break

            with torch.no_grad():
                R_est_t = rot6d_to_matrix(model.rot6d)
                T_est_t = model.transl

            R_est_np = np.from_dlpack(R_est_t.detach().cpu())
            T_est_np = np.from_dlpack(T_est_t.detach().cpu())

            # Debug: save final render
            with torch.no_grad():
                _cam_f = make_cameras(
                    R_est_t.unsqueeze(0), T_est_t.unsqueeze(0),
                    fx_ndc, fy_ndc, px_ndc, py_ndc,
                )
                _rendered_f = sil_renderer(meshes_world=mesh.clone(), cameras=_cam_f)
                _final_path = os.path.join(DEBUG_DIR, f"render_final_{_debug_tag}.png")
                save_tensor_as_png(_rendered_f[0, ..., 3].float(), _final_path)
            # ──────────────────────────────────────────────────────────────────

        # Errors
        rot_err   = rotation_error_deg(R_est_np, R_gt)
        trans_err = translation_error_mm(T_est_np, t_gt)
        add_val   = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)

        print(f"    Rotation error    : {rot_err:.4f} deg")
        print(f"    Translation error : {trans_err:.4f} mm")
        print(f"    ADD               : {add_val:.4f} mm")

        csv_rows.append({
            "frame_id":  frame_id,
            "rgb_file":  _rgb_on_disk[frame_id],
            "obj_id":    obj_id,
            # Estimated pose
            "est_R_00": R_est_np[0,0], "est_R_01": R_est_np[0,1], "est_R_02": R_est_np[0,2],
            "est_R_10": R_est_np[1,0], "est_R_11": R_est_np[1,1], "est_R_12": R_est_np[1,2],
            "est_R_20": R_est_np[2,0], "est_R_21": R_est_np[2,1], "est_R_22": R_est_np[2,2],
            "est_tx_mm": T_est_np[0],  "est_ty_mm": T_est_np[1],  "est_tz_mm": T_est_np[2],
            # Ground truth pose
            "gt_R_00": R_gt[0,0], "gt_R_01": R_gt[0,1], "gt_R_02": R_gt[0,2],
            "gt_R_10": R_gt[1,0], "gt_R_11": R_gt[1,1], "gt_R_12": R_gt[1,2],
            "gt_R_20": R_gt[2,0], "gt_R_21": R_gt[2,1], "gt_R_22": R_gt[2,2],
            "gt_tx_mm": t_gt[0],  "gt_ty_mm": t_gt[1],  "gt_tz_mm": t_gt[2],
            # Errors
            "init_rotation_error_deg":   init_rot_err,
            "init_translation_error_mm": init_trans_err,
            "rotation_error_deg":   rot_err,
            "translation_error_mm": trans_err,
            "ADD_mm":               add_val,
        })


# ── Write CSV ──────────────────────────────────────────────────────────────────
if csv_rows:
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved → {OUTPUT_CSV}  ({len(csv_rows)} rows)")

# ── Summary ────────────────────────────────────────────────────────────────────
if csv_rows:
    rot_errs   = [r["rotation_error_deg"]   for r in csv_rows]
    trans_errs = [r["translation_error_mm"] for r in csv_rows]
    add_errs   = [r["ADD_mm"]              for r in csv_rows]

    print("\n── Summary ─────────────────────────────────────────────")
    print(f"  Rows (frames × objects) : {len(csv_rows)}")
    print(f"  Mean rotation error     : {np.mean(rot_errs):.4f} deg"
          f"  (std {np.std(rot_errs):.4f})")
    print(f"  Mean translation error  : {np.mean(trans_errs):.4f} mm"
          f"  (std {np.std(trans_errs):.4f})")
    print(f"  Mean ADD                : {np.mean(add_errs):.4f} mm"
          f"  (std {np.std(add_errs):.4f})")
    print("────────────────────────────────────────────────────────")
