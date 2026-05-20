'''
── Summary ────────────────────────────────────
  Frames processed:       102
  Mean rotation error:    112.1737 deg  (std 0.1728)
  Mean translation error: 41.4058 mm  (std 0.0196)
  Mean ADD:               41.4032 mm  (std 0.0196)
───────────────────────────────────────────────
'''

import sys
import os
import json
import csv
import tempfile
import re
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras,
    look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)

# ─────────────────────────────────────────────
# CONFIG — edit these paths to match your setup
# ─────────────────────────────────────────────
DATASET_ROOT   = "../SIMPLELND_dataset"
CAMERA_DIR     = os.path.join(DATASET_ROOT, "camera_0", "000001")
SCENE_CAM_JSON = os.path.join(CAMERA_DIR, "scene_camera.json")
SCENE_GT_JSON  = os.path.join(CAMERA_DIR, "scene_gt.json")
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")
MESH_PATH      = "../../surgical_robotics_challenge/ADF/PSMs/LND_420006/high_res/tool pitch link.OBJ"
OUTPUT_GIF     = "../camera0_optimization.gif"
OUTPUT_CSV     = "../pose_estimation_results.csv"
NUM_ITERS      = 200
LR             = 0.05
LOSS_THRESHOLD = 200
SAVE_EVERY     = 10
RENDER_SIZE    = 256
# ─────────────────────────────────────────────


# ── Device ────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ── Load intrinsics from scene_camera.json ────
with open(SCENE_CAM_JSON, "r") as f:
    scene_cam = json.load(f)

first_key = sorted(scene_cam.keys(), key=lambda x: int(x))[0]
cam_data  = scene_cam[first_key]
K_flat    = cam_data["cam_K"]
fx, fy    = K_flat[0], K_flat[4]
cx, cy    = K_flat[2], K_flat[5]
print(f"Loaded intrinsics — fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


# ── Load ground truth poses from scene_gt.json ──
with open(SCENE_GT_JSON, "r") as f:
    scene_gt = json.load(f)

# ── Compute mean GT translation for initialisation ──
all_translations = []
for frame_key in sorted(scene_gt.keys(), key=lambda x: int(x)):
    objects = scene_gt[frame_key]
    if len(objects) > 0:
        t = objects[0]["cam_t_m2c"]
        all_translations.append(t)

mean_t_mm   = np.mean(all_translations, axis=0)
mean_t_pt3d = torch.tensor(mean_t_mm, dtype=torch.float32, device=device)
print(f"Mean GT translation (mm): {mean_t_mm}")


# ── Load mesh ──────────────────────────────────
def load_obj_no_mtl(path):
    """Load .obj stripping mtllib/usemtl lines to avoid PyTorch3D NumPy-2 crash."""
    with open(path, "r") as f:
        lines = f.readlines()
    clean = [l for l in lines if not re.match(r'\s*(mtllib|usemtl)\b', l)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp.writelines(clean)
        tmp_path = tmp.name
    verts, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    return verts, faces_idx

verts, faces_idx = load_obj_no_mtl(MESH_PATH)
faces    = faces_idx.verts_idx
verts_np = np.from_dlpack(verts.detach().cpu())
print(f"Mesh loaded: {verts.shape[0]} verts, {faces.shape[0]} faces")


# ── Image size from first RGB file ────────────
rgb_files = sorted([
    f for f in os.listdir(RGB_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
assert len(rgb_files) > 0, f"No images found in {RGB_DIR}"
img_W, img_H = Image.open(os.path.join(RGB_DIR, rgb_files[0])).size
print(f"Image size: {img_W}x{img_H}  |  {len(rgb_files)} frames found.")

fx_ndc = 2.0 * fx / img_W
fy_ndc = 2.0 * fy / img_H
px_ndc = (img_W / 2.0 - cx) / (img_W / 2.0)
py_ndc = (img_H / 2.0 - cy) / (img_H / 2.0)


# ── Camera / renderer factories ───────────────
def make_cameras(R=None, T=None):
    if R is None:
        R = torch.eye(3, device=device).unsqueeze(0)
    if T is None:
        T = torch.zeros(1, 3, device=device)
    return PerspectiveCameras(
        focal_length=((fx_ndc, fy_ndc),),
        principal_point=((px_ndc, py_ndc),),
        R=R, T=T,
        device=device,
    )

def make_silhouette_renderer():
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_sil = RasterizationSettings(
        image_size=RENDER_SIZE,
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=100,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=make_cameras(), raster_settings=raster_sil),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


# ── Optimisation model ─────────────────────────
class PoseModel(nn.Module):
    def __init__(self, meshes, renderer, image_ref_tensor, init_position):
        super().__init__()
        self.meshes   = meshes
        self.device   = meshes.device
        self.renderer = renderer

        mask = (image_ref_tensor.max(-1).values != 1.0).to(torch.float32)
        self.register_buffer("image_ref", mask)

        self.camera_position = nn.Parameter(
            init_position.clone().to(dtype=torch.float32, device=self.device)
        )

    def forward(self):
        R = look_at_rotation(self.camera_position[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        loss  = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, R, T


# ── Error metrics ──────────────────────────────
def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_diff    = R_est @ R_gt.T
    cos_angle = float(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))

def translation_error_mm(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    return float(np.linalg.norm(t_est - t_gt))

def add_metric(verts_np, R_est, t_est, R_gt, t_gt) -> float:
    pts_est = (R_est @ verts_np.T).T + t_est
    pts_gt  = (R_gt  @ verts_np.T).T + t_gt
    return float(np.mean(np.linalg.norm(pts_est - pts_gt, axis=1)))


# ── GT pose helpers ────────────────────────────
def gt_R_t(frame_key: str):
    obj    = scene_gt[frame_key][0]
    R      = np.array(obj["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t      = np.array(obj["cam_t_m2c"], dtype=np.float64)
    return R, t


# ── Build mesh (shared across frames) ─────────
def build_mesh():
    verts_rgb = torch.ones_like(verts)[None]
    textures  = TexturesVertex(verts_features=verts_rgb.to(device))
    return Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures,
    )


# ── Main loop ──────────────────────────────────
frame_keys = sorted(scene_gt.keys(), key=lambda x: int(x))
n_frames   = min(len(rgb_files), len(frame_keys))
print(f"Processing {n_frames} frames.\n")

csv_rows = []

for idx in range(n_frames):
    frame_key = frame_keys[idx]
    rgb_path  = os.path.join(RGB_DIR, rgb_files[idx])
    print(f"── Frame {frame_key} ({rgb_files[idx]}) ──────────────────")

    # ── Load & prepare reference image ──────────
    ref_pil   = Image.open(rgb_path).convert("RGB").resize(
        (RENDER_SIZE, RENDER_SIZE), Image.BILINEAR)
    _pil_bytes = list(ref_pil.tobytes())
    ref_tensor = torch.tensor(_pil_bytes, dtype=torch.float32).reshape(
        RENDER_SIZE, RENDER_SIZE, 3) / 255.0

    # ── Ground truth ─────────────────────────────
    R_gt, t_gt = gt_R_t(frame_key)

    # ── Build fresh mesh & silhouette renderer ───
    mesh         = build_mesh()
    sil_renderer = make_silhouette_renderer()

    # ── Initialise & run optimizer ───────────────
    model     = PoseModel(mesh, sil_renderer, ref_tensor, mean_t_pt3d).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    loop = tqdm(range(NUM_ITERS), desc=f"  optimising")
    for i in loop:
        optimizer.zero_grad()
        loss, _, _ = model()
        loss.backward()
        optimizer.step()
        loop.set_description(f"  optimising (loss {loss.item():.4f})")
        if loss.item() < LOSS_THRESHOLD:
            print(f"  Converged at iter {i}, loss={loss.item():.4f}")
            break

    # ── Extract final estimated R, T ─────────────
    with torch.no_grad():
        R_est_pt = look_at_rotation(model.camera_position[None, :], device=device)
        T_est_pt = -torch.bmm(R_est_pt.transpose(1, 2),
                               model.camera_position[None, :, None])[:, :, 0]

    R_est_np = np.from_dlpack(R_est_pt[0].detach().cpu())
    T_est_np = np.from_dlpack(T_est_pt[0].detach().cpu())

    # ── Compute errors ────────────────────────────
    rot_err   = rotation_error_deg(R_est_np, R_gt)
    trans_err = translation_error_mm(T_est_np, t_gt)
    add_val   = add_metric(verts_np, R_est_np, T_est_np, R_gt, t_gt)

    print(f"  Rotation error:    {rot_err:.4f} deg")
    print(f"  Translation error: {trans_err:.4f} mm")
    print(f"  ADD:               {add_val:.4f} mm\n")

    csv_rows.append({
        "frame_id": frame_key,
        "rgb_file": rgb_files[idx],
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
        "rotation_error_deg":   rot_err,
        "translation_error_mm": trans_err,
        "ADD_mm":               add_val,
    })


# ── Write CSV ──────────────────────────────────
fieldnames = list(csv_rows[0].keys())
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer_csv = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer_csv.writeheader()
    writer_csv.writerows(csv_rows)

print(f"CSV saved to {OUTPUT_CSV}")

# ── Summary stats ──────────────────────────────
rot_errors   = [r["rotation_error_deg"]   for r in csv_rows]
trans_errors = [r["translation_error_mm"] for r in csv_rows]
add_errors   = [r["ADD_mm"]              for r in csv_rows]

print("\n── Summary ────────────────────────────────────")
print(f"  Frames processed:       {len(csv_rows)}")
print(f"  Mean rotation error:    {np.mean(rot_errors):.4f} deg  (std {np.std(rot_errors):.4f})")
print(f"  Mean translation error: {np.mean(trans_errors):.4f} mm  (std {np.std(trans_errors):.4f})")
print(f"  Mean ADD:               {np.mean(add_errors):.4f} mm  (std {np.std(add_errors):.4f})")
print("───────────────────────────────────────────────")