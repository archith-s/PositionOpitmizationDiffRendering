"""
hand_eye_diff_render_comparison.py — MARKERLESS hand-eye calibration (cam_raw_T_base,
the PSM RCM base in the raw right-camera frame) by differentiable rendering of the
LND clevis (SurgRIPE joint.stl), compared against the marker-based calibrations of
a hand-eye capture.

Three calibrations, all scored against the same reference:

  * GROUND TRUTH : marker registration + 3D pose alignment (camera_registration.py
                   + hand_eye_rodrigues_dataset.py GUI) -- calibration_variants.json's
                   meta.cam_T_base_post_alignment.
  * MARKER ONLY  : the raw camera_registration.py result -- meta.cam_T_base_pre_alignment.
  * DIFF-RENDER  : estimated here with NO location prior of any kind.

What the differentiable rendering is given (and nothing else):
  * the rectified right AND left frames (right_rect-/left_rect-XXXX.png),
  * the camera calibration: right-rect intrinsics, rect_T_cam_raw, and the stereo
    projection matrices (calibration_variants.json meta.stereo_rectification),
  * the CAD model (joint.stl), and
  * the PSM kinematics (base_T_gripper / base_T_shaft_end) -- the same robot data the
    marker-based hand-eye calibration consumes; a camera<-base calibration cannot be
    expressed without it.
It never sees cam_T_* / cam_raw_T_* (they bake in the marker calibration), any
cam_T_base variant, or any annotation. load_reference_calibrations() -- the only
reader of those -- is called after the fit, for scoring.

Pipeline:
  1. TOOL BOX (stereo, per frame): SGBM disparity with the right image as reference
     (this rig's rectified pair is mirrored relative to OpenCV's left/right convention),
     left-right consistency check, a RANSAC plane for the board (disparity is affine in
     (x, y) for a plane), "near" = clearly in front of it. The tip is the largest
     near + metallic (low-saturation, bright) blob not touching the image border (the
     shaft enters from the border, the tip doesn't); frames whose blob depth disagrees
     with the other frames re-pick among their candidates.
  2. MASK: GrabCut initialised with that box.
  3. INIT: PnP (RANSAC, SQPnP) of the mask centroids vs the clevis centres from the
     kinematics -- a coarse calibration from images + kinematics alone.
  4. DIFF-RENDER: hand_eye_dr_refine.py's joint fit (one shared 6-DoF correction over
     all frames, same renderer/edge LUT/stages). After the first pass the masks are
     re-seeded with that script's GrabCut at the CURRENT DR estimate (clipped to the
     box), and the jaw/shaft/prong ignore zones always come from the current estimate.
     Low-IoU frames are dropped and the fit redone.
  5. EVAL (first --eval-frames frames): the clevis rendered at each calibration vs at
     the ground truth -- translation error (clevis origin, mm), rotation error (deg),
     IoU of the two silhouettes -- plus the IoU of each against the image mask.

OUTPUT: <eval-dir>/eval_<dataset>_markerless_dr_comparison.pdf + .json, and
<root>/hand_eye_diff_render_comparison/ (boxes, masks, overlays).

Run from PoseOptimizerCode/ with the diff_render env (FoundationPose/miniconda3/envs/diff_render)
or foundationpose; both have PyTorch3D + CUDA.
"""

import os, re, json, math, time, struct, hashlib, tempfile, datetime
import argparse
import numpy as np
import cv2
import trimesh
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from PIL import Image, ImageDraw, ImageFont

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
from pytorch3d.transforms import so3_exp_map


# ─────────────────────────── CONFIG ───────────────────────────────────────────
HAND_EYE_ROOT = "/home/aseshad4/hand_eye_collection_scripts/gt_dataset"
MODEL_ROOT    = "/home/aseshad4/TEST"                  # joint.stl + joint.npy
EVAL_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Eval_Results_DiffRender_HandEye")

GT_JSON       = "gripper_ground_truth.json"
VARIANTS_JSON = "calibration_variants.json"
IMAGE_FIELD   = "right_rect_image"
LEFT_FIELD    = "left_rect_image"
K_FIELD       = "camera_matrix_right_rect"
OUT_SUBDIR    = "hand_eye_diff_render_comparison"
# The ONLY per-record fields the calibration may read. cam_T_* / cam_raw_T_* are absent on purpose.
ALLOWED_FIELDS = ("frame_index", IMAGE_FIELD, LEFT_FIELD, K_FIELD, "rect_T_cam_raw",
                  "base_T_gripper", "base_T_shaft_end")

# ── Renderer (same values as hand_eye_dr_refine.py / hand_eye_diff_render_test.py) ──
RENDER_SIZE     = 192
OPT_FACES       = 1000
FACES_PER_PIXEL = 8
SIGMA_PX        = 0.5
BLUR_P_CUT      = 1e-2
CALIB_POSES     = 30
MESH_TAG        = "joint.stl"
PROXY_OBJ       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes",
                               f"joint_lnd_proxy_{OPT_FACES}.obj")   # see hand_eye_dr_refine.py

CROP_MARGIN_INIT_MM = 12.0
CROP_MARGIN_MM      = 2.0

# ── Tool geometry around the clevis (mm), as hand_eye_dr_refine.py ─────────────
JAW_R0_MM          = 2.0
JAW_LEN_MM         = 16.0
JAW_HALF_ANGLE_DEG = 35.0
JAW_HALF_WIDTH_MM  = 2.5
SHAFT_RADIUS_MM    = 4.4
SHAFT_S0_MM        = 3.3
SHAFT_S1_MM        = 150.0
RING_LEN_MM        = 9.0
PRONG_S0_MM        = -2.2
PRONG_LAT_MM       = (3.2, 4.6)
PRONG_HALF_MM      = 2.5
IGNORE_DILATE_MM   = 1.0

# ── GrabCut (estimate-seeded passes, as hand_eye_dr_refine.py) ────────────────
GC_ROI_MM   = 10.0
GC_BG_MM    = 3.5
GC_ITERS    = 6
GC_SEED_MAX_SAT = 110
JAW_SEED_HALF_ANGLE_DEG = 6.0
JAW_SEED_LEN_MM         = 9.0
JAW_SEED_HALF_WIDTH_MM  = 1.2

# ── Stereo tool box (pass 1 masks; image-only) ─────────────────────────────────
STEREO_SCALE    = 2       # SGBM at half resolution
MIN_DISP_PX     = 16.0    # full-res; farther than ~0.6 m is outside any endoscope's working range -> mismatch
NEAR_RATIO      = 1.12    # "in front of the board" = disparity > this x the board plane's
METAL_MAX_SAT   = 90
METAL_MIN_VAL   = 90
BOX_MARGIN      = 0.2     # of the blob's larger side
BOX_CLIP_PX     = 40      # estimate-seeded masks are kept within the box grown by this
DEPTH_AGREE     = 0.25    # a frame's tip blob may differ from the median tip disparity by this fraction

# ── Init ──────────────────────────────────────────────────────────────────────
PNP_RANSAC_PX = 150.0     # mask centroid vs clevis centre differ by up to ~80px (jaws / ring in the mask)

# ── Optimizer (as hand_eye_dr_refine.py) ──────────────────────────────────────
AREA_W     = 5e3
IOU_W      = 1.0
BATCH_SIZE = 16

SHARED_STAGES = [
    # kind      iters  lr_t   lr_r    img_blur  cosine  recrop
    ("centroid", 30,   1.0,   0.0,    0,        True,   False),
    ("mse",      30,   0.3,   0.0,    2,        True,   True),
    ("mse_iou",  90,   0.1,   0.003,  1,        False,  True),
    ("mse_iou",  90,   0.03,  0.001,  0,        False,  True),
]
OUTLIER_IOU      = 0.5
OUTLIER_IOU_DROP = 0.15

# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--hand-eye-root", type=str, default=HAND_EYE_ROOT,
                    help="Dataset directory (gripper_ground_truth.json + calibration_variants.json + *_rect-*.png)")
parser.add_argument("--model-root", type=str, default=MODEL_ROOT, help="Directory holding joint.stl + joint.npy")
parser.add_argument("--eval-dir", type=str, default=EVAL_DIR, help="Where the PDF / JSON go")
parser.add_argument("--out-dir", type=str, default=None, help=f"Intermediate outputs (default: <root>/{OUT_SUBDIR})")
parser.add_argument("--eval-frames", type=int, default=5, help="Report the first N frames")
parser.add_argument("--exclude", type=int, nargs="*", default=[], help="frame_index values to leave out")
parser.add_argument("--mask-passes", type=int, default=4,
                    help="Fit rounds; round 1 uses the stereo-box masks, later rounds re-seed from the DR estimate")
parser.add_argument("--batch", type=int, default=BATCH_SIZE)
parser.add_argument("--reuse-fit", action="store_true",
                    help="Skip the fit: rebuild the report from the previous run's diff-render result (the eval "
                         "JSON) and stereo boxes. Fits are non-deterministic, so this keeps the numbers fixed.")
args = parser.parse_args()

ROOT    = args.hand_eye_root
DATASET = os.path.basename(os.path.normpath(ROOT))
OUT_DIR = args.out_dir or os.path.join(ROOT, OUT_SUBDIR)
for sub in ("", "masks", "overlays"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
os.makedirs(args.eval_dir, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Hand-eye data → {ROOT}\nOutput        → {OUT_DIR}\nEval          → {args.eval_dir}\nDevice        → {device}")


# NumPy <-> torch always goes through DLPack (see hand_eye_dr_refine.py: the diff_render env's torch
# was built against NumPy 1.x and runs with NumPy 2).
def to_torch(a, dtype=torch.float32) -> torch.Tensor:
    a = np.ascontiguousarray(a)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8)
    return torch.from_dlpack(a).to(device=device, dtype=dtype)


def to_np(t: torch.Tensor) -> np.ndarray:
    return np.from_dlpack(t.detach().cpu().contiguous())


_arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=device)
GRID_Y, GRID_X = torch.meshgrid(_arange, _arange, indexing="ij")


# ── Pose helpers ──────────────────────────────────────────────────────────────
def joint_model_pose(T_shaft, T_gripper) -> np.ndarray:
    """Clevis (joint.stl) pose from the shaft-end (pitch pin) and gripper (yaw
    pin) frames -- as hand_eye_dr_refine.py."""
    shaft_origin = T_shaft[:3, 3]
    forward = T_gripper[:3, 3] - shaft_origin
    forward = forward / np.linalg.norm(forward)
    z_axis = T_gripper[:3, 0] - forward * np.dot(T_gripper[:3, 0], forward)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = -forward
    x_axis = np.cross(y_axis, z_axis)
    M = np.eye(4)
    M[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    M[:3, 3] = shaft_origin
    return M


def to_mm(T) -> np.ndarray:
    T = np.array(T, dtype=np.float64).reshape(4, 4).copy()
    T[:3, 3] *= 1000.0
    return T


def to_m(T) -> np.ndarray:
    T = np.array(T, dtype=np.float64).copy()
    T[:3, 3] /= 1000.0
    return T


def rot_deg(Ra, Rb) -> float:
    return float(np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude()))


def correction_matrix(Rd, pivot, tau) -> np.ndarray:
    """D = rotate by Rd about pivot, then translate by tau (raw camera frame, mm)."""
    D = np.eye(4)
    D[:3, :3] = Rd
    D[:3, 3] = (np.eye(3) - Rd) @ pivot + tau
    return D


# ── Load dataset (calibration inputs only) ────────────────────────────────────
with open(os.path.join(ROOT, GT_JSON)) as f:
    _records = [{k: r[k] for k in ALLOWED_FIELDS if k in r}
                for r in sorted(json.load(f), key=lambda r: int(r["frame_index"]))]
with open(os.path.join(ROOT, VARIANTS_JSON)) as f:
    _stereo = json.load(f)["meta"]["stereo_rectification"]   # camera calibration only

frames = []
for rec in _records:
    fid = int(rec["frame_index"])
    if fid in args.exclude:
        continue
    missing = [k for k in ALLOWED_FIELDS if k not in rec]
    if missing:
        raise SystemExit(f"frame {fid}: {GT_JSON} lacks {missing}")
    paths = [os.path.join(ROOT, rec[IMAGE_FIELD]), os.path.join(ROOT, rec[LEFT_FIELD])]
    if not all(os.path.exists(p) for p in paths):
        print(f"  frame {fid}: rectified image(s) missing, skipped")
        continue
    frames.append(dict(
        frame_id=fid, img_path=paths[0], left_path=paths[1],
        K3=np.array(rec[K_FIELD], dtype=np.float64),
        rect=to_mm(rec["rect_T_cam_raw"]),
        B=to_mm(joint_model_pose(np.array(rec["base_T_shaft_end"]), np.array(rec["base_T_gripper"]))),  # base_T_clevis
        G=to_mm(rec["base_T_gripper"]),                                                                    # base_T_gripper
    ))
if len(frames) < 3:
    raise SystemExit(f"need at least 3 frames, have {len(frames)}")
N = len(frames)
print(f"Frames: {N}  ({', '.join(str(f['frame_id']) for f in frames)})")

RECT = frames[0]["rect"]
if any(np.abs(f["rect"] - RECT).max() > 1e-9 for f in frames):
    raise SystemExit("rect_T_cam_raw differs between records -- not supported")

img_H, img_W = cv2.imread(frames[0]["img_path"]).shape[:2]
print(f"Image size: {img_W}×{img_H}   K: fx={frames[0]['K3'][0, 0]:.2f} cx={frames[0]['K3'][0, 2]:.2f} "
      f"cy={frames[0]['K3'][1, 2]:.2f}")


# ── Mesh / renderer / masks / joint fit: verbatim from hand_eye_dr_refine.py ──
def stl_to_obj(stl_path: str, obj_path: str):
    with open(stl_path, 'rb') as f:
        f.read(80)
        struct.unpack('<I', f.read(4))
        raw = f.read()
    vert_map, verts_l, faces_l = {}, [], []
    for i in range(len(raw) // 50):
        face = []
        for k in range(3):
            v = struct.unpack_from('<fff', raw, i * 50 + 12 + 12 * k)
            if v not in vert_map:
                vert_map[v] = len(verts_l) + 1
                verts_l.append(v)
            face.append(vert_map[v])
        faces_l.append(face)
    with open(obj_path, 'w') as fh:
        for x, y, z in verts_l:
            fh.write(f'v {x:.6f} {y:.6f} {z:.6f}\n')
        for a, b, c in faces_l:
            fh.write(f'f {a} {b} {c}\n')


def load_obj_no_mtl(path: str):
    with open(path) as fh:
        lines = fh.readlines()
    clean = [l for l in lines if not re.match(r"\s*(mtllib|usemtl)\b", l)]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".obj", delete=False) as tmp:
        tmp.writelines(clean)
        tmp_path = tmp.name
    v, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    return v, faces_idx.verts_idx


_obj_cache = os.path.join(tempfile.gettempdir(), "joint_lnd_handeye_converted.obj")
if not os.path.exists(_obj_cache):
    print("Converting joint.stl → OBJ (first run only) ...")
    stl_to_obj(os.path.join(args.model_root, "joint.stl"), _obj_cache)
verts, faces = load_obj_no_mtl(_obj_cache)
verts_np = np.from_dlpack(verts.detach().cpu())
_vmax = float(np.abs(verts_np).max())
if _vmax < 1.0:
    verts, verts_np = verts * 1000.0, verts_np * 1000.0

_bb_lo, _bb_hi = verts_np.min(0), verts_np.max(0)
SPHERE_C_NP = 0.5 * (_bb_lo + _bb_hi)
SPHERE_C = to_torch(SPHERE_C_NP)
SPHERE_R = float(np.linalg.norm(verts_np - SPHERE_C_NP, axis=1).max())
_BB_CORNERS = np.array([[x, y, z] for x in (_bb_lo[0], _bb_hi[0]) for y in (_bb_lo[1], _bb_hi[1])
                        for z in (_bb_lo[2], _bb_hi[2])], dtype=np.float64)

_trimesh_full = trimesh.Trimesh(vertices=np.array(verts_np, dtype=np.float64),
                                faces=np.array(to_np(faces), dtype=np.int64), process=False)
# The optimization proxy is decimated once and saved, so every env renders the same proxy. The renderer
# settings were tuned on open3d's quadric decimation (trimesh < 4.5, e.g. foundationpose), which hits
# OPT_FACES exactly; trimesh >= 4.5 uses fast_simplification instead (e.g. diff_render), which stalls at
# ~5400 faces on this non-watertight STL -- 5x the tuned proxy, and the self-test fails (1.5 mm / 10 deg).
if os.path.exists(PROXY_OBJ):
    _trimesh_opt = trimesh.load(PROXY_OBJ, force="mesh", process=False)
    _proxy_src = PROXY_OBJ
else:
    _trimesh_opt = _trimesh_full.simplify_quadric_decimation(face_count=OPT_FACES)
    if len(_trimesh_opt.faces) > 1.2 * OPT_FACES:
        raise SystemExit(
            f"Decimating joint.stl gave {len(_trimesh_opt.faces)} faces, not ~{OPT_FACES} (this env's trimesh "
            f"{trimesh.__version__} uses fast_simplification, which can't reduce this mesh that far). Run the "
            f"script once in the foundationpose env (open3d decimation) to write {PROXY_OBJ}; every env then "
            f"loads it.")
    _trimesh_opt.export(PROXY_OBJ)
    _proxy_src = f"decimated now, saved to {PROXY_OBJ}"
verts_opt = to_torch(_trimesh_opt.vertices).cpu()
faces_opt = to_torch(_trimesh_opt.faces, torch.int64).cpu()
print(f"Mesh: {faces.shape[0]} faces, optimization proxy {faces_opt.shape[0]} faces ({_proxy_src})")

model_pts = np.load(os.path.join(args.model_root, "joint.npy")).astype(np.float64)
if np.abs(model_pts).max() < 1.0:
    model_pts = model_pts * 1000.0
model_pts = model_pts[np.random.default_rng(0).choice(len(model_pts), min(3000, len(model_pts)), replace=False)]


# ── Renderers (from hand_eye_diff_render_test.py) ─────────────────────────────
_mesh_cache = {}


def batch_meshes(B) -> Meshes:
    if B not in _mesh_cache:
        v, f = verts_opt.to(device), faces_opt.to(device)
        _mesh_cache[B] = Meshes(verts=[v] * B, faces=[f] * B,
                                textures=TexturesVertex(verts_features=[torch.ones_like(v)] * B))
    return _mesh_cache[B]


_FLIP_T = to_torch(np.diag([-1., -1., 1.]))


def bop_to_pt3d_batch(R, t):
    return R.transpose(1, 2) @ _FLIP_T, t @ _FLIP_T


def window_cameras(R, t, win):
    Rp, Tp = bop_to_pt3d_batch(R, t)
    return PerspectiveCameras(focal_length=win[:, :2], principal_point=win[:, 2:], R=Rp, T=Tp, in_ndc=False,
                              image_size=torch.tensor([[RENDER_SIZE, RENDER_SIZE]] * R.shape[0], device=device),
                              device=device)


def crop_window(R, t, K, margin_mm):
    c = R @ SPHERE_C + t
    z = c[:, 2].clamp(min=1.0)
    u = K[:, 0] * c[:, 0] / z + K[:, 2]
    v = K[:, 1] * c[:, 1] / z + K[:, 3]
    half = K[:, 0] * (SPHERE_R + margin_mm) / z
    s = RENDER_SIZE / (2 * half)
    return torch.stack([K[:, 0] * s, K[:, 1] * s, (K[:, 2] - (u - half)) * s, (K[:, 3] - (v - half)) * s], 1)


def make_soft_renderer() -> MeshRenderer:
    sigma = (SIGMA_PX * 2.0 / RENDER_SIZE) ** 2
    raster = RasterizationSettings(
        image_size=RENDER_SIZE, blur_radius=float(np.log(1.0 / BLUR_P_CUT - 1.0)) * sigma,
        faces_per_pixel=FACES_PER_PIXEL, bin_size=16, max_faces_per_bin=faces_opt.shape[0])
    return MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster),
                        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=sigma)))


soft_renderer = make_soft_renderer()
_mesh_full = Meshes(verts=[verts.float().to(device)], faces=[faces.to(device)],
                    textures=TexturesVertex(verts_features=[torch.ones_like(verts.float()).to(device)]))


def render_crisp_alpha_t(R_bop, t_bop, K3) -> torch.Tensor:
    """Crisp full-resolution silhouette of the full mesh (rasterizes only the
    window around the projected bounding box)."""
    R = to_torch(R_bop)[None]
    t = to_torch(t_bop)[None]
    out = torch.zeros(img_H, img_W, device=device)
    pc = np.asarray(R_bop, dtype=np.float64) @ _BB_CORNERS.T + np.asarray(t_bop, dtype=np.float64)[:, None]
    if (pc[2] <= 1.0).any():
        u0, v0, u1, v1 = 0, 0, img_W, img_H
    else:
        uu = K3[0, 0] * pc[0] / pc[2] + K3[0, 2]
        vv = K3[1, 1] * pc[1] / pc[2] + K3[1, 2]
        u0, v0 = max(0, int(math.floor(uu.min() - 2))), max(0, int(math.floor(vv.min() - 2)))
        u1, v1 = min(img_W, int(math.ceil(uu.max() + 2))), min(img_H, int(math.ceil(vv.max() + 2)))
    if u1 <= u0 or v1 <= v0:
        return out
    h, w = v1 - v0, u1 - u0
    Rp, Tp = bop_to_pt3d_batch(R, t)
    cam = PerspectiveCameras(focal_length=((K3[0, 0], K3[1, 1]),), principal_point=((K3[0, 2] - u0, K3[1, 2] - v0),),
                             R=Rp, T=Tp, in_ndc=False, image_size=((h, w),), device=device)
    rast = MeshRasterizer(raster_settings=RasterizationSettings(
        image_size=(h, w), blur_radius=0.0, faces_per_pixel=1,
        bin_size=max(32, 1 << int(math.ceil(math.log2(max(h, w) / 16)))), max_faces_per_bin=faces.shape[0]))
    with torch.no_grad():
        out[v0:v1, u0:u1] = (rast(_mesh_full, cameras=cam).pix_to_face[0, ..., 0] >= 0).float()
    return out


# ── Binary mask -> window target, via the edge-profile table ──────────────────
_SS = 4


def mask_to_window(full_mask, K, win, ss=_SS):
    g = (torch.arange(RENDER_SIZE * ss, device=device, dtype=torch.float32) + 0.5) / ss - 0.5
    gy, gx = torch.meshgrid(g, g, indexing="ij")
    out = []
    for b in range(full_mask.shape[0]):
        s = win[b, 0] / K[b, 0]
        X = (gx - win[b, 2]) / s + K[b, 2]
        Y = (gy - win[b, 3]) / s + K[b, 3]
        grid = torch.stack([(X + 0.5) / img_W * 2 - 1, (Y + 0.5) / img_H * 2 - 1], -1)[None]
        out.append(F.grid_sample(full_mask[b][None, None], grid, mode="bilinear",
                                 align_corners=False, padding_mode="zeros")[0, 0])
    return torch.stack(out)


def mask_signed_distance(full_mask, K, win):
    m = to_np(mask_to_window(full_mask, K, win) > 0.5).astype(np.uint8)
    out = []
    for b in range(m.shape[0]):
        d_in  = cv2.distanceTransform(m[b],     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_out = cv2.distanceTransform(1 - m[b], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out.append(np.where(m[b] > 0, d_in - 0.5, -(d_out - 0.5)) / _SS)
    sdf = to_torch(np.stack(out))
    return F.avg_pool2d(sdf[:, None], _SS)[:, 0]


def window_weight(full_valid, K, win):
    """Fraction of each window pixel inside the valid (non-ignored, in-image) region."""
    return F.avg_pool2d(mask_to_window(full_valid, K, win)[:, None], _SS)[:, 0]


LUT_EDGES = torch.linspace(-6, 6, 241, device=device)


def synthetic_calibration_poses(n, K3, seed=99):
    rng = np.random.default_rng(seed)
    Kinv = np.linalg.inv(K3)
    c0 = 0.5 * (_bb_lo + _bb_hi)
    Rs, ts = [], []
    for _ in range(n):
        ray = Kinv @ np.array([rng.uniform(0.15, 0.85) * img_W, rng.uniform(0.15, 0.85) * img_H, 1.0])
        ray /= np.linalg.norm(ray)
        a, phi, roll = np.radians(rng.uniform(20, 90)), rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        p1 = np.cross(ray, [0, 1, 0]); p1 /= np.linalg.norm(p1); p2 = np.cross(ray, p1)
        y = -(math.cos(a) * ray + math.sin(a) * (math.cos(phi) * p1 + math.sin(phi) * p2))
        tmp = np.array([1., 0, 0]) if abs(y[0]) < 0.9 else np.array([0, 1., 0])
        z = np.cross(tmp, y); z /= np.linalg.norm(z); x = np.cross(y, z)
        R = np.column_stack([x, y, z]) @ Rotation.from_rotvec([0, roll, 0]).as_matrix()
        Rs.append(R); ts.append(ray * rng.uniform(65, 115) - R @ c0)
    return np.stack(Rs), np.stack(ts)


def edge_profile_table(K3) -> torch.Tensor:
    """Mean soft-rendered alpha vs signed distance to the true edge, cached on
    disk. The key includes the proxy actually loaded (face count + a vertex
    checksum), not just OPT_FACES: a table made for one proxy is wrong for
    another, and the old OPT_FACES-only key let a 5400-face proxy reuse the
    1000-face table."""
    key = json.dumps([MESH_TAG, RENDER_SIZE, OPT_FACES, FACES_PER_PIXEL, SIGMA_PX, BLUR_P_CUT, CALIB_POSES,
                      K3.round(3).tolist(), img_W, img_H, faces.shape[0], verts.shape[0], round(_vmax, 4),
                      int(faces_opt.shape[0]), round(float(verts_opt.double().abs().sum()), 3)])
    path = os.path.join(tempfile.gettempdir(), f"hand_eye_dr_edge_lut_{hashlib.md5(key.encode()).hexdigest()[:12]}.npy")
    if os.path.exists(path):
        print(f"Edge-profile table: cached ({path})")
        return to_torch(np.load(path))
    print(f"Edge-profile table: calibrating on {CALIB_POSES} synthetic poses (once) ...")
    Rs, ts = synthetic_calibration_poses(CALIB_POSES, K3)
    K = torch.tensor([[K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]]] * CALIB_POSES, dtype=torch.float32, device=device)
    R = to_torch(Rs); t = to_torch(ts)
    full = torch.stack([render_crisp_alpha_t(Rs[i], ts[i], K3) for i in range(CALIB_POSES)])
    sums = torch.zeros(len(LUT_EDGES) + 1, device=device); cnts = torch.zeros_like(sums)
    for margin in (CROP_MARGIN_MM + 1.0, 6.0, CROP_MARGIN_INIT_MM):
        win = crop_window(R, t, K, margin)
        sdf = mask_signed_distance(full, K, win)
        with torch.no_grad():
            a = soft_renderer(meshes_world=batch_meshes(CALIB_POSES), cameras=window_cameras(R, t, win))[..., 3]
        idx = torch.bucketize(sdf.flatten(), LUT_EDGES)
        sums.index_add_(0, idx, a.flatten()); cnts.index_add_(0, idx, torch.ones_like(a.flatten()))
    lut = sums / cnts.clamp(min=1)
    np.save(path, to_np(lut))
    return lut


def gblur(x, px):
    if not px:
        return x
    r = int(math.ceil(3 * px))
    g = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (g / px) ** 2); k = k / k.sum()
    y = F.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(y, k.view(1, 1, -1, 1), padding=(r, 0))[:, 0]


# ── Per-frame poses under a calibration X (4x4, mm, cam_raw_T_base) ───────────
def frame_poses_np(X, f):
    """-> cam_rect_T_clevis, cam_rect_T_gripper (mm) and the RCM position in the rect frame."""
    T = f["rect"] @ X
    return T @ f["B"], T @ f["G"], T[:3, 3]


def pixels_per_mm(T_clevis, K3) -> float:
    return float(K3[0, 0] / max((T_clevis[:3, :3] @ SPHERE_C_NP + T_clevis[:3, 3])[2], 1.0))


# ── Tool zones around the clevis (projected, never rendered) ──────────────────
def tool_zone_points(T_clevis, T_gripper, rcm):
    pin = T_clevis[:3, 3]
    a = rcm - pin; a /= np.linalg.norm(a)                       # pitch pin -> RCM, along the shaft
    e_p = T_clevis[:3, 0] - a * np.dot(T_clevis[:3, 0], a); e_p /= np.linalg.norm(e_p)   # pitch-pin axis
    e3 = np.cross(a, e_p)
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    circle = SHAFT_RADIUS_MM * (np.cos(phi)[:, None] * e_p + np.sin(phi)[:, None] * e3)

    # The shaft runs toward the camera (the RCM sits near the endoscope), so
    # clip it 10mm in front of the camera -- points behind it would be dropped
    # by hull_mask and the zone would collapse to the near end cap.
    s_cam = (pin[2] - 10.0) / -a[2] if a[2] < 0 else np.inf

    def cylinder(s0, s1):
        s1 = max(s0, min(s1, s_cam))
        return np.concatenate([pin + s0 * a + circle, pin + s1 * a + circle])

    def prong(sign):
        return np.array([pin + s * a + sign * l * e_p + w * e3
                         for s in (PRONG_S0_MM, SHAFT_S0_MM) for l in PRONG_LAT_MM for w in (-PRONG_HALF_MM, PRONG_HALF_MM)])

    o, gx, gy, gz = T_gripper[:3, 3], T_gripper[:3, 0], T_gripper[:3, 1], T_gripper[:3, 2]

    def wedge(r0, r1, half_deg, half_w):
        th = np.radians(np.linspace(-half_deg, half_deg, 9))
        return np.array([o + x * gx + r * (np.cos(t) * gz + np.sin(t) * gy)
                         for x in (-half_w, half_w) for r in (r0, r1) for t in th])

    return dict(
        shaft=cylinder(SHAFT_S0_MM, SHAFT_S1_MM), prong_p=prong(1.0), prong_m=prong(-1.0),
        jaw=wedge(JAW_R0_MM, JAW_LEN_MM, JAW_HALF_ANGLE_DEG, JAW_HALF_WIDTH_MM),
        ring_seed=cylinder(SHAFT_S0_MM, SHAFT_S0_MM + RING_LEN_MM),
        shaft_far=cylinder(SHAFT_S0_MM + RING_LEN_MM + 1.0, SHAFT_S1_MM),   # dark shaft past the silver ring
        jaw_seed=wedge(JAW_R0_MM, JAW_SEED_LEN_MM, JAW_SEED_HALF_ANGLE_DEG, JAW_SEED_HALF_WIDTH_MM),
    )


def hull_mask(pts3d, K3) -> np.ndarray:
    m = np.zeros((img_H, img_W), np.uint8)
    pts = pts3d[pts3d[:, 2] > 1.0]
    if len(pts) < 3:
        return m
    uv = (K3 @ pts.T).T
    uv = np.clip(uv[:, :2] / uv[:, 2:], -1e5, 1e5)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv).astype(np.int32)), 1)
    return m


def disk(r_px):
    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def valid_map(f, X) -> np.ndarray:
    """1 where the loss counts, 0 over the jaws / shaft / ring prongs (dilated)."""
    T_c, T_g, rcm = frame_poses_np(X, f)
    z = tool_zone_points(T_c, T_g, rcm)
    ign = np.zeros((img_H, img_W), np.uint8)
    for k in ("shaft", "prong_p", "prong_m", "jaw"):
        ign |= hull_mask(z[k], f["K3"])
    ign = cv2.dilate(ign, disk(IGNORE_DILATE_MM * pixels_per_mm(T_c, f["K3"])))
    return (1 - ign).astype(np.float32)


def grabcut_mask(img, f, X) -> np.ndarray:
    """Silver-tool mask (uint8 0/1) from GrabCut seeded at the clevis/jaw/ring
    projected under calibration X.

    Features are (HSV saturation, HSV value, Lab b) rather than BGR: the metal
    is near-achromatic, the wood saturated and the tape strongly blue, which
    BGR GMMs kept confusing. Seed pixels that are clearly saturated are not
    seeds (the projected clevis overlaps background where X is off), and the
    dark shaft past the ring -- ignored by the loss anyway -- is painted a
    colour that occurs nowhere else and fixed as background, so it can't drag
    the blue tape into the foreground model."""
    T_c, T_g, rcm = frame_poses_np(X, f)
    K3 = f["K3"]
    ppm = pixels_per_mm(T_c, K3)
    clevis = to_np(render_crisp_alpha_t(T_c[:3, :3], T_c[:3, 3], K3).cpu()) > 0.5
    z = tool_zone_points(T_c, T_g, rcm)
    seed = clevis | (hull_mask(z["jaw_seed"], K3) > 0) | (hull_mask(z["ring_seed"], K3) > 0)
    if not seed.any():
        return np.zeros((img_H, img_W), np.uint8)
    ys, xs = np.nonzero(seed)
    m = int(GC_ROI_MM * ppm)
    x0, x1 = max(0, xs.min() - m), min(img_W, xs.max() + m + 1)
    y0, y1 = max(0, ys.min() - m), min(img_H, ys.max() + m + 1)
    sub = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    feat = cv2.GaussianBlur(np.dstack([hsv[..., 1], hsv[..., 2], cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)[..., 2]]),
                            (5, 5), 0)
    far = (hull_mask(z["shaft_far"], K3) > 0)[y0:y1, x0:x1]
    seed_roi = (seed[y0:y1, x0:x1] & ~far).astype(np.uint8)
    feat[far] = (255, 0, 255)

    gc = np.full(seed_roi.shape, cv2.GC_PR_BGD, np.uint8)
    dist = cv2.distanceTransform(1 - seed_roi, cv2.DIST_L2, 5)
    gc[dist > GC_BG_MM * ppm] = cv2.GC_BGD
    gc[(seed_roi > 0) & (feat[..., 0] <= GC_SEED_MAX_SAT)] = cv2.GC_PR_FGD
    gc[far] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(feat, gc, None, bgd, fgd, GC_ITERS, cv2.GC_INIT_WITH_MASK)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, disk(max(1.0, 0.15 * ppm)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, disk(max(2.0, 0.4 * ppm)))
    # keep the blobs touching the projected clevis; fill their holes
    near = cv2.dilate(clevis[y0:y1, x0:x1].astype(np.uint8), disk(1.0 * ppm))
    n_lab, lab = cv2.connectedComponents(fg)
    keep = np.isin(lab, [l for l in range(1, n_lab) if (near[lab == l] > 0).any()])
    out_roi = np.zeros_like(fg)
    cnts, _ = cv2.findContours(keep.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out_roi, cnts, -1, 1, thickness=cv2.FILLED)
    out = np.zeros((img_H, img_W), np.uint8)
    out[y0:y1, x0:x1] = out_roi
    return out


# ── Optimization ──────────────────────────────────────────────────────────────
def frame_losses(kind, a, tgt, tgt_b, W, area_t, cx_t, cy_t, blur):
    if kind == "centroid":
        aw = a * W
        area = aw.sum((1, 2)).clamp(min=1e-6)
        cx = (GRID_X * aw).sum((1, 2)) / area
        cy = (GRID_Y * aw).sum((1, 2)) / area
        return (cx - cx_t) ** 2 + (cy - cy_t) ** 2 + AREA_W * (torch.sqrt(area / area_t) - 1) ** 2
    loss = ((gblur(a, blur) - tgt_b).pow(2) * W).sum((1, 2)) / area_t
    if kind == "mse_iou":
        inter = (a * tgt * W).sum((1, 2))
        union = ((a + tgt - a * tgt) * W).sum((1, 2)).clamp(min=1e-6)
        loss = loss + IOU_W * (1 - inter / union)
    return loss


def run_stages(stages, pose_fn, t_params, r_params, data, lut, desc):
    """pose_fn(idx) -> (R (b,3,3), t (b,3) mm) of the clevis in the rect frame
    for frame indices idx. Loss is the mean over frames; frames are rendered
    in chunks of --batch with gradients accumulated."""
    n = data["masks"].shape[0]
    idx_all = torch.arange(n, device=device)
    K = data["K"]
    win = None
    for si, (kind, iters, lr_t, lr_r, blur, cosine, recrop) in enumerate(stages):
        if win is None or recrop:
            with torch.no_grad():
                R, t = pose_fn(idx_all)
                win = crop_window(R, t, K, CROP_MARGIN_INIT_MM if win is None else CROP_MARGIN_MM)
                tgt = lut[torch.bucketize(mask_signed_distance(data["masks"], K, win), LUT_EDGES)]
                W = window_weight(data["valid"], K, win)
                tw = tgt * W
                area_t = tw.sum((1, 2)).clamp(min=1.0)
                cx_t = (GRID_X * tw).sum((1, 2)) / area_t
                cy_t = (GRID_Y * tw).sum((1, 2)) / area_t
        tgt_b = gblur(tgt, blur)
        groups = [{"params": t_params, "lr": lr_t}]
        if lr_r > 0:
            groups.append({"params": r_params, "lr": lr_r})
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters) if cosine else None
        for _ in tqdm(range(iters), desc=f"  {desc} stage {si + 1}/{len(stages)} ({kind})", leave=False):
            opt.zero_grad()
            for c0 in range(0, n, args.batch):
                idx = idx_all[c0:c0 + args.batch]
                R, t = pose_fn(idx)
                a = soft_renderer(meshes_world=batch_meshes(len(idx)), cameras=window_cameras(R, t, win[idx]))[..., 3]
                loss = frame_losses(kind, a, tgt[idx], tgt_b[idx], W[idx], area_t[idx], cx_t[idx], cy_t[idx], blur)
                (loss.sum() / n).backward()
            opt.step()
            if sched:
                sched.step()


def build_data(sel, masks_np, valid_np):
    return dict(
        masks=to_torch(np.stack([masks_np[i] for i in sel])),
        valid=to_torch(np.stack([valid_np[i] for i in sel])),
        K=torch.tensor([[frames[i]["K3"][0, 0], frames[i]["K3"][1, 1], frames[i]["K3"][0, 2], frames[i]["K3"][1, 2]]
                        for i in sel], dtype=torch.float32, device=device),
        RB=to_torch(np.stack([frames[i]["B"][:3, :3] for i in sel])),
        tB=to_torch(np.stack([frames[i]["B"][:3, 3] for i in sel])),
    )


def tool_pivot(X, sel) -> np.ndarray:
    """Mean clevis-centre position (raw camera frame, mm) — the rotation pivot."""
    return np.mean([(X @ frames[i]["B"])[:3, :3] @ SPHERE_C_NP + (X @ frames[i]["B"])[:3, 3] for i in sel], axis=0)


def fit_shared(sel, X_start, masks_np, valid_np, lut, desc="joint") -> np.ndarray:
    """Optimize D (rotation about the tool pivot + translation) shared by the
    frames in sel; returns D · X_start (mm)."""
    data = build_data(sel, masks_np, valid_np)
    f32 = to_torch
    pivot = tool_pivot(X_start, sel)
    R0, t0, p = f32(X_start[:3, :3]), f32(X_start[:3, 3]), f32(pivot)
    Rr, tr = f32(RECT[:3, :3]), f32(RECT[:3, 3])
    omega = torch.zeros(3, device=device, requires_grad=True)
    tau = torch.zeros(3, device=device, requires_grad=True)

    def pose_fn(idx):
        Rd = so3_exp_map(omega[None])[0]
        R_x = Rd @ R0
        t_x = Rd @ (t0 - p) + p + tau
        R = (Rr @ R_x) @ data["RB"][idx]
        t = data["tB"][idx] @ (Rr @ R_x).T + (Rr @ t_x + tr)
        return R, t

    run_stages(SHARED_STAGES, pose_fn, [tau], [omega], data, lut, desc)
    with torch.no_grad():
        Rd = to_np(so3_exp_map(omega[None])[0].double())
    return correction_matrix(Rd, pivot, to_np(tau.double())) @ X_start


# ── Metrics ───────────────────────────────────────────────────────────────────
def iou_valid(i, X, masks_np, valid_np) -> float:
    T = frame_poses_np(X, frames[i])[0]
    r = render_crisp_alpha_t(T[:3, :3], T[:3, 3], frames[i]["K3"]) > 0.5
    m = to_torch(masks_np[i] > 0, torch.bool)
    v = to_torch(valid_np[i] > 0.5, torch.bool)
    union = ((r | m) & v).sum().item()
    return float(((r & m) & v).sum().item() / union) if union else 0.0


def pose_displacement_mm(Ta, Tb) -> float:
    """Mean distance between clevis model points under two poses (ADD)."""
    pa = model_pts @ Ta[:3, :3].T + Ta[:3, 3]
    pb = model_pts @ Tb[:3, :3].T + Tb[:3, 3]
    return float(np.linalg.norm(pa - pb, axis=1).mean())


def calib_change(Xa, Xb, sel):
    """(mean clevis displacement over the frames in mm, rotation change deg) between two calibrations."""
    d = np.mean([pose_displacement_mm(frame_poses_np(Xa, frames[i])[0], frame_poses_np(Xb, frames[i])[0]) for i in sel])
    return float(d), rot_deg(Xa[:3, :3], Xb[:3, :3])


# ── Stereo tool box (images + stereo calibration only) ────────────────────────
# The right camera's P[0,3] = -fx·Tx. On this rig it is positive: a scene point sits further RIGHT in the
# right image than in the left (x_right = x_left + d), the mirror of OpenCV's convention, so SGBM is run with
# the right image as its reference ("left") image. The sign is read from the calibration, not assumed.
_RIGHT_IS_LEFT = float(np.array(_stereo["right"]["P"])[0, 3]) > 0


def _sgbm():
    return cv2.StereoSGBM_create(minDisparity=0, numDisparities=160, blockSize=5, P1=8 * 3 * 25, P2=32 * 3 * 25,
                                 uniquenessRatio=12, speckleWindowSize=100, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_HH)


def right_disparity(L, R) -> np.ndarray:
    """Full-res-unit disparity (>0) for each right-image pixel at 1/STEREO_SCALE resolution, 0 where invalid
    (SGBM failure, left-right inconsistency, or implausibly far)."""
    s = STEREO_SCALE
    Ls, Rs = [cv2.resize(x, None, fx=1 / s, fy=1 / s, interpolation=cv2.INTER_AREA) for x in (L, R)]
    sg = _sgbm()
    if _RIGHT_IS_LEFT:
        d_r = sg.compute(Rs, Ls).astype(np.float32) / 16                                  # right = reference
        d_l = cv2.flip(sg.compute(cv2.flip(Ls, 1), cv2.flip(Rs, 1)), 1).astype(np.float32) / 16
        sgn = -1
    else:
        d_r = cv2.flip(sg.compute(cv2.flip(Rs, 1), cv2.flip(Ls, 1)), 1).astype(np.float32) / 16
        d_l = sg.compute(Ls, Rs).astype(np.float32) / 16
        sgn = 1
    w = d_r.shape[1]
    x_l = np.clip(np.round(np.arange(w)[None, :] + sgn * d_r).astype(int), 0, w - 1)
    consistent = np.abs(d_r - np.take_along_axis(d_l, x_l, axis=1)) <= 2
    d = d_r * s
    return np.where((d >= MIN_DISP_PX) & consistent, d, 0).astype(np.float32)


def tip_candidates(L, R):
    """Near + metallic blobs not touching the border, largest first: (area, median disparity, full-res box)."""
    s = STEREO_SCALE
    d = right_disparity(L, R)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ok = d > 0
    A = np.column_stack([xx[ok], yy[ok], np.ones(ok.sum())]); dv = d[ok]
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):                       # board plane: disparity is affine in (x, y) for a plane
        idx = rng.choice(len(dv), 3, replace=False)
        try:
            coef = np.linalg.solve(A[idx], dv[idx])
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(A @ coef - dv) < 3.0
        if best is None or inl.sum() > best.sum():
            best = inl
    coef = np.linalg.lstsq(A[best], dv[best], rcond=None)[0]
    plane = coef[0] * xx + coef[1] * yy + coef[2]
    hsv = cv2.cvtColor(cv2.resize(R, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    cand = ok & (d > NEAR_RATIO * plane) & (hsv[..., 1] < METAL_MAX_SAT) & (hsv[..., 2] > METAL_MIN_VAL)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = st[k]
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:    # the shaft enters from the border, the tip doesn't
            continue
        dk = d[(lab == k) & ok]
        if len(dk) < 20:
            continue
        m = BOX_MARGIN * max(bw, bh)
        box = [int(max(0, x - m) * s), int(max(0, y - m) * s), int(min(w, x + bw + m) * s), int(min(h, y + bh + m) * s)]
        out.append((int(area), float(np.median(dk)), box))
    return sorted(out, key=lambda c: -c[0])


def stereo_boxes(images, lefts):
    """Largest candidate per frame, then re-pick in frames whose tip depth disagrees with the rest."""
    cands = [tip_candidates(lefts[i], images[i]) for i in tqdm(range(len(images)), desc="  stereo boxes", leave=False)]
    first = [c[0] for c in cands if c]
    if not first:
        raise SystemExit("no tool tip found in any frame")
    d_med = float(np.median([c[1] for c in first]))
    boxes = []
    for c in cands:
        agree = [x for x in c if abs(x[1] / d_med - 1) <= DEPTH_AGREE]
        pick = agree[0] if agree else (c[0] if c else None)
        boxes.append(pick[2] if pick else None)
    return boxes


def box_grabcut(img, box) -> np.ndarray:
    """GrabCut initialised with the box (same S/V/Lab-b features as grabcut_mask); largest blob, holes filled."""
    x0, y0, x1, y1 = box
    m = 60
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(img_W, x1 + m), min(img_H, y1 + m)
    sub = img[Y0:Y1, X0:X1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    feat = cv2.GaussianBlur(np.dstack([hsv[..., 1], hsv[..., 2], cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)[..., 2]]), (5, 5), 0)
    gc = np.zeros(sub.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(feat, gc, (x0 - X0, y0 - Y0, x1 - x0, y1 - y0), bgd, fgd, GC_ITERS, cv2.GC_INIT_WITH_RECT)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, disk(4))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, disk(8))
    out = np.zeros((img_H, img_W), np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        keep = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        roi = np.zeros_like(keep)
        cv2.drawContours(roi, cnts, -1, 1, thickness=cv2.FILLED)
        out[Y0:Y1, X0:X1] = roi
    return out


def box_region(box, pad) -> np.ndarray:
    r = np.zeros((img_H, img_W), np.uint8)
    x0, y0, x1, y1 = box
    r[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = 1
    return r


def pnp_init(sel, masks) -> np.ndarray:
    """cam_raw_T_base from mask centroids vs kinematic clevis centres (RANSAC SQPnP, then refit on inliers)."""
    p3 = np.array([frames[i]["B"][:3, :3] @ SPHERE_C_NP + frames[i]["B"][:3, 3] for i in sel])
    p2 = np.array([[np.nonzero(masks[i])[1].mean(), np.nonzero(masks[i])[0].mean()] for i in sel], dtype=np.float64)
    K3 = frames[0]["K3"]
    ok, rv, tv, inl = cv2.solvePnPRansac(p3, p2, K3, None, reprojectionError=PNP_RANSAC_PX,
                                         iterationsCount=500, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inl is None or len(inl) < 4:
        ok, rv, tv = cv2.solvePnP(p3, p2, K3, None, flags=cv2.SOLVEPNP_SQPNP)
        inl = np.arange(len(sel))[:, None]
    else:
        inl = inl.ravel()
        _, rv, tv = cv2.solvePnP(p3[inl], p2[inl], K3, None, flags=cv2.SOLVEPNP_SQPNP)
        inl = inl[:, None]
    Xr = np.eye(4)
    Xr[:3, :3] = cv2.Rodrigues(rv)[0]
    Xr[:3, 3] = tv.ravel()
    print(f"PnP init: {len(inl)}/{len(sel)} frames as inliers")
    return np.linalg.inv(RECT) @ Xr       # rect-frame result -> raw camera frame


def iou_full(a, b) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def render_mask(X, i) -> np.ndarray:
    T = frame_poses_np(X, frames[i])[0]
    return to_np(render_crisp_alpha_t(T[:3, :3], T[:3, 3], frames[i]["K3"]).cpu()) > 0.5


# ══ Main: markerless calibration ══════════════════════════════════════════════
t_start = time.time()
images = [cv2.imread(f["img_path"]) for f in frames]
ALL = list(range(N))
stem = f"eval_{DATASET}_markerless_dr_comparison"
pdf_path = os.path.join(args.eval_dir, stem + ".pdf")
json_path = os.path.join(args.eval_dir, stem + ".json")

if args.reuse_fit:
    # Only the fit's own outputs are taken from the previous run; the reference calibrations are re-read below.
    with open(json_path) as fh:
        prev = json.load(fh)
    with open(os.path.join(OUT_DIR, "stereo_boxes.json")) as fh:
        _boxes = json.load(fh)
    by_id = {f["frame_id"]: i for i, f in enumerate(frames)}
    X_init, X_dr = to_mm(prev["cam_T_base_pnp_init"]), to_mm(prev["cam_T_base_diff_render"])
    USE = [by_id[fid] for fid in prev["tip_found"]]
    inliers = [by_id[fid] for fid in prev["inliers"]]
    boxes = [_boxes.get(str(f["frame_id"])) for f in frames]
    args.mask_passes = prev["mask_passes"]
    masks = None
    fit_runtime = prev.get("fit_runtime_s")
    print(f"Reusing the diff-render result of {prev['generated']} ({json_path})")
else:
    lut = edge_profile_table(frames[0]["K3"])
    lefts = [cv2.imread(f["left_path"]) for f in frames]
    boxes = stereo_boxes(images, lefts)
    for i in ALL:
        if boxes[i] is None:
            print(f"  frame {frames[i]['frame_id']}: no tool tip found -- left out")
    USE = [i for i in ALL if boxes[i] is not None]
    if len(USE) < 3:
        raise SystemExit("tool tip found in fewer than 3 frames")
    box_masks = [box_grabcut(images[i], boxes[i]) if boxes[i] is not None else np.zeros((img_H, img_W), np.uint8)
                 for i in ALL]
    USE = [i for i in USE if box_masks[i].sum() > 0]

    X_init = pnp_init(USE, box_masks)
    X = X_init.copy()
    masks = box_masks
    for pas in range(args.mask_passes):
        if pas > 0:
            masks = [grabcut_mask(images[i], frames[i], X) * box_region(boxes[i], BOX_CLIP_PX) if i in USE
                     else box_masks[i] for i in tqdm(ALL, desc="  GrabCut masks", leave=False)]
        valid = [valid_map(frames[i], X) for i in ALL]
        X_prev, X = X, fit_shared(USE, X, masks, valid, lut, desc=f"pass {pas + 1}/{args.mask_passes}")
        dp, rp = calib_change(X_prev, X, USE)
        print(f"Pass {pas + 1}/{args.mask_passes}: moved {dp:.2f} mm / {rp:.2f} deg, mean valid IoU "
              f"{np.mean([iou_valid(i, X, masks, valid) for i in USE]):.3f}")

    ious = {i: iou_valid(i, X, masks, valid) for i in USE}
    thr = max(OUTLIER_IOU, float(np.median(list(ious.values()))) - OUTLIER_IOU_DROP)
    inliers = [i for i in USE if ious[i] >= thr]
    if len(inliers) < 3:
        print(f"Fewer than 3 frames reach IoU {thr:.3f} -- keeping all")
        inliers = USE
    elif len(inliers) < len(USE):
        print(f"Outliers (IoU < {thr:.3f}): {[frames[i]['frame_id'] for i in USE if i not in inliers]} "
              f"-- refitting on {len(inliers)} frames")
        X = fit_shared(inliers, X, masks, valid, lut, desc="inliers")
    X_dr = X
    fit_runtime = time.time() - t_start


# ══ Evaluation (reference calibrations are read only from here on) ════════════
def load_reference_calibrations():
    with open(os.path.join(ROOT, VARIANTS_JSON)) as fh:
        meta = json.load(fh)["meta"]
    return to_mm(meta["cam_T_base_post_alignment"]), to_mm(meta["cam_T_base_pre_alignment"])


X_gt, X_marker = load_reference_calibrations()
METHODS = [("hand-eye (marker only)", X_marker), ("diff-render (markerless)", X_dr)]


def frame_errors(X, i):
    """Clevis pose under X vs under the ground truth, in the rectified camera frame."""
    T, Tg = frame_poses_np(X, frames[i])[0], frame_poses_np(X_gt, frames[i])[0]
    return float(np.linalg.norm(T[:3, 3] - Tg[:3, 3])), rot_deg(T[:3, :3], Tg[:3, :3])


EVAL = ALL[:args.eval_frames]
rows = []
for i in EVAL:
    r_gt = render_mask(X_gt, i)
    row = dict(frame_id=frames[i]["frame_id"], inlier=i in inliers, box=boxes[i])
    for name, Xc in METHODS:
        r = render_mask(Xc, i)
        t_err, r_err = frame_errors(Xc, i)
        row[name] = dict(trans_err_mm=t_err, rot_err_deg=r_err, iou_vs_gt=iou_full(r, r_gt),
                         add_mm=pose_displacement_mm(frame_poses_np(Xc, frames[i])[0], frame_poses_np(X_gt, frames[i])[0]))
    rows.append(row)

overall = {}
for name, Xc in METHODS:
    d, r = calib_change(X_gt, Xc, ALL)
    overall[name] = dict(
        mean_clevis_displacement_mm=d, rotation_err_deg=r,
        base_origin_err_mm=float(np.linalg.norm(Xc[:3, 3] - X_gt[:3, 3])),
        eval_mean_trans_err_mm=float(np.mean([row[name]["trans_err_mm"] for row in rows])),
        eval_mean_rot_err_deg=float(np.mean([row[name]["rot_err_deg"] for row in rows])),
        eval_mean_iou_vs_gt=float(np.mean([row[name]["iou_vs_gt"] for row in rows])))
d0, r0 = calib_change(X_gt, X_init, ALL)

# ── Text summary ──────────────────────────────────────────────────────────────
lines = []
W_ = 104
lines.append("=" * W_)
lines.append("hand_eye_diff_render_comparison.py — markerless hand-eye calibration by differentiable rendering".center(W_))
lines.append("=" * W_)
lines.append(f"Dataset   : {ROOT}  ({N} frames; tip found in {len(USE)}, {len(inliers)} kept after outlier rejection)")
lines.append(f"Generated : {datetime.datetime.now().isoformat(timespec='seconds')}   fit runtime "
             f"{f'{fit_runtime:.0f}s' if fit_runtime is not None else 'n/a (report rebuilt with --reuse-fit)'}")
lines.append("Ground truth = marker registration + 3D pose alignment (cam_T_base_post_alignment); used ONLY for scoring.")
lines.append("DR inputs   = rectified right+left images, intrinsics / rect_T_cam_raw / stereo P, joint.stl, PSM kinematics.")
lines.append("              No marker calibration, no ground truth, no initial pose, no annotation.")
lines.append(f"Pipeline    = stereo tool box -> GrabCut -> PnP init -> {args.mask_passes} DR passes (masks re-seeded from the")
lines.append("              DR estimate after pass 1) -> outlier rejection + refit.")
lines.append("")
lines.append("Whole calibration vs ground truth (all frames):")
lines.append(f"  {'':<26}{'clevis displ. (mm)':>20}{'rotation (deg)':>16}{'base origin (mm)':>18}")
lines.append(f"  {'PnP init (no prior)':<26}{d0:>20.2f}{r0:>16.2f}{np.linalg.norm(X_init[:3, 3] - X_gt[:3, 3]):>18.2f}")
for name, _ in METHODS:
    o = overall[name]
    lines.append(f"  {name:<26}{o['mean_clevis_displacement_mm']:>20.2f}{o['rotation_err_deg']:>16.2f}"
                 f"{o['base_origin_err_mm']:>18.2f}")
lines.append("")
lines.append(f"First {len(EVAL)} frames (clevis pose vs its pose under the ground truth; IoU of the rendered silhouettes):")
hdr = f"  {'frame':>5} {'kept':>4} " + "".join(f"{n[:14] + ' t(mm)':>23}{'rot(deg)':>10}{'IoU':>7}" for n, _ in METHODS)
lines.append(hdr)
lines.append("  " + "-" * (len(hdr) - 2))
for row in rows:
    lines.append(f"  {row['frame_id']:>5} {'y' if row['inlier'] else 'n':>4} " +
                 "".join(f"{row[n]['trans_err_mm']:>23.2f}{row[n]['rot_err_deg']:>10.2f}{row[n]['iou_vs_gt']:>7.3f}"
                         for n, _ in METHODS))
lines.append("  " + "-" * (len(hdr) - 2))
lines.append(f"  {'mean':>5} {'':>4} " + "".join(
    f"{overall[n]['eval_mean_trans_err_mm']:>23.2f}{overall[n]['eval_mean_rot_err_deg']:>10.2f}"
    f"{overall[n]['eval_mean_iou_vs_gt']:>7.3f}" for n, _ in METHODS))
lines.append("  (t = clevis origin / pitch pin; rotation is the calibration's rotation error, the same for every frame)")
lines.append("")
lines.append("cam_raw_T_base (m), diff-render:")
lines += ["  " + np.array2string(row, precision=6, suppress_small=True) for row in to_m(X_dr)]
lines.append("=" * W_)
print("\n" + "\n".join(lines))


# ── Overlays + PDF + JSON ─────────────────────────────────────────────────────
def panel(i, Xc, col):
    """Image crop with the clevis contour at Xc (method colour) and at the ground truth (orange)."""
    o = images[i].copy()
    rects = []
    for Xd, c in ((X_gt, GT_COL), (Xc, col)):
        r = render_mask(Xd, i).astype(np.uint8)
        cnts, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cnts, -1, c, 3)
        if cnts:
            rects.append(cv2.boundingRect(np.concatenate(cnts)))
    if rects:
        x0 = min(b[0] for b in rects); y0 = min(b[1] for b in rects)
        x1 = max(b[0] + b[2] for b in rects); y1 = max(b[1] + b[3] for b in rects)
        m = int(0.35 * max(x1 - x0, y1 - y0)) + 30
        o = o[max(0, y0 - m):min(img_H, y1 + m), max(0, x0 - m):min(img_W, x1 + m)]
    return cv2.cvtColor(o, cv2.COLOR_BGR2RGB)


for i in ALL:
    fid = frames[i]["frame_id"]
    cv2.imwrite(os.path.join(OUT_DIR, "masks", f"mask-{fid:04d}.png"), masks[i] * 255)
    o = images[i].copy()
    if boxes[i] is not None:
        cv2.rectangle(o, tuple(boxes[i][:2]), tuple(boxes[i][2:]), (0, 0, 255), 3)
    cv2.imwrite(os.path.join(OUT_DIR, "overlays", f"box-{fid:04d}.jpg"), o)
with open(os.path.join(OUT_DIR, "stereo_boxes.json"), "w") as fh:
    json.dump({str(frames[i]["frame_id"]): boxes[i] for i in ALL}, fh)

stem = f"eval_{DATASET}_markerless_dr_comparison"
pdf_path = os.path.join(args.eval_dir, stem + ".pdf")
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=6.6, va="top", ha="left")
    pdf.savefig(fig); plt.close(fig)
    for row in rows:
        i = next(k for k in ALL if frames[k]["frame_id"] == row["frame_id"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
        for ax, (name, Xc, col) in zip(axes, METHODS):
            m = row[name]
            ax.imshow(panel(i, Xc, col))
            ax.set_title(f"{name}  vs  ground truth\ntranslation error {m['trans_err_mm']:.2f} mm   "
                         f"rotation error {m['rot_err_deg']:.2f} deg   IoU {m['iou_vs_gt']:.3f}",
                         fontsize=9, color=np.array(col[::-1]) / 255 * 0.8)
            ax.axis("off")
        fig.suptitle(f"{DATASET} frame {row['frame_id']}: clevis rendered at each calibration (coloured) and at "
                     f"the ground truth (orange){'' if row['inlier'] else '   [frame not used by the DR fit]'}",
                     fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        pdf.savefig(fig); plt.close(fig)

result = dict(
    generated=datetime.datetime.now().isoformat(timespec="seconds"), dataset=os.path.abspath(ROOT),
    dr_inputs=["right_rect + left_rect images", "camera_matrix_right_rect", "rect_T_cam_raw",
               "stereo_rectification P matrices", "joint.stl", "base_T_gripper", "base_T_shaft_end"],
    frames=[f["frame_id"] for f in frames], tip_found=[frames[i]["frame_id"] for i in USE],
    inliers=[frames[i]["frame_id"] for i in inliers], mask_passes=args.mask_passes,
    cam_T_base_ground_truth=to_m(X_gt).tolist(), cam_T_base_marker_only=to_m(X_marker).tolist(),
    cam_T_base_pnp_init=to_m(X_init).tolist(), cam_T_base_diff_render=to_m(X_dr).tolist(),
    pnp_init_vs_gt=dict(mean_clevis_displacement_mm=d0, rotation_err_deg=r0),
    overall=overall, eval_frames=rows,
)
json_path = os.path.join(args.eval_dir, stem + ".json")
with open(json_path, "w") as fh:
    json.dump(result, fh, indent=2)
print(f"Report → {pdf_path}\nResult → {json_path}")
