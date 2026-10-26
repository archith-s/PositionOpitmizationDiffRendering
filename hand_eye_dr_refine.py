"""
hand_eye_dr_refine.py — refine a dVRK PSM hand-eye calibration (cam_raw_T_base,
the PSM RCM base in the raw right-camera frame) by differentiable rendering of
the LND clevis (SurgRIPE joint.stl) against the real right_rect frames of a
hand-eye capture.

The automatic counterpart of hand_eye_collection_scripts/hand_eye_rodrigues_dataset.py's
manual pass: instead of nudging an overlay per frame and averaging the
per-frame corrections, ONE correction D (6-DoF, applied in the raw camera
frame, rotating about the tool's mean position) is optimized jointly over
every frame, each rendered from the kinematics:

    cam_rect_T_clevis_i = rect_T_cam_raw · D · X0 · base_T_clevis_i

where base_T_clevis_i comes from base_T_shaft_end + base_T_gripper (the PSM
forward kinematics, NOT the dataset's cam_T_* fields, which already bake in
the calibration being refined). Frames seen from different angles constrain
different directions of D, so the weak per-frame silhouette depth is
averaged out across views.

  * INPUT   : <root>/gripper_ground_truth.json + right_rect-XXXX.png — the same
              dataset hand_eye_diff_render_test.py reads. X0 is the calibration
              the dataset was made with (cam_raw_T_gripper · base_T_gripper⁻¹);
              <root>/calibration_variants.json, if present, adds the
              pre-manual-alignment calibration for comparison and the path of
              the hand-eye JSON.
  * MASKS   : the capture ships none, so they are made here: GrabCut on each
              frame, seeded with the clevis rendered at the current calibration
              (+ thin jaw / shaft-ring seeds). The masks cover the whole silver
              tool, so the jaws, shaft and the ring's pitch-pin prongs (none of
              which joint.stl models) are projected from the kinematics as
              IGNORE zones excluded from the loss. Masks and ignore zones are
              regenerated after each fit (--mask-passes, default 3).
  * RENDERER: identical to hand_eye_diff_render_test.py (192px crop window,
              ~1000-face proxy, edge-profile-calibrated targets, coarse-to-fine
              stages) — only the parameters differ: one shared (omega, tau)
              instead of a per-frame pose. (A per-frame free 6-DoF fit is NOT
              used as a check: the clevis is near-symmetric about its long axis
              and the ignore zones leave that roll unconstrained per frame, so
              it drifts by 10-20 deg even on perfect masks.)
  * CHECKS  : per-frame IoU (valid region) at the pre-alignment, dataset and
              refined calibrations; frames whose IoU is poor in absolute terms
              or relative to the rest (bad mask, occlusion) are dropped and the
              joint fit redone; leave-one-out IoU (fit on N-1, score
              the held-out frame — the only accuracy estimate with no ground
              truth; masks/ignore zones come from the all-frame fit, so it is
              slightly optimistic). --self-test MM DEG instead renders masks at
              a known perturbed calibration and reports how well it is recovered.
  * OUTPUT  : <out-dir>/ masks/, overlays/, hand_eye_dr_report.pdf,
              hand_eye_dr_result.json. The refined calibration is written back
              to the hand-eye JSON only on confirmation (or --apply), the same
              way hand_eye_rodrigues_dataset.py does it: timestamped backup,
              'pre-alignment' kept, an 'alignment-history' entry appended --
              plus a 'diff-render' block (this result + the registration/GUI
              calibration it replaced) that Comparative_calibration.py reads.

Run from PoseOptimizerCode/ with the diff_render env (FoundationPose/miniconda3/envs/diff_render)
or foundationpose; both have PyTorch3D + CUDA.
"""

import os, re, json, math, time, shutil, struct, hashlib, tempfile, textwrap, datetime, sys
import argparse
import numpy as np
import cv2
import trimesh
import torch
import torch.nn.functional as F
from tqdm import tqdm
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
from pytorch3d.transforms import so3_exp_map


# ─────────────────────────── CONFIG ───────────────────────────────────────────
HAND_EYE_ROOT = "/home/aseshad4/hand_eye_collection_scripts/gt_dataset"
MODEL_ROOT    = "/home/aseshad4/TEST"                  # joint.stl + joint.npy
DEFAULT_HAND_EYE_JSON = os.path.expanduser("~/PSM1-registration-open-cv.json")

GT_JSON       = "gripper_ground_truth.json"
VARIANTS_JSON = "calibration_variants.json"
IMAGE_FIELD   = "right_rect_image"
K_FIELD       = "camera_matrix_right_rect"
OUT_SUBDIR    = "hand_eye_dr_refine"

# ── Renderer (same values as hand_eye_diff_render_test.py; see its docstring) ──
RENDER_SIZE     = 192
OPT_FACES       = 1000
FACES_PER_PIXEL = 8
SIGMA_PX        = 0.5
BLUR_P_CUT      = 1e-2
CALIB_POSES     = 30
MESH_TAG        = "joint.stl"
PROXY_OBJ       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes",
                               f"joint_lnd_proxy_{OPT_FACES}.obj")   # see the proxy section below

CROP_MARGIN_INIT_MM = 12.0
CROP_MARGIN_MM      = 2.0

# ── Tool geometry around the clevis (mm) — NOT rendered, only projected as
# loss-ignore zones and GrabCut seeds. Clevis frame: pitch pin at the origin
# along X, yaw pin 9.1mm forward, body along -Y, extent y in [-10.8, 2.8],
# |x| <= 3.1. Gripper frame: origin on the yaw pin, X = pin axis, jaws along
# Z, opening in the Y-Z plane. The shaft runs from the pitch pin to the RCM
# (the PSM base origin). The jaw angle is not recorded, so the jaw zone is a
# wedge wide enough for any opening.
JAW_R0_MM          = 2.0    # clevis fork reaches ~1.7mm past the yaw pin
JAW_LEN_MM         = 16.0
JAW_HALF_ANGLE_DEG = 35.0
JAW_HALF_WIDTH_MM  = 2.5
SHAFT_RADIUS_MM    = 4.4    # 8.4mm LND shaft / ring
SHAFT_S0_MM        = 3.3    # ring starts just behind the clevis's rounded back (r <= 3.2 about the pitch pin)
SHAFT_S1_MM        = 150.0
RING_LEN_MM        = 9.0    # silver shaft-end ring (GrabCut seed only)
PRONG_S0_MM        = -2.2   # ring prongs holding the pitch pin flank the clevis back (end ~2mm past the pin)
PRONG_LAT_MM       = (3.2, 4.6)
PRONG_HALF_MM      = 2.5
IGNORE_DILATE_MM   = 1.0    # tolerance for calibration error in the zones themselves

# ── GrabCut mask generation ────────────────────────────────────────────────────
GC_ROI_MM   = 10.0   # ROI = seeds' bbox + this
GC_BG_MM    = 3.5    # farther than this from any seed = sure background (7mm let the tape leak in)
GC_ITERS    = 6
GC_SEED_MAX_SAT = 110   # HSV saturation above which a seed pixel is background, not metal
JAW_SEED_HALF_ANGLE_DEG = 6.0
JAW_SEED_LEN_MM         = 9.0
JAW_SEED_HALF_WIDTH_MM  = 1.2

# ── Optimizer ─────────────────────────────────────────────────────────────────
AREA_W     = 5e3
IOU_W      = 1.0
BATCH_SIZE = 16      # frames rendered per backward chunk (gradients accumulate across chunks)

# Shared correction: tau (mm, raw camera frame) and omega (rad, about the pivot).
SHARED_STAGES = [
    # kind      iters  lr_t   lr_r    img_blur  cosine  recrop
    ("centroid", 30,   1.0,   0.0,    0,        True,   False),
    ("mse",      30,   0.3,   0.0,    2,        True,   True),
    ("mse_iou",  90,   0.1,   0.003,  1,        False,  True),
    ("mse_iou",  90,   0.03,  0.001,  0,        False,  True),
]
OUTLIER_IOU      = 0.5    # absolute floor on a frame's refined IoU
OUTLIER_IOU_DROP = 0.15   # ... or this far below the median frame

# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--hand-eye-root", type=str, default=HAND_EYE_ROOT,
                    help="Dataset directory (gripper_ground_truth.json + right_rect-*.png)")
parser.add_argument("--model-root", type=str, default=MODEL_ROOT, help="Directory holding joint.stl + joint.npy")
parser.add_argument("--hand-eye-json", type=str, default=None,
                    help="Hand-eye calibration JSON to update (default: calibration_variants.json's "
                         f"meta.hand_eye_json, else {DEFAULT_HAND_EYE_JSON})")
parser.add_argument("--out-dir", type=str, default=None, help=f"Output directory (default: <root>/{OUT_SUBDIR})")
parser.add_argument("--exclude", type=int, nargs="*", default=[], help="frame_index values to leave out")
parser.add_argument("--mask-passes", type=int, default=3,
                    help="GrabCut-mask + fit rounds; each round re-seeds the masks from the latest fit")
parser.add_argument("--no-loo", action="store_true", help="Skip the leave-one-out evaluation")
parser.add_argument("--self-test", type=float, nargs=2, metavar=("MM", "DEG"), default=None,
                    help="Synthetic check: render masks at the dataset calibration perturbed by MM / DEG "
                         "and report how well the perturbation is recovered. Never writes the calibration.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--batch", type=int, default=BATCH_SIZE)
group = parser.add_mutually_exclusive_group()
group.add_argument("--apply", action="store_true", help="Write the refined calibration without asking")
group.add_argument("--dry-run", action="store_true", help="Never write the hand-eye calibration")
parser.add_argument("--allow-mismatch", action="store_true",
                    help="Let --apply write even when the hand-eye JSON no longer holds the calibration this "
                         "dataset was captured with (e.g. it was re-registered for a different camera placement)")
args = parser.parse_args()

ROOT    = args.hand_eye_root
OUT_DIR = args.out_dir or os.path.join(ROOT, OUT_SUBDIR)
for sub in ("", "masks", "overlays"):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Hand-eye data → {ROOT}\nOutput        → {OUT_DIR}\nDevice        → {device}")



# NumPy <-> torch always goes through DLPack. In the diff_render env torch (2.2, built against NumPy 1.x)
# runs with NumPy 2, so torch.from_numpy / tensor.numpy() raise "Numpy is not available" and
# torch.tensor(ndarray) silently falls back to an element-by-element copy (minutes for full-frame masks).
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
    pin) frames — same construction as hand_eye_diff_render_test.py's
    cam_T_joint_model. Built only from origins and axes, so it works in any
    frame (base or camera) and commutes with rigid transforms."""
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


# ── Load dataset ──────────────────────────────────────────────────────────────
with open(os.path.join(ROOT, GT_JSON)) as f:
    _records = sorted(json.load(f), key=lambda r: int(r["frame_index"]))
_variants_meta = {}
if os.path.exists(os.path.join(ROOT, VARIANTS_JSON)):
    with open(os.path.join(ROOT, VARIANTS_JSON)) as f:
        _variants_meta = json.load(f).get("meta", {})

frames = []
for rec in _records:
    fid = int(rec["frame_index"])
    if fid in args.exclude:
        continue
    missing = [k for k in ("base_T_gripper", "base_T_shaft_end", "cam_raw_T_gripper", "rect_T_cam_raw", K_FIELD)
               if k not in rec]
    if missing:
        raise SystemExit(f"frame {fid}: {GT_JSON} lacks {missing} -- re-generate it with the current "
                         f"image_generation_gt_hand_eye.py")
    img_path = os.path.join(ROOT, rec[IMAGE_FIELD])
    if not os.path.exists(img_path):
        print(f"  frame {fid}: {img_path} missing, skipped")
        continue
    base_T_gripper = np.array(rec["base_T_gripper"], dtype=np.float64)
    frames.append(dict(
        frame_id=fid, img_path=img_path,
        K3=np.array(rec[K_FIELD], dtype=np.float64),
        rect=to_mm(rec["rect_T_cam_raw"]),
        B=to_mm(joint_model_pose(np.array(rec["base_T_shaft_end"]), base_T_gripper)),   # base_T_clevis
        G=to_mm(base_T_gripper),                                                           # base_T_gripper
        X_rec=to_mm(np.array(rec["cam_raw_T_gripper"]) @ np.linalg.inv(base_T_gripper)),
    ))
if len(frames) < 2:
    raise SystemExit(f"need at least 2 frames, have {len(frames)}")
N = len(frames)
print(f"Frames: {N}  ({', '.join(str(f['frame_id']) for f in frames)})")

X0 = frames[0]["X_rec"]
_x_spread = max(max(np.linalg.norm(f["X_rec"][:3, 3] - X0[:3, 3]), rot_deg(f["X_rec"][:3, :3], X0[:3, :3]))
                for f in frames)
if _x_spread > 1e-3:
    print(f"  WARNING: the records disagree on the calibration they were made with (max {_x_spread:.4f} mm/deg); "
          f"using frame {frames[0]['frame_id']}'s")
RECT = frames[0]["rect"]
if any(np.abs(f["rect"] - RECT).max() > 1e-9 for f in frames):
    raise SystemExit("rect_T_cam_raw differs between records -- not supported")
X_pre = to_mm(_variants_meta["cam_T_base_pre_alignment"]) if "cam_T_base_pre_alignment" in _variants_meta else None

HAND_EYE_JSON = args.hand_eye_json or _variants_meta.get("hand_eye_json") or DEFAULT_HAND_EYE_JSON

img_H, img_W = cv2.imread(frames[0]["img_path"]).shape[:2]
print(f"Image size: {img_W}×{img_H}   K: fx={frames[0]['K3'][0, 0]:.2f} cx={frames[0]['K3'][0, 2]:.2f} "
      f"cy={frames[0]['K3'][1, 2]:.2f}")


# ── Mesh (same STL→OBJ conversion + cache as hand_eye_diff_render_test.py) ────
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


# ── Hand-eye JSON write-back (as hand_eye_rodrigues_dataset.py) ───────────────
def load_hand_eye_calibration(json_file: str) -> np.ndarray:
    with open(json_file) as fh:
        return np.array(json.load(fh)['base-frame']['transform']).reshape(4, 4)


def write_hand_eye_calibration(json_file: str, cam_T_base: np.ndarray, history_entry: dict = None) -> str:
    """base-frame.transform becomes the diff-render result (what the robot and
    every script loads). 'diff-render' records that result plus the
    calibration it replaced (camera_registration.py + GUI alignment) as
    previous_transform, so Comparative_calibration.py can compare all of them.
    Re-applying on top of an earlier diff-render result keeps the original
    previous_transform rather than the old diff-render result."""
    with open(json_file) as fh:
        data = json.load(fh)
    if 'pre-alignment' not in data:
        data['pre-alignment'] = {
            'reference-frame': data['base-frame'].get('reference-frame'),
            'transform': data['base-frame']['transform'],
            'recorded_at': time.time(),
        }
    current = np.array(data['base-frame']['transform']).reshape(4, 4)
    old_dr = data.get('diff-render')
    if old_dr is not None and np.allclose(np.array(old_dr['transform']).reshape(4, 4), current, atol=1e-9):
        previous, previous_source = old_dr['previous_transform'], old_dr.get('previous_source')
    else:
        previous, previous_source = current.tolist(), 'base-frame before diff-render (registration + GUI alignment)'
    data['diff-render'] = {
        'reference-frame': data['base-frame'].get('reference-frame'),
        'transform': cam_T_base.tolist(),
        'previous_transform': previous,
        'previous_source': previous_source,
        'recorded_at': time.time(),
        'dataset': history_entry.get('dataset') if history_entry else None,
    }
    data['base-frame']['transform'] = cam_T_base.tolist()
    if history_entry is not None:
        data.setdefault('alignment-history', []).append(history_entry)
    backup_path = f'{json_file}.bak-{int(time.time())}'
    shutil.copy2(json_file, backup_path)
    with open(json_file, 'w') as fh:
        json.dump(data, fh)
    return backup_path


# ── Overlays ──────────────────────────────────────────────────────────────────
def draw_overlay(img, i, mask, valid, calibs):
    """calibs: list of (label, X, BGR colour). Returns the crop around the tool."""
    o = img.copy()
    ign = valid < 0.5
    o[ign] = (0.6 * o[ign] + 0.4 * np.array([60, 60, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(o, cnts, -1, (255, 255, 255), 2)
    boxes = [cv2.boundingRect(np.concatenate(cnts))] if cnts else []
    for label, X, col in calibs:
        T = frame_poses_np(X, frames[i])[0]
        r = to_np(render_crisp_alpha_t(T[:3, :3], T[:3, 3], frames[i]["K3"]).cpu()).astype(np.uint8)
        rc, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, rc, -1, col, 2)
        if rc:
            boxes.append(cv2.boundingRect(np.concatenate(rc)))
    if boxes:
        x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes); y1 = max(b[1] + b[3] for b in boxes)
        m = int(0.25 * max(x1 - x0, y1 - y0)) + 20
        o = o[max(0, y0 - m):min(img_H, y1 + m), max(0, x0 - m):min(img_W, x1 + m)].copy()
    for k, (label, _, col) in enumerate(calibs):
        cv2.putText(o, label, (8, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(o, label, (8, 22 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
    return o


# ══ Main ══════════════════════════════════════════════════════════════════════
t_start = time.time()
lut = edge_profile_table(frames[0]["K3"])
images = [cv2.imread(f["img_path"]) for f in frames]
ALL = list(range(N))

X_true = None
if args.self_test:
    rng = np.random.default_rng(args.seed)
    ax_r = rng.normal(size=3); ax_r /= np.linalg.norm(ax_r)
    ax_t = rng.normal(size=3); ax_t /= np.linalg.norm(ax_t)
    X_true = correction_matrix(Rotation.from_rotvec(ax_r * np.radians(args.self_test[1])).as_matrix(),
                               tool_pivot(X0, ALL), ax_t * args.self_test[0]) @ X0
    masks = []
    for i in ALL:
        T = frame_poses_np(X_true, frames[i])[0]
        masks.append((to_np(render_crisp_alpha_t(T[:3, :3], T[:3, 3], frames[i]["K3"]).cpu()) > 0.5)
                     .astype(np.uint8))
    d, r = calib_change(X0, X_true, ALL)
    print(f"\nSELF-TEST: masks rendered at the dataset calibration perturbed by {args.self_test[0]:g} mm / "
          f"{args.self_test[1]:g} deg (mean clevis displacement {d:.2f} mm, rotation {r:.2f} deg)")

# Each pass: masks + ignore zones from the current calibration, then a joint
# fit on EVERY frame. Outliers are only judged after the last pass: the frames
# that start worst are typically the ones far across the workspace, which are
# exactly the ones that pin down the rotation (on LND006, dropping frames 7-9
# after pass 1 left a calibration 8 deg off that failed on them; keeping them
# fits all ten).
X = X0.copy()
for pas in range(args.mask_passes):
    if not args.self_test:
        masks = [grabcut_mask(images[i], frames[i], X) for i in tqdm(ALL, desc="  GrabCut masks", leave=False)]
    valid = [valid_map(frames[i], X) for i in ALL]
    for i in ALL:
        if masks[i].sum() == 0:
            print(f"  frame {frames[i]['frame_id']}: empty mask")
    X_prev, X = X, fit_shared(ALL, X, masks, valid, lut, desc=f"pass {pas + 1}/{args.mask_passes}")
    d, r = calib_change(X0, X, ALL)
    dp, rp = calib_change(X_prev, X, ALL)
    print(f"Pass {pas + 1}/{args.mask_passes}: vs dataset calibration {d:.2f} mm / {r:.2f} deg "
          f"(this pass moved it {dp:.2f} mm / {rp:.2f} deg)")

# ── Outlier rejection (bad mask / occlusion) ──────────────────────────────────
ious = [iou_valid(i, X, masks, valid) for i in ALL]
thr = max(OUTLIER_IOU, float(np.median(ious)) - OUTLIER_IOU_DROP)
inliers = [i for i in ALL if ious[i] >= thr]
if len(inliers) < 2:
    print(f"Fewer than 2 frames reach IoU {thr:.3f} -- keeping all frames")
    inliers = ALL
elif len(inliers) < N:
    print(f"Outliers (IoU < {thr:.3f}): {[frames[i]['frame_id'] for i in ALL if i not in inliers]} "
          f"-- refitting on {len(inliers)} frames")
    X = fit_shared(inliers, X, masks, valid, lut, desc="inliers")
X_dr = X

# ── Leave-one-out ─────────────────────────────────────────────────────────────
loo = {}
if not args.no_loo and len(inliers) >= 3:
    print(f"\nLeave-one-out over {len(inliers)} frames ...")
    for j in inliers:
        X_j = fit_shared([i for i in inliers if i != j], X_dr, masks, valid, lut, desc=f"loo-{frames[j]['frame_id']}")
        loo[j] = dict(X=X_j, iou=iou_valid(j, X_j, masks, valid), dev=calib_change(X_j, X_dr, ALL))

# ── Per-frame table ───────────────────────────────────────────────────────────
calibs = ([("pre-alignment", X_pre)] if X_pre is not None else []) + [("dataset", X0), ("diff-render", X_dr)]
if X_true is not None:
    calibs.append(("true (self-test)", X_true))
iou_tab = {name: [iou_valid(i, Xc, masks, valid) for i in ALL] for name, Xc in calibs}
per_frame = []
for i in ALL:
    row = dict(frame_id=frames[i]["frame_id"], inlier=i in inliers,
               mask_px=int(masks[i].sum()), valid_mask_px=int((masks[i] * valid[i]).sum()),
               iou={name: iou_tab[name][i] for name, _ in calibs})
    if i in loo:
        row["loo_iou"] = loo[i]["iou"]
        row["loo_calib_dev_mm"], row["loo_calib_dev_deg"] = loo[i]["dev"]
    per_frame.append(row)

lines = []
W_ = 100
lines.append("=" * W_)
lines.append("hand_eye_dr_refine.py — hand-eye refinement by differentiable rendering (joint.stl clevis)".center(W_))
lines.append("=" * W_)
lines.append(f"Dataset      : {ROOT}  ({N} frames, {len(inliers)} inliers)")
lines.append(f"Hand-eye JSON: {HAND_EYE_JSON}")
lines.append(f"Generated    : {datetime.datetime.now().isoformat(timespec='seconds')}   runtime {time.time() - t_start:.0f}s")
lines.append(f"Masks        : {'SYNTHETIC (self-test)' if args.self_test else 'GrabCut'}, {args.mask_passes} pass(es)")
lines.append("")
hdr = f"{'frame':>6} {'in':>3} " + "".join(f"{'IoU ' + n[:12]:>17}" for n, _ in calibs) + \
      f"{'LOO IoU':>9}"
lines.append(hdr)
lines.append("-" * len(hdr))
for row in per_frame:
    lines.append(f"{row['frame_id']:>6} {'y' if row['inlier'] else 'n':>3} " +
                 "".join(f"{row['iou'][n]:>17.4f}" for n, _ in calibs) +
                 (f"{row['loo_iou']:>9.4f}" if 'loo_iou' in row else f"{'-':>9}"))
lines.append("-" * len(hdr))
lines.append(f"{'mean':>6} {'':>3} " + "".join(f"{np.mean([iou_tab[n][i] for i in inliers]):>17.4f}" for n, _ in calibs) +
             (f"{np.mean([loo[j]['iou'] for j in loo]):>9.4f}" if loo else f"{'-':>9}"))
lines.append("  (IoU over the inlier frames, valid region only; LOO IoU = held-out frame scored with the "
             "calibration fit without it)")
lines.append("")
d, r = calib_change(X0, X_dr, ALL)
lines.append(f"Refined vs dataset calibration : mean clevis displacement {d:.2f} mm, rotation {r:.2f} deg, "
             f"translation of base origin {np.linalg.norm(X_dr[:3, 3] - X0[:3, 3]):.2f} mm")
if X_pre is not None:
    d2, r2 = calib_change(X_pre, X_dr, ALL)
    lines.append(f"Refined vs pre-alignment       : mean clevis displacement {d2:.2f} mm, rotation {r2:.2f} deg")
if loo:
    devs = np.array([loo[j]["dev"] for j in loo])
    lines.append(f"Leave-one-out calibration spread: {devs[:, 0].mean():.2f} mm (max {devs[:, 0].max():.2f}), "
                 f"{devs[:, 1].mean():.2f} deg (max {devs[:, 1].max():.2f})")
if X_true is not None:
    d0, r0 = calib_change(X0, X_true, ALL)
    d3, r3 = calib_change(X_dr, X_true, ALL)
    lines.append(f"SELF-TEST error: init {d0:.2f} mm / {r0:.2f} deg  ->  refined {d3:.2f} mm / {r3:.2f} deg")
lines.append("")
lines.append("cam_raw_T_base (m), refined:")
lines += ["  " + np.array2string(row, precision=6, suppress_small=True) for row in to_m(X_dr)]
lines.append("=" * W_)
print("\n" + "\n".join(lines))

# ── Save masks, overlays, PDF, JSON ───────────────────────────────────────────
colours = {"pre-alignment": (255, 0, 255), "dataset": (0, 200, 255), "diff-render": (0, 255, 0),
           "true (self-test)": (255, 128, 0)}
overlays = []
for i in ALL:
    fid = frames[i]["frame_id"]
    cv2.imwrite(os.path.join(OUT_DIR, "masks", f"mask-{fid:04d}.png"), masks[i] * 255)
    cv2.imwrite(os.path.join(OUT_DIR, "masks", f"valid-{fid:04d}.png"), (valid[i] * 255).astype(np.uint8))
    ov = draw_overlay(images[i], i, masks[i], valid[i],
                      [(f"{n}  IoU {iou_tab[n][i]:.3f}", Xc, colours[n]) for n, Xc in calibs])
    cv2.imwrite(os.path.join(OUT_DIR, "overlays", f"overlay-{fid:04d}.png"), ov)
    overlays.append((fid, ov))

pdf_path = os.path.join(OUT_DIR, "hand_eye_dr_report.pdf")
with PdfPages(pdf_path) as pdf:
    for k in range(0, len(lines), 70):
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.02, 0.98, "\n".join(lines[k:k + 70]), family="monospace", fontsize=6.5, va="top", ha="left")
        pdf.savefig(fig); plt.close(fig)
    for fid, ov in overlays:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {fid}: white = GrabCut mask, red tint = ignored (jaws / shaft / prongs), "
                     f"contours = clevis rendered at each calibration", fontsize=9)
        ax.axis("off")
        pdf.savefig(fig); plt.close(fig)

result = dict(
    generated=datetime.datetime.now().isoformat(timespec="seconds"), dataset=os.path.abspath(ROOT),
    hand_eye_json=HAND_EYE_JSON, self_test=args.self_test, mask_passes=args.mask_passes,
    frames=[f["frame_id"] for f in frames], inliers=[frames[i]["frame_id"] for i in inliers],
    cam_T_base_dataset=to_m(X0).tolist(), cam_T_base_refined=to_m(X_dr).tolist(),
    cam_T_base_pre_alignment=to_m(X_pre).tolist() if X_pre is not None else None,
    change_mm=d, change_deg=r, per_frame=per_frame,
)
if X_true is not None:
    result["cam_T_base_true"] = to_m(X_true).tolist()
with open(os.path.join(OUT_DIR, "hand_eye_dr_result.json"), "w") as fh:
    json.dump(result, fh, indent=2)
print(f"Report → {pdf_path}\nResult → {os.path.join(OUT_DIR, 'hand_eye_dr_result.json')}")

# ── Write back ────────────────────────────────────────────────────────────────
if args.self_test or args.dry_run:
    print("Not updating the hand-eye calibration (self-test / --dry-run).")
    sys.exit(0)
if not os.path.exists(HAND_EYE_JSON):
    print(f"{HAND_EYE_JSON} does not exist -- not updating it.")
    sys.exit(0)
current = to_mm(load_hand_eye_calibration(HAND_EYE_JSON))
dc, rc = np.linalg.norm(current[:3, 3] - X0[:3, 3]), rot_deg(current[:3, :3], X0[:3, :3])
# A hand-eye calibration only holds for the camera placement it was made for. If the file no longer holds the
# calibration this dataset was captured with, it was re-registered/re-aligned since -- possibly for another
# camera placement -- and writing this dataset's refinement over it could replace a newer, valid calibration.
mismatch = dc > 0.01 or rc > 0.01
if mismatch:
    print(f"WARNING: {HAND_EYE_JSON} no longer holds the calibration this dataset was captured with "
          f"({dc:.2f} mm / {rc:.2f} deg apart). It was re-registered or re-aligned after the capture; the refined "
          f"result is only valid for the camera placement of {ROOT}.")
# change_mm / change_deg as hand_eye_rodrigues_dataset.py computes them (base-origin translation, rotation)
cd, cr = float(np.linalg.norm(X_dr[:3, 3] - current[:3, 3])), rot_deg(current[:3, :3], X_dr[:3, :3])
cd_tool = calib_change(current, X_dr, ALL)[0]
print(f"Applying would change {HAND_EYE_JSON} by {cd:.2f} mm / {cr:.2f} deg "
      f"({cd_tool:.2f} mm mean displacement at the tool).")
if args.apply and mismatch and not args.allow_mismatch:
    answer = "n"
    print("--apply refused because of the mismatch above; add --allow-mismatch if the camera has not moved.")
elif args.apply:
    answer = "y"
elif sys.stdin.isatty():
    answer = input("Apply this refined calibration to the hand-eye file? [y/N]: ").strip().lower()
    if answer == "y" and mismatch:
        answer = input("The file holds a different, newer calibration (see WARNING). Replace it anyway? "
                       "[y/N]: ").strip().lower()
else:
    answer = "n"
    print("Non-interactive and no --apply -- not updating.")
if answer != "y":
    print("Not updating the hand-eye calibration file.")
    sys.exit(0)
# Same keys as hand_eye_rodrigues_dataset.py's entry (image_generation_gt_hand_eye.py copies these into
# calibration_variants.json), plus diff-render extras. 'spread' there is each frame's correction vs their
# mean; the closest equivalent here is each leave-one-out refit vs the all-frame result.
if loo:
    _t_dev = np.array([np.linalg.norm(loo[j]["X"][:3, 3] - X_dr[:3, 3]) for j in loo])
    _r_dev = np.array([rot_deg(loo[j]["X"][:3, :3], X_dr[:3, :3]) for j in loo])
    spread = {'trans_std_mm': float(_t_dev.std()), 'trans_max_mm': float(_t_dev.max()),
              'rot_std_deg': float(_r_dev.std()), 'rot_max_deg': float(_r_dev.max()),
              'source': 'leave-one-out refits vs the all-frame fit'}
else:
    spread = None
history_entry = {
    'timestamp': time.time(),
    'dataset': os.path.abspath(ROOT),
    'num_frames_saved': len(inliers),
    'num_frames': N,
    'cam_T_base_at_capture': to_m(X0).tolist(),
    'previous_transform': to_m(current).tolist(),
    'new_transform': to_m(X_dr).tolist(),
    'change_mm': cd,
    'change_deg': cr,
    'spread': spread,
    'method': 'diff_render (hand_eye_dr_refine.py, joint.stl clevis, GrabCut masks)',
    'frames_used': [frames[i]['frame_id'] for i in inliers],
    'change_mm_at_tool': cd_tool,
    'mean_iou_before': float(np.mean([iou_tab["dataset"][i] for i in inliers])),
    'mean_iou_after': float(np.mean([iou_tab["diff-render"][i] for i in inliers])),
    'loo_mean_iou': float(np.mean([loo[j]["iou"] for j in loo])) if loo else None,
}
backup = write_hand_eye_calibration(HAND_EYE_JSON, to_m(X_dr), history_entry)
print(f"Updated {HAND_EYE_JSON}: base-frame.transform is now the refined calibration (what every script loads); "
      f"the registration + GUI calibration it replaced is in 'diff-render'.previous_transform (and "
      f"alignment-history[-1].previous_transform), backed up to {backup}; "
      f"'pre-alignment' still holds the original camera_registration.py result.")
