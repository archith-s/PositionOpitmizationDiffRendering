"""
hand_eye_diff_render_test.py — differentiable-rendering pose optimizer run
against a dVRK hand-eye evaluation capture (default: LND006), using that
capture's gripper_ground_truth.json as the ground truth.

  * IMAGES        : <root>/right_rect-XXXX.png
                    (the RECTIFIED right frames — the frame gripper_ground_truth.json's
                    cam_T_gripper is expressed in, per image_generation_gt_hand_eye.py)
  * GROUND TRUTH  : <root>/gripper_ground_truth.json, fields
                    "cam_T_shaft_end" + "cam_T_gripper" (4x4, metres), combined
                    by cam_T_joint_model into the wrist-clevis pose the mesh is
                    defined in -> R + t(mm), model->camera, OpenCV/BOP
                    convention.
  * MODEL / CAMERA: the SurgRIPE LND joint.stl mesh + joint.npy model points
                    (default /home/aseshad4/TEST), LND instrument_type in
                    evaluate.py's Evaluator. Intrinsics are read PER RECORD from
                    the JSON's "camera_matrix_right_rect" (stereo-rectification
                    P, zero distortion); older datasets without it fall back to
                    the jhu_dVRK right camera_info projection matrix below.

The capture ships NO segmentation masks, so the observed silhouette is
stood in for by a crisp, full-resolution render of the FULL mesh at the JSON
GT pose — i.e. exactly what a perfect binary segmentation mask would be. The
optimizer only ever sees that binary mask (never the GT pose), so the
reported numbers are a pose-REFINEMENT result: how well the renderer +
optimizer recover the GT pose from a GT-perturbed init (+-10 deg / +-10 mm).
It does NOT fit the real tool pixels in the RGB frame, and it is not
comparable to the SurgRIPE paper's blind RGB pose-estimation numbers.

Pipeline (parameters tuned on the real LND006 poses + 30 synthetic poses
spanning tool-axis-vs-line-of-sight 20-90 deg, full 360 deg roll and 65-115 mm
depth, then validated on 40 unseen synthetic viewpoints; see CONFIG):

  1. Render a RENDER_SIZE^2 window around the tool (sized from the current
     pose estimate, so the clevis fills the window) instead of squashing the
     whole 1920x1080 frame, where it is only a few dozen pixels wide.
  2. Optimize against a ~1000-face decimated proxy, NOT face-doubled
     (PyTorch3D doesn't cull back faces, so doubling only wastes K slots).
     Soft-silhouette edge softness is set in render pixels (SIGMA_PX).
  3. Edge-profile calibration: the soft rasterizer's silhouette is not a
     0.5-at-the-edge step (several surface layers blend), which biased depth
     by ~+1 mm. The renderer's mean alpha vs signed distance to the true edge
     is measured once on synthetic calibration poses (cached to disk) and
     every binary target mask is mapped through that table, so render and
     target share one edge profile. Needs only the mask — works unchanged
     with real segmentation masks.
  4. Stages: centroid + area (translation incl. depth) -> silhouette MSE
     (translation) -> MSE+IoU (rotation+translation, coarse-to-fine image
     blur), re-cropping around the current estimate between stages.
  5. Frames are optimized in batches. Adam is elementwise, so a batch gives
     the same per-frame result as one-at-a-time, at a fraction of the cost.

Output mirrors EvalRealDiffRender.py: evaluate.py's Evaluator is fed every
(pose_pred, pose_gt) pair, the run saves the Evaluator.summarize() .npy, and a
single self-contained PDF report (labeled metrics, ADD accuracy-vs-threshold
curve, one GT-vs-estimate silhouette+axes overlay page per frame). Default is
the first 5 frames.
"""

import os, re, glob, struct, tempfile, textwrap, datetime, json, math, hashlib
import argparse
import time
import numpy as np
import cv2
import trimesh
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
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


# ─────────────────────────── STL → OBJ ────────────────────────────────────────
def stl_to_obj(stl_path: str, obj_path: str):
    """Convert binary STL to OBJ with deduplicated vertices (verbatim from
    EvalRealDiffRender.py — same mesh, same conversion)."""
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
HAND_EYE_ROOT = "/home/aseshad4/hand_eye_collection_scripts/LND006"   # images + gripper_ground_truth.json
MODEL_ROOT    = "/home/aseshad4/TEST"                  # joint.stl + joint.npy (same model as EvalRealDiffRender.py)

GT_JSON       = "gripper_ground_truth.json"
IMAGE_FIELD   = "right_rect_image"   # rectified right frame — cam_T_gripper's frame
POSE_FIELD    = "cam_T_gripper"      # 4x4 homogeneous, metres -- tool tip, on the yaw pin
SHAFT_FIELD   = "cam_T_shaft_end"    # 4x4 homogeneous, metres -- shaft end, on the pitch pin
K_FIELD       = "camera_matrix_right_rect"

# Fallback rectified-right intrinsics for datasets whose JSON has no K_FIELD
# (jhu_dVRK right camera_info projection matrix, 1920x1080, zero distortion).
CAM_FX, CAM_FY = 1705.507261, 1708.226705
CAM_CX, CAM_CY = 927.378712, 623.082993

OUTPUT_SUMMARY  = "../Eval_Results_DiffRender_HandEye/eval_summary_hand_eye_dr.npy"
INSTRUMENT_TYPE = 'LND'

USE_GT_INITIALIZATION = True
INIT_ROT_NOISE_DEG    = 10.0
INIT_TRANS_NOISE_MM   = 10.0
USE_GT_INIT_ONLY      = False

IOU_THRESHOLD = 0.5

# ── Renderer (each choice measured against the binary target, see docstring) ──
RENDER_SIZE     = 192     # px; 128 cost ~2x the error, 256 bought nothing over 192
OPT_FACES       = 1000    # decimated proxy; finer meshes need far more faces_per_pixel to fill in
FACES_PER_PIXEL = 8       # 4 leaves interior alpha ~0.9 and roughly doubled the error
SIGMA_PX        = 0.5     # soft-silhouette edge width, render px
BLUR_P_CUT      = 1e-2    # faces whose contribution is below this are culled (sets blur_radius)
CALIB_POSES     = 30      # synthetic poses for the edge-profile table (cached on disk)

# ── Crop window ────────────────────────────────────────────────────────────────
CROP_MARGIN_INIT_MM = 12.0   # around the INIT pose: must absorb the +-10mm init error
CROP_MARGIN_MM      = 2.0    # around the current estimate once translation has converged

# ── Optimizer ─────────────────────────────────────────────────────────────────
# Adam on (translation mm, 6-D rotation). LR_TRANS is large because the init is
# up to ~17mm off and Adam moves ~lr per step: the previous 0.05 capped
# translation travel at a few mm, which was most of the old error.
LR_TRANS   = 3.0
LR_ROT     = 0.01
ITERS_T    = 30       # per translation stage (x2)
ITERS_R    = 180      # rotation stages total; 120 left rotation ~0.4 deg worse, 240 bought nothing
AREA_W     = 5e3      # centroid stage depth term (sqrt-ratio form: bounded when the render leaves the window)
IOU_W      = 1.0      # on top of area-normalized MSE
IMG_BLUR   = (2, 1, 0)  # px Gaussian on render AND target for stages T2/R1/R2 (coarse-to-fine)
BATCH_SIZE = 16

STAGES = [
    # kind      iters          lr_t              lr_r          img_blur      cosine  recrop
    ("centroid", ITERS_T,      LR_TRANS,         0.0,          0,            True,   False),
    ("mse",      ITERS_T,      LR_TRANS * 0.3,   0.0,          IMG_BLUR[0],  True,   True),
    ("mse_iou",  ITERS_R // 2, LR_TRANS * 0.1,   LR_ROT,       IMG_BLUR[1],  False,  True),
    ("mse_iou",  ITERS_R - ITERS_R // 2, LR_TRANS * 0.03, LR_ROT * 0.3, IMG_BLUR[2], False, True),
]

# ──────────────────────────────────────────────────────────────────────────────


parser = argparse.ArgumentParser()
parser.add_argument("--all",    action="store_true", help="Run all frames in the JSON")
parser.add_argument("--frames", type=int, default=None,
                    help="Run only the first N frames (default: 5)")
parser.add_argument("--start-frame", type=int, default=None,
                    help="Skip the first N JSON records before applying --frames/--all")
parser.add_argument("--hand-eye-root", type=str, default=HAND_EYE_ROOT,
                    help="Directory holding right_rect-*.png + gripper_ground_truth.json")
parser.add_argument("--model-root", type=str, default=MODEL_ROOT,
                    help="Directory holding joint.stl + joint.npy (the LND model)")
parser.add_argument("--output_summary", type=str, default=OUTPUT_SUMMARY,
                    help="Where to save evaluate.py Evaluator.summarize() results.")
parser.add_argument("--batch", type=int, default=BATCH_SIZE,
                    help="Frames optimized together (same per-frame result; lower it if GPU memory is short)")
parser.add_argument("--seed", type=int, default=0, help="Seed for the GT-perturbed inits")
parser.add_argument("--obj", nargs="+", default=None,
                    help="Use jaw OBJ mesh(es) from ../meshes instead of joint.stl: tool id(s) "
                         "(e.g. 006 -> every tip_006_*.obj) and/or file names. Posed at "
                         "cam_T_gripper @ MESH_TO_GRIPPER (pin axis = mesh Z, see below).")
parser.add_argument("--scale", type=str, default="auto",
                    help="--obj only: multiplier taking OBJ units to mm, or 'auto' (power of ten "
                         "giving a 5-50 mm longest extent)")
args = parser.parse_args()

HAND_EYE_ROOT = args.hand_eye_root
MODEL_ROOT    = args.model_root
MESH_STL      = os.path.join(MODEL_ROOT, "joint.stl")
MODEL_PTS     = os.path.join(MODEL_ROOT, "joint.npy")

# --obj: jaw meshes, in a canonical frame with the jaw pivot pin at the
# origin, pin axis along Z and the jaws along +Y (tip_006 as shipped: its
# pulley disc is centred on the origin, normal to Z). cam_T_gripper's origin
# is on the same (yaw) pin with X = pin axis and Z along the jaws, so the
# mesh pose is cam_T_gripper @ MESH_TO_GRIPPER (mesh Z -> gripper X, mesh Y
# -> gripper Z, mesh X -> gripper Y). The JSON carries no jaw angle, so the
# jaws are rendered as modelled (closed). Checked against the real
# right_rect frames for tip_006 (LND006) and tip_093 (PGForceps093).
MESHES_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "meshes")
MESH_TO_GRIPPER = np.array([[0.0, 0.0, 1.0],
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0]])

# Per-file corrections (mm, after unit scaling) bringing an OBJ into the
# canonical frame above. tip_093_*: the shared pin hole (r~0.84mm) is at
# (0, 20, 0) and the jaws point -Y, so shift the pin to the origin and turn
# 180 deg about the pin; the hooked cable tangs more than 3mm behind the pin
# sit inside the clevis on the real tool (they render out past it into empty
# background in every frame), so they are dropped.
OBJ_FIXES = {
    "tip_093": dict(pin_mm=(0.0, 20.0, 0.0), flip_about_pin=True, trim_behind_pin_mm=3.0),
}


def resolve_obj_args(names):
    """Tool ids ("006" -> tip_006_*.obj) and/or file names (with or without
    .obj) in MESHES_DIR -> list of absolute paths."""
    paths = []
    for name in names:
        cand = os.path.join(MESHES_DIR, name if name.endswith(".obj") else name + ".obj")
        matches = [cand] if os.path.isfile(cand) else \
            sorted(glob.glob(os.path.join(MESHES_DIR, f"tip_{name}_*.obj")))
        if not matches:
            avail = ", ".join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(MESHES_DIR, "*.obj"))))
            raise SystemExit(f"--obj {name!r}: no match in {MESHES_DIR}. Available: {avail}")
        paths += [m for m in matches if m not in paths]
    return paths


OBJ_PATHS = resolve_obj_args(args.obj) if args.obj else []
MESH_TAG  = "+".join(os.path.splitext(os.path.basename(p))[0] for p in OBJ_PATHS) if OBJ_PATHS else "joint.stl"

os.makedirs(os.path.dirname(os.path.abspath(args.output_summary)), exist_ok=True)
print(f"Hand-eye data → {HAND_EYE_ROOT}")
print(f"Mesh          → {', '.join(os.path.basename(p) for p in OBJ_PATHS) if OBJ_PATHS else MESH_STL}")
print(f"Eval summary  → {args.output_summary}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
np.random.seed(args.seed)

_arange = torch.arange(RENDER_SIZE, dtype=torch.float32, device=device)
GRID_Y, GRID_X = torch.meshgrid(_arange, _arange, indexing="ij")


# ── Load JSON + build frame list ──────────────────────────────────────────────
with open(os.path.join(HAND_EYE_ROOT, GT_JSON)) as f:
    _records = json.load(f)
_records = sorted(_records, key=lambda r: int(r["frame_index"]))
print(f"JSON records: {len(_records)}")

if args.start_frame is not None:
    _records = _records[args.start_frame:]
    print(f"Skipping first {args.start_frame} records (--start-frame)")

if args.frames is not None:
    _records = _records[:args.frames]
    print(f"Running {len(_records)} frame(s) (--frames {args.frames})")
elif not args.all:
    _records = _records[:5]
    print(f"Running first {len(_records)} frames (--all for all)")
else:
    print(f"Running all {len(_records)} frames (--all)")


# ── Camera intrinsics (per record) ────────────────────────────────────────────
def record_K(rec) -> np.ndarray:
    """3x3 K the record's cam_T_* projects with. Datasets from the current
    image_generation_gt_hand_eye.py save it per record (stereo P's 3x3, zero
    distortion); older ones don't, so fall back to the constants above."""
    if K_FIELD in rec:
        return np.array(rec[K_FIELD], dtype=np.float64)
    return np.array([[CAM_FX, 0, CAM_CX], [0, CAM_FY, CAM_CY], [0, 0, 1]], dtype=np.float64)

_K0 = record_K(_records[0])
print(f"Camera K ({'from JSON' if K_FIELD in _records[0] else 'hardcoded -- JSON has none'}): "
      f"fx={_K0[0,0]:.2f}  fy={_K0[1,1]:.2f}  cx={_K0[0,2]:.2f}  cy={_K0[1,2]:.2f}")


def cam_T_joint_model(cam_T_shaft, cam_T_gripper) -> np.ndarray:
    """Wrist-clevis pose of the SurgRIPE joint mesh, from both dVRK axes.

    joint.stl/joint.npy is the clevis -- the link between the pitch and yaw
    pins, rigid with neither the shaft nor the jaws. In its own frame (mm)
    the pitch pin hole is at the origin along X, the yaw pin hole is 9.1mm
    forward at (0, -9.1, 0) along Z, and the body runs forward along -Y.
    On the dVRK side cam_T_shaft_end's origin sits on the pitch pin,
    cam_T_gripper's origin sits on the yaw pin (also 9.1mm apart), and
    gripper X is the yaw pin's own axis -- so the two axes together pin
    down the clevis pose completely. cam_T_gripper alone can't: its Z
    follows the jaws, which swing about the yaw pin relative to the clevis.
    The mesh is symmetric to ~0.05mm under a 180deg turn about its own Y, so
    mapping its Z to +gripper X rather than -gripper X is immaterial.
    """
    shaft_origin = cam_T_shaft[:3, 3]
    forward = cam_T_gripper[:3, 3] - shaft_origin
    forward = forward / np.linalg.norm(forward)
    z_axis = cam_T_gripper[:3, 0] - forward * np.dot(cam_T_gripper[:3, 0], forward)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = -forward
    x_axis = np.cross(y_axis, z_axis)
    M = np.eye(4)
    M[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    M[:3, 3] = shaft_origin
    return M


def _load_gt(rec):
    """Mesh pose (R 3x3, t 3 in mm), model->camera: the clevis from
    cam_T_shaft_end + cam_T_gripper for joint.stl, or cam_T_gripper @
    MESH_TO_GRIPPER for --obj jaw meshes."""
    if OBJ_PATHS:
        M = np.array(rec[POSE_FIELD], dtype=np.float64)
        return M[:3, :3] @ MESH_TO_GRIPPER, M[:3, 3] * 1000.0
    if SHAFT_FIELD not in rec:
        raise KeyError(
            f"frame {rec.get('frame_index')} has no '{SHAFT_FIELD}' -- the joint mesh's pose needs both "
            f"the shaft-end and gripper axes. Re-record the dataset with the current "
            f"image_generation_gt_hand_eye.py, which saves both.")
    M = cam_T_joint_model(np.array(rec[SHAFT_FIELD], dtype=np.float64),
                          np.array(rec[POSE_FIELD], dtype=np.float64))
    R = M[:3, :3].copy()
    t = (M[:3, 3].copy()) * 1000.0
    return R, t


# ── Image dimensions ──────────────────────────────────────────────────────────
_image_dir = HAND_EYE_ROOT
_sample_img = Image.open(os.path.join(_image_dir, _records[0][IMAGE_FIELD]))
img_W, img_H = _sample_img.size
print(f"Image size: {img_W}×{img_H}")


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


if OBJ_PATHS:
    # Load + concatenate the jaw OBJ part(s), scaled to mm.
    _raw = [load_obj_no_mtl(p) for p in OBJ_PATHS]
    _extent = float((torch.cat([v for v, _ in _raw]).max(0).values - torch.cat([v for v, _ in _raw]).min(0).values).max())
    if args.scale == "auto":
        MESH_SCALE = 10.0 ** int(np.ceil(np.log10(5.0 / _extent))) if _extent > 0 else 1.0
    else:
        MESH_SCALE = float(args.scale)
    _v_parts, _f_parts, _n = [], [], 0
    for p, (v, f) in zip(OBJ_PATHS, _raw):
        v = v.double().numpy() * MESH_SCALE; f = f.numpy()
        # Weld duplicate vertices: tip_093_* are stored as a triangle soup (3
        # unshared verts per face), which quadric decimation can't simplify --
        # the proxy came out full of holes and the optimizer drifted 10-18 deg.
        _m = trimesh.Trimesh(v, f, process=False); _m.merge_vertices()
        v, f = np.asarray(_m.vertices), np.asarray(_m.faces)
        fix = next((fx for pre, fx in OBJ_FIXES.items() if os.path.basename(p).startswith(pre)), None)
        if fix:
            v = v - np.array(fix["pin_mm"])
            if fix.get("flip_about_pin"):
                v = v * np.array([-1.0, -1.0, 1.0])
            if fix.get("trim_behind_pin_mm") is not None:
                f = f[(v[f][:, :, 1] >= -fix["trim_behind_pin_mm"]).all(1)]
            _m = trimesh.Trimesh(v, f, process=False); _m.remove_unreferenced_vertices()
            v, f = np.asarray(_m.vertices), np.asarray(_m.faces)
            print(f"  {os.path.basename(p)}: {len(v)} verts, {len(f)} faces (fixed: {fix})")
        else:
            print(f"  {os.path.basename(p)}: {len(v)} verts, {len(f)} faces")
        _v_parts.append(torch.tensor(v, dtype=torch.float32)); _f_parts.append(torch.tensor(f, dtype=torch.int64) + _n); _n += len(v)
    verts, faces = torch.cat(_v_parts, 0), torch.cat(_f_parts, 0)
    verts_np = np.from_dlpack(verts.detach().cpu())
    _vmax = float(np.abs(verts_np).max())
    print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces | scale ×{MESH_SCALE:g} → "
          f"extents {np.ptp(verts_np, axis=0).round(2)} mm")
    if not all(os.path.basename(p).startswith(("tip_006_", "tip_093_")) for p in OBJ_PATHS):
        print("  NOTE: only tip_006 and tip_093 are verified against cam_T_gripper — "
              "overlays for other meshes may be offset/rotated/mis-scaled.")
else:
    # Convert joint.stl → OBJ and load mesh (same as EvalRealDiffRender.py).
    _obj_cache = os.path.join(tempfile.gettempdir(), "joint_lnd_handeye_converted.obj")
    if not os.path.exists(_obj_cache):
        print("Converting joint.stl → OBJ (first run only) ...")
        stl_to_obj(MESH_STL, _obj_cache)
    else:
        print(f"Reusing cached OBJ: {_obj_cache}")
    verts, faces = load_obj_no_mtl(_obj_cache)
    verts_np = np.from_dlpack(verts.detach().cpu())
    _vmax = float(verts_np.__abs__().max())
    print(f"Mesh: {verts.shape[0]} verts, {faces.shape[0]} faces | max|v|={_vmax:.4f}")
    if _vmax < 1.0:
        print("  Auto-scaling ×1000 (m→mm)")
        verts = verts * 1000.0
        verts_np = verts_np * 1000.0
    else:
        print("  Vertices in mm.")

# Bounding sphere — sizes the render window around the tool.
_bb_lo, _bb_hi = verts_np.min(0), verts_np.max(0)
SPHERE_C = torch.tensor(0.5 * (_bb_lo + _bb_hi), dtype=torch.float32, device=device)
SPHERE_R = float(np.linalg.norm(verts_np - 0.5 * (_bb_lo + _bb_hi), axis=1).max())
_BB_CORNERS = np.array([[x, y, z] for x in (_bb_lo[0], _bb_hi[0]) for y in (_bb_lo[1], _bb_hi[1])
                        for z in (_bb_lo[2], _bb_hi[2])], dtype=np.float64)


# ── Decimated proxy mesh — used ONLY for the optimization loop ────────────────
_verts_f64 = np.array(verts_np, dtype=np.float64, copy=True)
_faces_i64 = np.array(np.from_dlpack(faces.detach().cpu()), dtype=np.int64, copy=True)
_trimesh_full = trimesh.Trimesh(vertices=_verts_f64, faces=_faces_i64, process=False)
try:
    _trimesh_opt = _trimesh_full.simplify_quadric_decimation(face_count=OPT_FACES)
    verts_opt = torch.tensor(np.array(_trimesh_opt.vertices), dtype=torch.float32)
    faces_opt = torch.tensor(np.array(_trimesh_opt.faces),    dtype=torch.int64)
except Exception as e:
    raise RuntimeError(
        f"Quadric decimation failed ({e!r}). The optimizer is tuned for a ~{OPT_FACES}-face proxy -- "
        f"install the fast_simplification package (pip install fast_simplification).")
print(f"Optimization mesh: {faces_opt.shape[0]} faces (decimated from {faces.shape[0]}, not face-doubled)")


# ── Load model point cloud for ADD metric + evaluate.py Evaluator ────────────
if OBJ_PATHS:
    # Sampled from the OBJ (same count as joint.npy); ADD thresholds use this
    # mesh's bounding-box diameter rather than the clevis's hardcoded one.
    model_pts_np, _ = trimesh.sample.sample_surface(_trimesh_full, 30000, seed=0)
    model_pts_np = np.asarray(model_pts_np, dtype=np.float64)
    MODEL_DIAM = float(np.linalg.norm(np.ptp(model_pts_np, axis=0)))
    EVAL_ROOT = tempfile.mkdtemp(prefix="hand_eye_dr_obj_")
    np.save(os.path.join(EVAL_ROOT, "joint.npy"), model_pts_np)   # Evaluator reads root_path/joint.npy
    print(f"Model pts: {model_pts_np.shape[0]} points sampled from the mesh | diameter={MODEL_DIAM:.2f} mm")
else:
    model_pts_np = np.load(MODEL_PTS).astype(np.float64)
    _pts_vmax = float(np.abs(model_pts_np).max())
    if _pts_vmax < 1.0:
        model_pts_np = model_pts_np * 1000.0
    MODEL_DIAM = model_diameter(INSTRUMENT_TYPE)
    EVAL_ROOT = MODEL_ROOT
    print(f"Model pts: {model_pts_np.shape[0]} points | max|v|={_pts_vmax:.4f}")


# ── Fallback translation (mean of all GT in the run) ─────────────────────────
_all_gt_t = [_load_gt(r)[1] for r in _records]
mean_t_fallback = np.mean(_all_gt_t, axis=0)
print(f"Mean GT translation: {mean_t_fallback.round(2)}")


# ── Meshes / cameras / renderers ──────────────────────────────────────────────
def batch_meshes(v, f, B) -> Meshes:
    v, f = v.to(device), f.to(device)
    return Meshes(verts=[v] * B, faces=[f] * B,
                  textures=TexturesVertex(verts_features=[torch.ones_like(v)] * B))


_FLIP_T = torch.tensor(np.diag([-1., -1., 1.]), dtype=torch.float32, device=device)


def bop_to_pt3d_batch(R, t):
    """BOP/OpenCV (B,3,3),(B,3) -> PyTorch3D row-vector R, T."""
    return R.transpose(1, 2) @ _FLIP_T, t @ _FLIP_T


def window_cameras(R, t, win):
    """win: (B,4) = fx, fy, cx, cy of the RENDER_SIZE^2 window, in window px."""
    Rp, Tp = bop_to_pt3d_batch(R, t)
    return PerspectiveCameras(focal_length=win[:, :2], principal_point=win[:, 2:], R=Rp, T=Tp, in_ndc=False,
                              image_size=torch.tensor([[RENDER_SIZE, RENDER_SIZE]] * R.shape[0], device=device),
                              device=device)


def crop_window(R, t, K, margin_mm):
    """Square window around the projected bounding sphere of the mesh at pose
    (R, t), padded by margin_mm, as (B,4) window intrinsics."""
    c = R @ SPHERE_C + t
    z = c[:, 2].clamp(min=1.0)
    u = K[:, 0] * c[:, 0] / z + K[:, 2]
    v = K[:, 1] * c[:, 1] / z + K[:, 3]
    half = K[:, 0] * (SPHERE_R + margin_mm) / z
    s = RENDER_SIZE / (2 * half)
    return torch.stack([K[:, 0] * s, K[:, 1] * s, (K[:, 2] - (u - half)) * s, (K[:, 3] - (v - half)) * s], 1)


def make_soft_renderer() -> MeshRenderer:
    sigma = (SIGMA_PX * 2.0 / RENDER_SIZE) ** 2               # PyTorch3D dists are squared NDC
    raster = RasterizationSettings(
        image_size=RENDER_SIZE, blur_radius=float(np.log(1.0 / BLUR_P_CUT - 1.0)) * sigma,
        faces_per_pixel=FACES_PER_PIXEL,
        # Binned raster; a per-bin cap of the whole mesh can never overflow
        # (overflow silently drops faces).
        bin_size=16, max_faces_per_bin=faces_opt.shape[0],
    )
    return MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster),
                        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=sigma)))


soft_renderer = make_soft_renderer()

# Crisp full-resolution silhouettes of the FULL mesh: the stand-in for the
# observed mask (at GT) and the IoU metric (at the estimate). Only the
# integer-aligned native-resolution window around the tool's bounding sphere
# is rasterized (identical pixels, ~7x fewer of them than the whole frame),
# then pasted into a full-frame mask. Binned with a cap of the whole mesh, so
# no overflow.
_mesh_full = batch_meshes(verts.float(), faces, 1)


def render_crisp_alpha_t(R_bop, t_bop, K3) -> torch.Tensor:
    R = torch.tensor(np.asarray(R_bop), dtype=torch.float32, device=device)[None]
    t = torch.tensor(np.asarray(t_bop), dtype=torch.float32, device=device)[None]
    out = torch.zeros(img_H, img_W, device=device)
    # Window = projected bounding-box corners (+2px): an exact bound under
    # perspective, unlike fx*R/z, which clipped long meshes pointing at the camera.
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


def render_crisp_alpha(R_bop, t_bop, K3) -> np.ndarray:
    return np.from_dlpack(render_crisp_alpha_t(R_bop, t_bop, K3).cpu())


# ── Binary mask -> window target, via the edge-profile table ──────────────────
_SS = 4   # supersampling of the mask resample / signed-distance computation


def mask_to_window(full_mask, K, win, ss=_SS):
    """Resample native-resolution masks (B,H,W) into each frame's window at
    ss x RENDER_SIZE resolution (B, ss*RS, ss*RS)."""
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
    """Signed distance (window px, + inside) to the mask edge at each window pixel centre."""
    m = (mask_to_window(full_mask, K, win) > 0.5).cpu().numpy().astype(np.uint8)
    out = []
    for b in range(m.shape[0]):
        d_in  = cv2.distanceTransform(m[b],     cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_out = cv2.distanceTransform(1 - m[b], cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out.append(np.where(m[b] > 0, d_in - 0.5, -(d_out - 0.5)) / _SS)
    sdf = torch.tensor(np.stack(out), device=device)
    return F.avg_pool2d(sdf[:, None], _SS)[:, 0]


LUT_EDGES = torch.linspace(-6, 6, 241, device=device)


def synthetic_calibration_poses(n, K3, seed=99):
    """Tool axis 20-90 deg from the line of sight, any roll, 65-115 mm deep,
    anywhere in the central 70% of the image — disjoint from the evaluated frames."""
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
    """Mean soft-rendered alpha vs signed distance to the TRUE silhouette edge.
    Depends only on the renderer + mesh (+ K for scale), so it is computed once
    on synthetic poses and cached on disk."""
    key = json.dumps([MESH_TAG, RENDER_SIZE, OPT_FACES, FACES_PER_PIXEL, SIGMA_PX, BLUR_P_CUT, CALIB_POSES,
                      K3.round(3).tolist(), img_W, img_H, faces.shape[0], verts.shape[0], round(_vmax, 4)])
    path = os.path.join(tempfile.gettempdir(), f"hand_eye_dr_edge_lut_{hashlib.md5(key.encode()).hexdigest()[:12]}.npy")
    if os.path.exists(path):
        print(f"Edge-profile table: cached ({path})")
        return torch.tensor(np.load(path), dtype=torch.float32, device=device)
    print(f"Edge-profile table: calibrating on {CALIB_POSES} synthetic poses (once) ...")
    Rs, ts = synthetic_calibration_poses(CALIB_POSES, K3)
    K = torch.tensor([[K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]]] * CALIB_POSES, dtype=torch.float32, device=device)
    R = torch.tensor(Rs, dtype=torch.float32, device=device); t = torch.tensor(ts, dtype=torch.float32, device=device)
    full = torch.stack([render_crisp_alpha_t(Rs[i], ts[i], K3) for i in range(CALIB_POSES)])
    mesh = batch_meshes(verts_opt, faces_opt, CALIB_POSES)
    sums = torch.zeros(len(LUT_EDGES) + 1, device=device); cnts = torch.zeros_like(sums)
    for margin in (CROP_MARGIN_MM + 1.0, 6.0, CROP_MARGIN_INIT_MM):     # cover the window scales used
        win = crop_window(R, t, K, margin)
        sdf = mask_signed_distance(full, K, win)
        with torch.no_grad():
            a = soft_renderer(meshes_world=mesh, cameras=window_cameras(R, t, win))[..., 3]
        idx = torch.bucketize(sdf.flatten(), LUT_EDGES)
        sums.index_add_(0, idx, a.flatten()); cnts.index_add_(0, idx, torch.ones_like(a.flatten()))
    lut = sums / cnts.clamp(min=1)
    np.save(path, lut.cpu().numpy())
    print(f"  alpha at the true edge = {lut[torch.bucketize(torch.zeros(1, device=device), LUT_EDGES)].item():.2f} "
          f"(0.5 would be unbiased) -> saved {path}")
    return lut


def gblur(x, px):
    """Separable Gaussian blur of (B,H,W) by px pixels (0 = identity)."""
    if not px:
        return x
    r = int(math.ceil(3 * px))
    g = torch.arange(-r, r + 1, device=x.device, dtype=torch.float32)
    k = torch.exp(-0.5 * (g / px) ** 2); k = k / k.sum()
    y = F.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, r))
    return F.conv2d(y, k.view(1, 1, -1, 1), padding=(r, 0))[:, 0]


# ── 6-D ↔ SO(3) ───────────────────────────────────────────────────────────────
def rot6d_to_matrix(r6d: torch.Tensor) -> torch.Tensor:
    """(B,6) -> (B,3,3); columns b1, b2, b1 x b2."""
    a1, a2 = r6d[:, :3], r6d[:, 3:]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    return torch.stack([b1, b2, torch.linalg.cross(b1, b2, dim=1)], dim=2)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    return R[:, :, :2].transpose(1, 2).reshape(R.shape[0], 6)


def perturb_pose(R_gt, t_gt, rot_noise_deg=10.0, trans_noise_mm=10.0):
    ax = np.random.randn(3)
    ax /= np.linalg.norm(ax)
    R_init = Rotation.from_rotvec(
        ax * np.deg2rad(np.random.uniform(-rot_noise_deg, rot_noise_deg))
    ).as_matrix() @ R_gt
    return R_init, t_gt + np.random.uniform(-trans_noise_mm, trans_noise_mm, 3)


# ── Batched optimizer ─────────────────────────────────────────────────────────
def optimize_batch(full_masks, K, init_R, init_t, lut, desc=""):
    """Recover poses whose crisp silhouettes are full_masks (B,H,W).
    K (B,4) = fx, fy, cx, cy; init_R (B,3,3), init_t (B,3) mm, BOP convention.
    Loss is a sum over frames, and Adam is elementwise, so each frame's
    trajectory is independent of the rest of the batch."""
    B = full_masks.shape[0]
    mesh = batch_meshes(verts_opt, faces_opt, B)
    rot6d = matrix_to_rot6d(init_R).clone().requires_grad_(True)
    transl = init_t.clone().requires_grad_(True)
    win = None
    for si, (kind, iters, lr_t, lr_r, blur, cosine, recrop) in enumerate(STAGES):
        if win is None or recrop:
            with torch.no_grad():
                win = crop_window(rot6d_to_matrix(rot6d), transl, K,
                                  CROP_MARGIN_INIT_MM if win is None else CROP_MARGIN_MM)
                target = lut[torch.bucketize(mask_signed_distance(full_masks, K, win), LUT_EDGES)]
                target_b = gblur(target, blur)
                area_t = target.sum((1, 2)).clamp(min=1.0)
                cx_t = (GRID_X * target).sum((1, 2)) / area_t
                cy_t = (GRID_Y * target).sum((1, 2)) / area_t
        else:
            target_b = gblur(target, blur)
        groups = [{"params": [transl], "lr": lr_t}]
        if lr_r > 0:
            groups.append({"params": [rot6d], "lr": lr_r})
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters) if cosine else None
        loop = tqdm(range(iters), desc=f"  {desc} stage {si + 1}/{len(STAGES)} ({kind})", leave=False)
        for _ in loop:
            opt.zero_grad()
            a = soft_renderer(meshes_world=mesh, cameras=window_cameras(rot6d_to_matrix(rot6d), transl, win))[..., 3]
            if kind == "centroid":
                area = a.sum((1, 2)).clamp(min=1e-6)
                cx = (GRID_X * a).sum((1, 2)) / area
                cy = (GRID_Y * a).sum((1, 2)) / area
                loss = (cx - cx_t) ** 2 + (cy - cy_t) ** 2 + AREA_W * (torch.sqrt(area / area_t) - 1) ** 2
            else:
                loss = (gblur(a, blur) - target_b).pow(2).sum((1, 2)) / area_t
                if kind == "mse_iou":
                    inter = (a * target).sum((1, 2))
                    union = (a + target - a * target).sum((1, 2)).clamp(min=1e-6)
                    loss = loss + IOU_W * (1 - inter / union)
            loss.sum().backward()
            opt.step()
            if sched:
                sched.step()
        loop.close()
    with torch.no_grad():
        return rot6d_to_matrix(rot6d).double().cpu().numpy(), transl.detach().double().cpu().numpy()


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
    mask_a = alpha_a > threshold
    mask_b = alpha_b > threshold
    union  = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter) / float(union)


def build_metric_descriptions(instrument_type: str) -> dict:
    diam = MODEL_DIAM
    thresh_mm = diam * 0.1
    return {
        'proj2d': (
            "Fraction of items where the mean 2D reprojection error (GT vs. "
            "predicted model points, projected using evaluate.py's fixed "
            f"'{instrument_type}' camera intrinsics — not this run's rectified-right "
            "camera K) is below 5 pixels."
        ),
        'add': (
            "Fraction of items where ADD (mean point-to-corresponding-point "
            f"distance between GT and predicted model points) is below 10% of "
            f"the {MESH_TAG} diameter ({thresh_mm:.2f} mm, "
            f"diameter={diam:.2f} mm)."
        ),
        'add-s': (
            "Fraction of items where ADD-S (nearest-neighbour matched ADD) is "
            f"below 10% of the {MESH_TAG} diameter ({thresh_mm:.2f} mm)."
        ),
        'ADD-distance': "Mean ADD distance across all evaluated items, in mm.",
        'ADD-distance list': (
            "Per-item ADD distance in mm, one entry per evaluated item, in "
            "the same order as pred_list/gt_list."
        ),
        'ADDS-distance': "Mean ADD-S distance across all evaluated items, in mm.",
        'avg_acc_0_5mm': (
            "SurgRIPE paper (arxiv 2501.02990) Table 4's 'Avg Acc (0-5mm)' "
            "column: mean of the ADD pass-rate at each integer threshold "
            "0,1,2,3,4,5mm."
        ),
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
            "Ground-truth pose (cam_T_gripper) as a 3x4 [R|t] matrix (t in mm) "
            "for each evaluated item — parallel to pred_list."
        ),
    }


def compute_threshold_curve(values: list, max_threshold: float, n_points: int = 21):
    vals = np.array(values, dtype=float)
    thresholds = np.linspace(0.0, max_threshold, n_points)
    accuracy = np.array([(vals <= t).mean() for t in thresholds])
    _trapz = getattr(np, "trapezoid", np.trapz)   # np.trapezoid is numpy>=2.0
    auc = float(_trapz(accuracy, thresholds) / max_threshold) if max_threshold > 0 else 0.0
    return thresholds, accuracy, auc


def build_report_lines(npy_path: str, descriptions: dict,
                       item_records: list, dataset_dir: str,
                       instrument_type: str) -> list:
    data = np.load(npy_path, allow_pickle=True).item()
    W = 90
    L = []
    L.append("=" * W)
    L.append("MASTER EVALUATION REPORT — hand_eye_diff_render_test.py".center(W))
    L.append("(diff-rendering pose optimizer, dVRK hand-eye capture)".center(W))
    L.append("(target = binary silhouette RENDERED at JSON GT — pose refinement)".center(W))
    L.append("=" * W)
    L.append(f"Hand-eye data    : {dataset_dir}")
    L.append(f"Mesh             : {MESH_TAG}  (diameter {MODEL_DIAM:.2f} mm)")
    L.append(f"Instrument type  : {instrument_type}")
    L.append(f"Items evaluated  : {len(item_records)}  (frames)")
    L.append(f"Generated        : {datetime.datetime.now().isoformat(timespec='seconds')}")
    L.append(f"Renderer         : {RENDER_SIZE}px window, {faces_opt.shape[0]}-face proxy, fpp={FACES_PER_PIXEL}, "
             f"sigma={SIGMA_PX}px, edge-profile-calibrated targets")
    L.append(f"Optimizer        : {sum(s[1] for s in STAGES)} iters, lr_t={LR_TRANS} lr_r={LR_ROT}, "
             f"batch={args.batch}, init noise +-{INIT_ROT_NOISE_DEG:g}deg/+-{INIT_TRANS_NOISE_MM:g}mm (seed {args.seed})")
    L.append("")

    L.append("-" * W)
    L.append("SECTION 1 — evaluate.py Evaluator metrics")
    L.append("(same metric set / same evaluate.py code SurgRIPE's run.py + estimator.py produce)")
    L.append("-" * W)
    for key in ['proj2d', 'add', 'add-s', 'ADD-distance', 'ADDS-distance',
                'avg_acc_0_5mm', 'cmd5', 'trans_error', 'rot_error']:
        if key not in data:
            continue
        L.append(f"{key:<16} = {float(data[key]):.4f}")
        for wline in textwrap.wrap(descriptions.get(key, ''), width=W - 4):
            L.append("    " + wline)
        L.append("")

    L.append("-" * W)
    L.append("SECTION 2 — Per-item domain metrics, aggregate")
    L.append("-" * W)

    def agg(key, label, unit):
        vals = np.array([r[key] for r in item_records if r.get(key) is not None], dtype=float)
        if len(vals) == 0:
            return
        L.append(f"{label:<34} mean={vals.mean():9.3f}{unit}   std={vals.std():9.3f}{unit}")

    agg('init_rot_err',   'Init rotation error',              ' deg')
    agg('init_trans_err', 'Init translation error',           ' mm')
    agg('rot_err',        'Rotation error',                   ' deg')
    agg('trans_err',      'Translation error',                ' mm')
    agg('add_val',        'ADD',                              ' mm')
    agg('iou_gt_vs_gt',   'IoU(gt,gt) [sanity check, ~1.0]',  '')
    agg('iou_gt_vs_est',  'IoU(gt,est)',                       '')
    agg('frame_time',     'Per-frame runtime (batch-amortized)', ' s')
    L.append("")

    L.append("-" * W)
    L.append("SECTION 3 — Per-item breakdown")
    L.append("-" * W)
    header = (f"{'frame':>8}  {'init_rot':>9}{'init_trans':>11}  "
              f"{'rot_err':>8}{'trans_err':>10}  {'ADD':>7}  "
              f"{'IoU(gt,gt)':>11}{'IoU(gt,est)':>12}{'time(s)':>9}")
    L.append(header)
    L.append("-" * len(header))
    for r in item_records:
        L.append(
            f"{r['frame_id']:>8}  "
            f"{r['init_rot_err']:>9.2f}{r['init_trans_err']:>11.2f}  "
            f"{r['rot_err']:>8.2f}{r['trans_err']:>10.2f}  {r['add_val']:>7.2f}  "
            f"{r['iou_gt_vs_gt']:>11.4f}{r['iou_gt_vs_est']:>12.4f}{r['frame_time']:>9.2f}"
        )
    L.append("")

    diam = MODEL_DIAM
    add_vals = [r['add_val'] for r in item_records]
    thr, add_acc, add_auc = compute_threshold_curve(add_vals, diam)
    L.append("-" * W)
    L.append("SECTION 4 — ADD accuracy-vs-threshold curve")
    L.append(f"(fraction of items with ADD <= threshold, swept 0 to the object diameter "
             f"{diam:.2f} mm; AUC normalized to [0,1].)")
    L.append("-" * W)
    L.append(f"ADD AUC = {add_auc:.4f}")
    L.append("")
    L.append(f"{'threshold(mm)':>16}{'ADD acc':>12}")
    for t, a in zip(thr, add_acc):
        L.append(f"{t:>16.2f}{a:>12.4f}")
    L.append("=" * W)
    return L


def write_pdf_report(pdf_path: str, npy_path: str, descriptions: dict,
                     item_records: list, dataset_dir: str, instrument_type: str):
    lines = build_report_lines(npy_path, descriptions, item_records,
                               dataset_dir, instrument_type)

    diam = MODEL_DIAM
    add_vals = [r['add_val'] for r in item_records]
    thr, add_acc, add_auc = compute_threshold_curve(add_vals, diam)

    with PdfPages(pdf_path) as pdf:
        lines_per_page = 62
        for i in range(0, len(lines), lines_per_page):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.02, 0.98, "\n".join(lines[i:i + lines_per_page]),
                     family='monospace', fontsize=7, va='top', ha='left')
            pdf.savefig(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.plot(thr, add_acc, label=f"ADD (AUC={add_auc:.3f})", color='tab:blue')
        ax.set_xlabel("Threshold (mm)")
        ax.set_ylabel("Fraction of items with ADD <= threshold")
        ax.set_title(f"ADD accuracy vs threshold — {MESH_TAG} (diameter={diam:.2f} mm)")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        for r in item_records:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.imshow(np.array(r['viz_image']))
            ax.set_title(r['label'])
            ax.axis('off')
            pdf.savefig(fig)
            plt.close(fig)


def draw_axes_and_silhouette(rgb_pil, alpha, R_bop, t_bop, K3,
                             color_sil=(0, 220, 0)) -> Image.Image:
    overlay = np.zeros((*alpha.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    rgb_arr = np.array(rgb_pil, dtype=np.float32)
    blended = np.clip(0.5 * alpha[..., None] * overlay + rgb_arr, 0, 255).astype(np.uint8)
    result  = Image.fromarray(blended)

    draw = ImageDraw.Draw(result)
    fx_, fy_, cx_, cy_ = K3[0, 0], K3[1, 1], K3[0, 2], K3[1, 2]
    def proj(p):
        pc = np.array(R_bop) @ np.array(p) + np.array(t_bop)
        return (int(round(fx_ * pc[0]/pc[2] + cx_)), int(round(fy_ * pc[1]/pc[2] + cy_))) \
               if pc[2] > 0 else None
    Lax = 15.0
    cen = proj([0, 0, 0])
    # Colors match image_generation_gt_hand_eye.py's draw_axis() convention —
    # that script draws on a cv2 BGR array with (255,0,0)/(0,255,0)/(0,0,255)
    # for X/Y/Z, which OpenCV's BGR->RGB write turns into X=blue, Y=green,
    # Z=red in the saved PNG. PIL takes RGB tuples literally, so to match:
    # X->blue, Y->green, Z->red.
    for end, col in zip([proj([Lax, 0, 0]), proj([0, Lax, 0]), proj([0, 0, Lax])],
                        [(50, 50, 220), (50, 220, 50), (220, 50, 50)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


def make_side_by_side(rgb_pil, alpha_gt, alpha_est, R_gt, t_gt, R_est, t_est, K3,
                      left_label, right_label) -> Image.Image:
    img_gt  = draw_axes_and_silhouette(rgb_pil, alpha_gt,  R_gt,  t_gt,  K3, (0, 220, 0))
    img_est = draw_axes_and_silhouette(rgb_pil, alpha_est, R_est, t_est, K3, (220, 80, 0))
    W, H = rgb_pil.size
    LH = 24
    combined = Image.new("RGB", (W * 2, H + LH), (40, 40, 40))
    combined.paste(img_gt,  (0, LH))
    combined.paste(img_est, (W, LH))
    d = ImageDraw.Draw(combined)
    d.text((8,     4), left_label,  fill=(100, 255, 100))
    d.text((W + 8, 4), right_label, fill=(255, 180, 80))
    return combined


# ── evaluate.py Evaluator setup ─────────────────────────────────────────────────
# Evaluator.__init__ loads root_path/joint.npy (mm-scale model points) — the
# LND_TEST dataset ships one, so point straight at MODEL_ROOT.
evaluator = Evaluator(root_path=EVAL_ROOT, instrument_type=INSTRUMENT_TYPE)
evaluator.diameter = MODEL_DIAM   # ADD/ADD-S 10%-diameter threshold for the mesh actually used
metric_descriptions = build_metric_descriptions(INSTRUMENT_TYPE)
print(f"Evaluator ready (instrument_type={INSTRUMENT_TYPE})")

edge_lut = edge_profile_table(_K0)

item_records = []


# ── Main loop (batches of frames) ─────────────────────────────────────────────
frames = []
for rec in _records:
    frame_id = int(rec["frame_index"])
    rgb_path = os.path.join(_image_dir, rec[IMAGE_FIELD])
    try:
        Image.open(rgb_path).verify()
    except OSError as e:
        print(f"  Frame {frame_id:04d} skipped — image load error: {e}")
        continue
    R_gt, t_gt = _load_gt(rec)
    if USE_GT_INITIALIZATION:
        init_R, init_t = perturb_pose(R_gt, t_gt, INIT_ROT_NOISE_DEG, INIT_TRANS_NOISE_MM)
    else:
        init_R, init_t = np.eye(3, dtype=np.float64), mean_t_fallback.copy()
    frames.append(dict(frame_id=frame_id, rgb_path=rgb_path, K3=record_K(rec),
                       R_gt=R_gt, t_gt=t_gt, init_R=init_R, init_t=init_t))

for b0 in range(0, len(frames), args.batch):
    chunk = frames[b0:b0 + args.batch]
    ids = [f["frame_id"] for f in chunk]
    print(f"\n══ Frames {ids[0]:04d}–{ids[-1]:04d} ({len(chunk)} in batch) ══")
    torch.cuda.synchronize(); t_batch = time.time()

    # Observed silhouettes (stand-in: crisp full-mesh render at the JSON GT).
    gt_masks = torch.stack([render_crisp_alpha_t(f["R_gt"], f["t_gt"], f["K3"]) for f in chunk])
    K = torch.tensor([[f["K3"][0, 0], f["K3"][1, 1], f["K3"][0, 2], f["K3"][1, 2]] for f in chunk],
                     dtype=torch.float32, device=device)

    if USE_GT_INIT_ONLY:
        R_est_all = np.stack([f["init_R"] for f in chunk]); t_est_all = np.stack([f["init_t"] for f in chunk])
    else:
        R_est_all, t_est_all = optimize_batch(
            gt_masks, K,
            torch.tensor(np.stack([f["init_R"] for f in chunk]), dtype=torch.float32, device=device),
            torch.tensor(np.stack([f["init_t"] for f in chunk]), dtype=torch.float32, device=device),
            edge_lut, desc=f"f{ids[0]:04d}-{ids[-1]:04d}")

    alpha_est_all = [render_crisp_alpha(R_est_all[i], t_est_all[i], f["K3"]) for i, f in enumerate(chunk)]
    torch.cuda.synchronize(); frame_time = (time.time() - t_batch) / len(chunk)

    for i, f in enumerate(chunk):
        R_gt, t_gt, R_est_np, T_est_np = f["R_gt"], f["t_gt"], R_est_all[i], t_est_all[i]
        init_rot_err   = rotation_error_deg(f["init_R"], R_gt)
        init_trans_err = translation_error_mm(f["init_t"], t_gt)
        rot_err   = rotation_error_deg(R_est_np, R_gt)
        trans_err = translation_error_mm(T_est_np, t_gt)
        add_val   = add_metric(model_pts_np, R_est_np, T_est_np, R_gt, t_gt)

        alpha_gt  = np.from_dlpack(gt_masks[i].cpu())
        alpha_est = alpha_est_all[i]
        iou_gt_vs_gt  = iou_metric(alpha_gt, alpha_gt)
        iou_gt_vs_est = iou_metric(alpha_gt, alpha_est)
        print(f"  f{f['frame_id']:04d}  init rot={init_rot_err:5.2f}° trans={init_trans_err:5.2f}mm  ->  "
              f"rot={rot_err:.2f}°  trans={trans_err:.2f}mm  ADD={add_val:.2f}mm  IoU(gt,est)={iou_gt_vs_est:.4f}")

        viz_image = make_side_by_side(
            Image.open(f["rgb_path"]).convert("RGB"), alpha_gt, alpha_est,
            R_gt, t_gt, R_est_np, T_est_np, f["K3"],
            left_label=f"GT  frame={f['frame_id']}",
            right_label=f"EST  rot={rot_err:.1f}deg trans={trans_err:.1f}mm "
                        f"ADD={add_val:.1f}mm IoU={iou_gt_vs_est:.3f}",
        )

        item_records.append({
            'frame_id': f['frame_id'],
            'init_rot_err': init_rot_err, 'init_trans_err': init_trans_err,
            'rot_err': rot_err, 'trans_err': trans_err, 'add_val': add_val,
            'iou_gt_vs_gt': iou_gt_vs_gt, 'iou_gt_vs_est': iou_gt_vs_est,
            'frame_time': frame_time,
            'viz_image': viz_image, 'label': f"frame_{f['frame_id']}",
        })

        pose_pred_34 = np.hstack([R_est_np, T_est_np.reshape(3, 1)])
        pose_gt_34   = np.hstack([R_gt,     t_gt.reshape(3, 1)])
        evaluator.evaluate(pose_gt=pose_gt_34, pose_pred=pose_pred_34)
    print(f"  batch time {frame_time * len(chunk):.1f}s  ({frame_time:.2f} s/frame)")


print("\nDone running frames — summarizing with evaluate.py's Evaluator")
evaluator.summarize(save_path=args.output_summary)
print(f"Eval summary saved to: {args.output_summary}")

pdf_summary_path = os.path.splitext(args.output_summary)[0] + '.pdf'
write_pdf_report(pdf_summary_path, args.output_summary, metric_descriptions,
                 item_records, HAND_EYE_ROOT, INSTRUMENT_TYPE)
print(f"Master report (pdf) saved to: {pdf_summary_path}")
