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
     (x, y) for a plane), "near" = clearly in front of it. Near + metallic (low-saturation,
     bright) blobs are cut to the pixels within TIP_LEN_MM of their tip end -- the point
     farthest from where the blob meets the image border (the shaft enters from the
     border, the tip doesn't), or for a blob clear of the border the end of its long axis
     farther from it -- and the largest is the tip; frames whose tip depth disagrees with
     the other frames re-pick among their candidates.
  2. MASK: GrabCut initialised with that box, kept within the near region.
  3. INIT: PnP (RANSAC, SQPnP) of the mask centroids vs the clevis centres from the
     kinematics -- a coarse calibration from images + kinematics alone.
  4. DIFF-RENDER: hand_eye_dr_refine.py's joint fit (one shared 6-DoF correction over
     all frames, same renderer/edge LUT/stages). After the first pass the masks are
     re-seeded with that script's GrabCut at the CURRENT DR estimate (clipped to the
     box), and the jaw/shaft/prong ignore zones always come from the current estimate.
     Low-IoU frames are dropped and the fit redone.
  5. EVAL on HELD-OUT frames: steps 1-4 see only the first --train-frac of the frames
     (by frame order, default 75%); the rest are never opened before the calibration is
     final. On each held-out frame every calibration is applied to that frame's
     kinematics (rect · cam_raw_T_base · base_T_clevis) and compared with the ground
     truth applied the same way -- translation error (clevis origin, mm), rotation
     error (deg), IoU of the two rendered silhouettes. What is scored is the
     calibration, not a per-frame clevis fit (there is none).

  6. PAGE 1 of the PDF: the LAST TRAINING frame -- diff-render vs ground truth overlay, ADD
     accuracy-vs-threshold curve, SurgRIPE metrics (evaluate.py's Evaluator, with the frame's own K),
     and the calibration's run time per image (load + stereo box, masks, equal share of the joint
     DR steps) and in total. Run times are only measured by a real fit (--reuse-fit reuses them).

  7. MASK BASELINE (after the fit; appended to the PDF): the final-pass masks are compared with a hand
     segmentation of the clevis using ~/hand_eye_collection_scripts/gt_segmentation_eval.py -- its
     two-panel window opens for frames not drawn yet (draw on the left, generated mask on the right;
     masks kept in <root>/hand_segmentation/ for every later run), then its metrics: IoU (whole image and
     the region the DR loss uses) and rot / trans / in-plane / depth error between the clevis poses best
     fitting each mask. A --reuse-fit whose masks are not on disk rebuilds them (same steps, seeded from the
     run's final calibration) and says so. --no-annotate scores existing drawings only;
     --no-mask-baseline skips it.

OUTPUT: <eval-dir>/eval_<dataset>_markerless_dr_comparison.pdf + .json, and
<root>/hand_eye_diff_render_comparison/ (boxes, masks, overlays). --reuse-fit rebuilds the
report from the previous run's diff-render result without refitting (fits are non-deterministic).

Run from PoseOptimizerCode/ with the diff_render env (FoundationPose/miniconda3/envs/diff_render)
or foundationpose; both have PyTorch3D + CUDA.
"""

import os, re, io, json, math, time, struct, hashlib, tempfile, textwrap, datetime, contextlib
_T_SCRIPT0 = time.time()     # one-time start-up (imports, CUDA, mesh, renderer) is reported separately
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

from evaluate import Evaluator, project as sr_project   # SurgRIPE's metric code (its run.py uses the same)


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
SEG_PDF_NAME  = "mask_vs_segmentation_eval.pdf"     # standalone report in gt_segmentation_eval.py format
HAND_SEG_SUBDIR = "hand_segmentation"                 # hand-drawn clevis masks (gt_segmentation_eval.py format)
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
NEAR_GROW_PX    = 20      # full-res; the box GrabCut mask must lie within the near region grown by this
METAL_MAX_SAT   = 90
METAL_MIN_VAL   = 90
BOX_MARGIN      = 0.2     # of the blob's larger side
TIP_LEN_MM      = 14.0    # the box covers the blob's pixels within this distance of its tip end (jaws + clevis)
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
parser.add_argument("--train-frac", type=float, default=0.75,
                    help="Fraction of the frames (in frame order) the calibration is fit on; the rest are held out "
                         "and are the only frames evaluated")
parser.add_argument("--exclude", type=int, nargs="*", default=[], help="frame_index values to leave out")
parser.add_argument("--mask-passes", type=int, default=4,
                    help="Fit rounds; round 1 uses the stereo-box masks, later rounds re-seed from the DR estimate")
parser.add_argument("--batch", type=int, default=BATCH_SIZE)
parser.add_argument("--reuse-fit", action="store_true",
                    help="Skip the fit: rebuild the report from the previous run's diff-render result (the eval "
                         "JSON) and stereo boxes. Fits are non-deterministic, so this keeps the numbers fixed.")
parser.add_argument("--no-mask-baseline", action="store_true",
                    help="Skip the mask baseline (generated masks vs hand-drawn segmentation)")
parser.add_argument("--no-annotate", action="store_true",
                    help="Mask baseline: do not open the drawing window; score only frames already hand-segmented")
parser.add_argument("--seg-tools", type=str, default="~/hand_eye_collection_scripts",
                    help="Directory holding gt_segmentation_eval.py (annotation window, silhouette fit, metrics)")
parser.add_argument("--hand-seg-dir", type=str, default=None,
                    help=f"Hand-drawn masks + segmentation_annotations.json (default: <root>/{HAND_SEG_SUBDIR})")
parser.add_argument("--seg-fit-restarts", type=int, default=4,
                    help="Mask baseline pose fit: starts per mask (gt_segmentation_eval.py --fit-restarts)")
parser.add_argument("--seg-seed", type=int, default=0)
parser.add_argument("--seg-frames", type=int, default=10,
                    help="Mask baseline: number of training frames to hand-segment and score, evenly spaced "
                         "over the frames with a generated mask (0 = all)")
parser.add_argument("--runtime-from", type=str, default=None,
                    help="With --reuse-fit, when the reused fit has no run-time log: an eval JSON of another fit "
                         "(same train frames) whose per-image run times go on page 1, labelled as from that run")
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
if len(frames) < 4:
    raise SystemExit(f"need at least 4 frames (3 to fit, 1 held out), have {len(frames)}")
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
# Baseline (mm) from P[0,3] = -fx·Tx (metres): at disparity d a millimetre spans d / BASELINE_MM pixels.
BASELINE_MM = 1000.0 * abs(float(np.array(_stereo["right"]["P"])[0, 3])) / float(np.array(_stereo["right"]["P"])[0, 0])


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


def tip_end(blob, touches_border):
    """(x, y) of the blob's tip end. The shaft enters from the image border, so when the blob touches it
    the tip is the blob pixel farthest from where it does; otherwise it is the end of the blob's long axis
    farther from the image border."""
    h, w = blob.shape
    ys, xs = np.nonzero(blob)
    if touches_border:
        edge = (xs <= 2) | (ys <= 2) | (xs >= w - 3) | (ys >= h - 3)
        k = np.argmax((xs - xs[edge].mean()) ** 2 + (ys - ys[edge].mean()) ** 2)
        return xs[k], ys[k]
    pts = np.column_stack([xs, ys]).astype(np.float64)
    axis = np.linalg.svd(pts - pts.mean(0), full_matrices=False)[2][0]
    s = (pts - pts.mean(0)) @ axis
    ends = [pts[np.argmin(s)], pts[np.argmax(s)]]
    border_dist = [min(x, y, w - 1 - x, h - 1 - y) for x, y in ends]
    return ends[int(np.argmax(border_dist))]


def tip_candidates(L, R):
    """Near + metallic blobs, largest first: (area, median disparity, full-res box), and the full-res map of
    everything in front of the board. When the tool is close its wrist, ring and shaft form one blob running
    off the border, so each blob is cut down to its pixels within TIP_LEN_MM of its tip end before boxing."""
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
    near = ok & (d > NEAR_RATIO * plane)
    near_full = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    near_full = cv2.dilate(cv2.resize(near_full, (img_W, img_H), interpolation=cv2.INTER_NEAREST), disk(NEAR_GROW_PX))
    cand = near & (hsv[..., 1] < METAL_MAX_SAT) & (hsv[..., 2] > METAL_MIN_VAL)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand)
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = st[k]
        blob = lab == k
        dk = d[blob & ok]
        if len(dk) < 20:
            continue
        touches = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        tx, ty = tip_end(blob, touches)
        r = TIP_LEN_MM * float(np.median(dk)) / BASELINE_MM / s    # TIP_LEN_MM in half-res pixels
        tip = blob & ((xx - tx) ** 2 + (yy - ty) ** 2 <= r * r)
        dk = d[tip & ok]
        if len(dk) < 20:
            continue
        ys, xs = np.nonzero(tip)
        x, y, bw, bh, area = xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, int(tip.sum())
        m = BOX_MARGIN * max(bw, bh)
        box = [int(max(0, x - m) * s), int(max(0, y - m) * s), int(min(w, x + bw + m) * s), int(min(h, y + bh + m) * s)]
        out.append((int(area), float(np.median(dk)), box))
    return sorted(out, key=lambda c: -c[0]), near_full


def stereo_boxes(images, lefts):
    """Largest candidate per frame, then re-pick in frames whose tip depth disagrees with the rest.
    Returns the boxes, each frame's near-the-camera map and each frame's stereo time (s)."""
    res, times = [], []
    for i in tqdm(range(len(images)), desc="  stereo boxes", leave=False):
        t0 = time.perf_counter()
        res.append(tip_candidates(lefts[i], images[i]))
        times.append(time.perf_counter() - t0)
    cands, nears = zip(*res)
    first = [c[0] for c in cands if c]
    if not first:
        raise SystemExit("no tool tip found in any frame")
    d_med = float(np.median([c[1] for c in first]))
    boxes = []
    for c in cands:
        agree = [x for x in c if abs(x[1] / d_med - 1) <= DEPTH_AGREE]
        pick = agree[0] if agree else (c[0] if c else None)
        boxes.append(pick[2] if pick else None)
    return boxes, list(nears), times


def box_grabcut(img, box, near) -> np.ndarray:
    """GrabCut initialised with the box (same S/V/Lab-b features as grabcut_mask), kept to the stereo near
    region (the tape and bright wood sit on the board and otherwise leak in); largest blob, holes filled."""
    x0, y0, x1, y1 = box
    m = 60
    X0, Y0, X1, Y1 = max(0, x0 - m), max(0, y0 - m), min(img_W, x1 + m), min(img_H, y1 + m)
    sub = img[Y0:Y1, X0:X1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    feat = cv2.GaussianBlur(np.dstack([hsv[..., 1], hsv[..., 2], cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)[..., 2]]), (5, 5), 0)
    gc = np.zeros(sub.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(feat, gc, (x0 - X0, y0 - Y0, x1 - x0, y1 - y0), bgd, fgd, GC_ITERS, cv2.GC_INIT_WITH_RECT)
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8) & near[Y0:Y1, X0:X1]
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
ALL = list(range(N))
# Train / held-out split by frame order. The calibration pipeline below only ever touches TRAIN frames.
n_train = min(N - 1, max(3, int(round(args.train_frac * N))))
TRAIN, TEST = ALL[:n_train], ALL[n_train:]
startup_s = time.time() - _T_SCRIPT0
t_start = time.time()
# Run time of the calibration as it would be done in practice (training frames only). Per image: reading the
# stereo pair, the stereo tool box, the masks (box GrabCut + the re-seeded GrabCut/ignore zones of every pass).
# Joint (all images at once): edge LUT, PnP init, DR passes, outlier check + refit.
rt_img = {i: dict(load_s=0.0, stereo_box_s=0.0, mask_s=0.0) for i in TRAIN}
rt_joint = dict(edge_lut_s=0.0, pnp_init_s=0.0, dr_passes_s=[], outlier_check_s=0.0, refit_s=0.0)
images = [None] * N
lefts = {}
for i in TRAIN:
    t0 = time.perf_counter()
    images[i], lefts[i] = cv2.imread(frames[i]["img_path"]), cv2.imread(frames[i]["left_path"])
    rt_img[i]["load_s"] = time.perf_counter() - t0
print(f"Split: fit on frames {[frames[i]['frame_id'] for i in TRAIN]}, "
      f"held out {[frames[i]['frame_id'] for i in TEST]}")
stem = f"eval_{DATASET}_markerless_dr_comparison"
pdf_path = os.path.join(args.eval_dir, stem + ".pdf")
json_path = os.path.join(args.eval_dir, stem + ".json")

if args.reuse_fit:
    # Only the fit's own outputs are taken from the previous run; the reference calibrations are re-read below.
    with open(json_path) as fh:
        prev = json.load(fh)
    if prev.get("train_frames") != [frames[i]["frame_id"] for i in TRAIN]:
        raise SystemExit(f"{json_path} was not fit on frames {[frames[i]['frame_id'] for i in TRAIN]} "
                         f"(it has train_frames={prev.get('train_frames')}) -- rerun without --reuse-fit")
    # The boxes only feed the debug overlays, which a reused fit does not rewrite -- so they are optional.
    _bpath = os.path.join(OUT_DIR, "stereo_boxes.json")
    _boxes = json.load(open(_bpath)) if os.path.exists(_bpath) else {}
    by_id = {f["frame_id"]: i for i, f in enumerate(frames)}
    X_init, X_dr = to_mm(prev["cam_T_base_pnp_init"]), to_mm(prev["cam_T_base_diff_render"])
    USE = [by_id[fid] for fid in prev["tip_found"]]
    inliers = [by_id[fid] for fid in prev["inliers"]]
    boxes = [_boxes.get(str(f["frame_id"])) for f in frames]
    args.mask_passes = prev["mask_passes"]
    masks = None
    fit_runtime = prev.get("fit_runtime_s")
    runtime = prev.get("runtime")
    if runtime is None and args.runtime_from:
        # The reused fit predates run-time logging: take the per-image breakdown from another (timed) run of the
        # same split, labelled as such on page 1; this fit's own total (fit_runtime_s) is still shown.
        with open(args.runtime_from) as fh:
            other = json.load(fh)
        if other.get("train_frames") != prev["train_frames"] or not other.get("runtime"):
            raise SystemExit(f"{args.runtime_from} has no run-time log for train_frames={prev['train_frames']}")
        runtime = dict(other["runtime"], source=dict(path=os.path.abspath(args.runtime_from),
                                                     generated=other["generated"]))
    # 'generated' is rewritten by every rebuild; the fit's own date is carried in 'generated_fit'
    fit_generated = prev.get("generated_fit") or prev["generated"]
    print(f"Reusing the diff-render result of {fit_generated} ({json_path})")
else:
    t0 = time.perf_counter()
    lut = edge_profile_table(frames[0]["K3"])
    rt_joint["edge_lut_s"] = time.perf_counter() - t0
    boxes, nears = [None] * N, [None] * N
    for i, b, nr, ts in zip(TRAIN, *stereo_boxes([images[i] for i in TRAIN], [lefts[i] for i in TRAIN])):
        boxes[i], nears[i] = b, nr
        rt_img[i]["stereo_box_s"] = ts
    for i in TRAIN:
        if boxes[i] is None:
            print(f"  frame {frames[i]['frame_id']}: no tool tip found -- left out")
    USE = [i for i in TRAIN if boxes[i] is not None]
    if len(USE) < 3:
        raise SystemExit("tool tip found in fewer than 3 training frames")
    box_masks = [None] * N
    for i in USE:
        t0 = time.perf_counter()
        box_masks[i] = box_grabcut(images[i], boxes[i], nears[i])
        rt_img[i]["mask_s"] += time.perf_counter() - t0
    USE = [i for i in USE if box_masks[i].sum() > 0]

    t0 = time.perf_counter()
    X_init = pnp_init(USE, box_masks)
    rt_joint["pnp_init_s"] = time.perf_counter() - t0
    X = X_init.copy()
    masks = box_masks
    for pas in range(args.mask_passes):
        if pas > 0:
            masks = [None] * N
            for i in tqdm(USE, desc="  GrabCut masks", leave=False):
                t0 = time.perf_counter()
                masks[i] = grabcut_mask(images[i], frames[i], X) * box_region(boxes[i], BOX_CLIP_PX)
                rt_img[i]["mask_s"] += time.perf_counter() - t0
        valid = [None] * N
        for i in USE:
            t0 = time.perf_counter()
            valid[i] = valid_map(frames[i], X)
            rt_img[i]["mask_s"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        X_prev, X = X, fit_shared(USE, X, masks, valid, lut, desc=f"pass {pas + 1}/{args.mask_passes}")
        rt_joint["dr_passes_s"].append(time.perf_counter() - t0)
        dp, rp = calib_change(X_prev, X, USE)
        print(f"Pass {pas + 1}/{args.mask_passes}: moved {dp:.2f} mm / {rp:.2f} deg, mean valid IoU "
              f"{np.mean([iou_valid(i, X, masks, valid) for i in USE]):.3f}")

    t0 = time.perf_counter()
    ious = {i: iou_valid(i, X, masks, valid) for i in USE}
    rt_joint["outlier_check_s"] = time.perf_counter() - t0
    thr = max(OUTLIER_IOU, float(np.median(list(ious.values()))) - OUTLIER_IOU_DROP)
    inliers = [i for i in USE if ious[i] >= thr]
    if len(inliers) < 3:
        print(f"Fewer than 3 frames reach IoU {thr:.3f} -- keeping all")
        inliers = USE
    elif len(inliers) < len(USE):
        print(f"Outliers (IoU < {thr:.3f}): {[frames[i]['frame_id'] for i in USE if i not in inliers]} "
              f"-- refitting on {len(inliers)} frames")
        t0 = time.perf_counter()
        X = fit_shared(inliers, X, masks, valid, lut, desc="inliers")
        rt_joint["refit_s"] = time.perf_counter() - t0
    X_dr = X
    fit_runtime = time.time() - t_start
    joint_s = rt_joint["edge_lut_s"] + rt_joint["pnp_init_s"] + sum(rt_joint["dr_passes_s"]) \
        + rt_joint["outlier_check_s"] + rt_joint["refit_s"]
    per_img_s = sum(sum(v.values()) for v in rt_img.values())
    runtime = dict(
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        n_images=len(TRAIN), n_images_fit=len(USE),
        per_image={str(frames[i]["frame_id"]): dict(
            **rt_img[i], used_in_fit=i in USE,
            joint_share_s=joint_s / len(USE) if i in USE else 0.0) for i in TRAIN},
        joint=rt_joint, joint_total_s=joint_s, per_image_total_s=per_img_s,
        other_s=fit_runtime - joint_s - per_img_s,        # bookkeeping, progress bars, Python overhead
        total_s=fit_runtime, startup_s=startup_s)


# ══ Evaluation (reference calibrations are read only from here on) ════════════
def load_reference_calibrations():
    with open(os.path.join(ROOT, VARIANTS_JSON)) as fh:
        meta = json.load(fh)["meta"]
    return to_mm(meta["cam_T_base_post_alignment"]), to_mm(meta["cam_T_base_pre_alignment"])


X_gt, X_marker = load_reference_calibrations()
for i in ALL:
    if images[i] is None:
        images[i] = cv2.imread(frames[i]["img_path"])
METHODS = [("hand-eye (marker only)", X_marker), ("diff-render (markerless)", X_dr)]


def frame_errors(X, i):
    """Clevis pose under X vs under the ground truth, in the rectified camera frame."""
    T, Tg = frame_poses_np(X, frames[i])[0], frame_poses_np(X_gt, frames[i])[0]
    return float(np.linalg.norm(T[:3, 3] - Tg[:3, 3])), rot_deg(T[:3, :3], Tg[:3, :3])


rows = []
for i in TEST:
    r_gt = render_mask(X_gt, i)
    row = dict(frame_id=frames[i]["frame_id"])
    for name, Xc in METHODS:
        r = render_mask(Xc, i)
        t_err, r_err = frame_errors(Xc, i)
        row[name] = dict(trans_err_mm=t_err, rot_err_deg=r_err, iou_vs_gt=iou_full(r, r_gt),
                         add_mm=pose_displacement_mm(frame_poses_np(Xc, frames[i])[0], frame_poses_np(X_gt, frames[i])[0]))
    rows.append(row)

overall = {}
for name, Xc in METHODS:
    d, r = calib_change(X_gt, Xc, TEST)
    overall[name] = dict(
        mean_clevis_displacement_mm=d, rotation_err_deg=r,
        base_origin_err_mm=float(np.linalg.norm(Xc[:3, 3] - X_gt[:3, 3])),
        eval_mean_trans_err_mm=float(np.mean([row[name]["trans_err_mm"] for row in rows])),
        eval_mean_rot_err_deg=float(np.mean([row[name]["rot_err_deg"] for row in rows])),
        eval_mean_iou_vs_gt=float(np.mean([row[name]["iou_vs_gt"] for row in rows])))
d0, r0 = calib_change(X_gt, X_init, TEST)

# ── Calibration image: the last training frame, scored with SurgRIPE's metrics (evaluate.py) ─────────────────
CAL = TRAIN[-1]
CAL_STATUS = ("used in the fit" if CAL in inliers else
              "dropped as an outlier" if CAL in USE else "no tool tip found -- not used in the fit")
SR_MODEL = np.load(os.path.join(args.model_root, "joint.npy")).astype(np.float64)   # what Evaluator loads
if np.abs(SR_MODEL).max() < 1.0:
    SR_MODEL = SR_MODEL * 1000.0
SR_DIAM = float(np.linalg.norm(np.ptp(SR_MODEL, axis=0)))   # = evaluate.py's LND diameter for joint.npy


def threshold_curve(values, max_thr, n=401):
    """Fraction of items with ADD <= threshold over 0..max_thr, and its normalised AUC
    (as hand_eye_diff_render_test.py's compute_threshold_curve, finer sampling)."""
    v = np.asarray(values, dtype=float)
    thr = np.linspace(0.0, max_thr, n)
    acc = np.array([(v <= t).mean() for t in thr])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return thr, acc, float(trapz(acc, thr) / max_thr)


def surgripe_metrics(X, i) -> dict:
    """evaluate.py's Evaluator on one frame. Its fixed SurgRIPE camera K is replaced by this frame's (the 2D
    reprojection test would otherwise use another camera); the ADD threshold is 10% of joint.npy's diameter."""
    ev = Evaluator(root_path=args.model_root, instrument_type="LND", is_save_pose=False)
    ev.model, ev.diameter, ev.K = SR_MODEL, SR_DIAM, frames[i]["K3"]
    Tp, Tg = frame_poses_np(X, frames[i])[0], frame_poses_np(X_gt, frames[i])[0]
    ev.evaluate(pose_gt=Tg[:3], pose_pred=Tp[:3])
    with contextlib.redirect_stdout(io.StringIO()):
        out = {k: float(v) for k, v in ev.summarize().items()}
    out["proj2d_err_px"] = float(np.linalg.norm(sr_project(SR_MODEL, frames[i]["K3"], Tp[:3])
                                                - sr_project(SR_MODEL, frames[i]["K3"], Tg[:3]), axis=1).mean())
    out["add_auc_0_diam"] = threshold_curve([out["ADD-distance"]], SR_DIAM)[2]
    out["iou_vs_gt"] = iou_full(render_mask(X, i), render_mask(X_gt, i))
    return out


cal_metrics = {name: surgripe_metrics(Xc, CAL) for name, Xc in METHODS}

# ── Text summary ──────────────────────────────────────────────────────────────
lines = []
W_ = 104
lines.append("=" * W_)
lines.append("hand_eye_diff_render_comparison.py — markerless hand-eye calibration by differentiable rendering".center(W_))
lines.append("=" * W_)
lines.append(f"Dataset   : {ROOT}  ({N} frames)")
lines.append(f"Fit on    : frames {[frames[i]['frame_id'] for i in TRAIN]} (first {args.train_frac:.0%}; tip found in "
             f"{len(USE)}, {len(inliers)} kept after outlier rejection)")
lines.append(f"Held out  : frames {[frames[i]['frame_id'] for i in TEST]} -- never opened by the fit; the only frames scored")
lines.append(f"Generated : {datetime.datetime.now().isoformat(timespec='seconds')}   fit runtime "
             f"{f'{fit_runtime:.0f}s' if fit_runtime is not None else 'n/a (report rebuilt with --reuse-fit)'}")
lines.append("Ground truth = marker registration + 3D pose alignment (cam_T_base_post_alignment); used ONLY for scoring.")
lines.append("DR inputs   = rectified right+left images, intrinsics / rect_T_cam_raw / stereo P, joint.stl, PSM kinematics.")
lines.append("              No marker calibration, no ground truth, no initial pose, no annotation.")
lines.append(f"Pipeline    = stereo tool box -> GrabCut -> PnP init -> {args.mask_passes} DR passes (masks re-seeded from the")
lines.append("              DR estimate after pass 1) -> outlier rejection + refit.")
lines.append("")
lines.append("Whole calibration vs ground truth (held-out frames):")
lines.append(f"  {'':<26}{'clevis displ. (mm)':>20}{'rotation (deg)':>16}{'base origin (mm)':>18}")
lines.append(f"  {'PnP init (no prior)':<26}{d0:>20.2f}{r0:>16.2f}{np.linalg.norm(X_init[:3, 3] - X_gt[:3, 3]):>18.2f}")
for name, _ in METHODS:
    o = overall[name]
    lines.append(f"  {name:<26}{o['mean_clevis_displacement_mm']:>20.2f}{o['rotation_err_deg']:>16.2f}"
                 f"{o['base_origin_err_mm']:>18.2f}")
lines.append("")
lines.append("Held-out frames (calibration applied to each frame's kinematics vs the ground truth applied the same "
             "way; IoU of the rendered silhouettes):")
hdr = f"  {'frame':>5} " + "".join(f"{n[:14] + ' t(mm)':>23}{'rot(deg)':>10}{'IoU':>7}" for n, _ in METHODS)
lines.append(hdr)
lines.append("  " + "-" * (len(hdr) - 2))
for row in rows:
    lines.append(f"  {row['frame_id']:>5} " +
                 "".join(f"{row[n]['trans_err_mm']:>23.2f}{row[n]['rot_err_deg']:>10.2f}{row[n]['iou_vs_gt']:>7.3f}"
                         for n, _ in METHODS))
lines.append("  " + "-" * (len(hdr) - 2))
lines.append(f"  {'mean':>5} " + "".join(
    f"{overall[n]['eval_mean_trans_err_mm']:>23.2f}{overall[n]['eval_mean_rot_err_deg']:>10.2f}"
    f"{overall[n]['eval_mean_iou_vs_gt']:>7.3f}" for n, _ in METHODS))
lines.append("  (t = clevis origin / pitch pin; rotation is the calibration's rotation error, the same for every frame)")
lines.append("")
lines.append("cam_raw_T_base (m), diff-render:")
lines += ["  " + np.array2string(row, precision=6, suppress_small=True) for row in to_m(X_dr)]
lines.append("=" * W_)
print("\n" + "\n".join(lines))


# ── Overlays + PDF + JSON ─────────────────────────────────────────────────────
# Per-frame figures follow hand_eye_diff_render_test.py's make_side_by_side: full frame, filled silhouette
# (ground truth green, calibration under test orange) plus the clevis axes, label bar on top.
_LABEL_FONT = ImageFont.load_default(size=34)


def draw_axes_and_silhouette(rgb_pil, alpha, T, K3, color_sil) -> Image.Image:
    overlay = np.zeros((*alpha.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    blended = np.clip(0.5 * alpha[..., None] * overlay + np.array(rgb_pil, dtype=np.float32), 0, 255).astype(np.uint8)
    result = Image.fromarray(blended)
    draw = ImageDraw.Draw(result)

    def proj(p):
        pc = T[:3, :3] @ np.array(p, dtype=np.float64) + T[:3, 3]
        return (int(round(K3[0, 0] * pc[0] / pc[2] + K3[0, 2])), int(round(K3[1, 1] * pc[1] / pc[2] + K3[1, 2]))) \
            if pc[2] > 0 else None
    cen = proj([0, 0, 0])
    # X->blue, Y->green, Z->red, as hand_eye_diff_render_test.py (image_generation_gt_hand_eye.py's convention)
    for end, col in zip([proj([15.0, 0, 0]), proj([0, 15.0, 0]), proj([0, 0, 15.0])],
                        [(50, 50, 220), (50, 220, 50), (220, 50, 50)]):
        if cen and end:
            draw.line([cen, end], fill=col, width=4)
    return result


def make_side_by_side(i, X_est, left_label, right_label) -> np.ndarray:
    rgb = Image.fromarray(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    T_gt, T_est = frame_poses_np(X_gt, frames[i])[0], frame_poses_np(X_est, frames[i])[0]
    img_gt = draw_axes_and_silhouette(rgb, render_mask(X_gt, i).astype(np.float32), T_gt, frames[i]["K3"], (0, 220, 0))
    img_est = draw_axes_and_silhouette(rgb, render_mask(X_est, i).astype(np.float32), T_est, frames[i]["K3"], (220, 80, 0))
    LH = 48
    combined = Image.new("RGB", (img_W * 2, img_H + LH), (40, 40, 40))
    combined.paste(img_gt, (0, LH))
    combined.paste(img_est, (img_W, LH))
    d = ImageDraw.Draw(combined)
    d.text((12, 6), left_label, fill=(100, 255, 100), font=_LABEL_FONT)
    d.text((img_W + 12, 6), right_label, fill=(255, 180, 80), font=_LABEL_FONT)
    return np.array(combined)


def _style_table(tab, ncols, fs):
    tab.auto_set_font_size(False); tab.set_fontsize(fs)
    for c in range(ncols):
        tab[0, c].set_facecolor("#e8e8e8"); tab[0, c].set_text_props(weight="bold")


def error_breakdown(X, i) -> dict:
    """Where the pose error lies, in the rectified camera frame: along the line of sight to the clevis (depth --
    the direction a single image constrains least), sideways, and the rotation split into roll about the clevis's
    long axis (near-symmetric part, hard to see in a silhouette), about the line of sight, and the rest."""
    Tp, Tg = frame_poses_np(X, frames[i])[0], frame_poses_np(X_gt, frames[i])[0]
    K3 = frames[i]["K3"]
    d = Tp[:3, 3] - Tg[:3, 3]
    view = Tg[:3, 3] / np.linalg.norm(Tg[:3, 3])
    along = float(d @ view)
    rv = Rotation.from_matrix(Tp[:3, :3] @ Tg[:3, :3].T).as_rotvec()
    axis = Tg[:3, 1]                                      # joint_model_pose: y = clevis long axis (-forward)
    size = lambda T: float(np.ptp(sr_project(SR_MODEL, K3, T[:3]), axis=0).max())
    return dict(depth_gt_mm=float(Tg[2, 3]), depth_est_mm=float(Tp[2, 3]),
                trans_along_sight_mm=along, trans_sideways_mm=float(np.linalg.norm(d - along * view)),
                rot_total_deg=float(np.degrees(np.linalg.norm(rv))),
                rot_about_clevis_axis_deg=float(np.degrees(abs(rv @ axis))),
                rot_about_sight_deg=float(np.degrees(abs(rv @ view))),
                projected_size_gt_px=size(Tg), projected_size_est_px=size(Tp))


def outline_crop(i, X_est, pad=110) -> np.ndarray:
    """Zoom on the tool: ground-truth (green) and estimated (orange) silhouette OUTLINES on the raw image, so a
    scale (depth) difference is visible -- filled full-frame overlays hide it."""
    img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB).copy()
    mg, me = render_mask(X_gt, i), render_mask(X_est, i)
    for m, col in ((mg, (0, 220, 0)), (me, (255, 130, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cs, -1, col, 3)
    ys, xs = np.nonzero(mg | me)
    return img[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def calibration_frame_pages(pdf):
    """Front section: diff-render alone, as a 6D pose estimator, on the last training image (2 pages)."""
    fid = frames[CAL]["frame_id"]
    DR = METHODS[1][0]
    m, eb = cal_metrics[DR], cal_breakdown
    head = (f"{DATASET}: diff-render (markerless) as a 6D pose estimator -- last training frame {fid} "
            f"({CAL_STATUS})\nscored against the ground-truth pose (marker + 3D pose alignment)")

    # ── Page 1: overlay, zoomed outlines, SurgRIPE metrics, error breakdown ──
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(head, fontsize=11, y=0.985)
    ax = fig.add_axes([0.01, 0.49, 0.98, 0.41])
    ax.imshow(make_side_by_side(CAL, X_dr, f"GT (marker + 3D alignment)  frame={fid}", DR))
    ax.set_title(f"{DR}:  translation error {m['trans_error']:.2f} mm   rotation error {m['rot_error']:.2f} deg   "
                 f"ADD {m['ADD-distance']:.2f} mm   IoU {m['iou_vs_gt']:.3f}", fontsize=10)
    ax.axis("off")

    ax = fig.add_axes([0.01, 0.07, 0.30, 0.38])
    ax.imshow(outline_crop(CAL, X_dr)); ax.axis("off")
    ax.set_title("zoom: outlines  GT (green) vs diff-render (orange)", fontsize=8.5)

    ax = fig.add_axes([0.35, 0.07, 0.27, 0.37]); ax.axis("off")
    ax.set_title(f"SurgRIPE metrics (evaluate.py), frame {fid}", fontsize=8.5)
    pf = lambda v: "pass" if v >= 0.5 else "fail"
    srows = [("ADD (mm)", "ADD-distance", "{:.2f}"),
             (f"ADD < 10% diam. ({0.1 * SR_DIAM:.2f} mm)", "add", pf),
             ("ADD-S (mm)", "ADDS-distance", "{:.2f}"),
             ("ADD-S < 10% diam.", "add-s", pf),
             ("2D proj. error (px)", "proj2d_err_px", "{:.1f}"),
             ("2D proj. < 5 px", "proj2d", pf),
             ("5 mm / 5 deg", "cmd5", pf),
             ("translation error (mm)", "trans_error", "{:.2f}"),
             ("rotation error (deg)", "rot_error", "{:.2f}"),
             ("Avg Acc (0-5 mm)", "avg_acc_0_5mm", "{:.3f}"),
             ("ADD AUC (0-diam.)", "add_auc_0_diam", "{:.3f}"),
             ("silhouette IoU vs GT *", "iou_vs_gt", "{:.3f}")]
    fmt = lambda f, v: f(v) if callable(f) else f.format(v)
    tab = ax.table(cellText=[[lab, fmt(f, m[k])] for lab, k, f in srows], colLabels=["Metric", "diff-render"],
                   cellLoc="center", colWidths=[0.62, 0.38], bbox=[0, 0, 1, 1])
    _style_table(tab, 2, 7)
    tab[0, 1].set_facecolor("#d9efd9")

    ax = fig.add_axes([0.66, 0.07, 0.33, 0.37]); ax.axis("off")
    ax.set_title(f"Where the error lies (camera frame), frame {fid}", fontsize=8.5)
    brows = [("clevis depth: GT / diff-render (mm)", f"{eb['depth_gt_mm']:.1f} / {eb['depth_est_mm']:.1f}"),
             ("translation along line of sight (mm)", f"{eb['trans_along_sight_mm']:+.2f}"),
             ("translation sideways (mm)", f"{eb['trans_sideways_mm']:.2f}"),
             ("rotation, total (deg)", f"{eb['rot_total_deg']:.2f}"),
             ("  about clevis long axis (roll)", f"{eb['rot_about_clevis_axis_deg']:.2f}"),
             ("  about line of sight", f"{eb['rot_about_sight_deg']:.2f}"),
             ("projected clevis size: GT / DR (px)",
              f"{eb['projected_size_gt_px']:.0f} / {eb['projected_size_est_px']:.0f}"),
             ("apparent scale DR vs GT", f"{eb['projected_size_est_px'] / eb['projected_size_gt_px']:.2f}x")]
    tab = ax.table(cellText=[list(r) for r in brows], colLabels=["", "value"], cellLoc="center",
                   colWidths=[0.68, 0.32], bbox=[0, 0.30, 1, 0.70])
    _style_table(tab, 2, 7)
    ax.text(0, 0.0, textwrap.fill(
        "Negative along-sight = diff-render places the clevis closer to the camera than the ground truth. A depth "
        "error mostly changes the silhouette's SIZE, which the filled full-frame overlay above hides; the zoomed "
        "outlines show it. Roll about the clevis's long axis barely changes its near-symmetric silhouette.", 70),
        fontsize=6.2, va="bottom", transform=ax.transAxes)

    fig.text(0.01, 0.008, textwrap.fill(
        f"Frame {fid} is the last image the markerless diff-render calibration was fit on; the calibration is applied "
        "to its kinematics and compared with the ground truth applied the same way (ground truth used only for "
        "scoring). Metrics: SurgRIPE's evaluate.py Evaluator (ADD / ADD-S pass at 10% of the joint.npy diameter, "
        "2D projection < 5 px with this frame's rectified K instead of SurgRIPE's fixed K, 5 mm 5 deg, Avg Acc "
        "0-5 mm as in the SurgRIPE paper's Table 4). * IoU is not a SurgRIPE metric. One image -> pass rates are 0/1.",
        230), fontsize=6.3, va="bottom")
    pdf.savefig(fig); plt.close(fig)

    # ── Page 2: ADD threshold curve + run time ──
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(head, fontsize=11, y=0.985)
    add = m["ADD-distance"]
    x_max = 5.0 * math.ceil(max(SR_DIAM, 1.15 * add) / 5.0)       # round axis end, past both the ADD and diameter
    ax = fig.add_axes([0.07, 0.18, 0.42, 0.66])
    thr, acc, _ = threshold_curve([add], x_max, n=2001)
    ax.plot(thr, acc, color="tab:green", lw=2.2, label=f"diff-render, frame {fid}  (AUC 0-diam. = {m['add_auc_0_diam']:.3f})")
    for x, t in ((0.1 * SR_DIAM, f"10% diam. = {0.1 * SR_DIAM:.2f} mm (ADD pass)"),
                 (SR_DIAM, f"diameter = {SR_DIAM:.2f} mm (AUC limit)")):
        ax.axvline(x, color="0.45", lw=0.9, ls="--")
        ax.text(x - 0.15, 0.55, t, fontsize=7.5, color="0.35", rotation=90, va="center", ha="right")
    ax.axvline(add, color="tab:green", lw=0.8, ls=":")
    ax.annotate(f"ADD = {add:.2f} mm", (add, 0.3), xytext=(add - 0.3, 0.3), fontsize=8, color="tab:green",
                va="center", ha="right", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(0, x_max); ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(np.arange(0, x_max + 1e-9, 1.0 if x_max <= 20 else 2.0))
    ax.set_xlabel("ADD threshold (mm)", fontsize=9); ax.set_ylabel("fraction of items with ADD <= threshold", fontsize=9)
    ax.set_title(f"ADD accuracy vs threshold, frame {fid} (joint.npy, diameter {SR_DIAM:.2f} mm)", fontsize=9.5)
    ax.tick_params(labelsize=8); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.text(0.07, 0.03, textwrap.fill(
        "A single image is one item, so the curve is a step at that image's ADD: 0 below it, 1 at and above it. "
        "SurgRIPE's AUC integrates the curve from 0 to the object diameter, so it is 0 whenever ADD exceeds the "
        "diameter.", 88), fontsize=7.5, va="bottom")

    ax = fig.add_axes([0.55, 0.06, 0.43, 0.80]); ax.axis("off")
    if runtime is None:
        ax.set_title("Calibration run time", fontsize=9.5)
        ax.text(0.5, 0.5, "n/a -- the reused fit predates run-time logging;\nrerun without --reuse-fit",
                ha="center", va="center", fontsize=9)
    else:
        dev = runtime["device"].replace("NVIDIA ", "").replace("GeForce ", "")
        ax.set_title(f"Calibration run time (s): {runtime['n_images']} training images, {dev}", fontsize=9.5)
        cells = []
        for k, v in runtime["per_image"].items():
            tot = v["load_s"] + v["stereo_box_s"] + v["mask_s"] + v["joint_share_s"]
            cells.append([k + ("" if v["used_in_fit"] else " (unused)"), f"{v['load_s'] + v['stereo_box_s']:.2f}",
                          f"{v['mask_s']:.2f}", f"{v['joint_share_s']:.2f}", f"{tot:.2f}"])
        nimg = len(cells)
        cells.append(["mean / image"] + [f"{np.mean([float(c[j]) for c in cells]):.2f}" for j in range(1, 5)])
        borrowed = "source" in runtime
        cells.append(["TOTAL" + (" (timed)" if borrowed else ""), "", "", "", f"{runtime['total_s']:.1f}"])
        if borrowed:
            cells.append(["TOTAL (this fit)", "", "", "",
                          f"{fit_runtime:.1f}" if fit_runtime is not None else "n/a"])
        tab = ax.table(cellText=cells, colLabels=["frame", "load +\nstereo box", "masks", "joint DR\nshare",
                                                  "total /\nimage"], cellLoc="center", bbox=[0, 0.2, 1, 0.8])
        _style_table(tab, 5, 8)
        for r in range(nimg + 1, len(cells) + 1):
            for c in range(5):
                tab[r, c].set_text_props(weight="bold")
        j = runtime["joint"]
        ax.text(0, 0.0, textwrap.fill(
            f"Joint steps (all {runtime['n_images_fit']} fit images at once, split equally above): edge LUT "
            f"{j['edge_lut_s']:.1f}, PnP init {j['pnp_init_s']:.2f}, DR passes "
            f"{' + '.join(f'{x:.1f}' for x in j['dr_passes_s'])}, outlier check {j['outlier_check_s']:.1f}, refit "
            f"{j['refit_s']:.1f} = {runtime['joint_total_s']:.1f} s. Python overhead {runtime['other_s']:.1f} s is "
            f"in TOTAL only. One-time start-up (imports, CUDA, mesh, renderer) {runtime['startup_s']:.1f} s "
            f"extra -> {runtime['total_s'] + runtime['startup_s']:.1f} s from launch."
            + (f" The fit in this report predates run-time logging (only its total was recorded); the "
               f"per-image times are from a separate timed fit of the same {runtime['n_images']} images "
               f"({runtime['source']['generated']})." if borrowed else ""), 95),
            fontsize=7, va="bottom", transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def metrics_table_page(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{DATASET}: hand-eye (marker only) vs diff-render (markerless), both scored against the ground truth "
                 f"(marker + 3D pose alignment)", fontsize=11, y=0.96)

    ax = fig.add_axes([0.04, 0.50, 0.92, 0.38]); ax.axis("off")
    ax.set_title(f"Held-out frames {[r['frame_id'] for r in rows]} (fit on frames "
                 f"{[frames[i]['frame_id'] for i in TRAIN]}; held-out images never seen by the fit)", fontsize=10, loc="left")
    cols = ["Frame"] + [f"{n.split(' (')[0]}\n{m}" for n, _ in METHODS
                        for m in ("trans. err (mm)", "rot. err (deg)", "IoU")]
    cells = [[f"{row['frame_id']}"] +
             [f"{row[n][k]:.{p}f}" for n, _ in METHODS
              for k, p in (("trans_err_mm", 2), ("rot_err_deg", 2), ("iou_vs_gt", 3))] for row in rows]
    cells.append(["Mean"] + [f"{overall[n][k]:.{p}f}" for n, _ in METHODS
                             for k, p in (("eval_mean_trans_err_mm", 2), ("eval_mean_rot_err_deg", 2),
                                          ("eval_mean_iou_vs_gt", 3))])
    tab = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.9)
    for c in range(len(cols)):
        tab[0, c].set_facecolor("#e8e8e8"); tab[0, c].set_text_props(weight="bold")
        tab[len(cells), c].set_text_props(weight="bold")
    for c in range(1, 4):
        tab[0, c].set_facecolor("#f6e3d2")        # hand-eye columns
    for c in range(4, 7):
        tab[0, c].set_facecolor("#d9efd9")        # diff-render columns

    ax2 = fig.add_axes([0.14, 0.24, 0.72, 0.22]); ax2.axis("off")
    ax2.set_title("Whole calibration, averaged over the held-out frames", fontsize=10, loc="left")
    cal = [["PnP init (no prior)", f"{d0:.2f}", f"{r0:.2f}", f"{np.linalg.norm(X_init[:3, 3] - X_gt[:3, 3]):.2f}"]]
    cal += [[n, f"{overall[n]['mean_clevis_displacement_mm']:.2f}", f"{overall[n]['rotation_err_deg']:.2f}",
             f"{overall[n]['base_origin_err_mm']:.2f}"] for n, _ in METHODS]
    tab2 = ax2.table(cellText=cal, colLabels=["Calibration", "mean clevis\ndisplacement (mm)", "rotation\nerr (deg)",
                                             "base-origin\nerr (mm)"], loc="upper center", cellLoc="center")
    tab2.auto_set_font_size(False); tab2.set_fontsize(9); tab2.scale(1, 1.9)
    for c in range(4):
        tab2[0, c].set_facecolor("#e8e8e8"); tab2[0, c].set_text_props(weight="bold")

    fig.text(0.04, 0.05, textwrap.fill(
        "trans. err = distance between the clevis origin (pitch pin) under the calibration and under the ground truth; "
        "rot. err = geodesic angle between the two clevis orientations (equal to the calibration's rotation error, so "
        "the same in every frame); IoU = overlap of the clevis silhouettes rendered at the calibration and at the "
        "ground truth. Each calibration is applied to the held-out frame's kinematics; nothing is fit per frame. "
        "Diff-render inputs (training frames only): "
        "rectified images, camera/stereo calibration, joint.stl, PSM kinematics -- no ground truth, marker calibration "
        "or initial pose.", 175), fontsize=7.5, va="bottom")
    pdf.savefig(fig); plt.close(fig)


if masks is not None:
    for i in USE:
        fid = frames[i]["frame_id"]
        cv2.imwrite(os.path.join(OUT_DIR, "masks", f"mask-{fid:04d}.png"), masks[i] * 255)
        o = images[i].copy()
        cv2.rectangle(o, tuple(boxes[i][:2]), tuple(boxes[i][2:]), (0, 0, 255), 3)
        cv2.imwrite(os.path.join(OUT_DIR, "overlays", f"box-{fid:04d}.jpg"), o)
    with open(os.path.join(OUT_DIR, "stereo_boxes.json"), "w") as fh:
        json.dump({str(frames[i]["frame_id"]): boxes[i] for i in ALL}, fh)

# ══ Mask baseline: the generated masks vs a hand-drawn segmentation ══════════════════════════════════════════
# The final-pass masks the diff-render fit used, scored against a hand segmentation of the clevis with
# gt_segmentation_eval.py's tools and metrics: its two-panel annotation window (draw on the left, the generated
# mask on the right), IoU, and the rotation / translation (in-plane, depth) error between the clevis pose best
# fitting the hand mask and the one best fitting the generated mask (its Nelder-Mead silhouette fit, same stages).
# The generated masks cover the whole silver tool (clevis + jaws + ring) while the hand masks are the clevis
# only (gt_segmentation_eval.py's instruction) and the DR loss ignores the jaw / shaft / ring-prong zones -- so
# the pose fits and the "DR region" IoU use only the pixels the DR loss uses (valid_map at the final estimate);
# the full-mask IoU is reported too. Runs after the fit, so nothing here can influence the calibration.
AUTO_MASK_DIR = os.path.join(OUT_DIR, "masks")
HAND_DIR = args.hand_seg_dir or os.path.join(ROOT, HAND_SEG_SUBDIR)
_md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
_mask_path = lambda fid: os.path.join(AUTO_MASK_DIR, f"mask-{fid:04d}.png")


def final_auto_masks():
    """{frame index: uint8 0/1 mask} of the masks the final fit used, and how they were obtained. A reused fit
    loads the saved masks if they are that run's (md5s in its JSON); otherwise they are rebuilt with the same steps
    (stereo box -> GrabCut, then the estimate-seeded GrabCut clipped to the box), seeded from the run's FINAL
    calibration because the estimate the last pass was seeded from was not saved."""
    if masks is not None:
        am = {i: masks[i] for i in USE if masks[i] is not None and masks[i].any()}
        return am, dict(reconstructed=False, md5={str(frames[i]["frame_id"]): _md5(m) for i, m in am.items()})
    saved = prev.get("auto_masks")
    if saved:
        am = {}
        for i in USE:
            p = _mask_path(frames[i]["frame_id"])
            if os.path.exists(p):
                am[i] = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if am and {str(frames[i]["frame_id"]): _md5(m) for i, m in am.items()} == saved["md5"]:
            print(f"Mask baseline: using the run's saved masks ({AUTO_MASK_DIR})")
            return am, saved
    print("Mask baseline: this run's masks are not on disk -- rebuilding them from its final calibration")
    bx, nr, _ = stereo_boxes([images[i] for i in TRAIN], [lefts[i] for i in TRAIN])
    bx, nr = dict(zip(TRAIN, bx)), dict(zip(TRAIN, nr))
    am = {}
    for i in tqdm(USE, desc="  rebuild masks", leave=False):
        if bx[i] is None:
            continue
        m = (grabcut_mask(images[i], frames[i], X_dr) * box_region(bx[i], BOX_CLIP_PX) if args.mask_passes > 1
             else box_grabcut(images[i], bx[i], nr[i]))
        if m.any():
            am[i] = m.astype(np.uint8)
            cv2.imwrite(_mask_path(frames[i]["frame_id"]), am[i] * 255)
    return am, dict(reconstructed=True,
                    note=(f"rebuilt {datetime.datetime.now().isoformat(timespec='seconds')}: the run predates saving "
                          "its masks; same mask steps, last pass seeded from the run's final calibration instead of "
                          f"its pass-{args.mask_passes - 1} estimate (not saved)"),
                    md5={str(frames[i]["frame_id"]): _md5(m) for i, m in am.items()})


def fit_pose_to_mask_weighted(seg, model, K, mask, weight, T0, window, restarts, rng):
    """gt_segmentation_eval.fit_pose_to_mask (same stages, simplex, restarts and selection), with both the target
    and the rendered silhouette multiplied by `weight` so ignored pixels count for neither."""
    x0, y0, x1, y1 = window
    scale = min(1.0, seg.FIT_MAX_SIDE_PX / max(x1 - x0, y1 - y0))
    fw, fh = max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))
    Kf, Kfull = seg.window_camera(K, x0, y0, scale), seg.window_camera(K, x0, y0, 1.0)
    rs = lambda a: cv2.resize(a[y0:y1, x0:x1].astype(np.float32), (fw, fh), interpolation=cv2.INTER_AREA)
    target, wgt = rs(mask), rs(weight)
    w_full = weight[y0:y1, x0:x1] > 0.5
    m_full = mask[y0:y1, x0:x1] > 0
    pivot = model.posed(T0)[0].mean(axis=0)

    def render_fit(x, cell_mm):
        return seg.render_silhouette(model, seg.delta_from_params(x, pivot) @ T0, Kf, (fw, fh), cell_mm,
                                     seg.FIT_SUPERSAMPLE).astype(np.float32) / 255.0

    starts = [np.zeros(6)]
    t_rng, r_rng = seg.FIT_RESTART_RANGE
    for _ in range(max(0, restarts - 1)):
        starts.append(np.concatenate([rng.uniform(-t_rng, t_rng, 3), rng.uniform(-r_rng, r_rng, 3)]))
    best_iou, best_pose = -1.0, T0
    for start in starts:
        x = start
        for cell_mm, sigma, (t_step, r_step), max_evals in seg.FIT_STAGES:
            tb = cv2.GaussianBlur(target, (0, 0), sigma) * wgt
            simplex = np.vstack([x] + [x + np.eye(6)[k] * (t_step if k < 3 else r_step) for k in range(6)])
            x = seg.minimize(
                lambda p: 1.0 - seg.soft_iou(cv2.GaussianBlur(render_fit(p, cell_mm), (0, 0), sigma) * wgt, tb),
                x, method="Nelder-Mead",
                options={"initial_simplex": simplex, "maxfev": max_evals, "xatol": 0.01, "fatol": 1e-5}).x
        pose = seg.delta_from_params(x, pivot) @ T0
        r = seg.render_silhouette(model, pose, Kfull, (x1 - x0, y1 - y0)) > 127
        score = seg.iou(r & w_full, m_full & w_full)
        if score > best_iou:
            best_iou, best_pose = score, pose
    return best_pose


def run_mask_baseline():
    import sys
    sys.path.insert(0, os.path.expanduser(args.seg_tools))
    import gt_segmentation_eval as seg
    os.makedirs(AUTO_MASK_DIR, exist_ok=True)
    auto, auto_info = final_auto_masks()
    if not auto:
        print("Mask baseline: no generated masks -- skipped")
        return None, auto_info, {}
    model = seg.ClevisModel(os.path.join(args.model_root, "joint.stl"))
    by_fid = {frames[i]["frame_id"]: i for i in auto}
    cand = sorted(auto)
    pick = cand if not 0 < args.seg_frames < len(cand) else \
        [cand[k] for k in np.unique(np.round(np.linspace(0, len(cand) - 1, args.seg_frames)).astype(int))]
    recs = [dict(frame_index=frames[i]["frame_id"], right_rect_image=os.path.abspath(frames[i]["img_path"]))
            for i in pick]
    os.makedirs(HAND_DIR, exist_ok=True)
    ann = seg.load_annotations(HAND_DIR)
    status = lambda r: ann["frames"].get(str(r["frame_index"]), {}).get("status", "pending")
    pending = [r["frame_index"] for r in recs if status(r) not in ("saved", "skipped")]
    if pending and not args.no_annotate:
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            print(f"Mask baseline: frames {pending} have no hand segmentation and there is no display -- "
                  f"rerun on a desktop (or with --reuse-fit) to draw them")
        else:
            print(f"Mask baseline: hand-segment the clevis in frames {pending} (left panel); the right panel shows "
                  f"the generated mask. Saved to {HAND_DIR}")
            seg.save_annotations(HAND_DIR, ann)
            label = "Generated mask (diff-render pipeline, final pass)" + \
                    (" -- rebuilt" if auto_info.get("reconstructed") else "")
            seg.Annotator(HAND_DIR, recs, ann, model,
                          reference=lambda r: (auto[by_fid[r["frame_index"]]], label)).run()
            ann = seg.load_annotations(HAND_DIR)
    done = [r for r in recs if status(r) == "saved"]
    if not done:
        print("Mask baseline: no hand-segmented frames yet -- not in the report")
        return None, auto_info, {}

    old = {r["frame_id"]: r for r in ((prev.get("mask_baseline") or {}).get("frames", []) if args.reuse_fit else [])}
    results, panels = [], {}
    for n, r in enumerate(done):
        fid = r["frame_index"]; i = by_fid[fid]; f = frames[i]
        hand = cv2.imread(os.path.join(HAND_DIR, ann["frames"][str(fid)]["mask"]), cv2.IMREAD_GRAYSCALE) > 127
        a = auto[i] > 0
        valid = valid_map(f, X_dr)
        v = valid > 0.5
        key = "|".join([_md5(hand), _md5(a), _md5(v), str(args.seg_fit_restarts), str(args.seg_seed)])
        t0 = time.time()
        if fid in old and old[fid].get("key") == key:
            res = old[fid]
        else:
            T_dr, T_gt = (to_m(frame_poses_np(X, f)[0]) for X in (X_dr, X_gt))
            win = seg.crop_window([a, hand], (img_W, img_H), margin_frac=0.5)
            fits = {}
            for name, m in (("hand", hand), ("auto", a)):
                fits[name] = fit_pose_to_mask_weighted(seg, model, f["K3"], m, valid, T_dr, win,
                                                       args.seg_fit_restarts,
                                                       np.random.default_rng(args.seg_seed + fid))
            res = dict(frame_id=fid, key=key, iou=seg.iou(a, hand), iou_dr_region=seg.iou(a & v, hand & v),
                       **seg.pose_error(fits["hand"], fits["auto"]),
                       hand_fit_vs_gt=seg.pose_error(T_gt, fits["hand"]),
                       hand_fit_vs_diff_render=seg.pose_error(T_dr, fits["hand"]),
                       pose_hand_fit=fits["hand"].tolist(), pose_auto_fit=fits["auto"].tolist())
        results.append(res)
        fm = {k: seg.render_silhouette(model, np.array(res[f"pose_{k}_fit"]), f["K3"], (img_W, img_H)) > 127
              for k in ("hand", "auto")}
        x0, y0, x1, y1 = seg.crop_window([hand, a, fm["hand"], fm["auto"]], (img_W, img_H))
        c = (slice(y0, y1), slice(x0, x1))
        rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)[c]
        p_iou = seg.iou_panel(rgb, hand[c], a[c])
        p_iou[~v[c]] = (0.35 * p_iou[~v[c]]).astype(np.uint8)          # ignored by the DR loss: darkened
        p_out = seg.outline_panel(rgb, [(a[c], (255, 0, 0), 2), (fm["auto"][c], (255, 0, 255), 2),
                                        (fm["hand"][c], (0, 255, 255), 2), (hand[c], (0, 200, 0), 1)])
        panels[fid] = (p_iou, p_out)
        print(f"  [{n + 1}/{len(done)}] frame {fid}: IoU {res['iou']:.3f} (DR region {res['iou_dr_region']:.3f})  "
              f"rot {res['rot_err_deg']:.2f} deg  trans {res['trans_err_mm']:.2f} mm (in-plane "
              f"{res['trans_err_inplane_mm']:.2f}, depth {res['trans_err_depth_mm']:.2f})  ({time.time() - t0:.0f}s)")
    meta = dict(hand_segmentation_dir=HAND_DIR, auto_mask_dir=AUTO_MASK_DIR, n_candidates=len(cand),
                fit_restarts=args.seg_fit_restarts,
                seed=args.seg_seed, start_pose="diff-render estimate", frames=results,
                skipped=[r["frame_index"] for r in recs if status(r) == "skipped"],
                pending=[r["frame_index"] for r in recs if status(r) == "pending"])
    return meta, auto_info, panels


def _stats_rows(results, cols):
    rows = [[str(r["frame_id"])] + [f"{g(r):.3f}" for _, g in cols] for r in results]
    for name, fn in (("mean", np.mean), ("median", np.median), ("std", np.std), ("max", np.max)):
        rows.append([name] + [f"{fn([g(r) for r in results]):.3f}" for _, g in cols])
    return rows


def _table(ax, rows, labels, fs=8):
    tab = ax.table(cellText=rows, colLabels=labels, cellLoc="center", bbox=[0, 0, 1, 1])
    _style_table(tab, len(labels), fs)
    for k, row in enumerate(rows, start=1):
        if not row[0].isdigit():
            for j in range(len(row)):
                tab[k, j].set_facecolor("#e8e8e8"); tab[k, j].set_text_props(weight="bold")


def mask_baseline_pages(pdf):
    from matplotlib.patches import Patch
    res = mask_baseline["frames"]
    src = ("REBUILT -- " + auto_mask_info["note"]) if auto_mask_info.get("reconstructed") else \
        "saved by the fit itself (final pass)"
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{DATASET}: mask baseline -- diff-render's generated masks vs hand-drawn clevis segmentation "
                 f"(gt_segmentation_eval.py metrics)", fontsize=11, y=0.975)
    fig.text(0.04, 0.915, "\n".join(textwrap.wrap(
        f"Generated masks: the masks of the diff-render fit in this report ({len(res)} training frames with a hand "
        f"mask), {src}. Hand masks: {HAND_DIR} (clevis only, drawn in gt_segmentation_eval.py's window). "
        "IoU = generated vs hand mask over the whole image; 'DR region' = only the pixels the diff-render loss "
        "uses (jaw / shaft / ring-prong zones excluded, as in the fit). Pose errors: clevis pose best fitting the "
        "generated mask vs best fitting the hand mask (gt_segmentation_eval.py's Nelder-Mead silhouette fit, "
        f"{mask_baseline['fit_restarts']} starts from the diff-render estimate, DR region only), camera frame at "
        "the pitch pin; depth = camera z, weakly observable from a silhouette.", 190)), fontsize=7.3, va="top")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.72]); ax.axis("off")
    cols = [("IoU", lambda r: r["iou"]), ("IoU\n(DR region)", lambda r: r["iou_dr_region"]),
            ("rot err\n(deg)", lambda r: r["rot_err_deg"]), ("trans err\n(mm)", lambda r: r["trans_err_mm"]),
            ("in-plane\n(mm)", lambda r: r["trans_err_inplane_mm"]), ("depth\n(mm)", lambda r: r["trans_err_depth_mm"])]
    _table(ax, _stats_rows(res, cols), ["frame"] + [c for c, _ in cols])
    if mask_baseline["skipped"] or mask_baseline["pending"]:
        fig.text(0.04, 0.015, f"Not scored: skipped (clevis hidden) {mask_baseline['skipped']}, not yet drawn "
                 f"{mask_baseline['pending']}.", fontsize=7.5)
    pdf.savefig(fig); plt.close(fig)

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(f"{DATASET}: mask baseline -- what the hand segmentation says about the poses", fontsize=11, y=0.975)
    fig.text(0.04, 0.915, "\n".join(textwrap.wrap(
        "Context, not a mask metric: the clevis pose best fitting the HAND mask (an image-only reference, as "
        "gt_segmentation_eval.py uses it) compared with the ground-truth pose and with the diff-render pose of "
        "each frame. Same fit and error definitions as the previous page.", 190)), fontsize=7.3, va="top")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.77]); ax.axis("off")
    cols = []
    for key, lab in (("hand_fit_vs_gt", "vs GT"), ("hand_fit_vs_diff_render", "vs DR")):
        cols += [(f"{lab}\nrot (deg)", lambda r, k=key: r[k]["rot_err_deg"]),
                 (f"{lab}\ntrans (mm)", lambda r, k=key: r[k]["trans_err_mm"]),
                 (f"{lab}\nin-plane (mm)", lambda r, k=key: r[k]["trans_err_inplane_mm"]),
                 (f"{lab}\ndepth (mm)", lambda r, k=key: r[k]["trans_err_depth_mm"])]
    _table(ax, _stats_rows(res, cols), ["frame"] + [c for c, _ in cols])
    pdf.savefig(fig); plt.close(fig)

    for r in res:
        p_iou, p_out = mask_baseline_panels[r["frame_id"]]
        fig, axes = plt.subplots(1, 2, figsize=(11, 8.5))
        dx, dy, dz = r["dxyz_mm"]
        fig.suptitle(f"mask baseline, frame {r['frame_id']}{' (generated mask rebuilt)' if auto_mask_info.get('reconstructed') else ''}"
                     f":  IoU {r['iou']:.3f}  (DR region {r['iou_dr_region']:.3f})\nfit to generated vs fit to hand "
                     f"mask: rot {r['rot_err_deg']:.2f} deg   trans {r['trans_err_mm']:.2f} mm (x {dx:+.2f}, "
                     f"y {dy:+.2f}, z {dz:+.2f}; in-plane {r['trans_err_inplane_mm']:.2f})", fontsize=10)
        axes[0].imshow(p_iou)
        axes[0].set_title("IoU: generated mask over the hand mask (darkened = ignored by the DR loss)", fontsize=8.5)
        axes[0].legend(handles=[Patch(color=(1, 0.86, 0), label="overlap"),
                                Patch(color=(1, 0, 0), label="generated only"),
                                Patch(color=(0, 0.78, 0), label="hand only")], loc="lower right", fontsize=7)
        axes[1].imshow(p_out)
        axes[1].set_title("outlines: masks and the clevis poses best fitting each", fontsize=8.5)
        axes[1].legend(handles=[Patch(color=(1, 0, 0), label="generated mask"),
                                Patch(color=(1, 0, 1), label="fit to generated"),
                                Patch(color=(0, 1, 1), label="fit to hand"),
                                Patch(color=(0, 0.78, 0), label="hand mask")], loc="lower right", fontsize=7)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig); plt.close(fig)


mask_baseline, auto_mask_info, mask_baseline_panels = (None, None, {}) if args.no_mask_baseline \
    else run_mask_baseline()


def write_mask_vs_segmentation_pdf(pdf_path):
    """Standalone report in gt_segmentation_eval.py's segmentation_eval.pdf format (its write_pdf layout: portrait,
    bold title, monospace meta block, frame table with mean / median / std / max rows, then one page per frame with
    the IoU panel above the outline panel) -- here generated mask vs hand mask instead of GT vs hand mask."""
    from matplotlib.patches import Patch
    res = mask_baseline["frames"]
    meta_lines = [
        f"dataset:       {os.path.abspath(ROOT)}",
        f"evaluated:     {time.ctime()}",
        "generated:     diff-render pipeline final-pass masks"
        + (" -- REBUILT (the run predates saving its masks)" if auto_mask_info.get("reconstructed") else ""),
        f"               fit of {fit_generated if args.reuse_fit else 'this run'}"
        + (f"; same mask steps, last pass seeded from its final calibration (its pass-{args.mask_passes - 1}"
           if auto_mask_info.get("reconstructed") else ""),
    ] + (["               estimate was not saved)"] if auto_mask_info.get("reconstructed") else []) + [
        f"               {AUTO_MASK_DIR}",
        f"hand masks:    {HAND_DIR} (clevis only)",
        f"frames:        {[r['frame_id'] for r in res]} ({len(res)} of the {mask_baseline['n_candidates']} "
        f"training frames, evenly spaced)",
        f"model:         {os.path.join(args.model_root, 'joint.stl')} (clevis only)",
        "IoU:           generated mask vs drawn mask, whole image; 'DR rgn' = only the pixels the diff-render",
        "               loss uses (jaw / shaft / ring-prong zones excluded -- the generated mask covers them)",
        "errors:        pose best fitting the generated mask vs pose best fitting the drawn mask (camera frame,",
        f"               at the pitch pin; gt_segmentation_eval.py's fit, {mask_baseline['fit_restarts']} starts, "
        "DR region only);",
        "               depth = camera z, only weakly observable from a silhouette",
    ]
    keys = [("iou", "IoU"), ("iou_dr_region", "IoU (DR rgn)"), ("rot_err_deg", "rot err [deg]"),
            ("trans_err_mm", "trans err [mm]"), ("trans_err_inplane_mm", "in-plane [mm]"),
            ("trans_err_depth_mm", "depth [mm]")]
    rows = [[str(r["frame_id"])] + [f"{r[k]:.3f}" for k, _ in keys] for r in res]
    for name, fn in (("mean", np.mean), ("median", np.median), ("std", np.std), ("max", np.max)):
        rows.append([name] + [f"{fn([r[k] for r in res]):.3f}" for k, _ in keys])

    with PdfPages(pdf_path) as pdf:
        per_page = 30
        for start in range(0, len(rows), per_page):
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.06, 0.96, "Generated masks vs hand segmentation", fontsize=15, weight="bold")
            fig.text(0.06, 0.94 - 0.016 * len(meta_lines), "\n".join(meta_lines), fontsize=7.5, va="bottom",
                     family="monospace")
            ax = fig.add_axes([0.06, 0.06, 0.88, 0.80 - 0.016 * len(meta_lines)])
            ax.axis("off")
            chunk = rows[start:start + per_page]
            table = ax.table(cellText=chunk, colLabels=["frame"] + [label for _, label in keys],
                             loc="upper center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8.5)
            table.scale(1, 1.3)
            for i, row in enumerate(chunk, start=1):
                if not row[0].isdigit():
                    for j in range(len(row)):
                        table[i, j].set_facecolor("#e8e8e8")
            pdf.savefig(fig)
            plt.close(fig)

        for r in res:
            panel_iou, panel_err = mask_baseline_panels[r["frame_id"]]
            fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
            dx, dy, dz = r["dxyz_mm"]
            fig.suptitle(f"frame {r['frame_id']}  ({os.path.basename(frames[mask_frame_idx[r['frame_id']]]['img_path'])})"
                         f"   IoU {r['iou']:.3f}  (DR region {r['iou_dr_region']:.3f})"
                         f"\nrot err {r['rot_err_deg']:.2f} deg   trans err {r['trans_err_mm']:.2f} mm "
                         f"(x {dx:+.2f}, y {dy:+.2f}, z {dz:+.2f} mm; in-plane {r['trans_err_inplane_mm']:.2f})",
                         fontsize=10)
            axes[0].imshow(panel_iou)
            axes[0].set_title("IoU: generated mask over the segmentation (darkened = ignored by the DR loss)",
                              fontsize=9)
            axes[0].legend(handles=[Patch(color=(1, 0.86, 0), label="overlap (generated and drawn)"),
                                    Patch(color=(1, 0, 0), label="generated only"),
                                    Patch(color=(0, 0.78, 0), label="drawn only")], loc="lower right", fontsize=7)
            axes[1].imshow(panel_err)
            axes[1].set_title("Errors: pose best fitting the generated mask vs pose best fitting the segmentation",
                              fontsize=9)
            axes[1].legend(handles=[Patch(color=(1, 0, 0), label="generated mask"),
                                    Patch(color=(1, 0, 1), label="fit to generated"),
                                    Patch(color=(0, 1, 1), label="fit to drawn"),
                                    Patch(color=(0, 0.78, 0), label="drawn")], loc="lower right", fontsize=7)
            for ax in axes:
                ax.axis("off")
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Mask vs segmentation report → {pdf_path}")


if mask_baseline:
    mask_frame_idx = {frames[i]["frame_id"]: i for i in ALL}
    write_mask_vs_segmentation_pdf(os.path.join(args.eval_dir, SEG_PDF_NAME))

cal_breakdown = error_breakdown(X_dr, CAL)
with PdfPages(pdf_path) as pdf:
    calibration_frame_pages(pdf)
    metrics_table_page(pdf)
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.02, 0.98, "\n".join(lines), family="monospace", fontsize=6.6, va="top", ha="left")
    pdf.savefig(fig); plt.close(fig)
    for row in rows:
        i = next(k for k in ALL if frames[k]["frame_id"] == row["frame_id"])
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        for ax, (name, Xc) in zip(axes, METHODS):
            m = row[name]
            ax.imshow(make_side_by_side(i, Xc, f"GT (marker + 3D alignment)  frame={row['frame_id']}", name))
            ax.set_title(f"{name}:  translation error {m['trans_err_mm']:.2f} mm   rotation error "
                         f"{m['rot_err_deg']:.2f} deg   IoU {m['iou_vs_gt']:.3f}", fontsize=10)
            ax.axis("off")
        fig.suptitle(f"frame_{row['frame_id']}   (held out: not used by the diff-render fit)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        pdf.savefig(fig); plt.close(fig)
    if mask_baseline:
        mask_baseline_pages(pdf)

result = dict(
    generated=datetime.datetime.now().isoformat(timespec="seconds"), dataset=os.path.abspath(ROOT),
    dr_inputs=["right_rect + left_rect images", "camera_matrix_right_rect", "rect_T_cam_raw",
               "stereo_rectification P matrices", "joint.stl", "base_T_gripper", "base_T_shaft_end"],
    frames=[f["frame_id"] for f in frames], train_frames=[frames[i]["frame_id"] for i in TRAIN],
    test_frames=[frames[i]["frame_id"] for i in TEST], tip_found=[frames[i]["frame_id"] for i in USE],
    inliers=[frames[i]["frame_id"] for i in inliers], mask_passes=args.mask_passes, fit_runtime_s=fit_runtime,
    cam_T_base_ground_truth=to_m(X_gt).tolist(), cam_T_base_marker_only=to_m(X_marker).tolist(),
    cam_T_base_pnp_init=to_m(X_init).tolist(), cam_T_base_diff_render=to_m(X_dr).tolist(),
    pnp_init_vs_gt_on_test=dict(mean_clevis_displacement_mm=d0, rotation_err_deg=r0),
    overall_on_test=overall, test_frame_results=rows,
    calibration_frame=dict(frame_id=frames[CAL]["frame_id"], status=CAL_STATUS, model_diameter_mm=SR_DIAM,
                           surgripe_metrics=cal_metrics, error_breakdown_diff_render=cal_breakdown),
    runtime=runtime, auto_masks=auto_mask_info, mask_baseline=mask_baseline,
)
if args.reuse_fit:
    result["generated_fit"] = fit_generated
with open(json_path, "w") as fh:
    json.dump(result, fh, indent=2)
print(f"Report → {pdf_path}\nResult → {json_path}")
