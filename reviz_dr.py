"""
Re-render Diff_Render_CSV_Visuals images from saved CSV poses.
Reads pose_estimation_results.csv and re-draws every side-by-side image
with the current draw_pose_overlay settings (L=50mm axes) without
re-running the optimization.

Run with:
  conda run -n diff_render python reviz_dr.py
"""
import os, re, csv, json, tempfile
import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, RasterizationSettings,
    SoftSilhouetteShader, BlendParams, PerspectiveCameras,
)
from pytorch3d.structures import Meshes

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(CODE_DIR, '../Diff_Render_CSV_Visuals/pose_estimation_results.csv')
VIZ_DIR    = os.path.join(CODE_DIR, '../Diff_Render_CSV_Visuals')
RGB_DIR    = os.path.join(CODE_DIR, '../SIMPLELND_data/camera_0/000001/rgb')
SCENE_CAM  = os.path.join(CODE_DIR, '../SIMPLELND_data/camera_0/000001/scene_camera.json')
MESH_PATH  = os.path.join(CODE_DIR, '../surgical_robotics_challenge/ADF/PSMs/'
                                    'LND_420006/high_res/tool pitch link.OBJ')
DEVICE     = 'cuda'
AXIS_L     = 50.0   # mm — matches FP's 50mm (scale=0.05 in metres)
AXIS_W     = 4      # line width pixels
SIL_SIGMA  = 0.005

_FLIP = np.diag([-1., -1., 1.])


# ── Mesh load (mirrors optimize_modelv2.py) ───────────────────────────────────
def _load_mesh():
    with open(MESH_PATH) as fh:
        lines = fh.readlines()
    clean = [l for l in lines if not re.match(r"\s*(mtllib|usemtl)\b", l)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.obj', delete=False) as tmp:
        tmp.writelines(clean); tmp_path = tmp.name
    verts, faces_idx, _ = load_obj(tmp_path)
    os.remove(tmp_path)
    faces = faces_idx.verts_idx
    if float(verts.abs().max()) < 1.0:
        verts = verts * 1000.0
    v = verts.to(DEVICE)
    f = faces.to(DEVICE)
    return Meshes(verts=[v], faces=[f])


# ── Camera / renderer helpers ─────────────────────────────────────────────────
# Screen-space (in_ndc=False) cameras with an explicit image_size, NOT
# PyTorch3D's default NDC convention -- NDC silently mishandles non-square
# aspect ratios, which produced a confirmed, measured pixel offset against a
# manual-rasterizer render of identical pose data (see sim_iter_images.py,
# fixed 2026-07-10).
def _make_cam(R_np, T_np, cam_K, W, H):
    R_pt3d = (R_np.T @ _FLIP).astype(np.float32).tolist()
    T_pt3d = (_FLIP @ T_np).astype(np.float32).tolist()
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    return PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        R=torch.tensor([R_pt3d], dtype=torch.float32, device=DEVICE),
        T=torch.tensor([T_pt3d], dtype=torch.float32, device=DEVICE),
        in_ndc=False, image_size=((H, W),), device=DEVICE,
    )


def _make_renderer(cam_K, W, H):
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    blur = float(np.log(1.0 / SIL_SIGMA - 1.0)) * SIL_SIGMA
    raster = RasterizationSettings(
        image_size=(H, W), blur_radius=blur,
        faces_per_pixel=100, bin_size=0,
    )
    dummy_cam = PerspectiveCameras(
        focal_length=((fx, fy),), principal_point=((cx, cy),),
        R=torch.eye(3, device=DEVICE).unsqueeze(0),
        T=torch.zeros(1, 3, device=DEVICE),
        in_ndc=False, image_size=((H, W),), device=DEVICE,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=dummy_cam, raster_settings=raster),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=SIL_SIGMA, gamma=SIL_SIGMA)),
    )


# ── Overlay function ──────────────────────────────────────────────────────────
def draw_pose_overlay(rgb_pil, R_bop, t_bop, cam_K, mesh, renderer, color_sil):
    W, H = rgb_pil.size
    cam = _make_cam(R_bop, t_bop, cam_K, W, H)
    with torch.no_grad():
        alpha = np.from_dlpack(
            renderer(meshes_world=mesh.clone(), cameras=cam)[0, ..., 3].detach().cpu()
        )
    overlay = np.zeros((*alpha.shape, 3), dtype=np.float32)
    overlay[..., 0], overlay[..., 1], overlay[..., 2] = color_sil
    blended = np.clip(
        0.5 * alpha[..., None] * overlay + np.array(rgb_pil, dtype=np.float32),
        0, 255,
    ).astype(np.uint8)
    result = Image.fromarray(blended)

    draw = ImageDraw.Draw(result)
    fx, fy, cx, cy = cam_K[0], cam_K[4], cam_K[2], cam_K[5]
    def proj(p):
        pc = R_bop @ np.array(p) + t_bop
        if pc[2] <= 0:
            return None
        return (int(round(fx * pc[0] / pc[2] + cx)),
                int(round(fy * pc[1] / pc[2] + cy)))
    cen = proj([0, 0, 0])
    for tip, col in zip(
        [proj([AXIS_L, 0, 0]), proj([0, AXIS_L, 0]), proj([0, 0, AXIS_L])],
        [(220, 50, 50), (50, 220, 50), (50, 50, 220)],
    ):
        if cen and tip:
            draw.line([cen, tip], fill=col, width=AXIS_W)
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading mesh...")
    mesh = _load_mesh()

    print("Loading scene camera K...")
    with open(SCENE_CAM) as f:
        scene_cam = json.load(f)

    print("Reading CSV...")
    with open(CSV_PATH, newline='') as f:
        rows = list(csv.DictReader(f))

    # Build renderer from first frame's K (same for all frames in this dataset)
    first_K = scene_cam[rows[0]['frame_id']]['cam_K']
    sample  = Image.open(os.path.join(RGB_DIR, rows[0]['rgb_file']))
    img_W, img_H = sample.size
    renderer = _make_renderer(first_K, img_W, img_H)
    print(f"Renderer ready — image {img_W}×{img_H}, axis={AXIS_L}mm")

    LH = 28
    for row in rows:
        frame_id = int(row['frame_id'])
        ob_id    = int(row['obj_id'])
        rgb_file = row['rgb_file']
        cam_K    = scene_cam[str(frame_id)]['cam_K']

        rgb_pil = Image.open(os.path.join(RGB_DIR, rgb_file)).convert('RGB')

        def _pose(prefix):
            R = np.array([[float(row[f'{prefix}R_{r}{c}']) for c in range(3)]
                          for r in range(3)])
            t = np.array([float(row[f'{prefix}tx_mm']),
                          float(row[f'{prefix}ty_mm']),
                          float(row[f'{prefix}tz_mm'])])
            return R, t

        R_gt,  t_gt  = _pose('gt_')
        R_est, t_est = _pose('est_')

        gt_ov  = draw_pose_overlay(rgb_pil, R_gt,  t_gt,  cam_K, mesh, renderer, (0, 220, 0))
        est_ov = draw_pose_overlay(rgb_pil, R_est, t_est, cam_K, mesh, renderer, (220, 80, 0))

        combined = Image.new('RGB', (img_W * 2, img_H + LH), (40, 40, 40))
        combined.paste(gt_ov,  (0,     LH))
        combined.paste(est_ov, (img_W, LH))

        d = ImageDraw.Draw(combined)
        rot_err   = float(row['rotation_error_deg'])
        trans_err = float(row['translation_error_mm'])
        add_val   = float(row['ADD_mm'])
        d.text((8,         5), f"GT  frame={frame_id:04d}  obj={ob_id}",
               fill=(100, 255, 100))
        d.text((img_W + 8, 5),
               f"EST  rot={rot_err:.1f}°  trans={trans_err:.1f}mm  ADD={add_val:.1f}mm",
               fill=(255, 180, 80))

        out = os.path.join(VIZ_DIR, f"viz_frame{frame_id:04d}_obj{ob_id}.png")
        combined.save(out)
        print(f"  → {out}")

    print(f"Done. {len(rows)} images updated in {VIZ_DIR}/")
