import sys
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
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
DATASET_ROOT   = "./SIMPLELND_dataset"
CAMERA_DIR     = os.path.join(DATASET_ROOT, "camera_0", "000001")
SCENE_CAM_JSON = os.path.join(CAMERA_DIR, "scene_camera.json")
RGB_DIR        = os.path.join(CAMERA_DIR, "rgb")
MESH_PATH      = "../surgical_robotics_challenge/ADF/PSMs/LND_420006/high_res/tool pitch link.OBJ"          # <-- point this to your .obj file
OUTPUT_GIF     = "./camera0_optimization.gif"
NUM_ITERS      = 200
LR             = 0.05
LOSS_THRESHOLD = 200
SAVE_EVERY     = 10
# ─────────────────────────────────────────────

# ── Device ────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


# ── Load intrinsics from scene_camera.json ────
# Expected BOP format:
# { "0": { "cam_K": [fx,0,cx, 0,fy,cy, 0,0,1], "depth_scale": ..., ... }, ... }
with open(SCENE_CAM_JSON, "r") as f:
    scene_cam = json.load(f)

# Use the first entry (frame 0 / key "0" or "1")
first_key = sorted(scene_cam.keys(), key=lambda x: int(x))[0]
cam_data  = scene_cam[first_key]
K_flat    = cam_data["cam_K"]          # 9 values, row-major
fx, fy    = K_flat[0], K_flat[4]
cx, cy    = K_flat[2], K_flat[5]
print(f"Loaded intrinsics — fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


# ── Load reference RGB image ───────────────────
# Pick the first image found in rgb/
rgb_files = sorted([
    f for f in os.listdir(RGB_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])
assert len(rgb_files) > 0, f"No images found in {RGB_DIR}"
ref_image_path = os.path.join(RGB_DIR, rgb_files[0])
print(f"Reference image: {ref_image_path}")

ref_pil   = Image.open(ref_image_path).convert("RGB")
img_W, img_H = ref_pil.size
print(f"Image size: {img_W}x{img_H}")

# Resize to a square power-of-2 for the renderer (keeps things fast)
RENDER_SIZE = 256
ref_pil_resized = ref_pil.resize((RENDER_SIZE, RENDER_SIZE), Image.BILINEAR)
# Convert PIL -> torch directly without touching the numpy bridge.
# torch.frombuffer warns about non-writable buffers, so we go via a list copy.
_pil_bytes = list(ref_pil_resized.tobytes())
ref_tensor = torch.tensor(_pil_bytes, dtype=torch.float32).reshape(RENDER_SIZE, RENDER_SIZE, 3) / 255.0
# Keep a numpy copy only for matplotlib display (PIL->numpy is safe, no torch bridge)
ref_np = np.array(ref_pil_resized).astype(np.float32) / 255.0


# ── Build PyTorch3D camera from intrinsics ─────
# Convert OpenCV K to PyTorch3D NDC focal / principal point.
# PyTorch3D PerspectiveCameras expect focal_length and principal_point
# in NDC space where image goes from -1 to 1.
# focal_ndc  = 2 * f / image_size
# pp_ndc     = (image_size/2 - c) / (image_size/2)   [x then y, note sign]
fx_ndc = 2.0 * fx / img_W
fy_ndc = 2.0 * fy / img_H
px_ndc = (img_W / 2.0 - cx) / (img_W / 2.0)
py_ndc = (img_H / 2.0 - cy) / (img_H / 2.0)

print(f"NDC focal: ({fx_ndc:.4f}, {fy_ndc:.4f})  principal: ({px_ndc:.4f}, {py_ndc:.4f})")


# ── Load mesh ──────────────────────────────────
# PyTorch3D's load_obj internally calls torch.from_numpy() while parsing .mtl
# material files, crashing under NumPy 2.x. We strip mtllib/usemtl lines from
# a temp copy of the .obj so PyTorch3D never enters the material loader.
import tempfile, re

def load_obj_no_mtl(path):
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
faces = faces_idx.verts_idx

verts_rgb = torch.ones_like(verts)[None]          # white vertices
textures  = TexturesVertex(verts_features=verts_rgb.to(device))

mesh = Meshes(
    verts=[verts.to(device)],
    faces=[faces.to(device)],
    textures=textures,
)
print(f"Mesh loaded: {verts.shape[0]} verts, {faces.shape[0]} faces")


# ── Build renderers ────────────────────────────
def make_cameras(R=None, T=None):
    """Create a PerspectiveCameras object using the loaded intrinsics.
    Always supplies valid R and T — passing None causes AttributeError deep
    inside cameras.py when the renderer inspects tensor shapes at init time.
    """
    if R is None:
        R = torch.eye(3, device=device).unsqueeze(0)   # (1, 3, 3) identity
    if T is None:
        T = torch.zeros(1, 3, device=device)            # (1, 3) no translation
    return PerspectiveCameras(
        focal_length=((fx_ndc, fy_ndc),),
        principal_point=((px_ndc, py_ndc),),
        R=R,
        T=T,
        device=device,
    )

# -- Silhouette renderer (soft, differentiable)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_sil = RasterizationSettings(
    image_size=RENDER_SIZE,
    blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
    faces_per_pixel=100,
)
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=make_cameras(),
        raster_settings=raster_sil,
    ),
    shader=SoftSilhouetteShader(blend_params=blend_params),
)

# -- Phong renderer (hard, for visualisation)
raster_phong = RasterizationSettings(
    image_size=RENDER_SIZE,
    blur_radius=0.0,
    faces_per_pixel=1,
)
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=make_cameras(),
        raster_settings=raster_phong,
    ),
    shader=HardPhongShader(device=device, cameras=make_cameras(), lights=lights),
)


# ── Render initial reference view ─────────────
distance, elevation, azimuth = 3.0, 50.0, 0.0
R_ref, T_ref = look_at_view_transform(distance, elevation, azimuth, device=device)

silhouette_init = silhouette_renderer(meshes_world=mesh, R=R_ref, T=T_ref)
phong_init      = phong_renderer(meshes_world=mesh, R=R_ref, T=T_ref)

sil_np   = np.from_dlpack(silhouette_init.detach().cpu())
phong_np = np.from_dlpack(phong_init.detach().cpu())

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(sil_np.squeeze()[..., 3]);  plt.title("Rendered silhouette"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(phong_np.squeeze());         plt.title("Rendered Phong");     plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(ref_np);                     plt.title("RGB reference");      plt.axis("off")
plt.tight_layout()
plt.savefig("initial_views.png", dpi=100)
plt.show()
print("Saved initial_views.png")


# ── Model ──────────────────────────────────────
class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref_tensor):
        super().__init__()
        self.meshes   = meshes
        self.device   = meshes.device
        self.renderer = renderer

        # image_ref_tensor is already a float32 torch tensor (H, W, 3) — no numpy bridge.
        # Build silhouette mask: any pixel that isn't white (1,1,1) is part of the object.
        mask = (image_ref_tensor.max(-1).values != 1.0).to(torch.float32)  # (H, W)
        self.register_buffer("image_ref", mask)

        # Optimisable camera position — initialise slightly off from ground truth
        # so the optimiser has something to do.
        self.camera_position = nn.Parameter(
            torch.tensor([3.0, 6.9, 2.5], dtype=torch.float32, device=self.device)
        )

    def forward(self):
        R = look_at_rotation(self.camera_position[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        loss  = torch.sum((image[..., 3] - self.image_ref) ** 2)
        return loss, image


# ── Optimisation loop ──────────────────────────
writer = imageio.get_writer(OUTPUT_GIF, mode="I", duration=0.3)

model     = Model(meshes=mesh, renderer=silhouette_renderer, image_ref_tensor=ref_tensor).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Show starting position
_, image_start = model()
start_np = np.from_dlpack(image_start.detach().squeeze().cpu())
ref_buf  = np.from_dlpack(model.image_ref.detach().cpu())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(start_np[..., 3]); plt.title("Starting silhouette"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(ref_buf.squeeze()); plt.title("Reference silhouette"); plt.axis("off")
plt.tight_layout()
plt.savefig("start_vs_reference.png", dpi=100)
plt.show()

loop = tqdm(range(NUM_ITERS))
for i in loop:
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    optimizer.step()
    loop.set_description("Optimising (loss %.4f)" % loss.item())

    if loss.item() < LOSS_THRESHOLD:
        print(f"Converged at iteration {i} with loss {loss.item():.4f}")
        break

    if i % SAVE_EVERY == 0:
        R = look_at_rotation(model.camera_position[None, :], device=model.device)
        T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]
        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)

        img_cpu   = image[0, ..., :3].detach().squeeze().cpu()
        img_np    = np.from_dlpack(img_cpu)
        img_ubyte = img_as_ubyte(img_np)
        writer.append_data(img_ubyte)

        plt.figure()
        plt.imshow(img_ubyte)
        plt.title("iter: %d, loss: %.2f" % (i, loss.item()))
        plt.axis("off")

writer.close()
print(f"GIF saved to {OUTPUT_GIF}")

# ── Final result ───────────────────────────────
print(f"\nOptimised camera position: {np.from_dlpack(model.camera_position.data.cpu())}")

R_final = look_at_rotation(model.camera_position[None, :], device=model.device)
T_final = -torch.bmm(R_final.transpose(1, 2), model.camera_position[None, :, None])[:, :, 0]
final_render = phong_renderer(meshes_world=model.meshes.clone(), R=R_final, T=T_final)
final_np     = np.from_dlpack(final_render[0, ..., :3].detach().cpu())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(final_np);  plt.title("Final render"); plt.axis("off")
plt.subplot(1, 2, 2); plt.imshow(ref_np);    plt.title("RGB reference"); plt.axis("off")
plt.tight_layout()
plt.savefig("final_vs_reference.png", dpi=100)
plt.show()
print("Saved final_vs_reference.png")