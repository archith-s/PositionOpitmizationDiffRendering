"""
hand_eye_multi_obj.py — run hand_eye_diff_render_test.py's differentiable-
rendering pose optimizer with jaw meshes from ../meshes/*.obj instead of the
SurgRIPE joint.stl clevis.

This used to be a separate copy of the optimizer, which had fallen behind
hand_eye_diff_render_test.py and was wrong in several ways:
  * MESH_TO_GRIPPER assumed the jaw OBJs have their pivot pin along mesh X;
    it is along mesh Z (checked on the real LND006 / PGForceps093 frames), so
    every overlay was rotated.
  * It carried the pre-2026-10-17 optimizer (face-doubled 3000-face proxy,
    fpp=4, no edge-profile table, lr_t=0.5), which got ~6 mm / 29 deg on
    LND006 where the current one gets ~0.3 mm / 0.8 deg.
  * No vertex welding or per-file fixes, so tip_093_* (a triangle soup with
    its pin hole at (0,20,0) and jaws along -Y) decimated into a proxy full
    of holes and was posed wrongly.
  * Its default dataset (gt_hand_eye_dataset5) no longer exists, and it wrote
    an fgmask PNG per frame into ../debug_renders.

hand_eye_diff_render_test.py --obj now does all of this correctly, so this
script only keeps the old command line (default --obj 006, first frame only,
--list-objs, eval_summary_hand_eye_multi_obj_<objs> output name) and hands
off to it. Any other option is passed straight through, e.g. --batch, --seed,
--all, --model-root.
"""

import argparse
import glob
import os
import subprocess
import sys

HERE          = os.path.dirname(os.path.abspath(__file__))
MESHES_DIR    = os.path.join(HERE, "..", "meshes")
TARGET        = os.path.join(HERE, "hand_eye_diff_render_test.py")
HAND_EYE_ROOT = "/home/aseshad4/hand_eye_collection_scripts/LND006"
OUTPUT_DIR    = os.path.join(HERE, "..", "Eval_Results_DiffRender_HandEye")

parser = argparse.ArgumentParser(description="Jaw-OBJ run of hand_eye_diff_render_test.py "
                                             "(unrecognised options are passed through to it)")
parser.add_argument("--obj", nargs="+", default=["006"],
                    help="Tool id(s) (e.g. 006 -> tip_006_*.obj) and/or OBJ file name(s) in ../meshes. "
                         "Default: 006 (both LND jaws).")
parser.add_argument("--list-objs", action="store_true", help="List the OBJ files in ../meshes and exit")
parser.add_argument("--frames", type=int, default=1, help="Run only the first N frames (default: 1)")
parser.add_argument("--hand-eye-root", type=str, default=HAND_EYE_ROOT,
                    help="Directory holding right_rect-*.png + gripper_ground_truth.json")
parser.add_argument("--output_summary", type=str, default=None,
                    help="Evaluator .npy path (default: ../Eval_Results_DiffRender_HandEye/"
                         "eval_summary_hand_eye_multi_obj_<objs>.npy; the PDF goes next to it)")
args, passthrough = parser.parse_known_args()

if args.list_objs:
    print("\n".join(sorted(os.path.basename(p) for p in glob.glob(os.path.join(MESHES_DIR, "*.obj")))))
    sys.exit(0)

if args.output_summary is None:
    args.output_summary = os.path.join(OUTPUT_DIR, f"eval_summary_hand_eye_multi_obj_{'_'.join(args.obj)}.npy")

cmd = [sys.executable, TARGET, "--obj", *args.obj, "--hand-eye-root", args.hand_eye_root,
       "--output_summary", args.output_summary]
if "--all" not in passthrough:
    cmd += ["--frames", str(args.frames)]
cmd += passthrough
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd, cwd=HERE))
