"""
Builds table4_summary_real_dr.csv from eval_summary_real_dr.npy: first row is
the aggregate average across all evaluated frames (matching evaluate.py's
Evaluator.summarize() aggregate metrics exactly), followed by one row per
frame. Reusable as-is for the 5-frame run and the full 373-frame run.
"""
import numpy as np
import csv

NPY_PATH = "../Eval_Results_DiffRender_Real/eval_summary_real_dr.npy"
CSV_PATH = "../Eval_Results_DiffRender_Real/table4_summary_real_dr.csv"
DIAMETER = 16.242301839504098  # LND, from evaluate.py's model_diameter()

FIELDNAMES = [
    'Image', 'ADD (10% diameter)', 'Avg Acc (0-5 MM)',
    'Avg Acc (0-10 MM, not a paper metric)',
    'Translation Error (mm)', 'Rotation Error (degree)', 'ADD Distance (mm)',
]


def rot_error_deg(R_est, R_gt):
    trace = min(np.trace(R_est @ R_gt.T), 3.0)
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    return float(np.degrees(np.arccos(error_cos)))


def trans_error_mm(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


def main():
    data = np.load(NPY_PATH, allow_pickle=True).item()
    add_dist_list = data['ADD-distance list']
    pred_list = data['pred_list']
    gt_list = data['gt_list']
    thresh_10pct = DIAMETER * 0.1

    rows = [{
        'Image': f'AVERAGE (n={len(add_dist_list)})',
        'ADD (10% diameter)': f"{data['add']:.4f}",
        'Avg Acc (0-5 MM)': f"{data['avg_acc_0_5mm']:.4f}",
        'Avg Acc (0-10 MM, not a paper metric)': f"{data['avg_acc_0_10mm']:.4f}",
        'Translation Error (mm)': f"{data['trans_error']:.4f}",
        'Rotation Error (degree)': f"{data['rot_error']:.4f}",
        'ADD Distance (mm)': f"{data['ADD-distance']:.4f}",
    }]

    for i, (pred, gt, add_dist) in enumerate(zip(pred_list, gt_list, add_dist_list), start=1):
        R_pred, t_pred = pred[:, :3], pred[:, 3]
        R_gt, t_gt = gt[:, :3], gt[:, 3]
        rows.append({
            'Image': f'frame_{i:04d}',
            'ADD (10% diameter)': 1 if add_dist < thresh_10pct else 0,
            'Avg Acc (0-5 MM)': f"{np.mean([1 if add_dist <= t else 0 for t in [0,1,2,3,4,5]]):.4f}",
            'Avg Acc (0-10 MM, not a paper metric)': f"{np.mean([1 if add_dist <= t else 0 for t in range(0,11)]):.4f}",
            'Translation Error (mm)': f"{trans_error_mm(t_pred, t_gt):.4f}",
            'Rotation Error (degree)': f"{rot_error_deg(R_pred, R_gt):.4f}",
            'ADD Distance (mm)': f"{add_dist:.4f}",
        })

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows (1 average + {len(rows)-1} frames) to {CSV_PATH}")


if __name__ == "__main__":
    main()
