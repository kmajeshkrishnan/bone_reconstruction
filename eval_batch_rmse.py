
import argparse
import os
import torch
import numpy as np
import open3d as o3d
from scipy.interpolate import NearestNDInterpolator
import matplotlib.pyplot as plt
from models import build_model_from_cfg
from utils.config import cfg_from_yaml_file
from utils import misc
import json
import re
import glob

# ============================================================================
# Reusing Logic from infer_learned_length.py
# (Duplicated here to keep script self-contained as requested)
# ============================================================================

def load_ply_or_npy(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    if filename.endswith('.npy'):
        pc = np.load(filename)
    else:
        pcd = o3d.io.read_point_cloud(filename)
        pc = np.asarray(pcd.points, dtype=np.float32)
    return pc

def align_bone_to_x_axis(points):
    """Align principal axis to X-axis"""
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    
    cov = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    longest_axis = eigenvectors[:, 0].real.astype(np.float32)
    longest_axis /= (np.linalg.norm(longest_axis) + 1e-8)
    
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    rotation_axis = np.cross(longest_axis, x_axis)
    dot = np.dot(longest_axis, x_axis)
    rotation_angle = np.arccos(np.clip(dot, -1.0, 1.0))
    
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis /= np.linalg.norm(rotation_axis)
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ], dtype=np.float32)
        R = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * (K @ K)
    else:
        R = np.eye(3) if dot > 0 else np.diag([-1, 1, 1])

    aligned_points = points_centered @ R.T
    return aligned_points.astype(np.float32)

def ensure_polarity(points):
    """Enforce canonical orientation: Thicker end at Negative X"""
    x = points[:, 0]
    x_min, x_max = x.min(), x.max()
    range_x = x_max - x_min
    
    left_mask = x < (x_min + range_x * 0.1)
    right_mask = x > (x_max - range_x * 0.1)
    
    r = np.sqrt(points[:, 1]**2 + points[:, 2]**2)
    
    r_left = np.median(r[left_mask]) if np.sum(left_mask) > 0 else 0
    r_right = np.median(r[right_mask]) if np.sum(right_mask) > 0 else 0
    
    if r_right > r_left:
        # Flip
        points[:, 0] *= -1
        points[:, 1] *= -1 
    return points

def cut_from_thinner_end(points, cut_ratio=0.1):
    """
    Cut a portion from the thinner end of the bone.
    """
    if cut_ratio <= 0:
        return points, {'cut': False}
    
    x = points[:, 0]
    x_min, x_max = x.min(), x.max()
    range_x = x_max - x_min
    
    # Identify left and right ends (assumes rough alignment)
    left_mask = x < (x_min + range_x * 0.1)
    right_mask = x > (x_max - range_x * 0.1)
    
    r = np.sqrt(points[:, 1]**2 + points[:, 2]**2)
    
    r_left = np.median(r[left_mask]) if np.sum(left_mask) > 0 else 0
    r_right = np.median(r[right_mask]) if np.sum(right_mask) > 0 else 0
    
    # Determine which end is thinner
    if r_right > r_left:
        # Right is thicker, so left is thinner - cut from left (min X)
        thinner_end = 'left'
        cut_threshold = x_min + (range_x * cut_ratio)
        mask = x >= cut_threshold
    else:
        # Left is thicker, so right is thinner - cut from right (max X)
        thinner_end = 'right'
        cut_threshold = x_max - (range_x * cut_ratio)
        mask = x <= cut_threshold
    
    cut_points = points[mask]
    
    info = {
        'cut': True,
        'thinner_end': thinner_end,
        'remaining': len(cut_points)
    }
    return cut_points.astype(np.float32), info

def resample_pcd(data, n_points):
    n = data.shape[0]
    if n != n_points:
        idx = np.random.choice(n, n_points, replace=n < n_points)
        data = data[idx]
    return data

def calculate_bone_length(points):
    """
    Calculate the length of the bone.
    Assumes bone is roughly aligned to X-axis, or uses PCA to find max extent.
    To be robust, we'll re-align using PCA first to be sure.
    """
    aligned = align_bone_to_x_axis(points)
    x = aligned[:, 0]
    return x.max() - x.min()

def save_attention_map(dense_points, keypoints, attn_weights, save_path):
    """
    Interpolate attention weights from keypoints to dense partial points,
    map to a color heatmap (viridis), and save as PLY.
    
    Args:
        dense_points: (N, 3) dense partial input / aligned
        keypoints: (K, 3) keypoints from the model
        attn_weights: (K,) aggregated attention scores
        save_path: path to save the .ply
    """
    if len(attn_weights) != len(keypoints):
        print(f"  Warning: Attention weights shape mismatch: {attn_weights.shape} vs {keypoints.shape}")
        return
        
    # Interpolate from keypoints to dense semantic points using nearest neighbors Inverse Distance Weighting
    interpolator = NearestNDInterpolator(keypoints, attn_weights)
    dense_scores = interpolator(dense_points)
    
    # Normalize between 0 and 1
    score_min, score_max = dense_scores.min(), dense_scores.max()
    if score_max > score_min:
        dense_scores = (dense_scores - score_min) / (score_max - score_min)
    else:
        dense_scores = np.zeros_like(dense_scores)
        
    # Map to colors using matplotlib viridis
    cmap = plt.get_cmap('jet')
    colors = cmap(dense_scores)[:, :3] # RGBA to RGB
    
    # Save open3d point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    o3d.io.write_point_cloud(save_path, pcd)

# ============================================================================
# Main Evaluation Logic
# ============================================================================

def run_evaluation(args):
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Stats
    with open(args.stats, 'r') as f:
        stats = json.load(f)
    global_scale = float(stats['global_max_radius'])
    print(f"Global Scale: {global_scale} mm")

    # 2. Load Model
    config = cfg_from_yaml_file(args.config)
    base_model = build_model_from_cfg(config.model)
    
    state_dict = torch.load(args.ckpt, map_location='cpu')
    if 'base_model' in state_dict: state_dict = state_dict['base_model']
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state)
    base_model.to(device)
    base_model.eval()

    # 3. Scan Folder and Pair Files
    # Expected format: ID2310-102_1stof4.ply, ID2310-102_complete.ply
    files = glob.glob(os.path.join(args.folder, "*.ply"))
    
    # Group by ID
    # Regex: (ID\d+-\d+)_(.+)
    pattern = re.compile(r"(ID\d+-\d+)_(.+)\.ply")
    
    groups = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        match = pattern.match(fname)
        if match:
            bone_id = match.group(1)
            part_suffix = match.group(2) # e.g. "complete", "1stof2", "2ndof2"
            
            if bone_id not in groups:
                groups[bone_id] = {'parts': [], 'gt': None}
                
            if part_suffix == "complete":
                groups[bone_id]['gt'] = fpath
            else:
                groups[bone_id]['parts'].append(fpath)
        else:
            if args.debug:
                print(f"[Skip] File didn't match pattern: {fname}")

    print(f"Found {len(groups)} unique bone IDs.")
    
    # 4. Iterate and Evaluate
    squared_errors = []
    absolute_errors = []
    results = []

    print("\n{:<15} {:<20} {:<15} {:<15} {:<15}".format("Bone ID", "Part", "GT Length", "Pred Length", "Abs Error"))
    print("-" * 80)

    for bone_id, data in groups.items():
        gt_path = data['gt']
        if not gt_path:
            if args.debug: print(f"[Skip] No GT found for {bone_id}")
            continue
            
        # Calc GT Length
        gt_points = load_ply_or_npy(gt_path)
        len_gt = calculate_bone_length(gt_points)
        
        for part_path in data['parts']:
            part_name = os.path.basename(part_path)
            
            # Filter parts if needed (default: only 2 parts support as per prompt?)
            # Prompt says: "As of now just consider samples with 2 parts, but maybe add an option for predicting for all parts."
            # We'll support all matching parts by default or filter if user wants.
            # Checking if file contains "of2"
            if not args.all_parts and "of2" not in part_name:
                continue

            # --- Inference Pipeline ---
            # 1. Load & Align
            partial_raw = load_ply_or_npy(part_path)
            partial_aligned = align_bone_to_x_axis(partial_raw)
            partial_aligned = ensure_polarity(partial_aligned)
            
            # Apply Cut if requested
            if args.cut > 0:
                partial_aligned, _ = cut_from_thinner_end(partial_aligned, args.cut)
            
            # 2. Normalize
            partial_norm = partial_aligned / global_scale
            centroid_norm = np.mean(partial_norm, axis=0)
            partial_input = partial_norm - centroid_norm
            
            # 3. Resample
            partial_input_resampled = resample_pcd(partial_input, 2048)
            
            # 4. Forward
            input_tensor = torch.from_numpy(partial_input_resampled).unsqueeze(0).float().to(device)
            with torch.no_grad():
                if args.save_attn and args.output_dir:
                    rets, attn_weights, keypoints = base_model(input_tensor, return_attn=True)
                    attn_weights = attn_weights.squeeze() # (1024, 512)
                    agg_attn = attn_weights.mean(dim=0).cpu().numpy()
                    keypoints_np = keypoints.squeeze().cpu().numpy()
                    keypoints_global = (keypoints_np + centroid_norm) * global_scale
                else:
                    rets = base_model(input_tensor)
                dense_pred_local = rets[2]
            
            # 5. Reconstruct
            dense_pred_local_np = dense_pred_local.squeeze().cpu().numpy()
            dense_pred_global_norm = dense_pred_local_np + centroid_norm
            pred_world = dense_pred_global_norm * global_scale
            
            # --- Measure Length ---
            len_pred = calculate_bone_length(pred_world)
            
            error = len_pred - len_gt
            squared_errors.append(error ** 2)
            absolute_errors.append(abs(error))
            
            print("{:<15} {:<20} {:<15.2f} {:<15.2f} {:<15.2f}".format(
                bone_id, part_name, len_gt, len_pred, abs(error)
            ))
            
            # Save Prediction if requested
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                save_name = f"{bone_id}_{part_name.replace('.ply', '')}_pred.ply"
                save_path = os.path.join(args.output_dir, save_name)
                misc.save_ply(pred_world, save_path)
                
                if args.save_attn:
                    attn_name = f"{bone_id}_{part_name.replace('.ply', '')}_attn.ply"
                    attn_path = os.path.join(args.output_dir, attn_name)
                    save_attention_map(partial_aligned, keypoints_global, agg_attn, attn_path)
            
            results.append({
                'id': bone_id, 'part': part_name, 
                'gt_len': float(len_gt), 'pred_len': float(len_pred), 
                'error': float(error)
            })

    if not squared_errors:
        print("No samples evaluated.")
        return

    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    mae = np.mean(absolute_errors)
    
    print("-" * 80)
    print(f"Total Samples Evaluated: {len(squared_errors)}")
    print(f"RMSE Length Error: {rmse:.4f} mm")
    print(f"MAE Length Error:  {mae:.4f} mm")
    print("-" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Evaluate Learned Length RMSE')
    parser.add_argument('--folder', type=str, required=True, help='Path to folder containing .ply models')
    parser.add_argument('--stats', type=str, required=True, help='Path to dataset_stats.json')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default="cfgs/CustomBones_models/SymmCompletion.yaml")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    parser.add_argument('--all-parts', action='store_true', help='Evaluate all parts, not just "of2"')
    parser.add_argument('--cut', type=float, default=0.0, help='Cut ratio from thinner end (e.g. 0.1)')
    parser.add_argument('--output-dir', type=str, default="Visualizations", help='Folder to save predicted PLY files')
    parser.add_argument('--save-attn', action='store_true', help='Save attention map visualization of partial input (Requires --output-dir)')

    args = parser.parse_args()
    
    run_evaluation(args)
