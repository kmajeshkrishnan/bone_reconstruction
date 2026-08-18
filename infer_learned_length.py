
import argparse
import os
import torch
import numpy as np
import open3d as o3d
from scipy.interpolate import NearestNDInterpolator
from models import build_model_from_cfg
from utils.config import cfg_from_yaml_file
from utils import misc
import json
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def load_ply_or_npy(filename):
    """Load point cloud from .ply or .npy file"""
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
    """
    Enforce canonical orientation: Thicker end at Negative X.
    Uses radius check at ends.
    """
    x = points[:, 0]
    x_min, x_max = x.min(), x.max()
    range_x = x_max - x_min
    
    left_mask = x < (x_min + range_x * 0.1)
    right_mask = x > (x_max - range_x * 0.1)
    
    r = np.sqrt(points[:, 1]**2 + points[:, 2]**2)
    
    r_left = np.median(r[left_mask]) if np.sum(left_mask) > 0 else 0
    r_right = np.median(r[right_mask]) if np.sum(right_mask) > 0 else 0
    
    print(f"  [Polarity] Left Radius: {r_left:.4f}, Right Radius: {r_right:.4f}")
    
    # We assume Canonical Template has Thick End at LEFT (Negative X).
    # If Right is THICKER than Left, it's flipped.
    if r_right > r_left:
        print("  [Polarity] Flipping Source to match Canonical Polarity...")
        points[:, 0] *= -1
        points[:, 1] *= -1 
        
    return points

def cut_from_thinner_end(points, cut_ratio=0.1):
    """
    Cut a portion from the thinner end of the bone using a planar cut.
    
    Args:
        points: Point cloud (N, 3)
        cut_ratio: Fraction to cut from thinner end (e.g., 0.1 = 10%)
    
    Returns:
        cut_points: Point cloud after cutting
        cut_info: Dictionary with cut statistics
    """
    if cut_ratio <= 0:
        return points, {'cut': False, 'cut_ratio': 0}
    
    x = points[:, 0]
    x_min, x_max = x.min(), x.max()
    range_x = x_max - x_min
    
    # Identify left and right ends
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
    
    original_points = len(points)
    cut_points = points[mask]
    removed_points = original_points - len(cut_points)
    
    print(f"  [Planar Cut] Thinner end: {thinner_end}")
    print(f"  [Planar Cut] Cut ratio: {cut_ratio*100:.1f}%")
    print(f"  [Planar Cut] Removed {removed_points} points (from {original_points} -> {len(cut_points)})")
    
    cut_info = {
        'cut': True,
        'cut_ratio': cut_ratio,
        'thinner_end': thinner_end,
        'original_points': original_points,
        'remaining_points': len(cut_points),
        'removed_points': removed_points
    }
    
    return cut_points.astype(np.float32), cut_info

    return cut_points.astype(np.float32), cut_info

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def resample_pcd(data, n_points):
    n = data.shape[0]
    if n != n_points:
        idx = np.random.choice(n, n_points, replace=n < n_points)
        data = data[idx]
    return data

def save_ply_as_png(ply_path, png_path):
    """Render point cloud from PLY file and save as PNG using matplotlib"""
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine color based on filename
        if 'partial' in ply_path.lower():
            color = 'blue'
            title = 'Input Partial Cloud'
        else:
            color = 'green'
            title = 'Completed Point Cloud'
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=1, alpha=0.7)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        plt.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"  Saved PNG: {png_path}")
    except Exception as e:
        print(f"  Warning: Failed to create PNG {png_path}: {e}")

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
    print(f"  Saved Attention Heatmap: {save_path}")

# ----------------------------------------------------------------------------
# Main Pipeline
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Inference with Learned Length (Global Norm)')
    parser.add_argument('--partial', type=str, required=True, help='Path to partial .ply')
    parser.add_argument('--stats', type=str, required=True, help='Path to dataset_stats.json (from new training data)')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--config', type=str, default="cfgs/CustomBones_models/SymmCompletion.yaml")
    parser.add_argument('--output_dir', type=str, default="./output_learned_length")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cut', type=float, default=0.1, help='Cut ratio from thinner end (e.g., 0.1 = 10%)')
    parser.add_argument('--uniform-target', type=int, default=0, help='Target points for uniform FPS resampling (e.g. 8192). 0 to disable.')
    parser.add_argument('--create-png', action='store_true', help='Create PNG visualizations (default: False)')
    parser.add_argument('--save-attn', action='store_true', help='Save attention map visualization of partial input')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Stats (The Global Scaler)
    with open(args.stats, 'r') as f:
        stats = json.load(f)
    print(f"Global Max Radius (Scaler): {stats['global_max_radius']} {stats.get('unit', '')}")
    global_scale = float(stats['global_max_radius'])

    # 2. Load Partial
    print(f"Loading Partial: {args.partial}")
    partial_raw = load_ply_or_npy(args.partial)
    print(f"Points: {len(partial_raw)}")

    # 3. Align to X-axis
    print("Aligning to X-axis...")
    partial_aligned = align_bone_to_x_axis(partial_raw)
    
    # 4. Polarity Check
    # Even without a template, we want consistent input orientation for the model.
    partial_aligned = ensure_polarity(partial_aligned)
    
    # 4b. Cut from Thinner End (Optional)
    if args.cut > 0:
        print(f"Cutting {args.cut*100:.1f}% from thinner end...")
        partial_aligned, cut_info = cut_from_thinner_end(partial_aligned, args.cut)
        print(f"  Result: {cut_info['remaining_points']} points remaining")

    # 5. Global Normalization
    print(f"Normalizing by Global Scale: {global_scale}")
    partial_norm = partial_aligned / global_scale
    
    # Check if we are inside unit sphere?
    max_r = np.max(np.linalg.norm(partial_norm, axis=1))
    print(f"Input Normalized Max Radius: {max_r:.4f} (Should be <= 1.0)")
    
    # 6. Prepare Model Input
    # Center locally (Model needs inputs centered at 0)
    # BUT we need to reconstruct this offset later.
    centroid_norm = np.mean(partial_norm, axis=0)
    partial_input = partial_norm - centroid_norm
    
    print(f"Partial Centroid (Normalized): {centroid_norm}")
    
    # Resample
    partial_input_resampled = resample_pcd(partial_input, 8192)
    
    # 7. Inference
    print("\n--- Inference ---")
    config = cfg_from_yaml_file(args.config)
    base_model = build_model_from_cfg(config.model)
    
    state_dict = torch.load(args.ckpt, map_location='cpu')
    if 'base_model' in state_dict: state_dict = state_dict['base_model']
    new_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(new_state)
    base_model.to(device)
    base_model.eval()
    
    input_tensor = torch.from_numpy(partial_input_resampled).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        if args.save_attn:
            rets, attn_weights, keypoints = base_model(input_tensor, return_attn=True)
            attn_weights = attn_weights.squeeze() # (1024, 512)
            # Average attention from all queries to each keypoint -> shape (512,)
            agg_attn = attn_weights.mean(dim=0).cpu().numpy()
            keypoints_np = keypoints.squeeze().cpu().numpy()
            
            # Keypoints are currently in local normalized space
            # Translate back to global world space to match dense points
            keypoints_global = (keypoints_np + centroid_norm) * global_scale
        else:
            rets = base_model(input_tensor)
            
        dense_pred_local = rets[2]
        
    dense_pred_local_np = dense_pred_local.squeeze().cpu().numpy()
    
    # 8. Reconstruction
    print("\n--- Reconstruction ---")
    # A. Normalized Space: Local -> Global
    dense_pred_global_norm = dense_pred_local_np + centroid_norm
    
    # B. World Space: Normalized -> World
    # Multiply by global_scale
    pred_world = dense_pred_global_norm * global_scale
    
    # 8b. Farthest Point Sampling (Uniform Density)
    if args.uniform_target > 0:
        if len(pred_world) >= args.uniform_target:
            print(f"Resampling to {args.uniform_target} points (FPS)...")
            pred_world = farthest_point_sample(pred_world, args.uniform_target)
        else:
             print(f"Warning: Output points ({len(pred_world)}) < Target ({args.uniform_target}), skipping FPS.")
    
    # 9. Save
    print(f"Saving to {args.output_dir}...")
    pred_path = os.path.join(args.output_dir, 'prediction_learned.ply')
    partial_path = os.path.join(args.output_dir, 'input_partial_aligned.ply')
    
    misc.save_ply(pred_world, pred_path)
    misc.save_ply(partial_aligned, partial_path)
    
    # Save Attention Map
    if args.save_attn:
        attn_path = os.path.join(args.output_dir, 'input_partial_attn.ply')
        save_attention_map(partial_aligned, keypoints_global, agg_attn, attn_path)
    
    # 10. Create PNG visualizations (optional)
    if args.create_png:
        print("Creating PNG visualizations...")
        save_ply_as_png(partial_path, partial_path.replace('.ply', '.png'))
        save_ply_as_png(pred_path, pred_path.replace('.ply', '.png'))
    
    print("Done.")

if __name__ == "__main__":
    main()
