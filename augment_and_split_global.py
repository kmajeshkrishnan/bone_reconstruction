import os
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# ==========================================================
# Geometric Utilities for Aligned Bones
# ==========================================================

def rotate_around_x_axis(points, angle):
    """
    Rotate point cloud around X-axis only (preserves bone alignment).
    
    Args:
        points: (N, 3) point cloud
        angle: rotation angle in radians
    
    Returns:
        rotated points
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[1, 0, 0],
                  [0, cos_a, -sin_a],
                  [0, sin_a, cos_a]])
    return points @ R.T


def anisotropic_scale(points, scale_long=1.0, scale_yz=1.0):
    """
    Anisotropic scaling along X (bone length) and Y/Z (cross-section).
    
    Args:
        points: (N, 3) point cloud
        scale_long: scale along X-axis (bone length)
        scale_yz: scale along Y/Z axes (bone thickness)
    
    Returns:
        scaled points and the scale factors used
    """
    S = np.diag([scale_long, scale_yz, scale_yz])
    return points @ S.T, scale_long, scale_yz


def add_epiphysis_noise(points, base_sigma=0.0015):
    """
    Add position-dependent noise (higher at bone ends = epiphysis).
    
    Args:
        points: (N, 3) point cloud
        base_sigma: base noise standard deviation
    
    Returns:
        noisy points and the sigma used
    """
    x = points[:, 0]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6)
    # Higher weight at ends (x_norm close to 0 or 1)
    weight = np.abs(x_norm - 0.5) * 2
    sigma = base_sigma * weight
    noise = np.random.normal(0, sigma[:, None], points.shape)
    return points + noise, float(base_sigma)


# ==========================================================
# Extract subject ID
# ==========================================================

def extract_subject_id(filename: str) -> str:
    name = os.path.splitext(filename)[0]
    parts = name.split('-', 1)
    core = parts[1] if len(parts) > 1 else parts[0]
    tokens = core.split('_')
    return "_".join(tokens[:2]) if len(tokens) >= 2 else core


# ==========================================================
# Augmentation stage
# ==========================================================

def augment_dataset(data_path, output_path, unique_files, tag, augmentations_per_sample=10):
    """
    Augment aligned bones with X-axis rotations.
    Scaling and Noise are currently disabled.
    
    Args:
        data_path: path to input dataset
        output_path: path to save augmented dataset
        unique_files: list of files to augment
        tag: 'train', 'val', or 'test' (test is not augmented)
        augmentations_per_sample: number of augmentations
    """
    print(f"\nProcessing {tag} set ({len(unique_files)} files)...")

    # If test, just copy and return
    if tag == 'test':
        final_list = []
        for base in tqdm(unique_files):
            cat_id = base.split('-')[0]
            src = os.path.join(data_path, cat_id, base + ".npy")
            pts = np.load(src)
            # Save original
            dst_dir = os.path.join(output_path, cat_id)
            os.makedirs(dst_dir, exist_ok=True)
            np.save(os.path.join(dst_dir, base + ".npy"), pts)
            final_list.append(base + ".npy")
        return final_list

    # For train/val, augment
    final_list = []
    aug_log = {}

    for base in tqdm(unique_files):
        cat_id = base.split("-")[0]
        src_path = os.path.join(data_path, cat_id, base + ".npy")
        if not os.path.exists(src_path):
             print(f"Warning: {src_path} not found.")
             continue
             
        orig = np.load(src_path).astype(np.float32)

        # Save original
        dst_dir = os.path.join(output_path, cat_id)
        os.makedirs(dst_dir, exist_ok=True)
        np.save(os.path.join(dst_dir, base + ".npy"), orig)
        final_list.append(f"{base}.npy")
        aug_log[base] = []

        # Augmented versions
        for i in range(augmentations_per_sample):
            aug = orig.copy()

            # ==========================================
            # AUGMENTATION 1: X-axis rotation
            # ==========================================
            if i % 2 == 0:
                angle_x = np.random.uniform(-np.pi, np.pi)
            else:
                angle_x = np.pi
            aug = rotate_around_x_axis(aug, angle_x)

            # ==========================================
            # AUGMENTATION 2: Anisotropic scaling (DISABLED)
            # ==========================================
            # scale_x = np.random.uniform(0.97, 1.03)
            # scale_yz = np.random.uniform(0.99, 1.01)
            # aug, s_x, s_yz = anisotropic_scale(aug, scale_x, scale_yz)

            # ==========================================
            # AUGMENTATION 3: Epiphysis noise (DISABLED)
            # ==========================================
            noise_sigma = 0
            if np.random.random() > 0.75:
                aug, noise_sigma = add_epiphysis_noise(aug, base_sigma=0.0015)

            new_name = f"{base}_aug{i}"
            np.save(os.path.join(dst_dir, new_name + ".npy"), aug.astype(np.float32))
            final_list.append(new_name + ".npy")

            aug_log[base].append({
                "aug": new_name,
                "rot_x": float(angle_x),
                "noise_sigma": float(noise_sigma)
            })

    # Save log for this split
    with open(os.path.join(output_path, f"augment_params_{tag}.json"), "w") as f:
        json.dump(aug_log, f, indent=2)

    return final_list


def parse_bone_and_side(fname, id_to_cat):
    """
    Parse bone (class) using category ID mapping.
    Expected format: {CategoryID}-{Name}.npy
    """
    base = os.path.splitext(fname)[0]
    parts = base.split('-')
    
    cat_id = parts[0]
    if cat_id in id_to_cat:
        bone = id_to_cat[cat_id]
    else:
        # Fallback if ID not found (though unlikely given structure)
        bone = "Unknown"

    return bone


def split_data_first(data_path, target_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Load all files and split them into Train/Val/Test sets BEFORE augmentation.
    Split Ratios: Train 65%, Val 20%, Test 15%
    """
    print("\n=== Initial Balanced Split (Before Augmentation) ===\n")
    random.seed(seed)
    
    # Load category mapping
    cat_path = os.path.join(data_path, 'categories.json')
    if not os.path.exists(cat_path):
        raise FileNotFoundError(f"categories.json not found in {data_path}")
        
    with open(cat_path, 'r') as f:
        categories = json.load(f)
    id_to_cat = categories.get("id_to_category", {})

    # Load all files
    with open(os.path.join(data_path, 'train.txt')) as f:
        f1 = [x.strip().replace('.npy', '') for x in f if x.strip()]
    with open(os.path.join(data_path, 'test.txt')) as f:
        f2 = [x.strip().replace('.npy', '') for x in f if x.strip()]
        
    all_files = list(set(f1 + f2))
    
    # Group by bone type using reliable ID mapping
    bone_to_files = defaultdict(list)
    for f in all_files:
        bone = parse_bone_and_side(f, id_to_cat)
        bone_to_files[bone].append(f)
        
    train_files = []
    val_files = []
    test_files = []
    
    print(f"Splitting unique bones (Target Ratios: {target_ratios}):")
    for bone, files in sorted(bone_to_files.items()):
        n = len(files)
        random.shuffle(files)
        
        n_train = int(round(n * target_ratios[0]))
        n_val   = int(round(n * target_ratios[1]))
        n_test  = n - n_train - n_val
        
        # Safety checks
        if n_train < 1: n_train = 1
        if n_test < 1: n_test = 1
        if n_val < 0: n_val = 0
        
        # Adjust if sum exceeds n (rare rounding edge case)
        if n_train + n_val + n_test > n:
            n_test = n - n_train - n_val
            
        train_files.extend(files[:n_train])
        val_files.extend(files[n_train:n_train+n_val])
        test_files.extend(files[n_train+n_val:])
        
        print(f"  {bone:10s}: Total={n:3d} -> Train={n_train:3d}, Val={n_val:3d}, Test={n_test:3d}")
        
    return train_files, val_files, test_files


if __name__ == "__main__":
    DATA_PATH = "./datasets/CustomBones/complete_global_norm_uniform"
    OUTPUT_PATH = "./datasets/CustomBones/complete_global_norm_uniform_balanced"
    AUG = 20

    print("=" * 60)
    print("    Split-First Augmentation Pipeline")
    print("=" * 60)
    
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Input directory {DATA_PATH} does not exist!")
        exit(1)
        
    # 1. Perform Split on Unique Files
    train_unique, val_unique, test_unique = split_data_first(DATA_PATH)
    
    # 2. Setup Output
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(DATA_PATH, 'categories.json')) as f:
        categories = json.load(f)
    with open(os.path.join(OUTPUT_PATH, "categories.json"), "w") as f:
        json.dump(categories, f, indent=2)
        
    # 3. Augment Train (x20)
    train_final = augment_dataset(DATA_PATH, OUTPUT_PATH, train_unique, 'train', AUG)
    
    val_final = augment_dataset(DATA_PATH, OUTPUT_PATH, val_unique, 'val', AUG)
    
    # 5. Copy Test (No Augmentation)
    test_final = augment_dataset(DATA_PATH, OUTPUT_PATH, test_unique, 'test', 0)
    
    # 6. Save Lists
    with open(os.path.join(OUTPUT_PATH, "train.txt"), "w") as f:
        for x in sorted(train_final): f.write(x + "\n")
    with open(os.path.join(OUTPUT_PATH, "val.txt"), "w") as f:
        for x in sorted(val_final): f.write(x + "\n")
    with open(os.path.join(OUTPUT_PATH, "test.txt"), "w") as f:
        for x in sorted(test_final): f.write(x + "\n")
        
    # 7. Copy Stats
    if os.path.exists(os.path.join(DATA_PATH, "dataset_stats.json")):
        with open(os.path.join(DATA_PATH, "dataset_stats.json")) as f:
            stats = json.load(f)
        with open(os.path.join(OUTPUT_PATH, "dataset_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)
            
    print("\n✓ Pipeline complete!")
    print(f"  Train: {len(train_final)} files")
    print(f"  Val:   {len(val_final)} files")
    print(f"  Test:  {len(test_final)} files (Originals only)")
