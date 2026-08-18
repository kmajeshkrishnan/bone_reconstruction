"""
====================================================================
Create .npy files for Custom Dataset (Bones) - ShapeNet-like format
Stratified & Balanced Train/Test Split by Bone Type
WITH GLOBAL NORMALIZATION: Preserves relative scale between bones.
====================================================================
"""

import os
import numpy as np
import json
import re
from pathlib import Path
from tqdm import tqdm
import trimesh

# ================================================================
#  Mesh Loading + Point Sampling (UNSCALED)
# ================================================================
def load_and_sample_mesh_raw(obj_path, n_points=16384):
    """
    Load mesh from OBJ file, uniformly sample points.
    Returns CENTERED but UNSCALED point cloud (in mm).
    """
    try:
        mesh = trimesh.load(obj_path, force='mesh')
        points, _ = trimesh.sample.sample_surface(mesh, n_points)

        # Center the point cloud
        centroid = points.mean(axis=0)
        points -= centroid
        
        # Do NOT normalize here. Return raw MM scale.

        return points.astype(np.float32)

    except Exception as e:
        print(f"[ERROR] Failed to load {obj_path}: {e}")
        return None


# ================================================================
#  Bone Alignment Functions
# ================================================================
def align_bone_to_x_axis(points):
    """
    Align point cloud so that:
    1. The longest axis aligns with the X-axis
    2. Bone center is at the origin
    """
    # Center the point cloud at origin (already centered but safe to repeat)
    centroid = points.mean(axis=0)
    points_centered = points - centroid
    
    # Compute PCA to find principal axes
    cov = np.cov(points_centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalues (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    longest_axis_direction = eigenvectors[:, 0].real.astype(np.float32)
    longest_axis_direction = longest_axis_direction / (np.linalg.norm(longest_axis_direction) + 1e-8)
    
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    rotation_axis = np.cross(longest_axis_direction, x_axis)
    rotation_angle = np.arccos(np.clip(np.dot(longest_axis_direction, x_axis), -1.0, 1.0))
    
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)
    else:
        if np.dot(longest_axis_direction, x_axis) < 0:
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float32)
        else:
            R = np.eye(3, dtype=np.float32)
    
    aligned_points = np.dot(points_centered, R.T)
    return aligned_points.astype(np.float32)


# ================================================================
#  Filename Parsing → Bone Type
# ================================================================
def extract_bone_type_from_filename(filename):
    known_bones = ["Femur", "Humerus", "Tibia", "Fibula", "Radius", "Ulna"]
    for b in known_bones:
        if b.lower() in filename.lower():
            return b.capitalize()
    
    pattern = r"_([A-Za-z]+)_[LR]"
    match = re.search(pattern, filename)
    if match:
        return match.group(1).capitalize()
    return "Unknown"


# ================================================================
#  Main dataset creation
# ================================================================
def create_npy_dataset_global_norm(
    obj_folder,
    output_folder,
    n_points=2048,
    test_size=0.2,
    random_state=42,
    bone_type_filter=None,
    align_bones=True
):
    print(f"\nSearching for OBJ files in: {obj_folder}")
    obj_files = list(Path(obj_folder).rglob("*.obj"))
    print(f"Found {len(obj_files)} OBJ files\n")
    if len(obj_files) == 0:
        raise ValueError("No OBJ files found.")

    # 1. Collect Metadata
    bone_data = []
    for obj_path in obj_files:
        fn = obj_path.name
        bone_type = extract_bone_type_from_filename(fn)
        if bone_type_filter and bone_type.lower() != bone_type_filter.lower():
            continue
        bone_data.append({
            "filename": fn,
            "path": str(obj_path),
            "bone_type": bone_type
        })

    print(f"Processing {len(bone_data)} bone models")
    bone_types = sorted(set([x["bone_type"] for x in bone_data]))
    
    os.makedirs(output_folder, exist_ok=True)
    category_to_id = {bt: f"{i:08d}" for i, bt in enumerate(bone_types)}
    id_to_category = {cid: bt for bt, cid in category_to_id.items()}

    for cat_id in id_to_category:
        os.makedirs(os.path.join(output_folder, cat_id), exist_ok=True)

    # ------------------------------------------------------------
    # Pass 1: Compute Global Scale (Max Radius across ALL bones)
    # ------------------------------------------------------------
    print("\n[Pass 1] Computing Global Maximum Radius...")
    global_max_radius = 0.0

    loaded_point_clouds = []
    
    for item in tqdm(bone_data):
        pts = load_and_sample_mesh_raw(item["path"], n_points)
        if pts is None: continue
        
        if align_bones:
            pts = align_bone_to_x_axis(pts)
            
        # Calc radius
        # Radius = max distance from origin (they are centered)
        radius = np.max(np.linalg.norm(pts, axis=1))
        if radius > global_max_radius:
            global_max_radius = radius
            
        loaded_point_clouds.append({
            "pts": pts,
            "meta": item
        })
        
    # Add a Safety Factor (e.g. 10%)
    # This ensures that if we encounter a bone slightly larger than the training set 
    # during inference, it will still fit within the Unit Sphere (approx).
    # e.g. Max Train = 500mm. Safety = 550mm. 
    # A 520mm test bone will be 520/550 = 0.94 (Safe < 1.0)
    safety_factor = 1.1
    print(f"\n[Pass 1] True Global Max Radius: {global_max_radius:.4f} mm")
    global_max_radius *= safety_factor
    print(f"         Applying Safety Factor (x{safety_factor}): {global_max_radius:.4f} mm")

    print(f"GLOBAL MAX RADIUS (Normalized by): {global_max_radius:.4f} mm")
    print("All bones will be normalized by this factor.\n")
    
    # Save Stats
    stats = {
        "global_max_radius": float(global_max_radius),
        "unit": "mm"
    }
    with open(os.path.join(output_folder, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)


    # ------------------------------------------------------------
    # Pass 2: Normalize and Save
    # ------------------------------------------------------------
    print("[Pass 2] Normalizing and Saving...")
    
    processed_samples = {bt: [] for bt in bone_types}
    
    for entry in tqdm(loaded_point_clouds):
        pts_raw = entry["pts"]
        item = entry["meta"]
        
        # GLOBAL NORMALIZATION
        pts_norm = pts_raw / global_max_radius
        
        # Save
        fn = item["filename"]
        cat_id = category_to_id[item["bone_type"]]
        unique_id = fn.replace(".obj", "").replace(" ", "_")
        
        npy_name = f"{cat_id}-{unique_id}.npy"
        npy_path = os.path.join(output_folder, cat_id, npy_name)
        
        np.save(npy_path, pts_norm)
        processed_samples[item["bone_type"]].append(npy_name.replace(".npy", ""))

    print(f"\nSaved normalized samples to {output_folder}")

    # ============================================================
    # Balanced Stratified Split
    # ============================================================
    print("\nPerforming balanced stratified train/test split...\n")
    np.random.seed(random_state)
    split_info = {"train": [], "test": []}

    for bt, items in processed_samples.items():
        items = np.array(items)
        np.random.shuffle(items)
        n_test = max(1, int(len(items) * test_size))
        test_items = items[:n_test]
        train_items = items[n_test:]
        split_info["train"].extend(train_items.tolist())
        split_info["test"].extend(test_items.tolist())

    with open(os.path.join(output_folder, "split.json"), "w") as f:
        json.dump(split_info, f, indent=4)
    with open(os.path.join(output_folder, "categories.json"), "w") as f:
        json.dump({"category_to_id": category_to_id, "id_to_category": id_to_category}, f, indent=4)
    with open(os.path.join(output_folder, "train.txt"), "w") as f:
        for x in split_info["train"]: f.write(x + ".npy\n")
    with open(os.path.join(output_folder, "test.txt"), "w") as f:
        for x in split_info["test"]: f.write(x + ".npy\n")

    print("\n=== Dataset creation complete (Global Normalization) ===")


if __name__ == "__main__":
    OBJ_FOLDER = "<path_to_3d_models>"
    # New output folder to avoid overwriting existing
    OUTPUT_FOLDER = "./datasets/CustomBones/complete_global_norm_uniform"

    N_POINTS = 16384
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    create_npy_dataset_global_norm(
        obj_folder=OBJ_FOLDER,
        output_folder=OUTPUT_FOLDER,
        n_points=N_POINTS,
        test_size=TEST_SIZE
    )
