import os
import torch
import numpy as np
import torch.utils.data as data
import open3d as o3d
from .io import IO
from .build import DATASETS
import random

@DATASETS.register_module()
class CustomBones(data.Dataset):
    """
    Custom Bones dataset: returns complete clouds, partials generated online.
    Returns (partial_local, gt_global, translation_offset)
    """

    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        if not os.path.exists(list_file):
            raise FileNotFoundError(f'List file not found: {list_file}')

        with open(list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_name': line
            })

        self.crop_method = getattr(config, 'CROP_METHOD', 'random_hole')
        print(f'[DATASET] {len(self.file_list)} instances were loaded. Crop Method: {self.crop_method}')

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def pc_norm(pc):
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        taxonomy_id = sample['taxonomy_id']
        file_name = sample['file_name']

        file_path = os.path.join(self.pc_path, taxonomy_id, file_name)
        data = IO.get(file_path).astype(np.float32)

        # Ensure fixed number of points
        n, c = data.shape
        if n != self.npoints:
            if n > self.npoints:
                idx = np.random.choice(n, self.npoints, replace=False)
            else:
                idx = np.random.choice(n, self.npoints, replace=True)
            data = data[idx]

        data = torch.from_numpy(data).float().numpy() # Back to numpy for Open3D/CPU ops

        # ONLINE DATA AUGMENTATION (Training only)
        if self.subset == 'train':
            # 1. Random X-axis Rotation (Preserves bone orientation)
            angle_x = np.random.uniform(-np.pi, np.pi)
            cos_a, sin_a = np.cos(angle_x), np.sin(angle_x)
            R_x = np.array([[1, 0, 0],
                            [0, cos_a, -sin_a],
                            [0, sin_a, cos_a]], dtype=np.float32)
            data = data @ R_x.T

            # 2. Random Uniform Scaling (Preserves biological proportions)
            # Uniform scale simulates a globally taller/shorter individual
            scale_factor = np.random.uniform(0.90, 1.10) # +/- 10% overall scale
            S = np.diag([scale_factor, scale_factor, scale_factor]).astype(np.float32)
            data = data @ S.T

            # 3. Random Jitter (Noise)
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.0015, data.shape).astype(np.float32)
                data = data + noise

        # 2. Generate Partial
        if self.crop_method == 'end_crop':
            # Spherical Cropping at Ends (simulating missing bone end)
            
            # PCA to find principal axis
            centroid = np.mean(data, axis=0)
            centered_data = data - centroid
            cov_matrix = np.cov(centered_data.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Project vertices onto principal axis to find bone length
            projections = np.dot(centered_data, principal_axis)
            bone_length = np.max(projections) - np.min(projections)
            
            # Find the two ends (min and max projections)
            min_proj_idx = np.argmin(projections)
            max_proj_idx = np.argmax(projections)
            
            # Randomly select which end to use as sphere center
            use_min_end = np.random.choice([True, False])
            if use_min_end:
                sphere_center = data[min_proj_idx]
            else:
                sphere_center = data[max_proj_idx]
                
            # Crop Ratio: Fraction of bone length to use as sphere radius
            crop_ratio = random.uniform(0.2, 0.75) 
            sphere_radius = crop_ratio * bone_length
            
            # Compute distances from the sphere center
            dists = np.linalg.norm(data - sphere_center, axis=1)
            
            # Keep vertices outside the sphere
            keep_mask = dists >= sphere_radius
            partial_raw = data[keep_mask]
            
        else: # 'random_hole' (Default)
            # Spherical Cropping (Hole Punching) somewhere on surface
            n_total = data.shape[0]
            crop_ratio = random.uniform(0.2, 0.75) # Crop 20% to 50%
            num_crop = int(n_total * crop_ratio)
            
            # Pick a random center point on the surface to be the center of the "Hole"
            center_idx = np.random.randint(0, n_total)
            center_point = data[center_idx]
            
            # Compute distances from this center
            dists = np.linalg.norm(data - center_point, axis=1)
            
            # Sort by distance: remove closest points (hole), keep furthest
            sort_idx = np.argsort(dists)
            keep_idx = sort_idx[num_crop:]
            partial_raw = data[keep_idx]
        
        # Safety check: if too few points, fallback to full
        if partial_raw.shape[0] < 512:
             partial_raw = data # Fallback to full if crop is too aggressive
             
        # FPS / Resampling to target_n (16384)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(partial_raw)
        
        # User requested INPUT size 16384 as well
        target_n = self.npoints 
        
        if len(pcd.points) > target_n:
            pcd_down = pcd.farthest_point_down_sample(target_n)
            partial = np.asarray(pcd_down.points).astype(np.float32)
        else:
            partial = np.asarray(pcd.points).astype(np.float32)
            
        # Resample to exact count
        if partial.shape[0] < target_n:
            choice = np.random.choice(partial.shape[0], target_n, replace=True)
            partial = partial[choice]
        elif partial.shape[0] > target_n:
             choice = np.random.choice(partial.shape[0], target_n, replace=False)
             partial = partial[choice]

        # 3. Calculate Centroid (Translation Offset) of the PARTIAL
        centroid = np.mean(partial, axis=0)
        
        # 4. Center the Partial
        partial = partial - centroid
        
        # Convert to Tensor
        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(data).float() # Original Global GT (but it was normalized in step 1, so it is centered global)
        centroid = torch.from_numpy(centroid).float()

        # Return: (partial_local, gt_global, translation_offset)
        return taxonomy_id, sample['model_id'], (partial, gt, centroid)
