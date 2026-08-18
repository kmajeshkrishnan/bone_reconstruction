import open3d as o3d
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Convert point cloud to mesh using Poisson reconstruction')
    parser.add_argument('--input', type=str, required=True, help='Path to input point cloud (PLY/PCD file)')
    parser.add_argument('--output', type=str, default='poisson_mesh.ply', help='Path to output mesh file (default: poisson_mesh.ply)')
    parser.add_argument('--depth', type=int, default=8, help='Poisson reconstruction depth (default: 8, higher = more detail but may extrapolate with sparse clouds)')
    parser.add_argument('--voxel_size', type=float, default=None, help='Voxel size for downsampling (default: None, skips downsampling for sparse clouds)')
    parser.add_argument('--knn', type=int, default=30, help='Number of nearest neighbors for normal estimation (default: 30)')
    parser.add_argument('--density_threshold', type=float, default=0.0001, help='Density threshold quantile to remove low-confidence vertices (default: 0.01)')
    parser.add_argument('--skip_downsample', action='store_true', help='Skip downsampling step')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    print(f"Loading point cloud: {args.input}")
    pcd = o3d.io.read_point_cloud(args.input)
    print(f"Loaded {len(pcd.points)} points")
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=args.knn))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    # Optional: remove outliers / downsample
    if not args.skip_downsample and args.voxel_size is not None:
        print(f"Downsampling with voxel size: {args.voxel_size}")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"Downsampled to {len(pcd.points)} points")
    else:
        print(f"Skipping downsampling (keeping {len(pcd.points)} points)")
    
    # Poisson reconstruction
    print(f"Running Poisson reconstruction (depth={args.depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    print(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Crop by density to remove low-confidence parts
    print(f"Filtering by density (threshold: {args.density_threshold})...")
    dens = np.asarray(densities)
    vertices_to_keep = dens > np.quantile(dens, args.density_threshold)
    mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])
    print(f"Remaining mesh: {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    # Save mesh
    print(f"Saving mesh to: {args.output}")
    o3d.io.write_triangle_mesh(args.output, mesh)
    print("Done!")

if __name__ == "__main__":
    main()