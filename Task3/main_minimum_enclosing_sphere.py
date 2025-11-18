#!/usr/bin/env python3
"""
Minimum Enclosing Sphere Optimization
Finds the smallest sphere that contains all points from two point clouds.
"""

import copy
import numpy as np
import open3d as o3d
from scipy.optimize import minimize, least_squares
import time


def compute_sphere_residuals(params, points):
    """
    Compute residuals for sphere constraint optimization.
    Residuals are positive when points are outside the sphere.
    
    Args:
        params: [xc, yc, zc, r] - sphere center and radius
        points: Nx3 array of points
    
    Returns:
        Array of residuals (distance_to_center - radius)
    """
    xc, yc, zc, r = params
    center = np.array([xc, yc, zc])
    
    # Calculate distances from center to all points
    distances = np.linalg.norm(points - center, axis=1)
    
    # Residuals: positive if point is outside sphere
    residuals = distances - r
    
    return residuals


def objective_function(params, points):
    """
    Objective function: minimize radius while ensuring all points are inside.
    
    Args:
        params: [xc, yc, zc, r] - sphere center and radius
        points: Nx3 array of points
    
    Returns:
        Objective value (radius + penalty for points outside)
    """
    xc, yc, zc, r = params
    center = np.array([xc, yc, zc])
    
    # Calculate distances from center to all points
    distances = np.linalg.norm(points - center, axis=1)
    
    # Maximum distance (this should equal radius at optimum)
    max_distance = np.max(distances)
    
    # Objective: minimize radius, with penalty if any point is outside
    # The penalty ensures all points stay inside
    penalty = np.maximum(0, max_distance - r).sum() * 1000
    
    return r + penalty


def constraint_all_points_inside(params, points):
    """
    Constraint function: all points must be inside the sphere.
    Returns negative values if constraint is satisfied.
    
    Args:
        params: [xc, yc, zc, r] - sphere center and radius
        points: Nx3 array of points
    
    Returns:
        Array of constraint values (should be <= 0)
    """
    xc, yc, zc, r = params
    center = np.array([xc, yc, zc])
    
    # Calculate distances from center to all points
    distances = np.linalg.norm(points - center, axis=1)
    
    # Constraint: distance - radius <= 0 (all points inside)
    return distances - r


def find_minimum_enclosing_sphere_constrained(points, initial_guess=None, pcd1=None, pcd2=None, visualize=True):
    """
    Find minimum enclosing sphere using constrained optimization.
    
    Args:
        points: Nx3 array of all points from both clouds
        initial_guess: Initial [xc, yc, zc, r] (optional)
        pcd1, pcd2: Point clouds for visualization (optional)
        visualize: Whether to show intermediate visualizations
    
    Returns:
        Optimized parameters [xc, yc, zc, r]
    """
    print("\n=== Method 1: Constrained Optimization ===")
    
    # Initial guess: centroid and maximum distance
    if initial_guess is None:
        centroid = np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points - centroid, axis=1))
        initial_guess = np.concatenate([centroid, [max_dist * 1.1]])
    
    print(f"Initial guess: center={initial_guess[:3]}, radius={initial_guess[3]:.4f}")
    
    # Visualize initial state
    if visualize and pcd1 is not None and pcd2 is not None:
        print("\n>>> Showing INITIAL sphere configuration...")
        visualize_sphere_iteration(pcd1, pcd2, initial_guess, "Initial Sphere (Before Optimization)")
    
    # Store intermediate results for visualization
    intermediate_params = []
    iteration_count = [0]  # Use list to modify in callback
    
    def callback(xk):
        """Callback function called at each iteration"""
        iteration_count[0] += 1
        if iteration_count[0] % 5 == 0:  # Store every 5th iteration
            intermediate_params.append(xk.copy())
    
    # Define constraint: all points must be inside sphere
    constraints = {
        'type': 'ineq',
        'fun': lambda params: -(constraint_all_points_inside(params, points).max())
    }
    
    # Bounds: radius must be positive
    bounds = [(None, None), (None, None), (None, None), (0.01, None)]
    
    start_time = time.time()
    
    # Optimize
    result = minimize(
        objective_function,
        initial_guess,
        args=(points,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        callback=callback,
        options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"Optimization completed in {elapsed_time:.3f} seconds")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.nit}")
    
    # Visualize intermediate states
    if visualize and pcd1 is not None and pcd2 is not None and len(intermediate_params) > 0:
        print(f"\n>>> Showing INTERMEDIATE sphere configurations ({len(intermediate_params)} states)...")
        for i, params in enumerate(intermediate_params):
            visualize_sphere_iteration(
                pcd1, pcd2, params, 
                f"Intermediate Sphere - Iteration {(i+1)*5} (radius={params[3]:.4f})"
            )
    
    return result.x


def find_minimum_enclosing_sphere_least_squares(points, initial_guess=None):
    """
    Find minimum enclosing sphere using least squares with soft constraints.
    This method minimizes the maximum violation.
    
    Args:
        points: Nx3 array of all points from both clouds
        initial_guess: Initial [xc, yc, zc, r] (optional)
    
    Returns:
        Optimized parameters [xc, yc, zc, r]
    """
    print("\n=== Method 2: Least Squares Optimization ===")
    
    # Initial guess: centroid and maximum distance
    if initial_guess is None:
        centroid = np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points - centroid, axis=1))
        initial_guess = np.concatenate([centroid, [max_dist]])
    
    print(f"Initial guess: center={initial_guess[:3]}, radius={initial_guess[3]:.4f}")
    
    start_time = time.time()
    
    # Use least squares to minimize residuals
    result = least_squares(
        compute_sphere_residuals,
        initial_guess,
        args=(points,),
        method='lm',
        ftol=1e-12,
        xtol=1e-12,
        max_nfev=1000
    )
    
    elapsed_time = time.time() - start_time
    
    # Adjust radius to ensure all points are inside
    xc, yc, zc, r = result.x
    center = np.array([xc, yc, zc])
    max_distance = np.max(np.linalg.norm(points - center, axis=1))
    
    # Set radius to maximum distance (ensuring all points are inside)
    result.x[3] = max_distance
    
    print(f"Optimization completed in {elapsed_time:.3f} seconds")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    return result.x


def create_sphere_wireframe(center, radius, color=[0.3, 0.5, 0.9], resolution=30):
    """
    Create a wireframe sphere for visualization (allows seeing points inside).
    
    Args:
        center: [xc, yc, zc] center coordinates
        radius: sphere radius
        color: RGB color [r, g, b]
        resolution: number of latitude/longitude lines
    
    Returns:
        Open3D LineSet representing the sphere wireframe
    """
    # Create sphere mesh first
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere_mesh.translate(center)
    
    # Convert to wireframe (LineSet from triangle edges)
    lines = o3d.geometry.LineSet.create_from_triangle_mesh(sphere_mesh)
    lines.paint_uniform_color(color)
    
    return lines


def create_sphere_point_cloud(center, radius, color=[0.3, 0.5, 0.9], num_points=5000):
    """
    Create a point cloud representation of sphere surface.
    
    Args:
        center: [xc, yc, zc] center coordinates
        radius: sphere radius
        color: RGB color [r, g, b]
        num_points: number of points on sphere surface
    
    Returns:
        Open3D PointCloud representing the sphere surface
    """
    # Generate random points on sphere surface using spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    theta = np.random.uniform(0, np.pi, num_points)
    
    x = radius * np.sin(theta) * np.cos(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(theta) + center[2]
    
    points = np.column_stack([x, y, z])
    
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(points)
    sphere_pcd.paint_uniform_color(color)
    
    return sphere_pcd


def visualize_sphere_iteration(pcd1, pcd2, params, iteration_name="Iteration", use_wireframe=True):
    """
    Visualize point clouds with sphere at a specific iteration.
    
    Args:
        pcd1, pcd2: Point clouds
        params: [xc, yc, zc, r] sphere parameters
        iteration_name: Name for the window title
        use_wireframe: If True, use wireframe; if False, use point cloud representation
    """
    # Create colored point clouds
    pcd1_vis = copy.deepcopy(pcd1)
    pcd1_vis.paint_uniform_color([1, 0, 0])  # Red
    
    pcd2_vis = copy.deepcopy(pcd2)
    pcd2_vis.paint_uniform_color([0, 1, 0])  # Green
    
    # Create sphere representation (wireframe or point cloud)
    if use_wireframe:
        sphere = create_sphere_wireframe(params[:3], params[3], color=[0.2, 0.4, 0.9], resolution=40)
    else:
        sphere = create_sphere_point_cloud(params[:3], params[3], color=[0.2, 0.4, 0.9], num_points=3000)
    
    # Create coordinate frame at sphere center
    center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=params[:3])
    
    # Display with draw_geometries
    o3d.visualization.draw_geometries(
        [pcd1_vis, pcd2_vis, sphere, center_frame],
        window_name=iteration_name,
        width=1280,
        height=720,
        point_show_normal=False
    )


def analyze_results(params, points1, points2):
    """
    Analyze and print statistics about the minimum enclosing sphere.
    
    Args:
        params: [xc, yc, zc, r] - optimized sphere parameters
        points1: Nx3 array of points from cloud 1
        points2: Mx3 array of points from cloud 2
    """
    xc, yc, zc, r = params
    center = np.array([xc, yc, zc])
    
    print("\n" + "="*60)
    print("MINIMUM ENCLOSING SPHERE RESULTS")
    print("="*60)
    
    print(f"\nSphere Parameters:")
    print(f"  Center: ({xc:.6f}, {yc:.6f}, {zc:.6f})")
    print(f"  Radius: {r:.6f}")
    print(f"  Volume: {(4/3) * np.pi * r**3:.6f} cubic units")
    print(f"  Surface Area: {4 * np.pi * r**2:.6f} square units")
    
    # Analyze point cloud 1
    distances1 = np.linalg.norm(points1 - center, axis=1)
    print(f"\nPoint Cloud 1 Statistics:")
    print(f"  Number of points: {len(points1)}")
    print(f"  Min distance to center: {np.min(distances1):.6f}")
    print(f"  Max distance to center: {np.max(distances1):.6f}")
    print(f"  Mean distance to center: {np.mean(distances1):.6f}")
    print(f"  Points on sphere boundary (within 0.1% of radius): {np.sum(np.abs(distances1 - r) < 0.001 * r)}")
    
    # Analyze point cloud 2
    distances2 = np.linalg.norm(points2 - center, axis=1)
    print(f"\nPoint Cloud 2 Statistics:")
    print(f"  Number of points: {len(points2)}")
    print(f"  Min distance to center: {np.min(distances2):.6f}")
    print(f"  Max distance to center: {np.max(distances2):.6f}")
    print(f"  Mean distance to center: {np.mean(distances2):.6f}")
    print(f"  Points on sphere boundary (within 0.1% of radius): {np.sum(np.abs(distances2 - r) < 0.001 * r)}")
    
    # Combined statistics
    all_distances = np.concatenate([distances1, distances2])
    print(f"\nCombined Statistics:")
    print(f"  Total points: {len(all_distances)}")
    print(f"  Max distance to center: {np.max(all_distances):.6f}")
    print(f"  Constraint satisfaction: ", end="")
    if np.all(all_distances <= r + 1e-6):
        print("✓ All points inside sphere")
    else:
        violations = np.sum(all_distances > r + 1e-6)
        print(f"✗ {violations} points outside sphere")
        print(f"  Max violation: {np.max(all_distances - r):.6f}")
    
    print("="*60)


def main():
    print("="*60)
    print("MINIMUM ENCLOSING SPHERE OPTIMIZATION")
    print("="*60)
    
    # Load point clouds from TUM dataset
    print("\nLoading point clouds from TUM dataset...")
    
    filename_rgb1 = '../tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)
    filename_depth1 = '../tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    
    filename_rgb2 = '../tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)
    filename_depth2 = '../tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    
    # Create point clouds
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    print(f"Point cloud 1: {len(pcd1.points)} points")
    print(f"Point cloud 2: {len(pcd2.points)} points")
    
    # Downsample for efficiency
    voxel_size = 0.05
    print(f"\nDownsampling with voxel size: {voxel_size}")
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)
    print(f"Point cloud 1: {len(pcd1.points)} -> {len(pcd1_down.points)} points")
    print(f"Point cloud 2: {len(pcd2.points)} -> {len(pcd2_down.points)} points")
    
    # Convert to numpy arrays
    points1 = np.asarray(pcd1_down.points)
    points2 = np.asarray(pcd2_down.points)
    
    # Combine all points for optimization
    all_points = np.vstack([points1, points2])
    print(f"\nTotal points for optimization: {len(all_points)}")
    
    # Method 1: Constrained optimization with visualization
    params_constrained = find_minimum_enclosing_sphere_constrained(
        all_points, pcd1=pcd1_down, pcd2=pcd2_down, visualize=True
    )
    analyze_results(params_constrained, points1, points2)
    
    # Method 2: Least squares
    params_ls = find_minimum_enclosing_sphere_least_squares(all_points)
    analyze_results(params_ls, points1, points2)
    
    # Use the better result (typically constrained gives better results)
    params_final = params_constrained
    
    # Visualization
    print("\n" + "="*60)
    print("FINAL VISUALIZATION")
    print("="*60)
    print("Preparing final visualization with transparent sphere...")
    
    # Visualize final result
    visualize_sphere_iteration(pcd1_down, pcd2_down, params_final, 
                              f"FINAL Minimum Enclosing Sphere (radius={params_final[3]:.4f})")
    
    print("\nVisualization Guide:")
    print("  - Red points: Point Cloud 1")
    print("  - Green points: Point Cloud 2")
    print("  - Blue wireframe/points: Minimum enclosing sphere boundary")
    print("  - RGB axes: Sphere center (Red=X, Green=Y, Blue=Z)")
    print("  - The sphere is shown as wireframe/points so you can see inside!")
    print("\nOptimization complete!")


if __name__ == '__main__':
    main()