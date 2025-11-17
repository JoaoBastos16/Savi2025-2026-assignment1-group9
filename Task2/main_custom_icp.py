#!/usr/bin/env python3
# shebang line for linux / mac
import copy
from copy import deepcopy
from functools import partial
import glob
from random import randint
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d
from scipy.optimize import least_squares
import cv2

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [10.0, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.87911045824568079, -0.1143707949631662, 0.46269225567601935],
                "lookat": [-14.857198715209961, 8.7558956146240234, 4.6393190026283264],
                "up": [-0.45122740480118839, 0.11291073802962912, 0.88523725316662361],
                "zoom": 0.53999999999999981
            }
        ],
    "version_major": 1,
    "version_minor": 0
}
def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def transform_points(points, params):
    """
    Aplica uma transformação rígida 3D definida por 6 parâmetros:
    3 de rotação (Rodrigues) e 3 de translação.
    """
    rx, ry, rz, tx, ty, tz = params
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    t = np.array([[tx], [ty], [tz]])
    transformed = (R @ points.T + t).T
    return transformed


def residual_function(params, src_pts, tgt_pts, tgt_normals=None, point_to_plane=False):
    """
    Função de erro para o least_squares.
    Retorna o vetor de resíduos (um valor por correspondência).
    """
    src_trans = transform_points(src_pts, params)
    diff = tgt_pts - src_trans

    if point_to_plane and tgt_normals is not None:
        # Erro Point-to-Plane (projeção ao longo da normal)
        residuals = np.sum(diff * tgt_normals, axis=1)
    else:
        # Erro Point-to-Point (distância euclidiana)
        residuals = np.linalg.norm(diff, axis=1)

    return residuals

def custom_icp(source_pcd, target_pcd, init_params=None,
               max_iterations=50, tolerance=1e-3, point_to_plane=True, 
               voxel_size=0.02, distance_threshold=0.05, visualize_iterations=True,
               visualize_interval=5):
    """
    Implementação do ICP personalizado com least_squares.
    source_pcd, target_pcd : open3d.geometry.PointCloud
    init_params : vetor de 6 elementos (rx, ry, rz, tx, ty, tz)
    voxel_size : tamanho do voxel para downsampling (None para desabilitar)
    distance_threshold : distância máxima para considerar correspondência válida
    visualize_iterations : se True, mostra visualizações intermediárias
    visualize_interval : intervalo entre visualizações (ex: 5 = mostra a cada 5 iterações)
    """
    if init_params is None:
        init_params = np.zeros(6)

    params = init_params.copy()
    
    # Downsample point clouds for efficiency
    if voxel_size is not None and voxel_size > 0:
        print(f"Downsampling with voxel size: {voxel_size}")
        src_down = source_pcd.voxel_down_sample(voxel_size)
        tgt_down = target_pcd.voxel_down_sample(voxel_size)
        print(f"Source points: {len(source_pcd.points)} -> {len(src_down.points)}")
        print(f"Target points: {len(target_pcd.points)} -> {len(tgt_down.points)}")
    else:
        src_down = source_pcd
        tgt_down = target_pcd
    
    # FIXED: Keep original points unchanged throughout iterations
    src_points = np.asarray(src_down.points).copy()
    tgt_points = np.asarray(tgt_down.points)
    tgt_normals = np.asarray(tgt_down.normals)
    kd_tree = o3d.geometry.KDTreeFlann(tgt_down)

    print("=== Início do ICP personalizado ===")

    for it in range(max_iterations):
        # 1️⃣ Encontrar correspondências (Nearest Neighbor)
        # FIXED: Transform points using current params, but don't modify original
        src_transformed = transform_points(src_points, params)
        
        correspondences = []
        distances = []
        for p in src_transformed:
            [_, idx, dist] = kd_tree.search_knn_vector_3d(p, 1)
            correspondences.append(idx[0])
            distances.append(dist[0])
        
        correspondences = np.array(correspondences)
        distances = np.array(distances)
        
        # Remove outlier correspondences based on distance threshold
        valid_mask = distances < distance_threshold
        n_valid = np.sum(valid_mask)
        n_total = len(valid_mask)
        
        if n_valid < 3:
            print(f"⚠️  Aviso: Apenas {n_valid} correspondências válidas. Parando ICP.")
            break
            
        print(f"   Correspondências válidas: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")
        
        # Filter points and correspondences
        src_points_filtered = src_points[valid_mask]
        correspondences_filtered = correspondences[valid_mask]
        matched_tgt = tgt_points[correspondences_filtered]
        matched_normals = tgt_normals[correspondences_filtered]

        # 2️⃣ Resolver otimização incremental (Least Squares)
        result = least_squares(
            residual_function, params,
            args=(src_points_filtered, matched_tgt, matched_normals, point_to_plane),
            method='lm'
        )

        delta = result.x - params
        params = result.x
        
        # Visualize only at specified intervals or first/last iteration
        should_visualize = (
            visualize_iterations and 
            (it == 0 or  # First iteration
             (it + 1) % visualize_interval == 0 or  # Every N iterations
             np.linalg.norm(delta) < tolerance)  # Last iteration (convergence)
        )
        
        if should_visualize:
            # Create visualization copies instead of modifying original
            src_vis = o3d.geometry.PointCloud()
            src_vis.points = o3d.utility.Vector3dVector(transform_points(src_points, params))
            src_vis.paint_uniform_color([1, 0, 0])
            
            tgt_vis = copy.deepcopy(tgt_down)
            tgt_vis.paint_uniform_color([0, 1, 0])
            
            o3d.visualization.draw_geometries([src_vis, tgt_vis], 
                window_name=f"ICP Customize – iteration {it+1:02d} : mean error = {np.mean(np.abs(result.fun)):.6f}")
            
        print(f"Iteração {it+1:02d}: erro médio = {np.mean(np.abs(result.fun)):.6f}")

        # 3️⃣ Critério de paragem
        if np.linalg.norm(delta) < tolerance:
            print("Convergência atingida!")
            
            # Show final iteration if not already shown
            if not should_visualize and visualize_iterations:
                src_vis = o3d.geometry.PointCloud()
                src_vis.points = o3d.utility.Vector3dVector(transform_points(src_points, params))
                src_vis.paint_uniform_color([1, 0, 0])
                
                tgt_vis = copy.deepcopy(tgt_down)
                tgt_vis.paint_uniform_color([0, 1, 0])
                
                o3d.visualization.draw_geometries([src_vis, tgt_vis], 
                    window_name=f"ICP Customize – iteration {it+1:02d} (FINAL) : mean error = {np.mean(np.abs(result.fun)):.6f}")
            
            break

    print("=== Fim do ICP ===")
    print("Transformação final (rx, ry, rz, tx, ty, tz):")
    print(params)
    
    # Return transformation matrix
    R, _ = cv2.Rodrigues(params[:3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = params[3:]
    
    return T


def main():

    # ------------------------------------
    # Visualize the point cloud
    # ------------------------------------
    filename_rgb1 = '../tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)

    filename_depth1 = '../tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)

    # Create the rgbd image
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    print(rgbd1)

    filename_rgb2 = '../tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)

    filename_depth2 = '../tum_dataset/depth/2.png'
    depth2 = o3d.io.read_image(filename_depth2)

    # Create the rgbd image
    rgbd2 = o3d.geometry.RGBDImage.create_from_tum_format(rgb2, depth2)
    print(rgbd2)

    # Obtain the point cloud from the rgbd image
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

 
    pcd_src = pcd1
    pcd_tgt = pcd2
    
    # Visualize initial alignment
    src_vis = copy.deepcopy(pcd_src)
    tgt_vis = copy.deepcopy(pcd_tgt)
    src_vis.paint_uniform_color([1, 0, 0])
    tgt_vis.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([src_vis, tgt_vis], window_name="Initial Alignment")
    
    # Estimação de normais (caso ainda não estejam calculadas)
    pcd_tgt.estimate_normals()
    pcd_src.estimate_normals()  # Estimate normals for source too

    # Transformação inicial manual (exemplo)
    init_params = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])  # pequena translação no eixo X

    # Executar ICP personalizado (now returns transformation matrix)
    # voxel_size: adjust based on your point cloud density
    # - Smaller (0.01): more points, slower but more accurate
    # - Larger (0.05): fewer points, faster but less detailed
    # distance_threshold: maximum distance for valid correspondence
    # - Smaller (0.02): stricter, removes more outliers
    # - Larger (0.1): more permissive, keeps more correspondences
    # visualize_interval: show visualization every N iterations
    # - 1: show every iteration
    # - 5: show every 5th iteration (default)
    # - Set visualize_iterations=False to disable intermediate visualization
    T_final = custom_icp(pcd_src, pcd_tgt, init_params, point_to_plane=True, 
                        voxel_size=0.02, distance_threshold=0.05,
                        visualize_iterations=True, visualize_interval=5)

    # FIXED: Apply final transformation to a copy for visualization
    pcd_src_aligned = copy.deepcopy(pcd_src)
    pcd_src_aligned.transform(T_final)
    pcd_src_aligned.paint_uniform_color([1, 0, 0])
    
    tgt_vis_final = copy.deepcopy(pcd_tgt)
    tgt_vis_final.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries([pcd_src_aligned, tgt_vis_final],
        window_name="ICP Customizado – Resultado Final")


if __name__ == '__main__':
    main()