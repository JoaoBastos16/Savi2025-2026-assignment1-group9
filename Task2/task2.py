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
               max_iterations=30, tolerance=1e-3, point_to_plane=True):
    """
    Implementação do ICP personalizado com least_squares.
    source_pcd, target_pcd : open3d.geometry.PointCloud
    init_params : vetor de 6 elementos (rx, ry, rz, tx, ty, tz)
    """
    if init_params is None:
        init_params = np.zeros(6)

    params = init_params.copy()
    src_points = np.asarray(source_pcd.points)
    tgt_points = np.asarray(target_pcd.points)
    tgt_normals = np.asarray(target_pcd.normals)
    kd_tree = o3d.geometry.KDTreeFlann(target_pcd)

    print("=== Início do ICP personalizado ===")

    for it in range(max_iterations):
        # 1️⃣ Encontrar correspondências (Nearest Neighbor)
        correspondences = []
        for p in transform_points(src_points, params):
            [_, idx, _] = kd_tree.search_knn_vector_3d(p, 1)
            correspondences.append(idx[0])

        matched_tgt = tgt_points[correspondences]
        matched_normals = tgt_normals[correspondences]

        # 2️⃣ Resolver otimização incremental (Least Squares)
        result = least_squares(
            residual_function, params,
            args=(src_points, matched_tgt, matched_normals, point_to_plane),
            method='lm'
        )

        delta = result.x - params
        params = result.x
        R, _ = cv2.Rodrigues(params[:3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = params[3:]

        source_pcd.transform(T)
        o3d.visualization.draw_geometries([
            source_pcd.paint_uniform_color([1, 0, 0]),
            target_pcd.paint_uniform_color([0, 1, 0]),
            ], window_name=f"ICP Customize — iteraction {it+1:02d} : medium error  = {np.mean(np.abs(result.fun)):.6f}")
            
        print(f"Iteração {it+1:02d}: erro médio = {np.mean(np.abs(result.fun)):.6f}")

        # 3️⃣ Critério de paragem
        if np.linalg.norm(delta) < tolerance:
            print("Convergência atingida!")
            break

    print("=== Fim do ICP ===")
    print("Transformação final (rx, ry, rz, tx, ty, tz):")
    print(params)
    return params





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

    # Show the images using matplotlib
    # plt.subplot(1, 2, 1)
    # plt.title('TUM grayscale image')
    # plt.imshow(rgbd1.color)
    # plt.subplot(1, 2, 2)
    # plt.title('TUM depth image')
    # plt.imshow(rgbd1.depth)
    # plt.show()

    # Obtain the point cloud from the rgbd image
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd1, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd2, o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

 
    pcd_src=pcd1
    pcd_tgt=pcd2
    o3d.visualization.draw_geometries([
        pcd_src.paint_uniform_color([1, 0, 0]),
        pcd_tgt.paint_uniform_color([0, 1, 0])
    ])
    # Estimação de normais (caso ainda não estejam calculadas)
    pcd_tgt.estimate_normals()

    # Transformação inicial manual (exemplo)
    init_params = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])  # pequena translação no eixo X

    # Executar ICP personalizado
    final_params = custom_icp(pcd_src, pcd_tgt, init_params, point_to_plane=True)

    # Aplicar transformação final e visualizar
    R, _ = cv2.Rodrigues(final_params[:3])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = final_params[3:]

    pcd_src.transform(T)
    o3d.visualization.draw_geometries([
        pcd_src.paint_uniform_color([1, 0, 0]),
        pcd_tgt.paint_uniform_color([0, 1, 0]),
    ]  ,window_name="ICP Customizado — Resultado Final",)


if __name__ == '__main__':
    main()


    
