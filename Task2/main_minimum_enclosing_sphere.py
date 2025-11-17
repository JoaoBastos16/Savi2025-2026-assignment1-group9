import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
import copy
import cv2

# Função para transformar os pontos usando parâmetros (rotação e translação)
def transform_points(points, params):
    rx, ry, rz, tx, ty, tz = params
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))
    t = np.array([[tx], [ty], [tz]])
    transformed = (R @ points.T + t).T
    return transformed

# Função para calcular a distância de cada ponto ao centro da esfera
def sphere_residual(params, src_pts, tgt_pts):
    xc, yc, zc, r = params
    all_points = np.concatenate([src_pts, tgt_pts], axis=0)
    # Distância euclidiana de cada ponto ao centro da esfera
    distances = np.linalg.norm(all_points - np.array([xc, yc, zc]), axis=1)
    residuals = distances - r
    return residuals

# Função para otimizar a esfera englobante mínima
def optimize_minimum_enclosing_sphere(src_points, tgt_points, init_params=None):
    if init_params is None:
        init_params = np.array([0.0, 0.0, 0.0, 1.0])  # centro (0, 0, 0) e raio inicial 1

    # Combina as duas nuvens de pontos
    all_points = np.concatenate([src_points, tgt_points], axis=0)

    # Otimização para minimizar o raio da esfera
    result = least_squares(sphere_residual, init_params, args=(src_points, tgt_points))

    # Parâmetros da esfera otimizada
    xc, yc, zc, r = result.x
    return xc, yc, zc, r

# Função principal
def main():
    # Carregar as imagens RGB e de profundidade
    filename_rgb1 = '/home/joao/savi_25-26/Parte08/tum_dataset/rgb/1.png'
    rgb1 = o3d.io.read_image(filename_rgb1)

    filename_depth1 = '/home/joao/savi_25-26/Parte08/tum_dataset/depth/1.png'
    depth1 = o3d.io.read_image(filename_depth1)

    # Create the rgbd image
    rgbd1 = o3d.geometry.RGBDImage.create_from_tum_format(rgb1, depth1)
    print(rgbd1)

    filename_rgb2 = '/home/joao/savi_25-26/Parte08/tum_dataset/rgb/2.png'
    rgb2 = o3d.io.read_image(filename_rgb2)

    filename_depth2 = '/home/joao/savi_25-26/Parte08/tum_dataset/depth/2.png'
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

    # Estimar as normais se não estiverem presentes
    if len(np.asarray(pcd1.normals)) == 0:
        pcd1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    if len(np.asarray(pcd2.normals)) == 0:
        pcd2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    src_points = np.asarray(pcd1.points)
    tgt_points = np.asarray(pcd2.points)

    # Inicializar parâmetros da esfera (posição e raio inicial)
    init_params = np.array([0.0, 0.0, 0.0, 1.0])  # centro da esfera no (0, 0, 0) e raio inicial 1

    # Otimizar a esfera englobante mínima
    xc, yc, zc, r = optimize_minimum_enclosing_sphere(src_points, tgt_points, init_params)

    # Visualizar o resultado
    print(f"Centro da esfera: ({xc:.3f}, {yc:.3f}, {zc:.3f})")
    print(f"Raio da esfera: {r:.3f}")

    # Visualizar as nuvens de pontos e a esfera otimizada
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=30)
    sphere.translate([xc, yc, zc])
    
    o3d.visualization.draw_geometries([
        pcd1.paint_uniform_color([1, 0, 0]),
        pcd2.paint_uniform_color([0, 1, 0]),
        sphere.paint_uniform_color([0, 0, 1])
    ], window_name="Resultado da Esfera Englobante Mínima")

if __name__ == "__main__":
    main()



    
