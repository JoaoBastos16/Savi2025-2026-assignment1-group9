#!/usr/bin/env python3
# shebang line for linux / mac

import copy
from functools import partial
import glob
from random import randint
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import argparse
import open3d as o3d


def draw_registration_result(source, target, transformation):
    # Create independent copies of the point clouds so the originals are not modified
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    # Paint the point clouds for visualization (yellow for source, blue for target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    # Apply the given transformation to the source cloud
    source_temp.transform(transformation)

    # Display the aligned point clouds in an Open3D visualization window
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,                     # camera zoom
        front=[0.9288, -0.2951, -0.2242], # camera orientation
        lookat=[1.6784, 2.0612, 1.4451],  # point where the camera is looking
        up=[-0.3402, -0.9189, -0.1996]    # camera "up" direction
    )


def main():

    # Load example point clouds provided by Open3D (source and target)
    demo_icp_pcds = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(demo_icp_pcds.paths[0])
    target = o3d.io.read_point_cloud(demo_icp_pcds.paths[1])

    # Maximum correspondence distance (2 cm)
    threshold = 0.02

    # Initial transformation guess (4x4 matrix: rotation + translation)
    trans_init = np.asarray([
        [0.862, 0.011, -0.507, 0.5],
        [-0.139, 0.967, -0.215, 0.7],
        [0.487, 0.255, 0.835, -1.4],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # Visualize the initial alignment before ICP refinement
    draw_registration_result(source, target, trans_init)

    print("Initial alignment")
    # Evaluate how well the initial transformation aligns the two point clouds
    evaluation = o3d.pipelines.registration.evaluate_registration(
        source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-plane ICP")
    # Apply ICP using the Point-to-Plane formulation
    # This usually converges better when point clouds contain planar surfaces
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )

    # Print the full ICP result, including RMSE, fitness, and transformation matrix
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)

    # Visualize the final alignment after ICP
    draw_registration_result(source, target, reg_p2l.transformation)


if __name__ == '__main__':
    main()
