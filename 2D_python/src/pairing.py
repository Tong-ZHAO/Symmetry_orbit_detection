#!/usr/bin/env python
# coding: utf8
# Author: Tong ZHAO

import numpy as np

from sklearn.neighbors import KDTree
from sklearn.metrics import pairwise_distances

def estimate_transform(point_1, point_2, normal_1, normal_2, curvature_1, curvature_2):
    """Find the transformation from point 1 to point 2

    Params:
        point_1     (np.array): the coordinate of the first point
        point_2     (np.array): the coordinate of the second point
        normal_1    (np.array): the normal of the first point
        normal_2    (np.array): the normal of the second point
        curvature_1 (float)   : the curvature of the first point
        curvature_2 (float)   : the curvature of the second point

    Return:
        T (np.array): translation matrix of size 3 x 3 (SIM(2))
    """

    # anti-clockwise: 1, clockwise: -1
    x = np.array([0, 1])
    flag_dir = 1 if np.dot(x, normal_1) >= np.dot(x, normal_2) else -1
    
    theta = np.arccos(np.clip(np.dot(normal_1, normal_2), None, 1.)) * flag_dir
    T = np.zeros((3, 3))
    # Rotation
    T[0, 0] = np.cos(theta)
    T[0, 1] = -np.sin(theta)
    T[1, 0] = np.sin(theta)
    T[1, 1] = np.cos(theta)
    
    # Scale
    T[2, 2] = np.abs(curvature_2 / curvature_1)
    
    # Translation
    offset = point_2 - T[:2, :2].dot(point_1) * T[2, 2]
    T[:2, -1] = offset
    
    return T

def transform_neighbors(point_1, point_2, pts, tree, T, radius):
    """Transform the neighbors of point 1 to point 2

    Params:
        point_1 (np.array): the coordinate of the first point
        point_2 (np.array): the coordinate of the second point
        pts     (np.array): the point cloud
        tree    (KDTree)  : the kdtree to find neighbors
        T       (np.array): the translation from point 1 to point 2
        radius  (float)   : the radius to collect neighbors

    Returns:
        neighbors_1   (np.array): the neighbor points of point 1
        neighbors_2   (np.array): the neighbor points of point 2
        new_neighbors (np.array): the transformed neigbor points of point 1
        distances     (float)   : the alignment error after transformation
    """
    
    neighbors_1 = pts[tree.query_radius(point_1.reshape((1, -1)), radius)[0]]
    new_neighbors = neighbors_1.dot(T[:2, :2].T) * T[2, 2] + T[:2, -1]
    max_dist = np.max(np.linalg.norm(new_neighbors - point_2, axis = 1))
    neighbors_2 = pts[tree.query_radius(point_2.reshape((1, -1)), max_dist + 0.01)[0]]
    
    distances = np.min(pairwise_distances(new_neighbors, neighbors_2), axis = 1)
    
    return neighbors_1, neighbors_2, new_neighbors, distances.sum() / len(distances)


def point_pairing(points, normals, curvatures, tree, radius, thresh, ratio = 0.2):
    """Calculate transformations between pairs of points

    Params:
        points     (np.array): the point cloud
        normals    (np.array): the normals
        curvatures (np.array): the curvatures
        tree       (KDTree)  : the tree to find neighbors of each point
        radius     (float)   : the radius to choose neighbors
        thresh     (float)   : the threshold to eliminate transformations with large alignment error
        ratio      (float)   : the proportion to choose a sub point set P' (fixed to 0.2)

    Returns:
        lT         (list[np.array]): the list of all transformations
        dict_index (dict)          : transformation index as the key, corresponding indices of points as the value
    """
    
    selected_num = int(ratio * len(points))
    selected_idx = np.random.choice(len(points), selected_num, replace = False)
    lT = []
    dict_index = {}
    index = 0
    
    for i in selected_idx:
        for j in range(len(points)):
            if i == j:
                pass
            else:
                T = estimate_transform(points[i], points[j], normals[i], normals[j], curvatures[i], curvatures[j])
                loss = transform_neighbors(points[i], points[j], points, tree, T, radius)[-1]

                if loss < thresh and T.sum() > 1e-6:
                    lT.append(T)
                    dict_index[index] = tuple((i, j))
                    index += 1
                
    lT = np.stack(lT)
    
    return lT, dict_index