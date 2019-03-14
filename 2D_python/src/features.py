#!/usr/bin/env python
# coding: utf8
# Author: Tong ZHAO

import numpy as np

from copy import deepcopy
from sklearn.neighbors import KDTree
from utils import data_preprocessing


def calc_curvature(pts, pts_left, pts_right):
    """Calculate curvature from a 2D point set

    Params:
        points    (np.array): the point cloud
        pts_left  (np.array): the left neighbor of each point
        pts_right (np.array): the right neighbor of each point
    Returns:
        curvatures (np.array): the estimated curvature
    """
    
    N = len(pts)
    
    dx = (pts[pts_right][:, 0] - pts[pts_left][:, 0]) / 2.0
    dy = (pts[pts_right][:, 1] - pts[pts_left][:, 1]) / 2.0
    dxx = (dx[pts_right] - dx[pts_left]) / 2.0
    dyy = (dy[pts_right] - dy[pts_left]) / 2.0
    
    numerator = dx * dyy - dxx * dy
    denominator = np.power(dx * dx + dy * dy, 1.5)
    
    return numerator / denominator


def standard_pca(X):
    """Calculate local pca around a center point

    Params:
        X (np.array): the centralized point 
    Returns:
        vec_eigen (np.array): the eigen vectors
        val_eigen (np.array): the eigen values
    """
    
    q = np.dot(X.T, X) / len(X)
    val_eigen, vec_eigen = np.linalg.eigh(q)

    return vec_eigen[:, ::-1], val_eigen[::-1].reshape((1, -1))


def calc_normal(points, tree, radius = 0.1):
    """Calculate unoriented normals from a 2D point set

    Params:
        points  (np.array): the point cloud
        tree    (KDTree)  : the kdtree to find neighbors
        radius  (float)   : the radius to collect neighbors
    Returns:
        normals (np.array): the estimated unoriented normal
    """
    
    
    nbs_list = tree.query_radius(points, radius)
    normals = np.zeros((len(points), 2))
    
    for i, nbs in enumerate(nbs_list):
        my_nbs = points[nbs]
        eigen_vectors, _ = standard_pca(data_preprocessing(my_nbs)[0])
        normals[i] = eigen_vectors[:, -1] 
        
    normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)
        
    return normals


def orient_normals(normals, pts_left, pts_right):
    """Calculate unoriented normals from a 2D point set

    Params:
        points  (np.array): the point cloud
    Returns:
        normals (np.array): the estimated unoriented normal
    """
    
    new_normals = deepcopy(normals)
    
    for i in range(len(normals)):
        if pts_right[i] > i:
            pass
        else:
            next_idx = pts_left[i]
            if np.dot(new_normals[i], new_normals[next_idx]) < 0:
                new_normals[next_idx] = -new_normals[next_idx]
    
    return new_normals