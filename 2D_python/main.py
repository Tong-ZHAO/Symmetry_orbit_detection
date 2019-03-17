#!/usr/bin/env python
# coding: utf8
# Author: Tong ZHAO

import os, sys
sys.path.append("src/")

import argparse
import numpy as np

from sklearn.neighbors import KDTree

from cluster import *
from detect import *
from features import *
from pairing import *
from utils import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type = str, default = '../2D_data/annulus.pslg', help = 'input pslg file')
    parser.add_argument('-m', '--mode', type = int, default = 0, help = '0 - Raw point cloud\n \
                                                                         1 - Horizental flip\n \
                                                                         2 - Horizental and vertical flip')
    parser.add_argument('-a', '--alpha', type = float, default = 1., help = 'alpha - coeff for theta')
    parser.add_argument('-b', '--beta', type = float, default = 1., help = 'beta - coeff for u')
    parser.add_argument('-g', '--gamma', type = float, default = 10., help = 'gamma - coeff for scaling factor')
    parser.add_argument('-r', '--radius', type = float, default = 0.02, help = 'the radius for normal estimation')
    parser.add_argument('-t', '--tpairing', type = float, default = 0.0005, help = 'the threshold for point pairing')
    parser.add_argument('-s', '--transac', type = float, default = 0.005, help = 'the threshold for ransac')
    parser.add_argument('-k', '--k', type = int, default = 2, help = 'the dimension of subspace in ransac')
    parser.add_argument('-e', '--tmean', type = float, default = 0.001, help = 'the threshold for mean-shift')
    parser.add_argument('-n', '--number', type = int, default = 250, help = 'the sub point number to do point pairing')
    parser.add_argument('--offset', type = float, default = 0.2, help = 'the distance from the raw point cloud to the symmetric axe \
                                                                         (used when mode = 1 or 2)')
                                                                        
    opt = parser.parse_args()

    # Read file
    pts, pts_left, pts_right = read_pslg(opt.data)
    print("Found %d points!" % len(pts))

    if opt.mode == 1 or opt.mode == 2:
        pts, pts_left, pts_right = generate_symmetry(pts, pts_left, pts_right, opt.mode, opt.offset)

    #show_point_set(pts, "Raw Point Cloud")

    # Calculate local features
    my_tree = KDTree(pts, 10)
    curvatures = calc_curvature(pts, pts_left, pts_right)
    normals = orient_normals(calc_normal(pts, my_tree, opt.radius), pts_left, pts_right)
    print("Curvatures and normal are calculated!")
    # Show local features
    #show_point_set_curvature(pts, curvatures, "curvature", 3)
    #show_point_set_normal(pts, -normals, "normals")

    # Test neighbor transformation
    if False:
        T = estimate_transform(pts[0], pts[1], normals[0], normals[1], curvatures[0], curvatures[1])
        n1, n2, n12, loss = transform_neighbors(pts[0], pts[1], pts, my_tree, T, opt.radius)
        show_transformed_point_set(n1, n2, n12, loss)

    # Select sub point cloud 
    print("Calculating transformations...")
    selected_idx = np.arange(0, len(pts), len(pts) // opt.number)
    my_subtree = KDTree(pts[selected_idx], 10)
    lT, dict_pairs = point_pairing(pts[selected_idx], normals[selected_idx], curvatures[selected_idx], my_subtree, opt.radius, opt.tpairing)
    print("%d pairs found!" % len(lT))

    # Clustering
    lT_log = np.clip(transform_embedding(lT), -20, 20)
    print("Lie algebra embedding found!")

    # Plot Embedding space
    #show_embedding(lT_log, [0, 1, 2], ["x1", "x2", "theta"])
    #show_embedding(lT_log, [0, 1, 3], ["x1", "x2", "lambd"])
    #show_embedding(lT_log, [0, 2, 3], ["x1", "theta", "lambd"])
    #show_embedding(lT_log, [1, 2, 3], ["x2", "theta", "lambd"])

    print(pts)

    if True:
    # Ransac - for orbits
        best_generators, best_score = ransac(lT_log, opt.k, opt.alpha, opt.beta, opt.gamma, opt.transac)
        print("Best score: ", best_score)
        select_transforms = in_plane(lT_log, best_generators.T, opt.alpha, opt.beta, opt.gamma, opt.transac)
        nb_pairs = []
        #pairs = [dict_pairs[index] for index in np.where(select_transforms == 1)[0]]
        for idx in np.where(select_transforms == 1)[0]:
            pair = [selected_idx[dict_pairs[idx][0]], selected_idx[dict_pairs[idx][1]]]
            nb_1, nb_2 = region_growing(pts, pts_left, pts_right, pair, lT[idx], 0.1) 
            nb_pairs.append([nb_1, nb_2])
        show_orbits(pts, nb_pairs[:3], "RANSAC")

    #print(lT_log)
    if False:
        # Mean shift - for symmetry
        sigma = sigma_estimation(lT_log, opt.alpha, opt.beta, opt.gamma) / 10.
        #sigma = 50.
        print("Estimated sigma: ", sigma)
        center = mean_shift(lT_log, sigma, opt.alpha, opt.beta, opt.gamma)
        #select_transforms = mean_neighbors(lT_log, center, opt.tmean, opt.alpha, opt.beta, opt.gamma)
        #pairs = [dict_pairs[index] for index in np.where(select_transforms == 1)[0]]
        #print("Number of pairs: ", len(pairs))
        #show_pair_points(pts[selected_idx], pairs, "Mean-shift")
        all_distances = distance_pts(lT_log, center, opt.alpha, opt.beta, opt.gamma)
        index = np.argmin(all_distances)
        pair = [selected_idx[dict_pairs[index][0]], selected_idx[dict_pairs[index][1]]] 
        nb_1, nb_2 = region_growing(pts, pts_left, pts_right, pair, lT[index], 0.5)
        print(nb_1)
        print(nb_2)
        show_symmetry(pts, nb_1, nb_2, "Symmetry")
