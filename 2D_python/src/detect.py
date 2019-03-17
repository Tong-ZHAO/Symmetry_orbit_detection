#!/usr/bin/env python
# coding: utf8
# Author: Tong ZHAO

import numpy as np
from sklearn.metrics import pairwise_distances


def distance_pts(log_mats, log_mat_ref, alpha, beta, gamma):
    """calculate the distances from a center point to all points in a set

    Params:
        log_mats    (np.array): a set of points in sim(2) of size n x 4
        log_mat_ref (np.array): a center point in sim(2) of size 1 x 4
        alpha       (float)   : coefficient for theta
        beta        (float)   : coefficient for u
        gamma       (float)   : coefficient for scaling factor
    Returns:
        distances (np.array): the calculated distances of size n x 1
    """
    
    if len(log_mats.shape) == 1:
        log_mats = log_mats.reshape((1, -1))
    
    weights = np.array([beta, beta, alpha, gamma])

    dist_sub = np.power(log_mats - log_mat_ref, 2).dot(weights)
    dist_sum = np.power(log_mats + log_mat_ref, 2).dot(weights)

    return np.minimum(dist_sub, dist_sum)


def distance_space(log_mats, generator, alpha, beta, gamma):
    """calculate the distances from a set of points to a plane

    Params:
        log_mats    (np.array): a set of points in sim(2) of size n x 4
        generator   (np.array): k point in sim(2) which generates the sub linear space of size k x 4
        alpha       (float)   : coefficient for theta
        beta        (float)   : coefficient for u
        gamma       (float)   : coefficient for scaling factor
    Returns:
        distances (np.array): the calculated distances of size n x 1
    """
    
    if len(log_mats.shape) == 1:
        log_mats = log_mats.reshape((1, -1))
    
    weights = np.array([beta, beta, alpha, gamma])
    
    log_mats = log_mats * weights
    generator = generator * weights.reshape((-1, 1))
    
    r = np.linalg.pinv(generator.T.dot(generator)).dot(generator.T).dot(log_mats.T)

    return np.linalg.norm(r, axis = 0)


def in_plane(points, generator, alpha, beta, gamma, thresh):
    """Given a point set and a linear space, find all points on the plane

    Params:
        points      (np.array): a set of points in sim(2) of size n x 4
        generator   (np.array): k point in sim(2) which generates the sub linear space of size 4 x k
        alpha       (float)   : coefficient for theta
        beta        (float)   : coefficient for u
        gamma       (float)   : coefficient for scaling factor
        thresh      (float)   : threshold to judge if a point is on the plane or not
    Returns:
        flags (np.array): an array indicating if a point belongs to the plane or not, of size n x 1
    """
    
    dists = distance_space(points, generator, alpha, beta, gamma)
    #print(dists)
    
    return (dists < thresh)


def ransac(points, k, alpha, beta, gamma, thresh, num_draws = 100):
    """RANSAC algorithm in high dimensional space

    Params:
        points    (np.array): a set of points in sim(2) of size n x 4
        k         (int)     : the dimension of the linear subspace
        alpha     (float)   : coefficient for theta
        beta      (float)   : coefficient for u
        gamma     (float)   : coefficient for scaling factor
        thresh    (float)   : threshold to judge if a point is on the plane or not
        num_draws (int)     : the number of random draw times
    Returns:
        best_generator (np.array): the best generators of the subspace of size k x 4
        best_score     (float)   : the best score
    """
    
    N = len(points)
    best_generator = np.zeros((4, k))
    best_score = 0
    
    for _ in range(num_draws):
        sampled_generator = points[np.random.choice(N, k, replace = False)]
        inliers = in_plane(points, sampled_generator.T, alpha, beta, gamma, thresh)
        
        if inliers.sum() > best_score:
            best_score = inliers.sum()
            best_generator = sampled_generator
            
    return best_generator, best_score


def sigma_estimation(points, alpha, beta, gamma, num_pairs = 10000):
    """Estimate gamma function for mean-shift by monte-carlo

    Params:
        points    (np.array): the point cloud
        alpha     (float)   : coefficient for theta
        beta      (float)   : coefficient for u
        gamma     (float)   : coefficient for scaling factor
        num_pairs (int)     : number of draw times

    Returns:
        avg_dist (float): average distance between pairs of points
    """
    
    avg_distances = 0
    N = len(points)
    
    for _ in range(num_pairs):
        
        idx_1, idx_2 = np.random.choice(N, 2, replace = False)
        avg_distances += np.squeeze(distance_pts(points[idx_1], points[idx_2], alpha, beta, gamma))
        
    return avg_distances / num_pairs


def mean_shift(points, sigma, alpha, beta, gamma, thresh = 1e-6, max_iter = 100):
    """Mean shift in sim(2) space to find symmetry

    Params:
        points    (np.array): the point cloud
        sigma     (float)   : variance of gaussian kernel
        alpha     (float)   : coefficient for theta
        beta      (float)   : coefficient for u
        gamma     (float)   : coefficient for scaling factor
        thresh    (float)   : early stop criteria
        max_iter  (int)     : number of maximum iteration
    Returns:
        center (np.array): the selected center
    """
    seed = np.random.choice(len(points))
    center = points[seed]
    
    for i in range(max_iter):
        
        distances = distance_pts(points - center, center, alpha, beta, gamma)
        k_distances = np.exp(-distances / (2 * sigma ** 2)).reshape((-1, 1))

        new_center = (k_distances * points).sum(0) / k_distances.sum()
        
        if distance_pts(new_center, center, alpha, beta, gamma) < thresh:
            return center
        
        center = new_center
        
    return center


def mean_neighbors(points, center, thresh, alpha, beta, gamma):
    """Given a selected center, find all neighbor points

    Params:
        points (np.array): the point cloud
        center (np.array): the center element in sim(2)
        thresh (float)   : the maximum distance from the center
        alpha  (float)   : coefficient for theta
        beta   (float)   : coefficient for u
        gamma  (float)   : coefficient for scaling factor
    Returns:
        flags (np.array): an array indicating if a point is around the center or not, of size n x 1
    """
    
    distances = distance_pts(points, center, alpha, beta, gamma)
    
    return distances < thresh


def region_growing_neighbors(neighbors_1, neighbors_2, T):
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
    
    new_neighbors = neighbors_1.dot(T[:2, :2].T) * T[2, 2] + T[:2, -1]
    distances = np.min(pairwise_distances(new_neighbors, neighbors_2), axis = 1)

    return new_neighbors, distances.mean()


def region_growing(points, points_left, points_right, pairs, T, threshold):

    nb_1 = [points_left[pairs[0]], pairs[0], points_right[pairs[0]]]
    nb_2 = [points_left[pairs[1]], pairs[1], points_right[pairs[1]]]

    while True:
        #print(nb_1)
        #print(nb_2)
        arr_1 = np.array([points[idx] for idx in nb_1])
        arr_2 = np.array([points[idx] for idx in nb_2])
        
        distances = region_growing_neighbors(arr_1, arr_2, T)[1]
        #print(distances)
        if distances < threshold:
            nb_1 = [points_left[nb_1[0]]] + nb_1 + [points_right[nb_1[-1]]]
            nb_2 = [points_left[nb_2[0]]] + nb_2 + [points_right[nb_2[-1]]]

            if len(set(nb_1 + nb_2)) < len(nb_1) + len(nb_2):
                break
        else:
            break

    arr_1 = np.array([points[idx] for idx in nb_1])
    arr_2 = np.array([points[idx] for idx in nb_2])

    return arr_1, arr_2

