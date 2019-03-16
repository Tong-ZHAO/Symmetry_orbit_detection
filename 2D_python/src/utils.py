#!/usr/bin/env python
# coding: utf8
# Author: Tong ZHAO

import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D 

def read_pslg(file_name):
    """Read data from pslg file

    Params:
        file_name (str): the absolute path for the file
    Returns:
        data          (np.array): the point cloud
        connect_left  (np.array): the left neighbor of each point, used for local feature calculation
        connect_right (np.array): the right neighbor of each point, used for local feature calculation
    """
    with open(file_name, "r") as fp:
        components = fp.read().split("component: ")[1:]
        
    lpoints, lconnect_left, lconnect_right = [], [], []
    index = 0
        
    for component in components:
        raw_points = component.split("\n")[1:-1]
        points = [list(map(float, line.split()[1:3])) for line in raw_points]
        connect_left = np.roll(np.arange(index, index + len(raw_points), 1), -1)
        connect_right = np.roll(connect_left, 2)
        
        lpoints = lpoints + points
        lconnect_left.append(connect_left)
        lconnect_right.append(connect_right)
        
        index += len(raw_points)
        
    return np.array(lpoints), np.hstack(lconnect_left).astype(int), np.hstack(lconnect_right).astype(int)


def generate_symmetry(points, pts_left, pts_right, mode = 1, offset = 0.2):
    """Generate symmetric figure by reflection.

    Params:
        points    (np.array): the point cloud
        pts_left  (np.array): the left neighbor of each point
        pts_right (np.array): the right neighbor of each point
        mode      (int)     : 1 - horizental reflection
                              2 - horizental + vertical reflection
        offset    (float)   : the distance between the boundary of the pattern and the symmetry axe
    Returns:
        points    (np.array): the updated point cloud
        pts_left  (np.array): the left neighbor of updated each point
        pts_right (np.array): the right neighbor of updated each point
    """

    assert(mode == 1 or mode == 2), "The selected mode (%d) is not defined!" % mode

    # horizental
    box_min, box_max = points.min(0), points.max(0)
    hor_axis = box_max[0] + offset
    
    new_comp = deepcopy(points)
    new_comp[:, 0] = 2 * hor_axis - new_comp[:, 0]
    
    points = np.vstack([points, new_comp])
    pts_left = np.hstack([pts_left, pts_left + len(pts_left)])
    pts_right = np.hstack([pts_right, pts_right + len(pts_right)])
    
    if mode == 2:
        # vertical
        ver_axis = box_min[1] + offset
        
        new_comp = deepcopy(points)
        new_comp[:, 1] = 2 * offset - new_comp[:, 1]
        
        points = np.vstack([points, new_comp])
        pts_left = np.hstack([pts_left, pts_left + len(pts_left)])
        pts_right = np.hstack([pts_right, pts_right + len(pts_right)])

    return points, pts_left, pts_right



def show_point_set(points, title, save_path = None):
    """ Show a point cloud in 2d or 3d
    Params:
        points (np.array): point cloud
        title       (str): title of the plot
        save_path   (str): where to save the plot
    """
    dim = points.shape[1]
    assert(dim == 2 or dim == 3), "Only data in 2 or 3 dimension space can be visualized."
    
    if dim == 2:
        fig = plt.figure(figsize = (6, 6))
        plt.title(title)
        plt.scatter(points[:, 0], points[:, 1], c = 'b', s = 5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis('equal')
        plt.grid()
        plt.show()
    elif dim == 3:
        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c = 'b', s = 5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        plt.show()

    if type(save_path) != type(None):
        fig.savefig(save_path)


def data_preprocessing(points):
    """Normalize and centralize a point cloud in a unit sphere

    Params:
        points (np.array): of size n x dim
    Returns:
        Processed point cloud
    """
    mean_coords = points.mean(0)
    points -= mean_coords
    
    max_norm = np.max(np.linalg.norm(points, axis = 1))
    points /= max_norm

    return points, mean_coords, max_norm


def show_point_set_curvature(points, curvatures, title, thresh = 0.9, save_path = None):
    """ Show a point cloud in 2d

    Params:
        points (np.array): point cloud
        title       (str): title of the plot
        save_path   (str): where to save the plot
    """
    dim = points.shape[1]
    assert(dim == 2), "Only data in 2 dimension space can be visualized."
    selected_pts = points[np.abs(curvatures) < thresh]
    
    fig = plt.figure(figsize = (6, 6))
    plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], c = 'b', s = 5, label = "point cloud")
    plt.scatter(selected_pts[:, 0], selected_pts[:, 1], c = 'r', s = 5, label = "selected point cloud")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
    
    if type(save_path) != type(None):
        fig.savefig(save_path)


def show_point_set_normal(points, normals, title, save_path = None):
    """ Show a point cloud in 2d

    Params:
        points (np.array): point cloud
        title       (str): title of the plot
        save_path   (str): where to save the plot
    """
    dim = points.shape[1]
    assert(dim == 2), "Only data in 2 dimension space can be visualized."
    
    fig = plt.figure(figsize = (6, 6))
    plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], c = 'b', s = 5, label = "point cloud")
    plt.quiver(points[:, 0], points[:, 1], normals[:, 0], normals[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
    
    if type(save_path) != type(None):
        fig.savefig(save_path)


def show_transformed_point_set(points_1, points_2, points_12, loss, save_path = None):
    """ Show a point cloud in 2d
    Params:
        points_1   (np.array): neighbors of point 1
        points_2   (np.array): neighbors of point 2
        points_12  (np.array): transformed neighbors of point 1
        loss       (float)   : alignment error
        title      (str)     : title of the plot
        save_path  (str)     : where to save the plot
    """

    fig = plt.figure(figsize = (6, 6))
    plt.title("Loss = %f" % loss)
    plt.scatter(points_1[:, 0], points_1[:, 1], s = 5, label = "point cloud 1")
    plt.scatter(points_2[:, 0], points_2[:, 1], s = 5, label = "point cloud 2")
    plt.scatter(points_12[:, 0], points_12[:, 1], s = 5, label = "point cloud 1 to 2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
    
    if type(save_path) != type(None):
        fig.savefig(save_path)


def show_embedding(lT_log, axis, labels, save_path = None):
    """Show embedding space along indicated axis

    Params:
        lT_log    (list[np.array]): the list of elements in sim(2)
        axis      (list[int])     : chosen axis
        labels    (list[str])     : the corresponding label to each axis
        save_path (str)           : where to save the plot
    """

    dim = len(axis)
    assert(dim == 2 or dim == 3), "Only data in 2 or 3 dimension space can be visualized."
    
    if dim == 2:
        fig = plt.figure(figsize = (6, 6))
        plt.scatter(lT_log[:, axis[0]], lT_log[:, axis[1]], c = 'b', s = 5)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.axis('equal')
        plt.grid()
        plt.show()
    elif dim == 3:
        fig = plt.figure(figsize = (6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(lT_log[:, axis[0]], lT_log[:, axis[1]], lT_log[:, axis[2]], c = 'b', s = 5)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        plt.show()

    if type(save_path) != type(None):
        fig.savefig(save_path)


def show_pair_points(points, pairs, title, save_path = None):
    """ Show a point cloud in 2d
    Params:
        points    (np.array): the whole point set
        pairs     (np.array): selected pairs of indices
        title     (str)     : title of the plot
        save_path (str)     : where to save the plot
    """

    fig = plt.figure(figsize = (6, 6))
    plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], c = 'b', s = 5, label = "point cloud")
    for pair in pairs:
        print(pair)
        point_1 = points[pair[0]]
        point_2 = points[pair[1]]
        plt.scatter([point_1[0], point_2[0]], [point_1[1], point_2[1]],  s = 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
    
    if type(save_path) != type(None):
        fig.savefig(save_path)


def show_symmetry(points, nb_1, nb_2, title, save_path = None):
    """ Show a point cloud in 2d
    Params:
        points    (np.array): the whole point set
        pairs     (np.array): selected pairs of indices
        title     (str)     : title of the plot
        save_path (str)     : where to save the plot
    """

    fig = plt.figure(figsize = (6, 6))
    plt.title(title)
    plt.scatter(points[:, 0], points[:, 1], c = 'b', s = 5, label = "point cloud")
    plt.scatter(nb_1[:, 0], nb_1[:, 1], c = 'r', s = 5, label = "symmetry_1")
    plt.scatter(nb_2[:, 0], nb_2[:, 1], c = 'g', s = 5, label = "symmetry_2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('equal')
    plt.grid()
    plt.legend()
    plt.show()
    
    if type(save_path) != type(None):
        fig.savefig(save_path)
