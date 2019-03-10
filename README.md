# Symmetry and Orbit Detection in Point Clouds

The repo is set for the final project for the MVA course: [Nuages de Points et Modélisation 3D](https://perso.telecom-paristech.fr/boubek/ens/master/mva/npm/index.html). I work on the paper: [Symmetry and Orbit Detection via Lie-Algebra Voting](http://www.geometry.caltech.edu/pubs/SADBH16.pdf).

The algorithm works on both 2D and 3D point clouds. The 2D version is implemented in Python. All the code for are written from scratch using simple functions in Numpy and Sklearn(only for KNN). The 3D version is implemented in C++, based on CGAL and Eigen library. It is faster and needs less known information about the input point cloud.

## Requirements

### 2D Implementation

* \>= Python 3

### 3D Implementation

* CGAL
* Eigen
* cpp 11+
