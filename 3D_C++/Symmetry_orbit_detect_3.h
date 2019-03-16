#ifndef SYMMETRY_ORBIT_DETECT_3_H
#define SYMMETRY_ORBIT_DETECT_3_H

#include <cmath>
#include <CGAL/array.h>
#include <CGAL/IO/read_xyz_points.h>
#include <CGAL/Monge_via_jet_fitting.h>
#include <CGAL/pca_estimate_normals.h>
#include <CGAL/property_map.h>
#include <CGAL/Real_timer.h>

#include <CGAL/Kd_tree.h>
#include <CGAL/Splitters.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <vector>
#include <utility>
#include <iostream>
#include <stdlib.h>

#include "Lie_matrix_3.h"

/// \internal
/// The Lie matrix class represents a transformation in 3D.
/// The purpose of this class is to create a tensor for each pair of points 

template<class Kernel>
class Symmetry_orbit_detect_3
{
// Public types
public:

    typedef typename Kernel::FT       FT;
    typedef typename Kernel::Point_3  Point;
    typedef typename Kernel::Vector_3 Vector; ///< Kernel's Vector_3 class.

    typedef typename std::vector<Point>         PointList;
    typedef typename std::pair<Vector, Vector>  Curve;
    typedef typename std::vector<Curve>         CurveList;
    typedef typename std::vector<Vector>        NormalList;
    typedef typename std::pair<size_t, size_t>  MatchPoint;
    typedef typename std::vector<MatchPoint>    MatchList;
    typedef typename std::vector<Lie_matrix_3<FT>>  MatrixList;
    typedef typename std::vector<size_t>        IndexList;


    typedef typename CGAL::Monge_via_jet_fitting<Kernel>      My_Monge_via_jet_fitting;
    typedef typename My_Monge_via_jet_fitting::Monge_form     My_Monge_form;

    typedef typename CGAL::Search_traits_3<Kernel>                            SearchTraits_3;
    typedef typename CGAL::Sliding_midpoint<SearchTraits_3>                   Splitter;
    typedef typename CGAL::Kd_tree<SearchTraits_3, Splitter, CGAL::Tag_true>  Tree;
    typedef typename CGAL::Fuzzy_sphere<SearchTraits_3>                       Sphere;
    typedef typename CGAL::Orthogonal_k_neighbor_search<SearchTraits_3>       Neighbor_search;
    typedef typename Neighbor_search::Tree                                    Neighbor_tree;

    typedef typename Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic>  EMatrix;
    typedef typename Eigen::Matrix<FT, Eigen::Dynamic, 1>  EVector;

    typedef typename CGAL::Real_timer Timer;

    PointList  my_pts;
    CurveList  my_curvatures;
    NormalList my_normals;
    std::vector<float> my_curve_1;
    std::vector<float> my_curve_2;
    Tree m_tree;

    MatrixList my_mats;
    MatchList  my_pairs;


    Symmetry_orbit_detect_3(std::string file_name, float radius, float d_fitting, float d_monge)
    {
        std::cerr << "Open " << file_name << " for reading..." << std::endl;
        std::ifstream stream (file_name.c_str());
        if(!stream ||
           !CGAL::read_xyz_points( stream,
                                   std::back_inserter(my_pts)))
        {
            std::cerr << "Error: cannot read file " << file_name << std::endl;
        }
        else
            std::cerr << "Number of points: " << my_pts.size() << std::endl;

        calculate_curvatures(radius, d_fitting, d_monge);
        
    }

    void calculate_curvatures(float radius, float d_fitting, float d_monge)
    {
        Timer t;
        
        // build kd-tree
        t.reset();  
        t.start();
        std::cerr << "Building Tree..." << std::endl;
        for(size_t i = 0; i < my_pts.size(); i++)
            m_tree.insert(my_pts[i]);
        t.stop();
        std::cerr << "Tree is built in " << t.time() << "s" << std::endl;

        // calculate curvature
        PointList my_nbs;
        My_Monge_form             monge_form;
        My_Monge_via_jet_fitting  monge_fit;

        t.reset();
        t.start();
        std::cerr << "Calculating curvatures..." << std::endl;
        for(size_t i = 0; i < my_pts.size(); i++)
        {
            Sphere m_sphere(my_pts[i], radius, 0);
            m_tree.search(std::back_inserter(my_nbs), m_sphere);
            monge_form = monge_fit(my_nbs.begin(), my_nbs.end(), d_fitting, d_monge);
            Curve my_curve = std::make_pair(monge_form.maximal_principal_direction(), 
                                            monge_form.minimal_principal_direction());
            my_curvatures.push_back(my_curve);
            my_curve_1.push_back(monge_form.principal_curvatures(0));
            my_curve_2.push_back(monge_form.principal_curvatures(1));
            my_normals.push_back(CGAL::cross_product(my_curve.first, my_curve.second));
            my_nbs.clear();
        }
        t.stop();
        std::cerr << "Curvatures are computed in " << t.time() << "s" << std::endl;
    }


    void pairing_points(float radius, float threshold)
    {

        // pruning

        PointList my_nbs_i, my_nbs_j;

        for(size_t i = 0; i < my_pts.size() - 1; i++)
        {
            EMatrix mat_i(3, 3);
            get_matrix(i, mat_i);

            Sphere m_sphere_i(my_pts[i], radius, 0);
            m_tree.search(std::back_inserter(my_nbs_i), m_sphere_i);

            EVector mat_point_i(3);
            mat_point_i << my_pts[i].x(), my_pts[i].y(), my_pts[i].z();

            std::cerr << "Neighbor i size: " << my_nbs_i.size() << std::endl;

            for(size_t j = i + 1; j < my_pts.size(); j++)
            {
                EMatrix mat_j(3, 3);
                get_matrix(j, mat_j);

                float scale = std::abs(0.5 * (my_curve_1[i] / my_curve_1[j] + my_curve_2[i] / my_curve_2[j]));
                EMatrix rotate_ij = mat_i.transpose() * mat_j;
                
                EVector mat_trans_i(3);
                mat_trans_i = rotate_ij.transpose() * mat_point_i * scale;
                Point vec_trans_i(mat_point_i(0), mat_point_i(1), mat_point_i(2));
                Vector offset = my_pts[j] - vec_trans_i;
                EVector mat_offset(3);
                mat_offset << offset.x(), offset.y(), offset.z();

                Sphere m_sphere_j(my_pts[j], radius * 5, 0);
                m_tree.search(std::back_inserter(my_nbs_j), m_sphere_j);

                PointList my_trans_pts;

                for(auto pt = my_nbs_i.begin(); pt != my_nbs_i.end(); pt++){
                    EVector my_point(3);
                    my_point << pt -> x(), pt -> y(), pt -> z();
                    my_point = rotate_ij.transpose() * my_point * scale;
                    Point my_tpoint(my_point(0), my_point(1), my_point(2));
                    my_tpoint = my_tpoint + offset;
                    my_trans_pts.push_back(my_tpoint);
                }

                bool flag = estimate_alignment_error(my_trans_pts, my_nbs_j, threshold);

                if(flag)
                {
                    MatchPoint m_index = std::make_pair(i, j);
                    my_pairs.push_back(m_index);

                    Lie_matrix_3<FT> my_mat(rotate_ij, mat_offset, scale);
                    my_mats.push_back(my_mat);
                }

                // std::cerr << "Flag j: " << flag << std::endl;

                my_nbs_j.clear();
            }

            my_nbs_i.clear();
            
        
        }

        std::cerr << my_mats.size() << std::endl;

    }

    bool estimate_alignment_error(PointList& trans_pts, PointList& ref_pts, float threshold)
    {
        Neighbor_tree local_tree(ref_pts.begin(), ref_pts.end());
        
        float error = 0.;

        for(size_t i = 0; i < trans_pts.size(); i++)
        {
            Neighbor_search search(local_tree, trans_pts[i], 1);
            for(auto it = search.begin(); it != search.end(); ++it)
                error += std::sqrt(it->second);
        }

        error /= trans_pts.size();

        // std::cerr << "Alignment Error: " << error << std::endl;

        if(error >= threshold)
            return false;
        else
            return true;
        

    }

    void get_matrix(size_t i, EMatrix& mat_i)
    {
        mat_i.row(0) << my_normals[i].x(), 
                        my_normals[i].y(), 
                        my_normals[i].z();

        mat_i.row(1) << my_curvatures[i].first.x(),
                        my_curvatures[i].first.y(),
                        my_curvatures[i].first.z();

        mat_i.row(2) << my_curvatures[i].second.x(),
                        my_curvatures[i].second.y(),
                        my_curvatures[i].second.z();
    }

    float distance_space(EVector& log_mat, EMatrix& my_generators, float alpha, float beta, float gamma)
    {
        EVector weight(7);
        weight << alpha, alpha, alpha, beta, beta, beta, gamma;
        weight = weight.cwiseSqrt();
        EMatrix weighted_generators = my_generators;
        weighted_generators = weighted_generators.array().colwise() * weight.array();
        EVector weighted_log_mat = log_mat.cwiseProduct(weight);

        float distance = ((weighted_generators.transpose() * weighted_generators).inverse() * 
                           weighted_generators.transpose() * weighted_log_mat).norm();
        return distance;
    }

    size_t in_plane(EMatrix& my_generators, float alpha, float beta, float gamma, float threshold)
    {
        size_t score = 0;

        for(size_t i = 0; i < my_mats.size(); i++)
        {
            float dist_i = distance_space(my_mats[i].log_tensor(), my_generators, alpha, beta, gamma);
            if(dist_i < threshold)
                score += 1;
            std::cerr << "i distance to space: " << dist_i << std::endl;

        }

        return score;
    }

    EMatrix ransac(int k, float threshold, float alpha = 10., float beta = 1., float gamma = 1., int num_draws = 100)
    {
        EMatrix best_generator(7, k);
        size_t best_score = 0;

        for(size_t i = 0; i < num_draws; i++)
        {
            EMatrix my_generators(7, k);

            for(size_t j = 0; j < k; j++){
                size_t index = rand() % my_mats.size();
                my_generators.col(j) = my_mats[index].log_tensor();
            }

            std::cerr << my_generators << std::endl;

            size_t score = in_plane(my_generators, alpha, beta, gamma, threshold);

            if(score > best_score){
                best_score = score;
                best_generator = my_generators;
            }
            
        }

        std::cerr << best_score << std::endl;

        return best_generator;
    }

    EVector distance_points(EMatrix& points, EVector& center, float alpha, float beta, float gamma)
    {
        EVector weight(7);
        weight << alpha, alpha, alpha, beta, beta, beta, gamma;

        EVector dists_sub = (points.array().colwise() - center.array()).pow(2).matrix().transpose() * weight;
        EVector dists_add = (points.array().colwise() + center.array()).pow(2).matrix().transpose() * weight;

        EVector results = dists_sub.cwiseMin(dists_add);

        std::cerr << "results: " << results.sum() << std::endl;

        return results;
    }

    EVector mean_shift(float sigma, float alpha = 10., float beta = 1., float gamma = 1., float threshold = 1e-6, int max_iter = 1)
    {
        size_t index = rand() % my_mats.size();
        EVector center = my_mats[index].log_tensor();

        std::cerr << "Init center: " << center << std::endl;

        EMatrix my_tensors(7, my_mats.size());
        EVector kernel_dists;

        for(size_t i = 0; i < my_mats.size();  i++)
            my_tensors.col(i) = my_mats[i].log_tensor();

        kernel_dists = distance_points(my_tensors, center, alpha, beta, gamma);

        for(int i = 0; i < max_iter; i++)
        {   
            //std::cerr << kernel_dists << std::endl;
            kernel_dists = kernel_dists / (-2. * pow(sigma, 2));
            kernel_dists = kernel_dists.array().exp();
            kernel_dists = kernel_dists / kernel_dists.sum();
            // weighted 
            EMatrix weighted_tensors = my_tensors.array().rowwise() * kernel_dists.transpose().array();
            center = weighted_tensors.rowwise().sum();
            
            kernel_dists = distance_points(my_tensors, center, alpha, beta, gamma);

            //std::cerr << "sum of distances: " << kernel_dists.sum() << std::endl;

            if(kernel_dists.sum() < threshold)
                break;
        }

        return center;
    }

    void mean_shift_neighbors(IndexList& selected_idx, EVector& center, float threshold, float alpha = 10., float beta = 1., float gamma = 1.)
    {
        EMatrix my_tensors(7, my_mats.size());
        EVector kernel_dists(my_mats.size());

        for(size_t i = 0; i < my_mats.size();  i++)
            my_tensors.col(i) = my_mats[i].log_tensor();

        kernel_dists = distance_points(my_tensors, center, alpha, beta, gamma);

        for(size_t i = 0; i < my_mats.size(); i++)
            if(kernel_dists(i) < threshold)
                selected_idx.push_back(i);
    }

};

#endif