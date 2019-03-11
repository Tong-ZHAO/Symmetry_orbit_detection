#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:$@$$) // boost::number_distance_distance()
                              // converts 64 to 32 bits integers
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Real_timer.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "Symmetry_orbit_detect_3.h"

typedef CGAL::Simple_cartesian<double>  Kernel;
typedef Kernel::Point_3                 Point;
typedef Eigen::Matrix<Kernel::FT, Eigen::Dynamic, Eigen::Dynamic>   EMatrix;
typedef Eigen::Matrix<Kernel::FT, Eigen::Dynamic, 1>                        EVector;

void usage(){
    std::cout << "Usage: ./main file.ply" << std::endl;
    std::cout << "file.ply: the file containing the point cloud" << std::endl;
}


int main(int argc, char** argv){
    // Check number of params
    if (argc != 2){
        usage();
        return EXIT_FAILURE;
    }

    std::string filename(argv[1]);
    Symmetry_orbit_detect_3<Kernel> my_points(filename, 0.18, 2, 2);
    my_points.pairing_points(0.18, 0.05);
    //my_points.ransac(2, 0.5);
    EVector mean_vector = my_points.mean_shift(100.);


    return EXIT_SUCCESS;
}