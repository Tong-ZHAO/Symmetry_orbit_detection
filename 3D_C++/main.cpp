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

#include "Symmetry_orbit_detect_3.h"

typedef CGAL::Simple_cartesian<double>  Kernel;
typedef Kernel::Point_3                 Point;

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
    Symmetry_orbit_detect_3<Kernel> my_points(filename, 0.1, 2, 2);
    my_points.pairing_points(0.1, 0.2);


    return EXIT_SUCCESS;
}