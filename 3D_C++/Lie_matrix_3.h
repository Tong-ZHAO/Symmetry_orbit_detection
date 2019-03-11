#ifndef LIE_MATRIX_3_H
#define LIE_MATRIX_3_H

#include <cmath>
#include <cassert>
#include <math.h>

#include <Eigen/Core>
#include <Eigen/Dense>

/// \internal
/// The Lie matrix class represents a transformation in 3D.
/// The purpose of this class is to create a tensor for each pair of points 

template <class FT>
class Lie_matrix_3
{
// Public types
public:

    typedef typename Eigen::Matrix<FT, Eigen::Dynamic, Eigen::Dynamic>      EMatrix;
    typedef typename Eigen::Matrix<FT, Eigen::Dynamic, 1>                   EVector;


// Private variables
private:

     EMatrix  m_rotation;
     EVector  m_offset;
     float    m_scale;
     EVector  m_log_tensor;

// Public methods
public:

    /// m_tensor is I_3 by default
    Lie_matrix_3(const EMatrix& rotation, const EVector& offset, const float scale):
    m_rotation(rotation), m_offset(offset), m_scale(scale), m_log_tensor(7)
    {
        build_log_matrix();
    }

    void build_log_matrix()
    {
        // log scale parameter
        m_log_tensor(6) = std::log(m_scale);

        // log rotation matrix
        float theta = acos((m_rotation.trace() - 1.) * 0.5);
        EMatrix log_r = (m_rotation - m_rotation.transpose()) * theta / sin(theta);
        //std::cerr << "log_r: " << log_r << std::endl;
        m_log_tensor(0) = log_r(2, 1);
        m_log_tensor(1) = log_r(0, 2);
        m_log_tensor(2) = log_r(1, 0);

        // log offset vector
        EMatrix mat_eye = EMatrix::Identity(3, 3); 
        EMatrix mat_v = mat_eye + (1 - cos(theta)) / pow(theta, 2) * log_r + (theta - sin(theta)) / pow(theta, 3) * log_r * log_r;
        EVector log_u = mat_v.inverse() * m_offset;
        //std::cerr << "log_u: " << log_u << std::endl;
        m_log_tensor(3) = log_u(0);
        m_log_tensor(4) = log_u(1);
        m_log_tensor(5) = log_u(2);
    }

    const EMatrix& rotation_matrix() const
    {
        return m_rotation;
    }

    const EVector& offset_vector() const
    {
        return m_offset;
    }

    const float scale_parameter() const
    {
        return m_scale;
    }

    const EVector& log_tensor() const
    {
        return m_log_tensor;
    }

    EMatrix& rotation_matrix()
    {
        return m_rotation;
    }

    EVector& offset_vector()
    {
        return m_offset;
    }

    float scale_parameter()
    {
        return m_scale;
    }

    EVector& log_tensor()
    {
        return m_log_tensor;
    }



};

#endif //LIE_MATRIX_3_H