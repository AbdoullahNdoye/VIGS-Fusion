/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <gaussian_splatting_slam/Utility.hpp>
#include <gaussian_splatting_slam/LocalParameterization.hpp>

namespace gaussian_splatting_slam
{
#if 0
}
#endif

    class PoseLocalParameterization : public gaussian_splatting_slam::LocalParameterizationBase
    {
    public:
        virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
        virtual bool ComputeJacobian(const double *x, double *jacobian) const;
        virtual int GlobalSize() const { return 7; };
        virtual int LocalSize() const { return 6; };

        virtual void boxMinus(const double *xi, const double *xj,
                              double *xi_minus_xj) const;
        virtual Eigen::MatrixXd boxMinusJacobianLeft(double const *xi, double const *xj) const;
        virtual Eigen::MatrixXd boxMinusJacobianRight(double const *xi, double const *xj) const;
    };

} // namespace gaussian_splatting_slam
