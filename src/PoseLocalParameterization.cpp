/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <gaussian_splatting_slam/PoseLocalParameterization.hpp>
#include <gaussian_splatting_slam/LocalParameterization.hpp>

namespace gaussian_splatting_slam
{
#if 0
}
#endif

    bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
    {
        Eigen::Map<const Eigen::Vector3d> _p(x);
        Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

        Eigen::Map<const Eigen::Vector3d> dp(delta);

        Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (_q * dq).normalized();

        /*std::cout << "Plus called. Position: " 
          << x[0] << ", " << x[1] << ", " << x[2] << ", "
          << x[3] << ", " << x[4] << ", " << x[5] << std::endl;
        std::cout << "Plus called. Delta: " 
          << delta[0] << ", " << delta[1] << ", " << delta[2] << ", "
          << delta[3] << ", " << delta[4] << ", " << delta[5] << std::endl;
        std::cout << "Plus called. Position + Delta: " 
          << x_plus_delta[0] << ", " << x_plus_delta[1] << ", " << x_plus_delta[2] << ", "
          << x_plus_delta[3] << ", " << x_plus_delta[4] << ", " << x_plus_delta[5] << std::endl;*/

        return true;
    }
    bool PoseLocalParameterization::ComputeJacobian(const double *, double *jacobian) const
    {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
        j.topRows<6>().setIdentity();
        j.bottomRows<1>().setZero();

        return true;
    }

    void PoseLocalParameterization::boxMinus(const double *xi, const double *xj,
                                             double *xi_minus_xj) const
    {
        Eigen::Map<const Eigen::Vector3d> pi(xi);
        // Eigen::Map<const Eigen::Quaterniond> qi(&xi[3]);

        Eigen::Map<const Eigen::Vector3d> pj(xj);
        // Eigen::Map<const Eigen::Quaterniond> qj(&xj[3]);

        Eigen::Map<Eigen::Vector3d> p(xi_minus_xj);
        Eigen::Map<Eigen::Vector3d> q(xi_minus_xj + 3);

        const Eigen::Quaterniond qi(xi[6], xi[3], xi[4], xi[5]);
        const Eigen::Quaterniond qj(xj[6], xj[3], xj[4], xj[5]);

        p = pi - pj;
        xi_minus_xj[3] = 2.0 * (-qi.w() * qj.x() + qi.x() * qj.w() - qi.y() * qj.z() + qi.z() * qj.y());
        xi_minus_xj[4] = 2.0 * (-qi.w() * qj.y() + qi.x() * qj.z() + qi.y() * qj.w() - qi.z() * qj.x());
        xi_minus_xj[5] = 2.0 * (-qi.w() * qj.z() - qi.x() * qj.y() + qi.y() * qj.x() + qi.z() * qj.w());
    }

    Eigen::MatrixXd PoseLocalParameterization::boxMinusJacobianLeft(double const *, double const *xj) const
    {
        // Eigen::Map<const Eigen::Quaterniond> qj(xj + 3);
        const Eigen::Quaterniond qj(xj[6], xj[3], xj[4], xj[5]);

        Eigen::MatrixXd J(6, 7);
        J.setZero();
        J.block<3, 3>(0, 0).setIdentity();
        J.block<3, 4>(3, 3) << qj.w(), -qj.z(), qj.y(), -qj.x(),
            qj.z(), qj.w(), -qj.x(), -qj.y(),
            -qj.y(), qj.x(), qj.w(), -qj.z();
        J.block<3, 4>(3, 3) *= 2.;

        return J;
    }

    Eigen::MatrixXd PoseLocalParameterization::boxMinusJacobianRight(double const *xi, double const *) const
    {
        // Eigen::Map<const Eigen::Quaterniond> qi(xi + 3);
        const Eigen::Quaterniond qi(xi[6], xi[3], xi[4], xi[5]);

        Eigen::MatrixXd J(6, 7);
        J.setZero();
        J.block<3, 3>(0, 0).setIdentity();
        J.block<3, 4>(3, 3) << -qi.w(), qi.z(), -qi.y(), qi.x(),
            -qi.z(), -qi.w(), qi.x(), qi.y(),
            qi.y(), -qi.x(), -qi.w(), qi.z();
        J.block<3, 4>(3, 3) *= 2.;

        return J;
    }

} // namespace gaussian_splatting_slam
