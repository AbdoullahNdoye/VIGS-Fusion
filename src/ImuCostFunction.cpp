#include <gaussian_splatting_slam/ImuCostFunction.hpp>
#include <gaussian_splatting_slam/Utility.hpp>
// #include <ros/ros.h>

namespace gaussian_splatting_slam
{
#if 0
}
#endif

    ImuCostFunction::ImuCostFunction(std::shared_ptr<Preintegration> &_pre_integration)
        : pre_integration(_pre_integration)
    {
    }

    bool ImuCostFunction::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> Pi(parameters[0]);
        Eigen::Map<const Eigen::Quaterniond> Qi(parameters[0] + 3);
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> Bai(parameters[1] + 3);
        Eigen::Map<const Eigen::Vector3d> Bgi(parameters[1] + 6);

        Eigen::Map<const Eigen::Vector3d> Pj(parameters[2]);
        Eigen::Map<const Eigen::Quaterniond> Qj(parameters[2] + 3);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> Baj(parameters[3] + 3);
        Eigen::Map<const Eigen::Vector3d> Bgj(parameters[3] + 6);


        if ((Bai - pre_integration->ba).norm() > 0.10 ||
            (Bgi - pre_integration->bg).norm() > 0.01)
        {
            pre_integration->repropagate(Bai, Bgi);
        }

        Eigen::Map<Eigen::Matrix<double, 15, 1>> residual(residuals);
        residual = pre_integration->evaluate(Pi, Qi, Vi, Bai, Bgi,
                                             Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix<double, 15, 15> sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();

        residual = sqrt_info * residual;

        if (jacobians)
        {
            double sum_dt = pre_integration->sum_dt;
            Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(0, 9);
            Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(0, 12);

            Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(3, 12);

            Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(6, 9);
            Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(6, 12);

            if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
            {
                // ROS_WARN("numerical unstable in preintegration");
                // std::cout << pre_integration->jacobian << std::endl;
                ///                ROS_BREAK();
            }

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();

                jacobian_pose_i.block<3, 3>(0, 0) = -Qi.inverse().toRotationMatrix();
                jacobian_pose_i.block<3, 3>(0, 3) = Utility::skewSymmetric(Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));

#if 0
            jacobian_pose_i.block<3, 3>(3, 3) = -(Qj.inverse() * Qi).toRotationMatrix();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->bg));
                jacobian_pose_i.block<3, 3>(3, 3) = -(Utility::Qleft(Qj.inverse() * Qi) * Utility::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_i.block<3, 3>(6, 3) = Utility::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));

                jacobian_pose_i = sqrt_info * jacobian_pose_i;

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
                {
                    // ROS_WARN("numerical unstable in preintegration");
                    // std::cout << sqrt_info << std::endl;
                    // ROS_BREAK();
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
                jacobian_speedbias_i.setZero();
                jacobian_speedbias_i.block<3, 3>(0, 6 - 6) = -Qi.inverse().toRotationMatrix() * sum_dt;
                jacobian_speedbias_i.block<3, 3>(0, 9 - 6) = -dp_dba;
                jacobian_speedbias_i.block<3, 3>(0, 12 - 6) = -dp_dbg;

#if 0
            jacobian_speedbias_i.block<3, 3>(3, 12 - 6) = -dq_dbg;
#else
                // Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->bg));
                // jacobian_speedbias_i.block<3, 3>(3, 12 - 6) = -Utility::Qleft(Qj.inverse() * Qi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
                jacobian_speedbias_i.block<3, 3>(3, 12 - 6) = -Utility::Qleft(Qj.inverse() * Qi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;
#endif

                jacobian_speedbias_i.block<3, 3>(6, 6 - 6) = -Qi.inverse().toRotationMatrix();
                jacobian_speedbias_i.block<3, 3>(6, 9 - 6) = -dv_dba;
                jacobian_speedbias_i.block<3, 3>(6, 12 - 6) = -dv_dbg;

                jacobian_speedbias_i.block<3, 3>(9, 9 - 6) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i.block<3, 3>(12, 12 - 6) = -Eigen::Matrix3d::Identity();

                jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;

                // ROS_ASSERT(fabs(jacobian_speedbias_i.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_speedbias_i.minCoeff()) < 1e8);
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
                jacobian_pose_j.setZero();

                jacobian_pose_j.block<3, 3>(0, 0) = Qi.inverse().toRotationMatrix();

#if 0
            jacobian_pose_j.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
#else
                Eigen::Quaterniond corrected_delta_q = pre_integration->delta_q * Utility::deltaQ(dq_dbg * (Bgi - pre_integration->bg));
                jacobian_pose_j.block<3, 3>(3, 3) = Utility::Qleft(corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
#endif

                jacobian_pose_j = sqrt_info * jacobian_pose_j;

                // ROS_ASSERT(fabs(jacobian_pose_j.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_pose_j.minCoeff()) < 1e8);
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
                jacobian_speedbias_j.setZero();

                jacobian_speedbias_j.block<3, 3>(6, 6 - 6) = Qi.inverse().toRotationMatrix();

                jacobian_speedbias_j.block<3, 3>(9, 9 - 6) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j.block<3, 3>(12, 12 - 6) = Eigen::Matrix3d::Identity();

                jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;

                // ROS_ASSERT(fabs(jacobian_speedbias_j.maxCoeff()) < 1e8);
                // ROS_ASSERT(fabs(jacobian_speedbias_j.minCoeff()) < 1e8);
            }
        }

        return true;
    }

} // namespace gaussian_splatting_slam
