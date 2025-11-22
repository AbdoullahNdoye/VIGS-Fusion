#include <gaussian_splatting_slam/GSPoseCost.hpp>
#include <thrust/count.h>
#include <iostream>
#include <gaussian_splatting_slam/cuda_error_check.h>
#include <gaussian_splatting_slam/GaussianSplattingSlam.hpp>

namespace gaussian_splatting_slam
{
    GSPoseCostFunction::GSPoseCostFunction(GaussianSplattingSlam *gss) : gss_(gss)
    {
    }
    GSPoseCostFunction::~GSPoseCostFunction()
    {
    }

    void GSPoseCostFunction::update(int l)
    {
        l_ = l;
    }

    bool GSPoseCostFunction::Evaluate(double const *const *parameters,
                                      double *residuals,
                                      double **jacobians) const
    {
        return const_cast<GSPoseCostFunction *>(this)->EvaluateNonConst(parameters,
                                                                        residuals,
                                                                        jacobians);
    }

    bool GSPoseCostFunction::EvaluateNonConst(double const *const *parameters,
                                              double *residuals,
                                              double **jacobians)
    {
        Eigen::Map<Eigen::Vector3d> P_imu(const_cast<double *>(parameters[0]));
        Eigen::Map<Eigen::Quaterniond> Q_imu(const_cast<double *>(parameters[0] + 3));

        // //Eigen::Vector3d P_cam = P_imu + Q_imu.normalized().toRotationMatrix() * P_imu_cam;
        // Eigen::Vector3d P_cam = P_imu + Q_imu * P_imu_cam;

        // Eigen::Quaterniond Q_cam = Q_imu * Q_imu_cam; // camera rotation
        // Q_cam.normalize();

        // // std::cout << "translation of the IMU  " << translation_imu.x() << " " << translation_imu.y() << " " << translation_imu.z() << std::endl;
        // // std::cout << "translation of the camera  " << translation.x() << " " << translation.y() << " " << translation.z() << std::endl;
        // // std::cout << "rotation of the camera  " << rotation.x() << " " << rotation.y() << " " << rotation.z() << " " << rotation.w() << std::endl;

        // Pose3D pose;
        // pose.position = make_float3(P_cam.x(), P_cam.y(), P_cam.z());
        // Eigen::Map<Eigen::Quaternionf>((float *)&pose.orientation) = Q_cam.cast<float>();

        //jtj_jtr = gss_->optimizePoseGNCeres(l_, pose);
        Eigen::Matrix<double, 6, 6> JtJ;
        Eigen::Vector<double, 6> Jtr;

        gss_->optimizePoseGNCeres(JtJ, Jtr, l_, P_imu, Q_imu);

        double eps = 0.0001;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(JtJ);
        Eigen::VectorXd S = Eigen::VectorXd((es.eigenvalues().array() > eps).select(es.eigenvalues().array(), 0));
        Eigen::VectorXd S_inv = Eigen::VectorXd((es.eigenvalues().array() > eps).select(es.eigenvalues().array().inverse(), 0));
        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
        Eigen::MatrixXd J_star = S_sqrt.asDiagonal() * es.eigenvectors().transpose();
        Eigen::VectorXd r_star = S_inv_sqrt.asDiagonal() * es.eigenvectors().transpose() * Jtr;

        residuals[0] = r_star(0);
        residuals[1] = r_star(1);
        residuals[2] = r_star(2);
        residuals[3] = r_star(3);
        residuals[4] = r_star(4);
        residuals[5] = r_star(5);

        if (jacobians != NULL)
        {
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            J_pose.setZero();
            J_pose.block<6, 6>(0, 0) = J_star;

            // Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> J_pose(jacobians[0]);
            // J_pose.setZero();

            // Eigen::Matrix3d R_imu_cam = Q_imu_cam.normalized().toRotationMatrix();
            // Eigen::Matrix3d R_imu = Q_imu.normalized().toRotationMatrix();

            // Eigen::Matrix3d P_imu_cam_skew{{0.0, -P_imu_cam.z(), P_imu_cam.y()},
            //                                {P_imu_cam.z(), 0.0, -P_imu_cam.x()},
            //                                {-P_imu_cam.y(), P_imu_cam.x(), 0.0}};

            // Eigen::Matrix<double, 6, 6> J_imu_cam;
            // J_imu_cam.block<3, 3>(0, 0).setIdentity();
            // J_imu_cam.block<3, 3>(0, 3) = -R_imu * P_imu_cam_skew; //* P_imu_cam_skew; //setZero();
            // J_imu_cam.block<3, 3>(3, 0).setZero();                 //* J_star.block<3, 3>(0, 3);
            // J_imu_cam.block<3, 3>(3, 3) = R_imu_cam.transpose();   //.transpose() ;   //* J_star.block<3, 3>(3, 3);

            // J_pose.block<6, 6>(0, 0) = J_star * J_imu_cam;
        }

        return true;
    }

} // namespace gaussian_splatting_slam