#pragma once

#include <Eigen/Dense>
#include <vector>

namespace gaussian_splatting_slam
{
#if 0
}
#endif

    class Preintegration
    {
    public:
        Preintegration();

        virtual void init(const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr,
                          const Eigen::Vector3d &_ba, const ::Eigen::Vector3d &_bg,
                          double acc_n, double gyr_n, double acc_w, double gyr_w);

        void add_imu(double _dt, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr);

        virtual void propagate(double _dt, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr);

        void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);

        // Eigen::Matrix<double, 15, 1>
        virtual Eigen::MatrixXd evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                         const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj);

        void predict(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                     Eigen::Vector3d &Pj, Eigen::Quaterniond &Qj, Eigen::Vector3d &Vj);

        const Eigen::Vector3d G = {0, 0, 9.81};

        bool is_initialized;

        std::vector<double> dt_buf;
        std::vector<Eigen::Vector3d> acc_buf;
        std::vector<Eigen::Vector3d> gyr_buf;
        double sum_dt;
        Eigen::Vector3d delta_p;
        Eigen::Quaterniond delta_q;
        Eigen::Vector3d delta_v;

        Eigen::Vector3d acc, gyr;
        Eigen::Vector3d ba, bg;

        Eigen::Vector3d linearized_acc, linearized_gyr;

        Eigen::MatrixXd jacobian, covariance, noise;
        //     Eigen::Matrix<double, 15, 15> jacobian, covariance;
        //     Eigen::Matrix<double, 18, 18> noise;
    };

} // namespace gaussian_splatting_slam
