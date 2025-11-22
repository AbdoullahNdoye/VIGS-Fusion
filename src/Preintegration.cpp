#include <gaussian_splatting_slam/Preintegration.hpp>

#include <iostream>

using namespace Eigen;

namespace gaussian_splatting_slam
{
#if 0
}
#endif

    template <typename Derived>
    Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
    {
        typedef typename Derived::Scalar Scalar_t;

        Eigen::Quaternion<Scalar_t> dq;
        Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
        half_theta /= static_cast<Scalar_t>(2.0);
        dq.w() = static_cast<Scalar_t>(1.0);
        dq.x() = half_theta.x();
        dq.y() = half_theta.y();
        dq.z() = half_theta.z();
        return dq;
    }

    // const Eigen::Vector3d G{0,0,9.81};

    Preintegration::Preintegration()
        : is_initialized(false)
    {
    }

    void Preintegration::init(const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr,
                              const Eigen::Vector3d &_ba, const ::Eigen::Vector3d &_bg,
                              double acc_n, double gyr_n, double acc_w, double gyr_w)
    {
        
        //std::cout << "IMU Preintegration initialized with  " << std::endl; 
        linearized_acc = _acc;
        linearized_gyr = _gyr;
        acc = _acc;
        gyr = _gyr;
        ba = _ba;
        bg = _bg;
        delta_p = Eigen::Vector3d::Zero();
        delta_q = Eigen::Quaterniond::Identity();
        delta_v = Eigen::Vector3d::Zero();
        sum_dt = 0.;

    
        jacobian = Eigen::MatrixXd::Identity(15, 15);
        //std::cout << "Jacobian size: " << jacobian.rows() << " x " << jacobian.cols() << std::endl;

        covariance = Eigen::MatrixXd::Zero(15, 15);
        noise = Eigen::MatrixXd::Zero(18, 18);

        noise.block<3, 3>(0, 0) = acc_n * acc_n * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = gyr_n * gyr_n * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = acc_n * acc_n * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = gyr_n * gyr_n * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = acc_w * acc_w * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = gyr_w * gyr_w * Eigen::Matrix3d::Identity();

        dt_buf.clear();
        acc_buf.clear();
        gyr_buf.clear();
        is_initialized = true;
    }

    void Preintegration::add_imu(double _dt, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr)
    {
        dt_buf.push_back(_dt);
        acc_buf.push_back(_acc);
        gyr_buf.push_back(_gyr);

        propagate(_dt, _acc, _gyr);
    }

    void Preintegration::propagate(double _dt, const Eigen::Vector3d &_acc, const Eigen::Vector3d &_gyr)
    {
        Vector3d acc0 = acc;
        Vector3d gyr0 = gyr;
        Quaterniond prev_delta_q = delta_q;

        Vector3d un_acc_0 = delta_q * (acc0 - ba);
        Vector3d un_gyr = 0.5 * (gyr0 + _gyr) - bg;
        delta_q = delta_q * Quaterniond(1, un_gyr(0) * _dt * 0.5, un_gyr(1) * _dt * 0.5, un_gyr(2) * _dt * 0.5);
        Vector3d un_acc_1 = delta_q * (_acc - ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
        delta_v = delta_v + un_acc * _dt;

        Vector3d w_x = 0.5 * (gyr0 + _gyr) - bg;
        Vector3d a_0_x = acc0 - ba;
        Vector3d a_1_x = _acc - ba;
        Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x << 0, -w_x(2), w_x(1),
            w_x(2), 0, -w_x(0),
            -w_x(1), w_x(0), 0;
        R_a_0_x << 0, -a_0_x(2), a_0_x(1),
            a_0_x(2), 0, -a_0_x(0),
            -a_0_x(1), a_0_x(0), 0;
        R_a_1_x << 0, -a_1_x(2), a_1_x(1),
            a_1_x(2), 0, -a_1_x(0),
            -a_1_x(1), a_1_x(0), 0;

        MatrixXd F = MatrixXd::Zero(15, 15);
        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
                              -0.25 * delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
        F.block<3, 3>(0, 6) = Matrix3d::Identity() * _dt;
        F.block<3, 3>(0, 9) = -0.25 * (prev_delta_q.toRotationMatrix() + delta_q.toRotationMatrix()) * _dt * _dt;
        F.block<3, 3>(0, 12) = -0.25 * delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
        F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
        F.block<3, 3>(3, 12) = -1.0 * Matrix3d::Identity() * _dt;
        F.block<3, 3>(6, 3) = -0.5 * prev_delta_q.toRotationMatrix() * R_a_0_x * _dt +
                              -0.5 * delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (prev_delta_q.toRotationMatrix() + delta_q.toRotationMatrix()) * _dt;
        F.block<3, 3>(6, 12) = -0.5 * delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * prev_delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 3) = 0.25 * -delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * 0.5 * _dt;
        V.block<3, 3>(0, 6) = 0.25 * delta_q.toRotationMatrix() * _dt * _dt;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) = 0.5 * Matrix3d::Identity() * _dt;
        V.block<3, 3>(3, 9) = 0.5 * Matrix3d::Identity() * _dt;
        V.block<3, 3>(6, 0) = 0.5 * prev_delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 3) = 0.5 * -delta_q.toRotationMatrix() * R_a_1_x * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) = 0.5 * delta_q.toRotationMatrix() * _dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = Matrix3d::Identity() * _dt;
        V.block<3, 3>(12, 15) = Matrix3d::Identity() * _dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();

        delta_q.normalize();
        sum_dt += _dt;
        acc = _acc;
        gyr = _gyr;
    }

    void Preintegration::repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
    {
        sum_dt = 0.0;
        acc = linearized_acc;
        gyr = linearized_gyr;
        delta_p.setZero();
        delta_q.setIdentity();
        delta_v.setZero();
        ba = _linearized_ba;
        bg = _linearized_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    Eigen::MatrixXd Preintegration::evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
                                             const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
    {
        Eigen::MatrixXd residuals(15, 1);

        //std::cout << "IMU Preintegration res calculation " << std::endl;    
        //std::cout << "Jacobian size: " << jacobian.rows() << " x " << jacobian.cols() << std::endl;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(0, 9);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(0, 12);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(3, 12);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(6, 9);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(6, 12);
        Eigen::Vector3d dba = Bai - ba;
        Eigen::Vector3d dbg = Bgi - bg;

        Eigen::Quaterniond corrected_delta_q = delta_q * deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(0, 0)  = (Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p);
        residuals.block<3, 1>(3, 0)  = (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(6, 0)  = (Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v);
        residuals.block<3, 1>(9, 0)  = (Baj - Bai);
        residuals.block<3, 1>(12, 0) = (Bgj - Bgi);

        //std::cout << "IMU Preintegration res calculation done " << std::endl;

        // Eigen::MatrixXd r(residuals);
        //     std::cout << "IMU Preintegration residual : " << residuals.transpose() << std::endl;
        //     std::cout << "Pi : " << Pi.transpose() << std::endl;
        //     std::cout << "Qi : " << Qi.x() << ' ' << Qi.y() << ' ' << Qi.z() << ' ' << Qi.w() << std::endl;
        //     std::cout << "Pj : " << Pj.transpose() << std::endl;
        //     std::cout << "Qj : " << Qj.x() << ' ' << Qj.y() << ' ' << Qj.z() << ' ' << Qj.w() << std::endl;
        //     std::cout << "nb IMU data : " << dt_buf.size() << std::endl;

        return residuals;
    }

    void Preintegration::predict(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
                                 Eigen::Vector3d &Pj, Eigen::Quaterniond &Qj, Eigen::Vector3d &Vj)
    {
        Qj = Qi * delta_q;
        Vj = Vi - G * sum_dt + Qi.toRotationMatrix() * delta_v;
        Pj = Pi + Vi * sum_dt - 0.5 * G * sum_dt * sum_dt + Qi.toRotationMatrix() * delta_p;
    }

} // namespace gaussian_splatting_slam