#pragma once

#include <Eigen/Geometry>

namespace gaussian_splatting_slam
{

    struct ImuData
    {
        Eigen::Vector3d Acc;
        Eigen::Vector3d Gyro;
        double acc_n;
        double gyr_n;
        double acc_w;
        double gyr_w;
        double g_norm;
    };

}
