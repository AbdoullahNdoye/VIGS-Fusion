#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <memory>
#include <gaussian_splatting_slam/Preintegration.hpp>

namespace gaussian_splatting_slam
{
#if 0
}
#endif

    class ImuCostFunction : public ceres::SizedCostFunction<15, 7, 9, 7, 9> // résidu taille 15, 4 blocks de paramètres taille 7,9,7,9
    {
    protected:
        std::shared_ptr<Preintegration> pre_integration;
        const Eigen::Vector3d G{0, 0, 9.8};

    public:
        ImuCostFunction(std::shared_ptr<Preintegration> &_pre_integration);
        virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const; // ceres evaluate
    };

} // namespace gaussian_splatting_slam
