#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <gaussian_splatting_slam/texture.hpp>
#include <gaussian_splatting_slam/LocalParameterization.hpp>
#include <gaussian_splatting_slam/PoseLocalParameterization.hpp>
#include <gaussian_splatting_slam/GaussianSplattingSlamKernels.hpp>
#include <gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp>
//#include <gaussian_splatting_slam/GaussianSplattingSlam.hpp>

using ceres::CostFunction;
using ceres::Problem;
using ceres::SizedCostFunction;
using ceres::Solver;

namespace gaussian_splatting_slam
{
/*#if 0
}
#endif*/
    class GaussianSplattingSlam;
    class GSPoseCostFunction : public ceres::SizedCostFunction<6, 7>
    {
    public:
        GSPoseCostFunction(GaussianSplattingSlam* gss);
        ~GSPoseCostFunction();
        /*bool Evaluate(double const *const *parameters,
                      double *residuals,
                      double **jacobians) const override;*/

        bool EvaluateNonConst(double const *const *parameters,
                              double *residuals,
                              double **jacobians);

        bool Evaluate(double const *const *parameters,
                              double *residuals,
                              double **jacobians) const;

        void update(int l);

    private:
        int l_;
        GaussianSplattingSlam* gss_;
    };

} // namespace gaussian_splatting_slam0