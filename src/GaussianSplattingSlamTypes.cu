#include <gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp>

namespace gaussian_splatting_slam
{

void Gaussians::resize(size_t len)
{
    positions.resize(len);
    scales.resize(len);
    orientations.resize(len);
    colors.resize(len);
    alphas.resize(len);
    imgPositions.resize(len);
    imgSigmas.resize(len);
    imgInvSigmas.resize(len);
    pHats.resize(len);
    normals.resize(len);
}

} // namespace gaussian_splatting_slam
