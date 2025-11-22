#include <gaussian_splatting_slam/GaussianSplattingKeyframe.hpp>

namespace gaussian_splatting_slam
{
    GaussianSplattingKeyframe::GaussianSplattingKeyframe(
        const cv::cuda::GpuMat &rgbImg,
        const cv::cuda::GpuMat &depthImg,
        const cv::cuda::GpuMat &normalImg,
        const Pose3D &cameraPose,
        const CameraParameters &cameraParams)
    {
        cameraPose_ = cameraPose;
        cameraParams_ = cameraParams;

        rgbImgGpu_ = rgbImg.clone();
        depthImgGpu_ = depthImg.clone();
        normalImgGpu_ = normalImg.clone();

        colorTex_.reset(new Texture<uchar4>(rgbImgGpu_));
        depthTex_.reset(new Texture<float>(depthImgGpu_));
        normalTex_.reset(new Texture<float4>(normalImgGpu_));
    }

}; // namespace gaussian_splatting_slam