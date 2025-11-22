#pragma once

#include <gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp>
#include <gaussian_splatting_slam/texture.hpp>
#include <opencv2/core/cuda.hpp>

namespace gaussian_splatting_slam
{
    class GaussianSplattingKeyframe
    {
    public:
        GaussianSplattingKeyframe(
            const cv::cuda::GpuMat &rgbImg,
            const cv::cuda::GpuMat &depthImg,
            const cv::cuda::GpuMat &normalImg,
            const Pose3D &cameraPose,
            const CameraParameters &cameraParams);

        inline const Pose3D& getCameraPose(){return cameraPose_;}
        inline const CameraParameters& getCameraParams(){return cameraParams_;}
        inline Texture<uchar4>& getColorTex(){ return *colorTex_;}
        inline Texture<float>& getDepthTex(){ return *depthTex_;}
        inline Texture<float4>& getNormalTex(){ return *normalTex_;}
        inline cv::cuda::GpuMat& getRgbImg(){return rgbImgGpu_;}
        inline cv::cuda::GpuMat& getDepthImg(){return depthImgGpu_;}
        inline cv::cuda::GpuMat& getNormalImg(){return normalImgGpu_;}
        inline const int getImgWidth(){return rgbImgGpu_.cols;}
        inline const int getImgHeight(){return rgbImgGpu_.rows;}
        
    protected:
        cv::cuda::GpuMat rgbImgGpu_, depthImgGpu_, normalImgGpu_;
        Pose3D cameraPose_;
        CameraParameters cameraParams_;
        std::shared_ptr<Texture<float>> depthTex_;
        std::shared_ptr<Texture<uchar4>> colorTex_;
        std::shared_ptr<Texture<float4>> normalTex_;

    }; // class GaussianSplattingFeyframe

}; // namespace gaussian_splatting_slam