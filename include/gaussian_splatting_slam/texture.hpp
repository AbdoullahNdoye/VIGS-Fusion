#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <opencv2/core/cuda.hpp>

namespace gaussian_splatting_slam
{

    template <typename T>
    class Texture
    {

    private:
        cudaTextureObject_t textureObject;
        cudaTextureFilterMode filterMode;

    public:
        Texture(const cv::cuda::GpuMat &img, cudaTextureFilterMode filterMode_ = cudaFilterModePoint);

        ~Texture();

        inline cudaTextureObject_t &getTextureObject() { return textureObject; }

    }; // class Texture

} // gaussian_splatting_slam

#include <gaussian_splatting_slam/texture_impl.hpp>

