#pragma once
#include <opencv2/core/cuda.hpp>

namespace gaussian_splatting_slam
{

    template <typename T>
    Texture<T>::Texture(const cv::cuda::GpuMat &img, cudaTextureFilterMode filterMode_)
    {
        filterMode = filterMode_;
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypePitch2D;
        resDesc.res.pitch2D.devPtr = img.data;
        resDesc.res.pitch2D.width = img.cols;
        resDesc.res.pitch2D.height = img.rows;
        resDesc.res.pitch2D.pitchInBytes = img.step;
        resDesc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = filterMode;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL);
    }

    template <typename T>
    Texture<T>::~Texture()
    {
        cudaDestroyTextureObject(textureObject);
    }

}; // namespace gaussian_splatting_slam
