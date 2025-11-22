#pragma once

#include <gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp>

#define GSS_BLOCK_X 16
#define GSS_BLOCK_Y 16
#define GSS_BLOCK_SIZE (GSS_BLOCK_X * GSS_BLOCK_Y)

namespace gaussian_splatting_slam
{
    __global__ void rasterizeGaussian(float3 *rgb,
                                      float *depth,
                                      Gaussian3D gaussian,
                                      Pose3D cameraPose,
                                      CameraParameters cameraParams,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t rgbStep,
                                      uint32_t depthStep);

    __global__ void generateGaussians_kernel(float3 *positions,
                                             float3 *scales,
                                             float4 *orientations,
                                             float3 *colors,
                                             float *alphas,
                                             uint32_t *instanceCounter,
                                             cudaTextureObject_t texRGBA,
                                             cudaTextureObject_t texDepth,
                                             cudaTextureObject_t texNormal,
                                             const Pose3D cameraPose,
                                             const CameraParameters cameraParams,
                                             uint32_t sample_dx,
                                             uint32_t sample_dy,
                                             uint32_t width,
                                             uint32_t height);

    __global__ void computeScreenSpaceParamsAndHashes_kernel(
        uint32_t *indices,
        uint64_t *hashes,
        uint32_t *instanceCounterPtr,
        float3 *imgPositions,
        float3 *imgInvSigmas,
        float2 *pHats,
        float3 *normals,
        const float3 *positions,
        const float3 *scales,
        const float4 *orientations,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float minDepth,
        uint2 tileSize,
        uint2 numTiles,
        uint32_t nbGaussians,
        uint32_t width,
        uint32_t height);

    __global__ void computeIndicesRanges_kernel(
        uint2 *ranges,
        const uint64_t *hashes,
        uint32_t nbInstances);

    __global__ void rasterizeGaussians_kernel(
        float3 *rgbData,
        float *depthData,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgInvSigmas,
        const float2 *pHats,
        const float3 *colors,
        const float *alphas,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep,
        uint32_t depthStep);

    __global__ void rasterizeGaussians_kernel(
        uchar4 *rgbData,
        float *depthData,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgInvSigmas,
        const float2 *pHats,
        const float3 *colors,
        const float *alphas,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep,
        uint32_t depthStep);

    __global__ void rasterizeGaussiansNormals_kernel(
        float3 *normalsData,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgInvSigmas,
        const float *alphas,
        const float3 *normals,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t normalsStep);

    __global__ void computeDensityMask_kernel(
        float *maskData,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgInvSigmas,
        const float2 *pHats,
        const float3 *colors,
        const float *alphas,
        // cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t maskStep);

    __global__ void optimizeGaussians_kernel(
        DeltaGaussian2D *deltaGaussians,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgSigmas,
        const float3 *imgInvSigmas,
        const float3 *colors,
        const float *alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizeGaussians2_kernel(
        DeltaGaussian2D *deltaGaussians,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgSigmas,
        const float3 *imgInvSigmas,
        const float2 *pHats,
        const float3 *colors,
        const float *alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        float w_depth,
        float w_dist,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void perTileBucketCount(
        uint32_t *bucketCount,
        const uint2 *ranges,
        int numTiles);

    __global__ void optimizeGaussiansForwardPass(
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgSigmas,
        const float3 *imgInvSigmas,
        const float2 *pHats,
        const float3 *colors,
        const float *alphas,
        const uint32_t *bucketOffsets,
        uint32_t *bucketToTile,
        float *sampled_T,
        float3 *sampled_ar,
        float *final_T,
        uint32_t *n_contrib,
        uint32_t *max_contrib,
        float3 *output_color,
        float *output_depth,
        float3 *color_error,
        float *depth_error,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizeGaussiansPerGaussianPass(
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        const uint32_t *__restrict__ bucketOffsets,
        const uint32_t *__restrict__ bucketToTile,
        const float *__restrict__ sampled_T,
        const float3 *__restrict__ sampled_ar,
        const float *__restrict__ final_T,
        const uint32_t *__restrict__ n_contrib,
        const uint32_t *__restrict__ max_contrib,
        const float3 *__restrict__ output_color,
        const float *__restrict__ output_depth,
        const float3 *__restrict__ color_error,
        const float *__restrict__ depth_error,
        DeltaGaussian2D *__restrict__ deltaGaussians,
        float3 bgColor,
        float w_depth,
        float w_dist,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t nbBuckets);

    __global__ void applyDeltaGaussians_kernel(
        float3 *positions,
        float3 *scales,
        float4 *orientations,
        float3 *colors,
        float *alphas,
        const DeltaGaussian2D *deltaGaussians,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        float lambda_iso,
        uint32_t nbGaussians);

    __global__ void applyDeltaGaussians2_kernel(
        float3 *positions,
        float3 *scales,
        float4 *orientations,
        float3 *colors,
        float *alphas,
        const DeltaGaussian2D *deltaGaussians,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        float lambda_iso,
        uint32_t nbGaussians);

    __global__ void computeDeltaGaussians3D_kernel(
        DeltaGaussian3D *__restrict__ deltaGaussians3D,
        const float3 *__restrict__ positions,
        const float3 *__restrict__ scales,
        const float4 *__restrict__ orientations,
        const float3 *__restrict__ colors,
        float *__restrict__ alphas,
        const DeltaGaussian2D *__restrict__ deltaGaussians2D,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float lambda_iso,
        uint32_t nbGaussians);

    __global__ void updateGaussiansParametersAdam_kernel(
        float3 *__restrict__ positions,
        float3 *__restrict__ scales,
        float4 *__restrict__ orientations,
        float3 *__restrict__ colors,
        float *__restrict__ alphas,
        AdamStateGaussian3D *__restrict__ adamStates,
        const DeltaGaussian3D *__restrict__ deltaGaussians3D,
        const float eta,
        const float beta1,
        const float beta2,
        const float epsilon,
        int nbGaussians);

    __global__ void optimizePose_kernel(
        DeltaGaussian2D *deltaGaussians,
        const uint2 *ranges,
        const uint32_t *indices,
        const float3 *imgPositions,
        const float3 *imgSigmas,
        const float3 *imgInvSigmas,
        const float3 *colors,
        const float *alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePose2_kernel(
        DeltaPose3D *__restrict__ deltaPose,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePose3_kernel(
        DeltaPose3D *__restrict__ deltaPose,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePoseGN_kernel(
        MotionTrackingData *__restrict__ mtd,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePoseGN2_kernel(
        MotionTrackingData *__restrict__ mtd,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePoseGN3_kernel(
        MotionTrackingData *__restrict__ mtd,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        cudaTextureObject_t texDx,
        cudaTextureObject_t texD,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePoseGN3_fast_kernel(
        MotionTrackingData *__restrict__ mtd,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        cudaTextureObject_t texDx,
        cudaTextureObject_t texD,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void optimizePoseGN_warping_kernel(
        MotionTrackingData *__restrict__ mtd,
        cudaTextureObject_t texRGBA1,
        cudaTextureObject_t texDepth1,
        cudaTextureObject_t texRGBA2,
        cudaTextureObject_t texDepth2,
        cudaTextureObject_t texDx,
        cudaTextureObject_t texDy,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float w_depth,
        float colorThresh,
        float depthThresh,
        uint32_t width,
        uint32_t height);

    __global__ void applyDeltaPose_kernel(
        DeltaPose3D *__restrict__ deltaPose,
        const float3 *__restrict__ positions,
        const float3 *__restrict__ scales,
        const float4 *__restrict__ orientations,
        const DeltaGaussian2D *__restrict__ deltaGaussians,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        uint32_t nbGaussians);

    __global__ void densifyGaussians_kernel(float3 *__restrict__ positions,
                                            float3 *__restrict__ scales,
                                            float4 *__restrict__ orientations,
                                            float3 *__restrict__ colors,
                                            float *__restrict__ alphas,
                                            uint32_t *__restrict__ instanceCounter,
                                            cudaTextureObject_t texRGBA,
                                            cudaTextureObject_t texDepth,
                                            cudaTextureObject_t texNormal,
                                            cudaTextureObject_t texMask,
                                            const Pose3D cameraPose,
                                            const CameraParameters cameraParams,
                                            uint32_t sample_dx,
                                            uint32_t sample_dy,
                                            uint32_t width,
                                            uint32_t height);

    __global__ void computePosGradVariance_kernel(PosGradVariance *data,
                                                  const uint2 *__restrict__ ranges,
                                                  const uint32_t *__restrict__ indices,
                                                  const float3 *__restrict__ imgPositions,
                                                  const float3 *__restrict__ imgInvSigmas,
                                                  const float2 *__restrict__ pHats,
                                                  const float3 *__restrict__ colors,
                                                  const float *__restrict__ alphas,
                                                  cudaTextureObject_t texRGBA,
                                                  cudaTextureObject_t texDepth,
                                                  float w_depth,
                                                  float w_dist,
                                                  float3 bgColor,
                                                  uint2 numTiles,
                                                  uint32_t width,
                                                  uint32_t height);

    __global__ void splitGaussians_kernel(const PosGradVariance *__restrict__ data,
                                          float3 *__restrict__ positions,
                                          float3 *__restrict__ scales,
                                          float4 *__restrict__ orientations,
                                          float3 *__restrict__ colors,
                                          float *__restrict__ alphas,
                                          uint32_t *__restrict__ instanceCounter,
                                          const Pose3D cameraPose,
                                          const CameraParameters cameraParams,
                                          float varThresh,
                                          uint32_t nbGaussians,
                                          uint32_t nbMaxGaussians);

    __global__ void pruneGaussians_kernel(unsigned int *__restrict__ nbRemoved,
                                          unsigned char *__restrict__ states,
                                          const float3 *__restrict__ scales,
                                          const float *__restrict__ alphas,
                                          float alphaThreshold,
                                          float scaleRatioThreshold,
                                          uint32_t nbGaussians);

    __global__ void computeGaussiansVisibility_kernel(
        unsigned char *__restrict__ visibilities,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float *__restrict__ alphas,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void computeGaussiansCovisibility_kernel(
        uint32_t *__restrict__ visibilityInter,
        uint32_t *__restrict__ visibilityUnion,
        unsigned char *__restrict__ visibilities1,
        unsigned char *__restrict__ visibilities2,
        uint32_t nbGaussians);

    __global__ void computeOutliers_kernel(
        float *__restrict__ outlierProb,
        float *__restrict__ totalAlpha,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        // const float *__restrict__ alphas,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height);

    __global__ void removeOutliers_kernel(
        unsigned int *__restrict__ nbRemoved,
        unsigned char *__restrict__ states,
        const float *__restrict__ outliersProb,
        const float *__restrict__ totalAlpha,
        float threshold,
        uint32_t nbGaussians);

    __global__ void computeNormalsFromDepth_kernel(
        float4 *normalsData,
        cudaTextureObject_t texDepth,
        const CameraParameters cameraParams,
        uint32_t width,
        uint32_t height,
        uint32_t normalsStep);

    __global__ void rasterizeGaussiansFill_kernel(
        float3 *__restrict__ rgbData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep);
    
    __global__ void rasterizeGaussiansBlobs_kernel(
        float3 *__restrict__ rgbData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        const float3 *__restrict__ positions,
        const float3 *__restrict__ scales,
        const float4 *__restrict__ orientations,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        const float3 lightDirection,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep);

    __global__ void rasterizeGaussiansError_kernel(
        float3 *__restrict__ rgbData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float *__restrict__ alphas,
        const PosGradVariance *__restrict__ posGrad,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep);

    __global__ void exportPLYGaussians_kernel(
        float *__restrict__ buffer,
        const float3 *__restrict__ positions,
        const float3 *__restrict__ scales,
        const float4 *__restrict__ orientations,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        uint32_t nbGaussians);
}
