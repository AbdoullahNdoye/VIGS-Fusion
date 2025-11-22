#pragma once

#include <cstdint>
#include <vector_types.h>
#include <thrust/device_vector.h>
#include <Eigen/Dense>
#include <cuda_utils/vector_math.cuh>

namespace gaussian_splatting_slam
{

    struct Gaussians
    {
        thrust::device_vector<float3> positions;    // gaussian mean
        thrust::device_vector<float3> scales;       // gaussian scaling parameters
        thrust::device_vector<float4> orientations; // quaternion
        thrust::device_vector<float3> colors;       // color
        thrust::device_vector<float> alphas;        // opacity

        thrust::device_vector<float3> imgPositions; // (u, v, depth)
        thrust::device_vector<float3> imgSigmas;    // (uu, uv, vv)
        thrust::device_vector<float3> imgInvSigmas; // (uu, uv, vv)
        thrust::device_vector<float2> pHats;
        thrust::device_vector<float3> normals;

        void resize(size_t len);

        typedef thrust::tuple<thrust::device_vector<float3>::iterator,
                              thrust::device_vector<float3>::iterator,
                              thrust::device_vector<float4>::iterator,
                              thrust::device_vector<float3>::iterator,
                              thrust::device_vector<float>::iterator>
            Tuple;

        typedef thrust::zip_iterator<Tuple> iterator;

        iterator begin()
        {
            return thrust::make_zip_iterator(thrust::make_tuple(positions.begin(),
                                                                scales.begin(),
                                                                orientations.begin(),
                                                                colors.begin(),
                                                                alphas.begin()));
        }
        iterator end()
        {
            return thrust::make_zip_iterator(thrust::make_tuple(positions.end(),
                                                                scales.end(),
                                                                orientations.end(),
                                                                colors.end(),
                                                                alphas.end()));
        }
    };

    struct Gaussian3D
    {
        float3 position;    // gaussian mean
        float3 scale;       // gaussian scaling parameters
        float4 orientation; // quaternion
        float3 color;       // color
        float alpha;        // opacity
    };

    struct SplattedGaussian
    {
        float3 position; // gaussian mean (u,v,depth)
        // float3 sigma;    // img covariance
        float3 invSigma; // inverse covariance
        float3 color;    // color
        //float3 normal; // normal (in camera frame)
        float alpha;     // opacity
        float2 pHat;
        float3 normal;
    };

    struct DeltaGaussian2D
    {
        // float3 position;
        // float3 scale;
        // float3 angles;
        float2 meanImg;
        float3 invSigmaImg;
        float3 color;
        float depth;
        float alpha;
        float2 pHat;
        unsigned int n;

        __device__ __host__ inline DeltaGaussian2D &operator+=(const DeltaGaussian2D &d)
        {
            // position.x += d.position.x;
            // position.y += d.position.y;
            // position.z += d.position.z;

            meanImg.x += d.meanImg.x;
            meanImg.y += d.meanImg.y;

            invSigmaImg.x += d.invSigmaImg.x;
            invSigmaImg.y += d.invSigmaImg.y;
            invSigmaImg.z += d.invSigmaImg.z;

            color.x += d.color.x;
            color.y += d.color.y;
            color.z += d.color.z;

            depth += d.depth;

            alpha += d.alpha;

            pHat.x += d.pHat.x;
            pHat.y += d.pHat.y;

            n += d.n;

            return *this;
        }
    };

    struct DeltaGaussian3D
    {
        float3 position;
        float3 scale;
        float3 orientation;
        float3 color;
        float alpha;
        int n;
    };

    struct AdamStateGaussian3D
    {
        float3 m_position;
        float3 v_position;
        float3 m_scale;
        float3 v_scale;
        float3 m_orientation;
        float3 v_orientation;
        float3 m_color;
        float3 v_color;
        float m_alpha;
        float v_alpha;
        float t;
    };

    struct Pose3D
    {
        float3 position;
        float4 orientation;
    };

    struct JTJ_JTR_DATA
    {
        Eigen::Matrix<float, 6, 6> JTJ;
        Eigen::Vector<float, 6> JTr;
    };

    struct DeltaPose3D
    {
        float3 dp;
        float3 dq;
        unsigned int n;
        __device__ __host__ inline DeltaPose3D &operator+=(const DeltaPose3D &d)
        {
            dp.x += d.dp.x;
            dp.y += d.dp.y;
            dp.z += d.dp.z;

            dq.x += d.dq.x;
            dq.y += d.dq.y;
            dq.z += d.dq.z;

            n += d.n;

            return *this;
        }
    };

    struct PosGradVariance
    {
        float w, dx, dy, dx2, dy2, dxdy;
    };

    struct MotionTrackingData
    {
        float JtJ[21];
        //float JtJ[36];
        float Jtr[6];
        //int n;

        __device__ __host__ inline MotionTrackingData &operator+=(const MotionTrackingData &m)
        {
            #pragma unroll
            for(int i=0; i<21; i++)
            {
                JtJ[i]+=m.JtJ[i];
            }
            #pragma unroll
            for(int i=0; i<6; i++)
            {
                Jtr[i]+=m.Jtr[i];
            }
            //n+=m.n;
            return *this;
        }
    };

    struct CameraParameters
    {
        float2 f;
        float2 c;
    };

    typedef enum PoseEstimationMethod
    {
        PoseEstimationMethodFull = 0,
        PoseEstimationMethodWarpingSingleRendering=1,
        PoseEstimationMethodWarpingMultipleRendering=2,
    } PoseEstimationMethod;
} // namespace gaussian_splatting_slam
