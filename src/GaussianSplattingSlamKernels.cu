#include <gaussian_splatting_slam/GaussianSplattingSlamKernels.hpp>
#include <Eigen/Dense>
#include <Eigen/QR>

#include <cuda_utils/vector_math.cuh>
#include <cuda_utils/cuda_utils_dev.cuh>
#include <cuda_utils/reduce_dev.cuh>
#include <cstdio>

#include <cub/cub.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE (GSS_BLOCK_X * GSS_BLOCK_Y)
#define NUM_WARPS (BLOCK_SIZE / 32)

// #define USE_MEAN_DEPTH
#define USE_MEDIAN_DEPTH

namespace gaussian_splatting_slam
{
    inline __device__ void forwardPass(
        float3 &output_color,
        float &output_depth,
        float &final_T,
        uint32_t &n_contrib,
        int x,
        int y,
        uint32_t tile_id,
        SplattedGaussian *__restrict__ splattedGaussian_sh,
        uint32_t *__restrict__ gids_sh,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        const float3 &bgColor,
        bool inside)
    {
        auto block = cg::this_thread_block();
        int tid = block.thread_rank();
        //__shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        //__shared__ uint32_t gids_sh[BLOCK_SIZE];
        uint2 range = ranges[tile_id];
        int n = range.y - range.x;
        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);
        float depth = 0.f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;
        float T = 1.f;
        bool done = !inside;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                contributor++;
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f || v <= 0.f)
                    continue;
                float test_T = T * (1.f - alpha_i);
                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }
                color += splattedGaussian_sh[i].color * alpha_i * T;

                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                if (T > 0.5f && test_T < 0.5f)
                {
                    depth = d;
                }

                T = test_T;
                last_contributor = contributor;
            }
        }

        if (inside)
        {
            final_T = T;
            n_contrib = last_contributor;
            color += T * bgColor;
            output_color = color;
            output_depth = depth;
        }
    }

    inline __device__ void forwardPassDepthOnly(
        float &output_depth,
        float &final_T,
        uint32_t &n_contrib,
        int x,
        int y,
        uint32_t tile_id,
        SplattedGaussian *__restrict__ splattedGaussian_sh,
        uint32_t *__restrict__ gids_sh,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float *__restrict__ alphas,
        bool inside)
    {
        auto block = cg::this_thread_block();
        int tid = block.thread_rank();
        uint2 range = ranges[tile_id];
        int n = range.y - range.x;
        float prod_alpha = 1.f;
        float depth = 0.f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;
        float T = 1.f;
        bool done = !inside;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                contributor++;
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f || v <= 0.f)
                    continue;
                float test_T = T * (1 - alpha_i);
                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }

                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                if (T > 0.5f && test_T < 0.5f)
                {
                    depth = d;
                }

                T = test_T;
                last_contributor = contributor;
            }
        }

        if (inside)
        {
            final_T = T;
            n_contrib = last_contributor;
            output_depth = depth;
        }
    }

    __global__ void rasterizeGaussian(float3 *rgb,
                                      float *depth,
                                      Gaussian3D gaussian,
                                      Pose3D cameraPose,
                                      CameraParameters cameraParams,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t rgbStep,
                                      uint32_t depthStep)
    {
        int u = blockIdx.x * blockDim.x + threadIdx.x;
        int v = blockIdx.y * blockDim.y + threadIdx.y;
        if (u >= width || v >= height)
            return;

        Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&gaussian.position);
        Eigen::Map<const Eigen::Quaternionf> q_gauss((float *)&gaussian.orientation);
        Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&gaussian.scale);

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);

        Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);

        Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                                     {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

        Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
        Eigen::Matrix3f R = q_gauss.toRotationMatrix();

        Eigen::Matrix<float, 2, 3> M = J * W * R * s_gauss.asDiagonal();
        // Eigen::Matrix2f Sigma = Eigen::Matrix2f::Zero();
        // Sigma.selfadjointView<Eigen::Lower>().rankUpdate(J * W * R * s_gauss.asDiagonal());

        Eigen::Vector2f mu_img(cameraParams.f.x * mu_cam.x() / mu_cam.z() + cameraParams.c.x,
                               cameraParams.f.y * mu_cam.y() / mu_cam.z() + cameraParams.c.y);

        Eigen::Vector2f uv(u, v);
        Eigen::Vector2f dx(mu_img - uv);

        Eigen::Matrix2f Sigma = M * M.transpose();
        Eigen::Matrix2f Sigma_i = Sigma.inverse();

        // Eigen::Vector3f Mdx = M.transpose()*dx;

        if (u == 0 && v == 0)
        {
            printf("M :\n%f %f %f\n%f %f %f\n", M(0, 0), M(0, 1), M(0, 2), M(1, 0), M(1, 1), M(1, 2));
            printf("Sigma: \n%f %f\n%f %f\n", Sigma(0, 0), Sigma(0, 1), Sigma(1, 0), Sigma(1, 1));
        }

        // float alpha = gaussian.alpha * expf(-0.5f * Mdx.dot(Mdx));
        float alpha = gaussian.alpha * expf(-0.5f * (dx.transpose() * Sigma_i * dx).value());

        // float alpha = dx.norm();
        rgb[v * rgbStep / sizeof(float3) + u] = alpha * gaussian.color;
        depth[v * depthStep / sizeof(float) + u] = alpha * mu_cam.z();
    }

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
                                             uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        float u = (x + 0.5f) * sample_dx;
        float v = (y + 0.5f) * sample_dy;
        // float u = x * sample_dx + sample_dx / 2;
        // float v = y * sample_dy + sample_dy / 2;

        if (u >= width || v >= height)
        {
            return;
        }

        float depth = tex2D<float>(texDepth, u, v);
        // printf("%f %f %f\n", u, v, depth);
        if (depth < 0.5f)
        {
            // depth = 1.f;
            return;
        }

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);

        uint32_t idx = atomicAggInc(instanceCounter);

        Eigen::Vector3f pos_cam(depth * (u - cameraParams.c.x) / cameraParams.f.x,
                                depth * (v - cameraParams.c.y) / cameraParams.f.y,
                                depth);

        Eigen::Map<Eigen::Vector3f> position((float *)&positions[idx]);
        position = p_cam + q_cam * pos_cam;
        // positions[idx] = pos;

        float scale_x = 0.8f * depth * sample_dx / cameraParams.f.x;
        float scale_y = 0.8f * depth * sample_dy / cameraParams.f.y;

        float3 scale = make_float3(0.5f * (scale_x + scale_y),
                                   0.5f * (scale_x + scale_y),
                                   0.1f * (scale_x + scale_y));
        // float3 scale = make_float3(0.05f, 0.05f, 0.01f);

        scales[idx] = scale;

        float4 n = tex2D<float4>(texNormal, u, v);
        uchar4 rgba = tex2D<uchar4>(texRGBA, u, v);

        Eigen::Quaternionf q;

        // printf("n : %f %f %f\n", n.x, n.y, n.z);

        Eigen::Vector3f u0(0.f, 0.f, 1.f);
        Eigen::Vector3f u1(n.x, n.y, n.z);

        if (u1.z() < 0.f)
        {
            u1 = -u1;
        }
        q = Eigen::Quaternionf::FromTwoVectors(u0, u1);
        // q = Eigen::Quaternionf::FromTwoVectors(u1, u0);

        /*
        if(n.w==0.f)
        {
            q = Eigen::Quaternionf(1.f, 0.f, 0.f, 0.f);
        }else{
            //q = Eigen::Quaternionf(1.f, 0.f, 0.f, 0.f);
            if()
            q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0.f, 0.f, 1.f), Eigen::Vector3f(n.x, n.y, n.z));
            printf("q : %f %f %f %f\n", q.x(), q.y(), q.z(), q.w());

            // q.setFromTwoVectors(Eigen::Vector3f(n.x, n.y, n.z), Eigen::Vector3f(0.f, 0.f, 1.f));
        }
        */

        q = q_cam * q;
        orientations[idx] = make_float4(q.x(), q.y(), q.z(), q.w());
        // orientations[idx] = make_float4(0.f, 0.f, 0.f, 1.f);

        colors[idx] = make_float3(rgba.x / 255.f,
                                  rgba.y / 255.f,
                                  rgba.z / 255.f);
        alphas[idx] = 1.f;

        // printf("%d\n", idx);
    }

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
        uint32_t height)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);

        Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);

        if (mu_cam.z() < minDepth)
        {
            // printf("mucam.z < minDepth\n");
            return;
        }

        Eigen::Map<const Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
        Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);

        Eigen::Matrix<float, 3, 3> J;
        J << cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z()),
            0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z()),
            0.f, 0.f, 1.f;

        Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
        Eigen::Matrix3f R = q_gauss.toRotationMatrix();

        Eigen::Matrix<float, 3, 3> M = J * W * R * s_gauss.asDiagonal();

        Eigen::Matrix3f Sigma = M * M.transpose();
        Eigen::Matrix3f Sigma_i = Sigma.inverse();

        Eigen::Matrix2f Sigma2D_i = Sigma.topLeftCorner<2, 2>().inverse();

        /*if (isnan(Sigma_i(0, 0)) || isnan(Sigma_i(0, 1)) || isnan(Sigma_i(1, 1)))
        {
            //printf("Sigma_i is nan\n");
            printf("Sigma : %f %f %f %f\n", Sigma(0, 0), Sigma(0, 1), Sigma(1, 0), Sigma(1, 1));
            printf("scale : %f %f %f\n", s_gauss.x(), s_gauss.y(), s_gauss.z());
            printf("orientation : %f %f %f %f\n", q_gauss.x(), q_gauss.y(), q_gauss.z(), q_gauss.w());
        }*/
        /*printf("Sigma_i : %f %f %f %f %f %f\n",
               Sigma_i(0, 0), Sigma_i(0, 1), Sigma_i(0, 2),
               Sigma_i(1, 1), Sigma_i(1, 2), Sigma_i(2, 2));
        printf("Sigma2_i : %f %f %f\n",
               Sigma2D_i(0, 0), Sigma2D_i(0, 1), Sigma2D_i(1, 1));*/

        Eigen::Vector2f mu_img(cameraParams.f.x * mu_cam.x() / mu_cam.z() + cameraParams.c.x,
                               cameraParams.f.y * mu_cam.y() / mu_cam.z() + cameraParams.c.y);

        if (mu_img.x() < 0.f || mu_img.x() > width - 1 || mu_img.y() < 0.f || mu_img.y() > height - 1)
        {
            return;
        }

        Eigen::Vector2f img_gauss_size = 3.f * Sigma.topLeftCorner<2, 2>().diagonal().cwiseSqrt();

        int tile_min_x = (mu_img.x() - img_gauss_size.x()) / tileSize.x;
        int tile_min_y = (mu_img.y() - img_gauss_size.y()) / tileSize.y;

        int tile_max_x = (mu_img.x() + img_gauss_size.x()) / tileSize.x;
        int tile_max_y = (mu_img.y() + img_gauss_size.y()) / tileSize.y;

        // int num_tiles_x = (width + tile_width - 1) / tile_width;
        // int num_tiles_y = (height + tile_height - 1) / tile_height;

        // if(Sigma(0,0) > 1000.f || Sigma(1,1) > 1000.f)
        // {
        //     //printf("very large projected Gaussian : %f %f %f (%f %f %f)\n", Sigma(0,0), Sigma(0,1), Sigma(1,1), s_gauss.x(), s_gauss.y(), s_gauss.z());
        //     return;
        // }

        if (tile_max_x < 0 || tile_max_y < 0 || tile_min_x >= (int)numTiles.x || tile_min_y >= (int)numTiles.y)
        {
            /*
            printf("gaussian totaly outside %d %d %d %d \n", tile_min_x, tile_min_y, tile_max_x, tile_max_y);
            printf("numTiles : %d %d\n", numTiles.x, numTiles.y);
            */
            return;
        }

        tile_min_x = max(0, tile_min_x);
        tile_min_y = max(0, tile_min_y);

        tile_max_x = min((int)numTiles.x - 1, tile_max_x);
        tile_max_y = min((int)numTiles.y - 1, tile_max_y);

        Eigen::Vector3f pHat = Sigma_i.col(2);
        pHat *= mu_cam.z() / (mu_cam.norm() * pHat.z());

        Eigen::Vector3f normal(Sigma_i(2, 0) / Sigma_i(2, 2), Sigma_i(2, 1) / Sigma_i(2, 2), 1.f);
        normal = -J.transpose() * normal;

        normal /= normal.norm();

        imgPositions[idx] = make_float3(mu_img.x(), mu_img.y(), mu_cam.z());
        // imgSigmas[idx] = make_float3(Sigma(0, 0), Sigma(0, 1), Sigma(1, 1));
        imgInvSigmas[idx] = make_float3(Sigma2D_i(0, 0), Sigma2D_i(0, 1), Sigma2D_i(1, 1));
        pHats[idx] = make_float2(pHat.x(), pHat.y());
        normals[idx] = make_float3(normal.x(), normal.y(), normal.z());

        // printf("pHat : %f %f\n", pHat.x(), pHat.y());

        for (int j = tile_min_y; j <= tile_max_y; j++)
        {
            for (int i = tile_min_x; i <= tile_max_x; i++)
            {
                uint32_t instance = atomicAggInc(instanceCounterPtr);
                indices[instance] = idx;
                uint64_t hash = (uint64_t)(__float_as_uint(mu_cam.z())) | ((uint64_t)(j * numTiles.x + i) << 32);
                hashes[instance] = hash;
            }
        }
    }

    __global__ void computeIndicesRanges_kernel(
        uint2 *ranges,
        const uint64_t *hashes,
        uint32_t nbInstances)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= nbInstances)
            return;

        if (idx == 0)
        {
            uint32_t hash = hashes[idx] >> 32;
            ranges[hash].x = 0;
            return;
        }

        // printf("%lu\n", hashes[idx]);

        uint32_t prev_hash = hashes[idx - 1] >> 32;
        uint32_t hash = hashes[idx] >> 32;

        if (prev_hash != hash)
        {
            ranges[prev_hash].y = idx;
            ranges[hash].x = idx;

            // printf("%d %d\n", hash, idx);
        }

        if (idx == nbInstances - 1)
        {
            ranges[hash].y = idx + 1;
        }
    }

    //     __global__ void rasterizeGaussians_kernel(
    //         float3 *__restrict__ rgbData,
    //         float *__restrict__ depthData,
    //         const uint2 *__restrict__ ranges,
    //         const uint32_t *__restrict__ indices,
    //         const float3 *__restrict__ imgPositions,
    //         const float3 *__restrict__ imgInvSigmas,
    //         const float2 *__restrict__ pHats,
    //         const float3 *__restrict__ colors,
    //         const float *__restrict__ alphas,
    //         float3 bgColor,
    //         uint2 numTiles,
    //         uint32_t width,
    //         uint32_t height,
    //         uint32_t rgbStep,
    //         uint32_t depthStep)
    //     {
    //         int x = blockIdx.x * blockDim.x + threadIdx.x;
    //         int y = blockIdx.y * blockDim.y + threadIdx.y;

    //         auto block = cg::this_thread_block();

    //         __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
    //         //__shared__ uint32_t gids[BLOCK_SIZE];

    //         int tileId = blockIdx.y * numTiles.x + blockIdx.x;
    //         uint2 range = ranges[tileId];

    //         bool inside = x < width && y < height;

    //         float prod_alpha = 1.f;
    //         float3 color = make_float3(0.f);

    //         float depth = 0.f;
    //         float T = 1.f;

    //         // if(threadIdx.x==0 && threadIdx.y==0)
    //         // {
    //         //     printf("ranges[%d] : %u %u\n", tileId, range.x, range.y);
    //         // }

    //         int n = min(range.y - range.x, BLOCK_SIZE);

    //         // collect gaussians data
    //         if (block.thread_rank() < n)
    //         {
    //             uint32_t gid = indices[range.x + block.thread_rank()];
    //             // gids[block.thread_rank()] = gid;
    //             splattedGaussians[block.thread_rank()].position = imgPositions[gid];
    //             splattedGaussians[block.thread_rank()].invSigma = imgInvSigmas[gid];
    //             splattedGaussians[block.thread_rank()].color = colors[gid];
    //             splattedGaussians[block.thread_rank()].alpha = alphas[gid];
    //             splattedGaussians[block.thread_rank()].pHat = pHats[gid];
    //         }
    //         block.sync();

    //         if (inside)
    //         {

    //             for (int i = 0; i < n; i++)
    //             {
    //                 float dx = splattedGaussians[i].position.x - x;
    //                 float dy = splattedGaussians[i].position.y - y;
    //                 float v = splattedGaussians[i].invSigma.x * dx * dx + 2.f * splattedGaussians[i].invSigma.y * dx * dy + splattedGaussians[i].invSigma.z * dy * dy;

    //                 float alpha_i = min(0.99f, splattedGaussians[i].alpha * expf(-0.5f * v));

    //                 if (alpha_i < 1.f / 255.f)
    //                     continue;

    //                 color += splattedGaussians[i].color * alpha_i * prod_alpha;
    // #ifdef USE_MEAN_DEPTH
    //                 // mean depth
    //                 depth += (splattedGaussians[i].position.z + dx * splattedGaussians[i].pHat.x + dy * splattedGaussians[i].pHat.y) * alpha_i * prod_alpha;
    // #endif
    //                 prod_alpha *= (1.f - alpha_i);

    // #ifdef USE_MEDIAN_DEPTH
    //                 // median depth
    //                 if (T > 0.5f && prod_alpha < 0.5)
    //                 {
    //                     T = prod_alpha;
    //                     depth = splattedGaussians[i].position.z + dx * splattedGaussians[i].pHat.x + dy * splattedGaussians[i].pHat.y;
    //                 }
    // #endif
    //                 if (prod_alpha < 0.001f)
    //                 {
    //                     break;
    //                 }
    //             }

    //             prod_alpha = max(0.f, min(1.f, prod_alpha));
    //             color += prod_alpha * bgColor;

    //             float3 *rgb_row = (float3 *)&((unsigned char *)rgbData)[y * rgbStep];
    //             rgb_row[x] = color;
    //             // rgbData[y * (rgbStep / sizeof(float3)) + x] = color;

    //             depthData[y * (depthStep / sizeof(float)) + x] = depth;
    //         }
    //     }

    __global__ void rasterizeGaussians_kernel(
        float3 *__restrict__ rgbData,
        float *__restrict__ depthData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep,
        uint32_t depthStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;

        bool inside = x < width && y < height;

        float3 color;
        float depth;
        float final_T;
        uint32_t n_contrib;

        forwardPass(
            color,
            depth,
            final_T,
            n_contrib,
            x,
            y,
            tileId,
            splattedGaussian_sh,
            gids_sh,
            ranges,
            indices,
            imgPositions,
            imgInvSigmas,
            pHats,
            colors,
            alphas,
            bgColor,
            inside);

        if (inside)
        {
            float3 *rgb_row = (float3 *)&((unsigned char *)rgbData)[y * rgbStep];
            rgb_row[x] = color;
            // rgbData[y * (rgbStep / sizeof(float3)) + x] = color;
            depthData[y * (depthStep / sizeof(float)) + x] = depth;
        }
    }

    __global__ void rasterizeGaussians_kernel(
        uchar4 *__restrict__ rgbData,
        float *__restrict__ depthData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t rgbStep,
        uint32_t depthStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;

        bool inside = x < width && y < height;

        float3 color;
        float depth;
        float final_T;
        uint32_t n_contrib;

        forwardPass(
            color,
            depth,
            final_T,
            n_contrib,
            x,
            y,
            tileId,
            splattedGaussian_sh,
            gids_sh,
            ranges,
            indices,
            imgPositions,
            imgInvSigmas,
            pHats,
            colors,
            alphas,
            bgColor,
            inside);

        if (inside)
        {
            uchar4 *rgb_row = (uchar4 *)&((unsigned char *)rgbData)[y * rgbStep];
            rgb_row[x].x = 255 * color.x;
            rgb_row[x].y = 255 * color.y;
            rgb_row[x].z = 255 * color.z;

            depthData[y * (depthStep / sizeof(float)) + x] = final_T > 0.1f ? 0.f : depth;
        }
    }

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
        uint32_t normalsStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();

        __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
        //__shared__ uint32_t gids[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        bool inside = x < width && y < height;

        float prod_alpha = 1.f;
        float T = 1.f;

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (block.thread_rank() < n)
        {
            uint32_t gid = indices[range.x + block.thread_rank()];
            // gids[block.thread_rank()] = gid;
            splattedGaussians[block.thread_rank()].position = imgPositions[gid];
            splattedGaussians[block.thread_rank()].invSigma = imgInvSigmas[gid];
            splattedGaussians[block.thread_rank()].alpha = alphas[gid];
            splattedGaussians[block.thread_rank()].normal = normals[gid];
        }
        block.sync();

        if (inside)
        {
            float3 normal = {0.f, 0.f, 0.f};

            for (int i = 0; i < n; i++)
            {
                float dx = splattedGaussians[i].position.x - x;
                float dy = splattedGaussians[i].position.y - y;
                float v = splattedGaussians[i].invSigma.x * dx * dx + 2.f * splattedGaussians[i].invSigma.y * dx * dy + splattedGaussians[i].invSigma.z * dy * dy;

                float alpha_i = min(0.99f, splattedGaussians[i].alpha * expf(-0.5f * v));

                if (alpha_i < 1.f / 255.f)
                    continue;

#ifdef USE_MEAN_DEPTH
                // mean depth
                // depth += (splattedGaussians[i].position.z + dx * splattedGaussians[i].pHat.x + dy * splattedGaussians[i].pHat.y) * alpha_i * prod_alpha;
#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha < 0.5)
                {
                    T = prod_alpha;
                    normal = splattedGaussians[i].normal;
                }
#endif
                if (prod_alpha < 0.001f)
                {
                    break;
                }
            }

            float3 *normals_row = (float3 *)&((unsigned char *)normalsData)[y * normalsStep];
            normals_row[x] = normal;
        }
    }

    __global__ void optimizeGaussians_kernel(
        DeltaGaussian2D *__restrict__ deltaGaussians,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ DeltaGaussian2D deltaGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        bool inside = x < width && y < height;

        int n = min(range.y - range.x, BLOCK_SIZE);

        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);
        float depth = 0.f;
        float T = 1.f;

        // collect gaussians data
        if (range.x + tid < range.y)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
        }
        block.sync();

        if (inside)
        {
            for (int i = 0; i < n; i++)
            {
                float dx = x - splattedGaussian_sh[i].position.x;
                float dy = y - splattedGaussian_sh[i].position.y;
                float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f)
                    continue;

                color += splattedGaussian_sh[i].color * alpha_i * prod_alpha;
#ifdef USE_MEAN_DEPTH
                // mean depth
                depth += splattedGaussian_sh[i].position.z * alpha_i * prod_alpha;
#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha < 0.5)
                {
                    T = prod_alpha;
                    depth = splattedGaussian_sh[i].position.z;
                }
#endif
                if (prod_alpha < 0.0001f)
                {
                    break;
                }
            }
        }

        const float final_T = prod_alpha;

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        const float3 color_error = color - make_float3(rgba.x / 255.f,
                                                       rgba.y / 255.f,
                                                       rgba.z / 255.f);

        const float imgDepth = tex2D<float>(texDepth, x, y);

        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        block.sync();

        prod_alpha = 1.f;
        T = 1.f;

        float3 acc_c = make_float3(0.f);
        float alpha_prev = 0.f;
        float3 color_prev = make_float3(0.f);
        // float T = final_T;

        for (int i = 0; i < n; i++)
        {
            const float dx = x - splattedGaussian_sh[i].position.x;
            const float dy = y - splattedGaussian_sh[i].position.y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float G = expf(-0.5f * v);
            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);

            float3 d_alpha = splattedGaussian_sh[i].color * prod_alpha;

            acc_c += splattedGaussian_sh[i].color * alpha_i * prod_alpha;

            d_alpha -= (color - acc_c) / (1.f - alpha_i);

            d_alpha = -color_error * d_alpha;

            const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;
            // const float dl_alpha = d_alpha.x;

            const float G_dl = G * dl_alpha;
            const float a_G_dl = alpha_i * dl_alpha;
            // const float a_G_dl = splattedGaussian_sh[i].alpha * /* alpha_i **/ dl_alpha;

            // float3 dl_alpha_i = G * (splattedGaussian_sh[i].alpha * (1.f - splattedGaussian_sh[i].alpha)) * d_alpha;
            // deltaGaussian_sh[tid].alpha = (dl_alpha_i.x+dl_alpha_i.y+dl_alpha_i.z)/3.f;

            if (inside && alpha_i * prod_alpha > 0.01f)
            {

                deltaGaussian_sh[tid].color = -alpha_i * prod_alpha * color_error;
                deltaGaussian_sh[tid].n = 1.f;

                // deltaGaussian_sh[tid].alpha = G_dl * (splattedGaussian_sh[i].alpha * (1.f - splattedGaussian_sh[i].alpha));
                deltaGaussian_sh[tid].alpha = G_dl;

#ifdef USE_MEAN_DEPTH
                // mean depth
                deltaGaussian_sh[tid].depth = -alpha_i * prod_alpha * depth_error;
#endif

                deltaGaussian_sh[tid].meanImg.x = a_G_dl * (splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy);
                deltaGaussian_sh[tid].meanImg.y = a_G_dl * (splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);
                deltaGaussian_sh[tid].invSigmaImg.x = -0.5f * a_G_dl * dx * dx;
                deltaGaussian_sh[tid].invSigmaImg.y = -0.5f * a_G_dl * dx * dy;
                deltaGaussian_sh[tid].invSigmaImg.z = -0.5f * a_G_dl * dy * dy;

                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha <= 0.5f)
                {
                    T = prod_alpha;
                    deltaGaussian_sh[tid].depth = -depth_error;
                }
                else
                {
                    deltaGaussian_sh[tid].depth = 0.f;
                }
#endif
                /*
                T = T / alpha_i;
                acc_c = alpha_prev * color_prev + (1.f - alpha_prev) * acc_c;
                color_prev = splattedGaussian_sh[i].color;
                float3 d_alpha = (color_prev - acc_c) * color_error;
                float dl_alpha = d_alpha.x + d_alpha.y + d_alpha.z;
                dl_alpha *= T;
                alpha_prev = alpha_i;
                deltaGaussian_sh[tid].alpha = G * dl_alpha * (splattedGaussian_sh[i].alpha * (1.f - splattedGaussian_sh[i].alpha));
                */
            }
            else
            {
                // deltaGaussian_sh[tid].position = make_float3(0.f, 0.f, 0.f);
                // deltaGaussian_sh[tid].scale = make_float3(0.f, 0.f, 0.f);
                // deltaGaussian_sh[tid].angles = make_float3(0.f, 0.f, 0.f);
                deltaGaussian_sh[tid].meanImg = make_float2(0.f, 0.f);
                deltaGaussian_sh[tid].invSigmaImg = make_float3(0.f, 0.f, 0.f);
                deltaGaussian_sh[tid].color = make_float3(0.f, 0.f, 0.f);
                deltaGaussian_sh[tid].depth = 0.f;
                deltaGaussian_sh[tid].alpha = 0.f;
                deltaGaussian_sh[tid].n = 0.f;
            }

            block.sync();

            reduce<BLOCK_SIZE>(deltaGaussian_sh, tid);

            if (tid == 0)
            {
                DeltaGaussian2D &dg = deltaGaussian_sh[0];
                uint32_t gid = gids_sh[i];
                // atomicAdd(&deltaGaussians[gid].position.x, dg.position.x);
                // atomicAdd(&deltaGaussians[gid].position.y, dg.position.y);
                // atomicAdd(&deltaGaussians[gid].position.z, dg.position.z);
                // atomicAdd(&deltaGaussians[gid].scale.x, dg.scale.x);
                // atomicAdd(&deltaGaussians[gid].scale.y, dg.scale.y);
                // atomicAdd(&deltaGaussians[gid].scale.z, dg.scale.z);
                // atomicAdd(&deltaGaussians[gid].angles.x, dg.angles.x);
                // atomicAdd(&deltaGaussians[gid].angles.y, dg.angles.y);
                // atomicAdd(&deltaGaussians[gid].angles.z, dg.angles.z);
                atomicAdd(&deltaGaussians[gid].meanImg.x, dg.meanImg.x);
                atomicAdd(&deltaGaussians[gid].meanImg.y, dg.meanImg.y);
                atomicAdd(&deltaGaussians[gid].invSigmaImg.x, dg.invSigmaImg.x);
                atomicAdd(&deltaGaussians[gid].invSigmaImg.y, dg.invSigmaImg.y);
                atomicAdd(&deltaGaussians[gid].invSigmaImg.z, dg.invSigmaImg.z);
                atomicAdd(&deltaGaussians[gid].color.x, dg.color.x);
                atomicAdd(&deltaGaussians[gid].color.y, dg.color.y);
                atomicAdd(&deltaGaussians[gid].color.z, dg.color.z);
                atomicAdd(&deltaGaussians[gid].depth, dg.depth);
                atomicAdd(&deltaGaussians[gid].alpha, dg.alpha);
                atomicAdd(&deltaGaussians[gid].n, dg.n);
            }
            block.sync();
        }
    }

    __global__ void optimizeGaussians2_kernel(
        DeltaGaussian2D *__restrict__ deltaGaussians,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (tid < n)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
            splattedGaussian_sh[tid].pHat = pHats[gid];
        }
        block.sync();

        if (x >= width || y >= height)
            return;

        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);
        float depth = 0.f;
        float T = 1.f;

        float omegas[BLOCK_SIZE];
        float depths[BLOCK_SIZE];

        for (int i = 0; i < n; i++)
        {
            const float dx = splattedGaussian_sh[i].position.x - x;
            const float dy = splattedGaussian_sh[i].position.y - y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
            if (alpha_i < 1.f / 255.f || v <= 0.f)
            {
                omegas[i] = 0.f;
                depths[i] = 0.f;
                continue;
            }

            float omega = alpha_i * prod_alpha;

            color += splattedGaussian_sh[i].color * omega;

            float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

#ifdef USE_MEAN_DEPTH
            // mean depth
            depth += omega * d;
#endif
            omegas[i] = omega;
            depths[i] = d;

            prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
            // median depth
            if (T > 0.5f && prod_alpha < 0.5)
            {
                T = prod_alpha;
                // depth = splattedGaussian_sh[i].position.z;
                depth = d;
            }
#endif
            if (prod_alpha < 0.001f)
            {
                // n = i+1;
                break;
            }
        }

        const float final_T = max(0.f, min(1.f, prod_alpha));

        // add bg color
        color += final_T * bgColor;

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        const float3 color_error = color - make_float3(rgba.x / 255.f,
                                                       rgba.y / 255.f,
                                                       rgba.z / 255.f);
        const float imgDepth = tex2D<float>(texDepth, x, y);

        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        block.sync();

        prod_alpha = 1.f;
        T = 1.f;

        float3 acc_c = make_float3(0.f);
        /// float acc_d = 0.f;

        float alpha_prev = 0.f;
        float3 color_prev = make_float3(0.f);
        // float T = final_T;

        for (int i = 0; i < n; i++)
        {
            const float dx = splattedGaussian_sh[i].position.x - x;
            const float dy = splattedGaussian_sh[i].position.y - y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float G = expf(-0.5f * v);
            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
            if (alpha_i < 1.f / 255.f)
                continue;

            float3 d_alpha = splattedGaussian_sh[i].color * prod_alpha;

            acc_c += d_alpha * alpha_i;

            d_alpha -= (color - acc_c) / (1.f - alpha_i);

            d_alpha = -color_error * d_alpha;

            const float dl_alpha = d_alpha.x + d_alpha.y + d_alpha.z;

            const float G_dl = G * dl_alpha;
            const float a_G_dl = alpha_i * dl_alpha;

            DeltaGaussian2D dg;
            uint32_t gid = gids_sh[i];

            dg.color = -alpha_i * prod_alpha * color_error;
            dg.alpha = G_dl;

#ifdef USE_MEAN_DEPTH
            // mean depth
            // dg.depth = -alpha_i * prod_alpha * depth_error;
            // atomicAdd(&deltaGaussians[gid].depth, -alpha_i * prod_alpha * depth_error);
            // dg.alpha -= depth_error * G * dl_alpha;
            float omega = alpha_i * prod_alpha;
            dg.depth = -w_depth * omega * depth_error;
            dg.meanImg.x -= w_depth * omega * depth_error * splattedGaussian_sh[i].pHat.x;
            dg.meanImg.y -= w_depth * omega * depth_error * splattedGaussian_sh[i].pHat.y;

            dg.pHat.x = -w_depth * omega * depth_error * dx;
            dg.pHat.y = -w_depth * omega * depth_error * dy;

#endif

            dg.meanImg.x = -a_G_dl * (splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy);
            dg.meanImg.y = -a_G_dl * (splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);
            dg.invSigmaImg.x = -0.5f * a_G_dl * dx * dx;
            dg.invSigmaImg.y = -0.5f * a_G_dl * dx * dy;
            dg.invSigmaImg.z = -0.5f * a_G_dl * dy * dy;

            prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
            // median depth
            if (T > 0.5f && prod_alpha <= 0.5f)
            {
                float w_depth_norm = w_depth / (imgDepth + 0.1f);
                T = prod_alpha;
                dg.depth = -w_depth_norm * depth_error;
                dg.meanImg.x -= w_depth_norm * depth_error * splattedGaussian_sh[i].pHat.x;
                dg.meanImg.y -= w_depth_norm * depth_error * splattedGaussian_sh[i].pHat.y;

                dg.pHat.x = -w_depth_norm * depth_error * dx;
                dg.pHat.y = -w_depth_norm * depth_error * dy;
            }
            else
            {
                dg.depth = 0.f;
                dg.pHat.x = 0.f;
                dg.pHat.y = 0.f;
            }
#endif

            for (int j = 0; j < n; j++)
            {
                float dd = depths[i] - depths[j];
                float ww = omegas[i] * omegas[j] * w_dist;
                float coeff = ww * dd;
                if (coeff == 0.f)
                    continue;

                dg.depth -= coeff;

                dg.meanImg.x -= coeff * splattedGaussian_sh[i].pHat.x;
                dg.meanImg.y -= coeff * splattedGaussian_sh[i].pHat.y;

                dg.pHat.x -= coeff * dx;
                dg.pHat.y -= coeff * dy;
            }

            atomicAdd(&deltaGaussians[gid].depth, dg.depth);
            atomicAdd(&deltaGaussians[gid].pHat.x, dg.pHat.x);
            atomicAdd(&deltaGaussians[gid].pHat.y, dg.pHat.y);

            atomicAdd(&deltaGaussians[gid].meanImg.x, dg.meanImg.x);
            atomicAdd(&deltaGaussians[gid].meanImg.y, dg.meanImg.y);
            atomicAdd(&deltaGaussians[gid].invSigmaImg.x, dg.invSigmaImg.x);
            atomicAdd(&deltaGaussians[gid].invSigmaImg.y, dg.invSigmaImg.y);
            atomicAdd(&deltaGaussians[gid].invSigmaImg.z, dg.invSigmaImg.z);
            atomicAdd(&deltaGaussians[gid].color.x, dg.color.x);
            atomicAdd(&deltaGaussians[gid].color.y, dg.color.y);
            atomicAdd(&deltaGaussians[gid].color.z, dg.color.z);
            // atomicAdd(&deltaGaussians[gid].depth, dg.depth);
            atomicAdd(&deltaGaussians[gid].alpha, dg.alpha);
            atomicAggInc(&deltaGaussians[gid].n);
            // atomicAdd(&deltaGaussians[gid].n, 1.f);
            // atomicAdd(&deltaGaussians[gid].n, alpha_i);
            if (prod_alpha < 0.001f)
            {
                break;
            }
        }
    }

    __global__ void perTileBucketCount(
        uint32_t *__restrict__ bucketCount,
        const uint2 *__restrict__ ranges,
        int numTiles)
    {
        auto idx = cg::this_grid().thread_rank();
        if (idx >= numTiles)
            return;

        uint2 range = ranges[idx];
        // int num_splats = min(range.y - range.x, BLOCK_SIZE);
        int num_splats = range.y - range.x;
        int num_buckets = (num_splats + 31) / 32;
        bucketCount[idx] = (uint32_t)num_buckets;
    }

    // __global__ void optimizeGaussiansForwardPass(
    //     const uint2 *__restrict__ ranges,
    //     const uint32_t *__restrict__ indices,
    //     const float3 *__restrict__ imgPositions,
    //     const float3 *__restrict__ imgSigmas,
    //     const float3 *__restrict__ imgInvSigmas,
    //     const float2 *__restrict__ pHats,
    //     const float3 *__restrict__ colors,
    //     const float *__restrict__ alphas,
    //     const uint32_t *__restrict__ bucketOffsets,
    //     uint32_t *__restrict__ bucketToTile,
    //     float *__restrict__ sampled_T,
    //     float3 *__restrict__ sampled_ar,
    //     float *__restrict__ final_T,
    //     uint32_t *__restrict__ n_contrib,
    //     uint32_t *__restrict__ max_contrib,
    //     float3 *__restrict__ output_color,
    //     float3 *__restrict__ color_error,
    //     float *__restrict__ depth_error,
    //     cudaTextureObject_t texRGBA,
    //     cudaTextureObject_t texDepth,
    //     float3 bgColor,
    //     uint2 numTiles,
    //     uint32_t width,
    //     uint32_t height)
    // {
    //     auto block = cg::this_thread_block();
    //     int tid = block.thread_rank();

    //     int x = blockIdx.x * blockDim.x + threadIdx.x;
    //     int y = blockIdx.y * blockDim.y + threadIdx.y;
    //     uint32_t pix_id = width * y + x;

    //     uint32_t tileId = blockIdx.y * numTiles.x + blockIdx.x;
    //     uint2 range = ranges[tileId];
    //     int n = min(range.y - range.x, BLOCK_SIZE);

    //     uint32_t bbm = tileId == 0 ? 0 : bucketOffsets[tileId - 1];
    //     int num_buckets = (n + 31) / 32;
    //     for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
    //     {
    //         int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
    //         if (bucket_idx < num_buckets)
    //         {
    //             bucketToTile[bbm + bucket_idx] = tileId;
    //         }
    //     }

    //     __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
    //     __shared__ uint32_t gids_sh[BLOCK_SIZE];

    //     // collect gaussians data
    //     if (tid < n)
    //     {
    //         uint32_t gid = indices[range.x + tid];
    //         gids_sh[tid] = gid;
    //         splattedGaussian_sh[tid].position = imgPositions[gid];
    //         splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
    //         splattedGaussian_sh[tid].color = colors[gid];
    //         splattedGaussian_sh[tid].alpha = alphas[gid];
    //         splattedGaussian_sh[tid].pHat = pHats[gid];
    //     }
    //     block.sync();

    //     bool inside = (x < width && y < height);
    //     bool done = !inside;

    //     float prod_alpha = 1.f;
    //     float3 color = make_float3(0.f);
    //     float depth = 0.f;
    //     float T = 1.f;
    //     uint32_t contributor = 0;
    //     uint32_t last_contributor = 0;

    //     for (int i = 0; !done && i < n; i++)
    //     {
    //         if (i % 32 == 0)
    //         {
    //             sampled_T[(bbm * BLOCK_SIZE) + tid] = T;
    //             sampled_ar[(bbm * BLOCK_SIZE) + tid] = color;
    //             ++bbm;
    //         }

    //         contributor++;
    //         const float dx = splattedGaussian_sh[i].position.x - x;
    //         const float dy = splattedGaussian_sh[i].position.y - y;
    //         const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
    //         const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
    //         if (alpha_i < 1.f / 255.f || v <= 0.f)
    //             continue;
    //         float test_T = T * (1 - alpha_i);
    //         if (test_T < 0.0001f)
    //         {
    //             done = true;
    //             continue;
    //         }

    //         color += splattedGaussian_sh[i].color * alpha_i * T;
    //         float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

    //         if (T > 0.5f && test_T < 0.5f)
    //         {
    //             depth = d;
    //         }

    //         T = test_T;
    //         last_contributor = contributor;
    //     }

    //     if (inside)
    //     {
    //         final_T[pix_id] = T;
    //         n_contrib[pix_id] = last_contributor;
    //         color += T * bgColor;
    //         uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);
    //         color_error[pix_id] = color - make_float3(rgba.x / 255.f,
    //                                                   rgba.y / 255.f,
    //                                                   rgba.z / 255.f);
    //         output_color[pix_id] = color;

    //         const float imgDepth = tex2D<float>(texDepth, x, y);
    //         const float depth_err = imgDepth > 0.1f ? depth - imgDepth : 0.f;
    //         depth_error[pix_id] = depth_err;
    //     }

    //     typedef cub::BlockReduce<uint32_t, GSS_BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
    //                              GSS_BLOCK_Y>
    //         BlockReduce;
    //     __shared__ typename BlockReduce::TempStorage temp_storage;
    //     last_contributor =
    //         BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
    //     if (block.thread_rank() == 0)
    //     {
    //         max_contrib[tileId] = last_contributor;
    //     }
    // }

    __global__ void optimizeGaussiansForwardPass(
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        const uint32_t *__restrict__ bucketOffsets,
        uint32_t *__restrict__ bucketToTile,
        float *__restrict__ sampled_T,
        float3 *__restrict__ sampled_ar,
        float *__restrict__ final_T,
        uint32_t *__restrict__ n_contrib,
        uint32_t *__restrict__ max_contrib,
        float3 *__restrict__ output_color,
        float *__restrict__ output_depth,
        float3 *__restrict__ color_error,
        float *__restrict__ depth_error,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        float3 bgColor,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        uint32_t pix_id = width * y + x;

        uint32_t tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = min(range.y - range.x, BLOCK_SIZE);

        uint32_t bbm = tileId == 0 ? 0 : bucketOffsets[tileId - 1];
        int num_buckets = (n + 31) / 32;
        for (int i = 0; i < (num_buckets + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
        {
            int bucket_idx = i * BLOCK_SIZE + block.thread_rank();
            if (bucket_idx < num_buckets)
            {
                bucketToTile[bbm + bucket_idx] = tileId;
            }
        }

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        bool inside = (x < width && y < height);
        bool done = !inside;

        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);
        float depth = 0.f;
        float T = 1.f;
        uint32_t contributor = 0;
        uint32_t last_contributor = 0;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; !done && i < BLOCK_SIZE && i + k * BLOCK_SIZE < n; i++)
            {
                if ((i + k * BLOCK_SIZE) % 32 == 0)
                {
                    sampled_T[(bbm * BLOCK_SIZE) + tid] = T;
                    sampled_ar[(bbm * BLOCK_SIZE) + tid] = color;
                    ++bbm;
                }

                contributor++;
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f || v <= 0.f)
                    continue;
                float test_T = T * (1 - alpha_i);
                if (test_T < 0.0001f)
                {
                    done = true;
                    continue;
                }

                color += splattedGaussian_sh[i].color * alpha_i * T;
                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                if (T > 0.5f && test_T < 0.5f)
                {
                    depth = d;
                }

                T = test_T;
                last_contributor = contributor;
            }
        }

        if (inside)
        {
            final_T[pix_id] = T;
            n_contrib[pix_id] = last_contributor;
            color += T * bgColor;
            uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);
            color_error[pix_id] = color - make_float3(rgba.x / 255.f,
                                                      rgba.y / 255.f,
                                                      rgba.z / 255.f);
            output_color[pix_id] = color;
            output_depth[pix_id] = depth;

            const float imgDepth = tex2D<float>(texDepth, x, y);
            // const float depth_err = imgDepth > 0.1f ? depth - imgDepth : 0.f;
            // depth_error[pix_id] = depth_err;

            depth_error[pix_id] = imgDepth;
        }

        typedef cub::BlockReduce<uint32_t, GSS_BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS,
                                 GSS_BLOCK_Y>
            BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        last_contributor =
            BlockReduce(temp_storage).Reduce(last_contributor, cub::Max());
        if (block.thread_rank() == 0)
        {
            max_contrib[tileId] = last_contributor;
        }
    }

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
        uint32_t nbBuckets)
    {
        auto block = cg::this_thread_block();
        auto my_warp = cg::tiled_partition<32>(block);
        uint32_t global_bucket_idx =
            block.group_index().x * my_warp.meta_group_size() +
            my_warp.meta_group_rank();
        bool valid_bucket = global_bucket_idx < nbBuckets;
        if (!valid_bucket)
            return;

        bool valid_splat = false;
        uint32_t tile_id, bbm;
        uint2 range;
        int num_splats_in_tile, bucket_idx_in_tile;
        int splat_idx_in_tile, splat_idx_global;

        tile_id = bucketToTile[global_bucket_idx];
        range = ranges[tile_id];
        num_splats_in_tile = min(range.y - range.x, BLOCK_SIZE);

        bbm = tile_id == 0 ? 0 : bucketOffsets[tile_id - 1];
        bucket_idx_in_tile = global_bucket_idx - bbm;
        splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
        splat_idx_global = range.x + splat_idx_in_tile;
        valid_splat = (splat_idx_in_tile < num_splats_in_tile);

        // if first gaussian in bucket is useless, then others are also useless
        if (bucket_idx_in_tile * 32 >= max_contrib[tile_id])
        {
            return;
        }

        uint32_t gaussian_idx = 0;

        SplattedGaussian splattedGaussian;

        if (valid_splat)
        {
            gaussian_idx = indices[splat_idx_global];
            splattedGaussian.position = imgPositions[gaussian_idx];
            splattedGaussian.invSigma = imgInvSigmas[gaussian_idx];
            splattedGaussian.color = colors[gaussian_idx];
            splattedGaussian.alpha = alphas[gaussian_idx];
            splattedGaussian.pHat = pHats[gaussian_idx];
        }

        DeltaGaussian2D deltaGaussian;
        deltaGaussian.meanImg = {0.f, 0.f};
        deltaGaussian.invSigmaImg = {0.f, 0.f, 0.f};
        deltaGaussian.color = {0.f, 0.f, 0.f};
        deltaGaussian.depth = 0.f;
        deltaGaussian.alpha = 0.f;
        deltaGaussian.pHat = {0.f, 0.f};
        deltaGaussian.n = 0;

        const uint2 tile = {tile_id % numTiles.x, tile_id / numTiles.x};
        const uint2 pix_min = {tile.x * GSS_BLOCK_X, tile.y * GSS_BLOCK_Y};

        float T;
        float T_final;
        float last_contributor;
        float3 acc_c;
        float3 color_err;
        float3 color;
        float img_depth; // depth_err;
        float depth;

        for (int i = 0; i < BLOCK_SIZE + 31; ++i)
        {
            T = my_warp.shfl_up(T, 1);
            last_contributor = my_warp.shfl_up(last_contributor, 1);
            T_final = my_warp.shfl_up(T_final, 1);
            acc_c = my_warp.shfl_up(acc_c, 1);
            color = my_warp.shfl_up(color, 1);
            color_err = my_warp.shfl_up(color_err, 1);
            img_depth = my_warp.shfl_up(img_depth, 1);
            depth = my_warp.shfl_up(depth, 1);

            // which pixel index should this thread deal with?
            int idx = i - my_warp.thread_rank();
            const uint2 pix = {pix_min.x + idx % GSS_BLOCK_X, pix_min.y + idx / GSS_BLOCK_X};
            const uint32_t pix_id = width * pix.y + pix.x;
            const float2 pixf = {(float)pix.x, (float)pix.y};
            bool valid_pixel = pix.x < width && pix.y < height;

            // every 32nd thread should read the stored state from memory
            // TODO: perhaps store these things in shared memory?
            if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 &&
                idx < BLOCK_SIZE)
            {
                T = sampled_T[global_bucket_idx * BLOCK_SIZE + idx];
                T_final = final_T[pix_id];

                acc_c = sampled_ar[global_bucket_idx * BLOCK_SIZE + idx];
                color = output_color[pix_id];
                depth = output_depth[pix_id];
                last_contributor = n_contrib[pix_id];
                color_err = color_error[pix_id];
                img_depth = depth_error[pix_id];
            }

            // do work
            if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE)
            {
                if (width <= pix.x || height <= pix.y)
                    continue;
                if (splat_idx_in_tile >= last_contributor)
                    continue;

                // compute blending values
                const float dx = splattedGaussian.position.x - pix.x;
                const float dy = splattedGaussian.position.y - pix.y;
                const float v = splattedGaussian.invSigma.x * dx * dx + 2.f * splattedGaussian.invSigma.y * dx * dy + splattedGaussian.invSigma.z * dy * dy;
                const float G = expf(-0.5f * v);
                const float alpha = min(0.99f, splattedGaussian.alpha * G);
                if (alpha < 1.f / 255.f)
                    continue;

                float3 d_alpha = splattedGaussian.color * T;
                acc_c += d_alpha * alpha;
                d_alpha -= (color - acc_c) / (1.f - alpha);
                d_alpha = -color_err * d_alpha;
                const float dl_alpha = d_alpha.x + d_alpha.y + d_alpha.z;
                const float G_dl = G * dl_alpha;
                const float a_G_dl = alpha * dl_alpha;

                deltaGaussian.n++;

                deltaGaussian.color -= alpha * T * color_err;
                deltaGaussian.alpha -= a_G_dl;
                deltaGaussian.meanImg.x -= a_G_dl * (splattedGaussian.invSigma.x * dx + splattedGaussian.invSigma.y * dy);
                deltaGaussian.meanImg.y -= a_G_dl * (splattedGaussian.invSigma.y * dx + splattedGaussian.invSigma.z * dy);
                deltaGaussian.invSigmaImg.x -= 0.5f * a_G_dl * dx * dx;
                deltaGaussian.invSigmaImg.y -= 0.5f * a_G_dl * dx * dy;
                deltaGaussian.invSigmaImg.z -= 0.5f * a_G_dl * dy * dy;

                float test_T = T * (1.f - alpha);

                if (T > 0.5f && test_T <= 0.5f)
                {
                    const float depth_err = img_depth > 0.1f ? depth - img_depth : 0.f;
                    deltaGaussian.depth -= w_depth * depth_err;
                    deltaGaussian.meanImg.x -= w_depth * depth_err * splattedGaussian.pHat.x;
                    deltaGaussian.meanImg.y -= w_depth * depth_err * splattedGaussian.pHat.y;

                    deltaGaussian.pHat.x -= w_depth * depth_err * dx;
                    deltaGaussian.pHat.y -= w_depth * depth_err * dy;
                }

                // simplified depth distortion
                float di = splattedGaussian.position.z + dx * splattedGaussian.pHat.x + dy * splattedGaussian.pHat.y;
                float dd = di - depth;
                float dist_coeff = w_dist * alpha * T * dd;
                deltaGaussian.depth -= dist_coeff;
                deltaGaussian.meanImg.x -= dist_coeff * splattedGaussian.pHat.x;
                deltaGaussian.meanImg.y -= dist_coeff * splattedGaussian.pHat.y;

                deltaGaussian.pHat.x -= dist_coeff * dx;
                deltaGaussian.pHat.y -= dist_coeff * dy;

                T = test_T;
            }
        }

        if (valid_splat && deltaGaussian.n > 0)
        {
            atomicAdd(&deltaGaussians[gaussian_idx].depth, deltaGaussian.depth);
            atomicAdd(&deltaGaussians[gaussian_idx].pHat.x, deltaGaussian.pHat.x);
            atomicAdd(&deltaGaussians[gaussian_idx].pHat.y, deltaGaussian.pHat.y);
            atomicAdd(&deltaGaussians[gaussian_idx].meanImg.x, deltaGaussian.meanImg.x);
            atomicAdd(&deltaGaussians[gaussian_idx].meanImg.y, deltaGaussian.meanImg.y);
            atomicAdd(&deltaGaussians[gaussian_idx].invSigmaImg.x, deltaGaussian.invSigmaImg.x);
            atomicAdd(&deltaGaussians[gaussian_idx].invSigmaImg.y, deltaGaussian.invSigmaImg.y);
            atomicAdd(&deltaGaussians[gaussian_idx].invSigmaImg.z, deltaGaussian.invSigmaImg.z);
            atomicAdd(&deltaGaussians[gaussian_idx].color.x, deltaGaussian.color.x);
            atomicAdd(&deltaGaussians[gaussian_idx].color.y, deltaGaussian.color.y);
            atomicAdd(&deltaGaussians[gaussian_idx].color.z, deltaGaussian.color.z);
            atomicAdd(&deltaGaussians[gaussian_idx].alpha, deltaGaussian.alpha);
            atomicAdd(&deltaGaussians[gaussian_idx].n, deltaGaussian.n);
        }
    }

    __global__ void applyDeltaGaussians_kernel(
        float3 *__restrict__ positions,
        float3 *__restrict__ scales,
        float4 *__restrict__ orientations,
        float3 *__restrict__ colors,
        float *__restrict__ alphas,
        const DeltaGaussian2D *__restrict__ deltaGaussians,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        float lambda_iso,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        DeltaGaussian2D dg = deltaGaussians[idx];
        if (dg.n == 0)
        {
            // alphas[idx] = 0.f;
            // printf("d %f %f %f\n", positions[idx].x, positions[idx].y, positions[idx].z);
            return;
        }

        // dg.n = 1.f;

        colors[idx] = min(make_float3(1.f, 1.f, 1.f), max(make_float3(0.f, 0.f, 0.f), colors[idx] + eta * dg.color / dg.n));

        /*sigmoid parameterization*/
        float alpha = alphas[idx];

        float alpha_s = logf(alpha / (1.f - alpha)) + eta * (dg.alpha / dg.n) / (alpha - alpha * alpha);
        alphas[idx] = max(0.01f, min(0.99f, 1.f / (1.f + expf(-alpha_s))));
        /// alphas[idx] = min(0.99f, max(0.01f, alphas[idx] + eta * dg.alpha / dg.n));

        // compute delta mean3D
        Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);
        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);
        Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
        Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);

        Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                                     {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

        Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
        Eigen::Matrix3f R = q_gauss.toRotationMatrix();

        Eigen::Vector2f dl_mean2d(dg.meanImg.x / dg.n, dg.meanImg.y / dg.n);

        // dl_mean2d.normalize();

        // Eigen::Matrix<float, 3, 2> Ji = ((J.transpose() * J).inverse()) * J.transpose();
        // Eigen::Matrix<float, 3, 2> Ji = J.transpose() * ((J * J.transpose()).inverse());
        // Eigen::Vector3f dl_mean3d = q_cam * (Ji * dl_mean2d);

        // dg.depth = 0.f;

        Eigen::Vector3f dl_mean3d = q_cam * (J.transpose() * dl_mean2d + (dg.depth / dg.n) * Eigen::Vector3f(mu_cam.x() / mu_cam.z(), mu_cam.y() / mu_cam.z(), 1.f));

        /*update position from 2D mean grad*/

        positions[idx].x += eta * dl_mean3d.x();
        positions[idx].y += eta * dl_mean3d.y();
        positions[idx].z += eta * dl_mean3d.z();

        /* compute derivative of inverse Cov2 wrt cov2*/
        const Eigen::Matrix<float, 2, 3> T = J * W;
        const Eigen::Matrix3f RS = R * s_gauss.asDiagonal();
        const Eigen::Matrix<float, 2, 3> M = T * RS;
        const Eigen::Matrix2f Cov2d = M * M.transpose();

        float a = Cov2d(0, 0) + 0.1f;
        float b = Cov2d(1, 0);
        float c = Cov2d(1, 1) + 0.1f;

        float denom = a * c - b * b;
        float dL_da = 0, dL_db = 0, dL_dc = 0;
        float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

        float3 dL_dconic = dg.invSigmaImg / dg.n;

        // derivative of Cov2 wrt Cov2^-1 :
        dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
        // cov2D = T * Cov3d * T;
        const Eigen::Matrix3f Cov3D = RS * (RS.transpose());

        float dL_dT00 = 2 * (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_da +
                        (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_db;
        float dL_dT01 = 2 * (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_da +
                        (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_db;
        float dL_dT02 = 2 * (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_da +
                        (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_db;
        float dL_dT10 = 2 * (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_db;
        float dL_dT11 = 2 * (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_db;
        float dL_dT12 = 2 * (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_db;
        // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
        // T = J * W
        float dL_dJ00 = W(0, 0) * dL_dT00 + W(0, 1) * dL_dT01 + W(0, 2) * dL_dT02;
        float dL_dJ02 = W(2, 0) * dL_dT00 + W(2, 1) * dL_dT01 + W(2, 2) * dL_dT02;
        float dL_dJ11 = W(1, 0) * dL_dT10 + W(1, 1) * dL_dT11 + W(1, 2) * dL_dT12;
        float dL_dJ12 = W(2, 0) * dL_dT10 + W(2, 1) * dL_dT11 + W(2, 2) * dL_dT12;

        float tz = 1.f / mu_cam.z();
        float tz2 = tz * tz;
        float tz3 = tz2 * tz;

        // Gradients of loss w.r.t. transformed Gaussian mean t
        float dL_dtx = -cameraParams.f.x * tz2 * dL_dJ02;
        float dL_dty = -cameraParams.f.y * tz2 * dL_dJ12;
        float dL_dtz = -cameraParams.f.x * tz2 * dL_dJ00 - cameraParams.f.y * tz2 * dL_dJ11 + (2 * cameraParams.f.x * mu_cam.x()) * tz3 * dL_dJ02 + (2 * cameraParams.f.y * mu_cam.y()) * tz3 * dL_dJ12;
        Eigen::Vector3f dl_dmean = q_cam.inverse() * Eigen::Vector3f(dL_dtx, dL_dty, dL_dtz);

        // printf("dL_dtx = %f %f %f\n", dl_dmean.x(), dl_dmean.y(), dl_dmean.z());
        // printf("dl_dmean = %f %f %f\n", dl_dmean.x(), dl_dmean.y(), dl_dmean.z());

        /*update position from covariance grad*/

        positions[idx].x += eta * dl_dmean.x();
        positions[idx].y += eta * dl_dmean.y();
        positions[idx].z += eta * dl_dmean.z();

        Eigen::Matrix3f dL_dcov3D;
        dL_dcov3D(0, 0) = (T(0, 0) * T(0, 0) * dL_da + T(0, 0) * T(1, 0) * dL_db + T(1, 0) * T(1, 0) * dL_dc);
        dL_dcov3D(1, 1) = (T(0, 1) * T(0, 1) * dL_da + T(0, 1) * T(1, 1) * dL_db + T(1, 1) * T(1, 1) * dL_dc);
        dL_dcov3D(2, 2) = (T(0, 2) * T(0, 2) * dL_da + T(0, 2) * T(1, 2) * dL_db + T(1, 2) * T(1, 2) * dL_dc);
        dL_dcov3D(1, 0) = dL_dcov3D(0, 1) = T(0, 0) * T(0, 1) * dL_da + 0.5f * (T(0, 0) * T(1, 1) + T(0, 1) * T(1, 0)) * dL_db + T(1, 0) * T(1, 1) * dL_dc;
        dL_dcov3D(2, 0) = dL_dcov3D(0, 2) = T(0, 0) * T(0, 2) * dL_da + 0.5f * (T(0, 0) * T(1, 2) + T(0, 2) * T(1, 0)) * dL_db + T(1, 0) * T(1, 2) * dL_dc;
        dL_dcov3D(2, 1) = dL_dcov3D(1, 2) = T(0, 2) * T(0, 1) * dL_da + 0.5f * (T(0, 1) * T(1, 2) + T(0, 2) * T(1, 1)) * dL_db + T(1, 1) * T(1, 2) * dL_dc;

        Eigen::Matrix3f dL_dM = 2.0f * dL_dcov3D * RS; // 2.0f * RS * dL_dcov3D;
        // Eigen::Matrix3f dL_dMt = 2.0f * dL_dcov3D * RS;

        Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dM(0, 0) + R(1, 0) * dL_dM(1, 0) + R(2, 0) * dL_dM(2, 0),
                                                    R(0, 1) * dL_dM(0, 1) + R(1, 1) * dL_dM(1, 1) + R(2, 1) * dL_dM(2, 1),
                                                    R(0, 2) * dL_dM(0, 2) + R(1, 2) * dL_dM(1, 2) + R(2, 2) * dL_dM(2, 2));
        // Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dMt(0, 0) + R(1, 0) * dL_dMt(1, 0) + R(2, 0) * dL_dMt(2, 0),
        //                                             R(0, 1) * dL_dMt(0, 1) + R(1, 1) * dL_dMt(1, 1) + R(2, 1) * dL_dMt(2, 1),
        //                                             R(0, 2) * dL_dMt(0, 2) + R(1, 2) * dL_dMt(1, 2) + R(2, 2) * dL_dMt(2, 2));

        const float3 scale = scales[idx];
        float3 scale_log = make_float3(logf(scale.x),
                                       logf(scale.y),
                                       logf(scale.z));

        // isotropic regulation
        float mean_scale = (scale.x + scale.y + scale.z) / 3.f;
        float3 dl_iso = make_float3(scale.x - mean_scale,
                                    scale.y - mean_scale,
                                    scale.z - mean_scale);

        /// dL_dscale -= lambda_iso * (1.f / 3.f) * Eigen::Vector3f(2.f * dl_iso.x - dl_iso.y - dl_iso.z, -dl_iso.x + 2.f * dl_iso.y - dl_iso.z, -dl_iso.x - dl_iso.y + 2.f * dl_iso.z);

        scale_log += eta * min(make_float3(2.f, 2.f, 2.f), make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z()) / scale);
        // scale_log += eta * make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z()) / scale;

        // dL_dM = s_gauss.asDiagonal() * dL_dM;
        // dL_dM.row(0) *= scale.x;
        // dL_dM.row(1) *= scale.y;
        // dL_dM.row(2) *= scale.z;
        dL_dM.col(0) *= scale.x;
        dL_dM.col(1) *= scale.y;
        dL_dM.col(2) *= scale.z;

        // Gradient of loss wrt to the gaussian orientation in the tangent space

        Eigen::Vector3f dL_dq = (R.col(0).cross(dL_dM.col(0)) + R.col(1).cross(dL_dM.col(1)) + R.col(2).cross(dL_dM.col(2)));
        q_gauss = q_gauss * Eigen::Quaternionf(1.f, eta * dL_dq.x(), eta * dL_dq.y(), eta * dL_dq.z());
        q_gauss.normalize();

        // Gradients of loss w.r.t. normalized quaternion
        /*
        Eigen::Quaternionf dL_dq;
        const float x = q_gauss.x();
        const float y = q_gauss.y();
        const float z = q_gauss.z();
        const float w = q_gauss.w();

        dL_dq.w() = 2 * z * (dL_dM(0, 1) - dL_dM(1, 0)) + 2 * y * (dL_dM(2, 0) - dL_dM(0, 2)) + 2 * x * (dL_dM(1, 2) - dL_dM(2, 1));
        dL_dq.x() = 2 * y * (dL_dM(1, 0) + dL_dM(0, 1)) + 2 * z * (dL_dM(2, 0) + dL_dM(0, 2)) + 2 * w * (dL_dM(1, 2) - dL_dM(2, 1)) - 4 * x * (dL_dM(2, 2) + dL_dM(1, 1));
        dL_dq.y() = 2 * x * (dL_dM(1, 0) + dL_dM(0, 1)) + 2 * w * (dL_dM(2, 0) - dL_dM(0, 2)) + 2 * z * (dL_dM(1, 2) + dL_dM(2, 1)) - 4 * y * (dL_dM(2, 2) + dL_dM(0, 0));
        dL_dq.z() = 2 * w * (dL_dM(0, 1) - dL_dM(1, 0)) + 2 * x * (dL_dM(2, 0) + dL_dM(0, 2)) + 2 * y * (dL_dM(1, 2) + dL_dM(2, 1)) - 4 * z * (dL_dM(1, 1) + dL_dM(0, 0));

        // update orientation
        q_gauss.x() += dL_dq.x();
        q_gauss.y() += dL_dq.y();
        q_gauss.z() += dL_dq.z();
        q_gauss.w() += dL_dq.w();
        q_gauss.normalize();
        */

        // update scale
        // TODO : check saturation
        // scales[idx] = min(make_float3(0.15f), make_float3(expf(scale_log.x),
        //                                                   expf(scale_log.y),
        //                                                   expf(scale_log.z)));

        // scales[idx] = max(make_float3(0.001f), min(make_float3(0.2f), scales[idx] + eta * make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z())));

        // scales[idx] = max(make_float3(0.001f), min(make_float3(0.4f), make_float3(expf(scale_log.x),
        //                                                                           expf(scale_log.y),
        //                                                                           expf(scale_log.z))));
        scales[idx] = max(make_float3(0.0001f), make_float3(expf(scale_log.x),
                                                            expf(scale_log.y),
                                                            expf(scale_log.z)));

        /*
        printf("dl_mean2d : %f %f\n", dl_mean2d.x(), dl_mean2d.y());
        printf("dl_mean3d : %f %f %f\n", dl_mean3d.x(), dl_mean3d.y(), dl_mean3d.z());
        printf("mu_cam : %f %f %f\n", mu_cam.x(), mu_cam.y(), mu_cam.z());
        printf("Ji :\n %f %f\n %f %f\n%f %f\n",
               Ji(0, 0), Ji(0, 1),
               Ji(1, 0), Ji(1, 1),
               Ji(2, 0), Ji(2, 1));
        */
        /*
        printf("dl_alpha : %f\n", dg.alpha / dg.n);
        printf("dl_dinvcov2D : %f %f %f\n", dL_dconic.x, dL_dconic.z, dL_dconic.z);
        printf("dl_dcov2D : %f %f %f\n", dL_da, dL_db, dL_dc);
        printf("dl_dscale : %f %f %f\n", dL_dscale.x, dL_dscale.y, dL_dscale.z);
        */
        // printf("dL_dq : %f %f %f %f\n", dL_dq.x(), dL_dq.y(), dL_dq.z(), dL_dq.w());

        /*if (idx == 742)
        {
            printf("%f %f %f\n\n %f %f\n%f %f\n%f %f\n\n",
                   p_gauss.x(), p_gauss.y(), p_gauss.z(),
                   J(0, 0), J(1, 0),
                   J(0, 1), J(1, 1),
                   J(0, 2), J(1, 2));

            printf("delta mean : %f %f -> %f %f %f\n", dl_mean2d.x(), dl_mean2d.y(), dl_mean3d.x(), dl_mean3d.y(), dl_mean3d.z());
        }
        */
    }

    __global__ void applyDeltaGaussians2_kernel(
        float3 *__restrict__ positions,
        float3 *__restrict__ scales,
        float4 *__restrict__ orientations,
        float3 *__restrict__ colors,
        float *__restrict__ alphas,
        const DeltaGaussian2D *__restrict__ deltaGaussians,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        float lambda_iso,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        DeltaGaussian2D dg = deltaGaussians[idx];
        if (dg.n == 0)
        {
            // alphas[idx] = 0.f;
            // printf("d %f %f %f\n", positions[idx].x, positions[idx].y, positions[idx].z);
            return;
        }

        // dg.n = 1.f;

        const float eta_n = eta / (10.f + dg.n);

        colors[idx] = min(make_float3(1.f, 1.f, 1.f), max(make_float3(0.f, 0.f, 0.f), colors[idx] + eta_n * dg.color /*/ dg.n*/));

        /*sigmoid parameterization*/
        float alpha = alphas[idx];

        float alpha_s = logf(alpha / (1.f - alpha)) + eta_n * (dg.alpha /*/ dg.n*/) / (alpha - alpha * alpha);
        alphas[idx] = max(0.01f, min(0.99f, 1.f / (1.f + expf(-alpha_s))));
        /// alphas[idx] = min(0.99f, max(0.01f, alphas[idx] + eta * dg.alpha / dg.n));

        // compute delta mean3D
        Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);
        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);
        Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
        Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);

        Eigen::Matrix<float, 3, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                                     {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())},
                                     {0.f, 0.f, 1.f}};

        Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
        Eigen::Matrix3f R = q_gauss.toRotationMatrix();

        Eigen::Vector3f dl_mean(dg.meanImg.x /*/ dg.n*/, dg.meanImg.y /*/ dg.n*/, dg.depth /*/ dg.n*/);

        /* compute derivative of inverse Cov2 wrt cov2*/
        const Eigen::Matrix<float, 3, 3> T = J * W;
        const Eigen::Matrix3f RS = R * s_gauss.asDiagonal();
        const Eigen::Matrix<float, 3, 3> M = T * RS;
        const Eigen::Matrix3f Sigma = M * M.transpose();
        Eigen::Matrix3f Sigma_i = Sigma.inverse();

        Eigen::Vector3f dl_mean2d(dg.meanImg.x /*/ dg.n*/, dg.meanImg.y /*/ dg.n*/, 0.);
        Eigen::Vector3f dl_mean3d = q_cam * (J.transpose() * dl_mean2d + dg.depth * Eigen::Vector3f(mu_cam.x() / mu_cam.z(), mu_cam.y() / mu_cam.z(), 1.f));
        // Eigen::Vector3f dl_mean3d = q_cam * (J.transpose() * dl_mean);

        Eigen::Vector2f dl_pHat(dg.pHat.x /*/ dg.n*/, dg.pHat.y /*/ dg.n*/);

        // derivatives of pHat wrt position
        float norm2_mucam = mu_cam.dot(mu_cam);
        float norm_mucam = sqrtf(norm2_mucam);
        float demon_phat_mean = 1.f / (norm2_mucam * norm_mucam);
        Eigen::Vector3f dpHat_mean(-mu_cam.x() * mu_cam.z() * demon_phat_mean,
                                   -mu_cam.y() * mu_cam.z() * demon_phat_mean,
                                   (mu_cam.x() * mu_cam.x() + mu_cam.y() * mu_cam.y()) * demon_phat_mean);
        dl_mean3d += (q_cam * (dpHat_mean * ((Sigma_i(2, 0) * dl_pHat.x() + Sigma_i(2, 1) * dl_pHat.y()) / Sigma_i(2, 2))));

        float a = Sigma(0, 0) + 0.001f;
        float b = Sigma(1, 0);
        float c = Sigma(1, 1) + 0.001f;

        float denom = a * c - b * b;
        float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

        float3 dL_dconic = dg.invSigmaImg /*/ dg.n*/;

        // printf("dL_dconic : %f %f %f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);

        // derivative of Cov2 wrt Cov2^-1 :
        float dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        float dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        float dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        // printf("Cov2 : %f %f %f\n", a, b, c);
        // printf("dL_Cov2 : %f %f %f\n", dL_da, dL_db, dL_dc);

        // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
        // cov2D = T * Cov3d * T;
        const Eigen::Matrix3f Cov3D = RS * (RS.transpose());

        float dL_dT00 = 2 * (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_da +
                        (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_db;
        float dL_dT01 = 2 * (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_da +
                        (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_db;
        float dL_dT02 = 2 * (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_da +
                        (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_db;
        float dL_dT10 = 2 * (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_db;
        float dL_dT11 = 2 * (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_db;
        float dL_dT12 = 2 * (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_db;
        // float dL_dT20 = 0.f;
        // float dL_dT21 = 0.f;
        // float dL_dT22 = 0.f;

        // derivative of qHat wrt Sigma_i
        float zc_tc = mu_cam.z() / norm_mucam;
        Eigen::Matrix3f dL_dSigma_i;
        // CHECK HERE
        dL_dSigma_i << 0.f, 0.f, 0.5f * zc_tc * dl_pHat.x() / Sigma_i(2, 2),
            0.f, 0.f, 0.5f * zc_tc * dl_pHat.y() / Sigma_i(2, 2),
            0.5f * zc_tc * dl_pHat.x() / Sigma_i(2, 2), 0.5f * zc_tc * dl_pHat.y() / Sigma_i(2, 2), -zc_tc * (dl_pHat.x() * Sigma_i(2, 0) + dl_pHat.y() * Sigma_i(2, 1)) / (Sigma_i(2, 2) * Sigma_i(2, 2));
        // dL_dSigma_i << 0.f, 0.f, zc_tc * dl_pHat.x() / Sigma_i(2, 2),
        //     0.f, 0.f, zc_tc * dl_pHat.y() / Sigma_i(2, 2),
        //     zc_tc * dl_pHat.x() / Sigma_i(2, 2), zc_tc * dl_pHat.y() / Sigma_i(2, 2), -zc_tc * (dl_pHat.x() * Sigma_i(2, 0) + dl_pHat.y() * Sigma_i(2, 1)) / (Sigma_i(2, 2) * Sigma_i(2, 2));

        Eigen::Matrix3f dL_dSigma = -Sigma_i * dL_dSigma_i * Sigma_i;

        // printf("dL_dSigma : \n%e %e %e\n%e %e %e\n%e %e %e\n",
        //         dL_dSigma(0, 0), dL_dSigma(0, 1), dL_dSigma(0, 2),
        //         dL_dSigma(1, 0), dL_dSigma(1, 1), dL_dSigma(1, 2),
        //         dL_dSigma(2, 0), dL_dSigma(2, 1), dL_dSigma(2, 2));
        // printf("Sigma : \n%f %f %f\n%f %f %f\n%f %f %f\n",
        //        Sigma(0, 0), Sigma(0, 1), Sigma(0, 2),
        //        Sigma(1, 0), Sigma(1, 1), Sigma(1, 2),
        //        Sigma(2, 0), Sigma(2, 1), Sigma(2, 2));

        Eigen::Matrix3f TCov3D = T * Cov3D;
        Eigen::Matrix3f Cov3DT = Cov3D * T.transpose();
        dL_dT00 += TCov3D.col(0).dot(dL_dSigma.col(0)) + Cov3DT.row(0).dot(dL_dSigma.row(0));
        dL_dT01 += TCov3D.col(1).dot(dL_dSigma.col(0)) + Cov3DT.row(1).dot(dL_dSigma.row(0));
        dL_dT02 += TCov3D.col(2).dot(dL_dSigma.col(0)) + Cov3DT.row(2).dot(dL_dSigma.row(0));
        dL_dT10 += TCov3D.col(0).dot(dL_dSigma.col(1)) + Cov3DT.row(0).dot(dL_dSigma.row(1));
        dL_dT11 += TCov3D.col(1).dot(dL_dSigma.col(1)) + Cov3DT.row(1).dot(dL_dSigma.row(1));
        dL_dT12 += TCov3D.col(2).dot(dL_dSigma.col(1)) + Cov3DT.row(2).dot(dL_dSigma.row(1));
        // dL_dT20 += TCov3D.col(0).dot(dL_dSigma.col(2)) + Cov3DT.row(0).dot(dL_dSigma.row(2));
        // dL_dT21 += TCov3D.col(1).dot(dL_dSigma.col(2)) + Cov3DT.row(1).dot(dL_dSigma.row(2));
        // dL_dT22 += TCov3D.col(2).dot(dL_dSigma.col(2)) + Cov3DT.row(2).dot(dL_dSigma.row(2));

        // dL_dT00 += TCov3D.col(0).dot(dL_dSigma.col(0)) + Cov3DT.row(0).dot(dL_dSigma.row(0));
        // dL_dT10 += TCov3D.col(1).dot(dL_dSigma.col(0)) + Cov3DT.row(1).dot(dL_dSigma.row(0));
        // dL_dT20 += TCov3D.col(2).dot(dL_dSigma.col(0)) + Cov3DT.row(2).dot(dL_dSigma.row(0));
        // dL_dT01 += TCov3D.col(0).dot(dL_dSigma.col(1)) + Cov3DT.row(0).dot(dL_dSigma.row(1));
        // dL_dT11 += TCov3D.col(1).dot(dL_dSigma.col(1)) + Cov3DT.row(1).dot(dL_dSigma.row(1));
        // dL_dT21 += TCov3D.col(2).dot(dL_dSigma.col(1)) + Cov3DT.row(2).dot(dL_dSigma.row(1));
        // dL_dT02 += TCov3D.col(0).dot(dL_dSigma.col(2)) + Cov3DT.row(0).dot(dL_dSigma.row(2));
        // dL_dT12 += TCov3D.col(1).dot(dL_dSigma.col(2)) + Cov3DT.row(1).dot(dL_dSigma.row(2));
        // dL_dT22 += TCov3D.col(2).dot(dL_dSigma.col(2)) + Cov3DT.row(2).dot(dL_dSigma.row(2));

        // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
        // T = J * W
        float dL_dJ00 = W(0, 0) * dL_dT00 + W(0, 1) * dL_dT01 + W(0, 2) * dL_dT02;
        float dL_dJ02 = W(2, 0) * dL_dT00 + W(2, 1) * dL_dT01 + W(2, 2) * dL_dT02;
        float dL_dJ11 = W(1, 0) * dL_dT10 + W(1, 1) * dL_dT11 + W(1, 2) * dL_dT12;
        float dL_dJ12 = W(2, 0) * dL_dT10 + W(2, 1) * dL_dT11 + W(2, 2) * dL_dT12;

        float tz = 1.f / mu_cam.z();
        float tz2 = tz * tz;
        float tz3 = tz2 * tz;

        // Gradients of loss w.r.t. transformed Gaussian mean t
        float dL_dtx = -cameraParams.f.x * tz2 * dL_dJ02;
        float dL_dty = -cameraParams.f.y * tz2 * dL_dJ12;
        float dL_dtz = -cameraParams.f.x * tz2 * dL_dJ00 - cameraParams.f.y * tz2 * dL_dJ11 + (2 * cameraParams.f.x * mu_cam.x()) * tz3 * dL_dJ02 + (2 * cameraParams.f.y * mu_cam.y()) * tz3 * dL_dJ12;

        // Eigen::Vector3f dl_dmean = q_cam.inverse() * Eigen::Vector3f(dL_dtx, dL_dty, dL_dtz);
        dl_mean3d += q_cam * Eigen::Vector3f(dL_dtx, dL_dty, dL_dtz);
        // printf("dL_dtx = %f %f %f\n", dl_dmean.x(), dl_dmean.y(), dl_dmean.z());
        // printf("dl_dmean = %f %f %f\n", dl_dmean.x(), dl_dmean.y(), dl_dmean.z());

        /*update position from 2D + z mean + covariance grad*/

        positions[idx].x += eta_n * dl_mean3d.x();
        positions[idx].y += eta_n * dl_mean3d.y();
        positions[idx].z += eta_n * dl_mean3d.z();

        /*update position from covariance grad*/
        /*
        positions[idx].x += eta * dl_dmean.x();
        positions[idx].y += eta * dl_dmean.y();
        positions[idx].z += eta * dl_dmean.z();
        */

        Eigen::Matrix3f dL_dcov3D;
        dL_dcov3D(0, 0) = (T(0, 0) * T(0, 0) * dL_da + T(0, 0) * T(1, 0) * dL_db + T(1, 0) * T(1, 0) * dL_dc);
        dL_dcov3D(1, 1) = (T(0, 1) * T(0, 1) * dL_da + T(0, 1) * T(1, 1) * dL_db + T(1, 1) * T(1, 1) * dL_dc);
        dL_dcov3D(2, 2) = (T(0, 2) * T(0, 2) * dL_da + T(0, 2) * T(1, 2) * dL_db + T(1, 2) * T(1, 2) * dL_dc);
        dL_dcov3D(1, 0) = dL_dcov3D(0, 1) = T(0, 0) * T(0, 1) * dL_da + 0.5f * (T(0, 0) * T(1, 1) + T(0, 1) * T(1, 0)) * dL_db + T(1, 0) * T(1, 1) * dL_dc;
        dL_dcov3D(2, 0) = dL_dcov3D(0, 2) = T(0, 0) * T(0, 2) * dL_da + 0.5f * (T(0, 0) * T(1, 2) + T(0, 2) * T(1, 0)) * dL_db + T(1, 0) * T(1, 2) * dL_dc;
        dL_dcov3D(2, 1) = dL_dcov3D(1, 2) = T(0, 2) * T(0, 1) * dL_da + 0.5f * (T(0, 1) * T(1, 2) + T(0, 2) * T(1, 1)) * dL_db + T(1, 1) * T(1, 2) * dL_dc;

        // d_hat covariance
        dL_dcov3D += T.transpose() * dL_dSigma * T;

        // printf("cov3D : \n%e %e %e\n%e %e %e\n%e %e %e\n",
        //         Cov3D(0, 0), Cov3D(0, 1), Cov3D(0, 2),
        //         Cov3D(1, 0), Cov3D(1, 1), Cov3D(1, 2),
        //         Cov3D(2, 0), Cov3D(2, 1), Cov3D(2, 2));
        // printf("dL_dcov3D : \n%e %e %e\n%e %e %e\n%e %e %e\n",
        //         dL_dcov3D(0, 0), dL_dcov3D(0, 1), dL_dcov3D(0, 2),
        //         dL_dcov3D(1, 0), dL_dcov3D(1, 1), dL_dcov3D(1, 2),
        //         dL_dcov3D(2, 0), dL_dcov3D(2, 1), dL_dcov3D(2, 2));

        // Eigen::Matrix3f dL_dM = 2.0f * dL_dcov3D * RS; // 2.0f * RS * dL_dcov3D;
        // Eigen::Matrix3f dL_dM = 2.0f * RS * dL_dcov3D;
        Eigen::Matrix3f dL_dM = 2.0f * RS.transpose() * dL_dcov3D;
        // Eigen::Matrix3f dL_dM = 2.0f * dL_dcov3D * RS.transpose();

        // Eigen::Matrix3f dL_dM = RS * dL_dcov3D + dL_dcov3D * RS.transpose();

        // Eigen::Matrix3f dL_dMt = 2.0f * dL_dcov3D * RS;

        // printf("RS : \n%f %f %f\n%f %f %f\n%f %f %f\n",
        //         RS(0, 0), RS(0, 1), RS(0, 2),
        //         RS(1, 0), RS(1, 1), RS(1, 2),
        //         RS(2, 0), RS(2, 1), RS(2, 2));

        // printf("dL_dM : \n%e %e %e\n%e %e %e\n%e %e %e\n",
        //         dL_dM(0, 0), dL_dM(0, 1), dL_dM(0, 2),
        //         dL_dM(1, 0), dL_dM(1, 1), dL_dM(1, 2),
        //         dL_dM(2, 0), dL_dM(2, 1), dL_dM(2, 2));

        // Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dM(0, 0) + R(1, 0) * dL_dM(1, 0) + R(2, 0) * dL_dM(2, 0),
        //                                             R(0, 1) * dL_dM(0, 1) + R(1, 1) * dL_dM(1, 1) + R(2, 1) * dL_dM(2, 1),
        //                                             R(0, 2) * dL_dM(0, 2) + R(1, 2) * dL_dM(1, 2) + R(2, 2) * dL_dM(2, 2));
        Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dM(0, 0) + R(1, 0) * dL_dM(0, 1) + R(2, 0) * dL_dM(0, 2),
                                                    R(0, 1) * dL_dM(1, 0) + R(1, 1) * dL_dM(1, 1) + R(2, 1) * dL_dM(1, 2),
                                                    R(0, 2) * dL_dM(2, 0) + R(1, 2) * dL_dM(2, 1) + R(2, 2) * dL_dM(2, 2));
        // Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dM(0, 0) + R(0, 1) * dL_dM(1, 0) + R(0, 2) * dL_dM(2, 0),
        //                                             R(1, 0) * dL_dM(0, 1) + R(1, 1) * dL_dM(1, 1) + R(1, 2) * dL_dM(2, 1),
        //                                             R(2, 0) * dL_dM(0, 2) + R(2, 1) * dL_dM(1, 2) + R(2, 2) * dL_dM(2, 2));

        // Eigen::Vector3f dL_dscale = q_gauss.inverse() * Eigen::Vector3f(R(0, 0) * dL_dM(0, 0) + R(1, 0) * dL_dM(1, 0) + R(2, 0) * dL_dM(2, 0),
        //                                             R(0, 1) * dL_dM(0, 1) + R(1, 1) * dL_dM(1, 1) + R(2, 1) * dL_dM(2, 1),
        //                                             R(0, 2) * dL_dM(0, 2) + R(1, 2) * dL_dM(1, 2) + R(2, 2) * dL_dM(2, 2));
        // Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dMt(0, 0) + R(1, 0) * dL_dMt(1, 0) + R(2, 0) * dL_dMt(2, 0),
        //                                             R(0, 1) * dL_dMt(0, 1) + R(1, 1) * dL_dMt(1, 1) + R(2, 1) * dL_dMt(2, 1),
        //                                             R(0, 2) * dL_dMt(0, 2) + R(1, 2) * dL_dMt(1, 2) + R(2, 2) * dL_dMt(2, 2));
        // Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dMt(0, 0) + R(0, 1) * dL_dMt(1, 0) + R(0, 2) * dL_dMt(2, 0),
        //                                            R(1, 0) * dL_dMt(0, 1) + R(1, 1) * dL_dMt(1, 1) + R(1, 2) * dL_dMt(2, 1),
        //                                            R(2, 0) * dL_dMt(0, 2) + R(2, 1) * dL_dMt(1, 2) + R(2, 2) * dL_dMt(2, 2));

        // printf("dL_dscale : %e %e %e\n", dL_dscale.x(), dL_dscale.y(), dL_dscale.z());

        const float3 scale = scales[idx];
        float3 scale_log = make_float3(logf(scale.x),
                                       logf(scale.y),
                                       logf(scale.z));

        // isotropic regulation
        float mean_scale = (scale.x + scale.y + scale.z) / 3.f;
        float3 dl_iso = make_float3(scale.x - mean_scale,
                                    scale.y - mean_scale,
                                    scale.z - mean_scale);
        dL_dscale -= lambda_iso * (1.f / 3.f) * Eigen::Vector3f(2.f * dl_iso.x - dl_iso.y - dl_iso.z, -dl_iso.x + 2.f * dl_iso.y - dl_iso.z, -dl_iso.x - dl_iso.y + 2.f * dl_iso.z);

        // printf("scale_log : %f %f %f\n", scale_log.x, scale_log.y, scale_log.z);
        float3 dL_dscale_log = make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z()) / scale;
        // printf("dL_dscale_log : %f %f %f\n", dL_dscale_log.x, dL_dscale_log.y, dL_dscale_log.z);
        scale_log += 10.f * eta_n * min(make_float3(2.f, 2.f, 2.f), dL_dscale_log);
        // scale_log += eta * make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z()) / scale;

        // printf("scale_log updated : %f %f %f\n", scale_log.x, scale_log.y, scale_log.z);

        // dL_dM = s_gauss.asDiagonal() * dL_dM;
        dL_dM.row(0) *= scale.x;
        dL_dM.row(1) *= scale.y;
        dL_dM.row(2) *= scale.z;
        // dL_dM.col(0) *= scale.x;
        // dL_dM.col(1) *= scale.y;
        // dL_dM.col(2) *= scale.z;

        // printf("dL_dM : \n%e %e %e\n%e %e %e\n%e %e %e\n",
        //         dL_dM(0, 0), dL_dM(0, 1), dL_dM(0, 2),
        //         dL_dM(1, 0), dL_dM(1, 1), dL_dM(1, 2),
        //         dL_dM(2, 0), dL_dM(2, 1), dL_dM(2, 2));

        // Gradient of loss wrt to the gaussian orientation in the tangent space

        // Eigen::Vector3f dL_dq = -10. * eta_n * (R.col(0).cross(dL_dM.col(0)) + R.col(1).cross(dL_dM.col(1)) + R.col(2).cross(dL_dM.col(2)));
        Eigen::Vector3f dL_dq = -/*10.f **/ eta_n * (R.row(0).cross(dL_dM.col(0)) + R.row(1).cross(dL_dM.col(1)) + R.row(2).cross(dL_dM.col(2)));

        // Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
        // Eigen::Vector3f dL_dq = -10.f * eta_n * (I.row(0).cross(dL_dM.col(0)) + I.row(1).cross(dL_dM.col(1)) + I.row(2).cross(dL_dM.col(2)));

        // printf("dL_dq : %f %f %f\n", dL_dq.x(), dL_dq.y(), dL_dq.z());

        q_gauss = q_gauss * Eigen::Quaternionf(1.f, dL_dq.x(), dL_dq.y(), dL_dq.z());
        // q_gauss = Eigen::Quaternionf(1.f, dL_dq.x(), dL_dq.y(), dL_dq.z()) * q_gauss;
        q_gauss.normalize();

        // update scale
        // TODO : check saturation
        // scales[idx] = min(make_float3(0.15f), make_float3(expf(scale_log.x),
        //                                                   expf(scale_log.y),
        //                                                   expf(scale_log.z)));

        // scales[idx] = max(make_float3(0.001f), min(make_float3(0.2f), scales[idx] + eta * make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z())));

        // scales[idx] = max(make_float3(0.001f), min(make_float3(0.4f), make_float3(expf(scale_log.x),
        //                                                                           expf(scale_log.y),
        //                                                                           expf(scale_log.z))));
        /* Sigmoid scale */
        /*scales[idx] = max(make_float3(0.0001f), make_float3(expf(scale_log.x),
                                                            expf(scale_log.y),
                                                            expf(scale_log.z)));
        */
        /* Linear scale */
        scales[idx] = max(make_float3(0.0001f), scale + eta_n * make_float3(dL_dscale.x(),
                                                                            dL_dscale.y(),
                                                                            dL_dscale.z()));

        /*
        printf("dl_mean2d : %f %f\n", dl_mean2d.x(), dl_mean2d.y());
        printf("dl_mean3d : %f %f %f\n", dl_mean3d.x(), dl_mean3d.y(), dl_mean3d.z());
        printf("mu_cam : %f %f %f\n", mu_cam.x(), mu_cam.y(), mu_cam.z());
        printf("Ji :\n %f %f\n %f %f\n%f %f\n",
               Ji(0, 0), Ji(0, 1),
               Ji(1, 0), Ji(1, 1),
               Ji(2, 0), Ji(2, 1));
        */
        /*
        printf("dl_alpha : %f\n", dg.alpha / dg.n);
        printf("dl_dinvcov2D : %f %f %f\n", dL_dconic.x, dL_dconic.z, dL_dconic.z);
        */
        // printf("dl_dcov2D : %f %f %f\n", dL_da, dL_db, dL_dc);
        // printf("dl_dscale : %f %f %f\n", dL_dscale.x(), dL_dscale.y(), dL_dscale.z());
        // printf("scale : %f %f %f\n", scales[idx].x, scales[idx].y, scales[idx].z);
        // printf("dL_dq : %f %f %f\n", dL_dq.x(), dL_dq.y(), dL_dq.z());
        // printf("dL_dR : \n%f %f %f\n%f %f %f\n%f %f %f\n",
        //        dL_dM(0, 0), dL_dM(0, 1), dL_dM(0, 2),
        //        dL_dM(1, 0), dL_dM(1, 1), dL_dM(1, 2),
        //        dL_dM(2, 0), dL_dM(2, 1), dL_dM(2, 2));

        /*if (idx == 742)
        {
            printf("%f %f %f\n\n %f %f\n%f %f\n%f %f\n\n",
                   p_gauss.x(), p_gauss.y(), p_gauss.z(),
                   J(0, 0), J(1, 0),
                   J(0, 1), J(1, 1),
                   J(0, 2), J(1, 2));

            printf("delta mean : %f %f -> %f %f %f\n", dl_mean2d.x(), dl_mean2d.y(), dl_mean3d.x(), dl_mean3d.y(), dl_mean3d.z());
        }
        */
    }

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
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        DeltaGaussian2D dg = deltaGaussians2D[idx];

        DeltaGaussian3D dg3D;
        if (dg.n == 0)
        {
            deltaGaussians3D[idx].n = 0;
            return;
        }

        dg3D.n = dg.n;
        const float eta_n = 1.f / (5.f + dg.n);

        dg3D.color = eta_n * dg.color;
        dg3D.alpha = eta_n * dg.alpha;

        // compute delta mean3D
        Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);
        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);
        Eigen::Map<const Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
        Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);

        Eigen::Matrix<float, 3, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                                     {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())},
                                     {0.f, 0.f, 1.f}};

        Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
        Eigen::Matrix3f R = q_gauss.toRotationMatrix();

        Eigen::Vector3f dl_mean(dg.meanImg.x /*/ dg.n*/, dg.meanImg.y /*/ dg.n*/, dg.depth /*/ dg.n*/);

        /* compute derivative of inverse Cov2 wrt cov2*/
        const Eigen::Matrix<float, 3, 3> T = J * W;
        const Eigen::Matrix3f RS = R * s_gauss.asDiagonal();
        const Eigen::Matrix<float, 3, 3> M = T * RS;
        const Eigen::Matrix3f Sigma = M * M.transpose();
        Eigen::Matrix3f Sigma_i = Sigma.inverse();

        Eigen::Vector3f dl_mean2d(dg.meanImg.x /*/ dg.n*/, dg.meanImg.y /*/ dg.n*/, 0.);
        Eigen::Vector3f dl_mean3d = q_cam * (J.transpose() * dl_mean2d + dg.depth * Eigen::Vector3f(mu_cam.x() / mu_cam.z(), mu_cam.y() / mu_cam.z(), 1.f));
        // Eigen::Vector3f dl_mean3d = q_cam * (J.transpose() * dl_mean);

        Eigen::Vector2f dl_pHat(dg.pHat.x /*/ dg.n*/, dg.pHat.y /*/ dg.n*/);

        // derivatives of pHat wrt position
        float norm2_mucam = mu_cam.dot(mu_cam);
        float norm_mucam = sqrtf(norm2_mucam);
        float demon_phat_mean = 1.f / (norm2_mucam * norm_mucam);
        Eigen::Vector3f dpHat_mean(-mu_cam.x() * mu_cam.z() * demon_phat_mean,
                                   -mu_cam.y() * mu_cam.z() * demon_phat_mean,
                                   (mu_cam.x() * mu_cam.x() + mu_cam.y() * mu_cam.y()) * demon_phat_mean);
        dl_mean3d += (q_cam * (dpHat_mean * ((Sigma_i(2, 0) * dl_pHat.x() + Sigma_i(2, 1) * dl_pHat.y()) / Sigma_i(2, 2))));

        float a = Sigma(0, 0) + 0.001f;
        float b = Sigma(1, 0);
        float c = Sigma(1, 1) + 0.001f;

        float denom = a * c - b * b;
        float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

        float3 dL_dconic = dg.invSigmaImg /*/ dg.n*/;

        // printf("dL_dconic : %f %f %f\n", dL_dconic.x, dL_dconic.y, dL_dconic.z);

        // derivative of Cov2 wrt Cov2^-1 :
        float dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
        float dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
        float dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

        // printf("Cov2 : %f %f %f\n", a, b, c);
        // printf("dL_Cov2 : %f %f %f\n", dL_da, dL_db, dL_dc);

        // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
        // cov2D = T * Cov3d * T;
        const Eigen::Matrix3f Cov3D = RS * (RS.transpose());

        float dL_dT00 = 2 * (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_da +
                        (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_db;
        float dL_dT01 = 2 * (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_da +
                        (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_db;
        float dL_dT02 = 2 * (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_da +
                        (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_db;
        float dL_dT10 = 2 * (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_db;
        float dL_dT11 = 2 * (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_db;
        float dL_dT12 = 2 * (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_dc +
                        (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_db;

        // derivative of qHat wrt Sigma_i
        float zc_tc = mu_cam.z() / norm_mucam;
        Eigen::Matrix3f dL_dSigma_i;
        // CHECK HERE
        dL_dSigma_i << 0.f, 0.f, 0.5f * zc_tc * dl_pHat.x() / Sigma_i(2, 2),
            0.f, 0.f, 0.5f * zc_tc * dl_pHat.y() / Sigma_i(2, 2),
            0.5f * zc_tc * dl_pHat.x() / Sigma_i(2, 2), 0.5f * zc_tc * dl_pHat.y() / Sigma_i(2, 2), -zc_tc * (dl_pHat.x() * Sigma_i(2, 0) + dl_pHat.y() * Sigma_i(2, 1)) / (Sigma_i(2, 2) * Sigma_i(2, 2));

        Eigen::Matrix3f dL_dSigma = -Sigma_i * dL_dSigma_i * Sigma_i;

        Eigen::Matrix3f TCov3D = T * Cov3D;
        Eigen::Matrix3f Cov3DT = Cov3D * T.transpose();
        dL_dT00 += TCov3D.col(0).dot(dL_dSigma.col(0)) + Cov3DT.row(0).dot(dL_dSigma.row(0));
        dL_dT01 += TCov3D.col(1).dot(dL_dSigma.col(0)) + Cov3DT.row(1).dot(dL_dSigma.row(0));
        dL_dT02 += TCov3D.col(2).dot(dL_dSigma.col(0)) + Cov3DT.row(2).dot(dL_dSigma.row(0));
        dL_dT10 += TCov3D.col(0).dot(dL_dSigma.col(1)) + Cov3DT.row(0).dot(dL_dSigma.row(1));
        dL_dT11 += TCov3D.col(1).dot(dL_dSigma.col(1)) + Cov3DT.row(1).dot(dL_dSigma.row(1));
        dL_dT12 += TCov3D.col(2).dot(dL_dSigma.col(1)) + Cov3DT.row(2).dot(dL_dSigma.row(1));

        // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
        // T = J * W
        float dL_dJ00 = W(0, 0) * dL_dT00 + W(0, 1) * dL_dT01 + W(0, 2) * dL_dT02;
        float dL_dJ02 = W(2, 0) * dL_dT00 + W(2, 1) * dL_dT01 + W(2, 2) * dL_dT02;
        float dL_dJ11 = W(1, 0) * dL_dT10 + W(1, 1) * dL_dT11 + W(1, 2) * dL_dT12;
        float dL_dJ12 = W(2, 0) * dL_dT10 + W(2, 1) * dL_dT11 + W(2, 2) * dL_dT12;

        float tz = 1.f / mu_cam.z();
        float tz2 = tz * tz;
        float tz3 = tz2 * tz;

        // Gradients of loss w.r.t. transformed Gaussian mean t
        float dL_dtx = -cameraParams.f.x * tz2 * dL_dJ02;
        float dL_dty = -cameraParams.f.y * tz2 * dL_dJ12;
        float dL_dtz = -cameraParams.f.x * tz2 * dL_dJ00 - cameraParams.f.y * tz2 * dL_dJ11 + (2 * cameraParams.f.x * mu_cam.x()) * tz3 * dL_dJ02 + (2 * cameraParams.f.y * mu_cam.y()) * tz3 * dL_dJ12;

        // Eigen::Vector3f dl_dmean = q_cam.inverse() * Eigen::Vector3f(dL_dtx, dL_dty, dL_dtz);
        dl_mean3d += q_cam * Eigen::Vector3f(dL_dtx, dL_dty, dL_dtz);

        /*update position from 2D + z mean + covariance grad*/

        dg3D.position.x = eta_n * dl_mean3d.x();
        dg3D.position.y = eta_n * dl_mean3d.y();
        dg3D.position.z = eta_n * dl_mean3d.z();

        Eigen::Matrix3f dL_dcov3D;
        dL_dcov3D(0, 0) = (T(0, 0) * T(0, 0) * dL_da + T(0, 0) * T(1, 0) * dL_db + T(1, 0) * T(1, 0) * dL_dc);
        dL_dcov3D(1, 1) = (T(0, 1) * T(0, 1) * dL_da + T(0, 1) * T(1, 1) * dL_db + T(1, 1) * T(1, 1) * dL_dc);
        dL_dcov3D(2, 2) = (T(0, 2) * T(0, 2) * dL_da + T(0, 2) * T(1, 2) * dL_db + T(1, 2) * T(1, 2) * dL_dc);
        dL_dcov3D(1, 0) = dL_dcov3D(0, 1) = T(0, 0) * T(0, 1) * dL_da + 0.5f * (T(0, 0) * T(1, 1) + T(0, 1) * T(1, 0)) * dL_db + T(1, 0) * T(1, 1) * dL_dc;
        dL_dcov3D(2, 0) = dL_dcov3D(0, 2) = T(0, 0) * T(0, 2) * dL_da + 0.5f * (T(0, 0) * T(1, 2) + T(0, 2) * T(1, 0)) * dL_db + T(1, 0) * T(1, 2) * dL_dc;
        dL_dcov3D(2, 1) = dL_dcov3D(1, 2) = T(0, 2) * T(0, 1) * dL_da + 0.5f * (T(0, 1) * T(1, 2) + T(0, 2) * T(1, 1)) * dL_db + T(1, 1) * T(1, 2) * dL_dc;

        // d_hat covariance
        dL_dcov3D += T.transpose() * dL_dSigma * T;

        Eigen::Matrix3f dL_dM = 2.0f * RS.transpose() * dL_dcov3D;
        Eigen::Vector3f dL_dscale = Eigen::Vector3f(R(0, 0) * dL_dM(0, 0) + R(1, 0) * dL_dM(0, 1) + R(2, 0) * dL_dM(0, 2),
                                                    R(0, 1) * dL_dM(1, 0) + R(1, 1) * dL_dM(1, 1) + R(2, 1) * dL_dM(1, 2),
                                                    R(0, 2) * dL_dM(2, 0) + R(1, 2) * dL_dM(2, 1) + R(2, 2) * dL_dM(2, 2));

        const float3 scale = scales[idx];
        float3 scale_log = make_float3(logf(scale.x),
                                       logf(scale.y),
                                       logf(scale.z));

        // isotropic regulation
        float mean_scale = (scale.x + scale.y + scale.z) / 3.f;
        float3 dl_iso = make_float3(scale.x - mean_scale,
                                    scale.y - mean_scale,
                                    scale.z - mean_scale);
        dL_dscale -= lambda_iso * (1.f / 3.f) * Eigen::Vector3f(2.f * dl_iso.x - dl_iso.y - dl_iso.z, -dl_iso.x + 2.f * dl_iso.y - dl_iso.z, -dl_iso.x - dl_iso.y + 2.f * dl_iso.z);

        // printf("scale_log : %f %f %f\n", scale_log.x, scale_log.y, scale_log.z);
        float3 dL_dscale_log = make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z()) / scale;
        // printf("dL_dscale_log : %f %f %f\n", dL_dscale_log.x, dL_dscale_log.y, dL_dscale_log.z);
        scale_log += 10.f * eta_n * min(make_float3(2.f, 2.f, 2.f), dL_dscale_log);
        // scale_log += eta * make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z()) / scale;

        // printf("scale_log updated : %f %f %f\n", scale_log.x, scale_log.y, scale_log.z);

        // dL_dM = s_gauss.asDiagonal() * dL_dM;
        dL_dM.row(0) *= scale.x;
        dL_dM.row(1) *= scale.y;
        dL_dM.row(2) *= scale.z;
        // Gradient of loss wrt to the gaussian orientation in the tangent space

        Eigen::Vector3f dL_dq = -/*10.f **/ eta_n * (R.row(0).cross(dL_dM.col(0)) + R.row(1).cross(dL_dM.col(1)) + R.row(2).cross(dL_dM.col(2)));

        dg3D.orientation.x = dL_dq.x();
        dg3D.orientation.y = dL_dq.y();
        dg3D.orientation.z = dL_dq.z();

        // update scale
        // TODO : check saturation
        // scales[idx] = min(make_float3(0.15f), make_float3(expf(scale_log.x),
        //                                                   expf(scale_log.y),
        //                                                   expf(scale_log.z)));

        // scales[idx] = max(make_float3(0.001f), min(make_float3(0.2f), scales[idx] + eta * make_float3(dL_dscale.x(), dL_dscale.y(), dL_dscale.z())));

        // scales[idx] = max(make_float3(0.001f), min(make_float3(0.4f), make_float3(expf(scale_log.x),
        //                                                                           expf(scale_log.y),
        //                                                                           expf(scale_log.z))));
        /* Sigmoid scale */
        /*scales[idx] = max(make_float3(0.0001f), make_float3(expf(scale_log.x),
                                                            expf(scale_log.y),
                                                            expf(scale_log.z)));
        */
        /* Linear scale */
        dg3D.scale.x = eta_n * dL_dscale.x();
        dg3D.scale.y = eta_n * dL_dscale.y();
        dg3D.scale.z = eta_n * dL_dscale.z();

        deltaGaussians3D[idx] = dg3D;
    }

    inline __device__ float updateAdam(float &m,
                                       float &v,
                                       float grad,
                                       const float eta,
                                       const float alpha1,
                                       const float beta1,
                                       const float beta1t,
                                       const float alpha2,
                                       const float beta2,
                                       const float beta2t,
                                       const float epsilon)
    {
        m = alpha1 * grad + beta1 * m;
        v = alpha2 * grad * grad + beta2 * v;
        // return eta * m * __frsqrt_rn(v + epsilon);
        float m_hat = beta1t * m;
        float v_hat = beta2t * v;
        return eta * m_hat * __frsqrt_rn(v_hat + epsilon);
    }

    inline __device__ float3 updateAdam(float3 &m,
                                        float3 &v,
                                        float3 grad,
                                        const float eta,
                                        const float alpha1,
                                        const float beta1,
                                        const float beta1t,
                                        const float alpha2,
                                        const float beta2,
                                        const float beta2t,
                                        const float epsilon)
    {
        float3 res;

        res.x = updateAdam(m.x, v.x, grad.x,
                           eta,
                           alpha1, beta1, beta1t,
                           alpha2, beta2, beta2t,
                           epsilon);
        res.y = updateAdam(m.y, v.y, grad.y,
                           eta,
                           alpha1, beta1, beta1t,
                           alpha2, beta2, beta2t,
                           epsilon);
        res.z = updateAdam(m.z, v.z, grad.z,
                           eta,
                           alpha1, beta1, beta1t,
                           alpha2, beta2, beta2t,
                           epsilon);
        return res;
    }

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
        int nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        DeltaGaussian3D dg3D = deltaGaussians3D[idx];
        if (dg3D.n == 0)
            return;
        AdamStateGaussian3D adamState = adamStates[idx];

        adamState.t += 1.f;

        float alpha1 = 1.f - beta1;
        float alpha2 = 1.f - beta2;

        float beta1t = __frcp_rn(1.f - __powf(beta1, adamState.t));
        float beta2t = __frcp_rn(1.f - __powf(beta2, adamState.t));

        float3 position = positions[idx];
        float3 scale = scales[idx];
        float4 orientation = orientations[idx];
        float3 color = colors[idx];
        float alpha = alphas[idx];

        position += updateAdam(adamState.m_position,
                               adamState.v_position,
                               dg3D.position,
                               eta,
                               alpha1, beta1, beta1t,
                               alpha2, beta2, beta2t,
                               epsilon);

        float3 dq = updateAdam(adamState.m_orientation,
                               adamState.v_orientation,
                               dg3D.orientation,
                               eta,
                               alpha1, beta1, beta1t,
                               alpha2, beta2, beta2t,
                               epsilon);
        Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientation);
        q_gauss = q_gauss * Eigen::Quaternionf(1.f, 0.5f * dq.x, 0.5f * dq.y, 0.5f * dq.z);
        q_gauss.normalize();

        scale += updateAdam(adamState.m_scale,
                            adamState.v_scale,
                            dg3D.scale,
                            eta,
                            alpha1, beta1, beta1t,
                            alpha2, beta2, beta2t,
                            epsilon);

        color += updateAdam(adamState.m_color,
                            adamState.v_color,
                            dg3D.color,
                            eta,
                            alpha1, beta1, beta1t,
                            alpha2, beta2, beta2t,
                            epsilon);

        float dalpha = updateAdam(adamState.m_alpha,
                                  adamState.v_alpha,
                                  dg3D.alpha,
                                  eta,
                                  alpha1, beta1, beta1t,
                                  alpha2, beta2, beta2t,
                                  epsilon);
        float alpha_s = __logf(alpha / (1.f - alpha)) + dalpha / (alpha - alpha * alpha);
        alpha = max(0.01f, min(0.99f, 1.f / (1.f + __expf(-alpha_s))));

        positions[idx] = position;
        scales[idx] = max(make_float3(0.001f), scale);
        orientations[idx] = orientation;
        colors[idx] = min(make_float3(1.f, 1.f, 1.f), max(make_float3(0.f, 0.f, 0.f), color));
        alphas[idx] = alpha;

        adamStates[idx] = adamState;
    }

    __global__ void optimizePose_kernel(
        DeltaGaussian2D *__restrict__ deltaGaussians,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgSigmas,
        const float3 *__restrict__ imgInvSigmas,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (range.x + tid < range.y)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
        }
        block.sync();

        if (x >= width || y >= height)
            return;

        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);
        float depth = 0.f;
        float T = 1.f;

        for (int i = 0; i < n; i++)
        {
            const float dx = x - splattedGaussian_sh[i].position.x;
            const float dy = y - splattedGaussian_sh[i].position.y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
            if (alpha_i < 1.f / 255.f)
                continue;

            color += splattedGaussian_sh[i].color * alpha_i * prod_alpha;
#ifdef USE_MEAN_DEPTH
            // mean depth
            depth += splattedGaussian_sh[i].position.z * alpha_i * prod_alpha;
#endif
            prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
            // median depth
            if (T > 0.5f && prod_alpha < 0.5)
            {
                T = prod_alpha;
                depth = splattedGaussian_sh[i].position.z;
            }
#endif
            if (prod_alpha < 0.001f)
            {
                break;
            }
        }

        // const float final_T = prod_alpha;
        if (prod_alpha > 0.2f)
        {
            return;
        }

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        const float3 color_error = color - make_float3(rgba.x / 255.f,
                                                       rgba.y / 255.f,
                                                       rgba.z / 255.f);

        const float imgDepth = tex2D<float>(texDepth, x, y);

        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        block.sync();

        prod_alpha = 1.f;
        T = 1.f;

        float3 acc_c = make_float3(0.f);
        float alpha_prev = 0.f;
        float3 color_prev = make_float3(0.f);
        // float T = final_T;

        for (int i = 0; i < n; i++)
        {
            const float dx = x - splattedGaussian_sh[i].position.x;
            const float dy = y - splattedGaussian_sh[i].position.y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float G = expf(-0.5f * v);
            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
            if (alpha_i < 1.f / 255.f)
                continue;

            float3 d_alpha = splattedGaussian_sh[i].color * prod_alpha;

            acc_c += splattedGaussian_sh[i].color * alpha_i * prod_alpha;

            d_alpha -= (color - acc_c) / (1.f - alpha_i);

            d_alpha = -color_error * d_alpha;

            const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;
            // const float dl_alpha = d_alpha.x;

            const float G_dl = G * dl_alpha;
            const float a_G_dl = alpha_i * dl_alpha;
            // const float a_G_dl = splattedGaussian_sh[i].alpha * /* alpha_i **/ dl_alpha;

            // float3 dl_alpha_i = G * (splattedGaussian_sh[i].alpha * (1.f - splattedGaussian_sh[i].alpha)) * d_alpha;
            // deltaGaussian_sh[tid].alpha = (dl_alpha_i.x+dl_alpha_i.y+dl_alpha_i.z)/3.f;

            uint32_t gid = gids_sh[i];

#ifdef USE_MEAN_DEPTH
            // mean depth
            // dg.depth = -alpha_i * prod_alpha * depth_error;
            atomicAdd(&deltaGaussians[gid].depth, -alpha_i * prod_alpha * depth_error);
#endif

            atomicAdd(&deltaGaussians[gid].meanImg.x, a_G_dl * (splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy));
            atomicAdd(&deltaGaussians[gid].meanImg.y, a_G_dl * (splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy));

            prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
            // median depth
            if (T > 0.5f && prod_alpha <= 0.5f)
            {
                T = prod_alpha;
                // dg.depth = - depth_error;
                atomicAdd(&deltaGaussians[gid].depth, -depth_error);
            }
            // else{
            //     dg.depth = 0.f;
            // }
#endif
            atomicAggInc(&deltaGaussians[gid].n);
            // atomicAdd(&deltaGaussians[gid].n, 1.f);
            if (prod_alpha < 0.001f)
            {
                break;
            }
        }
    }

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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (range.x + tid < range.y)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
        }
        block.sync();

        if (x >= width || y >= height)
            return;

        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);
        float depth = 0.f;
        float T = 1.f;

        for (int i = 0; i < n; i++)
        {
            const float dx = x - splattedGaussian_sh[i].position.x;
            const float dy = y - splattedGaussian_sh[i].position.y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
            if (alpha_i < 1.f / 255.f)
                continue;

            color += splattedGaussian_sh[i].color * alpha_i * prod_alpha;
#ifdef USE_MEAN_DEPTH
            // mean depth
            depth += splattedGaussian_sh[i].position.z * alpha_i * prod_alpha;
#endif
            prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
            // median depth
            if (T > 0.5f && prod_alpha < 0.5)
            {
                T = prod_alpha;
                depth = splattedGaussian_sh[i].position.z;
            }
#endif
            if (prod_alpha < 0.001f)
            {
                break;
            }
        }

        // const float final_T = prod_alpha;
        if (prod_alpha > 0.2f)
        {
            return;
        }

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        const float3 color_error = color - make_float3(rgba.x / 255.f,
                                                       rgba.y / 255.f,
                                                       rgba.z / 255.f);

        const float imgDepth = tex2D<float>(texDepth, x, y);

        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        block.sync();

        prod_alpha = 1.f;
        T = 1.f;

        float3 acc_c = make_float3(0.f);
        float alpha_prev = 0.f;
        float3 color_prev = make_float3(0.f);
        // float T = final_T;

        // Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
        //                              {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                            (y - cameraParams.c.y) / cameraParams.f.y,
                            1.f);
        DeltaPose3D dl_pose;
        dl_pose.dp = make_float3(0.f);
        dl_pose.dq = make_float3(0.f);
        dl_pose.n = 0;
        Eigen::Map<Eigen::Vector3f> dp((float *)&dl_pose.dp);
        Eigen::Map<Eigen::Vector3f> dq((float *)&dl_pose.dq);

        for (int i = 0; i < n; i++)
        {
            float z = splattedGaussian_sh[i].position.z;
            Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / z, 0.f},
                                          {0.f, cameraParams.f.y / z},
                                          {-cameraParams.f.x * ray.x() / z, -cameraParams.f.y * ray.y() / z}};

            const float dx = x - splattedGaussian_sh[i].position.x;
            const float dy = y - splattedGaussian_sh[i].position.y;
            const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

            const float G = expf(-0.5f * v);
            const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
            if (alpha_i < 1.f / 255.f)
                continue;

            float3 d_alpha = splattedGaussian_sh[i].color * prod_alpha;

            acc_c += splattedGaussian_sh[i].color * alpha_i * prod_alpha;

            d_alpha -= (color - acc_c) / (1.f - alpha_i);

            d_alpha = -color_error * d_alpha;

            const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;
            // const float dl_alpha = d_alpha.x;

            const float G_dl = G * dl_alpha;
            const float a_G_dl = alpha_i * dl_alpha;

            uint32_t gid = gids_sh[i];

            Eigen::Vector2f dl_mean2d(a_G_dl * (splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy),
                                      a_G_dl * (splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy));

            Eigen::Vector3f d_mu_cam = -Jt * dl_mean2d;

#ifdef USE_MEAN_DEPTH
            // mean depth
            d_mu_cam += alpha_i * prod_alpha * depth_error * ray;

#endif

            prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
            // median depth
            if (T > 0.5f && prod_alpha <= 0.5f)
            {
                T = prod_alpha;
                d_mu_cam += depth_error * ray;
            }
            // else{
            //     dg.depth = 0.f;
            // }
#endif

            dp += q_cam * d_mu_cam;
            dq += q_cam * (z * ray.cross(d_mu_cam));
            dl_pose.n++;

            // atomicAggInc(&deltaGaussians[gid].n);
            //  atomicAdd(&deltaGaussians[gid].n, 1.f);
            if (prod_alpha < 0.001f)
            {
                break;
            }
        }

        atomicAdd(&deltaPose->dp.x, dl_pose.dp.x);
        atomicAdd(&deltaPose->dp.y, dl_pose.dp.y);
        atomicAdd(&deltaPose->dp.z, dl_pose.dp.z);
        atomicAdd(&deltaPose->dq.x, dl_pose.dq.x);
        atomicAdd(&deltaPose->dq.y, dl_pose.dq.y);
        atomicAdd(&deltaPose->dq.z, dl_pose.dq.z);
        atomicAdd(&deltaPose->n, dl_pose.n);
    }

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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];
        __shared__ DeltaPose3D deltaPose_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (range.x + tid < range.y)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
        }

        DeltaPose3D &dl_pose = deltaPose_sh[tid];

        deltaPose_sh[tid].dp = make_float3(0.f);
        deltaPose_sh[tid].dq = make_float3(0.f);
        deltaPose_sh[tid].n = 0;

        block.sync();

        if (x < width && y < height)
        {
            float prod_alpha = 1.f;
            float3 color = make_float3(0.f);
            float depth = 0.f;
            float T = 1.f;

            for (int i = 0; i < n; i++)
            {
                const float dx = x - splattedGaussian_sh[i].position.x;
                const float dy = y - splattedGaussian_sh[i].position.y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f)
                    continue;

                color += splattedGaussian_sh[i].color * alpha_i * prod_alpha;
#ifdef USE_MEAN_DEPTH
                // mean depth
                depth += splattedGaussian_sh[i].position.z * alpha_i * prod_alpha;
#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha < 0.5)
                {
                    T = prod_alpha;
                    depth = splattedGaussian_sh[i].position.z;
                }
#endif
                if (prod_alpha < 0.001f)
                {
                    break;
                }
            }

            // const float final_T = prod_alpha;
            if (prod_alpha < 0.1f)
            {

                uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

                const float3 color_error = color - make_float3(rgba.x / 255.f,
                                                               rgba.y / 255.f,
                                                               rgba.z / 255.f);

                const float imgDepth = tex2D<float>(texDepth, x, y);

                const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

                prod_alpha = 1.f;
                T = 1.f;

                float3 acc_c = make_float3(0.f);
                float alpha_prev = 0.f;
                float3 color_prev = make_float3(0.f);
                // float T = final_T;

                // Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                //                              {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

                Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                                    (y - cameraParams.c.y) / cameraParams.f.y,
                                    1.f);
                Eigen::Map<Eigen::Vector3f> dp((float *)&dl_pose.dp);
                Eigen::Map<Eigen::Vector3f> dq((float *)&dl_pose.dq);

                for (int i = 0; i < n; i++)
                {
                    float z = splattedGaussian_sh[i].position.z;
                    Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / z, 0.f},
                                                  {0.f, cameraParams.f.y / z},
                                                  {-cameraParams.f.x * ray.x() / z, -cameraParams.f.y * ray.y() / z}};

                    const float dx = x - splattedGaussian_sh[i].position.x;
                    const float dy = y - splattedGaussian_sh[i].position.y;
                    const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                    const float G = expf(-0.5f * v);
                    const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
                    if (alpha_i < 1.f / 255.f)
                        continue;

                    float3 d_alpha = splattedGaussian_sh[i].color * prod_alpha;

                    acc_c += splattedGaussian_sh[i].color * alpha_i * prod_alpha;

                    d_alpha -= (color - acc_c) / (1.f - alpha_i);

                    d_alpha = -color_error * d_alpha;

                    const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;
                    // const float dl_alpha = d_alpha.x;

                    const float G_dl = G * dl_alpha;
                    const float a_G_dl = alpha_i * dl_alpha;

                    uint32_t gid = gids_sh[i];

                    Eigen::Vector2f dl_mean2d(a_G_dl * (splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy),
                                              a_G_dl * (splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy));

                    Eigen::Vector3f d_mu_cam = -Jt * dl_mean2d;

#ifdef USE_MEAN_DEPTH
                    // mean depth
                    d_mu_cam += alpha_i * prod_alpha * depth_error * ray;

#endif

                    prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                    // median depth
                    if (T > 0.5f && prod_alpha <= 0.5f)
                    {
                        T = prod_alpha;
                        d_mu_cam += depth_error * ray;
                    }
                    // else{
                    //     dg.depth = 0.f;
                    // }
#endif

                    dp += q_cam * d_mu_cam;
                    dq += /*q_cam **/ (z * ray.cross(d_mu_cam));
                    dl_pose.n++;

                    // atomicAggInc(&deltaGaussians[gid].n);
                    //  atomicAdd(&deltaGaussians[gid].n, 1.f);
                    if (prod_alpha < 0.001f)
                    {
                        break;
                    }
                }
            }
        }

        block.sync();
        reduce<BLOCK_SIZE>(deltaPose_sh, tid);
        if (tid == 0)
        {
            DeltaPose3D &delta = deltaPose_sh[0];
            atomicAdd(&deltaPose->dp.x, delta.dp.x);
            atomicAdd(&deltaPose->dp.y, delta.dp.y);
            atomicAdd(&deltaPose->dp.z, delta.dp.z);
            atomicAdd(&deltaPose->dq.x, delta.dq.x);
            atomicAdd(&deltaPose->dq.y, delta.dq.y);
            atomicAdd(&deltaPose->dq.z, delta.dq.z);
            atomicAdd(&deltaPose->n, delta.n);
        }
    }

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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];
        __shared__ MotionTrackingData mtd_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (range.x + tid < range.y)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
        }

        block.sync();

        float JtJ_data[36];
        // float Jtr_data[6];
        // Eigen::Matrix<float, 6, 6> JtJ = Eigen::Matrix<float, 6, 6>::Zero();
        // Eigen::Vector<float, 6> Jtr = Eigen::Vector<float, 6>::Zero();

        MotionTrackingData &mtd_i = mtd_sh[tid];

        Eigen::Map<Eigen::Matrix<float, 6, 6>> JtJ(JtJ_data);
        Eigen::Map<Eigen::Vector<float, 6>> Jtr(mtd_i.Jtr);

        JtJ.setZero();
        Jtr.setZero();

        // MotionTrackingData &mtd_i = mtd_sh[tid];
        //  #pragma unroll
        //  for(int i=0; i<21; i++)
        //  {
        //      mtd_i.JtJ[i]=0.f;
        //  }
        //  #pragma unroll
        //  for(int i=0; i<6; i++)
        //  {
        //      mtd_i.Jtr[i]=0.f;
        //  }

        if (x < width && y < height)
        {
            float prod_alpha = 1.f;
            Eigen::Vector3f color(0.f, 0.f, 0.f);
            float depth = 0.f;
            float T = 1.f;

            for (int i = 0; i < n; i++)
            {
                const float dx = x - splattedGaussian_sh[i].position.x;
                const float dy = y - splattedGaussian_sh[i].position.y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f)
                    continue;

                color.x() += splattedGaussian_sh[i].color.x * alpha_i * prod_alpha;
                color.y() += splattedGaussian_sh[i].color.y * alpha_i * prod_alpha;
                color.z() += splattedGaussian_sh[i].color.z * alpha_i * prod_alpha;
#ifdef USE_MEAN_DEPTH
                // mean depth
                depth += splattedGaussian_sh[i].position.z * alpha_i * prod_alpha;
#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha < 0.5)
                {
                    T = prod_alpha;
                    depth = splattedGaussian_sh[i].position.z;
                }
#endif
                if (prod_alpha < 0.001f)
                {
                    break;
                }
            }

            // const float final_T = prod_alpha;
            if (prod_alpha < alphaThresh)
            {
                color += prod_alpha * Eigen::Vector3f(bgColor.x, bgColor.y, bgColor.z);

                uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

                const Eigen::Vector3f color_error = color - Eigen::Vector3f(rgba.x / 255.f,
                                                                            rgba.y / 255.f,
                                                                            rgba.z / 255.f);

                const float imgDepth = tex2D<float>(texDepth, x, y);

                const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

                prod_alpha = 1.f;
                T = 1.f;

                Eigen::Vector3f acc_c(0.f, 0.f, 0.f);
                float alpha_prev = 0.f;
                Eigen::Vector3f color_prev(0.f, 0.f, 0.f);
                // float T = final_T;

                // Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                //                              {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

                Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                                    (y - cameraParams.c.y) / cameraParams.f.y,
                                    1.f);

                Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                          {ray.z(), 0.f, -ray.x()},
                                          {-ray.y(), ray.x(), 0.f}};

                Eigen::Matrix3f R = q_cam.toRotationMatrix();

                for (int i = 0; i < n; i++)
                {
                    float z = splattedGaussian_sh[i].position.z;
                    // Eigen::Vector3f ray((splattedGaussian_sh[i].position.x - cameraParams.c.x) / cameraParams.f.x,
                    //                     (splattedGaussian_sh[i].position.y - cameraParams.c.y) / cameraParams.f.y,
                    //                     1.f);

                    // Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                    //                           {ray.z(), 0.f, -ray.x()},
                    //                           {-ray.y(), ray.x(), 0.f}};

                    Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / z, 0.f},
                                                  {0.f, cameraParams.f.y / z},
                                                  {-cameraParams.f.x * ray.x() / z, -cameraParams.f.y * ray.y() / z}};

                    const float dx = x - splattedGaussian_sh[i].position.x;
                    const float dy = y - splattedGaussian_sh[i].position.y;
                    const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                    const float G = expf(-0.5f * v);
                    const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
                    if (alpha_i < 1.f / 255.f)
                        continue;

                    Eigen::Vector3f d_alpha(splattedGaussian_sh[i].color.x * prod_alpha,
                                            splattedGaussian_sh[i].color.y * prod_alpha,
                                            splattedGaussian_sh[i].color.z * prod_alpha);

                    acc_c += alpha_i * d_alpha;

                    d_alpha -= (color - acc_c) / (1.f - alpha_i);

                    // d_alpha = -color_error * d_alpha;

                    // const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;

                    // const float a_G_dl = alpha_i * dl_alpha;

                    // uint32_t gid = gids_sh[i];

                    Eigen::Vector2f dl_mean2d(splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy,
                                              splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);

                    // Eigen::Vector3f d_mu_cam = -Jt * dl_mean2d;

                    // huber loss for the color
                    float lc = color_error.norm();
                    float wc = lc < colorThresh ? 1.f : colorThresh / lc;

                    Eigen::Matrix3f Jt_cam = Jt * dl_mean2d * alpha_i * d_alpha.transpose();

                    Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
                    Eigen::Vector3f Jtr_cam = wc * Jt_cam * color_error;

#ifdef USE_MEAN_DEPTH
                    // mean depth
                    // d_mu_cam += alpha_i * prod_alpha * depth_error * ray;

#endif

                    prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                    // median depth
                    if (T > 0.5f && prod_alpha <= 0.5f)
                    {
                        T = prod_alpha;
                        // d_mu_cam += depth_error * ray;

                        float ld = fabsf(depth_error);
                        float wd = ld < depthThresh ? 1.f : depthThresh / ld;

                        JtJ_cam += wd * ray * ray.transpose();
                        Jtr_cam += wd * depth_error * ray;
                    }
                    // else{
                    //     dg.depth = 0.f;
                    // }
#endif
                    Eigen::Matrix<float, 6, 3> Jpose;
                    Jpose.block<3, 3>(0, 0) = R;
                    Jpose.block<3, 3>(3, 0) = /*R * */ z * ray_cross;

                    JtJ += Jpose * JtJ_cam * Jpose.transpose();
                    Jtr += Jpose * Jtr_cam;

                    // dp += q_cam * d_mu_cam;
                    // dq += q_cam * (z * ray.cross(d_mu_cam));
                    // dl_pose.n++;

                    if (prod_alpha < 0.001f)
                    {
                        break;
                    }
                }
            }
        }

        block.sync();

        int k = 0;
#pragma unroll
        for (int i = 0; i < 6; i++)
#pragma unroll
            for (int j = i; j < 6; j++, k++)
            {
                mtd_i.JtJ[k] = JtJ_data[6 * i + j];
            }

        /*float *Jtr_i = mtd_sh[tid].Jtr;
        #pragma unroll
        for (int i = 0; i < 6; i++)
        {
            Jtr_i[i] = Jtr_data[i];
        }
        */

        block.sync();

        reduce<BLOCK_SIZE>(mtd_sh, tid);

        if (tid == 0)
        {
            MotionTrackingData &mtd0 = mtd_sh[0];
            for (int i = 0; i < 21; i++)
            {
                atomicAdd(&mtd->JtJ[i], mtd0.JtJ[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                atomicAdd(&mtd->Jtr[i], mtd0.Jtr[i]);
            }
        }
    }

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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];
        __shared__ MotionTrackingData mtd_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = min(range.y - range.x, BLOCK_SIZE);

        // collect gaussians data
        if (range.x + tid < range.y)
        {
            uint32_t gid = indices[range.x + tid];
            gids_sh[tid] = gid;
            splattedGaussian_sh[tid].position = imgPositions[gid];
            splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
            splattedGaussian_sh[tid].color = colors[gid];
            splattedGaussian_sh[tid].alpha = alphas[gid];
            splattedGaussian_sh[tid].pHat = pHats[gid];
        }

        block.sync();

        float JtJ_data[36];
        // float Jtr_data[6];
        // Eigen::Matrix<float, 6, 6> JtJ = Eigen::Matrix<float, 6, 6>::Zero();
        // Eigen::Vector<float, 6> Jtr = Eigen::Vector<float, 6>::Zero();

        MotionTrackingData &mtd_i = mtd_sh[tid];

        Eigen::Map<Eigen::Matrix<float, 6, 6>> JtJ(JtJ_data);
        Eigen::Map<Eigen::Vector<float, 6>> Jtr(mtd_i.Jtr);

        JtJ.setZero();
        Jtr.setZero();

        // MotionTrackingData &mtd_i = mtd_sh[tid];
        //  #pragma unroll
        //  for(int i=0; i<21; i++)
        //  {
        //      mtd_i.JtJ[i]=0.f;
        //  }
        //  #pragma unroll
        //  for(int i=0; i<6; i++)
        //  {
        //      mtd_i.Jtr[i]=0.f;
        //  }

        if (x < width && y < height)
        {
            float prod_alpha = 1.f;
            Eigen::Vector3f color(0.f, 0.f, 0.f);
            float depth = 0.f;
            float T = 1.f;

            for (int i = 0; i < n; i++)
            {
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                if (alpha_i < 1.f / 255.f)
                    continue;

                float omega = alpha_i * prod_alpha;
                color.x() += splattedGaussian_sh[i].color.x * omega;
                color.y() += splattedGaussian_sh[i].color.y * omega;
                color.z() += splattedGaussian_sh[i].color.z * omega;

                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

#ifdef USE_MEAN_DEPTH
                // mean depth
                depth += omega * d;
#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha < 0.5)
                {
                    T = prod_alpha;
                    depth = d;
                }
#endif
                if (prod_alpha < 0.001f)
                {
                    break;
                }
            }

            // const float final_T = prod_alpha;
            if (prod_alpha < alphaThresh)
            {
                const float final_T = max(0.f, min(1.f, prod_alpha));

                color += final_T * Eigen::Vector3f(bgColor.x, bgColor.y, bgColor.z);

                uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

                const Eigen::Vector3f color_error = color - Eigen::Vector3f(rgba.x / 255.f,
                                                                            rgba.y / 255.f,
                                                                            rgba.z / 255.f);

                const float imgDepth = tex2D<float>(texDepth, x, y);

                const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

                prod_alpha = 1.f;
                T = 1.f;

                Eigen::Vector3f acc_c(0.f, 0.f, 0.f);
                float alpha_prev = 0.f;
                Eigen::Vector3f color_prev(0.f, 0.f, 0.f);
                // float T = final_T;

                // Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                //                              {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

                Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                                    (y - cameraParams.c.y) / cameraParams.f.y,
                                    1.f);

                Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                          {ray.z(), 0.f, -ray.x()},
                                          {-ray.y(), ray.x(), 0.f}};

                Eigen::Matrix3f R = q_cam.toRotationMatrix();

                for (int i = 0; i < n; i++)
                {
                    // float z = splattedGaussian_sh[i].position.z;
                    const float dx = splattedGaussian_sh[i].position.x - x;
                    const float dy = splattedGaussian_sh[i].position.y - y;
                    const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                    const float G = expf(-0.5f * v);
                    const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
                    if (alpha_i < 1.f / 255.f)
                        continue;

                    float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                    Eigen::Vector3f d_alpha(splattedGaussian_sh[i].color.x * prod_alpha,
                                            splattedGaussian_sh[i].color.y * prod_alpha,
                                            splattedGaussian_sh[i].color.z * prod_alpha);

                    acc_c += alpha_i * d_alpha;

                    d_alpha -= (color - acc_c) / (1.f - alpha_i);

                    // d_alpha = -color_error * d_alpha;

                    // const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;

                    // const float a_G_dl = alpha_i * dl_alpha;

                    // uint32_t gid = gids_sh[i];

                    Eigen::Vector2f dl_mean2d(splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy,
                                              splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);

                    // Eigen::Vector3f d_mu_cam = -Jt * dl_mean2d;

                    // huber loss for the color
                    float lc = color_error.norm();
                    float wc = lc < colorThresh ? 1.f : colorThresh / lc;

                    Eigen::Vector3f ray((splattedGaussian_sh[i].position.x - cameraParams.c.x) / cameraParams.f.x,
                                        (splattedGaussian_sh[i].position.y - cameraParams.c.y) / cameraParams.f.y,
                                        1.f);

                    Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                              {ray.z(), 0.f, -ray.x()},
                                              {-ray.y(), ray.x(), 0.f}};

                    Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / d, 0.f},
                                                  {0.f, cameraParams.f.y / d},
                                                  {-cameraParams.f.x * ray.x() / d, -cameraParams.f.y * ray.y() / d}};

                    Eigen::Matrix3f Jt_cam = Jt * dl_mean2d * alpha_i * d_alpha.transpose();

                    Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
                    Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error;

#ifdef USE_MEAN_DEPTH
                    // mean depth
                    // d_mu_cam += alpha_i * prod_alpha * depth_error * ray;

#endif

                    prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                    // median depth
                    if (T > 0.5f && prod_alpha <= 0.5f && imgDepth > 0.5f)
                    {
                        T = prod_alpha;
                        // d_mu_cam += depth_error * ray;

                        float ld = fabsf(depth_error);
                        float wd = ld < depthThresh ? 1.f : depthThresh / ld;

                        wd /= imgDepth;

                        JtJ_cam += wd * ray * ray.transpose();
                        Jtr_cam += wd * depth_error * ray;

                        // JtJ_cam(2,2) += wd;
                        // Jtr_cam.z() += wd * depth_error;
                    }
                    // else{
                    //     dg.depth = 0.f;
                    // }
#endif
                    Eigen::Matrix<float, 6, 3> Jpose;
                    Jpose.block<3, 3>(0, 0) = R;
                    Jpose.block<3, 3>(3, 0) = /*R **/ d * ray_cross;

                    JtJ += Jpose * JtJ_cam * Jpose.transpose();
                    Jtr += Jpose * Jtr_cam;

                    // dp += q_cam * d_mu_cam;
                    // dq += q_cam * (z * ray.cross(d_mu_cam));
                    // dl_pose.n++;

                    if (prod_alpha < 0.001f)
                    {
                        break;
                    }
                }
            }
        }

        block.sync();

        int k = 0;
#pragma unroll
        for (int i = 0; i < 6; i++)
#pragma unroll
            for (int j = i; j < 6; j++, k++)
            {
                mtd_i.JtJ[k] = JtJ_data[6 * i + j];
            }

        /*float *Jtr_i = mtd_sh[tid].Jtr;
        #pragma unroll
        for (int i = 0; i < 6; i++)
        {
            Jtr_i[i] = Jtr_data[i];
        }
        */

        block.sync();

        reduce<BLOCK_SIZE>(mtd_sh, tid);

        if (tid == 0)
        {
            MotionTrackingData &mtd0 = mtd_sh[0];
            for (int i = 0; i < 21; i++)
            {
                atomicAdd(&mtd->JtJ[i], mtd0.JtJ[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                atomicAdd(&mtd->Jtr[i], mtd0.Jtr[i]);
            }
        }
    }

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
        cudaTextureObject_t texDy,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ MotionTrackingData mtd_sh[BLOCK_SIZE];
        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        float JtJ_data[36];

        MotionTrackingData &mtd_i = mtd_sh[tid];

        Eigen::Map<Eigen::Matrix<float, 6, 6>> JtJ(JtJ_data);
        Eigen::Map<Eigen::Vector<float, 6>> Jtr(mtd_i.Jtr);

        JtJ.setZero();
        Jtr.setZero();

        const float imgDepth = tex2D<float>(texDepth, x, y);

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;

        bool inside = x < width && y < height && imgDepth > 0.1f;

        float3 color = make_float3(0.f);
        float depth = 0.f;
        float final_T;
        uint32_t n_contrib;
        uint32_t last_contributor = 0;

        forwardPass(
            color,
            depth,
            final_T,
            n_contrib,
            x,
            y,
            tileId,
            splattedGaussian_sh,
            gids_sh,
            ranges,
            indices,
            imgPositions,
            imgInvSigmas,
            pHats,
            colors,
            alphas,
            bgColor,
            inside);

        uint2 range = ranges[tileId];
        int n = range.y - range.x;
        // float T = 1.f;

        // bool done = !inside;
        // for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        // {
        //     // collect gaussians data
        //     if (k * BLOCK_SIZE + tid < n)
        //     {
        //         uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
        //         gids_sh[tid] = gid;
        //         splattedGaussian_sh[tid].position = imgPositions[gid];
        //         splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
        //         splattedGaussian_sh[tid].color = colors[gid];
        //         splattedGaussian_sh[tid].alpha = alphas[gid];
        //         splattedGaussian_sh[tid].pHat = pHats[gid];
        //     }
        //     block.sync();

        //     for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
        //     {
        //         contributor++;
        //         const float dx = splattedGaussian_sh[i].position.x - x;
        //         const float dy = splattedGaussian_sh[i].position.y - y;
        //         const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
        //         const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
        //         if (alpha_i < 1.f / 255.f || v <= 0.f)
        //             continue;
        //         float test_T = T * (1 - alpha_i);
        //         if (test_T < 0.0001f)
        //         {
        //             done = true;
        //             continue;
        //         }
        //         color += splattedGaussian_sh[i].color * alpha_i * T;

        //         float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

        //         if (T > 0.5f && test_T < 0.5f)
        //         {
        //             depth = d;
        //         }

        //         T = test_T;
        //         n_contrib = contributor;
        //     }
        // }

        // if (inside)
        // {
        //     final_T = T;
        //     n_contrib;
        //     color += T * bgColor;
        // }

        inside &= final_T < alphaThresh;

        float T = 1.f;
        float prod_alpha = 1.f;

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        Eigen::Vector3f color_eig(color.x, color.y, color.z);
        const Eigen::Vector3f color_error = color_eig - Eigen::Vector3f(rgba.x / 255.f, rgba.y / 255.f, rgba.z / 255.f);
        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        Eigen::Vector3f acc_c(0.f, 0.f, 0.f);
        float alpha_prev = 0.f;
        Eigen::Vector3f color_prev(0.f, 0.f, 0.f);

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                            (y - cameraParams.c.y) / cameraParams.f.y,
                            1.f);

        Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                  {ray.z(), 0.f, -ray.x()},
                                  {-ray.y(), ray.x(), 0.f}};

        Eigen::Matrix3f R = q_cam.toRotationMatrix();

        // const float final_T = prod_alpha;
        if (inside)
        {
            float4 gradX = tex2D<float4>(texDx, x, y);
            float4 gradY = tex2D<float4>(texDy, x, y);

            Eigen::Matrix<float, 2, 3> dl_Img{{gradX.x, gradX.y, gradX.z},
                                              {gradY.x, gradY.y, gradY.z}};

            // huber loss for the color
            float lc = color_error.norm();
            float wc = lc < colorThresh ? 1.f : colorThresh / lc;

            Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / imgDepth, 0.f},
                                          {0.f, cameraParams.f.y / imgDepth},
                                          {-cameraParams.f.x * ray.x() / imgDepth, -cameraParams.f.y * ray.y() / imgDepth}};

            Eigen::Matrix3f Jt_cam = Jt * dl_Img;

            Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
            Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error;
            // Eigen::Vector3f Jtr_cam = wc * Jt_cam * color_error;

            float ld = fabsf(depth_error);
            float wd = ld < depthThresh ? 1.f : depthThresh / ld;

            wd /= imgDepth;
            JtJ_cam += wd * ray * ray.transpose();
            Jtr_cam += wd * depth_error * ray;
            // Jtr_cam += - wd * depth_error * ray;

            Eigen::Matrix<float, 6, 3> Jpose;

            Jpose.block<3, 3>(0, 0) = R;
            Jpose.block<3, 3>(3, 0) = /*R **/ imgDepth * ray_cross;

            // Jpose.block<3, 3>(0, 0).setIdentity();
            // Jpose.block<3, 3>(3, 0) = /*R **/ - imgDepth * ray_cross;

            JtJ += Jpose * JtJ_cam * Jpose.transpose();
            Jtr += Jpose * Jtr_cam;
        }

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; inside && i + k * BLOCK_SIZE < n_contrib && i < BLOCK_SIZE; i++)
            {
                // float z = splattedGaussian_sh[i].position.z;
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                const float G = expf(-0.5f * v);
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
                if (alpha_i < 1.f / 255.f)
                    continue;

                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                Eigen::Vector3f d_alpha(splattedGaussian_sh[i].color.x * prod_alpha,
                                        splattedGaussian_sh[i].color.y * prod_alpha,
                                        splattedGaussian_sh[i].color.z * prod_alpha);

                acc_c += alpha_i * d_alpha;

                d_alpha -= (color_eig - acc_c) / (1.f - alpha_i);

                // d_alpha = -color_error * d_alpha;

                // const float dl_alpha = (d_alpha.x + d_alpha.y + d_alpha.z) / 3.f;

                // const float a_G_dl = alpha_i * dl_alpha;

                // uint32_t gid = gids_sh[i];

                Eigen::Vector2f dl_mean2d(splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy,
                                          splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);

                // Eigen::Vector3f d_mu_cam = -Jt * dl_mean2d;

                // huber loss for the color
                float lc = color_error.norm();
                float wc = lc < colorThresh ? 1.f : colorThresh / lc;

                Eigen::Vector3f ray((splattedGaussian_sh[i].position.x - cameraParams.c.x) / cameraParams.f.x,
                                    (splattedGaussian_sh[i].position.y - cameraParams.c.y) / cameraParams.f.y,
                                    1.f);

                Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                          {ray.z(), 0.f, -ray.x()},
                                          {-ray.y(), ray.x(), 0.f}};

                Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / d, 0.f},
                                              {0.f, cameraParams.f.y / d},
                                              {-cameraParams.f.x * ray.x() / d, -cameraParams.f.y * ray.y() / d}};

                Eigen::Matrix3f Jt_cam = Jt * dl_mean2d * alpha_i * d_alpha.transpose();

                Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
                Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error;

#ifdef USE_MEAN_DEPTH
                // mean depth
                // d_mu_cam += alpha_i * prod_alpha * depth_error * ray;

#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha <= 0.5f && imgDepth > 0.5f)
                {
                    T = prod_alpha;
                    // d_mu_cam += depth_error * ray;

                    float ld = fabsf(depth_error);
                    float wd = ld < depthThresh ? 1.f : depthThresh / ld;

                    wd /= imgDepth;

                    JtJ_cam += wd * ray * ray.transpose();
                    Jtr_cam += wd * depth_error * ray;

                    // JtJ_cam(2,2) += wd;
                    // Jtr_cam.z() += wd * depth_error;
                }
                // else{
                //     dg.depth = 0.f;
                // }
#endif
                Eigen::Matrix<float, 6, 3> Jpose;
                Jpose.block<3, 3>(0, 0) = R;
                Jpose.block<3, 3>(3, 0) = /*R **/ d * ray_cross;

                JtJ += Jpose * JtJ_cam * Jpose.transpose();
                Jtr += Jpose * Jtr_cam;

                // dp += q_cam * d_mu_cam;
                // dq += q_cam * (z * ray.cross(d_mu_cam));
                // dl_pose.n++;

                if (prod_alpha < 0.001f)
                {
                    break;
                }
            }
        }

        block.sync();

        int k = 0;
#pragma unroll
        for (int i = 0; i < 6; i++)
#pragma unroll
            for (int j = i; j < 6; j++, k++)
            {
                mtd_i.JtJ[k] = JtJ_data[6 * i + j];
            }

        block.sync();

        reduce<BLOCK_SIZE>(mtd_sh, tid);

        if (tid == 0)
        {
            MotionTrackingData &mtd0 = mtd_sh[0];
            for (int i = 0; i < 21; i++)
            {
                atomicAdd(&mtd->JtJ[i], mtd0.JtJ[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                atomicAdd(&mtd->Jtr[i], mtd0.Jtr[i]);
            }
        }
    }

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
        cudaTextureObject_t texDy,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float3 bgColor,
        float alphaThresh,
        float colorThresh,
        float depthThresh,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ MotionTrackingData mtd_sh[BLOCK_SIZE];
        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        float JtJ_data[36];

        MotionTrackingData &mtd_i = mtd_sh[tid];

        Eigen::Map<Eigen::Matrix<float, 6, 6>> JtJ(JtJ_data);
        Eigen::Map<Eigen::Vector<float, 6>> Jtr(mtd_i.Jtr);

        JtJ.setZero();
        Jtr.setZero();

        const float imgDepth = tex2D<float>(texDepth, x, y);

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;

        bool inside = x < width && y < height && imgDepth > 0.1f;

        float3 color = make_float3(0.f);
        float depth = 0.f;
        float final_T;
        uint32_t n_contrib;
        uint32_t last_contributor = 0;

        forwardPass(
            color,
            depth,
            final_T,
            n_contrib,
            x,
            y,
            tileId,
            splattedGaussian_sh,
            gids_sh,
            ranges,
            indices,
            imgPositions,
            imgInvSigmas,
            pHats,
            colors,
            alphas,
            bgColor,
            inside);

        uint2 range = ranges[tileId];
        int n = range.y - range.x;

        inside &= final_T < alphaThresh;

        float T = 1.f;
        float prod_alpha = 1.f;

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        Eigen::Vector3f color_eig(color.x, color.y, color.z);
        const Eigen::Vector3f color_error = color_eig - Eigen::Vector3f(rgba.x / 255.f, rgba.y / 255.f, rgba.z / 255.f);
        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        Eigen::Vector3f acc_c(0.f, 0.f, 0.f);
        float alpha_prev = 0.f;
        Eigen::Vector3f color_prev(0.f, 0.f, 0.f);

        Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
        Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
        Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                            (y - cameraParams.c.y) / cameraParams.f.y,
                            1.f);

        Eigen::Matrix3f ray_cross{{0.f, -ray.z(), ray.y()},
                                  {ray.z(), 0.f, -ray.x()},
                                  {-ray.y(), ray.x(), 0.f}};

        Eigen::Matrix3f R = q_cam.toRotationMatrix();

        // const float final_T = prod_alpha;
        if (inside)
        {
            float4 gradX = tex2D<float4>(texDx, x, y);
            float4 gradY = tex2D<float4>(texDy, x, y);

            Eigen::Matrix<float, 2, 3> dl_Img{{gradX.x, gradX.y, gradX.z},
                                              {gradY.x, gradY.y, gradY.z}};

            // huber loss for the color
            float lc = color_error.norm();
            float wc = lc < colorThresh ? 1.f : colorThresh / lc;

            Eigen::Matrix<float, 3, 2> Jt{{cameraParams.f.x / imgDepth, 0.f},
                                          {0.f, cameraParams.f.y / imgDepth},
                                          {-cameraParams.f.x * ray.x() / imgDepth, -cameraParams.f.y * ray.y() / imgDepth}};

            Eigen::Matrix3f Jt_cam = Jt * dl_Img;

            Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
            Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error;
            // Eigen::Vector3f Jtr_cam = wc * Jt_cam * color_error;

            float ld = fabsf(depth_error);
            float wd = ld < depthThresh ? 1.f : depthThresh / ld;

            wd /= imgDepth;
            JtJ_cam += wd * ray * ray.transpose();
            Jtr_cam += wd * depth_error * ray;
            // Jtr_cam += - wd * depth_error * ray;

            Eigen::Matrix<float, 6, 3> Jpose;

            Jpose.block<3, 3>(0, 0) = R;
            Jpose.block<3, 3>(3, 0) = /*R **/ imgDepth * ray_cross;

            // Jpose.block<3, 3>(0, 0).setIdentity();
            // Jpose.block<3, 3>(3, 0) = /*R **/ - imgDepth * ray_cross;

            JtJ += Jpose * JtJ_cam * Jpose.transpose();
            Jtr += Jpose * Jtr_cam;
        }

        block.sync();

        int k = 0;
#pragma unroll
        for (int i = 0; i < 6; i++)
#pragma unroll
            for (int j = i; j < 6; j++, k++)
            {
                mtd_i.JtJ[k] = JtJ_data[6 * i + j];
            }

        block.sync();

        reduce<BLOCK_SIZE>(mtd_sh, tid);

        if (tid == 0)
        {
            MotionTrackingData &mtd0 = mtd_sh[0];
            for (int i = 0; i < 21; i++)
            {
                atomicAdd(&mtd->JtJ[i], mtd0.JtJ[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                atomicAdd(&mtd->Jtr[i], mtd0.Jtr[i]);
            }
        }
    }

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
        uint32_t height)
    {
        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ MotionTrackingData mtd_sh[GSS_BLOCK_SIZE];
        float JtJ_data[36];
        MotionTrackingData &mtd_i = mtd_sh[tid];
        Eigen::Map<Eigen::Matrix<float, 6, 6>> JtJ(JtJ_data);
        // Eigen::Map<Eigen::Matrix<float, 6, 1>> Jtr(mtd_i.Jtr);
        Eigen::Map<Eigen::Vector<float, 6>> Jtr(mtd_i.Jtr);
        JtJ.setZero();
        Jtr.setZero();

        if (x < width && y < height)
        {
            float depth1 = tex2D<float>(texDepth1, x, y);
            if (depth1 > 0.1f)
            {
                Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                                    (y - cameraParams.c.y) / cameraParams.f.y,
                                    1.f);
                Eigen::Matrix3f R = q_cam.toRotationMatrix();
                Eigen::Vector3f X1 = depth1 * ray;
                Eigen::Vector3f X2 = q_cam.conjugate() * (X1 - p_cam);
                Eigen::Vector2f X2p(cameraParams.f.x * X2.x() / X2.z() + cameraParams.c.x,
                                    cameraParams.f.y * X2.y() / X2.z() + cameraParams.c.y);
                float2 xy2 = {X2p.x(), X2p.y()};
                if (xy2.x >= 0 && xy2.x < width && xy2.y >= 0 && xy2.y < height)
                {
                    uchar4 rgba1 = tex2D<uchar4>(texRGBA1, x, y);
                    float3 rgb1 = make_float3(rgba1.x / 255.f,
                                              rgba1.y / 255.f,
                                              rgba1.z / 255.f);

                    uchar4 rgba2 = tex2D<uchar4>(texRGBA2, xy2.x, xy2.y);
                    float3 rgb2 = make_float3(rgba2.x / 255.f,
                                              rgba2.y / 255.f,
                                              rgba2.z / 255.f);

                    float depth2 = tex2D<float>(texDepth2, xy2.x, xy2.y);

                    float3 color_error = rgb2 - rgb1;
                    Eigen::Map<const Eigen::Vector3f> color_error_eig((float *)&color_error);

                    float lc = color_error_eig.norm();
                    float wc = lc < colorThresh ? 1.f : colorThresh / lc;

                    float4 gradX = tex2D<float4>(texDx, xy2.x, xy2.y);
                    float4 gradY = tex2D<float4>(texDy, xy2.x, xy2.y);

                    Eigen::Matrix<float, 3, 2> Jt;
                    Jt << cameraParams.f.x / X2.z(), 0.f,
                        0.f, cameraParams.f.y / X2.z(),
                        -cameraParams.f.x * X2.x() / (X2.z() * X2.z()), -cameraParams.f.y * X2.y() / (X2.z() * X2.z());

                    Eigen::Matrix<float, 2, 3> grad;
                    grad << gradX.x, gradX.y, gradX.z,
                        gradY.x, gradY.y, gradY.z;
                    Eigen::Matrix3f Jt_cam = Jt * grad;

                    Eigen::Matrix3f JtJ_cam = wc * Jt_cam * Jt_cam.transpose();
                    Eigen::Vector3f Jtr_cam = -wc * Jt_cam * color_error_eig;

                    if (depth2 > 0.1f)
                    {
                        float depth_error = X2.z() - depth2;

                        float ld = fabsf(depth_error);
                        float wd = ld < depthThresh ? 1.f : depthThresh / ld;

                        Eigen::Vector3f ray2 = X2 / X2.z();
                        JtJ_cam += wd * ray2 * ray2.transpose();
                        Jtr_cam -= wd * depth_error * ray2;
                    }

                    Eigen::Matrix3f X1_cross;
                    X1_cross << 0.f, -X1.z(), X1.y(),
                        X1.z(), 0.f, -X1.x(),
                        -X1.y(), X1.x(), 0.f;
                    Eigen::Matrix<float, 3, 6> Jpose;
                    Jpose.block<3, 3>(0, 0) = -R.transpose();
                    Jpose.block<3, 3>(0, 3) = R * X1_cross * R.transpose();

                    JtJ = Jpose.transpose() * JtJ_cam * Jpose;
                    Jtr = Jpose.transpose() * Jtr_cam;
                }
            }
        }

        block.sync();
        int k = 0;
#pragma unroll
        for (int i = 0; i < 6; i++)
#pragma unroll
            for (int j = i; j < 6; j++, k++)
            {
                mtd_i.JtJ[k] = JtJ_data[6 * i + j];
            }

        block.sync();

        reduce<BLOCK_SIZE>(mtd_sh, tid);

        if (tid == 0)
        {
            MotionTrackingData &mtd0 = mtd_sh[0];
            for (int i = 0; i < 21; i++)
            {
                atomicAdd(&mtd->JtJ[i], mtd0.JtJ[i]);
            }
            for (int i = 0; i < 6; i++)
            {
                atomicAdd(&mtd->Jtr[i], mtd0.Jtr[i]);
            }
        }
    }

    __global__ void applyDeltaPose_kernel(
        DeltaPose3D *__restrict__ deltaPose,
        const float3 *__restrict__ positions,
        const float3 *__restrict__ scales,
        const float4 *__restrict__ orientations,
        const DeltaGaussian2D *__restrict__ deltaGaussians,
        const Pose3D cameraPose,
        const CameraParameters cameraParams,
        float eta,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ DeltaPose3D deltaPose_sh[BLOCK_SIZE];

        if (idx >= nbGaussians)
        {
            deltaPose_sh[tid].dp = make_float3(0.f);
            deltaPose_sh[tid].dq = make_float3(0.f);
            deltaPose_sh[tid].n = 0.f;
        }
        else
        {
            DeltaGaussian2D dg = deltaGaussians[idx];
            if (dg.n == 0.f)
            {
                deltaPose_sh[tid].dp = make_float3(0.f);
                deltaPose_sh[tid].dq = make_float3(0.f);
                deltaPose_sh[tid].n = 0.f;
            }
            else
            {
                deltaPose_sh[tid].n = dg.n;

                // compute delta mean3D
                Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);
                Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);
                Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
                Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);

                Eigen::Matrix<float, 2, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                                             {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())}};

                Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
                Eigen::Matrix3f R = q_gauss.toRotationMatrix();

                // Eigen::Vector2f dl_mean2d(dg.meanImg.x / dg.n, dg.meanImg.y / dg.n);
                Eigen::Vector2f dl_mean2d(dg.meanImg.x, dg.meanImg.y);

                Eigen::Map<Eigen::Vector3f> dp((float *)&deltaPose_sh[tid].dp);
                Eigen::Map<Eigen::Vector3f> dq((float *)&deltaPose_sh[tid].dq);
                // dp = q_cam * (J.transpose() * dl_mean2d + Eigen::Vector3f(0.f, 0.f, dg.depth / dg.n));

                Eigen::Vector3f d_mu_cam = -J.transpose() * dl_mean2d - dg.depth * Eigen::Vector3f(mu_cam.x() / mu_cam.z(), mu_cam.y() / mu_cam.z(), 1.f);

                dp = q_cam * d_mu_cam;
                dq = q_cam * (mu_cam.cross(d_mu_cam));

                // deltaPose_sh[tid].dq = make_float3(0.f);

                /*if (isnan(dp.x()))
                {
                    printf("nan value %f %f %f %f\n", dp.x(), dl_mean2d.x(), dl_mean2d.y(), dg.depth);
                }*/

// dq = mu_cam.cross(q_cam*d_mu_cam);

// #define USE_COV_FOR_POSE
#ifdef USE_COV_FOR_POSE

                /* compute derivative of inverse Cov2 wrt cov2*/
                const Eigen::Matrix<float, 2, 3> T = J * W;
                const Eigen::Matrix3f RS = R * s_gauss.asDiagonal();
                const Eigen::Matrix<float, 2, 3> M = T * RS;
                const Eigen::Matrix2f Cov2d = M * M.transpose();

                float a = Cov2d(0, 0) + 0.1f;
                float b = Cov2d(1, 0);
                float c = Cov2d(1, 1) + 0.1f;

                float denom = a * c - b * b;
                float dL_da = 0, dL_db = 0, dL_dc = 0;
                float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

                float3 dL_dconic = dg.invSigmaImg / dg.n;

                // derivative of Cov2 wrt Cov2^-1 :
                dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
                dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
                dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

                // Eigen::Matrix3f dL_dcov3D;
                // dL_dcov3D(0, 0) = (T(0, 0) * T(0, 0) * dL_da + T(0, 0) * T(1, 0) * dL_db + T(1, 0) * T(1, 0) * dL_dc);
                // dL_dcov3D(1, 1) = (T(0, 1) * T(0, 1) * dL_da + T(0, 1) * T(1, 1) * dL_db + T(1, 1) * T(1, 1) * dL_dc);
                // dL_dcov3D(2, 2) = (T(0, 2) * T(0, 2) * dL_da + T(0, 2) * T(1, 2) * dL_db + T(1, 2) * T(1, 2) * dL_dc);
                // dL_dcov3D(1, 0) = dL_dcov3D(0, 1) = T(0, 0) * T(0, 1) * dL_da + (T(0, 0) * T(1, 1) + T(0, 1) * T(1, 0)) * dL_db + T(1, 0) * T(1, 1) * dL_dc;
                // dL_dcov3D(2, 0) = dL_dcov3D(0, 2) = T(0, 0) * T(0, 2) * dL_da + (T(0, 0) * T(1, 2) + T(0, 2) * T(1, 0)) * dL_db + T(1, 0) * T(1, 2) * dL_dc;
                // dL_dcov3D(2, 1) = dL_dcov3D(1, 2) = T(0, 2) * T(0, 1) * dL_da + (T(0, 1) * T(1, 2) + T(0, 2) * T(1, 1)) * dL_db + T(1, 1) * T(1, 2) * dL_dc;

                // Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
                // cov2D = T * Cov3d * T;
                const Eigen::Matrix3f Cov3D = RS * (RS.transpose());

                float dL_dT00 = 2 * (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_da +
                                (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_db;
                float dL_dT01 = 2 * (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_da +
                                (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_db;
                float dL_dT02 = 2 * (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_da +
                                (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_db;
                float dL_dT10 = 2 * (T(1, 0) * Cov3D(0, 0) + T(1, 1) * Cov3D(0, 1) + T(1, 2) * Cov3D(0, 2)) * dL_dc +
                                (T(0, 0) * Cov3D(0, 0) + T(0, 1) * Cov3D(0, 1) + T(0, 2) * Cov3D(0, 2)) * dL_db;
                float dL_dT11 = 2 * (T(1, 0) * Cov3D(1, 0) + T(1, 1) * Cov3D(1, 1) + T(1, 2) * Cov3D(1, 2)) * dL_dc +
                                (T(0, 0) * Cov3D(1, 0) + T(0, 1) * Cov3D(1, 1) + T(0, 2) * Cov3D(1, 2)) * dL_db;
                float dL_dT12 = 2 * (T(1, 0) * Cov3D(2, 0) + T(1, 1) * Cov3D(2, 1) + T(1, 2) * Cov3D(2, 2)) * dL_dc +
                                (T(0, 0) * Cov3D(2, 0) + T(0, 1) * Cov3D(2, 1) + T(0, 2) * Cov3D(2, 2)) * dL_db;
                // Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
                // T = J * W
                float dL_dJ00 = W(0, 0) * dL_dT00 + W(0, 1) * dL_dT01 + W(0, 2) * dL_dT02;
                float dL_dJ02 = W(2, 0) * dL_dT00 + W(2, 1) * dL_dT01 + W(2, 2) * dL_dT02;
                float dL_dJ11 = W(1, 0) * dL_dT10 + W(1, 1) * dL_dT11 + W(1, 2) * dL_dT12;
                float dL_dJ12 = W(2, 0) * dL_dT10 + W(2, 1) * dL_dT11 + W(2, 2) * dL_dT12;

                float tz = 1.f / mu_cam.z();
                float tz2 = tz * tz;
                float tz3 = tz2 * tz;

                // Gradients of loss w.r.t. transformed Gaussian mean t
                // float dL_dtx = -cameraParams.f.x * tz2 * dL_dJ02;
                // float dL_dty = -cameraParams.f.y * tz2 * dL_dJ12;
                // float dL_dtz = -cameraParams.f.x * tz2 * dL_dJ00 - camep/var/lib/docker -raParams.f.y * tz2 * dL_dJ11 + (2 * cameraParams.f.x * mu_cam.x()) * tz3 * dL_dJ02 + (2 * cameraParams.f.y * mu_cam.y()) * tz3 * dL_dJ12;
                Eigen::Vector3f dL_W_cam(-cameraParams.f.x * tz2 * dL_dJ02,
                                         -cameraParams.f.y * tz2 * dL_dJ12,
                                         -cameraParams.f.x * tz2 * dL_dJ00 - cameraParams.f.y * tz2 * dL_dJ11 + (2 * cameraParams.f.x * mu_cam.x()) * tz3 * dL_dJ02 + (2 * cameraParams.f.y * mu_cam.y()) * tz3 * dL_dJ12);

                dp -= (q_cam * dL_W_cam);
                dq -= (q_cam * (mu_cam.cross(dL_W_cam)));

                Eigen::Matrix3f dL_dW;
                dL_dW(0, 0) = J(0, 0) * dL_dT00;
                dL_dW(0, 1) = J(0, 0) * dL_dT01;
                dL_dW(0, 2) = J(0, 0) * dL_dT02;
                dL_dW(1, 0) = J(1, 1) * dL_dT10;
                dL_dW(1, 1) = J(1, 1) * dL_dT11;
                dL_dW(1, 2) = J(1, 1) * dL_dT12;
                dL_dW(2, 0) = J(0, 2) * dL_dT00 + J(1, 2) * dL_dT10;
                dL_dW(2, 1) = J(0, 2) * dL_dT01 + J(1, 2) * dL_dT11;
                dL_dW(2, 2) = J(0, 2) * dL_dT02 + J(1, 2) * dL_dT12;

                dq -= (q_cam * (W.col(0).cross(dL_dW.col(0)) + W.col(1).cross(dL_dW.col(1)) + W.col(2).cross(dL_dW.col(2))));

#endif // USE_COV_FOR_POSE
            }
        }
        block.sync();
        reduce<BLOCK_SIZE>(deltaPose_sh, tid);
        if (tid == 0)
        {
            DeltaPose3D &delta = deltaPose_sh[0];
            atomicAdd(&deltaPose->dp.x, delta.dp.x);
            atomicAdd(&deltaPose->dp.y, delta.dp.y);
            atomicAdd(&deltaPose->dp.z, delta.dp.z);
            atomicAdd(&deltaPose->dq.x, delta.dq.x);
            atomicAdd(&deltaPose->dq.y, delta.dq.y);
            atomicAdd(&deltaPose->dq.z, delta.dq.z);
            atomicAdd(&deltaPose->n, delta.n);
        }
    }

    __global__ void computeDensityMask_kernel(
        float *maskData,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float2 *__restrict__ pHats,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        // cudaTextureObject_t texRGBA,
        cudaTextureObject_t texDepth,
        uint2 numTiles,
        uint32_t width,
        uint32_t height,
        uint32_t maskStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();

        __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
        //__shared__ uint32_t gids[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = range.y - range.x;
        const float imgDepth = tex2D<float>(texDepth, x, y);

        bool inside = x < width && y < height;

        imgDepth > 0.5f;

        float prod_alpha = 1.f;
        float3 color = make_float3(0.f);

        float depth = 0.f;
        float T = 1.f;
        bool done = (!inside || imgDepth < 0.5f);
        float val = 0.f;

        // if(threadIdx.x==0 && threadIdx.y==0)
        // {
        //     printf("ranges[%d] : %u %u\n", tileId, range.x, range.y);
        // }
        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (range.x + block.thread_rank() < range.y)
            {
                uint32_t gid = indices[range.x + block.thread_rank()];
                // gids[block.thread_rank()] = gid;
                splattedGaussians[block.thread_rank()].position = imgPositions[gid];
                splattedGaussians[block.thread_rank()].invSigma = imgInvSigmas[gid];
                splattedGaussians[block.thread_rank()].pHat = pHats[gid];
                splattedGaussians[block.thread_rank()].color = colors[gid];
                splattedGaussians[block.thread_rank()].alpha = alphas[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                float dx = splattedGaussians[i].position.x - x;
                float dy = splattedGaussians[i].position.y - y;
                float v = splattedGaussians[i].invSigma.x * dx * dx + 2.f * splattedGaussians[i].invSigma.y * dx * dy + splattedGaussians[i].invSigma.z * dy * dy;

                float alpha_i = min(0.99f, splattedGaussians[i].alpha * expf(-0.5f * v));

                if (alpha_i < 1.f / 255.f)
                    continue;

                color += splattedGaussians[i].color * alpha_i * prod_alpha;
#ifdef USE_MEAN_DEPTH
                // mean depth
                depth += (splattedGaussians[i].position.z + dx * splattedGaussians[i].pHat.x + dy * splattedGaussians[i].pHat.y) * alpha_i * prod_alpha;

#endif
                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha < 0.5)
                {
                    T = prod_alpha;
                    depth = splattedGaussians[i].position.z + dx * splattedGaussians[i].pHat.x + dy * splattedGaussians[i].pHat.y;
                }
#endif
                if (prod_alpha < 0.0001f)
                {
                    done = true;
                    break;
                }
            }
        }

        if (inside)
        {
            if (imgDepth > 0.5f)
            {
                if (prod_alpha > 0.5f)
                {
                    val = prod_alpha;
                }
                else
                {
                    const float depth_error = imgDepth > 0.5f ? depth - imgDepth : 0.f;

                    if (imgDepth < depth && depth_error > 0.2f * imgDepth)
                    {
                        val = 1.f;
                    }

                    // uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);
                    // const float3 color_error = color - make_float3(rgba.x / 255.f,
                    //                                            rgba.y / 255.f,
                    //                                         rgba.z / 255.f);
                    // if(length(color_error) > 0.2f)
                    // {
                    //     val = 1.f;
                    // }
                }
            }
            maskData[y * (maskStep / sizeof(float)) + x] = val;
        }
    }

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
                                            uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (sample_dx > 1)
        {
            int u_min = x * sample_dx;
            int v_min = y * sample_dy;

            if (u_min >= width || v_min >= height)
            {
                return;
            }

            float3 img_pos = make_float3(0.f);
            float4 rgba = make_float4(0.f);
            int n = 0;
            for (int u = u_min; u < u_min + sample_dx && u < width; u++)
            {
                for (int v = v_min; v < v_min + sample_dy && v < height; v++)
                {
                    if (tex2D<float>(texMask, u, v) > 0.f)
                    {
                        float depth = tex2D<float>(texDepth, u, v);
                        uchar4 color = tex2D<uchar4>(texRGBA, u, v);

                        img_pos.x += u;
                        img_pos.y += v;
                        img_pos.z += depth;
                        rgba.x += color.x / 255.f;
                        rgba.y += color.y / 255.f;
                        rgba.z += color.z / 255.f;
                        n++;
                    }
                }
            }

            if (n < ((sample_dx * sample_dy) >> 1)) // >>2
                return;

            Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
            Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);

            uint32_t idx = atomicAggInc(instanceCounter);

            img_pos = img_pos / n;

            Eigen::Vector3f pos_cam(img_pos.z * (img_pos.x - cameraParams.c.x) / cameraParams.f.x,
                                    img_pos.z * (img_pos.y - cameraParams.c.y) / cameraParams.f.y,
                                    img_pos.z);

            Eigen::Vector3f pos = p_cam + q_cam * pos_cam;

            positions[idx].x = pos.x();
            positions[idx].y = pos.y();
            positions[idx].z = pos.z();

            float scale_x = 1.f * img_pos.z * sample_dx / cameraParams.f.x;
            float scale_y = 1.f * img_pos.z * sample_dy / cameraParams.f.y;

            float3 scale = make_float3(0.5f * (scale_x + scale_y),
                                       0.5f * (scale_x + scale_y),
                                       0.1f * (scale_x + scale_y));

            scales[idx] = scale;
            float4 normal = tex2D<float4>(texNormal, img_pos.x, img_pos.y);

            Eigen::Vector3f u0(0.f, 0.f, 1.f);
            Eigen::Vector3f u1(normal.x, normal.y, normal.z);

            if (u1.z() < 0.f)
            {
                u1 = -u1;
            }
            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(u0, u1);
            q = q_cam * q; // CHECK (q*q_cam ?)

            orientations[idx] = make_float4(q.x(), q.y(), q.z(), q.w());
            colors[idx].x = rgba.x / n;
            colors[idx].y = rgba.y / n;
            colors[idx].z = rgba.z / n;

            alphas[idx] = 1.f;
        }

        else
        {
            if (x >= width || y >= height)
            {
                return;
            }

            int u = x;
            int v = y;

            if (tex2D<float>(texMask, u, v) <= 0.f)
            {
                return;
            }
            float3 img_pos = make_float3(0.f);
            float4 rgba = make_float4(0.f);
            int n = 0;
            float depth = tex2D<float>(texDepth, u, v);
            uchar4 color = tex2D<uchar4>(texRGBA, u, v);
            img_pos.x = u;
            img_pos.y = v;
            img_pos.z = depth;
            rgba.x = color.x / 255.f;
            rgba.y = color.y / 255.f;
            rgba.z = color.z / 255.f;
            n = 1;
            if (n < 1)
            {
                return;
            }
            Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
            Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);

            uint32_t idx = atomicAggInc(instanceCounter);

            Eigen::Vector3f pos_cam(img_pos.z * (img_pos.x - cameraParams.c.x) / cameraParams.f.x,
                                    img_pos.z * (img_pos.y - cameraParams.c.y) / cameraParams.f.y,
                                    img_pos.z);

            Eigen::Vector3f pos = p_cam + q_cam * pos_cam;

            positions[idx].x = pos.x();
            positions[idx].y = pos.y();
            positions[idx].z = pos.z();

            float scale_x = 0.8f * img_pos.z * sample_dx / cameraParams.f.x;
            float scale_y = 0.8f * img_pos.z * sample_dy / cameraParams.f.y;

            float3 scale = make_float3(0.5f * (scale_x + scale_y),
                                       0.5f * (scale_x + scale_y),
                                       0.1f * (scale_x + scale_y));

            scales[idx] = scale;
            float4 normal = tex2D<float4>(texNormal, u, v);

            Eigen::Vector3f u0(0.f, 0.f, 1.f);
            Eigen::Vector3f u1(normal.x, normal.y, normal.z);

            if (u1.z() < 0.f)
            {
                u1 = -u1;
            }
            Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(u0, u1);
            q = q_cam * q; // CHECK (q*q_cam ?)
            orientations[idx] = make_float4(q.x(), q.y(), q.z(), q.w());
            colors[idx].x = rgba.x;
            colors[idx].y = rgba.y;
            colors[idx].z = rgba.z;

            alphas[idx] = 1.f;
        }
    }

    __global__ void computePosGradVariance_kernel(
        PosGradVariance *data,
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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        bool inside = x < width && y < height;
        int n = range.y - range.x;

        float3 color;
        float depth;
        float final_T;
        uint32_t n_contrib;

        forwardPass(
            color,
            depth,
            final_T,
            n_contrib,
            x,
            y,
            tileId,
            splattedGaussian_sh,
            gids_sh,
            ranges,
            indices,
            imgPositions,
            imgInvSigmas,
            pHats,
            colors,
            alphas,
            bgColor,
            inside);

        uchar4 rgba = tex2D<uchar4>(texRGBA, x, y);

        const float3 color_error = color - make_float3(rgba.x / 255.f,
                                                       rgba.y / 255.f,
                                                       rgba.z / 255.f);

        const float imgDepth = tex2D<float>(texDepth, x, y);

        const float depth_error = imgDepth > 0.1f ? depth - imgDepth : 0.f;

        float prod_alpha = 1.f;
        float T = 1.f;

        float3 acc_c = make_float3(0.f);
        /// float acc_d = 0.f;

        float alpha_prev = 0.f;
        float3 color_prev = make_float3(0.f);
        // float T = final_T;

        bool done = !inside;
        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;

                const float G = expf(-0.5f * v);
                const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * G);
                if (alpha_i < 1.f / 255.f)
                    continue;

                float3 d_alpha = splattedGaussian_sh[i].color * prod_alpha;

                acc_c += d_alpha * alpha_i;

                d_alpha -= (color - acc_c) / (1.f - alpha_i);

                d_alpha = -color_error * d_alpha;

                const float dl_alpha = d_alpha.x + d_alpha.y + d_alpha.z;

                const float G_dl = G * dl_alpha;
                const float a_G_dl = alpha_i * dl_alpha;

                uint32_t gid = gids_sh[i];

                float meanImgX = -a_G_dl * (splattedGaussian_sh[i].invSigma.x * dx + splattedGaussian_sh[i].invSigma.y * dy);
                float meanImgY = -a_G_dl * (splattedGaussian_sh[i].invSigma.y * dx + splattedGaussian_sh[i].invSigma.z * dy);

#ifdef USE_MEAN_DEPTH
                // mean depth
                // dg.depth = -alpha_i * prod_alpha * depth_error;
                // atomicAdd(&deltaGaussians[gid].depth, -alpha_i * prod_alpha * depth_error);
                // dg.alpha -= depth_error * G * dl_alpha;
                float omega = alpha_i * prod_alpha;
                meanImgX -= w_depth * omega * depth_error * splattedGaussian_sh[i].pHat.x;
                meanImgY -= w_depth * omega * depth_error * splattedGaussian_sh[i].pHat.y;

#endif

                prod_alpha *= (1.f - alpha_i);

#ifdef USE_MEDIAN_DEPTH
                // median depth
                if (T > 0.5f && prod_alpha <= 0.5f)
                {
                    float w_depth_norm = w_depth / (imgDepth + 0.1f);
                    T = prod_alpha;
                    meanImgX -= w_depth_norm * depth_error * splattedGaussian_sh[i].pHat.x;
                    meanImgY -= w_depth_norm * depth_error * splattedGaussian_sh[i].pHat.y;
                }
#endif

                // for (int j = 0; j < n; j++)
                // {
                //     float dd = depths[i] - depths[j];
                //     float ww = omegas[i] * omegas[j] * w_dist;
                //     float coeff = ww * dd;
                //     if (coeff == 0.f)
                //         continue;

                //     meanImgX -= coeff * splattedGaussian_sh[i].pHat.x;
                //     meanImgY -= coeff * splattedGaussian_sh[i].pHat.y;
                // }

                atomicAdd(&data[gid].w, 1.f);
                atomicAdd(&data[gid].dx, meanImgX);
                atomicAdd(&data[gid].dy, meanImgY);
                atomicAdd(&data[gid].dx2, meanImgX * meanImgX);
                atomicAdd(&data[gid].dy2, meanImgY * meanImgY);
                atomicAdd(&data[gid].dxdy, meanImgX * meanImgY);

                if (prod_alpha < 0.001f)
                {
                    done = true;
                    continue;
                }
            }
        }
    }

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
                                          uint32_t nbMaxGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        PosGradVariance posGradVar = data[idx];
        if (posGradVar.w == 0.f)
            return;

        float3 cov = {posGradVar.dx2 / posGradVar.w - posGradVar.dx * posGradVar.dx / (posGradVar.w * posGradVar.w),
                      posGradVar.dxdy / posGradVar.w - posGradVar.dx * posGradVar.dy / (posGradVar.w * posGradVar.w),
                      posGradVar.dy2 / posGradVar.w - posGradVar.dy * posGradVar.dy / (posGradVar.w * posGradVar.w)};

        float cov_trace = cov.x + cov.z;

        if (cov_trace > varThresh)
        // if (grad.x * grad.x + grad.y * grad.y > varThresh)
        {
            // split gaussian
            // printf("split\n");

            uint32_t new_idx = atomicAggInc(instanceCounter);

            if (new_idx >= nbMaxGaussians)
            {
                atomicAggDec(instanceCounter);
            }
            else
            {
                Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);
                Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);
                Eigen::Map<Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
                Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);

                Eigen::Matrix<float, 3, 3> J{{cameraParams.f.x / mu_cam.z(), 0.f, -cameraParams.f.x * mu_cam.x() / (mu_cam.z() * mu_cam.z())},
                                             {0.f, cameraParams.f.y / mu_cam.z(), -cameraParams.f.y * mu_cam.y() / (mu_cam.z() * mu_cam.z())},
                                             {0.f, 0.f, 1.f}};

                Eigen::Matrix3f W = q_cam.inverse().toRotationMatrix();
                Eigen::Matrix3f R = q_gauss.toRotationMatrix();

                // Compute split direction in image space
                float z_x = cov.z - cov.x;
                float d = 0.5f * (z_x + sqrtf(z_x * z_x + 4.f * cov.y * cov.y)) / cov.y;
                float div = 1.f / sqrtf(1 + d * d);
                Eigen::Vector3f dir2d(1 * div, d * div, 0.f);
                Eigen::Vector3f dir3d = q_cam * (J.transpose() * dir2d);
                dir3d.normalize();
                Eigen::Matrix<float, 3, 3> M = R * s_gauss.asDiagonal();

                float s = 0.5f * sqrtf((dir3d.transpose() * M * M.transpose() * dir3d).value());

                if (s > s_gauss.x() && s > s_gauss.y() && s > s_gauss.z())
                {
                    printf("%f %f %f %f\n", s, s_gauss.x(), s_gauss.y(), s_gauss.z());
                }

                Eigen::Vector3f p1 = p_gauss + s * dir3d;
                Eigen::Vector3f p2 = p_gauss - s * dir3d;
                positions[idx] = {p1.x(), p1.y(), p1.z()};
                positions[new_idx] = {p2.x(), p2.y(), p2.z()};
                // scales[idx] *= 0.6f;
                scales[new_idx] = scales[idx];
                orientations[new_idx] = orientations[idx];
                colors[new_idx] = colors[idx];
                alphas[new_idx] = 0.5f * alphas[idx];
                alphas[idx] = 0.5f * alphas[idx];

                // Eigen::Map<const Eigen::Vector3f> p((float *)&positions[idx]);
                // Eigen::Map<const Eigen::Quaternionf> q((float *)&orientations[idx]);
                // Eigen::Vector3f p1 = p + q.inverse() * Eigen::Vector3f(0.5f * scales[idx].x, 0.f, 0.f);
                // Eigen::Vector3f p2 = p - q.inverse() * Eigen::Vector3f(0.5f * scales[idx].x, 0.f, 0.f);
                // positions[idx] = {p1.x(), p1.y(), p1.z()};
                // positions[new_idx] = {p2.x(), p2.y(), p2.z()};
                // scales[idx] *= 0.6f;
                // scales[new_idx] = scales[idx];
                // orientations[new_idx] = orientations[idx];
                // colors[new_idx] = colors[idx];
                // alphas[new_idx] = alphas[idx];
            }
        }
    }

    inline __device__ void swap(float &x, float &y)
    {
        float tmp = x;
        x = y;
        y = tmp;
    }

    __global__ void pruneGaussians_kernel(
        unsigned int *__restrict__ nbRemoved,
        unsigned char *__restrict__ states,
        const float3 *__restrict__ scales,
        const float *__restrict__ alphas,
        float alphaThreshold,
        float scaleRatioThreshold,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        unsigned char state = 0;

        if (alphas[idx] < alphaThreshold)
        {
            state = 0xff;
        }
        else
        {
            float3 s = scales[idx];

            // sort x,y,z
            if (s.x > s.y)
            {
                swap(s.x, s.y);
            }
            if (s.y > s.z)
            {
                swap(s.y, s.z);
            }
            if (s.x > s.y)
            {
                swap(s.x, s.y);
            }

            if (s.y / s.z < scaleRatioThreshold || s.z < 0.005f)
            {
                state = 0xff;
            }

            // float maxScale = fmaxf(fmaxf(s.x, s.y), s.z);
            // float minScale = fminf(fminf(s.x, s.y), s.z);
            // // float maxScale = fmaxf(s.x, s.y);
            // // float minScale = fminf(s.x, s.y);
            // if (minScale / maxScale < scaleRatioThreshold || maxScale < 0.005f)
            // {
            //     state = 0xff;
            // }
        }

        if (state != 0)
            atomicAggInc(nbRemoved);

        states[idx] = state;
    }

    __global__ void computeGaussiansVisibility_kernel(
        unsigned char *__restrict__ visibilities,
        const uint2 *__restrict__ ranges,
        const uint32_t *__restrict__ indices,
        const float3 *__restrict__ imgPositions,
        const float3 *__restrict__ imgInvSigmas,
        const float *__restrict__ alphas,
        uint2 numTiles,
        uint32_t width,
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
        __shared__ uint32_t gids[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        int n = range.y - range.x;

        bool inside = x < width && y < height;

        float prod_alpha = 1.f;
        float depth = 0.f;
        float T = 1.f;

        bool done = !inside;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids[tid] = gid;
                splattedGaussians[tid].position = imgPositions[gid];
                splattedGaussians[tid].invSigma = imgInvSigmas[gid];
                splattedGaussians[tid].alpha = alphas[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                float dx = splattedGaussians[i].position.x - x;
                float dy = splattedGaussians[i].position.y - y;
                float v = splattedGaussians[i].invSigma.x * dx * dx + 2.f * splattedGaussians[i].invSigma.y * dx * dy + splattedGaussians[i].invSigma.z * dy * dy;

                float alpha_i = min(0.99f, splattedGaussians[i].alpha * expf(-0.5f * v));

                if (alpha_i < 1.f / 255.f)
                    continue;
                prod_alpha *= (1.f - alpha_i);

                visibilities[gids[i]] = 1;

                if (prod_alpha < 0.5f)
                {
                    done = true;
                    continue;
                }
                prod_alpha = max(0.f, min(1.f, prod_alpha));
            }
        }
    }

    __global__ void computeGaussiansCovisibility_kernel(
        uint32_t *__restrict__ visibilityInter,
        uint32_t *__restrict__ visibilityUnion,
        unsigned char *__restrict__ visibilities1,
        unsigned char *__restrict__ visibilities2,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        unsigned char vis1 = visibilities1[idx];
        unsigned char vis2 = visibilities2[idx];

        if (vis1 | vis2)
            atomicAggInc(visibilityUnion);

        if (vis1 & vis2)
            atomicAggInc(visibilityInter);
    }

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
        uint32_t height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tile_id = blockIdx.y * numTiles.x + blockIdx.x;
        bool inside = x < width && y < height;
        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        uint2 range = ranges[tile_id];
        int n = range.y - range.x;
        // float prod_alpha = 1.f;
        // float T = 1.f;

        const float imgDepth = tex2D<float>(texDepth, x, y);
        // inside = inside && imgDepth > 0.1f;
        bool done = !inside;

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                // splattedGaussian_sh[tid].alpha = alphas[gid];
                splattedGaussian_sh[tid].pHat = pHats[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                const float dx = splattedGaussian_sh[i].position.x - x;
                const float dy = splattedGaussian_sh[i].position.y - y;
                const float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
                // const float alpha_i = min(0.99f, splattedGaussian_sh[i].alpha * expf(-0.5f * v));
                const float alpha_i = expf(-0.5f * v);
                if (alpha_i < 1.f / 255.f || v <= 0.f)
                    continue;
                // float test_T = T * (1 - alpha_i);
                // if (test_T < 0.0001f)
                // {
                //     done = true;
                //     continue;
                // }

                float d = splattedGaussian_sh[i].position.z + dx * splattedGaussian_sh[i].pHat.x + dy * splattedGaussian_sh[i].pHat.y;

                if (imgDepth > 0.1f && d < 0.8f * imgDepth)
                {
                    atomicAdd(&outlierProb[gids_sh[i]], alpha_i);
                }
                atomicAdd(&totalAlpha[gids_sh[i]], alpha_i);

                // T = test_T;
            }
        }
    }

    __global__ void removeOutliers_kernel(
        unsigned int *__restrict__ nbRemoved,
        unsigned char *__restrict__ states,
        const float *__restrict__ outliersProb,
        const float *__restrict__ totalAlpha,
        float threshold,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        unsigned char state = 0;

        if (totalAlpha[idx] > 1.f && outliersProb[idx] / totalAlpha[idx] > threshold)
        {
            atomicAggInc(nbRemoved);
            state = 0xff;
        }

        states[idx] = state;
    }

    // __global__ void computeNormalsFromDepth_kernel(
    //     float4 *normalsData,
    //     cudaTextureObject_t texDepth,
    //     const CameraParameters cameraParams,
    //     uint32_t width,
    //     uint32_t height,
    //     uint32_t normalsStep)
    // {
    //     int x = blockIdx.x * blockDim.x + threadIdx.x;
    //     int y = blockIdx.y * blockDim.y + threadIdx.y;

    //     if (x >= width - 1 || y >= height - 1)
    //     {
    //         return;
    //     }

    //     float depths[9];
    //     int k=0;
    //     for (int j = -1; j <= 1; j++)
    //     {
    //         for (int i = -1; i <= 1; i++, k++)
    //         {
    //             depths[k] = tex2D<float>(texDepth, x+j, y+i);
    //         }
    //     }

    //     float depth = depths[4];
    //     float dzdx = (-0.125*depths[0] - 0.25f*depths[1] - 0.125f*depths[2] + 0.125*depths[6] + 0.25f*depths[7] + 0.125f*depths[8]) * cameraParams.f.x / depth;
    //     float dzdy = (-0.125*depths[0] - 0.25f*depths[3] - 0.125f*depths[6] + 0.125*depths[2] + 0.25f*depths[5] + 0.125f*depths[8]) * cameraParams.f.y / depth;
    //     //float dzdy = (0.5f * tex2D<float>(texDepth, x, y + 1) - tex2D<float>(texDepth, x, y - 1)) * cameraParams.f.y / depth;
    //     float3 d = {dzdx, dzdy, -1.f};
    //     float3 n = normalize(d);

    //     float4 *normal_row = (float4 *)&((unsigned char *)normalsData)[y * normalsStep];
    //     normal_row[x] = make_float4(n.x, n.y, n.z, 0.f);
    // }

#define GSS_DEPTH_HS 4
    __global__ void computeNormalsFromDepth_kernel(
        float4 *normalsData,
        cudaTextureObject_t texDepth,
        const CameraParameters cameraParams,
        uint32_t width,
        uint32_t height,
        uint32_t normalsStep)
    {
        const int nbShPoints = (GSS_BLOCK_X + 2 * GSS_DEPTH_HS) * (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS);
        __shared__ float disps[(GSS_BLOCK_X + 2 * GSS_DEPTH_HS) * (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS)];

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        for (int k = tid; k < nbShPoints; k += GSS_BLOCK_SIZE)
        {
            int x = blockIdx.x * blockDim.x - GSS_DEPTH_HS + k / (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS);
            int y = blockIdx.y * blockDim.y - GSS_DEPTH_HS + k % (GSS_BLOCK_Y + 2 * GSS_DEPTH_HS);

            if (x < 0 || x >= width || y < 0 || y >= height)
            {
                disps[k] = 0.f;
            }
            else
            {
                float depth = tex2D<float>(texDepth, x, y);

                if (depth == 0.f)
                {
                    disps[k] = 0.f;
                }
                else
                {
                    disps[k] = 1. / depth;
                }
            }
        }

        block.sync();

        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width - 1 || y >= height - 1)
        {
            return;
        }

        double sx, sy, sz, sw;
        double sxx, sxy, sxz, syy, syz, szz;
        sx = sy = sz = sw = sxx = sxy = sxz = syy = syz = szz = 0.f;

        for (int j = -GSS_DEPTH_HS; j <= GSS_DEPTH_HS; j++)
        {
            for (int i = -GSS_DEPTH_HS; i <= GSS_DEPTH_HS; i++)
            {
                float disp = disps[(threadIdx.x + GSS_DEPTH_HS + j) * (blockDim.y + 2 * GSS_DEPTH_HS) + threadIdx.y + GSS_DEPTH_HS + i];
                if (disp > 0.f)
                {
                    const float w = 1.f; // expf(-0.5f*(i*i+j*j)*0.5f);
                    sx += w * j;
                    sy += w * i;
                    sz += w * disp;
                    sw += w;
                    sxx += w * j * j;
                    sxy += w * j * i;
                    sxz += w * j * disp;
                    syy += w * i * i;
                    syz += w * i * disp;
                    szz += w * disp * disp;
                }
            }
        }

        Eigen::Matrix3d A;
        A << sxx, sxy, sx,
            sxy, syy, sy,
            sx, sy, sw;

        Eigen::Vector3d b(sxz, syz, sz);

        float3 n = {0.f, 0.f, 0.f};

        float4 *normal_row = (float4 *)&((unsigned char *)normalsData)[y * normalsStep];

        if (A.determinant() > 1e-6)
        {
            Eigen::Vector3d plane = A.inverse() * b;

            n.x = plane.x() * cameraParams.f.x;
            n.y = plane.y() * cameraParams.f.y;
            n.z = plane.z() + plane.x() * (cameraParams.c.x - x) + plane.y() * (cameraParams.c.y - y);

            // n = {(float)plane.x(), (float)plane.y(), (float)plane.z()};
            n = normalize(n);
            if (n.z > 0.f)
            {
                n = -n;
            }
            normal_row[x] = make_float4(n.x, n.y, n.z, 1.f);
        }
        else
        {
            normal_row[x] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        // normal_row[x] = make_float4(p0.z, p0.z, p0.z, 0.f);
    }

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
        uint32_t rgbStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();

        __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
        //__shared__ uint32_t gids[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        bool inside = x < width && y < height;

        // if(threadIdx.x==0 && threadIdx.y==0)
        // {
        //     printf("ranges[%d] : %u %u\n", tileId, range.x, range.y);
        // }

        // collect gaussians data
        if (range.x + block.thread_rank() < range.y)
        {
            uint32_t gid = indices[range.x + block.thread_rank()];
            // gids[block.thread_rank()] = gid;
            splattedGaussians[block.thread_rank()].position = imgPositions[gid];
            splattedGaussians[block.thread_rank()].invSigma = imgInvSigmas[gid];
            splattedGaussians[block.thread_rank()].color = colors[gid];
            splattedGaussians[block.thread_rank()].alpha = alphas[gid];
        }
        block.sync();

        if (inside)
        {
            int n = min(range.y - range.x, BLOCK_SIZE);
            float3 color = make_float3(0.f, 0.f, 0.f);

            for (int i = 0; i < n; i++)
            {
                float dx = x - splattedGaussians[i].position.x;
                float dy = y - splattedGaussians[i].position.y;
                float v = splattedGaussians[i].invSigma.x * dx * dx + 2.f * splattedGaussians[i].invSigma.y * dx * dy + splattedGaussians[i].invSigma.z * dy * dy;

                if (v < 1.2f)
                {
                    float l = expf(-15.f * (v - 1.f) * (v - 1.f));
                    color = splattedGaussians[i].color;
                    color = max(color, make_float3(l, l, l));
                    break;
                }
            }

            float3 *rgb_row = (float3 *)&((unsigned char *)rgbData)[y * rgbStep];
            rgb_row[x] = color;
        }
    }

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
        uint32_t rgbStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();
        int tid = block.thread_rank();

        __shared__ SplattedGaussian splattedGaussian_sh[BLOCK_SIZE];
        __shared__ uint32_t gids_sh[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];
        int n = range.y - range.x;

        bool inside = x < width && y < height;
        bool done = !inside;

        float3 color = make_float3(0.f, 0.f, 0.f);

        Eigen::Vector3f ray((x - cameraParams.c.x) / cameraParams.f.x,
                            (y - cameraParams.c.y) / cameraParams.f.y,
                            1.f);
        ray.normalize();

        for (int k = 0; k < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; k++)
        {
            // collect gaussians data
            if (k * BLOCK_SIZE + tid < n)
            {
                uint32_t gid = indices[range.x + k * BLOCK_SIZE + tid];
                gids_sh[tid] = gid;
                splattedGaussian_sh[tid].position = imgPositions[gid];
                splattedGaussian_sh[tid].invSigma = imgInvSigmas[gid];
                splattedGaussian_sh[tid].color = colors[gid];
                splattedGaussian_sh[tid].alpha = alphas[gid];
            }
            block.sync();

            for (int i = 0; !done && i + k * BLOCK_SIZE < n && i < BLOCK_SIZE; i++)
            {
                float dx = x - splattedGaussian_sh[i].position.x;
                float dy = y - splattedGaussian_sh[i].position.y;
                float v = splattedGaussian_sh[i].invSigma.x * dx * dx + 2.f * splattedGaussian_sh[i].invSigma.y * dx * dy + splattedGaussian_sh[i].invSigma.z * dy * dy;
                float alpha = splattedGaussian_sh[i].alpha;

                if (v < 2.f && alpha > 0.01f)
                {
                    color = splattedGaussian_sh[i].color;

                    int idx = gids_sh[i];

                    // compute gaussian ellipsoid intersection with camera ray
                    Eigen::Map<const Eigen::Vector3f> p_gauss((float *)&positions[idx]);
                    Eigen::Map<const Eigen::Vector3f> p_cam((float *)&cameraPose.position);
                    Eigen::Map<const Eigen::Quaternionf> q_cam((float *)&cameraPose.orientation);
                    Eigen::Vector3f mu_cam = q_cam.inverse() * (p_gauss - p_cam);
                    Eigen::Map<const Eigen::Quaternionf> q_gauss((float *)&orientations[idx]);
                    Eigen::Map<const Eigen::Vector3f> s_gauss((float *)&scales[idx]);
                    Eigen::Matrix3f W = q_cam.conjugate().toRotationMatrix();
                    Eigen::Matrix3f R = q_gauss.toRotationMatrix();

                    Eigen::Vector3f is2_gauss(1.f / (s_gauss.x() * s_gauss.x()),
                                              1.f / (s_gauss.y() * s_gauss.y()),
                                              1.f / (s_gauss.z() * s_gauss.z()));

                    Eigen::Matrix<float, 3, 3> Sigma_i = W * R * is2_gauss.asDiagonal() * R.transpose() * W.transpose();

                    float a = ray.dot(Sigma_i * ray);
                    float b = 2.f * ray.dot(Sigma_i * mu_cam);
                    float c = mu_cam.dot(Sigma_i * mu_cam) - 2.f;
                    float delta = b * b - 4.f * a * c;

                    if (delta < 1e-6f)
                    {
                        // no intersection
                        continue;
                    }
                    else
                    {
                        // compute intersection
                        float t = (b - sqrtf(delta)) / (2.f * a);
                        Eigen::Vector3f p = t * ray;
                        // ompute normal
                        Eigen::Vector3f n = q_cam * (Sigma_i * (p - mu_cam));
                        n.normalize();
                        // compute lighting
                        Eigen::Map<const Eigen::Vector3f> lightDirection_eigen((float *)&lightDirection);
                        float l = fmaxf(0.f, n.dot(lightDirection_eigen));

                        color = (0.1f + 0.9f * l) * splattedGaussian_sh[i].color;
                        done = true;
                    }
                }
            }
        }

        if (inside)
        {
            float3 *rgb_row = (float3 *)&((unsigned char *)rgbData)[y * rgbStep];
            rgb_row[x] = color;
        }
    }

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
        uint32_t rgbStep)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        auto block = cg::this_thread_block();

        __shared__ SplattedGaussian splattedGaussians[BLOCK_SIZE];
        __shared__ uint32_t gids[BLOCK_SIZE];

        int tileId = blockIdx.y * numTiles.x + blockIdx.x;
        uint2 range = ranges[tileId];

        bool inside = x < width && y < height;

        // if(threadIdx.x==0 && threadIdx.y==0)
        // {
        //     printf("ranges[%d] : %u %u\n", tileId, range.x, range.y);
        // }

        // collect gaussians data
        if (range.x + block.thread_rank() < range.y)
        {
            uint32_t gid = indices[range.x + block.thread_rank()];
            gids[block.thread_rank()] = gid;
            splattedGaussians[block.thread_rank()].position = imgPositions[gid];
            splattedGaussians[block.thread_rank()].invSigma = imgInvSigmas[gid];
            splattedGaussians[block.thread_rank()].alpha = alphas[gid];
        }
        block.sync();

        if (inside)
        {
            int n = min(range.y - range.x, BLOCK_SIZE);
            float3 color = make_float3(0.f, 0.f, 0.f);

            for (int i = 0; i < n; i++)
            {
                float dx = x - splattedGaussians[i].position.x;
                float dy = y - splattedGaussians[i].position.y;
                float v = splattedGaussians[i].invSigma.x * dx * dx + 2.f * splattedGaussians[i].invSigma.y * dx * dy + splattedGaussians[i].invSigma.z * dy * dy;

                if (v < 1.f)
                {
                    PosGradVariance posGradVar = posGrad[gids[i]];
                    if (posGradVar.w == 0.f)
                        continue;

                    float2 grad = {posGradVar.dx / posGradVar.w,
                                   posGradVar.dy / posGradVar.w};

                    float3 cov = {posGradVar.dx2 / posGradVar.w - posGradVar.dx * posGradVar.dx / (posGradVar.w * posGradVar.w),
                                  posGradVar.dxdy / posGradVar.w - posGradVar.dx * posGradVar.dy / (posGradVar.w * posGradVar.w),
                                  posGradVar.dy2 / posGradVar.w - posGradVar.dy * posGradVar.dy / (posGradVar.w * posGradVar.w)};

                    // float grad_norm = hypotf(grad.x, grad.y);
                    // color.x = fabsf(1000.f*grad.x);
                    // color.y = fabsf(1000.f*grad.y);
                    // color.z = 0.f;//1000.f*grad_norm;

                    float cov_trace = cov.x + cov.z;
                    color.x = 1.e5 * cov_trace;
                    color.y = 1.e5 * cov_trace;
                    color.z = 1.e5 * cov_trace; // 1000.f*grad_norm;

                    break;
                }
            }

            float3 *rgb_row = (float3 *)&((unsigned char *)rgbData)[y * rgbStep];
            rgb_row[x] = color;
        }
    }

    __global__ void exportPLYGaussians_kernel(
        float *__restrict__ buffer,
        const float3 *__restrict__ positions,
        const float3 *__restrict__ scales,
        const float4 *__restrict__ orientations,
        const float3 *__restrict__ colors,
        const float *__restrict__ alphas,
        uint32_t nbGaussians)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= nbGaussians)
            return;

        Gaussian3D g;
        g.position = positions[idx];
        g.scale = scales[idx];
        g.orientation = orientations[idx];
        g.color = colors[idx];
        g.alpha = alphas[idx];

        buffer[14 * idx] = g.position.x;
        buffer[14 * idx + 1] = g.position.y;
        buffer[14 * idx + 2] = g.position.z;

        buffer[14 * idx + 3] = logf(g.scale.x);
        buffer[14 * idx + 4] = logf(g.scale.y);
        buffer[14 * idx + 5] = logf(g.scale.z);

        buffer[14 * idx + 6] = g.orientation.w;
        buffer[14 * idx + 7] = g.orientation.x;
        buffer[14 * idx + 8] = g.orientation.y;
        buffer[14 * idx + 9] = g.orientation.z;

        buffer[14 * idx + 10] = (g.color.z - 0.5f) / 0.2820948f;
        buffer[14 * idx + 11] = (g.color.y - 0.5f) / 0.2820948f;
        buffer[14 * idx + 12] = (g.color.x - 0.5f) / 0.2820948f;

        buffer[14 * idx + 13] = logf(g.alpha / (1.f - g.alpha));
    }
}
