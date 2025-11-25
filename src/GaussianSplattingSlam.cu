#include <gaussian_splatting_slam/GaussianSplattingSlam.hpp>
#include <gaussian_splatting_slam/GaussianSplattingSlamKernels.hpp>
#include "gaussian_splatting_slam/GSPoseCost.hpp"
#include <gaussian_splatting_slam/Preintegration.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/quaternion.hpp>

#include <cuda_utils/CudaErrorCheck.h>

#include <Eigen/Dense>

#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/scan.h>

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <chrono>

#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>

namespace gaussian_splatting_slam
{
    GaussianSplattingSlam::GaussianSplattingSlam()
        : nbGaussians(0), nbGaussiansMax(1000000), instanceCounterPtr(1, true, true),
          firstImage(true), deltaPosePtr(1, true, true), mtdPtr(1, true, true), initialized(false),
          visUnionPtr(1, true, true), visInterPtr(1, true, true), nbPyrLevels(3), w_depth(1.0), w_dist(0.1),
          poseIterations(5), updateIterations(10), etaPose(0.01f), etaUpdate(0.002f),
          gaussInitSizePx(7), iterationsSinceDensification(0), preint(new Preintegration)
    {
        gaussians.resize(nbGaussiansMax);
        deltaGaussians.resize(nbGaussiansMax);

        cameraParams.resize(nbPyrLevels);

        bgColor = make_float3(0.5f, 0.5f, 0.5f);

        indices.resize(nbGaussiansMax * 80);
        hashes.resize(nbGaussiansMax * 80);

        states.resize(nbGaussiansMax);

        keyframeVis.resize(nbGaussiansMax);
        frameVis.resize(nbGaussiansMax);

        posGradVar.resize(nbGaussiansMax);

        deltaGaussian3Ds.resize(nbGaussiansMax);
        adamStates.resize(nbGaussiansMax);
        cudaMemset(thrust::raw_pointer_cast(adamStates.data()), 0, sizeof(AdamStateGaussian3D) * nbGaussiansMax);

        outlierProb.resize(nbGaussiansMax);
        totalAlpha.resize(nbGaussiansMax);

        tileSize = make_uint2(16, 16);

        dxFilter = cv::cuda::createDerivFilter(CV_8UC4, CV_32FC4, 1, 0, 3, true, 1 / 255.);
        dyFilter = cv::cuda::createDerivFilter(CV_8UC4, CV_32FC4, 0, 1, 3, true, 1 / 255.);

        Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur, 9);
        Eigen::Map<Eigen::VectorXd> VB_prev_eig(VB_prev, 9);
        Eigen::Map<Eigen::Vector3d> P_prev_eig(P_prev);
        Eigen::Map<Eigen::Vector3d> P_cur_eig(P_cur);
        Eigen::Map<Eigen::Vector3d> V_prev_eig(VB_prev);

        VB_cur_eig.setZero();
        VB_prev_eig.setZero();

        GSPoseCost = new GSPoseCostFunction(this);
        imuCost = new ImuCostFunction(preint);
        marginalizationCost = new MarginalizationFactor;

        problem.AddParameterBlock(P_prev, 7, new PoseLocalParameterization());
        problem.AddParameterBlock(P_cur, 7);
        problem.SetParameterization(P_cur, new PoseLocalParameterization());
        problem.AddParameterBlock(VB_prev, 9);
        problem.AddParameterBlock(VB_cur, 9);

        problem.AddResidualBlock(GSPoseCost, nullptr, P_cur);
        problem.AddResidualBlock(imuCost, NULL, P_prev, VB_prev, P_cur, VB_cur);

        marginalization_info.addResidualBlockInfo(new ResidualBlockInfo(GSPoseCost, NULL, std::vector<double *>{P_cur}, std::vector<int>{}));
        marginalization_info.addResidualBlockInfo(new ResidualBlockInfo(imuCost, NULL, std::vector<double *>{P_prev, VB_prev, P_cur, VB_cur}, std::vector<int>{0, 1}));

        marginalization_info.init();
        std::unordered_map<long, double *> addr_shift;
        addr_shift[reinterpret_cast<long>(P_cur)] = P_prev;
        addr_shift[reinterpret_cast<long>(VB_cur)] = VB_prev;
        std::vector<double *> params = marginalization_info.getParameterBlocks(addr_shift);
        marginalizationCost->init(&marginalization_info);
        marginalization_info.addResidualBlockInfo(new ResidualBlockInfo(marginalizationCost, NULL, params, std::vector<int>{}));
        problem.AddResidualBlock(marginalizationCost, NULL, params);

        options.max_num_iterations = poseIterations;
        options.linear_solver_type = ceres::DENSE_QR;
        initialized = true;
    }
    GaussianSplattingSlam::~GaussianSplattingSlam()
    {
        delete GSPoseCost;
        stopOptimization = true;
        stopDisplay = true;
        optimizeThread.join();
        displayThread.join();
    }

    void GaussianSplattingSlam::setCameraParameters(const CameraParameters &params)
    {
        cameraParams[0] = params;
        for (int i = 1; i < nbPyrLevels; i++)
        {
            cameraParams[i].f = cameraParams[i - 1].f / 2.f;
            cameraParams[i].c = cameraParams[i - 1].c / 2.f;
        }
    }

    void GaussianSplattingSlam::setPose(const Pose3D &Pose)
    {
        cameraPose.position.x = Pose.position.x;       // t.x
        cameraPose.position.y = Pose.position.y;       // t.y
        cameraPose.position.z = Pose.position.z;       // t.z
        cameraPose.orientation.x = Pose.orientation.x; // x
        cameraPose.orientation.y = Pose.orientation.y; // y
        cameraPose.orientation.z = Pose.orientation.z; // z
        cameraPose.orientation.w = Pose.orientation.w; // w
    }

    void GaussianSplattingSlam::initialize(const Eigen::Vector3d t_imu_cam, const Eigen::Quaterniond q_imu_cam, const Pose3D &Pose, const Pose3D &PoseCam)
    {

        q_imu_cam_ = q_imu_cam;
        t_imu_cam_ = t_imu_cam;
        InitialPose.position.x = Pose.position.x;       // t.x
        InitialPose.position.y = Pose.position.y;       // t.y
        InitialPose.position.z = Pose.position.z;       // t.z
        InitialPose.orientation.x = Pose.orientation.x; // x
        InitialPose.orientation.y = Pose.orientation.y; // y
        InitialPose.orientation.z = Pose.orientation.z; // z
        InitialPose.orientation.w = Pose.orientation.w; // w

        InitialPoseCam.position.x = PoseCam.position.x;       // t.x
        InitialPoseCam.position.y = PoseCam.position.y;       // t.y
        InitialPoseCam.position.z = PoseCam.position.z;       // t.z
        InitialPoseCam.orientation.x = PoseCam.orientation.x; // x
        InitialPoseCam.orientation.y = PoseCam.orientation.y; // y
        InitialPoseCam.orientation.z = PoseCam.orientation.z; // z
        InitialPoseCam.orientation.w = PoseCam.orientation.w; // w
    }

    void GaussianSplattingSlam::setImuBias(const Eigen::Vector3d &b_a,
                                           const Eigen::Vector3d &b_g)
    {
        Eigen::Map<Eigen::Vector3d> BA_prev_eig(VB_prev + 3);
        Eigen::Map<Eigen::Vector3d> BG_prev_eig(VB_prev + 6);
        Eigen::Map<Eigen::Vector3d> BA_cur_eig(VB_cur + 3);
        Eigen::Map<Eigen::Vector3d> BG_cur_eig(VB_cur + 6);

        BA_cur_eig = BA_prev_eig = b_a;
        BG_cur_eig = BG_prev_eig = b_g;
    }

    void GaussianSplattingSlam::processIMU(const double t, const ImuData &imud)
    {
        if (preint->is_initialized)
        {
            double dt = t - last_imu_time;
            preint->add_imu(dt, imud.Acc, imud.Gyro);
        }
        else
        {
            Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur, 9);
            preint->init(imud.Acc, imud.Gyro,
                         VB_cur_eig.segment(3, 3), VB_cur_eig.segment(6, 3),

                         imud.acc_n, imud.gyr_n, imud.acc_w, imud.gyr_w);
        }
        last_imu = imud;
        last_imu_time = t;
    }

    void GaussianSplattingSlam::setNbPyrLevels(int n)
    {
        nbPyrLevels = n;
        setCameraParameters(cameraParams[0]);
    }

    void GaussianSplattingSlam::initAndCopyImgs(const cv::Mat &rgbImg, const cv::Mat &depthImg)
    {
        if (pyrColor.empty() || pyrColor[0].size() != rgbImg.size())
        {
            pyrColor.clear();
            pyrDepth.clear();
            pyrDepthTex.clear();
            pyrColorTex.clear();
            pyrNormal.clear();
            renderedRgbGpu.create(rgbImg.size(), CV_32FC3);
            renderedErrorGpu.create(rgbImg.size(), CV_32FC3);
            renderedDepthGpu.create(rgbImg.size(), CV_32FC1);
            renderedNormalsGpu.create(rgbImg.size(), CV_32FC3);
            densityMask.create(rgbImg.size(), CV_32FC1);

            texMask.reset(new Texture<float>(densityMask));

            pyrColor.resize(nbPyrLevels);
            pyrDepth.resize(nbPyrLevels);
            pyrNormal.resize(nbPyrLevels);
            pyrDx.resize(nbPyrLevels);
            pyrDy.resize(nbPyrLevels);
            pyrColorWarping.resize(nbPyrLevels);
            pyrDepthWarping.resize(nbPyrLevels);

            pyrColor[0].create(rgbImg.size(), CV_8UC4);
            pyrDepth[0].create(rgbImg.size(), CV_32FC1);
            pyrNormal[0].create(rgbImg.size(), CV_32FC4);

            pyrColorWarping[0].create(rgbImg.size(), CV_8UC4);
            pyrDepthWarping[0].create(rgbImg.size(), CV_32FC1);

            rgbImgGpu.upload(rgbImg);
            pyrDepth[0].upload(depthImg);

            cv::cuda::cvtColor(rgbImgGpu, pyrColor[0], cv::COLOR_BGR2BGRA, 4);

            for (int i = 1; i < nbPyrLevels; i++)
            {
                cv::cuda::pyrDown(pyrColor[i - 1], pyrColor[i]);
                cv::cuda::pyrDown(pyrDepth[i - 1], pyrDepth[i]);

                cv::cuda::pyrDown(pyrColorWarping[i - 1], pyrColorWarping[i]);
                cv::cuda::pyrDown(pyrDepthWarping[i - 1], pyrDepthWarping[i]);
            }

            pyrColorTex.resize(nbPyrLevels);
            pyrDepthTex.resize(nbPyrLevels);
            pyrNormalTex.resize(nbPyrLevels);
            pyrDxTex.resize(nbPyrLevels);
            pyrDyTex.resize(nbPyrLevels);

            pyrColorWarpingTex.resize(nbPyrLevels);
            pyrDepthWarpingTex.resize(nbPyrLevels);

            for (int i = 0; i < nbPyrLevels; i++)
            {
                pyrColorTex[i].reset(new Texture<uchar4>(pyrColor[i]));
                pyrDepthTex[i].reset(new Texture<float>(pyrDepth[i]));
                // pyrDepthTex.push_back(new Texture<float>(pyrDepth[i]));

                pyrColorWarpingTex[i].reset(new Texture<uchar4>(pyrColorWarping[i]));
                pyrDepthWarpingTex[i].reset(new Texture<float>(pyrDepthWarping[i]));

                pyrDx[i].create(pyrColor[i].size(), CV_32FC4);
                pyrDy[i].create(pyrColor[i].size(), CV_32FC4);

                dxFilter->apply(pyrColor[i], pyrDx[i]);
                dyFilter->apply(pyrColor[i], pyrDy[i]);

                pyrDxTex[i].reset(new Texture<float4>(pyrDx[i]));
                pyrDyTex[i].reset(new Texture<float4>(pyrDy[i]));
            }

            computeNormals();

            for (int i = 1; i < nbPyrLevels; i++)
            {
                cv::cuda::pyrDown(pyrNormal[i - 1], pyrNormal[i]);
            }
            for (int i = 0; i < nbPyrLevels; i++)
            {
                pyrNormalTex[i].reset(new Texture<float4>(pyrNormal[i]));
            }
        }
        else
        {
            rgbImgGpu.upload(rgbImg);
            pyrDepth[0].upload(depthImg);
            cv::cuda::cvtColor(rgbImgGpu, pyrColor[0], cv::COLOR_BGR2BGRA, 4);
            computeNormals();

            for (int i = 1; i < nbPyrLevels; i++)
            {
                cv::cuda::pyrDown(pyrColor[i - 1], pyrColor[i]);
                cv::cuda::pyrDown(pyrDepth[i - 1], pyrDepth[i]);
                cv::cuda::pyrDown(pyrNormal[i - 1], pyrNormal[i]);
            }

            for (int i = 0; i < nbPyrLevels; i++)
            {
                dxFilter->apply(pyrColor[i], pyrDx[i]);
                dyFilter->apply(pyrColor[i], pyrDy[i]);
            }
        }
    }

    void GaussianSplattingSlam::compute(const cv::Mat &rgbImg, const cv::Mat &depthImg, const Pose3D &odomPose)
    {
        cudaGetLastError();

        optimizationMutex.lock();

        initAndCopyImgs(rgbImg, depthImg);

        if (firstImage)
        {

            P_cur[0] = InitialPose.position.x;
            P_cur[1] = InitialPose.position.y;
            P_cur[2] = InitialPose.position.z;
            P_cur[3] = InitialPose.orientation.x;
            P_cur[4] = InitialPose.orientation.y;
            P_cur[5] = InitialPose.orientation.z;
            P_cur[6] = InitialPose.orientation.w;

            P_prev[0] = InitialPose.position.x;    // t.x
            P_prev[1] = InitialPose.position.y;    // t.y
            P_prev[2] = InitialPose.position.z;    // t.z
            P_prev[3] = InitialPose.orientation.x; // x
            P_prev[4] = InitialPose.orientation.y; // y
            P_prev[5] = InitialPose.orientation.z; // z
            P_prev[6] = InitialPose.orientation.w; // w

            cameraPose.position.x = InitialPoseCam.position.x;       // t.x
            cameraPose.position.y = InitialPoseCam.position.y;       // t.y
            cameraPose.position.z = InitialPoseCam.position.z;       // t.z
            cameraPose.orientation.x = InitialPoseCam.orientation.x; // x
            cameraPose.orientation.y = InitialPoseCam.orientation.y; // y
            cameraPose.orientation.z = InitialPoseCam.orientation.z; // z
            cameraPose.orientation.w = InitialPoseCam.orientation.w; // w

            std::cout << " Position initiale IMU x " << P_cur[0] << "  y " << P_cur[1] << "  z " << P_cur[2] << std::endl;
            std::cout << " Orientation initiale IMU x " << P_cur[3] << "  y " << P_cur[4] << "  z " << P_cur[5] << "  w " << P_cur[6] << std::endl;

            std::cout << " Position initiale CAMERA x " << cameraPose.position.x << "  y " << cameraPose.position.y << "  z " << cameraPose.position.z << std::endl;
            std::cout << " Orientation initiale CAMERA x " << cameraPose.orientation.x << "  y " << cameraPose.orientation.y << "  z " << cameraPose.orientation.z << "  w " << cameraPose.orientation.w << std::endl;

            prevOdomPose = odomPose;

            std::cerr << "generate Gaussians" << std::endl;
            generateGaussians();

            addKeyframe();

            optimizeThread = std::thread(&GaussianSplattingSlam::optimizationLoop, this);

            firstImage = false;

            std::cerr << "nb Gaussians : " << nbGaussians << std::endl;
        } // firstImage

        if (preint->is_initialized)
        {
            float et;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            Eigen::Map<Eigen::Quaterniond> Q_cur_eig(P_cur + 3);
            Eigen::Map<Eigen::Vector3d> P_cur_eig(P_cur);
            Eigen::Map<Eigen::Quaterniond> Q_prev_eig(P_prev + 3);
            Eigen::Map<Eigen::Vector3d> P_prev_eig(P_prev);
            Eigen::Map<Eigen::VectorXd> VB_cur_eig(VB_cur, 9);
            Eigen::Map<Eigen::VectorXd> VB_prev_eig(VB_prev, 9);

            Eigen::Map<Eigen::Vector3d> V_prev_eig(VB_prev);
            Eigen::Map<Eigen::Vector3d> V_cur_eig(VB_cur);

            options.max_num_iterations = poseIterations;

            Eigen::Vector3d Pj, Vj;
            Eigen::Quaterniond Qj;

            preint->predict(P_prev_eig, Q_prev_eig, V_prev_eig,
                            Pj, Qj, Vj);
            P_cur_eig = Pj;
            Q_cur_eig = Qj;
            V_cur_eig = Vj;

            // std::cout << "P_prev_eig : " << P_prev_eig.transpose() << std::endl;
            // std::cout << "Q_prev_eig : " << Q_prev_eig.vec().transpose() << std::endl;
            // std::cout << "Q_prev_Mat : " << Q_prev_eig.toRotationMatrix() << std::endl;
            // std::cout << "V_prev_eig : " << V_prev_eig.transpose() << std::endl;
            // std::cout << "P_cur_eig : " << P_cur_eig.transpose() << std::endl;
            // std::cout << "Q_cur_eig : " << Q_cur_eig.vec().transpose() << std::endl;
            // std::cout << "V_cur_eig : " << V_cur_eig.transpose() << std::endl;

            // std::cout << "preint sum_dt : " << preint->sum_dt << std::endl;
            // std::cout << " V_prev_eig x " << V_prev_eig[0] << "  y " << V_prev_eig[1] << "  z " << V_prev_eig[2] << std::endl;
            // std::cout << " P_prev_eig Pose x " << P_prev_eig[0] << "  y " << P_prev_eig[1] << "  z " << P_prev_eig[2] << std::endl;
            // std::cout << " P_prev_eig orient x " << P_prev_eig[3] << "   y " << P_prev_eig[4] << "   z " << P_prev_eig[5] << "   w " << P_prev_eig[6] << std::endl;
            // std::cout << " V_cur_eig x " << V_cur_eig[0] << "  y " << V_cur_eig[1] << "  z " << V_cur_eig[2] << std::endl;
            // std::cout << " P_cur_eig Pose x " << P_cur_eig[0] << "  y " << P_cur_eig[1] << "  z " << P_cur_eig[2] << std::endl;
            // std::cout << " Q_cur_eig orient x " << Q_cur_eig.x() << "   y " << Q_cur_eig.y() << "   z " << Q_cur_eig.z() << "   w " << Q_cur_eig.w() << std::endl;

            // std::cout << "delta_v : " << preint->delta_v.transpose() << std::endl;

            if (poseEstimationMethod == PoseEstimationMethodWarpingSingleRendering)
            {
                Pose3D camPose;

                Eigen::Vector3d P_cam = Pj + Qj * t_imu_cam_;
                Eigen::Quaterniond Q_cam = Qj * q_imu_cam_;

                // Eigen::Vector3d P_cam = P_prev_eig + Q_prev_eig * t_imu_cam_;
                // Eigen::Quaterniond Q_cam = Q_prev_eig * q_imu_cam_;

                camPose.position.x = P_cam.x();
                camPose.position.y = P_cam.y();
                camPose.position.z = P_cam.z();
                camPose.orientation.x = Q_cam.x();
                camPose.orientation.y = Q_cam.y();
                camPose.orientation.z = Q_cam.z();
                camPose.orientation.w = Q_cam.w();

                initWarping(camPose);
            }

            for (int l = nbPyrLevels - 1; l >= 1; l--)
            {
                GSPoseCost->update(l);
                Solve(options, &problem, &summary);
            }

            // std::cout << " P_cur_eig Pose x " << P_cur_eig[0] << "  y " << P_cur_eig[1] << "  z " << P_cur_eig[2] << std::endl;
            // std::cout << " Q_cur_eig orient x " << Q_cur_eig.x() << "   y " << Q_cur_eig.y() << "   z " << Q_cur_eig.z() << "   w " << Q_cur_eig.w() << std::endl;

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&et, start, stop);

            //std::cout << "pose estimation time : " << et << " ms" << std::endl;

            Eigen::Vector3f imu_trans = Eigen::Vector3f(P_cur[0], P_cur[1], P_cur[2]);
            Eigen::Quaternionf imu_rot = Eigen::Quaternionf(P_cur[6], P_cur[3], P_cur[4], P_cur[5]);
            Eigen::Vector3f position_cam = imu_trans + imu_rot.normalized().toRotationMatrix() * t_imu_cam_.cast<float>();
            cameraPose.position = make_float3(position_cam.x(), position_cam.y(), position_cam.z());
            Eigen::Quaternionf orientation_cam = imu_rot * q_imu_cam_.cast<float>();
            orientation_cam.normalize();
            cameraPose.orientation = make_float4(orientation_cam.x(), orientation_cam.y(), orientation_cam.z(), orientation_cam.w());

            removeOutliers();

            float covisRatio = computeCovisibilityRatio(keyframes[currentKeyframe]);
            // if (covisRatio < covisibilityThreshold)
            // {
            //     for (int i = 0; i < keyframes.size(); i++)
            //     {
            //         if (i == currentKeyframe)
            //         {
            //             continue;
            //         }
            //         covisRatio = computeCovisibilityRatio(keyframes[i]);
            //         if (covisRatio > covisibilityThreshold)
            //         {
            //             currentKeyframe = i;
            //             break;
            //         }
            //     }
            // }

            if (covisRatio < covisibilityThreshold)
            {
                std::cerr << "New keyframe" << std::endl;
                options.max_num_iterations = poseIterations;

                auto start_process_frame = std::chrono::high_resolution_clock::now();
                for (int l = nbPyrLevels - 1; l >= 0; l--)
                {
                    GSPoseCost->update(l);
                    Solve(options, &problem, &summary);
                }
                auto end_process_frame = std::chrono::high_resolution_clock::now();
                auto duration_process_frame = std::chrono::duration_cast<std::chrono::milliseconds>(end_process_frame - start_process_frame);
                std::cout << " Time to process the keyframe: " << duration_process_frame.count() << " milliseconds" << std::endl;

                imu_trans = Eigen::Vector3f(P_cur[0], P_cur[1], P_cur[2]);
                imu_rot = Eigen::Quaternionf(P_cur[6], P_cur[3], P_cur[4], P_cur[5]);
                position_cam = imu_trans + imu_rot.normalized().toRotationMatrix() * t_imu_cam_.cast<float>();
                cameraPose.position = make_float3(position_cam.x(), position_cam.y(), position_cam.z());
                orientation_cam = imu_rot * q_imu_cam_.cast<float>();
                orientation_cam.normalize();
                cameraPose.orientation = make_float4(orientation_cam.x(), orientation_cam.y(), orientation_cam.z(), orientation_cam.w());

                prune();
                addKeyframe();
                densify(keyframes[currentKeyframe]);
            }

            // cameraPose.position = make_float3(P_cur[0], P_cur[1], P_cur[2]);
            // cameraPose.orientation = make_float4(P_cur[3], P_cur[4], P_cur[5], P_cur[6]);

            // std::cout << " Imu Pose x " << P_cur[0] << "  y " << P_cur[1] << "  z " << P_cur[2] << std::endl;
            // std::cout << " IMu orient x " << P_cur[3] << "   y " << P_cur[4] << "   z " << P_cur[5] << "   w " << P_cur[6] << std::endl;
            // std::cout << " camera pose X : " << cameraPose.position.x << " Y : " << cameraPose.position.y << " Z : " << cameraPose.position.z << std::endl;
            // std::cout << " camera orient x : " << cameraPose.orientation.x << " Yrot : " << cameraPose.orientation.y << " Zrot : " << cameraPose.orientation.z << " Wrot : " << cameraPose.orientation.w << std::endl;

            marginalization_info.preMarginalize();
            marginalization_info.marginalize();

            Q_prev_eig = Q_cur_eig;
            P_prev_eig = P_cur_eig;
            VB_prev_eig = VB_cur_eig;

            preint->init(last_imu.Acc, last_imu.Gyro,
                         VB_cur_eig.segment(3, 3), VB_cur_eig.segment(6, 3),
                         last_imu.acc_n, last_imu.gyr_n, last_imu.acc_w, last_imu.gyr_w);
        }

        // std::cout << " X : " << cameraPose.position.x << " Y : " << cameraPose.position.y << " Z : " << cameraPose.position.z << std::endl;
        nbImagesProcessed++;
        prevOdomPose = odomPose;

        optimizationMutex.unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    void GaussianSplattingSlam::display()
    {
        optimizationMutex.lock();
        if (nbGaussians == 0)
        {
            optimizationMutex.unlock();
            return;
        }

        rasterize();
        // rasterizeBlobs(renderedBlobsGpu);
        // rasterizeFill();
        //  rasterizeNormals();
        // rasterizeError(keyframes[currentKeyframe]);
        pyrColor[0].download(rgbImg);
        pyrDepth[0].download(depthImg);

        // render3dView();

        optimizationMutex.unlock();

        cv::imshow("rgb", rgbImg);
        cv::imshow("depth", 0.15f * depthImg);

        renderedRgbGpu.download(renderedRgb);
        // rgbaImgGpu.download(renderedRgb);
        renderedDepthGpu.download(renderedDepth);
        // renderedErrorGpu.download(renderedError);

        // renderedBlobsGpu.download(renderedBlobs);

        // pyrDx[1].download(imgDx);
        // cv::imshow("dx", imgDx/255);

        // densityMask.download(maskImg);

        // renderedNormalsGpu.download(renderedNormals);
        // pyrNormal[0].download(computedNormals);

        cv::imshow("rendered_rgb", renderedRgb);
        // cv::imshow("rendered_error", renderedError);
        cv::imshow("rendered_depth", 0.15f * renderedDepth);
        // cv::imshow("rendered_blob", renderedBlobs);
        // cv::imshow("mask", maskImg);
        // cv::imshow("rendered_normals", cv::Scalar(0.5f) + 0.5f * renderedNormals);
        // cv::imshow("computed_normals", cv::Scalar(0.5f) + 0.5f * computedNormals);

        if (cv::waitKey(1) == 's')
        {
            std::cout << "save model to /tmp/model.ply" << std::endl;

            optimizationMutex.lock();
            exportPLY("/tmp/model.ply");
            optimizationMutex.unlock();
        }

        // for (int i = 0; i < keyframes.size(); i++)
        // {
        //     renderKeyframe(keyframes[i]);

        //     renderedRgbGpu.download(renderedRgb);
        //     renderedDepthGpu.download(renderedDepth);
        //     /*
        //     keyframes[i].getRgbImg().download(rgbImg);
        //     keyframes[i].getDepthImg().download(depthImg);
        //     cv::imshow(std::string("rgb_")+std::to_string(i), rgbImg);
        //     cv::imshow(std::string("depth_")+std::to_string(i), 0.15f * depthImg);
        //     */
        //     cv::imshow(std::string("rendered_rgb_") + std::to_string(i), renderedRgb);
        //     cv::imshow(std::string("rendered_depth_") + std::to_string(i), 0.15f * renderedDepth);
        // }
    }

    void GaussianSplattingSlam::generateGaussians()
    {
        /* generate gaussians */
        // CudaCheckError();

        *instanceCounterPtr.data_host() = 0;
        instanceCounterPtr.upload();

        dim3 dimBlock(16, 16);
        dim3 dimGridGen(((pyrColor[0].cols / gaussInitSizePx) + dimBlock.x - 1) / dimBlock.x,
                        ((pyrColor[0].rows / gaussInitSizePx) + dimBlock.y - 1) / dimBlock.y);

        generateGaussians_kernel<<<dimGridGen, dimBlock>>>(
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            instanceCounterPtr.data(),
            pyrColorTex[0]->getTextureObject(),
            pyrDepthTex[0]->getTextureObject(),
            pyrNormalTex[0]->getTextureObject(),
            cameraPose,
            cameraParams[0],
            gaussInitSizePx,
            gaussInitSizePx,
            pyrColor[0].cols,
            pyrColor[0].rows);

        // CudaCheckError();

        instanceCounterPtr.download();
        nbGaussians = *instanceCounterPtr.data_host();
    }

    void GaussianSplattingSlam::prepareRasterization(const Pose3D &cameraPose_,
                                                     const CameraParameters &cameraParams_,
                                                     int width,
                                                     int height)
    {
        // std::cerr<<"begin prepare rasterization" << std::endl;

        *instanceCounterPtr.data_host() = 0;
        instanceCounterPtr.upload();

        numTiles = make_uint2((width + tileSize.x - 1) / tileSize.x,
                              (height + tileSize.y - 1) / tileSize.y);
        ranges.resize(numTiles.x * numTiles.y);

        computeScreenSpaceParamsAndHashes_kernel<<<(nbGaussians + 127) / 128, 128>>>(
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(hashes.data()),
            instanceCounterPtr.data(),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.normals.data()),
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            cameraPose_,
            cameraParams_,
            0.4f,
            tileSize,
            numTiles,
            nbGaussians,
            width,
            height);

        instanceCounterPtr.download();
        uint32_t nbInstances = *instanceCounterPtr.data_host();

        // std::cout << "nbGaussians : " << nbGaussians << std::endl;
        // std::cout << "nbInstances : " << nbInstances << std::endl;
        uint2 nullRange = make_uint2(0, 0);
        thrust::fill(ranges.begin(), ranges.end(), nullRange);

        if (nbInstances <= 0)
            return;

        thrust::sort_by_key(thrust::cuda::par(allocator),
                            hashes.begin(),
                            hashes.begin() + nbInstances,
                            indices.begin());

        computeIndicesRanges_kernel<<<(nbInstances + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(hashes.data()),
            nbInstances);

        // std::cerr<<"end prepare rasterization" << std::endl;
    }

    void GaussianSplattingSlam::rasterize()
    {
        prepareRasterization(cameraPose, cameraParams[0], pyrColor[0].cols, pyrColor[0].rows);

        rasterizeGaussians_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)renderedRgbGpu.data,
            (float *)renderedDepthGpu.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            bgColor,
            numTiles,
            renderedRgbGpu.cols,
            renderedRgbGpu.rows,
            renderedRgbGpu.step,
            renderedDepthGpu.step);

        // CudaCheckError();
    }

    void GaussianSplattingSlam::rasterizeNormals()
    {
        prepareRasterization(cameraPose, cameraParams[0], pyrColor[0].cols, pyrColor[0].rows);

        rasterizeGaussiansNormals_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)renderedNormalsGpu.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(gaussians.normals.data()),
            numTiles,
            renderedNormalsGpu.cols,
            renderedNormalsGpu.rows,
            renderedNormalsGpu.step);
    }

    void GaussianSplattingSlam::rasterizeFill(cv::cuda::GpuMat &img)
    {
        img.create(pyrColor[0].size(), CV_32FC3);
        prepareRasterization(cameraPose, cameraParams[0], pyrColor[0].cols, pyrColor[0].rows);

        rasterizeGaussiansFill_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)img.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            numTiles,
            img.cols,
            img.rows,
            img.step);
    }

    void GaussianSplattingSlam::rasterizeBlobs(cv::cuda::GpuMat &img)
    {
        img.create(pyrColor[0].size(), CV_32FC3);

        prepareRasterization(cameraPose, cameraParams[0], pyrColor[0].cols, pyrColor[0].rows);

        float3 lightDirection = normalize(make_float3(-0.3, 1., 1.f));

        rasterizeGaussiansBlobs_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)img.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            cameraPose,
            cameraParams[0],
            lightDirection,
            numTiles,
            img.cols,
            img.rows,
            img.step);
    }

    void GaussianSplattingSlam::computeNormals()
    {
        dim3 dimBlock(GSS_BLOCK_X, GSS_BLOCK_Y);
        dim3 dimGridGen((pyrNormal[0].cols + dimBlock.x - 1) / dimBlock.x,
                        (pyrNormal[0].rows + dimBlock.y - 1) / dimBlock.y);

        computeNormalsFromDepth_kernel<<<dimGridGen, dimBlock>>>(
            (float4 *)pyrNormal[0].data,
            pyrDepthTex[0]->getTextureObject(),
            cameraParams[0],
            pyrNormal[0].cols,
            pyrNormal[0].rows,
            pyrNormal[0].step);
    }

    void GaussianSplattingSlam::rasterizeError(GaussianSplattingKeyframe &keyframe)
    {
        prepareRasterization(keyframe.getCameraPose(),
                             keyframe.getCameraParams(),
                             keyframe.getImgWidth(),
                             keyframe.getImgHeight());

        PosGradVariance nullPosGradVar = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        thrust::fill(posGradVar.begin(), posGradVar.begin() + nbGaussians, nullPosGradVar);
        computePosGradVariance_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(posGradVar.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            keyframe.getColorTex().getTextureObject(),
            keyframe.getDepthTex().getTextureObject(),
            1.f,
            5.f,
            bgColor,
            numTiles,
            densityMask.cols,
            densityMask.rows);

        rasterizeGaussiansError_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)renderedErrorGpu.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(posGradVar.data()),
            numTiles,
            renderedErrorGpu.cols,
            renderedErrorGpu.rows,
            renderedErrorGpu.step);
    }

    void GaussianSplattingSlam::optimizeGaussians(int nbIterations, float eta)
    {
        int k;
        for (k = 0; k < nbIterations;)
        {
            for (int i = keyframes.size() - 1; i >= ((int)keyframes.size()) - 2 && i >= 0; i--, k++)
            {
                optimizeGaussiansKeyframe2(keyframes[i], eta);
            }

            if (keyframes.size() > 2)
            {
                for (int i = 0; i < 2; i++, k++)
                {
                    int keyframeIdx = keyframes.size() - 1 - betaBinomialGenerator.sampleBetaBinomial(keyframes.size() - 1, 0.7, 2.);

                    optimizeGaussiansKeyframe2(keyframes[keyframeIdx], eta);
                    // optimizeGaussiansKeyframe2(keyframes[rand() % (keyframes.size() - 2)], eta);
                }
            }
        }

        iterationsSinceDensification += k;
        if (iterationsSinceDensification > 400)
        {
            std::cout << "prune & split gaussians" << std::endl;
            prune();
            // splitGaussians(keyframes[currentKeyframe]);
            iterationsSinceDensification = 0;
        }
    }

    void GaussianSplattingSlam::optimizeGaussiansKeyframe(GaussianSplattingKeyframe &keyframe, float eta)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        prepareRasterization(keyframe.getCameraPose(),
                             keyframe.getCameraParams(),
                             keyframe.getImgWidth(),
                             keyframe.getImgHeight());

        cudaMemset((void *)thrust::raw_pointer_cast(deltaGaussians.data()),
                   0,
                   nbGaussians * sizeof(DeltaGaussian2D));

        float3 bg = {drand48(), drand48(), drand48()};

        optimizeGaussians2_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(deltaGaussians.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            keyframe.getColorTex().getTextureObject(),
            keyframe.getDepthTex().getTextureObject(),
            1., // w_depth // 1.
            5., // w_dist   // 5.
            bg,
            numTiles,
            keyframe.getImgWidth(),
            keyframe.getImgHeight());

        applyDeltaGaussians2_kernel<<<(nbGaussians + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(deltaGaussians.data()),
            keyframe.getCameraPose(),
            keyframe.getCameraParams(),
            eta,  // eta
            0.1f, // 0.5f,  // 0.01f,  // lamda_iso
            nbGaussians);

        float et;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        std::cout << "optimization iteration time : " << et << " ms" << std::endl;
    }

    void GaussianSplattingSlam::optimizeGaussiansKeyframe2(GaussianSplattingKeyframe &keyframe, float eta)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // CudaCheckError();

        prepareRasterization(keyframe.getCameraPose(),
                             keyframe.getCameraParams(),
                             keyframe.getImgWidth(),
                             keyframe.getImgHeight());

        int num_tiles = numTiles.x * numTiles.y;
        perTileBuckets.resize(num_tiles);
        perTileBucketCount<<<(num_tiles + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(perTileBuckets.data()),
            thrust::raw_pointer_cast(ranges.data()),
            num_tiles);
        thrust::inclusive_scan(thrust::cuda::par(allocator), perTileBuckets.begin(), perTileBuckets.end(), perTileBuckets.begin());
        uint32_t numBuckets = perTileBuckets[perTileBuckets.size() - 1];

        uint32_t numPixels = keyframe.getImgWidth() * keyframe.getImgHeight();

        bucketToTile.resize(numBuckets);
        sampled_T.resize(numBuckets * GSS_BLOCK_SIZE);
        sampled_ar.resize(numBuckets * GSS_BLOCK_SIZE);

        cudaMemset((void *)thrust::raw_pointer_cast(deltaGaussians.data()),
                   0,
                   nbGaussians * sizeof(DeltaGaussian2D));

        float3 bg = {drand48(), drand48(), drand48()};

        final_T.resize(numPixels);
        n_contrib.resize(numPixels);
        max_contrib.resize(num_tiles);
        output_color.resize(numPixels);
        output_depth.resize(numPixels);
        color_error.resize(numPixels);
        depth_error.resize(numPixels);

        // std::cout << "numBuckets : " << numBuckets << std::endl;
        // std::cout << "numTiles : " << num_tiles << std::endl;

        optimizeGaussiansForwardPass<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(perTileBuckets.data()),
            thrust::raw_pointer_cast(bucketToTile.data()),
            thrust::raw_pointer_cast(sampled_T.data()),
            thrust::raw_pointer_cast(sampled_ar.data()),
            thrust::raw_pointer_cast(final_T.data()),
            thrust::raw_pointer_cast(n_contrib.data()),
            thrust::raw_pointer_cast(max_contrib.data()),
            thrust::raw_pointer_cast(output_color.data()),
            thrust::raw_pointer_cast(output_depth.data()),
            thrust::raw_pointer_cast(color_error.data()),
            thrust::raw_pointer_cast(depth_error.data()),
            keyframe.getColorTex().getTextureObject(),
            keyframe.getDepthTex().getTextureObject(),
            bg,
            numTiles,
            keyframe.getImgWidth(),
            keyframe.getImgHeight());

        optimizeGaussiansPerGaussianPass<<<numBuckets, 32>>>(
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(perTileBuckets.data()),
            thrust::raw_pointer_cast(bucketToTile.data()),
            thrust::raw_pointer_cast(sampled_T.data()),
            thrust::raw_pointer_cast(sampled_ar.data()),
            thrust::raw_pointer_cast(final_T.data()),
            thrust::raw_pointer_cast(n_contrib.data()),
            thrust::raw_pointer_cast(max_contrib.data()),
            thrust::raw_pointer_cast(output_color.data()),
            thrust::raw_pointer_cast(output_depth.data()),
            thrust::raw_pointer_cast(color_error.data()),
            thrust::raw_pointer_cast(depth_error.data()),
            thrust::raw_pointer_cast(deltaGaussians.data()),
            bg,
            w_depth, // w_depth 1.f
            w_dist,  // w_dist 0.1 0.2f 5.
            numTiles,
            keyframe.getImgWidth(),
            keyframe.getImgHeight(),
            numBuckets);

        // CudaCheckError();

        // DeltaGaussian dg = deltaGaussians[1000];
        // std::cout << "dl_color : " << dg.color.x << ' ' << dg.color.y << ' ' << dg.color.z << std::endl;
        // std::cout << "dl_n : " << dg.n << std::endl;

        if (optimizer == "grad")
        {

            applyDeltaGaussians2_kernel<<<(nbGaussians + 255) / 256, 256>>>(
                thrust::raw_pointer_cast(gaussians.positions.data()),
                thrust::raw_pointer_cast(gaussians.scales.data()),
                thrust::raw_pointer_cast(gaussians.orientations.data()),
                thrust::raw_pointer_cast(gaussians.colors.data()),
                thrust::raw_pointer_cast(gaussians.alphas.data()),
                thrust::raw_pointer_cast(deltaGaussians.data()),
                keyframe.getCameraPose(),
                keyframe.getCameraParams(),
                eta,  // eta
                0.1f, // 0.5f,  // 0.01f,  // lamda_iso
                nbGaussians);
        }
        else if (optimizer == "adam")
        {

            computeDeltaGaussians3D_kernel<<<(nbGaussians + 255) / 256, 256>>>(
                thrust::raw_pointer_cast(deltaGaussian3Ds.data()),
                thrust::raw_pointer_cast(gaussians.positions.data()),
                thrust::raw_pointer_cast(gaussians.scales.data()),
                thrust::raw_pointer_cast(gaussians.orientations.data()),
                thrust::raw_pointer_cast(gaussians.colors.data()),
                thrust::raw_pointer_cast(gaussians.alphas.data()),
                thrust::raw_pointer_cast(deltaGaussians.data()),
                keyframe.getCameraPose(),
                keyframe.getCameraParams(),
                0.01f, // 0.5f,  // 0.01f,  // lamda_iso
                nbGaussians);
            updateGaussiansParametersAdam_kernel<<<(nbGaussians + 255) / 256, 256>>>(
                thrust::raw_pointer_cast(gaussians.positions.data()),
                thrust::raw_pointer_cast(gaussians.scales.data()),
                thrust::raw_pointer_cast(gaussians.orientations.data()),
                thrust::raw_pointer_cast(gaussians.colors.data()),
                thrust::raw_pointer_cast(gaussians.alphas.data()),
                thrust::raw_pointer_cast(adamStates.data()),
                thrust::raw_pointer_cast(deltaGaussian3Ds.data()),
                adamEta,
                adamBeta1,
                adamBeta2,
                adamEpsilon,
                nbGaussians);
            CudaCheckError();
        }

        float et;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&et, start, stop);
        // std::cout << "optimization2 iteration time : " << et << " ms" << std::endl;
    }

    void GaussianSplattingSlam::optimizePose(int nbIterations, float eta, int minLevel)
    {
        float dlength = 1.;
        for (int l = nbPyrLevels - 1; l >= minLevel; l--)
        {
            for (int k = 0; dlength > 1e-4 && k < nbIterations; k++)
            {
                prepareRasterization(cameraPose, cameraParams[l], pyrColor[l].cols, pyrColor[l].rows);

                /*
                cudaMemset((void *)thrust::raw_pointer_cast(deltaGaussians.data()),
                           0,
                           nbGaussians * sizeof(DeltaGaussian));

                optimizePose_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                    thrust::raw_pointer_cast(deltaGaussians.data()),
                    thrust::raw_pointer_cast(ranges.data()),
                    thrust::raw_pointer_cast(indices.data()),
                    thrust::raw_pointer_cast(gaussians.imgPositions.data()),
                    thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.colors.data()),
                    thrust::raw_pointer_cast(gaussians.alphas.data()),
                    texRGBA->getTextureObject(),
                    texDepth->getTextureObject(),
                    numTiles,
                    rgbImgGpu.cols,
                    rgbImgGpu.rows);

                cudaMemset(deltaPosePtr.data(),
                           0,
                           sizeof(DeltaPose3D));

                applyDeltaPose_kernel<<<(nbGaussians + GSS_BLOCK_SIZE - 1) / GSS_BLOCK_SIZE, GSS_BLOCK_SIZE>>>(
                    deltaPosePtr.data(),
                    thrust::raw_pointer_cast(gaussians.positions.data()),
                    thrust::raw_pointer_cast(gaussians.scales.data()),
                    thrust::raw_pointer_cast(gaussians.orientations.data()),
                    thrust::raw_pointer_cast(deltaGaussians.data()),
                    cameraPose,
                    cameraParams,
                    eta, // eta
                    nbGaussians);

                */
                cudaMemset(deltaPosePtr.data(),
                           0,
                           sizeof(DeltaPose3D));

                optimizePose3_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                    deltaPosePtr.data(),
                    thrust::raw_pointer_cast(ranges.data()),
                    thrust::raw_pointer_cast(indices.data()),
                    thrust::raw_pointer_cast(gaussians.imgPositions.data()),
                    thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.colors.data()),
                    thrust::raw_pointer_cast(gaussians.alphas.data()),
                    pyrColorTex[l]->getTextureObject(),
                    pyrDepthTex[l]->getTextureObject(),
                    cameraPose,
                    cameraParams[0],
                    eta, // eta
                    numTiles,
                    pyrColor[0].cols,
                    pyrColor[0].rows);

                deltaPosePtr.download();

                /*std:: cout << "dp : " << deltaPosePtr.data_host()->dp.x
                << " " << deltaPosePtr.data_host()->dp.y
                << " " << deltaPosePtr.data_host()->dp.z
                << " / n : " << deltaPosePtr.data_host()->n << std::endl;
                */
                float n = deltaPosePtr.data_host()->n;
                cameraPose.position = cameraPose.position + eta * deltaPosePtr.data_host()->dp / n;
                // std::cout << "delta pos : "
                //           << deltaPosePtr.data_host()->dp.x / n << " "
                //           << deltaPosePtr.data_host()->dp.y / n << " "
                //           << deltaPosePtr.data_host()->dp.z / n << std::endl;
                // std::cout << "delta q : "
                //           << deltaPosePtr.data_host()->dq.x / n << " "
                //           << deltaPosePtr.data_host()->dq.y / n << " "
                //           << deltaPosePtr.data_host()->dq.z / n << std::endl;

                Eigen::Map<Eigen::Vector3f> delta_q((float *)&(deltaPosePtr.data_host()->dq));
                Eigen::Map<Eigen::Quaternionf> q((float *)&(cameraPose.orientation));
                const float mult = eta * 0.5f / n;
                q = q * Eigen::Quaternionf(1.f, mult * delta_q.x(), mult * delta_q.y(), mult * delta_q.z());
                q.normalize();

                dlength = std::max(length(deltaPosePtr.data_host()->dp) / n, length(deltaPosePtr.data_host()->dq) / n);
                if (dlength < 1e-4)
                {
                    std::cout << "dlength<1e-4" << std::endl;
                }
            }
        }
    }

    void GaussianSplattingSlam::optimizePoseGN(int nbIterations, float eta, int minLevel)
    {
        Eigen::Matrix<float, 6, 6> JtJ;
        Eigen::Vector<float, 6> Jtr;

        Pose3D currentPose = cameraPose;

        int nb_it = 0;
        float total_time = 0.f;
        for (int l = nbPyrLevels - 1; l >= minLevel; l--)
        {
            for (int it = 0; it < nbIterations; it++)
            {
                nb_it++;

                // CudaCheckError();
                prepareRasterization(currentPose, cameraParams[l], pyrColor[l].cols, pyrColor[l].rows);

                cudaMemset(mtdPtr.data(),
                           0,
                           sizeof(MotionTrackingData));
                float et;
                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);

                optimizePoseGN3_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                    mtdPtr.data(),
                    thrust::raw_pointer_cast(ranges.data()),
                    thrust::raw_pointer_cast(indices.data()),
                    thrust::raw_pointer_cast(gaussians.imgPositions.data()),
                    thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.pHats.data()),
                    thrust::raw_pointer_cast(gaussians.colors.data()),
                    thrust::raw_pointer_cast(gaussians.alphas.data()),
                    pyrColorTex[l]->getTextureObject(),
                    pyrDepthTex[l]->getTextureObject(),
                    pyrDxTex[l]->getTextureObject(),
                    pyrDyTex[l]->getTextureObject(),
                    currentPose,
                    cameraParams[l],
                    bgColor,
                    0.2f, // alpha threshold
                    0.1f, // color threshold
                    0.2f, // depth threshold
                    numTiles,
                    pyrColor[l].cols,
                    pyrColor[l].rows);

                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&et, start, stop);
                total_time += et;

                mtdPtr.download();

                int k = 0;
                for (int i = 0; i < 6; i++)
                {
                    for (int j = i; j < 6; j++, k++)
                    {
                        JtJ(i, j) = JtJ(j, i) = mtdPtr.data_host()->JtJ[k];
                    }
                }
                for (int i = 0; i < 6; i++)
                {
                    Jtr(i) = mtdPtr.data_host()->Jtr[i];
                }

                JtJ.diagonal() += Eigen::Vector<float, 6>::Constant(eta / (1 << l));

                auto decomp = JtJ.ldlt();
                // auto decomp = JtJ.block<3,3>(0,0).ldlt();

                Eigen::Matrix<float, 6, 1> deltax = decomp.solve(Jtr);
                // Eigen::Matrix<float, 3, 1> deltax = decomp.solve(Jtr.head<3>());

                // std::cout << "delta pose : " << deltax.transpose() << std::endl;
                if (deltax.norm() > 10.f)
                {
                    std::cerr << "Error : too large pose displacement, ignoring frame for pose estimation" << std::endl;
                    return;
                }
                else
                {
                    currentPose.position = currentPose.position + make_float3(deltax(0), deltax(1), deltax(2));

                    Eigen::Map<Eigen::Quaternionf> q((float *)&(currentPose.orientation));
                    q = q * Eigen::Quaternionf(1.f, 0.5f * deltax(3), 0.5f * deltax(4), 0.5f * deltax(5));
                    q.normalize();

                    if (deltax.norm() < 1e-5)
                    {
                        // std::cout << "deltapose norm < 1e-5" << std::endl;
                        break;
                    }
                }
            }
        }

        cameraPose = currentPose;

        // std::cout << "average optimizePoseGN_kernel time : " << total_time / nb_it << " ms" << std::endl;

        // CudaCheckError();
    }

    // JTJ_JTR_DATA GaussianSplattingSlam::optimizePoseGNCeres(int l, Pose3D &cameraPose)
    // {
    //     Eigen::Matrix<float, 6, 6> JtJ;
    //     Eigen::Vector<float, 6> Jtr;
    //     Pose3D currentPose = cameraPose;

    //     prepareRasterization(currentPose, cameraParams[l], pyrColor[l].cols, pyrColor[l].rows);

    //     cudaMemset(mtdPtr.data(),
    //                0,
    //                sizeof(MotionTrackingData));
    //     if (useFastPose)
    //     {
    //         optimizePoseGN3_fast_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
    //             mtdPtr.data(),
    //             thrust::raw_pointer_cast(ranges.data()),
    //             thrust::raw_pointer_cast(indices.data()),
    //             thrust::raw_pointer_cast(gaussians.imgPositions.data()),
    //             thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
    //             thrust::raw_pointer_cast(gaussians.pHats.data()),
    //             thrust::raw_pointer_cast(gaussians.colors.data()),
    //             thrust::raw_pointer_cast(gaussians.alphas.data()),
    //             pyrColorTex[l]->getTextureObject(),
    //             pyrDepthTex[l]->getTextureObject(),
    //             pyrDxTex[l]->getTextureObject(),
    //             pyrDyTex[l]->getTextureObject(),
    //             currentPose,
    //             cameraParams[l],
    //             bgColor,
    //             0.1f, // alpha threshold 0.2
    //             0.2f, // color threshold 0.1
    //             0.4f, // depth threshold 0.1
    //             numTiles,
    //             pyrColor[l].cols,
    //             pyrColor[l].rows);
    //     }
    //     else
    //     {
    //         optimizePoseGN3_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
    //             mtdPtr.data(),
    //             thrust::raw_pointer_cast(ranges.data()),
    //             thrust::raw_pointer_cast(indices.data()),
    //             thrust::raw_pointer_cast(gaussians.imgPositions.data()),
    //             thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
    //             thrust::raw_pointer_cast(gaussians.pHats.data()),
    //             thrust::raw_pointer_cast(gaussians.colors.data()),
    //             thrust::raw_pointer_cast(gaussians.alphas.data()),
    //             pyrColorTex[l]->getTextureObject(),
    //             pyrDepthTex[l]->getTextureObject(),
    //             pyrDxTex[l]->getTextureObject(),
    //             pyrDyTex[l]->getTextureObject(),
    //             currentPose,
    //             cameraParams[l],
    //             bgColor,
    //             0.1f, // alpha threshold 0.2
    //             0.2f, // color threshold 0.1
    //             0.4f, // depth threshold 0.1
    //             numTiles,
    //             pyrColor[l].cols,
    //             pyrColor[l].rows);
    //     }
    //     mtdPtr.download();

    //     int k = 0;
    //     for (int i = 0; i < 6; i++)
    //     {
    //         for (int j = i; j < 6; j++, k++)
    //         {
    //             JtJ(i, j) = JtJ(j, i) = mtdPtr.data_host()->JtJ[k];
    //         }
    //     }
    //     for (int i = 0; i < 6; i++)
    //     {
    //         Jtr(i) = -1 * mtdPtr.data_host()->Jtr[i];
    //     }

    //     cameraPose = currentPose;

    //     JTJ_JTR_DATA jtj_jtr;
    //     jtj_jtr.JTJ = JtJ;
    //     jtj_jtr.JTr = Jtr;

    //     return jtj_jtr;
    // }

    void GaussianSplattingSlam::optimizePoseGNCeres(Eigen::Matrix<double, 6, 6> &JtJ,
                                                    Eigen::Vector<double, 6> &Jtr,
                                                    int l,
                                                    const Eigen::Vector3d &P_imu,
                                                    const Eigen::Quaterniond &Q_imu)
    {
        // Eigen::Matrix<float, 6, 6> JtJ;
        // Eigen::Vector<float, 6> Jtr;

        Eigen::Vector3d P_cam = P_imu + Q_imu * t_imu_cam_;
        Eigen::Quaterniond Q_cam = Q_imu * q_imu_cam_;

        cudaMemset(mtdPtr.data(),
                   0,
                   sizeof(MotionTrackingData));

        if (poseEstimationMethod == PoseEstimationMethodWarpingSingleRendering)
        {
            Eigen::Vector3d P_warping(poseWarping.position.x,
                                      poseWarping.position.y,
                                      poseWarping.position.z);
            Eigen::Quaterniond Q_warping(poseWarping.orientation.w,
                                         poseWarping.orientation.x,
                                         poseWarping.orientation.y,
                                         poseWarping.orientation.z);
            Eigen::Vector3d P = Q_warping.conjugate() * (P_cam - P_warping);
            Eigen::Quaterniond Q = Q_warping.conjugate() * Q_cam;

            Pose3D warpPose;
            warpPose.position.x = P.x();
            warpPose.position.y = P.y();
            warpPose.position.z = P.z();
            warpPose.orientation.x = Q.x();
            warpPose.orientation.y = Q.y();
            warpPose.orientation.z = Q.z();
            warpPose.orientation.w = Q.w();

            optimizePoseGN_warping_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                mtdPtr.data(),
                pyrColorWarpingTex[l]->getTextureObject(),
                pyrDepthWarpingTex[l]->getTextureObject(),
                pyrColorTex[l]->getTextureObject(),
                pyrDepthTex[l]->getTextureObject(),
                pyrDxTex[l]->getTextureObject(),
                pyrDyTex[l]->getTextureObject(),
                warpPose,
                cameraParams[l],
                1.f,  // w_depth
                0.2f, // color threshold 0.1
                0.4f, // depth threshold 0.1
                pyrColor[l].cols,
                pyrColor[l].rows);
        }
        else
        {
            Pose3D currentPose;
            currentPose.position = make_float3(P_cam.x(), P_cam.y(), P_cam.z());
            Eigen::Map<Eigen::Quaternionf>((float *)&currentPose.orientation) = Q_cam.cast<float>();

            prepareRasterization(currentPose, cameraParams[l], pyrColor[l].cols, pyrColor[l].rows);

            if (poseEstimationMethod == PoseEstimationMethodWarpingMultipleRendering)
            {
                // std::cout << "OptimizePoseGN WarpingMultipleRendeging" << std::endl;
                optimizePoseGN3_fast_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                    mtdPtr.data(),
                    thrust::raw_pointer_cast(ranges.data()),
                    thrust::raw_pointer_cast(indices.data()),
                    thrust::raw_pointer_cast(gaussians.imgPositions.data()),
                    thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.pHats.data()),
                    thrust::raw_pointer_cast(gaussians.colors.data()),
                    thrust::raw_pointer_cast(gaussians.alphas.data()),
                    pyrColorTex[l]->getTextureObject(),
                    pyrDepthTex[l]->getTextureObject(),
                    pyrDxTex[l]->getTextureObject(),
                    pyrDyTex[l]->getTextureObject(),
                    currentPose,
                    cameraParams[l],
                    bgColor,
                    0.1f, // alpha threshold 0.2
                    0.2f, // color threshold 0.1
                    0.4f, // depth threshold 0.1
                    numTiles,
                    pyrColor[l].cols,
                    pyrColor[l].rows);
            }
            else
            {
                optimizePoseGN3_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                    mtdPtr.data(),
                    thrust::raw_pointer_cast(ranges.data()),
                    thrust::raw_pointer_cast(indices.data()),
                    thrust::raw_pointer_cast(gaussians.imgPositions.data()),
                    thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
                    thrust::raw_pointer_cast(gaussians.pHats.data()),
                    thrust::raw_pointer_cast(gaussians.colors.data()),
                    thrust::raw_pointer_cast(gaussians.alphas.data()),
                    pyrColorTex[l]->getTextureObject(),
                    pyrDepthTex[l]->getTextureObject(),
                    pyrDxTex[l]->getTextureObject(),
                    pyrDyTex[l]->getTextureObject(),
                    currentPose,
                    cameraParams[l],
                    bgColor,
                    0.1f, // alpha threshold 0.2
                    0.2f, // color threshold 0.1
                    0.4f, // depth threshold 0.1
                    numTiles,
                    pyrColor[l].cols,
                    pyrColor[l].rows);
            }
        }
        mtdPtr.download();

        int k = 0;
        for (int i = 0; i < 6; i++)
        {
            for (int j = i; j < 6; j++, k++)
            {
                JtJ(i, j) = JtJ(j, i) = mtdPtr.data_host()->JtJ[k];
            }
        }
        for (int i = 0; i < 6; i++)
        {
            Jtr(i) = -1 * mtdPtr.data_host()->Jtr[i];
        }

        if (poseEstimationMethod == PoseEstimationMethodWarpingSingleRendering)
        {
            Eigen::Quaterniond Q_warping(poseWarping.orientation.w,
                                         poseWarping.orientation.x,
                                         poseWarping.orientation.y,
                                         poseWarping.orientation.z);
            Eigen::Matrix3d R_warping = Q_warping.toRotationMatrix();

            Eigen::Matrix<double, 6, 6> J_warp_cam;
            J_warp_cam.block<3, 3>(0, 0) = R_warping.transpose();
            J_warp_cam.block<3, 3>(0, 3).setZero();
            J_warp_cam.block<3, 3>(3, 0).setZero();
            J_warp_cam.block<3, 3>(3, 3) = R_warping.transpose();
            JtJ = J_warp_cam.transpose() * JtJ * J_warp_cam;
            Jtr = J_warp_cam.transpose() * Jtr;
        }

        Eigen::Matrix3d R_imu_cam = q_imu_cam_.toRotationMatrix();
        Eigen::Matrix3d R_imu = Q_imu.toRotationMatrix();
        Eigen::Matrix3d P_imu_cam_skew{{0.0, -t_imu_cam_.z(), t_imu_cam_.y()},
                                       {t_imu_cam_.z(), 0.0, -t_imu_cam_.x()},
                                       {-t_imu_cam_.y(), t_imu_cam_.x(), 0.0}};
        Eigen::Matrix<double, 6, 6> J_cam_imu;
        J_cam_imu.block<3, 3>(0, 0).setIdentity();
        J_cam_imu.block<3, 3>(0, 3) = -R_imu * P_imu_cam_skew;
        J_cam_imu.block<3, 3>(3, 0).setZero();
        J_cam_imu.block<3, 3>(3, 3) = R_imu_cam.transpose();

        JtJ = J_cam_imu.transpose() * JtJ * J_cam_imu;
        Jtr = J_cam_imu.transpose() * Jtr;

        // JTJ_JTR_DATA jtj_jtr;
        // jtj_jtr.JTJ = JtJ;
        // jtj_jtr.JTr = Jtr;

        // return jtj_jtr;
    }

    void GaussianSplattingSlam::densify(GaussianSplattingKeyframe &keyframe)
    {
        prepareRasterization(keyframe.getCameraPose(), keyframe.getCameraParams(), keyframe.getImgWidth(), keyframe.getImgHeight());

        *instanceCounterPtr.data_host() = nbGaussians;
        instanceCounterPtr.upload();

        dim3 dimBlock(16, 16);
        dim3 dimGridGen(((rgbImgGpu.cols / gaussInitSizePx) + dimBlock.x - 1) / dimBlock.x,
                        ((rgbImgGpu.rows / gaussInitSizePx) + dimBlock.y - 1) / dimBlock.y);

        computeDensityMask_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float *)densityMask.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            // pyrColorTex[0]->gJetTextureObject(),
            // pyrDepthTex[0]->getTextureObject(),
            keyframe.getDepthTex().getTextureObject(),
            numTiles,
            densityMask.cols,
            densityMask.rows,
            densityMask.step);

        /* generate gaussians */

        densifyGaussians_kernel<<<dimGridGen, dimBlock>>>(
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            instanceCounterPtr.data(),
            keyframe.getColorTex().getTextureObject(),
            keyframe.getDepthTex().getTextureObject(),
            keyframe.getNormalTex().getTextureObject(),
            texMask->getTextureObject(),
            keyframe.getCameraPose(),
            keyframe.getCameraParams(),
            gaussInitSizePx,
            gaussInitSizePx,
            keyframe.getImgWidth(),
            keyframe.getImgHeight());

        instanceCounterPtr.download();
        nbGaussians = *instanceCounterPtr.data_host();

        std::cout << "nbGaussians : " << nbGaussians << std::endl;

        // CudaCheckError();
    }

    void GaussianSplattingSlam::splitGaussians(GaussianSplattingKeyframe &keyframe)
    {
        prepareRasterization(keyframe.getCameraPose(), keyframe.getCameraParams(), keyframe.getImgWidth(), keyframe.getImgHeight());

        // split gaussians
        *instanceCounterPtr.data_host() = nbGaussians;
        instanceCounterPtr.upload();

        dim3 dimBlock(16, 16);
        dim3 dimGridGen(((rgbImgGpu.cols / gaussInitSizePx) + dimBlock.x - 1) / dimBlock.x,
                        ((rgbImgGpu.rows / gaussInitSizePx) + dimBlock.y - 1) / dimBlock.y);

        PosGradVariance nullPosGradVar = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        thrust::fill(posGradVar.begin(), posGradVar.begin() + nbGaussians, nullPosGradVar);
        computePosGradVariance_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(posGradVar.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            keyframe.getColorTex().getTextureObject(),
            keyframe.getDepthTex().getTextureObject(),
            1.f,
            5.f,
            bgColor,
            numTiles,
            densityMask.cols,
            densityMask.rows);

        splitGaussians_kernel<<<(nbGaussians + GSS_BLOCK_SIZE - 1) / GSS_BLOCK_SIZE, GSS_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(posGradVar.data()),
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            instanceCounterPtr.data(),
            keyframe.getCameraPose(), keyframe.getCameraParams(),
            1e-5f,
            nbGaussians,
            nbGaussiansMax);

        instanceCounterPtr.download();
        nbGaussians = *instanceCounterPtr.data_host();

        std::cout << "nbGaussians : " << nbGaussians << std::endl;

        // CudaCheckError();
    }

    void GaussianSplattingSlam::removeOutliers()
    {
        prepareRasterization(cameraPose, cameraParams[0], pyrDepth[0].cols, pyrDepth[0].rows);
        thrust::fill(outlierProb.begin(), outlierProb.begin() + nbGaussians, 0.f);
        thrust::fill(totalAlpha.begin(), totalAlpha.begin() + nbGaussians, 0.f);

        computeOutliers_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(outlierProb.data()),
            thrust::raw_pointer_cast(totalAlpha.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            pyrDepthTex[0]->getTextureObject(),
            numTiles,
            pyrDepth[0].cols,
            pyrDepth[0].rows);

        *instanceCounterPtr.data_host() = 0;
        instanceCounterPtr.upload();

        removeOutliers_kernel<<<(nbGaussians + GSS_BLOCK_SIZE - 1) / GSS_BLOCK_SIZE, GSS_BLOCK_SIZE>>>(
            instanceCounterPtr.data(),
            thrust::raw_pointer_cast(states.data()),
            thrust::raw_pointer_cast(outlierProb.data()),
            thrust::raw_pointer_cast(totalAlpha.data()),
            0.6f,
            nbGaussians);

        auto it = thrust::make_zip_iterator(thrust::make_tuple(
            gaussians.positions.begin(),
            gaussians.scales.begin(),
            gaussians.orientations.begin(),
            gaussians.colors.begin(),
            gaussians.alphas.begin(),
            adamStates.begin()));
        thrust::sort_by_key(states.begin(), states.begin() + nbGaussians,
                            it);

        instanceCounterPtr.download();
        int nb_removed = *instanceCounterPtr.data_host();
        nbGaussians -= nb_removed;

        cudaMemset(&thrust::raw_pointer_cast(adamStates.data())[nbGaussians], 0, sizeof(AdamStateGaussian3D) * nb_removed);

        // std::cout << nb_removed << " outliers removed" << std::endl;
    }

    void GaussianSplattingSlam::prune()
    {
        *instanceCounterPtr.data_host() = 0;
        instanceCounterPtr.upload();

        // dim3 dimBlock(16, 16);
        // dim3 dimGridGen(((rgbImgGpu.cols / 6) + dimBlock.x - 1) / dimBlock.x,
        //                 ((rgbImgGpu.rows / 6) + dimBlock.y - 1) / dimBlock.y);

        pruneGaussians_kernel<<<(nbGaussians + GSS_BLOCK_SIZE - 1) / GSS_BLOCK_SIZE, GSS_BLOCK_SIZE>>>(
            instanceCounterPtr.data(),
            thrust::raw_pointer_cast(states.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            0.05f, // alpha threshold
            0.05f, // scale ratio
            nbGaussians);

        // thrust::sort_by_key(states.begin(), states.begin() + nbGaussians,
        //                     gaussians.begin());

        auto it = thrust::make_zip_iterator(thrust::make_tuple(
            gaussians.positions.begin(),
            gaussians.scales.begin(),
            gaussians.orientations.begin(),
            gaussians.colors.begin(),
            gaussians.alphas.begin(),
            adamStates.begin()));
        thrust::sort_by_key(states.begin(), states.begin() + nbGaussians,
                            it);

        instanceCounterPtr.download();
        int nb_removed = *instanceCounterPtr.data_host();
        nbGaussians -= nb_removed;

        cudaMemset(&thrust::raw_pointer_cast(adamStates.data())[nbGaussians], 0, sizeof(AdamStateGaussian3D) * nb_removed);
        // CudaCheckError();
    }

    void GaussianSplattingSlam::displayGrad()
    {
        rasterize();

        renderedRgbGpu.download(renderedRgb);
        // rgbaImgGpu.download(renderedRgb);
        renderedDepthGpu.download(renderedDepth);

        cudaMemset((void *)thrust::raw_pointer_cast(deltaGaussians.data()),
                   0,
                   nbGaussians * sizeof(DeltaGaussian2D));

        optimizePose_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(deltaGaussians.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgSigmas.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            pyrColorTex[0]->getTextureObject(),
            pyrDepthTex[0]->getTextureObject(),
            numTiles,
            pyrColor[0].cols,
            pyrColor[0].rows);

        for (int i = 0; i < nbGaussians; i++)
        {
            float3 p = gaussians.imgPositions[i];
            float4 q = gaussians.orientations[i];

            float siny_cosp = 2 * (q.w * q.z + q.x * q.y);
            float cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
            cv::line(renderedRgb, cv::Point(p.x, p.y), cv::Point(p.x + 10 * cosy_cosp, p.y + 10 * siny_cosp), cv::Scalar(1., 0., 0.));

            /*
            DeltaGaussian dg = deltaGaussians[i];
            float n = dg.n;
            if (n > 0.f)
            {
                cv::line(renderedRgb, cv::Point(p.x, p.y), cv::Point(p.x + 1.e5 * dg.meanImg.x / n, p.y + 1.e5 * dg.meanImg.y / n), cv::Scalar(1., 0., 0.));
            }
            */
        }

        cv::imshow("grad", renderedRgb);
    }

    void GaussianSplattingSlam::addKeyframe()
    {
        keyframes.emplace_back(pyrColor[0], pyrDepth[0], pyrNormal[0], cameraPose, cameraParams[0]);
        currentKeyframe = keyframes.size() - 1;
    }

    float GaussianSplattingSlam::computeCovisibilityRatio(GaussianSplattingKeyframe &keyframe)
    {
        *visUnionPtr.data_host() = 0;
        visUnionPtr.upload();
        *visInterPtr.data_host() = 0;
        visInterPtr.upload();

        thrust::fill(keyframeVis.begin(), keyframeVis.begin() + nbGaussians, 0);
        thrust::fill(frameVis.begin(), frameVis.begin() + nbGaussians, 0);

        prepareRasterization(keyframe.getCameraPose(), keyframe.getCameraParams(), keyframe.getImgWidth(), keyframe.getImgHeight());
        computeGaussiansVisibility_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(keyframeVis.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            numTiles,
            keyframe.getImgWidth(),
            keyframe.getImgHeight());

        prepareRasterization(cameraPose, cameraParams[0], pyrColor[0].cols, pyrColor[0].rows);
        computeGaussiansVisibility_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            thrust::raw_pointer_cast(frameVis.data()),
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            numTiles,
            pyrColor[0].cols,
            pyrColor[0].rows);

        computeGaussiansCovisibility_kernel<<<(nbGaussians + GSS_BLOCK_SIZE - 1) / GSS_BLOCK_SIZE, GSS_BLOCK_SIZE>>>(
            visInterPtr.data(),
            visUnionPtr.data(),
            thrust::raw_pointer_cast(keyframeVis.data()),
            thrust::raw_pointer_cast(frameVis.data()),
            nbGaussians);

        visInterPtr.download();
        visUnionPtr.download();

        // CudaCheckError();

        return *visInterPtr.data_host() / (float)*visUnionPtr.data_host();
    }

    void GaussianSplattingSlam::renderKeyframe(GaussianSplattingKeyframe &keyframe)
    {
        prepareRasterization(keyframe.getCameraPose(), keyframe.getCameraParams(), keyframe.getImgWidth(), keyframe.getImgHeight());

        rasterizeGaussians_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)renderedRgbGpu.data,
            (float *)renderedDepthGpu.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            bgColor,
            numTiles,
            renderedRgbGpu.cols,
            renderedRgbGpu.rows,
            renderedRgbGpu.step,
            renderedDepthGpu.step);
    }

    void GaussianSplattingSlam::exportPLY(const std::string &filename)
    {
        thrust::device_vector<float> buffer_dev(nbGaussians * 14);
        exportPLYGaussians_kernel<<<(nbGaussians + GSS_BLOCK_SIZE - 1) / GSS_BLOCK_SIZE, GSS_BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(buffer_dev.data()),
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            nbGaussians);
        thrust::host_vector<float> buffer_host = buffer_dev;

        // std::cout << "sizeof(Gaussian3D) = " << sizeof(Gaussian3D) << std::endl;
        std::ofstream of(filename);
        of << "ply" << std::endl
           << "format binary_little_endian 1.0" << std::endl
           << "element vertex " << nbGaussians << std::endl
           << "property float x" << std::endl
           << "property float y" << std::endl
           << "property float z" << std::endl
           << "property float scale_0" << std::endl
           << "property float scale_1" << std::endl
           << "property float scale_2" << std::endl
           << "property float rot_0" << std::endl
           << "property float rot_1" << std::endl
           << "property float rot_2" << std::endl
           << "property float rot_3" << std::endl
           << "property float f_dc_0" << std::endl
           << "property float f_dc_1" << std::endl
           << "property float f_dc_2" << std::endl
           << "property float opacity" << std::endl
           << "end_header" << std::endl;
        of.write((char *)buffer_host.data(), nbGaussians * sizeof(Gaussian3D));
        of.close();
    }

    void GaussianSplattingSlam::optimizationLoop()
    {
        while (!stopOptimization)
        {
            optimizationMutex.lock();
            optimizeGaussians(updateIterations, etaUpdate);
            optimizationMutex.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void GaussianSplattingSlam::displayLoop()
    {
        while (!stopDisplay)
        {
            display();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }

    void GaussianSplattingSlam::startDisplayLoop()
    {
        stopDisplay = false;
        displayThread = std::thread(&GaussianSplattingSlam::displayLoop, this);
    }

    void GaussianSplattingSlam::setAdamParameters(float eta, float beta1, float beta2, float epsilon)
    {
        adamEta = eta;
        adamBeta1 = beta1;
        adamBeta2 = beta2;
        adamEpsilon = epsilon;
    }

    void GaussianSplattingSlam::initWarping(const Pose3D &pose)
    {
        poseWarping = pose;

        // std::cout << "initWarping, pose = "
        //           << pose.position.x << " "
        //           << pose.position.y << " "
        //           << pose.position.z << std::endl;
        for (int l = 0; l < nbPyrLevels; l++)
        {
            prepareRasterization(poseWarping, cameraParams[l], pyrColorWarping[l].cols, pyrColorWarping[l].rows);

            rasterizeGaussians_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
                (uchar4 *)pyrColorWarping[l].data,
                (float *)pyrDepthWarping[l].data,
                thrust::raw_pointer_cast(ranges.data()),
                thrust::raw_pointer_cast(indices.data()),
                thrust::raw_pointer_cast(gaussians.imgPositions.data()),
                thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
                thrust::raw_pointer_cast(gaussians.pHats.data()),
                thrust::raw_pointer_cast(gaussians.colors.data()),
                thrust::raw_pointer_cast(gaussians.alphas.data()),
                bgColor,
                numTiles,
                pyrColorWarping[l].cols,
                pyrColorWarping[l].rows,
                pyrColorWarping[l].step,
                pyrDepthWarping[l].step);
        }
        // for (int i = 1; i < nbPyrLevels; i++)
        // {
        //     cv::cuda::pyrDown(pyrColor[i - 1], pyrColor[i]);
        //     cv::cuda::pyrDown(pyrDepth[i - 1], pyrDepth[i]);
        // }
    }

    void GaussianSplattingSlam::render3dView(cv::cuda::GpuMat &renderedRGB_,
                                             cv::cuda::GpuMat &renderedDepth_,
                                             const Pose3D &renderedCameraPose_,
                                             const CameraParameters &renderedCameraParams_,
                                             int width_,
                                             int height_)
    {
        renderedRGB_.create(height_, width_, CV_8UC4);
        renderedDepth_.create(height_, width_, CV_32FC1);

        if (nbGaussians == 0)
        {
            renderedRGB_.setTo(cv::Scalar(0, 0, 0, 0));
            renderedDepth_.setTo(0.f);
            return;
        }

        optimizationMutex.lock();

        prepareRasterization(renderedCameraPose_, renderedCameraParams_, width_, height_);

        rasterizeGaussians_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (uchar4 *)renderedRGB_.data,
            (float *)renderedDepth_.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.pHats.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            bgColor,
            numTiles,
            width_,
            height_,
            renderedRGB_.step,
            renderedDepth_.step);

        optimizationMutex.unlock();

        //    renderedViewGpu.download(renderedView);
        // cv::viz::imshow("3d view", renderedView);
        // std::cout << "showImage" << std::endl;
        // cv::imshow("3d view", renderedView);
    }

    void GaussianSplattingSlam::render3dViewBlobs(cv::cuda::GpuMat &renderedRGB_,
                                                  const Pose3D &renderedCameraPose_,
                                                  const CameraParameters &renderedCameraParams_,
                                                  int width_,
                                                  int height_)
    {
        renderedRGB_.create(height_, width_, CV_32FC3);

        if (nbGaussians == 0)
        {
            renderedRGB_.setTo(cv::Scalar(0, 0, 0, 0));
            return;
        }
        float3 lightDirection = normalize(make_float3(-0.3, 1., 1.f));

        optimizationMutex.lock();

        prepareRasterization(renderedCameraPose_, renderedCameraParams_, width_, height_);

        rasterizeGaussiansBlobs_kernel<<<dim3(numTiles.x, numTiles.y), dim3(tileSize.x, tileSize.y)>>>(
            (float3 *)renderedRGB_.data,
            thrust::raw_pointer_cast(ranges.data()),
            thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(gaussians.imgPositions.data()),
            thrust::raw_pointer_cast(gaussians.imgInvSigmas.data()),
            thrust::raw_pointer_cast(gaussians.colors.data()),
            thrust::raw_pointer_cast(gaussians.alphas.data()),
            thrust::raw_pointer_cast(gaussians.positions.data()),
            thrust::raw_pointer_cast(gaussians.scales.data()),
            thrust::raw_pointer_cast(gaussians.orientations.data()),
            renderedCameraPose_,
            renderedCameraParams_,
            lightDirection,
            numTiles,
            width_,
            height_,
            renderedRGB_.step);

        optimizationMutex.unlock();

        //    renderedViewGpu.download(renderedView);
        // cv::viz::imshow("3d view", renderedView);
        // std::cout << "showImage" << std::endl;
        // cv::imshow("3d view", renderedView);
    }

    void GaussianSplattingSlam::testOpt()
    {
        nbPyrLevels = 1;
        nbGaussians = 1;

        gaussians.positions[0] = make_float3(0.f, 0.f, 0.f);

        // gaussians.scales[0] = make_float3(0.15f, 0.04f, 0.01f);
        // gaussians.orientations[0] = make_float4(0.5, -0.5, 0.5, -0.5);
        gaussians.scales[0] = make_float3(0.01f, 0.15f, 0.04f);
        gaussians.orientations[0] = make_float4(0.f, 0.f, 0.f, 1.f);

        // gaussians.orientations[0] = make_float4(0.f, 0.f, 0.8660254, 0.5);
        // gaussians.orientations[0] = make_float4(0.f, 0.3826834, 0., 0.9238795);

        // gaussians.orientations[0] = make_float4(0, 0.0871557, 0, 0.9961947);
        // gaussians.orientations[0] = make_float4(0, 0.0436194, 0, 0.9990482 );
        /// gaussians.orientations[0] = make_float4(0, 0, 0.3826834, 0.9238795);
        // gaussians.orientations[0] = make_float4(0, 0, 0.7071068, 0.7071068);
        gaussians.colors[0] = make_float3(1.f, 0.1f, 0.1f);
        gaussians.alphas[0] = 0.9f;

        gaussians.positions[1] = make_float3(-0.6f, 0.f, 0.f);
        gaussians.scales[1] = make_float3(0.3f, 0.1f, 0.01f);
        gaussians.orientations[1] = make_float4(0.3826834, 0., 0., 0.9238795);
        gaussians.colors[1] = make_float3(1.f, 1.f, 0.1f);
        gaussians.alphas[1] = 0.9f;

        /*gaussians.positions[1] = make_float3(0.1f, 0.2f, 0.95f);
        gaussians.scales[1] = make_float3(0.08f, 0.12f, 0.007f);
        gaussians.orientations[1] = make_float4(0.f, 0.f, 0.f, 1.f);
        gaussians.colors[1] = make_float3(0.1f, 1.f, 0.1f);
        gaussians.alphas[1] = 0.6f;*/

        gaussians.positions[2] = make_float3(-0.6f, -0.4f, 0.2f);
        gaussians.scales[2] = make_float3(0.12f, 0.12f, 0.005f);
        gaussians.orientations[2] = make_float4(0.f, 0.f, 0.f, 1.f);
        gaussians.colors[2] = make_float3(0.5f, 0.5f, 1.f);
        gaussians.alphas[2] = 0.7f;

        gaussians.positions[3] = make_float3(0.6f, -0.5f, -0.2f);
        gaussians.scales[3] = make_float3(0.1f, 0.1f, 0.005f);
        gaussians.orientations[3] = make_float4(0.f, 0.f, 0.f, 1.f);
        gaussians.colors[3] = make_float3(0.05f, 0.05f, 0.3f);
        gaussians.alphas[3] = 0.95f;

        gaussians.positions[4] = make_float3(0.6f, 0.5f, 0.1f);
        gaussians.scales[4] = make_float3(0.1f, 0.1f, 0.005f);
        gaussians.orientations[4] = make_float4(0.f, 0.f, 0.f, 1.f);
        gaussians.colors[4] = make_float3(0.05f, 0.3f, 0.05f);
        gaussians.alphas[4] = 0.95f;

        gaussians.positions[5] = make_float3(-0.5f, 0.5f, 0.5f);
        gaussians.scales[5] = make_float3(0.1f, 0.1f, 0.005f);
        gaussians.orientations[5] = make_float4(0.f, 0.f, 0.f, 1.f);
        gaussians.colors[5] = make_float3(0.3f, 0.05f, 0.05f);
        gaussians.alphas[5] = 0.95f;

        gaussians.positions[6] = make_float3(0.6f, 0.f, 0.f);
        // gaussians.scales[6] = make_float3(0.2f, 0.1f, 0.01f);
        // gaussians.orientations[6] = make_float4(0.f, 0.3826834, 0, 0.9238795);
        // gaussians.colors[6] = make_float3(0.8f, 0.05f, 0.05f);
        // gaussians.alphas[6] = 0.95f;

        cameraPose = {{-1., 0., 0.},
                      {0.5, -0.5, 0.5, -0.5}};
        // cameraPose = {{-1., 0., 0.},
        //               {0, 0.7071068, 0, 0.7071068}};

        //{0., 0., 0., 1.}};
        //{0, 0, 0.6816388, 0.7316889 }};
        //{0., 0.1736482, 0., 0.9848078}};

        CameraParameters cameraParameters = {{400., 400.},
                                             {320., 240.}};

        setCameraParameters(cameraParameters);

        cv::Mat rgbImg(480, 640, CV_8UC3);
        cv::Mat depthImg(480, 640, CV_32FC1);

        initAndCopyImgs(rgbImg, depthImg);
        rasterize();

        renderedRgbGpu.convertTo(rgbImgGpu, CV_8UC3, 255.);
        renderedDepthGpu.copyTo(depthImgGpu);

        rgbImgGpu.download(rgbImg);
        depthImgGpu.download(depthImg);
        initAndCopyImgs(rgbImg, depthImg);

        /*cv::cuda::cvtColor(rgbImgGpu, rgbaImgGpu, cv::COLOR_BGR2BGRA, 4);

        texRGBA.reset(new Texture<uchar4>(rgbaImgGpu));
        texDepth.reset(new Texture<float>(depthImgGpu));
        */

        // cv::imshow("rgb", rgbImg);
        // cv::imshow("depth", 0.15f * depthImg);

        // renderedRgbGpu.download(renderedRgb);

        cv::Mat initialRGB, initialDepth;
        renderedRgbGpu.download(initialRGB);
        renderedDepthGpu.download(initialDepth);
        cv::imshow("initial_rgb", initialRGB);
        cv::imshow("initial_depth", 0.5f * initialDepth);

#define TEST_GAUSS_OPT
#ifdef TEST_GAUSS_OPT

        {
            std::cout << "real state :" << std::endl;
            float3 mean = gaussians.positions[0];
            float3 scale = gaussians.scales[0];
            float4 orientation = gaussians.orientations[0];
            float3 color = gaussians.colors[0];
            float alpha = gaussians.alphas[0];

            std::cout << "  mean : "
                      << mean.x << " "
                      << mean.y << " "
                      << mean.z << std::endl;
            std::cout << "  scale : "
                      << scale.x << " "
                      << scale.y << " "
                      << scale.z << std::endl;
            std::cout << "  orientation : "
                      << orientation.x << " "
                      << orientation.y << " "
                      << orientation.z << " "
                      << orientation.w << std::endl;
            std::cout << "  color : "
                      << color.x << " "
                      << color.y << " "
                      << color.z << std::endl;
            std::cout << "  alpha : "
                      << alpha << std::endl;
        }

        // gaussians.positions[0] += make_float3(0.f, 0.0f, 0.1f);
        // gaussians.positions[0] = make_float3(0.6f, 0.f, 0.1f);
        gaussians.scales[0] += make_float3(0.02f, 0.02f, 0.0f);
        // gaussians.orientations[0] = make_float4(0.f, 0.f, 0.819152, 0.5735764);
        // gaussians.orientations[0] = make_float4(0.f, 0.f, 0.f, 1.f);
        // gaussians.colors[0] = make_float3(0.5f, 0.5f, 0.5f);
        // gaussians.alphas[0] = 0.8f;

        float4 q0 = gaussians.orientations[0];
        Eigen::Quaternionf q(q0.w,
                             q0.x,
                             q0.y,
                             q0.z);
        q = Eigen::Quaternionf(1., 0.2, 0., 0.) * q;
        q.normalize();
        gaussians.orientations[0] = make_float4(q.x(), q.y(), q.z(), q.w());

        {
            std::cout << "init state :" << std::endl;
            float3 mean = gaussians.positions[0];
            float3 scale = gaussians.scales[0];
            float4 orientation = gaussians.orientations[0];
            float3 color = gaussians.colors[0];
            float alpha = gaussians.alphas[0];

            std::cout << "  mean : "
                      << mean.x << " "
                      << mean.y << " "
                      << mean.z << std::endl;
            std::cout << "  scale : "
                      << scale.x << " "
                      << scale.y << " "
                      << scale.z << std::endl;
            std::cout << "  orientation : "
                      << orientation.x << " "
                      << orientation.y << " "
                      << orientation.z << " "
                      << orientation.w << std::endl;
            std::cout << "  color : "
                      << color.x << " "
                      << color.y << " "
                      << color.z << std::endl;
            std::cout << "  alpha : "
                      << alpha << std::endl;
        }

        for (int k = 0; k < 10; k++)
        {
            optimizeGaussians(10, 1e-6);
            float3 mean = gaussians.positions[0];
            float3 scale = gaussians.scales[0];
            float4 orientation = gaussians.orientations[0];
            float3 color = gaussians.colors[0];
            float alpha = gaussians.alphas[0];

            std::cout << "mean : "
                      << mean.x << " "
                      << mean.y << " "
                      << mean.z << std::endl;
            std::cout << "scale : "
                      << scale.x << " "
                      << scale.y << " "
                      << scale.z << std::endl;
            std::cout << "orientation : "
                      << orientation.x << " "
                      << orientation.y << " "
                      << orientation.z << " "
                      << orientation.w << std::endl;
            std::cout << "color : "
                      << color.x << " "
                      << color.y << " "
                      << color.z << std::endl;
            std::cout << "alpha : "
                      << alpha << std::endl;
        }
#endif

// #define TEST_POSE_OPT
#ifdef TEST_POSE_OPT
        cameraPose.position += make_float3(0.0f, 0.f, -0.1f);
        // cameraPose.orientation = {0, 0, 0.6051864, 0.7960838};
        // cameraPose.orientation = {-0.0460361, 0.0772334, 0.0460361, 0.9948851};
        Eigen::Quaternionf q0(1.f, 0.0f, 0.f, 0.1f);
        q0.normalize();

        cameraPose.orientation = make_float4(q0.x(), q0.y(), q0.z(), q0.w());

        for (int k = 0; k < 1; k++)
        {
            optimizePoseGN(20, 0.1);
            std::cout << "position : "
                      << cameraPose.position.x << " "
                      << cameraPose.position.y << " "
                      << cameraPose.position.z << std::endl;
            std::cout << "orientation : "
                      << cameraPose.orientation.x << " "
                      << cameraPose.orientation.y << " "
                      << cameraPose.orientation.z << " "
                      << cameraPose.orientation.w << std::endl;
        }

#endif

        // cv::imshow("rendered_rgb", renderedRgb);
        // cv::imshow("rendered_depth", 0.15f * renderedDepth);

        rasterize();
        rasterizeNormals();
        renderedRgbGpu.download(renderedRgb);
        renderedDepthGpu.download(renderedDepth);
        renderedNormalsGpu.download(renderedNormals);
        cv::imshow("rgb", renderedRgb);
        cv::imshow("depth", 0.5f * renderedDepth);
        cv::imshow("normals", 0.5f + 0.5f * renderedNormals);
        int i = 1;
        for (;;)
        {
            if (cv::waitKey(5) == ' ')
            {
                i = (i + 1) % 2;
                if (i == 0)
                {
                    std::cout << "display reference image" << std::endl;
                    cv::imshow("rgb", initialRGB);
                    cv::imshow("depth", 0.5f * initialDepth);
                }
                else
                {
                    std::cout << "display final image" << std::endl;
                    cv::imshow("rgb", renderedRgb);
                    cv::imshow("depth", 0.5f * renderedDepth);
                }
            }
        }
    }
}
