#pragma once

#include <gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp>
#include <gaussian_splatting_slam/GSPoseCost.hpp>
#include <gaussian_splatting_slam/texture.hpp>
#include <gaussian_splatting_slam/GaussianSplattingKeyframe.hpp>
#include <gaussian_splatting_slam/Preintegration.hpp>
#include <gaussian_splatting_slam/ImuCostFunction.hpp>
#include <gaussian_splatting_slam/MarginalizationFactor.hpp>
#include <gaussian_splatting_slam/BetaBinomialGenerator.hpp>
#include <opencv2/core/cuda.hpp>
#include <cuda_utils/cudasharedptr.h>
#include <cuda_utils/CachedAllocator.h>
#include <opencv2/cudafilters.hpp>
#include "ceres/ceres.h"
#include <gaussian_splatting_slam/IMU_data.hpp>
#include <opencv2/viz.hpp>

#include <memory>
#include <vector>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thread>
#include <mutex>

namespace gaussian_splatting_slam
{
    class GaussianSplattingSlam
    {
    public:
        GaussianSplattingSlam();
        ~GaussianSplattingSlam();

        void setCameraParameters(const CameraParameters &params);

        // void compute();

        void compute(const cv::Mat &rgbImg, const cv::Mat &depthImg, const Pose3D &odomPose);

        void initAndCopyImgs(const cv::Mat &rgbImg, const cv::Mat &depthImg);

        void generateGaussians();

        // void prepareRasterization(int level = 0);
        void prepareRasterization(const Pose3D &cameraPose,
                                  const CameraParameters &cameraParams,
                                  int width,
                                  int height);

        void rasterize();
        void rasterizeFill(cv::cuda::GpuMat &img);
        void rasterizeNormals();

        void rasterizeBlobs(cv::cuda::GpuMat &img);

        void rasterizeError(GaussianSplattingKeyframe &keyframe);

        void optimizeGaussians(int nbIterations, float eta = 0.2);
        void optimizeGaussiansKeyframe(GaussianSplattingKeyframe &keyframe, float eta = 0.2);
        void optimizeGaussiansKeyframe2(GaussianSplattingKeyframe &keyframe, float eta = 0.2);

        void optimizePose(int nbIterations, float eta = 0.5, int minLevel = 0);
        void optimizePoseGN(int nbIterations, float eta = 0.5, int minLevel = 0);

        // JTJ_JTR_DATA optimizePoseGNCeres(int l, Pose3D &cameraPose);
        void optimizePoseGNCeres(
            Eigen::Matrix<double, 6, 6> &JtJ,
            Eigen::Vector<double, 6> &Jtr,
            int l,
            const Eigen::Vector3d &P_imu,
            const Eigen::Quaterniond &Q_imu);

        void setPose(const Pose3D &Pose);
        void initialize(const Eigen::Vector3d t_imu_cam, const Eigen::Quaterniond q_imu_cam, const Pose3D &Pose, const Pose3D &PoseCam);

        void setImuBias(const Eigen::Vector3d &b_a,
                        const Eigen::Vector3d &b_g);
        void processIMU(const double t, const ImuData &imud);
        inline void getBiases(Eigen::Vector3d &ba,
                              Eigen::Vector3d &bg) const
        {
            ba = Eigen::Map<const Eigen::Vector3d>(VB_cur + 3);
            bg = Eigen::Map<const Eigen::Vector3d>(VB_cur + 6);
        }
        inline bool isInitialized() { return initialized; }

        void densify(GaussianSplattingKeyframe &keyframe);
        void splitGaussians(GaussianSplattingKeyframe &keyframe);
        void prune();

        void removeOutliers();

        void displayGrad();

        void addKeyframe();

        inline const Pose3D &getCameraPose() { return cameraPose; }
        inline const double *getImuPose() { return P_cur; }
        inline const double *getImuVelocity() { return VB_cur; }

        void testOpt();

        void computeNormals();

        inline void setPoseIterations(int it) { poseIterations = it; }
        inline void setUpdateIterations(int it) { updateIterations = it; }
        inline void setEtaPose(float eta) { etaPose = eta; }
        inline void setEtaUpdate(float eta) { etaUpdate = eta; }
        inline void setGaussInitSizePx(int s) { gaussInitSizePx = s; }
        inline void setWDepth(float WDepth) { w_depth = WDepth; }
        inline void setWDist(float WDist) { w_dist = WDist; }
        inline void setOptimizer(std::string t) { optimizer = t; }

        inline void setPoseEstimationMethod(PoseEstimationMethod method) { poseEstimationMethod = method; }

        inline void getState(Eigen::Vector3d &pos,
                             Eigen::Quaterniond &rot,
                             Eigen::Vector3d &vel) const
        {
            pos = Eigen::Map<const Eigen::Vector3d>(P_cur);
            rot = Eigen::Map<const Eigen::Quaterniond>(P_cur + 3);
            vel = Eigen::Map<const Eigen::Vector3d>(VB_cur);
        }

        void setNbPyrLevels(int n);

        float computeCovisibilityRatio(GaussianSplattingKeyframe &keyframe);

        void display();

        void renderKeyframe(GaussianSplattingKeyframe &keyframe);

        void exportPLY(const std::string &filename);

        void optimizationLoop();

        void displayLoop();
        void startDisplayLoop();

        void setAdamParameters(float eta, float beta1, float beta2, float epsilon);
        inline void setCovisibilityThreshold(float t) { covisibilityThreshold = t; }

        void render3dView(cv::cuda::GpuMat &renderedRGB_,
                          cv::cuda::GpuMat &renderedDepth_,
                          const Pose3D &renderedCameraPose_,
                          const CameraParameters &renderedCameraParams_,
                          int width_,
                          int height_);

        void render3dViewBlobs(cv::cuda::GpuMat &renderedRGB_,
                          const Pose3D &renderedCameraPose_,
                          const CameraParameters &renderedCameraParams_,
                          int width_,
                          int height_);
        

    protected:
        void initWarping(const Pose3D &pose);

        GSPoseCostFunction *GSPoseCost;
        ImuCostFunction *imuCost;

        double P_cur[7], P_prev[7];
        double VB_cur[9], VB_prev[9];

        MarginalizationInfo marginalization_info;
        MarginalizationFactor *marginalizationCost;

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        ceres::Problem problem;

        std::shared_ptr<Preintegration> preint;
        ImuData last_imu;
        double last_imu_time;

        int nbPyrLevels;
        int poseIterations, updateIterations;
        float etaPose, etaUpdate;
        int gaussInitSizePx;

        cv::cuda::GpuMat rgbImgGpu, rgbaImgGpu, depthImgGpu;
        cv::cuda::GpuMat renderedRgbGpu, renderedDepthGpu, renderedBlobsGpu, renderedNormalsGpu; //renderedViewGpu, renderedViewDepthGpu
        cv::cuda::GpuMat renderedErrorGpu;

        cv::cuda::GpuMat densityMask;

        cv::Mat rgbImg, depthImg, renderedRgb, renderedDepth, maskImg, renderedBlobs, renderedNormals, computedNormals; // renderedView, 
        cv::Mat renderedError;
        cv::Mat imgDx, imgDy;

        cuda_utils::CachedAllocator allocator;

        std::shared_ptr<Texture<float>> texDepth, texMask;
        std::shared_ptr<Texture<uchar4>> texRGBA;

        std::vector<cv::cuda::GpuMat> pyrColor, pyrDepth, pyrNormal, pyrDx, pyrDy;
        std::vector<std::shared_ptr<Texture<float>>> pyrDepthTex;
        std::vector<std::shared_ptr<Texture<uchar4>>> pyrColorTex;
        std::vector<std::shared_ptr<Texture<float4>>> pyrNormalTex, pyrDxTex, pyrDyTex;

        std::vector<cv::cuda::GpuMat> pyrColorWarping, pyrDepthWarping;
        std::vector<std::shared_ptr<Texture<float>>> pyrDepthWarpingTex;
        std::vector<std::shared_ptr<Texture<uchar4>>> pyrColorWarpingTex;

        cv::Ptr<cv::cuda::Filter> dxFilter, dyFilter;

        std::vector<CameraParameters> cameraParams;
        Pose3D cameraPose = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
        Pose3D InitialPose = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
        Pose3D InitialPoseCam = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
        Pose3D prevOdomPose = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
        Eigen::Quaterniond q_imu_cam_;
        Eigen::Vector3d t_imu_cam_;

        Pose3D poseWarping = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};

        float3 bgColor;

        std::vector<GaussianSplattingKeyframe> keyframes;

        int currentKeyframe;

        Gaussians gaussians;

        thrust::device_vector<uint32_t> indices;
        thrust::device_vector<uint64_t> hashes;
        thrust::device_vector<uint2> ranges;
        thrust::device_vector<uint8_t> states;
        thrust::device_vector<DeltaGaussian2D> deltaGaussians;
        thrust::device_vector<DeltaGaussian3D> deltaGaussian3Ds;
        thrust::device_vector<AdamStateGaussian3D> adamStates;

        thrust::device_vector<uint32_t> perTileBuckets;
        thrust::device_vector<uint32_t> bucketToTile;
        thrust::device_vector<float> sampled_T;
        thrust::device_vector<float3> sampled_ar;
        thrust::device_vector<float> final_T;
        thrust::device_vector<uint32_t> n_contrib;
        thrust::device_vector<uint32_t> max_contrib;
        thrust::device_vector<float3> output_color;
        thrust::device_vector<float> output_depth;
        thrust::device_vector<float3> color_error;
        thrust::device_vector<float> depth_error;

        thrust::device_vector<float> outlierProb;
        thrust::device_vector<float> totalAlpha;

        fun::cuda::shared_ptr<uint32_t> instanceCounterPtr;

        fun::cuda::shared_ptr<DeltaPose3D> deltaPosePtr;
        fun::cuda::shared_ptr<MotionTrackingData> mtdPtr;

        thrust::device_vector<uint8_t> keyframeVis, frameVis;
        fun::cuda::shared_ptr<uint32_t> visUnionPtr, visInterPtr;

        thrust::device_vector<PosGradVariance> posGradVar;

        uint2 tileSize, numTiles;

        unsigned int iterationsSinceDensification;

        unsigned int nbGaussians = 0;
        unsigned int nbGaussiansMax;

        bool firstImage, initialized;

        unsigned int nbImagesProcessed = 0;

        float adamEta = 1e-3;
        float adamBeta1 = 0.9f;
        float adamBeta2 = 0.999f;
        float adamEpsilon = 1.e-8;

        PoseEstimationMethod poseEstimationMethod = PoseEstimationMethodFull;

        float covisibilityThreshold = 0.95;

        float w_depth;
        float w_dist;

        std::thread optimizeThread, displayThread;
        bool stopOptimization = false;
        bool stopDisplay = false;
        std::mutex optimizationMutex;
        BetaBinomialGenerator betaBinomialGenerator;
        std::string optimizer = "adam";

        // Pose3D viewPose = {{0.f, 0.f, 1.f}, {-0.5f, 0.5f, -0.5f, 0.5f}};
        // CameraParameters viewCameraParams;

    }; // class GaussianSplattingSlam
} // namespace gaussian_splatting_slam
