#pragma once
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <gaussian_splatting_slam/GaussianSplattingSlamTypes.hpp>
#include <gaussian_splatting_slam/GaussianSplattingSlam.hpp>
#include <Eigen/Dense>

#include <thread>

namespace gaussian_splatting_slam
{
    class GaussianSplattingViewer
    {

    public:
        GaussianSplattingViewer(GaussianSplattingSlam &gsslam_);

        ~GaussianSplattingViewer();

        void startThread();

    protected:
        static void mouseCallbackStatic(int event, int x, int y, int flags, void *userdata);        
        void mouseCallback(int event, int x, int y, int flags);
        void keyCallback(int key);
        void renderLoop();
        void render();
        void resetView();

        GaussianSplattingSlam& gsslam;

        cv::Mat renderedRgb, renderedDepth;
        cv::cuda::GpuMat renderedRgbGpu, renderedDepthGpu;

        CameraParameters cameraParams;
        Pose3D cameraPose;
        int width, height;
        double fov;

        bool follow = true;

        Eigen::Vector3f cameraViewPosition;
        double yaw, pitch;
        Eigen::Vector3f focalPoint;
        double distance;

        int prevMouseX, prevMouseY;
        
        std::thread renderThread;

        enum {
            RENDER_TYPE_RGB = 0,
            RENDER_TYPE_DEPTH = 1,
            RENDER_TYPE_BLOBS = 2,
            RENDER_TYPE_NUM
        } renderType = RENDER_TYPE_RGB;

    }; // class GaussianSplattingViewer
} // namespace gaussian_splatting_slam