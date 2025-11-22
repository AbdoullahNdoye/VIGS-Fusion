#pragma once

#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/subscriber.h>
#include <message_filters/cache.h>
#include <sensor_msgs/msg/camera_info.h>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <image_transport/image_transport.hpp>
#include <image_transport/subscriber_filter.hpp>

#include <gaussian_splatting_slam/GaussianSplattingSlam.hpp>
#include <gaussian_splatting_slam/Preintegration.hpp>
#include <gaussian_splatting_slam/Utility.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_listener.h>
#include "tf2_ros/transform_broadcaster.h"
#include <tf2_ros/buffer.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_eigen/tf2_eigen.hpp"

#include <gaussian_splatting_slam/GaussianSplattingViewer.hpp>

#include <memory>

namespace gaussian_splatting_slam
{

    class GaussianSplattingSlamNode : public rclcpp::Node
    {
    public:
        GaussianSplattingSlamNode();
        ~GaussianSplattingSlamNode();

        void RGBDCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg_rgb, const sensor_msgs::msg::Image::ConstSharedPtr msg_depth);
        void camInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr info);
        void IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg_imu);
        void sync_rgbd_imu();

    protected:
        GaussianSplattingSlam gss;

        std::shared_ptr<GaussianSplattingViewer> viewer;
        
        gaussian_splatting_slam::Preintegration preint;

        image_transport::SubscriberFilter rgbSub, depthSub;
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image, sensor_msgs::msg::Image> RGBDSyncPolicy;
        message_filters::Synchronizer<RGBDSyncPolicy> sync;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camInfoSub;

        message_filters::Subscriber<sensor_msgs::msg::Imu> imuSub;
        message_filters::Cache<sensor_msgs::msg::Imu> imuCache;
        message_filters::Cache<sensor_msgs::msg::Imu> IMUCache_preint;

        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_preintSub;

        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odomPub, odomImuPub;

        std::string worldFrameId;
        nav_msgs::msg::Odometry odomMsg, odomImuMsg;

        Pose3D odomPose = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
        Pose3D odomPose_init = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 1.f}};
        rclcpp::Time lastStampIMU{0, 0, RCL_ROS_TIME};
        rclcpp::Time lastProcessedFrameStamp{0, 0, RCL_ROS_TIME};
        rclcpp::Time lastRGBTime{0, 0, RCL_ROS_TIME};
        rclcpp::Time lastDEPTHTime{0, 0, RCL_ROS_TIME};
        sensor_msgs::msg::Image::ConstSharedPtr last_rgb_msg;
        int64_t depth_last_time, rgb_last_time;
        

        bool downsample, imu_initialized = false;
        cv::Mat rgbDownsampleImg, depthDownsampleImg;
        cv::Mat localDepthImg, localRgbImg;

        cv::Mat depthImg, rgbImg;
        bool isProcessing = false, firstImu = false;
        bool processIMU = false;

        ImuData imud;
        std::string imuFrameId;
        int nb_init_imu=0;
        Eigen::Vector3d avg_acc;
        Eigen::Vector3d acc_bias, avg_gyro;
        Eigen::Quaterniond q_imu_cam;
        Eigen::Vector3d t_imu_cam;

        cv::Rect roi;
        bool hasCameraInfo;

        std::shared_ptr<tf2_ros::TransformListener> tfListener{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tfBuffer;
        std::unique_ptr<tf2_ros::TransformBroadcaster> br;

        int frequence_;

    }; // class GaussianSplattingSlamNode

} // namespace gaussian_splatting_slam
