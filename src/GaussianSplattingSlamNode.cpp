#include <gaussian_splatting_slam/GaussianSplattingSlamNode.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <limits>

namespace gaussian_splatting_slam
{
    GaussianSplattingSlamNode::GaussianSplattingSlamNode()
        : Node("gaussian_splatting_slam"),
          imuSub(this, "imu"),
          IMUCache_preint(1000),
          sync(RGBDSyncPolicy(2), rgbSub, depthSub),
          hasCameraInfo(false),
          nb_init_imu(0),
          avg_acc(Eigen::Vector3d::Zero()),
          avg_gyro(Eigen::Vector3d::Zero())
    {
        tfBuffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        rmw_qos_profile_t qos_profile = rmw_qos_profile_default;
        qos_profile.depth = 1;

        this->declare_parameter("world_frame_id", "world");
        worldFrameId = this->get_parameter("world_frame_id").as_string();
        this->declare_parameter("optimizer", "adam");

        this->declare_parameter("pose_iterations", 4);
        this->declare_parameter("update_iterations", 10);
        this->declare_parameter("eta_pose", 0.01);
        this->declare_parameter("eta_update", 0.002);
        this->declare_parameter("gauss_init_size_px", 7);
        this->declare_parameter("nb_pyr_levels", 3);
        this->declare_parameter("downsample", false);
        this->declare_parameter("w_depth", 1.0);
        this->declare_parameter("w_dist", 0.1);
        this->declare_parameter("frequence", 30);

        this->declare_parameter<double>("acc_n", 0.1);
        this->declare_parameter<double>("gyr_n", 0.01);
        this->declare_parameter<double>("acc_w", 0.001);
        this->declare_parameter<double>("gyr_w", 0.0001);

        this->get_parameter<double>("acc_n", imud.acc_n);
        this->get_parameter<double>("gyr_n", imud.gyr_n);
        this->get_parameter<double>("acc_w", imud.acc_w);
        this->get_parameter<double>("gyr_w", imud.gyr_w);

        this->get_parameter<double>("acc_n", imud.acc_n);
        this->get_parameter<double>("gyr_n", imud.gyr_n);
        this->get_parameter<double>("acc_w", imud.acc_w);
        this->get_parameter<double>("gyr_w", imud.gyr_w);
        
        downsample = this->get_parameter("downsample").as_bool();
        frequence_=this->get_parameter("frequence").as_int();


        this->declare_parameter<double>("adam_eta", 1.e-5);
        this->declare_parameter<double>("adam_beta1", 0.9);
        this->declare_parameter<double>("adam_beta2", 0.999);
        this->declare_parameter<double>("adam_epsilon", 1.e-8);

        gss.setAdamParameters(this->get_parameter("adam_eta").as_double(),
                              this->get_parameter("adam_beta1").as_double(),
                              this->get_parameter("adam_beta2").as_double(),
                              this->get_parameter("adam_epsilon").as_double());

        this->declare_parameter<double>("covisibility_threshold", 0.95);
        gss.setCovisibilityThreshold(this->get_parameter("covisibility_threshold").as_double());
        
        gss.setPoseIterations(this->get_parameter("pose_iterations").as_int());
        gss.setUpdateIterations(this->get_parameter("update_iterations").as_int());
        gss.setEtaPose(this->get_parameter("eta_pose").as_double());
        gss.setEtaUpdate(this->get_parameter("eta_update").as_double());
        gss.setGaussInitSizePx(this->get_parameter("gauss_init_size_px").as_int());
        gss.setNbPyrLevels(this->get_parameter("nb_pyr_levels").as_int());
        gss.setWDepth(this->get_parameter("w_depth").as_double());
        gss.setWDist(this->get_parameter("w_dist").as_double());
        gss.startDisplayLoop();
        gss.setOptimizer(this->get_parameter("optimizer").as_string());


        odomMsg.header.frame_id = worldFrameId;
        odomImuMsg.header.frame_id = worldFrameId;


        this->declare_parameter<std::string>("pose_estimation_method", std::string("full"));
        std::string method = this->get_parameter("pose_estimation_method").as_string();
        if(method == std::string("Full"))
        {
            gss.setPoseEstimationMethod(PoseEstimationMethodFull);
        }else if(method == std::string("WarpingSingleRendering"))
        {
            gss.setPoseEstimationMethod(PoseEstimationMethodWarpingSingleRendering);
        }else if(method == std::string("WarpingMultipleRendering"))
        {
            gss.setPoseEstimationMethod(PoseEstimationMethodWarpingMultipleRendering);
        }else
        {
            RCLCPP_WARN(this->get_logger(), "Unknown Pose Estimation Method, using Full method");
            gss.setPoseEstimationMethod(PoseEstimationMethodFull);
        }

        this->declare_parameter("color_transport", "raw");
        this->declare_parameter("depth_transport", "raw");

        rgbSub.subscribe(this, "image_color", this->get_parameter("color_transport").as_string(), qos_profile);
        depthSub.subscribe(this, "image_depth", this->get_parameter("depth_transport").as_string(), qos_profile);

        camInfoSub = this->create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", 1, std::bind(&GaussianSplattingSlamNode::camInfoCallback, this, std::placeholders::_1));
        sync.registerCallback(std::bind(&GaussianSplattingSlamNode::RGBDCallback, this, std::placeholders::_1, std::placeholders::_2));
        odomPub = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
        odomImuPub = this->create_publisher<nav_msgs::msg::Odometry>("odom_imu", 10);
        imu_preintSub = this->create_subscription<sensor_msgs::msg::Imu>("imu_preint", 100, std::bind(&GaussianSplattingSlamNode::IMUCallback, this, std::placeholders::_1));
        imuCache.connectInput(imuSub);


        this->declare_parameter("viewer", true);
        if(this->get_parameter("viewer").as_bool())
        {
            viewer = std::make_shared<GaussianSplattingViewer>(gss);
            viewer->startThread();
        }
    }

    GaussianSplattingSlamNode::~GaussianSplattingSlamNode()
    {
    }

    void GaussianSplattingSlamNode::RGBDCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg_rgb, const sensor_msgs::msg::Image::ConstSharedPtr msg_depth)
    {

        if (!hasCameraInfo)
        {
            return;
        }

        if (imu_initialized && gss.isInitialized())
        {
            // RCLCPP_WARN(this->get_logger(), " RGB callback ");
            lastRGBTime = rclcpp::Time(msg_rgb->header.stamp, RCL_ROS_TIME);
            lastDEPTHTime = rclcpp::Time(msg_depth->header.stamp, RCL_ROS_TIME);
            last_rgb_msg = msg_rgb;
            try
            {
                depthImg = cv_bridge::toCvShare(msg_depth)->image;
            }
            catch (cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "could not convert depth image with encoding '%s'", msg_depth->encoding.c_str());
                return;
            }
            try
            {
                rgbImg = cv_bridge::toCvShare(msg_rgb, "bgr8")->image;
            }
            catch (cv_bridge::Exception &e)
            {
                RCLCPP_ERROR(this->get_logger(),
                             "could not convert color image with encoding '%s'.", msg_rgb->encoding.c_str());
                return;
            }

            localDepthImg = depthImg.clone();
            localRgbImg = rgbImg.clone();

            sync_rgbd_imu();
        }
    }

    void GaussianSplattingSlamNode::sync_rgbd_imu()

    {
        // RCLCPP_WARN(this->get_logger(), " enter sync ");
        if (isProcessing)
        {
            // RCLCPP_WARN(this->get_logger(), " is porcessing");
            return;
        }

        if (lastRGBTime <= lastProcessedFrameStamp)
        {
            // RCLCPP_WARN(this->get_logger(), " lastrgb<lasprocessstamp");
            return;
        }
        rclcpp::Time lastIMUTime(IMUCache_preint.getLatestTime(), RCL_ROS_TIME);

        if (lastIMUTime < lastRGBTime)
        {
            // RCLCPP_WARN(this->get_logger(), " lastimu < lastrgb");
            return;
        }

        auto imuInterval = IMUCache_preint.getInterval(lastProcessedFrameStamp, lastRGBTime);
        if (imuInterval.empty())
        {
            // RCLCPP_WARN(this->get_logger(), " imuinterval is empty");
            return;
        }

        for (auto it = imuInterval.begin(); it != imuInterval.end(); it++)
        {
            imud.Acc = Eigen::Vector3d((*it)->linear_acceleration.x,
                                       (*it)->linear_acceleration.y,
                                       (*it)->linear_acceleration.z);
            imud.Gyro = Eigen::Vector3d((*it)->angular_velocity.x,
                                        (*it)->angular_velocity.y,
                                        (*it)->angular_velocity.z);

            gss.processIMU(rclcpp::Time((*it)->header.stamp).seconds(), imud);
        }
        lastProcessedFrameStamp = lastRGBTime;

        cv::Mat rgbDownsampleImg, depthDownsampleImg;
        isProcessing = true;

        if (downsample)
        {
            cv::pyrDown(localRgbImg, rgbDownsampleImg);
            cv::pyrDown(localDepthImg, depthDownsampleImg);
            gss.compute(rgbDownsampleImg, depthDownsampleImg, odomPose_init);
        }
        else
        {
            gss.compute(localRgbImg, localDepthImg, odomPose_init);
        }

        /*const Pose3D &pose = gss.getCameraPose();
        odomMsg.header.stamp = last_rgb_msg->header.stamp;
        odomMsg.child_frame_id = last_rgb_msg->header.frame_id;
        odomMsg.pose.pose.position.x = pose.position.x;
        odomMsg.pose.pose.position.y = pose.position.y;
        odomMsg.pose.pose.position.z = pose.position.z;
        odomMsg.pose.pose.orientation.x = pose.orientation.x;
        odomMsg.pose.pose.orientation.y = pose.orientation.y;
        odomMsg.pose.pose.orientation.z = pose.orientation.z;
        odomMsg.pose.pose.orientation.w = pose.orientation.w;*/

        const double *pose = gss.getImuPose(); 
        const double *velocity = gss.getImuVelocity(); 


        if (pose && velocity)
        {
            odomMsg.pose.pose.position.x = pose[0];
            odomMsg.pose.pose.position.y = pose[1];
            odomMsg.pose.pose.position.z = pose[2];
            odomMsg.pose.pose.orientation.x = pose[3];
            odomMsg.pose.pose.orientation.y = pose[4];
            odomMsg.pose.pose.orientation.z = pose[5];
            odomMsg.pose.pose.orientation.w = pose[6];

            odomMsg.twist.twist.linear.x = velocity[0];
            odomMsg.twist.twist.linear.y = velocity[1];
            odomMsg.twist.twist.linear.z = velocity[2];
            odomMsg.twist.twist.angular.x = velocity[3];
            odomMsg.twist.twist.angular.y = velocity[4];
            odomMsg.twist.twist.angular.z = velocity[5];
            

        }
        else
        {
            RCLCPP_WARN(this->get_logger(), "getImuPose() returned a null pointer!");
        }

        odomPub->publish(odomMsg);

        auto remainingImu = IMUCache_preint.getInterval(lastProcessedFrameStamp, IMUCache_preint.getLatestTime());

        auto first = remainingImu.front();
        Eigen::Vector3d acc(first->linear_acceleration.x,
                            first->linear_acceleration.y,
                            first->linear_acceleration.z);
        Eigen::Vector3d gyr(first->angular_velocity.x,
                            first->angular_velocity.y,
                            first->angular_velocity.z);
        Eigen::Vector3d ba, bg;

        gss.getBiases(ba, bg);

        rclcpp::Time t_imu(first->header.stamp, RCL_ROS_TIME);
        preint.init(acc, gyr, ba, bg,
                    imud.acc_n, imud.gyr_n, imud.acc_w, imud.gyr_w);

        for (auto it = remainingImu.begin(); it != remainingImu.end(); it++)
        {
            imud.Acc = Eigen::Vector3d((*it)->linear_acceleration.x,
                                       (*it)->linear_acceleration.y,
                                       (*it)->linear_acceleration.z);
            imud.Gyro = Eigen::Vector3d((*it)->angular_velocity.x,
                                        (*it)->angular_velocity.y,
                                        (*it)->angular_velocity.z);
            preint.add_imu((rclcpp::Time((*it)->header.stamp, RCL_ROS_TIME) - t_imu).seconds(),
                           imud.Acc, imud.Gyro);

            t_imu = rclcpp::Time((*it)->header.stamp, RCL_ROS_TIME);
        }

        // RCLCPP_INFO_STREAM(this->get_logger(), " Position : " << pose.position.x << " " << pose.position.y << " " << pose.position.z);
        // RCLCPP_INFO_STREAM(this->get_logger(), " Orientation : " << pose.orientation.x << " " << pose.orientation.y << " " << pose.orientation.z << " " << pose.orientation.w);
        isProcessing = false;
    }

    void GaussianSplattingSlamNode::IMUCallback(const sensor_msgs::msg::Imu::ConstSharedPtr msg_imu)
    {
        Eigen::Vector3d raw_acc(msg_imu->linear_acceleration.x,
                                msg_imu->linear_acceleration.y,
                                msg_imu->linear_acceleration.z);

        Eigen::Vector3d raw_gyro(msg_imu->angular_velocity.x,
                                 msg_imu->angular_velocity.y,
                                 msg_imu->angular_velocity.z);

        imuFrameId = msg_imu->header.frame_id;
        rclcpp::Time stampIMU(msg_imu->header.stamp, RCL_ROS_TIME);

        if (!hasCameraInfo)
        {
            return;
        }

        if (nb_init_imu < 100)
        {
            avg_acc += raw_acc;
            avg_gyro += raw_gyro;
            nb_init_imu++;
            lastStampIMU = stampIMU;
        }

        double dt = (stampIMU - lastStampIMU).seconds();
        lastStampIMU = stampIMU;

        /*double alpha_acc = dt == 0. ? 1. : dt / (dt + (1. / 200.));
        double alpha_gyro = dt == 0. ? 1. : dt / (dt + (1. / 200.));
        imud.Acc = (1. - alpha_acc) * imud.Acc + alpha_acc * raw_acc;
        imud.Gyro = (1. - alpha_gyro) * imud.Gyro + alpha_gyro * raw_gyro;*/

        imud.Acc = raw_acc;
        imud.Gyro = raw_gyro;

        if (nb_init_imu >= 100 && gss.isInitialized())
        {
            if (!imu_initialized)
            {
                avg_acc /= nb_init_imu;
                avg_gyro /= nb_init_imu;
                Eigen::Matrix3d R0 = Utility::g2R(avg_acc);
                Eigen::Isometry3d init_pose;
                init_pose.setIdentity();
                init_pose.linear() = R0;

                Eigen::Vector3d G(0, 0, 9.81);
                acc_bias = avg_acc - R0.inverse() * G;

                Pose3D init_pose_cam, init_pose_imu;
                Eigen::Quaterniond Q_init_cam, Q_init_imu;
                Eigen::Vector3d translation_init_ = init_pose.translation();
                Q_init_imu = Eigen::Quaterniond(init_pose.linear());
                Q_init_cam = Q_init_imu * q_imu_cam;

                init_pose_imu.position.x = translation_init_.x();
                init_pose_imu.position.y = translation_init_.y();
                init_pose_imu.position.z = translation_init_.z();
                init_pose_imu.orientation.x = Q_init_imu.x();
                init_pose_imu.orientation.y = Q_init_imu.y();
                init_pose_imu.orientation.z = Q_init_imu.z();
                init_pose_imu.orientation.w = Q_init_imu.w();

                init_pose_cam.position.x = t_imu_cam.x();
                init_pose_cam.position.y = t_imu_cam.y();
                init_pose_cam.position.z = t_imu_cam.z();
                init_pose_cam.orientation.x = Q_init_cam.x();
                init_pose_cam.orientation.y = Q_init_cam.y();
                init_pose_cam.orientation.z = Q_init_cam.z();
                init_pose_cam.orientation.w = Q_init_cam.w();

                gss.initialize(t_imu_cam, q_imu_cam, init_pose_imu, init_pose_cam);
                gss.setImuBias(acc_bias, avg_gyro);
                preint.init(imud.Acc, imud.Gyro, acc_bias, avg_gyro,
                            imud.acc_n, imud.gyr_n, imud.acc_w, imud.gyr_w);

                RCLCPP_INFO_STREAM(this->get_logger(), " Position IMU initiale: " << translation_init_.x() << " " << translation_init_.y() << " " << translation_init_.z());
                RCLCPP_INFO_STREAM(this->get_logger(), " orientation IMU initiale: " << Q_init_imu.x() << " " << Q_init_imu.y() << " " << Q_init_imu.z() << " " << Q_init_imu.w());
                RCLCPP_INFO_STREAM(this->get_logger(), " ROTATION CAM IN IMU : " << q_imu_cam.x() << " " << q_imu_cam.y() << " " << q_imu_cam.z() << " " << q_imu_cam.w());
                RCLCPP_INFO_STREAM(this->get_logger(), " Position CAM in IMU : " << t_imu_cam.x() << " " << t_imu_cam.y() << " " << t_imu_cam.z());
                RCLCPP_INFO_STREAM(this->get_logger(), " Preintegration IMU vo_imu is initialized: ");
                imu_initialized = true;
            }
            else
            {
                preint.add_imu(dt, imud.Acc, imud.Gyro);

                Eigen::Vector3d pos_imu, vel_imu;
                Eigen::Quaterniond rot_imu;

                Eigen::Vector3d P;
                Eigen::Quaterniond Q;
                Eigen::Vector3d V;

                gss.getState(P, Q, V);
                preint.predict(P, Q, V,
                               pos_imu, rot_imu, vel_imu);
                odomImuMsg.header.stamp = msg_imu->header.stamp;
                odomImuMsg.child_frame_id = msg_imu->header.frame_id;
                odomImuMsg.pose.pose.position.x = pos_imu.x();
                odomImuMsg.pose.pose.position.y = pos_imu.y();
                odomImuMsg.pose.pose.position.z = pos_imu.z();
                odomImuMsg.pose.pose.orientation.x = rot_imu.x();
                odomImuMsg.pose.pose.orientation.y = rot_imu.y();
                odomImuMsg.pose.pose.orientation.z = rot_imu.z();
                odomImuMsg.pose.pose.orientation.w = rot_imu.w();
                odomImuMsg.twist.twist.linear.x = vel_imu.x();
		        odomImuMsg.twist.twist.linear.y = vel_imu.y();
		        odomImuMsg.twist.twist.linear.z = vel_imu.z();

                odomImuPub->publish(odomImuMsg);

                sensor_msgs::msg::Imu::SharedPtr imu_cache(new sensor_msgs::msg::Imu);
                imu_cache->header = msg_imu->header;
                imu_cache->angular_velocity.x = imud.Gyro.x();
                imu_cache->angular_velocity.y = imud.Gyro.y();
                imu_cache->angular_velocity.z = imud.Gyro.z();
                imu_cache->linear_acceleration.x = imud.Acc.x();
                imu_cache->linear_acceleration.y = imud.Acc.y();
                imu_cache->linear_acceleration.z = imud.Acc.z();

                IMUCache_preint.add(imu_cache);
                // RCLCPP_WARN(this->get_logger(), " IMU added to IMUCache_preint");

                sync_rgbd_imu();
            }
        }
    }

    void GaussianSplattingSlamNode::camInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr info)
    {

        CameraParameters params;
        params.f.x = info->k[0];
        params.c.x = info->k[2];//- roi.x;
        params.f.y = info->k[4];
        params.c.y = info->k[5];// - roi.y;

        if (downsample)
        {
            params.f.x /= 2.;
            params.c.x /= 2.;
            params.f.y /= 2.;
            params.c.y /= 2.;
        }

        gss.setCameraParameters(params);
        geometry_msgs::msg::TransformStamped t;

        try
        {
            t = tfBuffer->lookupTransform(
                imuFrameId, info->header.frame_id,
                tf2::TimePointZero);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), " No TRANSFORM YET");
            return;
        }

        tf2::fromMsg(t.transform.rotation, q_imu_cam);    // rotate from camera to IMU
        tf2::fromMsg(t.transform.translation, t_imu_cam); // pos camera in imu frame go from imu to camera
        hasCameraInfo = true;
    }

} // namespace gaussian_splatting_slam
