import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
         Node(package='tf2_ros',
             executable='static_transform_publisher',
             output='screen',
             #arguments = ["0.130913", "0.0429078", "-0.0263298",
             #             "0.0018806", "0.00625239", "-0.000737243", "0.999975",
             #             "xsens_imu_link", "camera_link"],
             arguments = ["0.0456823", " 0.0360948", "-0.0745264",
                          "-0.00910219", "0.0119687", "-0.0126518", "0.999804",
                          "xsens_imu_link", "camera_link"],
             parameters=[{'use_sim_time': True}],
             ),
        Node(package='cuda_depth_register',
             executable='cuda_depth_register_node',
             namespace='/camera',
             parameters=[{
                 'median_filter':True,
                 'encoding':'16UC1',
                 'use_sim_time': True,
                 }],
             ),
        Node(
            package='gaussian_splatting_slam',
            executable='gaussian_splatting_slam_node',
            output='screen',
            #prefix=['xterm -e gdb -ex run --args'],
            parameters=[{
                'color_transport':'compressed',
                # 'roi_x':64,
                # 'roi_y':8,
                # 'roi_w':1280-64-16,
                # 'roi_h':720-16,
                'pose_iterations':12,
                'update_iterations':8,
                'eta_pose':1.e-5, #1e-5
                'eta_update':0.5e-3, #1e-5, 0.002,
                'gauss_init_size_px':3, #4,
                'nb_pyr_levels':3,
                'downsample':True,
                'acc_n':0.2,  
                'gyr_n':0.0002, 
                'acc_w':0.02, 
                'gyr_w':0.0000002,
                'adam_eta':1.e-4,
                'adam_beta1':0.9, # 0.8
                'adam_beta2':0.999,
                'adam_epsilon':1.e-8,
                'covisibility_threshold':0.85,
                'w_depth':1.0,
                'w_dist':0.1,
                'world_frame_id':'map',
                'pose_estimation_method': 'WarpingMultipleRendering', #'WarpingMultipleRendering', 'WarpingSingleRendering' or 'Full'
                'viewer': True,  # Set to True to enable the viewer
                'use_sim_time': True,
            }],
            remappings=[
                ('camera_info', '/camera/color/camera_info'),
                ('image_color/compressed', '/camera/color/image_raw/compressed'),
                ('image_depth', '/camera/depth_registered/image_rect_raw'),
                ('odom', '/odom_gs'),
                #('imu', '/imu_attitude')
                ('imu_preint', '/filtered_imu_xsens')
            ],
        )
    ])
