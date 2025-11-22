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
            package='vigs_fusion',
            executable='gaussian_splatting_slam_node',
            output='screen',
             #prefix=['xterm -e gdb -ex run --args'],
            parameters=[{
                'color_transport':'compressed',
                'pose_iterations':12,
                'update_iterations':8,
                'eta_pose':1.e-5, 
                'eta_update':0.5e-3, 
                'gauss_init_size_px':3,
                'nb_pyr_levels':3,
                'downsample':True,
                'acc_n':0.2,  
                'gyr_n':0.0002, 
                'acc_w':0.02, 
                'gyr_w':0.0000002,
                'adam_eta':1.e-4,
                'adam_beta1':0.9, 
                'adam_beta2':0.999,
                'adam_epsilon':1.e-8,
                'covisibility_threshold':0.85,
                'optimizer':'adam',
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
                ('imu_preint', '/filtered_imu_xsens')
            ],
        )
    ])
