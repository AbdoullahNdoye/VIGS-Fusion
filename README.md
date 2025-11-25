Readme to be updated....
# VIGS-Fusion 
Fast Gaussian Splatting SLAM processed onboard a small quadrirotor in real time. 

Check out the video:

[![VIGS-Fusion](http://img.youtube.com/vi/9MY-AvcOoUE/0.jpg)](https://www.youtube.com/watch?v=9MY-AvcOoUE "My YouTube Video")

## Related Publications 
If you find VIGS-Fusion useful in your research or projects, please cite our work:

```
@inproceedings{ndoye2025vigs,
  title={VIGS-Fusion: Fast Gaussian Splatting SLAM processed onboard a small quadrirotor},
  author={Ndoye, Abdoullah and N{\`e}gre, Amaury and Marchand, Nicolas and Ruffier, Franck},
  booktitle={International Conference on Advanced Robotics (ICAR)},
  year={2025}
}
```
Note that the software is experimental and some changes have been done since the publication of the paper.

## Requirements 

- ### Ubuntu 22.04 or 24.04 
The system has been tested both on Ubuntu 22.04 and 24.04.

- ### [ROS Rolling](https://docs.ros.org/en/rolling/)
We used ROS 2 Rolling.

- ### [CUDA](https://developer.nvidia.com/cuda-downloads) >= 12.0 
The system has been tested on platforms with CUDA 12.3 and CUDA 12.9. It can be installed using the official .run/.deb installer.

- ### [OpenCV](https://opencv.org/) 
We use OpenCV 4.8 but any version of OpenCV 4 should work. OpenCV has to be installed with CUDA support and with contrib modules.
- **Download OpenCV 4.8.0 and OpenCV Contrib**
```bash
cd ~
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
git checkout 4.8.0
cd ../opencv_contrib
git checkout 4.8.0
cd ~/opencv
mkdir build && cd build
```
- **Configure the Build (CMake)**
```bash
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D WITH_TBB=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D CUDA_ARCH_BIN=8.6 \
-D OPENCV_EXTRA_MODULES_PATH=$HOME/opencv_contrib/modules \
-D BUILD_EXAMPLES=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D HAVE_opencv_python3=ON \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3.10/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_opencv_cudaarithm=OFF  \
-D BUILD_opencv_cudaimgproc=OFF \
-D BUILD_opencv_photo=OFF \
-D BUILD_opencv_cudalegacy=OFF \
..
```
Note: Set CUDA_ARCH_BIN to match your GPU’s compute capability.You can check yours here: https://developer.nvidia.com/cuda-gpus
- **Build and install OpenCV**
```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

Note that, to be able to use with ROS a version of OpenCV different from the default ROS OpenCV version (which doesn't have CUDA support), you might have to rebuild all ROS packages that require OpenCV against your specific version, particularly the [vision_opencv](http://wiki.ros.org/vision_opencv) package that provides [cv_bridge](http://wiki.ros.org/cv_bridge).

- ### [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) (5.10.1)
We used particular version of SuiteSparse (5.10.1). It can be installed: 
```bash
cd ~
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
git checkout v5.10.1
make -j$(nproc)
sudo make install
sudo ldconfig
```
- ### Eigen
```bash
sudo apt-get install libeigen3-dev
```

- ### glog 
```bash
sudo apt-get install -y libgoogle-glog-dev
```

- ### [Ceres Solver](http://ceres-solver.org) 2.0.0
Ceres solver was used with IMU, RGB-D, and Marginalization Jacobians calculated by hand for optimization.
```bash
cd ~
git clone https://github.com/ceres-solver/ceres-solver.git
cd ceres-solver
git checkout 2.0.0
mkdir build && cd build
cmake .. \
-DBUILD_TESTING=OFF \
-DBUILD_EXAMPLES=OFF \
-DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig
```
## Build VIGS-Fusion 

The system is provided as a ROS package which can be copied or cloned into your workspace and built directly.
### 1. Clone the repository

```bash
cd ~/ros2_work_space/src
git clone https://github.com/AbdoullahNdoye/VIGS-Fusion.git
```

### 2. Build 
Just go to your workspace root directory and build using `colcon build`.

```bash
cd ~/ros2_work_space
colcon build
```

## Usage
VIGS-Fusion can be tested with our dataset available here: https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/CI0K9G. 

Just play a bag file and launch: 
```bash
ros2 launch vigs_fusion vigs_fusion_launch.py
```