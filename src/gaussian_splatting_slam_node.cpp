#include <rclcpp/rclcpp.hpp>
#include <gaussian_splatting_slam/GaussianSplattingSlamNode.hpp>

using namespace gaussian_splatting_slam;

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GaussianSplattingSlamNode>());
    rclcpp::shutdown();
    
    return 0;
}
