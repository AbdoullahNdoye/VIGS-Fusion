#include <gaussian_splatting_slam/GaussianSplattingViewer.hpp>

namespace gaussian_splatting_slam
{
    GaussianSplattingViewer::GaussianSplattingViewer(GaussianSplattingSlam &gsslam_)
        : gsslam(gsslam_), focalPoint(3.0, 0.0, 0.0), distance(2.0), yaw(0.0), pitch(0.2), width(848), height(480), fov(90.0), prevMouseX(0), prevMouseY(0)
    {
        resetView();
    }

    GaussianSplattingViewer::~GaussianSplattingViewer()
    {
    }

    void GaussianSplattingViewer::resetView()
    {
        focalPoint = Eigen::Vector3f(2.0, 0.0, 0.0);
        distance = 3.0;

        cameraViewPosition = Eigen::Vector3f(0.0, 0.0, 1.0);

        yaw = 0.0;
        pitch = 0.2;
        fov = 60.;
    }

    void GaussianSplattingViewer::mouseCallbackStatic(int event, int x, int y, int flags, void *userdata)
    {
        GaussianSplattingViewer *viewer = static_cast<GaussianSplattingViewer *>(userdata);
        viewer->mouseCallback(event, x, y, flags);
    }

    void GaussianSplattingViewer::mouseCallback(int event, int x, int y, int flags)
    {
        // Handle mouse events here
        if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN || event == cv::EVENT_MBUTTONDOWN)
        {
            prevMouseX = x;
            prevMouseY = y;
        }
        else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON))
        {
            int dx = x - prevMouseX;
            int dy = y - prevMouseY;

            yaw -= dx * 0.01;   // Adjust sensitivity as needed
            pitch += dy * 0.01; // Adjust sensitivity as needed

            prevMouseX = x;
            prevMouseY = y;
        }
        else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_RBUTTON))
        {
            int dx = x - prevMouseX;
            int dy = y - prevMouseY;

            // distance *= std::exp2f(dy * 0.01); // Adjust sensitivity as needed
            Eigen::Quaternionf orientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
            cameraViewPosition += orientation * Eigen::Vector3f(0., dx * 0.01, dy * 0.01); // Adjust sensitivity as needed


            //Eigen::Quaternionf cameraOrientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * Eigen::Quaternionf(-0.5, 0.5, -0.5, 0.5);
            //cameraViewPosition += cameraOrientation * Eigen::Vector3f(-dx * 0.01, -dy * 0.01, 0); // Adjust sensitivity as needed

            prevMouseX = x;
            prevMouseY = y;
        }
        else if (event == cv::EVENT_MOUSEWHEEL)
        {
            int delta = cv::getMouseWheelDelta(flags);
            // distance *= std::exp2f(delta * 0.1); // Adjust sensitivity as needed
            Eigen::Quaternionf orientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
            cameraViewPosition += orientation * Eigen::Vector3f(-delta *0.1, 0.f, 0.f); // Adjust sensitivity as needed

            // Eigen::Quaternionf cameraOrientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * Eigen::Quaternionf(-0.5, 0.5, -0.5, 0.5);
            // cameraViewPosition += cameraOrientation * Eigen::Vector3f(0.f, 0.f, -delta * 0.1f); // Adjust sensitivity as needed
        }
        else if (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_MBUTTON))
        {
            int dx = x - prevMouseX;
            int dy = y - prevMouseY;

            //Eigen::Quaternionf cameraOrientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) * Eigen::Quaternionf(-0.5, 0.5, -0.5, 0.5);
            //focalPoint += distance * (cameraOrientation * Eigen::Vector3f(-dx * 0.001, -dy * 0.001, 0)); // Adjust sensitivity as needed

            prevMouseX = x;
            prevMouseY = y;
        }
    }

    void GaussianSplattingViewer::keyCallback(int key)
    {
        if (key == int(' '))
        {
            // Toggle render type
            renderType = static_cast<decltype(renderType)>((renderType + 1) % RENDER_TYPE_NUM);
        }
        else if (key == int('r'))
        {
            resetView();
        }
        else if (key == int('f'))
        {
            resetView();
            follow = !follow;
        }
        else if(key == 82 || key == 84) // Up or Down arrow keys
        {
            float d = (key == 82 ? 0.1 : -0.1);

            Eigen::Quaternionf orientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
            cameraViewPosition += orientation * Eigen::Vector3f(d, 0.f, 0.); // Adjust sensitivity as needed
        }
        else if(key == 81 || key == 83) // Left or Right arrow keys
        {
            float d = (key == 81 ? 0.1 : -0.1);

            Eigen::Quaternionf orientation = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
            cameraViewPosition += orientation * Eigen::Vector3f(0., d, 0.f); // Adjust sensitivity as needed
        }
        // else
        // {
        //     std::cout << "Key pressed: " << key << std::endl;
        // }
        std::cout << "Key pressed: " << key << std::endl;
        std::cout << "yaw: " << yaw << std::endl;
        std::cout << "pitch: " << pitch << std::endl;
    }

    void GaussianSplattingViewer::startThread()
    {
        renderThread = std::thread(&GaussianSplattingViewer::renderLoop, this);
    }

    void GaussianSplattingViewer::renderLoop()
    {
        cv::namedWindow("Gaussian Splatting Viewer", cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback("Gaussian Splatting Viewer", mouseCallbackStatic, this);

        while (true)
        {
            render();

            int key = cv::waitKey(200);

            if(key>=0)
                keyCallback(key);

            // if (cv::waitKey(300) >= 0)
            //     break; // Exit on any key press
        }
    }

    void GaussianSplattingViewer::render()
    {
        // compute the camera pose based on yaw, pitch, and distance
        Eigen::Map<Eigen::Quaternionf> cameraOrientation(reinterpret_cast<float *>(&cameraPose.orientation));
        Eigen::Quaternionf pitchyaw = Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());
        cameraOrientation = pitchyaw * Eigen::Quaternionf(-0.5, 0.5, -0.5, 0.5);

        Eigen::Vector3f globalFocalPoint = focalPoint;

        /*if (follow)
        {
            Eigen::Vector3d p;
            Eigen::Vector3d v;
            Eigen::Quaterniond q;

            gsslam.getState(p, q, v);

            // focalPoint = (p-prevImuPosition).cast<float>() + Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ())*focalPoint;

            float yawImu = q.toRotationMatrix().eulerAngles(0, 1, 2).z();

            // cameraOrientation = q.cast<float>()*cameraOrientation;
            cameraOrientation = Eigen::AngleAxisf(yaw + yawImu, Eigen::Vector3f::UnitZ()) * cameraOrientation;

            globalFocalPoint = p.cast<float>() + Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) * focalPoint;
        }*/

        Eigen::Map<Eigen::Vector3f> cameraPosition(reinterpret_cast<float *>(&cameraPose.position));

        cameraPosition = cameraViewPosition;

        // cameraPosition = globalFocalPoint - distance * (cameraOrientation * Eigen::Vector3f(0, 0, 1));

        // compute camera parameters
        cameraParams.f.x = width / (2.0 * tan(fov * M_PI / 360.0));
        cameraParams.f.y = cameraParams.f.x; // Assuming square pixels
        cameraParams.c.x = width / 2.0;
        cameraParams.c.y = height / 2.0;

        if (renderType == RENDER_TYPE_RGB)
        {
            gsslam.render3dView(renderedRgbGpu, renderedDepthGpu, cameraPose, cameraParams, width, height);

            // Convert GPU images to CPU for display
            renderedRgbGpu.download(renderedRgb);
            cv::imshow("Gaussian Splatting Viewer", renderedRgb);
        }
        else if (renderType == RENDER_TYPE_DEPTH)
        {
            gsslam.render3dView(renderedRgbGpu, renderedDepthGpu, cameraPose, cameraParams, width, height);
            // Convert GPU images to CPU for display
            renderedDepthGpu.download(renderedDepth);
            cv::imshow("Gaussian Splatting Viewer", 0.15 * renderedDepth);
        }
        else if (renderType == RENDER_TYPE_BLOBS)
        {
            gsslam.render3dViewBlobs(renderedRgbGpu, cameraPose, cameraParams, width, height);
            // Convert GPU images to CPU for display
            renderedRgbGpu.download(renderedRgb);
            cv::imshow("Gaussian Splatting Viewer", renderedRgb);
        }
    }
} // namespace gaussian_splatting_slam