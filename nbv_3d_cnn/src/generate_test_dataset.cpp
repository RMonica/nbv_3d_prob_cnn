#include "generate_test_dataset.h"

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// STL
#include <string>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <sstream>

// custom
#include "generate_test_dataset_opencl.h"
#include "origin_visibility.h"
#include "generate_single_image.h"
#include "voxelgrid.h"

class GenerateTestDataset
{
public:
  typedef uint64_t uint64;
  typedef int64_t int64;
  typedef uint8_t uint8;

  typedef std::vector<float> FloatVector;
  typedef std::vector<std::string> StringVector;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > Matrix3fVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;

  GenerateTestDataset(ros::NodeHandle & nh): m_nh(nh), m_opencl(m_nh)
  {
    std::string param_string;
    int param_int;
    double param_double;

    m_timer = m_nh.createTimer(ros::Duration(1.0), &GenerateTestDataset::onTimer, this, true);

    m_nh.param<std::string>(PARAM_NAME_3D_MODE, param_string, PARAM_DEFAULT_3D_MODE);
    if (param_string == PARAM_VALUE_3D_MODE_2D)
      m_is_3d = false;
    else if (param_string == PARAM_VALUE_3D_MODE_3D)
      m_is_3d = true;
    else
    {
      ROS_ERROR("generate_test_dataset: invalid value for %s: %s", (const char *)PARAM_NAME_3D_MODE,
                param_string.c_str());
      exit(1);
    }

    {
      float sensor_range_voxels;
      m_nh.param<float>(PARAM_NAME_SENSOR_RANGE_VOXELS, sensor_range_voxels, PARAM_DEFAULT_SENSOR_RANGE_VOXELS);

      int sensor_resolution_x;
      m_nh.param<int>(PARAM_NAME_SENSOR_RESOLUTION_X, sensor_resolution_x, PARAM_DEFAULT_SENSOR_RESOLUTION_X);
      int sensor_resolution_y;
      m_nh.param<int>(PARAM_NAME_SENSOR_RESOLUTION_Y, sensor_resolution_y, PARAM_DEFAULT_SENSOR_RESOLUTION_Y);

      float sensor_focal_length;
      m_nh.param<float>(PARAM_NAME_SENSOR_FOCAL_LENGTH, sensor_focal_length, PARAM_DEFAULT_SENSOR_FOCAL_LENGTH);

      int view_cube_resolution;
      m_nh.param<int>(PARAM_NAME_VIEW_CUBE_RESOLUTION, view_cube_resolution, PARAM_DEFAULT_VIEW_CUBE_RESOLUTION);

      int submatrix_resolution;
      m_nh.param<int>(PARAM_NAME_SUBMATRIX_RESOLUTION, submatrix_resolution,
                      PARAM_DEFAULT_SUBMATRIX_RESOLUTION);

      m_generate_single_image.reset(new GenerateSingleImage(m_nh, m_opencl, m_is_3d,
                                                            sensor_range_voxels, sensor_resolution_x,
                                                            sensor_resolution_y,
                                                            sensor_focal_length, view_cube_resolution,
                                                            submatrix_resolution));
    }

    m_nh.param<std::string>(PARAM_NAME_SOURCE_IMAGES_PREFIX, m_source_images_prefix,
                            PARAM_DEFAULT_SOURCE_IMAGES_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_SOURCE_IMAGES_SUFFIX, m_source_images_suffix,
                            PARAM_DEFAULT_SOURCE_IMAGES_SUFFIX);
    m_nh.param<std::string>(PARAM_NAME_DEST_IMAGES_PREFIX, m_dest_images_prefix, PARAM_DEFAULT_DEST_IMAGES_PREFIX);

    m_nh.param<int>(PARAM_NAME_NUM_VIEW_POSES_MAX, param_int, PARAM_DEFAULT_NUM_VIEW_POSES_MAX);
    m_num_view_poses_max = param_int;

    m_nh.param<int>(PARAM_NAME_NUM_VIEW_POSES_MIN, param_int, PARAM_DEFAULT_NUM_VIEW_POSES_MIN);
    m_num_view_poses_min = param_int;

    m_nh.param<std::string>(PARAM_NAME_ENVIRONMENT_RESIZE, param_string, PARAM_DEFAULT_ENVIRONMENT_RESIZE);
    m_environment_resize = Eigen::Vector2i(0, 0);
    if (!param_string.empty())
    {
      std::istringstream istr(param_string);
      istr >> m_environment_resize.x();
      istr >> m_environment_resize.y();
      if (!istr || (m_environment_resize.array() <= 0).any())
      {
        ROS_ERROR("generate_test_dataset: could not parse resize string \"%s\", ignoring resize.",
                  param_string.c_str());
        m_environment_resize = Eigen::Vector2i(0, 0);
      }
    }

    m_debug_environment_pub = m_nh.advertise<sensor_msgs::Image>("debug_environment_image", 1);
    m_debug_empty_observed_pub = m_nh.advertise<sensor_msgs::Image>("debug_empty_observed_image", 1);
    m_debug_occupied_observed_pub = m_nh.advertise<sensor_msgs::Image>("debug_occupied_observed_image", 1);
    m_debug_view_cube_eval_pub = m_nh.advertise<sensor_msgs::Image>("debug_view_cube_eval_image", 1);
    m_debug_frontier_pub = m_nh.advertise<sensor_msgs::Image>("debug_frontier_image", 1);

    m_nh.param<std::string>(PARAM_NAME_PREFIX_LIST, param_string, PARAM_DEFAULT_PREFIX_LIST);
    {
      std::istringstream istr(param_string);
      std::string str;
      while (istr >> str)
      {
        m_prefixes.push_back(str);
        ROS_INFO("generate_test_dataset: loaded prefix: %s", str.c_str());
      }
    }

    m_counter = 1;
    m_prefix_counter = 0;
    m_image_counter = 0;
  }

  void PublishCvImage(ros::Publisher & publisher, const cv::Mat & cv_mat, const std::string & type)
  {
    if (publisher.getNumSubscribers() == 0)
      return;

    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);

    cv_ptr->encoding = type;
    cv_ptr->header.stamp = ros::Time::now();
    cv_ptr->image = cv_mat;

    publisher.publish(cv_ptr->toImageMsg());
  }

  Eigen::Vector3f FindAvailableOrigin(const Voxelgrid::ConstPtr environment)
  {
    const uint64 width = environment->GetWidth();
    const uint64 height = environment->GetHeight();
    const uint64 depth = environment->GetDepth();

    uint64 count = 0;
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          if (!environment->at(x, y, z))
            count++;
        }
    if (count == 0)
      return Eigen::Vector3f(NAN, NAN, NAN);

    const uint64 selected = rand() % count;
    count = 0;
    for (uint64 z = 0; z < depth && count <= selected; z++)
      for (uint64 y = 0; y < height && count <= selected; y++)
        for (uint64 x = 0; x < width && count <= selected; x++)
        {
          if (!environment->at(x, y, z))
          {
            if (count == selected)
              return Eigen::Vector3f(x, y, z);
            count++;
          }
        }

    return Eigen::Vector3f(NAN, NAN, NAN);
  }

  cv::Mat GenerateViewCubeEvaluationVis(const Voxelgrid::ConstPtr environment,
                                        const Voxelgrid & cumulative_empty_observation,
                                        const Voxelgrid & view_cube_evaluation)
  {
    const uint64 width = environment->GetWidth();
    const uint64 height = environment->GetHeight();
    cv::Mat view_cube_evaluation_vis(width, height, CV_8UC3);

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (environment->at(x, y, 0))
        {
          view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[0] = 255;
          view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[1] = 255;
          view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[2] = 255;
          continue;
        }

        if (!cumulative_empty_observation.at(x, y, 0))
        {
          view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[0] = 0;
          view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[1] = 0;
          view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[2] = 0;
          continue;
        }

        view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[0] = 255;
        view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[1] = view_cube_evaluation.at(x, y, 0) * 255;
        view_cube_evaluation_vis.at<cv::Vec3b>(y, x)[2] = 0;
      }

    return view_cube_evaluation_vis;
  }

  Voxelgrid::Ptr LoadEnvironment(const uint64 prefix_counter, const uint64 counter)
  {
    Voxelgrid::Ptr environment;
    const std::string name = m_prefixes[m_prefix_counter] + std::to_string(m_counter);
    const std::string filename = m_source_images_prefix + name + m_source_images_suffix;
    ROS_INFO("generate_test_dataset: loading image: \"%s\"", filename.c_str());
    if (!m_is_3d)
      environment = Voxelgrid::Load2DOpenCV(filename);
    else
      environment = Voxelgrid::Load3DOctomap(filename);

    return environment;
  }

  void onTimer(const ros::TimerEvent &)
  {
    Voxelgrid::Ptr environment = LoadEnvironment(m_prefix_counter, m_counter);
    if (!environment)
    {
      m_prefix_counter++;
      m_counter = 1;

      if (m_prefix_counter >= m_prefixes.size())
      {
        ROS_INFO("generate_test_dataset: all images processed.");
        return;
      }

      ROS_INFO("generate_test_dataset: could not load image, switching to prefix %s",
               m_prefixes[m_prefix_counter].c_str());

      environment = LoadEnvironment(m_prefix_counter, m_counter);

      if (!environment)
      {
        ROS_ERROR("generate_test_dataset: could not load image for new prefix.");
        return;
      }
    }

    const Eigen::Vector3i voxelgrid_size = environment->GetSize();

    const uint64 view_cube_depth = m_is_3d ? 4 : 1;

    Voxelgrid cumulative_empty_observation(voxelgrid_size);
    Voxelgrid cumulative_occupied_observation(voxelgrid_size);
    Voxelgrid cumulative_frontier_observation(voxelgrid_size);
    Voxelgrid view_cube_evaluation(voxelgrid_size);
    Voxelgrid4 directional_scoreangle_evaluation(voxelgrid_size);
    Voxelgrid directional_view_cube_evaluation(voxelgrid_size.x() * 4, voxelgrid_size.y() * 4,
                                               voxelgrid_size.z() * view_cube_depth);
    Voxelgrid smooth_directional_view_cube_evaluation(voxelgrid_size.x() * 4, voxelgrid_size.y() * 4,
                                                      voxelgrid_size.z() * view_cube_depth);

    if (ros::isShuttingDown())
      return;

    ROS_INFO("generate_test_dataset: image %d", unsigned(m_image_counter));

    if (!m_is_3d && m_environment_resize != Eigen::Vector2i::Ones())
    {
      Eigen::Vector2f resize = Eigen::Vector2f::Ones().array() / m_environment_resize.cast<float>().array();
      environment = environment->Resize(resize);
    }

    Vector3fVector origins;
    QuaternionfVector orientations;

    const uint64 num_poses_diff = m_num_view_poses_max - m_num_view_poses_min;
    const uint64 num_poses = (num_poses_diff ? (uint64(rand()) % num_poses_diff) : 0) + m_num_view_poses_min;
    for (uint64 i = 0; i < num_poses; i++)
    {
      Eigen::Vector3f origin = FindAvailableOrigin(environment);
      if (origin.array().isNaN().any())
        continue;

      Eigen::Quaternionf orientation;
      if (!m_is_3d)
      {
        const float orient = (float(rand()) / RAND_MAX) * M_PI * 2.0f;
        orientation = Eigen::Quaternionf(
              Eigen::AngleAxisf(orient, Eigen::Vector3f::UnitZ()) *
              Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitX()) *
              Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitY())
              );
      }
      else
      {
        const Eigen::Vector3f random_axis = Eigen::Vector3f::Random().normalized();
        const float random_angle = (float(rand()) / RAND_MAX) * M_PI * 2.0f;
        orientation = Eigen::Quaternionf(Eigen::AngleAxisf(random_angle, random_axis));
//        origin = Eigen::Vector3f(7.0f, 7.0f, 15.0f);
//        orientation = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
      }
      origins.push_back(origin);
      orientations.push_back(orientation);

      ROS_INFO_STREAM("generate_test_dataset: viewpoint origin: " << origin.transpose());
      ROS_INFO_STREAM("generate_test_dataset: viewpoint orientation: " << orientation.vec().transpose()
                                                                       << " " << orientation.w());
    }

    m_generate_single_image->Run(*environment, origins, orientations,
                                 cumulative_empty_observation, cumulative_occupied_observation,
                                 cumulative_frontier_observation, view_cube_evaluation,
                                 directional_view_cube_evaluation,
                                 smooth_directional_view_cube_evaluation,
                                 directional_scoreangle_evaluation);

    if (!m_is_3d)
    {
      ROS_INFO("generate_test_dataset: visualizing...");
      cv::Mat view_cube_evaluation_vis = GenerateViewCubeEvaluationVis(environment,
                                                                       cumulative_empty_observation,
                                                                       view_cube_evaluation);

      PublishCvImage(m_debug_environment_pub, environment->ToOpenCVImage2D(), "8UC1");
      PublishCvImage(m_debug_empty_observed_pub, cumulative_empty_observation.ToOpenCVImage2D(), "8UC1");
      PublishCvImage(m_debug_occupied_observed_pub, cumulative_occupied_observation.ToOpenCVImage2D(), "8UC1");
      PublishCvImage(m_debug_view_cube_eval_pub, view_cube_evaluation_vis, "rgb8");
      PublishCvImage(m_debug_frontier_pub, cumulative_frontier_observation.ToOpenCVImage2D(), "8UC1");
    }

    ROS_INFO("generate_test_dataset: saving files to folder %s", m_dest_images_prefix.c_str());

    std::string environment_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_environment";
    environment->Save2D3D(environment_filename, m_is_3d);

    std::string gt_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_gt";
    view_cube_evaluation.Save2D3D(gt_filename, m_is_3d);

    std::string gt_scoreangle_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_scoreangle_gt";
    directional_scoreangle_evaluation.Save2D3D(gt_scoreangle_filename, m_is_3d);

    std::string directional_gt_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_directional_gt";
    directional_view_cube_evaluation.Save2D3D(directional_gt_filename, m_is_3d);

    std::string smooth_directional_gt_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_smooth_directional_gt";
    smooth_directional_view_cube_evaluation.Save2D3D(smooth_directional_gt_filename, m_is_3d);

    std::string frontier_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_frontier";
    cumulative_frontier_observation.Save2D3D(frontier_filename, m_is_3d);

    std::string empty_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_empty";
    cumulative_empty_observation.Save2D3D(empty_filename, m_is_3d);

    std::string occupied_filename = m_dest_images_prefix +
        std::to_string(m_image_counter) + "_occupied";
    cumulative_occupied_observation.Save2D3D(occupied_filename, m_is_3d);

    m_image_counter++;

    ROS_INFO("generate_test_dataset: finished.");

    m_counter++;
    m_timer.stop();
    m_timer.start();
  }

private:
  ros::Timer m_timer;

  uint64 m_counter;
  uint64 m_prefix_counter;
  uint64 m_image_counter;
  StringVector m_prefixes;

  bool m_is_3d;

  ros::NodeHandle & m_nh;

  std::string m_source_images_prefix;
  std::string m_dest_images_prefix;
  std::string m_source_images_suffix;

  Eigen::Vector2i m_environment_resize;
  uint64 m_num_view_poses_min;
  uint64 m_num_view_poses_max;

  ros::Publisher m_debug_environment_pub;
  ros::Publisher m_debug_empty_observed_pub;
  ros::Publisher m_debug_occupied_observed_pub;
  ros::Publisher m_debug_frontier_pub;
  ros::Publisher m_debug_view_cube_eval_pub;

  GenerateTestDatasetOpenCL m_opencl;

  std::shared_ptr<GenerateSingleImage> m_generate_single_image;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "generate_test_dataset");

  ros::NodeHandle nh("~");

  int param_int;
  nh.param<int>(PARAM_NAME_RANDOM_SEED, param_int, PARAM_DEFAULT_RANDOM_SEED);
  srand(param_int);

  GenerateTestDataset gtd(nh);

  ros::spin();

  return 0;
}
