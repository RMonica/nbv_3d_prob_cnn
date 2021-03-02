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
#include <nbv_3d_cnn/origin_visibility.h>

typedef uint8_t uint8;
typedef uint64_t uint64;
typedef std::vector<float> FloatVector;
typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "test_visibility_matrix");

  ros::NodeHandle nh("~");
  {
  cv::Mat test_image = cv::Mat(4, 4, CV_8UC1);
  test_image = 0;
  //test_image.at<uint8>(3, 2) = 255;
  test_image.at<uint8>(0, 3) = 255;
  //test_image.at<uint8>(0, 2) = 255;
  //test_image.at<uint8>(0, 1) = 255;
  //test_image.at<uint8>(1, 3) = 255;
  //test_image.at<uint8>(1, 2) = 255;
  //test_image.at<uint8>(1, 1) = 255;

  OriginVisibility ov = OriginVisibility::FromVisibilityMatrix(Eigen::Vector3f(0, 0, 0), 16, test_image);
  cv::Mat test_image_out = ov.GetVisibilityMatrix(4);
  cv::Mat test_image_out_int;
  test_image_out.convertTo(test_image_out_int, CV_8UC1, 255.0f);

  std::cout << "visibility matrix:\n";
  for (uint64 y = 0; y < test_image_out.rows; y++)
  {
    for (uint64 x = 0; x < test_image_out.cols; x++)
    {
      std::cout << int(test_image_out_int.at<uint8>(y, x)) << "\t";
    }
    std::cout << "\n";
  }

  cv::Mat test_image_out_smooth = ov.SmoothByHFOV(30.0f / 180.0f * M_PI).GetVisibilityMatrix(4);
  cv::Mat test_image_out_smooth_int;
  test_image_out_smooth.convertTo(test_image_out_smooth_int, CV_8UC1, 255.0f);

  std::cout << "visibility matrix smoothed:\n";
  for (uint64 y = 0; y < test_image_out_smooth.rows; y++)
  {
    for (uint64 x = 0; x < test_image_out_smooth.cols; x++)
    {
      std::cout << int(test_image_out_smooth_int.at<uint8>(y, x)) << "\t";
    }
    std::cout << "\n";
  }

  std::cout << std::flush;
  }
  {
  // ------------------------ 3D ------------
  Voxelgrid voxelgrid(4, 4, 4);
  voxelgrid.at(2, 3, 2) = 1.0f;
  voxelgrid.at(1, 3, 1) = 1.0f;
  std::cout << voxelgrid.ToString() << std::endl;

  OriginVisibility ov = OriginVisibility::FromVoxelgrid(Eigen::Vector3f(0, 0, 0), 16, voxelgrid);

  float best_gain;
  FloatVector gains;
  const QuaternionfVector test_orientations = OriginVisibility::GenerateStandardOrientationSet(8);
  const Eigen::Quaternionf orient =
    ov.GetBestSensorOrientationManyViews(test_orientations, Eigen::Vector2f::Ones() * M_PI / 6.0f,
                                         best_gain, gains);
  const Eigen::Vector3f bearing = orient * Eigen::Vector3f::UnitZ();
  std::cout << "Best bearing: " << bearing.transpose() << std::endl;
  std::cout << "Best gain: " << best_gain << std::endl;

  const Eigen::Quaternionf orient2 =
    ov.GetBestSensorOrientationHighResolution(test_orientations, Eigen::Vector2f::Ones() * M_PI / 6.0f,
                                              best_gain, gains);
  const Eigen::Vector3f bearing2 = orient2 * Eigen::Vector3f::UnitZ();
  std::cout << "Best bearing2: " << bearing2.transpose() << std::endl;
  std::cout << "Best gain2: " << best_gain << std::endl;

  //voxelgrid = ov.GetVisibilityVoxelgrid(4);
  //std::cout << voxelgrid.ToString() << std::endl;
  }

  return 0;
}
