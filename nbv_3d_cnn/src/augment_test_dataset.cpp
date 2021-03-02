#include "augment_test_dataset.h"

#include <nbv_3d_cnn/generate_test_dataset_opencl.h>

#include <stdint.h>
#include <string>
#include <sstream>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <ros/ros.h>

#include <nbv_3d_cnn/voxelgrid.h>
#include <nbv_3d_cnn/origin_visibility.h>

class AugmentTestDataset
{
  public:
  typedef uint64_t uint64;

  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<float> FloatVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;

  AugmentTestDataset(ros::NodeHandle & nh): m_nh(nh), m_opencl(m_nh)
  {
    int param_int;
    std::string param_string;

    m_nh.param<std::string>(PARAM_NAME_SOURCE_IMAGES_PREFIX, m_source_prefix, PARAM_DEFAULT_SOURCE_IMAGES_PREFIX);

    m_nh.param<std::string>(PARAM_NAME_DEST_IMAGES_PREFIX, m_dest_prefix, PARAM_DEFAULT_DEST_IMAGES_PREFIX);

    m_nh.param<int>(PARAM_NAME_SUBMATRIX_RESOLUTION, param_int, PARAM_DEFAULT_SUBMATRIX_RESOLUTION);
    m_submatrix_resolution = param_int;

    m_nh.param<int>(PARAM_NAME_SENSOR_FOCAL_LENGTH, param_int, PARAM_DEFAULT_SENSOR_FOCAL_LENGTH);
    m_sensor_focal_length = param_int;

    m_nh.param<int>(PARAM_NAME_SENSOR_RESOLUTION_X, param_int, PARAM_DEFAULT_SENSOR_RESOLUTION_X);
    m_sensor_resolution_x = param_int;

    m_nh.param<int>(PARAM_NAME_SENSOR_RESOLUTION_Y, param_int, PARAM_DEFAULT_SENSOR_RESOLUTION_Y);
    m_sensor_resolution_y = param_int;

    m_nh.param<std::string>(PARAM_NAME_AUGMENT_ORIENTATIONS, param_string, PARAM_DEFAULT_AUGMENT_ORIENTATIONS);
    {
      std::istringstream istr(param_string);
      istr >> m_augment_orientations.x() >> m_augment_orientations.y() >> m_augment_orientations.z();
      if (!istr)
      {
        ROS_FATAL("augment_test_dataset: invalid augment orientation string: %s", param_string.c_str());
        exit(1);
      }

      ROS_INFO_STREAM("augment_test_dataset: augment orientations: " << m_augment_orientations.transpose());
    }

    m_timer = m_nh.createTimer(ros::Duration(0.1), &AugmentTestDataset::onTimer, this, true);

    m_counter = 0;
  }

  void onTimer(const ros::TimerEvent &)
  {
    Vector3iVector rotations;
    for (uint64 a = 0; a < m_augment_orientations.x(); a++)
      for (uint64 b = 0; b < m_augment_orientations.y(); b++)
        for (uint64 c = 0; c < m_augment_orientations.z(); c++)
          rotations.push_back(Eigen::Vector3i(a, b, c));

    const std::string directional_gt_filename = m_source_prefix + std::to_string(m_counter) +
                                                "_directional_gt.binvoxelgrid";
    const std::string frontier_filename = m_source_prefix + std::to_string(m_counter) +
        "_frontier.binvoxelgrid";
    const std::string empty_filename = m_source_prefix + std::to_string(m_counter) +
        "_empty.binvoxelgrid";

    ROS_INFO("augment_test_dataset: loading file %s", directional_gt_filename.c_str());
    Voxelgrid::Ptr directional_gt_ptr = Voxelgrid::FromFileBinary(directional_gt_filename);
    if (!directional_gt_ptr)
    {
      ROS_ERROR("augment_test_dataset: could not load file: %s", directional_gt_filename.c_str());
      return;
    }

    ROS_INFO("augment_test_dataset: loading file %s", frontier_filename.c_str());
    Voxelgrid::Ptr frontier_ptr = Voxelgrid::FromFileBinary(frontier_filename);
    if (!frontier_ptr)
    {
      ROS_ERROR("augment_test_dataset: could not load file: %s", frontier_filename.c_str());
      return;
    }

    ROS_INFO("augment_test_dataset: loading file %s", empty_filename.c_str());
    Voxelgrid::Ptr empty_ptr = Voxelgrid::FromFileBinary(empty_filename);
    if (!empty_ptr)
    {
      ROS_ERROR("augment_test_dataset: could not load file: %s", empty_filename.c_str());
      return;
    }

    uint64 augment_counter = 0;
    for (const Eigen::Vector3i & rotation : rotations)
    {
      ROS_INFO("  augment_counter: %u", unsigned(augment_counter));

      Voxelgrid::Ptr r_empty = empty_ptr;
      Voxelgrid::Ptr r_frontier = frontier_ptr;
      Voxelgrid::Ptr r_gt = directional_gt_ptr;

      r_empty = r_empty->Rotate90n(0, 1, rotation[2]);
      r_frontier = r_frontier->Rotate90n(0, 1, rotation[2]);
      r_gt = r_gt->Rotate90n(0, 1, rotation[2]);

      r_empty = r_empty->Rotate90n(0, 2, rotation[1]);
      r_frontier = r_frontier->Rotate90n(0, 2, rotation[1]);
      r_gt = r_gt->Rotate90n(0, 2, rotation[1]);

      r_empty = r_empty->Rotate90n(1, 2, rotation[0]);
      r_frontier = r_frontier->Rotate90n(1, 2, rotation[0]);
      r_gt = r_gt->Rotate90n(1, 2, rotation[0]);

      const Voxelgrid & empty = *r_empty;
      const Voxelgrid & frontier = *r_frontier;
      const Voxelgrid & directional_gt = *r_gt;

      const Eigen::Vector3i size = directional_gt.GetSize() / m_submatrix_resolution;

      const float hfov_x = std::atan2(float(m_sensor_resolution_x) / 2.0f, float(m_sensor_focal_length));
      const float hfov_y = std::atan2(float(m_sensor_resolution_y) / 2.0f, float(m_sensor_focal_length));
      const QuaternionfVector orientations = OriginVisibility::GenerateStandardOrientationSet(8);
      const Eigen::Vector2f hfov(hfov_x, hfov_y);

      Voxelgrid4 scoreangle(size);

      Voxelgrid flat(Eigen::Vector3i(size.x() * orientations.size(), size.y(), size.z()));

      FloatVector gains;
      for (uint64 z = 0; z < uint64(size.z()); z++)
      {
        ROS_INFO("  at z: %u", unsigned(z));
        for (uint64 y = 0; y < uint64(size.y()); y++)
          for (uint64 x = 0; x < uint64(size.x()); x++)
          {
            if (ros::isShuttingDown())
              return;

            const Eigen::Vector3i xyz(x, y, z);
            const Eigen::Vector3i submatrix_origin = xyz * m_submatrix_resolution;
            const Eigen::Vector3i submatrix_size = Eigen::Vector3i::Ones() * m_submatrix_resolution;
            Voxelgrid::ConstPtr submatrix_ptr = directional_gt.GetSubmatrix(submatrix_origin, submatrix_size);
            const Voxelgrid & submatrix = *submatrix_ptr;

            const OriginVisibility ov = OriginVisibility::FromVoxelgrid(xyz.cast<float>(), 4, submatrix);

            float best_gain;
            const Eigen::Quaternionf best_orient =
              ov.GetBestSensorOrientationOCL(&m_opencl, orientations, hfov, best_gain, gains);

            scoreangle.at(0).at(x, y, z) = best_orient.x();
            scoreangle.at(1).at(x, y, z) = best_orient.y();
            scoreangle.at(2).at(x, y, z) = best_orient.z();
            scoreangle.at(3).at(x, y, z) = best_orient.w();

            for (uint64 i = 0; i < orientations.size(); i++)
              flat.at(i + x * orientations.size(), y, z) = gains[i];
          }
      }

      const std::string scoreangle_gt_filename = m_dest_prefix + std::to_string(m_counter) + "_scoreangle_gte_" +
                                                 std::to_string(augment_counter) + "_";
      ROS_INFO("augment_test_dataset: saving file: %s", scoreangle_gt_filename.c_str());
      scoreangle.Save2D3D(scoreangle_gt_filename, true);
      const std::string smooth_directional_gt_filename = m_dest_prefix + std::to_string(m_counter) +
          "_smooth_directional_gte_" +
          std::to_string(augment_counter) + "";
      ROS_INFO("augment_test_dataset: saving file: %s", smooth_directional_gt_filename.c_str());
      flat.Save2D3D(smooth_directional_gt_filename, true);
      const std::string empty_filename = m_dest_prefix + std::to_string(m_counter) + "_empty_gte_" +
                                                 std::to_string(augment_counter) + "";
      ROS_INFO("augment_test_dataset: saving file: %s", empty_filename.c_str());
      empty.Save2D3D(empty_filename, true);
      const std::string frontier_filename = m_dest_prefix + std::to_string(m_counter) + "_frontier_gte_" +
                                                 std::to_string(augment_counter) + "";
      ROS_INFO("augment_test_dataset: saving file: %s", frontier_filename.c_str());
      frontier.Save2D3D(frontier_filename, true);
      augment_counter++;
    }

    m_counter++;
    m_timer.stop();
    m_timer.start();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  ros::NodeHandle & m_nh;

  GenerateTestDatasetOpenCL m_opencl;

  std::string m_source_prefix;
  std::string m_dest_prefix;

  Eigen::Vector3i m_augment_orientations;

  uint64 m_sensor_resolution_x;
  uint64 m_sensor_resolution_y;
  float m_sensor_focal_length;

  uint64 m_submatrix_resolution;

  ros::Timer m_timer;
  uint64 m_counter;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "augment_test_dataset");

  ros::NodeHandle nh("~");

  AugmentTestDataset atd(nh);

  ros::spin();

  return 0;
}
