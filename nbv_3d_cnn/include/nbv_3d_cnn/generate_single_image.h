#ifndef GENERATE_SINGLE_IMAGE_H
#define GENERATE_SINGLE_IMAGE_H

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
#include <nbv_3d_cnn/generate_test_dataset_opencl.h>
#include <nbv_3d_cnn/origin_visibility.h>
#include <nbv_3d_cnn/voxelgrid.h>

class GenerateSingleImage
{
  public:
  typedef uint64_t uint64;
  typedef int64_t int64;
  typedef uint8_t uint8;

  typedef std::vector<float> FloatVector;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;
  typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > Matrix3fVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;

  static Eigen::Quaternionf Bearing2DToQuat(const Eigen::Vector3f bearing)
  {
    Eigen::Quaternionf axisq;
    {
      Eigen::Matrix3f rot;
      rot.col(2) = bearing;
      rot.col(1) = -Eigen::Vector3f::UnitZ();
      rot.col(0) = rot.col(1).cross(rot.col(2));
      axisq = Eigen::Quaternionf(rot);
    }

    return axisq;
  }

  GenerateSingleImage(ros::NodeHandle & nh, GenerateTestDatasetOpenCL & opencl, const bool is_3d,
                      const float sensor_range,
                      const float sensor_min_range,
                      const uint64 sensor_resolution_x,
                      const uint64 sensor_resolution_y,
                      const float sensor_focal_length, const uint64 view_cube_resolution,
                      const uint64 submatrix_resolution);

  void Run(const Voxelgrid & environment, const Vector3fVector & origins, const QuaternionfVector & orientations,
           const uint64 accuracy_skip_voxels,
           Voxelgrid &cumulative_empty_observation, Voxelgrid &cumulative_occupied_observation,
           Voxelgrid &cumulative_frontier_observation, Voxelgrid &view_cube_evaluation,
           Voxelgrid &directional_view_cube_evaluation, Voxelgrid &smooth_directional_view_cube_evaluation,
           Voxelgrid4 &directional_scoreangle_evaluation);

  void GenerateTestSample(const Voxelgrid &environment, const Vector3fVector &origins,
                          const QuaternionfVector &orientations,
                          Voxelgrid &cumulative_empty_observation, Voxelgrid &cumulative_occupied_observation,
                          Voxelgrid &cumulative_frontier_observation);

  Vector3fVector GenerateSensorRayBearings(const Eigen::Quaternionf &bearing,
                                           const float sensor_f, const Eigen::Vector2i &sensor_resolution);

  Vector3fVector GenerateCubeRayBearings(const Eigen::Vector3i &resolution) const;

  Vector3iVector SimulateView(const Voxelgrid & environment,
                              const Eigen::Vector3f & origin, const Eigen::Quaternionf &bearing,
                              const float sensor_f, const Eigen::Vector2i & sensor_resolution,
                              const float max_range, const float min_range,
                              FloatVector & nearest_dist, Vector3fVector & ray_bearings);

  OriginVisibilityVector EvaluateMultiViewCubes(const Voxelgrid & environment,
                                                const Voxelgrid & known_occupied,
                                                const Voxelgrid & known_empty,
                                                const Vector3fVector & origins,
                                                const float max_range,
                                                const float min_range);

  OriginVisibilityVector InformationGainMultiViewCubes(const Voxelgrid & environment,
                                                       const Voxelgrid & known_occupied,
                                                       const Voxelgrid & known_empty,
                                                       const Vector3fVector & origins,
                                                       const float max_range,
                                                       const float min_range,
                                                       const bool stop_at_first_hit,
                                                       const float a_priori_occupied_prob);

  Voxelgrid::Ptr FillViewKnown(const Voxelgrid & environment, const Eigen::Vector3f &origin,
                               const Eigen::Quaternionf &bearing,
                               const float sensor_f,
                               const Eigen::Vector2i &sensor_resolution,
                               const float max_range,
                               const float min_range,
                               Voxelgrid &observed_surface);

  Voxelgrid::Ptr FillViewKnownUsingCubeResolution(const Voxelgrid & environment,
                                                  const Eigen::Vector3f & origin,
                                                  const Eigen::Quaternionf & bearing,
                                                  const float max_range,
                                                  const float min_range,
                                                  Voxelgrid & observed_surface);

  Voxelgrid::Ptr FrontierFromObservedAndEmpty(const Voxelgrid & occupied_observed, const Voxelgrid & empty_observed) const;

  private:

  void UpdateViewCubeEvaluationPixel(const Eigen::Vector3i & pixel,
                                     const OriginVisibility & ov,
                                     Voxelgrid & view_cube_evaluation,
                                     Voxelgrid & directional_view_cube_evaluation,
                                     Voxelgrid & smooth_directional_view_cube_evaluation,
                                     Voxelgrid4 & directional_scoreangle_evaluation);

  ros::NodeHandle & m_nh;

  float m_sensor_range_voxels;
  float m_sensor_min_range_voxels;
  uint64 m_sensor_resolution_x;
  uint64 m_sensor_resolution_y;
  float m_sensor_focal_length;

  uint64 m_view_cube_resolution;
  uint64 m_submatrix_resolution;

  bool m_is_3d;

  GenerateTestDatasetOpenCL & m_opencl;
};

#endif // GENERATE_SINGLE_IMAGE_H
