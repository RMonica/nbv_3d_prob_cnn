#include "generate_single_image.h"

#define SQR(x) ((x)*(x))

GenerateSingleImage::GenerateSingleImage(ros::NodeHandle & nh, GenerateTestDatasetOpenCL & opencl,
                                         const bool is_3d,
                                         const float sensor_range, const uint64 sensor_resolution_x,
                                         const uint64 sensor_resolution_y,
                                         const float sensor_focal_length, const uint64 view_cube_resolution,
                                         const uint64 submatrix_resolution):
  m_nh(nh), m_opencl(opencl)
{
  m_sensor_range_voxels = sensor_range;
  m_sensor_focal_length = sensor_focal_length;
  m_sensor_resolution_x = sensor_resolution_x;
  m_sensor_resolution_y = sensor_resolution_y;
  m_view_cube_resolution = view_cube_resolution;
  m_submatrix_resolution = submatrix_resolution;

  m_is_3d = is_3d;
}

void GenerateSingleImage::GenerateTestSample(const Voxelgrid & environment, const Vector3fVector & origins,
                                             const QuaternionfVector & orientations,
                                             Voxelgrid & cumulative_empty_observation,
                                             Voxelgrid & cumulative_occupied_observation,
                                             Voxelgrid & cumulative_frontier_observation)
{
  cumulative_empty_observation = *environment.FilledWith(0.0);
  cumulative_occupied_observation = *environment.FilledWith(0.0);

  const float max_range = m_sensor_range_voxels;

  const uint64 num_poses = origins.size();
  for (uint64 i = 0; i < num_poses; i++)
  {
    ROS_INFO("generate_test_dataset: rendering view %u", unsigned(i));

    const Eigen::Vector3f origin = origins[i];
    const Eigen::Quaternionf sensor_orientation = orientations[i];
    const Eigen::Vector2i resolution(m_sensor_resolution_x, m_sensor_resolution_y);

    Voxelgrid occupied_observed_by_this;
    const Voxelgrid empty_observed_by_this = *FillViewKnown(environment, origin, sensor_orientation,
                                                            m_sensor_focal_length,
                                                            resolution, max_range,
                                                            occupied_observed_by_this);

    cumulative_empty_observation = *cumulative_empty_observation.Or(empty_observed_by_this);
    cumulative_occupied_observation = *cumulative_occupied_observation.Or(occupied_observed_by_this);
  }

  cumulative_frontier_observation = *FrontierFromObservedAndEmpty(cumulative_occupied_observation,
                                                                  cumulative_empty_observation);
}

Voxelgrid::Ptr GenerateSingleImage::FrontierFromObservedAndEmpty(const Voxelgrid & occupied_observed,
                                                                 const Voxelgrid & empty_observed) const
{
  const uint64 width = occupied_observed.GetWidth();
  const uint64 height = occupied_observed.GetHeight();
  const uint64 depth = occupied_observed.GetDepth();

  Voxelgrid::Ptr result_ptr = occupied_observed.FilledWith(0.0f);
  Voxelgrid & result = *result_ptr;
  {
    const Voxelgrid empty_observation_expand =
      *(m_is_3d ? empty_observed.DilateCross(Eigen::Vector3i::Ones())
                : empty_observed.DilateRect(Eigen::Vector3i::Ones()));

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          if (!empty_observed.at(x, y, z) &&
              empty_observation_expand.at(x, y, z) &&
              !occupied_observed.at(x, y, z))
            result.at(x, y, z) = 1.0;
        }
  }

  return result_ptr;
}

void GenerateSingleImage::UpdateViewCubeEvaluationPixel(const Eigen::Vector3i & pixel,
                                                        const OriginVisibility & ov,
                                                        Voxelgrid & view_cube_evaluation,
                                                        Voxelgrid & directional_view_cube_evaluation,
                                                        Voxelgrid & smooth_directional_view_cube_evaluation,
                                                        Voxelgrid4 & directional_scoreangle_evaluation)
{
  view_cube_evaluation.at(pixel.x(), pixel.y(), pixel.z()) = ov.AverageVisibility();

  const Voxelgrid visibility_matrix = ov.GetVisibilityVoxelgrid(m_submatrix_resolution);
  directional_view_cube_evaluation.SetSubmatrix(pixel * m_submatrix_resolution, visibility_matrix);

  const float hfov_x = std::atan2(float(m_sensor_resolution_x) / 2.0f, float(m_sensor_focal_length));
  const float hfov_y = std::atan2(float(m_sensor_resolution_y) / 2.0f, float(m_sensor_focal_length));

  if (m_is_3d)
  {
    float best_score;
    const QuaternionfVector orientations = ov.GenerateStandardOrientationSet(8);
    FloatVector gains;
    const Eigen::Quaternionf best_orient = ov.GetBestSensorOrientationHighResolution(orientations,
                                                                                     Eigen::Vector2f(hfov_x, hfov_y),
                                                                                     best_score,
                                                                                     gains);

    Voxelgrid smooth_visibility_matrix(gains.size(), 1, 1);
    for (uint64 i = 0; i < gains.size(); i++)
      smooth_visibility_matrix.at(i, 0, 0) = gains[i];
    smooth_directional_view_cube_evaluation.SetSubmatrix(Eigen::Vector3i(pixel.x() * gains.size(), pixel.y(), pixel.z()),
                                                                         smooth_visibility_matrix);

    directional_scoreangle_evaluation.SetAt(pixel,
      Eigen::Vector4f(best_score * best_orient.x() * 0.5f + 0.5f,
                      best_score * best_orient.y() * 0.5f + 0.5f,
                      best_score * best_orient.z() * 0.5f + 0.5f,
                      best_score * best_orient.w() * 0.5f + 0.5f));
  }
  else
  {
    const OriginVisibility smooth_ov = ov.SmoothByHFOV(hfov_x);
    const Voxelgrid smooth_visibility_matrix = ov.GetVisibilityVoxelgrid(m_submatrix_resolution);
    smooth_directional_view_cube_evaluation.SetSubmatrix(pixel * m_submatrix_resolution, smooth_visibility_matrix);

    Eigen::Vector3i index;
    Eigen::Vector3f best_bearing;
    const float best_score = smooth_ov.MaxVisibility(index, best_bearing);
    directional_scoreangle_evaluation.SetAt(pixel,
      Eigen::Vector3f(best_score * best_bearing.x() * 0.5f + 0.5f, best_score * best_bearing.y() * 0.5f + 0.5f, 0.0f));
  }

}

void GenerateSingleImage::Run(const Voxelgrid & environment, const Vector3fVector & origins,
                              const QuaternionfVector &orientations,
                              Voxelgrid & cumulative_empty_observation,
                              Voxelgrid & cumulative_occupied_observation,
                              Voxelgrid & cumulative_frontier_observation,
                              Voxelgrid & view_cube_evaluation,
                              Voxelgrid & directional_view_cube_evaluation,
                              Voxelgrid & smooth_directional_view_cube_evaluation,
                              Voxelgrid4 & directional_scoreangle_evaluation)
{
  GenerateTestSample(environment, origins, orientations, cumulative_empty_observation,
                     cumulative_occupied_observation,
                     cumulative_frontier_observation);

  const uint64 width = environment.GetWidth();
  const uint64 height = environment.GetHeight();
  const uint64 depth = environment.GetDepth();
  const Eigen::Vector3i size = environment.GetSize();
  const float max_range = m_sensor_range_voxels;

  view_cube_evaluation = *environment.FilledWith(0.0f);

  directional_scoreangle_evaluation = Voxelgrid4(width, height, depth);
  if (!m_is_3d)
    directional_scoreangle_evaluation.Fill(Eigen::Vector4f::UnitW()); // w is alpha
  else
    directional_scoreangle_evaluation.Fill(Eigen::Vector4f::Zero());

  directional_view_cube_evaluation = Voxelgrid(size.x() * m_submatrix_resolution,
                                               size.y() * m_submatrix_resolution,
                                               size.z() * (m_is_3d ? m_submatrix_resolution : 1));
  directional_view_cube_evaluation.Fill(0.0f);

  if (!m_is_3d)
    smooth_directional_view_cube_evaluation = directional_view_cube_evaluation;
  else
  {
    const uint64 standard_orientations_size = OriginVisibility::GenerateStandardOrientationSet(8).size();
    smooth_directional_view_cube_evaluation = Voxelgrid(size.x() * standard_orientations_size, size.y(), size.z());
    smooth_directional_view_cube_evaluation.Fill(0.0f);
  }

  Vector3fVector batch_origins;
  Vector3iVector batch_coords;
  const uint64 threads_per_batch = SQR(m_view_cube_resolution - 1) * 6;
  const uint64 BATCH_SIZE = std::max<uint64>((1ul << 15ul) / (threads_per_batch), 1);
  batch_origins.reserve(BATCH_SIZE);
  batch_coords.reserve(BATCH_SIZE);
  ROS_INFO("generate_test_dataset: evaluating view poses...");
  for (uint64 z = 0; z < depth; z++)
  {
    for (uint64 y = 0; y < height; y++)
    {
      //ROS_INFO("generate_test_dataset: evaluating view poses 0-%u %u", unsigned(width - 1), unsigned(y));
      for (uint64 x = 0; x < width; x++)
      {
        if (!cumulative_empty_observation.at(x, y, z))
          continue;

        if (ros::isShuttingDown())
          return;

        const Eigen::Vector3f origin(x, y, z);
        batch_origins.push_back(origin);
        batch_coords.push_back(Eigen::Vector3i(x, y, z));

        if (batch_origins.size() >= BATCH_SIZE)
        {
          if (m_is_3d)
            ROS_INFO("generate_test_dataset: processing batch at x,y,z = %u,%u,%u",
                     unsigned(x), unsigned(y), unsigned(z));
          else
            ROS_INFO("generate_test_dataset: processing batch at x,y = %u,%u", unsigned(x), unsigned(y));
          OriginVisibilityVector origin_visibility =
            EvaluateMultiViewCubes(environment, cumulative_occupied_observation, cumulative_empty_observation,
                                   batch_origins, max_range);
          ROS_INFO("generate_test_dataset: saving batch at x,y = %u,%u", unsigned(x), unsigned(y));
          for (uint64 i = 0; i < batch_origins.size(); i++)
            UpdateViewCubeEvaluationPixel(batch_coords[i], origin_visibility[i],
                                          view_cube_evaluation, directional_view_cube_evaluation,
                                          smooth_directional_view_cube_evaluation,
                                          directional_scoreangle_evaluation);

          batch_origins.clear();
          batch_coords.clear();
        }
      }
    }
  }

  if (!batch_origins.empty())
  {
    ROS_INFO("generate_test_dataset: processing last batch");
    OriginVisibilityVector origin_visibility =
      EvaluateMultiViewCubes(environment, cumulative_occupied_observation, cumulative_empty_observation,
                             batch_origins, max_range);
    ROS_INFO("generate_test_dataset: saving last batch");
    for (uint64 i = 0; i < batch_origins.size(); i++)
      UpdateViewCubeEvaluationPixel(batch_coords[i], origin_visibility[i],
                                    view_cube_evaluation, directional_view_cube_evaluation,
                                    smooth_directional_view_cube_evaluation,
                                    directional_scoreangle_evaluation);
    batch_origins.clear();
    batch_coords.clear();
  }
}

GenerateSingleImage::Vector3fVector GenerateSingleImage::GenerateSensorRayBearings(
  const Eigen::Quaternionf & orientation, const float sensor_f, const Eigen::Vector2i & sensor_resolution)
{
  Vector3fVector ray_bearings(sensor_resolution.prod());

  for (uint64 iy = 0; iy < sensor_resolution.y(); iy++)
    for (uint64 ix = 0; ix < sensor_resolution.x(); ix++)
    {
      const Eigen::Vector3f v = Eigen::Vector3f(float(ix) - (sensor_resolution.x() / 2.0f) + 0.5f,
                                                float(iy) - (sensor_resolution.y() / 2.0f) + 0.5f,
                                                sensor_f
                                                ).normalized();
      const Eigen::Vector3f b = orientation * v;
      ray_bearings[ix + iy * sensor_resolution.x()] = b;
    }

  return ray_bearings;
}

GenerateSingleImage::Vector3fVector GenerateSingleImage::GenerateCubeRayBearings(
  const Eigen::Vector3i & resolution) const
{

  Vector3fVector cube_bearings;

  const Eigen::Vector3f matrix_center = resolution.cast<float>() / 2.0f - 0.5f * Eigen::Vector3f::Ones();

  Eigen::Vector3i index;
  const Eigen::Vector3i resolution1 = (resolution - Eigen::Vector3i::Ones());
  for (index.z() = 0; index.z() < resolution.z(); index.z()++)
    for (index.y() = 0; index.y() < resolution.y(); index.y()++)
     for (index.x() = 0; index.x() < resolution.x(); index.x()++)
     {
       if ((index.array() != 0).all() &&
           (index.head<2>().array() != resolution1.head<2>().array()).all() &&
           (!m_is_3d || index.z() != resolution1.z()))
       {
         cube_bearings.push_back(Eigen::Vector3f::Zero());
         continue; // in cube center, bearings are invalid
       }

       const Eigen::Vector3f ray = (index.cast<float>() - matrix_center).normalized();
       cube_bearings.push_back(ray);
     }

  return cube_bearings;
}

GenerateSingleImage::Vector3iVector GenerateSingleImage::SimulateView(const Voxelgrid & environment,
                            const Eigen::Vector3f & origin, const Eigen::Quaternionf & bearing,
                            const float sensor_f, const Eigen::Vector2i & sensor_resolution,
                            const float max_range,
                            FloatVector & nearest_dist, Vector3fVector & ray_bearings)
{
  nearest_dist.assign(sensor_resolution.prod(), -1.0);
  Vector3iVector nearest_point(sensor_resolution.prod());
  ray_bearings = GenerateSensorRayBearings(bearing, sensor_f, sensor_resolution);

  m_opencl.SimulateView(environment, origin, ray_bearings, max_range,
                        nearest_dist, nearest_point);

  return nearest_point;
}

OriginVisibilityVector GenerateSingleImage::EvaluateMultiViewCubes(const Voxelgrid & environment,
                                                                   const Voxelgrid & known_occupied,
                                                                   const Voxelgrid & known_empty,
                                                                   const Vector3fVector & origins,
                                                                   const float max_range)
{
  OriginVisibilityVector result;
  for (const Eigen::Vector3f & origin : origins)
    result.push_back(OriginVisibility(origin, m_view_cube_resolution, m_is_3d));

  const Voxelgrid environment_shink = *environment.ErodeCross(Eigen::Vector3i::Ones());
  const Voxelgrid known_environment = *(known_empty.Or(known_occupied)->Or(environment_shink));

  const uint64 width = known_environment.GetWidth();
  const uint64 height = known_environment.GetHeight();
  const uint64 depth = known_environment.GetDepth();

  const Eigen::Vector3i resolution = Eigen::Vector3i(m_view_cube_resolution, m_view_cube_resolution,
                                                     m_is_3d ? m_view_cube_resolution : 1);

  const Vector3fVector cube_bearings = GenerateCubeRayBearings(resolution);

  Vector3fVector all_bearings;
  all_bearings.reserve(cube_bearings.size() * origins.size());
  for (uint64 i = 0; i < origins.size(); i++)
    all_bearings.insert(all_bearings.end(), cube_bearings.begin(), cube_bearings.end());
  Vector3fVector all_origins;
  all_origins.reserve(cube_bearings.size() * origins.size());
  for (uint64 i = 0; i < origins.size(); i++)
    for (uint64 h = 0; h < cube_bearings.size(); h++)
      all_origins.push_back(origins[i]);

  FloatVector nearest_dist;
  Vector3iVector observed_points;
  m_opencl.SimulateMultiRayWI(environment, all_origins, all_bearings, max_range, nearest_dist, observed_points);

  for (uint64 ov_i = 0; ov_i < result.size(); ov_i++)
  { 
    if (ros::isShuttingDown())
      return result;
    Uint64Vector hits(cube_bearings.size());
    Uint64Vector miss(cube_bearings.size());

    FloatVector nearest_dist_current_view(nearest_dist.begin() +
                                          ov_i * cube_bearings.size(),
                                          nearest_dist.begin() +
                                          (ov_i + 1) * cube_bearings.size());

    const Eigen::Vector3f origin = origins[ov_i];
    const uint64 side = 2 * max_range + 1;
    const Eigen::Vector3i env_side = Eigen::Vector3i(width, height, depth);
    const Eigen::Vector3i side3 = Eigen::Vector3i(side, side, m_is_3d ? side : uint64(1));
    Voxelgrid sub_known_environment(side3);
    sub_known_environment.Fill(1.0f); // sub environment is all known-empty, unless global env says otherwise
    Eigen::Vector3f new_origin;
    {
      const Eigen::Vector3i i_origin = origin.cast<int>();
      const int64 i_max_range = max_range;

      // copy environment part to local sub-environment
      const Eigen::Vector3i i_max_range3 = i_max_range * Eigen::Vector3i(1, 1, m_is_3d ? 1 : 0);
      const Eigen::Vector3i origin_in = (i_origin - i_max_range3).array().max(0);
      const Eigen::Vector3i origin_out = (-(i_origin - i_max_range3)).array().max(0);
      const Eigen::Vector3i size_in = (origin_in + side3).array().min(env_side.array()) - origin_in.array();

      new_origin = i_max_range3.cast<float>();
      sub_known_environment.SetSubmatrix(origin_out,
                                         *known_environment.GetSubmatrix(origin_in, size_in));
    }

    m_opencl.ComputeGainFromViewCube(sub_known_environment, new_origin, resolution, max_range,
                                     nearest_dist_current_view, hits, miss);

    OriginVisibility & ov = result[ov_i];
    for (uint64 i = 0; i < cube_bearings.size(); i++)
    {
      if (cube_bearings[i] == Eigen::Vector3f::Zero())
        continue;
      ov.IntegrateObservation(cube_bearings[i], OriginVisibility::OBSTYPE_EMPTY, miss[i]);
      ov.IntegrateObservation(cube_bearings[i], OriginVisibility::OBSTYPE_UNKNOWN, hits[i]);
    }
  }


  return result;
}

OriginVisibilityVector GenerateSingleImage::InformationGainMultiViewCubes(const Voxelgrid & environment,
                                                                          const Voxelgrid & known_occupied,
                                                                          const Voxelgrid & known_empty,
                                                                          const Vector3fVector & origins,
                                                                          const float max_range,
                                                                          const float min_range,
                                                                          const bool stop_at_first_hit,
                                                                          const float a_priori_occupied_prob)
{
  OriginVisibilityVector result;
  for (const Eigen::Vector3f & origin : origins)
    result.push_back(OriginVisibility(origin, m_view_cube_resolution, m_is_3d));

  const float sensor_f = m_view_cube_resolution / 2.0f;
  const Eigen::Vector3i resolution = Eigen::Vector3i(m_view_cube_resolution,
                                                     m_view_cube_resolution,
                                                     m_is_3d ? m_view_cube_resolution : 1);

  const Vector3fVector cube_bearings = GenerateCubeRayBearings(resolution);

  Vector3fVector all_bearings;
  all_bearings.reserve(cube_bearings.size() * origins.size());
  for (uint64 i = 0; i < origins.size(); i++)
    all_bearings.insert(all_bearings.end(), cube_bearings.begin(), cube_bearings.end());
  Vector3fVector all_origins;
  all_origins.reserve(cube_bearings.size() * origins.size());
  for (uint64 i = 0; i < origins.size(); i++)
    for (uint64 h = 0; h < cube_bearings.size(); h++)
      all_origins.push_back(origins[i]);

  FloatVector hits;
  FloatVector miss;
  m_opencl.SimulateMultiRayWithInformationGainBatched(known_empty, known_occupied, all_origins, all_bearings,
                                                      sensor_f, max_range, min_range,
                                                      stop_at_first_hit, a_priori_occupied_prob,
                                                      hits, miss);

  for (uint64 ov_i = 0; ov_i < result.size(); ov_i++)
  {
    OriginVisibility & ov = result[ov_i];
    for (uint64 i = 0; i < cube_bearings.size(); i++)
    {
      const uint64 i2 = i + ov_i * cube_bearings.size();

      if (cube_bearings[i] == Eigen::Vector3f::Zero())
        continue;

      const float h = hits[i2];
      const float m = miss[i2];

      ov.IntegrateObservation(cube_bearings[i], OriginVisibility::OBSTYPE_OCCUPIED, m);
      ov.IntegrateObservation(cube_bearings[i], OriginVisibility::OBSTYPE_UNKNOWN, h);
    }
  }

  return result;
}

// fills a view, as estimated by the view cube
Voxelgrid::Ptr GenerateSingleImage::FillViewKnownUsingCubeResolution(const Voxelgrid & environment,
                                                                     const Eigen::Vector3f & origin,
                                                                     const Eigen::Quaternionf & bearing,
                                                                     const float max_range,
                                                                     Voxelgrid & observed_surface)
{
  observed_surface = *environment.FilledWith(0.0f);

  const Eigen::Vector3i resolution = Eigen::Vector3i(m_view_cube_resolution,
                                                     m_view_cube_resolution,
                                                     m_is_3d ? m_view_cube_resolution : 1);
  const Vector3fVector cube_bearings = GenerateCubeRayBearings(resolution);

  Vector3fVector origins(cube_bearings.size(), origin);

  FloatVector nearest_dist;
  Vector3iVector observed_points;
  m_opencl.SimulateMultiRayWI(environment, origins, cube_bearings, max_range, nearest_dist, observed_points);

  const float hfov_x = std::atan2(float(m_sensor_resolution_x) / 2.0f, float(m_sensor_focal_length));
  const float hfov_y = std::atan2(float(m_sensor_resolution_y) / 2.0f, float(m_sensor_focal_length));
  const Eigen::Vector2f hfov(hfov_x, hfov_y);

  Voxelgrid::Ptr result(environment.FilledWith(0.0f));

  m_opencl.FillEnvironmentFromViewCube(*result, origin, bearing, hfov, resolution,
                                       nearest_dist, observed_points, *result);

  {
    Voxelgrid environment_shrink = *environment.ErodeRect(Eigen::Vector3i::Ones());
    Voxelgrid observable_surface = *environment.AndNot(environment_shrink);

    observed_surface = *result->And(observable_surface);
  }

  result = result->AndNot(environment);

  return result;
}

Voxelgrid::Ptr GenerateSingleImage::FillViewKnown(const Voxelgrid & environment, const Eigen::Vector3f & origin,
                                                  const Eigen::Quaternionf & bearing,
                                                  const float sensor_f, const Eigen::Vector2i & sensor_resolution,
                                                  const float max_range,
                                                  Voxelgrid & observed_surface)
{
  Voxelgrid::Ptr result_ptr = environment.FilledWith(0.0f);

  observed_surface = *environment.FilledWith(0.0f);

  FloatVector nearest_dist(sensor_resolution.prod(), NAN);
  Vector3fVector ray_bearings(sensor_resolution.prod());
  const Vector3iVector nearest = SimulateView(environment, origin, bearing,
                                              sensor_f, sensor_resolution, max_range,
                                              nearest_dist, ray_bearings);

  for (uint64 i = 0; i < nearest.size(); i++)
  {
    if ((nearest[i].array() < 0.0f).any())
      continue;
    const Eigen::Vector3i & pt = nearest[i];
    if (!environment.at(pt))
      continue;
    observed_surface.at(pt) = 1.0f;
  }

  m_opencl.FillEnvironmentFromView(*result_ptr, origin, bearing, sensor_f, sensor_resolution,
                                   nearest_dist, nearest, *result_ptr);

  result_ptr = result_ptr->AndNot(environment);

  return result_ptr;
}
