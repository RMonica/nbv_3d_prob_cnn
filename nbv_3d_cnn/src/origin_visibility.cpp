#include <nbv_3d_cnn/origin_visibility.h>

OriginVisibility::OriginVisibility(const Eigen::Vector3f & p_pos, const uint64 res,
                                   const bool p_is_3d)
{
  pos = p_pos;
  is_3d = p_is_3d;
  resolution = res;
  focal_length = float(res) / 2.0f;

  if (is_3d)
    virtual_frames.resize(res * res * res);
  else
    virtual_frames.resize(res * res);
}

Eigen::Matrix3f OriginVisibility::GetAxisFromFrame(const TFrame frame) const
{
  Eigen::Matrix3f result;

  switch (frame)
  {
    case FRAME_TOP:
      result.col(0) = Eigen::Vector3f(-1.0f, 0.0f, 0.0f);
      result.col(1) = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
      result.col(2) = Eigen::Vector3f(0.0f, -1.0f, 0.0f);
      break;
    case FRAME_LEFT:
      result.col(0) = Eigen::Vector3f( 0.0f, 1.0f, 0.0f);
      result.col(1) = Eigen::Vector3f( 0.0f, 0.0f,-1.0f);
      result.col(2) = Eigen::Vector3f(-1.0f, 0.0f, 0.0f);
      break;
    case FRAME_BOTTOM:
      result.col(0) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
      result.col(1) = Eigen::Vector3f(0.0f, 0.0f,-1.0f);
      result.col(2) = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
      break;
    case FRAME_RIGHT:
      result.col(2) = Eigen::Vector3f(0.0f,-1.0f, 0.0f);
      result.col(2) = Eigen::Vector3f(0.0f, 0.0f,-1.0f);
      result.col(2) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
      break;
    case FRAME_FAR:
      result.col(0) = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
      result.col(1) = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
      result.col(2) = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
      break;
    case FRAME_NEAR:
      result.col(0) = Eigen::Vector3f(-1.0f, 0.0f, 0.0f);
      result.col(1) = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
      result.col(2) = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
      break;
    default:
      ROS_FATAL("generate_test_dataset: unknown frame type %u", unsigned(frame));
      exit(1);
  }

  return result;
}

OriginVisibility::TFrame OriginVisibility::BearingToFrame(const Eigen::Vector3f & bearing) const
{
  float min_dist;
  TFrame result;
  for (uint64 i = 0; i < FRAME_MAX; i++)
    if (i == 0 || bearing.dot(GetAxisFromFrame(TFrame(i)).col(2)) > min_dist)
    {
      min_dist = bearing.dot(GetAxisFromFrame(TFrame(i)).col(2));
      result = TFrame(i);
    }

  return result;
}

Eigen::Vector3i OriginVisibility::BearingToIndex(const Eigen::Vector3f & bearing) const
{
  const Eigen::Vector3f ones_v(1.0f, 1.0f, is_3d ? 1.0f : 0.0f);

  uint64 max_index;
  for (uint64 i = 0; i < 3; i++)
    if (i == 0 || std::abs(bearing[i]) > std::abs(bearing[max_index]))
      max_index = i;

  const Eigen::Vector3f normalized_bearing = bearing / std::abs(bearing[max_index]);
  const Eigen::Vector3i index = ((normalized_bearing + ones_v) *
                                 resolution / 2.0f).array().floor().cast<int>().
                                 max(0).min(resolution - 1);

  return index;
}

Eigen::Vector3f OriginVisibility::IndexToBearing(const Eigen::Vector3i index) const
{
  const float matrix_center = resolution / 2.0f - 0.5f;
  const Eigen::Vector3f matrix_center_v(matrix_center, matrix_center, is_3d ? matrix_center : 0.0f);

  const Eigen::Vector3f bearing = index.cast<float>() - matrix_center_v;

  return bearing.normalized();
}

void OriginVisibility::IntegrateObservation(const Eigen::Quaternionf &orientation, const TObsType type,
                                            const double count)
{
  const Eigen::Vector3f bearing = orientation * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  IntegrateObservation(bearing, type, count);
}

void OriginVisibility::IntegrateObservation(const Eigen::Vector3f &bearing, const TObsType type, const double count)
{
  if (bearing == Eigen::Vector3f::Zero())
    return;

  const Eigen::Vector3i index = BearingToIndex(bearing);

  if (type == OBSTYPE_OCCUPIED || type == OBSTYPE_EMPTY)
    virtual_frames[IndexToI(index)].miss += count;
  if (type == OBSTYPE_UNKNOWN)
    virtual_frames[IndexToI(index)].hits += count;
}

Eigen::Vector3i OriginVisibility::GetClockwiseIndexFrame(const Eigen::Vector3i & index) const
{
  const int resolution1 = resolution - 1;

  if (index.y() == 0 && index.x() < resolution1)
    return index + Eigen::Vector3i::UnitX();
  if (index.x() == resolution1 && index.y() < resolution1)
    return index + Eigen::Vector3i::UnitY();
  if (index.y() == resolution1 && index.x() > 0)
    return index - Eigen::Vector3i::UnitX();
  if (index.x() == 0 && index.y() > 0)
    return index - Eigen::Vector3i::UnitY();

  ROS_ERROR_STREAM("GetClockwiseIndexFrame: huh? " << index.transpose());
  exit(1);
}

Eigen::Vector3i OriginVisibility::GetCounterClockwiseIndexFrame(const Eigen::Vector3i & index) const
{
  const int resolution1 = resolution - 1;

  if (index.y() == 0 && index.x() > 0)
    return index - Eigen::Vector3i::UnitX();
  if (index.x() == 0 && index.y() < resolution1)
    return index + Eigen::Vector3i::UnitY();
  if (index.y() == resolution1 && index.x() < resolution1)
    return index + Eigen::Vector3i::UnitX();
  if (index.x() == resolution1 && index.y() > 0)
    return index - Eigen::Vector3i::UnitY();

  ROS_ERROR_STREAM("GetCounterClockwiseIndexFrame: huh? " << index.transpose());
  exit(1);
}

bool OriginVisibility::IsValidIndex(const Eigen::Vector3i & index) const
{
  const int resolution1 = resolution - 1;
  if ((index.array() < 0).any())
    return false;
  if ((index.array() >= resolution).any())
    return false;

  if ((index.array() == 0).any())
    return true;
  if ((index.head<2>().array() == resolution1).any())
    return true;
  if (is_3d && index.z() == resolution1)
    return true;
  return false;
}

void OriginVisibility::ForeachIndex(const std::function<bool(const Eigen::Vector3i & index)> & f) const
{
  for (uint64 z = 0; z < (is_3d ? resolution : 1); z++)
    for (uint64 y = 0; y < resolution; y++)
      for (uint64 x = 0; x < resolution; x++)
      {
        const Eigen::Vector3i index(x, y, z);
        if (!IsValidIndex(index))
          continue;

        if (!f(index))
          break;
      }
}

float OriginVisibility::AverageVisibility() const
{
  float sum = 0.0f;
  uint64 count = 0;

  ForeachIndex([&](const Eigen::Vector3i & index) -> bool
  {
    const PixelInfo & v = virtual_frames[IndexToI(index)];
    float visibility = 0.0f;
    if (v.miss + v.hits > 0)
      visibility = (v.hits) / (v.miss + v.hits);
    sum += visibility;
    count++;
    return true;
  });

  return sum / float(count);
}

cv::Mat OriginVisibility::GetVisibilityMatrix(const uint64 res_out) const
{
  cv::Mat result = cv::Mat(res_out, res_out, CV_32FC1);
  result = 0.0f;
  cv::Mat counters = cv::Mat(res_out, res_out, CV_32FC1);
  counters = 0;

  ForeachIndex([&](const Eigen::Vector3i & index) -> bool
  {
    const float hits = virtual_frames[IndexToI(index)].hits;
    const float miss = virtual_frames[IndexToI(index)].miss;

    const Eigen::Vector3i result_pixel = index * res_out / resolution;

    if ((result_pixel.array() < 0).any() || (result_pixel.array() >= res_out).any())
      return true; // out of the matrix

    result.at<float>(result_pixel.y(), result_pixel.x()) += hits;
    counters.at<float>(result_pixel.y(), result_pixel.x()) += hits + miss;
    return true;
  });

  counters = cv::max(counters, 1.0f); // prevent division by zero

  result /= counters;

  return result;
}

Voxelgrid OriginVisibility::GetVisibilityVoxelgrid(const uint64 res_out) const
{
  const Eigen::Vector3i ones_out(res_out, res_out, is_3d ? res_out : 1);
  Voxelgrid result(ones_out);
  result.Fill(0.0f);
  Voxelgrid counters(ones_out);
  counters.Fill(0.0f);

  ForeachIndex([&](const Eigen::Vector3i & index) -> bool
  {
    const float hits = virtual_frames[IndexToI(index)].hits;
    const float miss = virtual_frames[IndexToI(index)].miss;

    const Eigen::Vector3i result_pixel = index * res_out / resolution;
    if ((result_pixel.array() < 0).any() || (result_pixel.array() >= res_out).any())
      return true;

    result.at(result_pixel) += hits;
    counters.at(result_pixel) += hits + miss;

    return true;
  });

  counters.Max(1.0f); // prevent division by zero

  result.DivideBy(counters);

  return result;
}

OriginVisibility OriginVisibility::FromVoxelgrid(const Eigen::Vector3f & p_pos,
                                                 const uint64 &res, const Voxelgrid & voxelgrid)
{
  const uint64 input_res = voxelgrid.GetWidth();
  const Eigen::Vector3i input_size = voxelgrid.GetSize();

  OriginVisibility result(p_pos, res, true);

  if (input_res < res) // upscaling
  {
    result.ForeachIndex([&](const Eigen::Vector3i & index) -> bool
    {
      const Eigen::Vector3i matrix_pixel = index * input_res / result.resolution;
      if ((matrix_pixel.array() < 0).any() || (matrix_pixel.array() >= input_res).any())
        return true;

      const float v = voxelgrid.at(matrix_pixel);

      result.virtual_frames[result.IndexToI(index)].hits += v;
      result.virtual_frames[result.IndexToI(index)].miss += 1.0f - v;

      return true;
    });
  }
  else // downscaling
  {
    for (uint64 z = 0; z < input_size.z(); z++)
      for (uint64 y = 0; y < input_size.y(); y++)
        for (uint64 x = 0; x < input_size.x(); x++)
        {
          if (y != 0 && x != 0 && y != (input_res - 1) && x != (input_res - 1))
            continue; // only on the outer circle

          const Eigen::Vector3i matrix_pixel(x, y, z);
          const Eigen::Vector3i index = matrix_pixel * result.resolution / input_res;

          const float v = voxelgrid.at(matrix_pixel);

          result.virtual_frames[result.IndexToI(index)].hits += v;
          result.virtual_frames[result.IndexToI(index)].miss += 1.0f - v;
        }
  }

  return result;
}

OriginVisibility OriginVisibility::FromVisibilityMatrix(const Eigen::Vector3f & p_pos,
                                                        const uint64 & res, const cv::Mat matrix)
{
  const uint64 input_res = matrix.rows;

  if (matrix.type() == CV_32FC1)
    matrix.convertTo(matrix, CV_8UC1, 255.0f);

  OriginVisibility result(p_pos, res, false);

  if (input_res < res) // upscaling
  {
    result.ForeachIndex([&](const Eigen::Vector3i & index) -> bool
    {
      const Eigen::Vector3i matrix_pixel = index * input_res / result.resolution;
      if ((matrix_pixel.array() < 0).any() || (matrix_pixel.array() >= input_res).any())
        return true;

      const uint8 v = matrix.at<uint8>(matrix_pixel.y(), matrix_pixel.x());

      result.virtual_frames[result.IndexToI(index)].hits += v;
      result.virtual_frames[result.IndexToI(index)].miss += 255 - v;

      return true;
    });
  }
  else // downscaling
  {
    for (uint64 y = 0; y < input_res; y++)
      for (uint64 x = 0; x < input_res; x++)
      {
        if (y != 0 && x != 0 && y != (input_res - 1) && x != (input_res - 1))
          continue; // only on the outer circle

        const Eigen::Vector3i matrix_pixel(x, y, 0);
        const Eigen::Vector3i index = matrix_pixel * result.resolution / input_res;

        const uint8 v = matrix.at<uint8>(matrix_pixel.y(), matrix_pixel.x());

        result.virtual_frames[result.IndexToI(index)].hits += v;
        result.virtual_frames[result.IndexToI(index)].miss += 255 - v;
      }
  }

  return result;
}

OriginVisibility OriginVisibility::SmoothByHFOV(const float hfov) const
{
  OriginVisibility result(pos, resolution, is_3d);

  ForeachIndex([&](const Eigen::Vector3i & index) -> bool
  {
    const Eigen::Vector3f initial_bearing = IndexToBearing(index);

    result.virtual_frames[IndexToI(index)].hits += virtual_frames[IndexToI(index)].hits;
    result.virtual_frames[IndexToI(index)].miss += virtual_frames[IndexToI(index)].miss;

    if (index.x() != 0 && index.y() != 0 &&
        index.x() != resolution-1 && index.y() != resolution-1)
      return true;

    // expand clockwise (-1) and counter-clockwise (+1)
    for (int64 sign = -1; sign <= 1; sign+=2)
    {
      Eigen::Vector3i index_frame = index;
      for (uint64 cwise = 1; cwise < resolution * 4; cwise++)
      {
        if (sign == -1)
          index_frame = GetClockwiseIndexFrame(index_frame);
        else
          index_frame = GetCounterClockwiseIndexFrame(index_frame);

        const Eigen::Vector3f bearing = IndexToBearing(index_frame);

        if (bearing.dot(initial_bearing) < std::cos(hfov))
          break; // out of the cone

        result.virtual_frames[IndexToI(index_frame)].hits +=
          virtual_frames[IndexToI(index)].hits;
        result.virtual_frames[IndexToI(index_frame)].miss +=
          virtual_frames[IndexToI(index)].miss;
      }
    }

    return true;
  });

  return result;
}

float OriginVisibility::MaxVisibility(Eigen::Vector3i &index, Eigen::Vector3f & bearing) const
{
  index = Eigen::Vector3i::Zero();
  float result = -1.0;

  ForeachIndex([&](const Eigen::Vector3i & index_i) -> bool
  {
    const float v = virtual_frames[IndexToI(index_i)].GetVisibility();
    if (v > result)
    {
      index = index_i;
      result = v;
    }

    return true;
  });

  bearing = IndexToBearing(index);

  return result;
}

float OriginVisibility::MaxVisibility(Eigen::Vector3i &index, Eigen::Quaternionf & orientation) const
{
  Eigen::Vector3f bearing;
  const float result = MaxVisibility(index, bearing);

  Eigen::Vector3f far_axis = Eigen::Vector3f::Unit(0);
  for (uint64 i = 1; i < 3; i++)
    if (std::abs(far_axis.dot(bearing)) > std::abs(Eigen::Vector3f::Unit(i).dot(bearing)))
      far_axis = Eigen::Vector3f::Unit(i);

  Eigen::Matrix3f rot;
  rot.col(2) = bearing;
  rot.col(1) = (far_axis - bearing * far_axis.dot(bearing)).normalized();
  rot.col(0) = rot.col(1).cross(rot.col(2));
  orientation = Eigen::Quaternionf(rot);

  return result;
}

std::string OriginVisibility::ToString() const
{
  std::ostringstream ostr;
  ostr << "OV " << resolution << " " << resolution << " " << (is_3d ? resolution : 1) << "\n";
  ostr << "Origin " << pos.transpose() << "\n";
  for (uint64 z = 0; z < (is_3d ? resolution : 1); z++)
  {
    ostr << "Depth " << z << "\n";
    for (uint64 y = 0; y < resolution; y++)
    {
      for (uint64 x = 0; x < resolution; x++)
      {
        ostr << virtual_frames[IndexToI(Eigen::Vector3i(x, y, z))].hits << "/" <<
                virtual_frames[IndexToI(Eigen::Vector3i(x, y, z))].miss << " ";
      }
      ostr << "\n";
    }
  }
  ostr << "VO\n";

  return ostr.str();
}

OriginVisibility OriginVisibility::SmoothByHFOV3D(const Eigen::Vector2f & hfov) const
{
  const uint64 sampling = resolution;
  OriginVisibility result(pos, resolution, is_3d);

  ForeachIndex([&](const Eigen::Vector3i & index_i) -> bool
  {
    const Eigen::Vector3f bearing = IndexToBearing(index_i);
    const Eigen::Quaternionf rotation1 = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitZ(), bearing);
    double hits = 0.0;
    double miss = 0.0;
    for (uint64 i = 0; i < resolution; i++)
    {
      const float angle = i * 2.0f * M_PI / resolution;

      const Eigen::Quaternionf rotation2(Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitZ()));
      const Eigen::Quaternionf rotation = rotation1 * rotation2;
      double hhits;
      double hmiss;
      GetGainAtSensorOrientation(rotation, hfov, hhits, hmiss);
      hits += hhits;
      miss += hmiss;
    }
    hits /= sampling;
    miss /= sampling;

    result.IntegrateObservation(bearing, OBSTYPE_EMPTY, miss);
    result.IntegrateObservation(bearing, OBSTYPE_UNKNOWN, hits);
    
    return true;
  });

  return result;
}

Eigen::Quaternionf OriginVisibility::GetBestSensorOrientationHighResolution(const QuaternionfVector &orientations,
                                                                            const Eigen::Vector2f & hfov,
                                                                            float & best_gain,
                                                                            FloatVector & gains) const
{
  Eigen::Quaternionf best = orientations[0];
  best_gain = 0.0f;
  gains.clear();

  for (const Eigen::Quaternionf & orient : orientations)
  {
    double hits, miss;
    const double gain = GetGainAtSensorOrientation(orient, hfov, hits, miss);
    gains.push_back(gain);
    if (gain > best_gain)
    {
      best = orient;
      best_gain = gain;
    }
  }

  return best;
}

Eigen::Quaternionf OriginVisibility::GetBestSensorOrientationManyViews(const QuaternionfVector &orientations,
                                                                       const Eigen::Vector2f & hfov,
                                                                       float & best_gain,
                                                                       FloatVector & gains) const
{
  Eigen::Quaternionf best;
  best_gain = 0.0f;
  gains.clear();

  const Eigen::Vector2f tan_hfov = hfov.array().tan();

  Matrix3fVector orientation_mats;

  for (const Eigen::Quaternionf & orient : orientations)
  {
    const Eigen::Vector3f fwd = orient * Eigen::Vector3f(0, 0, 1);
    const Eigen::Vector3f right = orient * Eigen::Vector3f(1, 0, 0);
    const Eigen::Vector3f down = orient * Eigen::Vector3f(0, 1, 0);
    Eigen::Matrix3f mat;
    mat.col(2) = fwd;
    mat.col(0) = right;
    mat.col(1) = down;

    orientation_mats.push_back(mat);
  }

  FloatVector hits(orientations.size(), 0.0f);
  FloatVector miss(orientations.size(), 0.0f);

  ForeachIndex([&](const Eigen::Vector3i & index) -> bool
  {
    for (uint64 vi = 0; vi < orientations.size(); vi++)
    {
      if (!IsIndexWithinOrientationMatrix(tan_hfov, orientation_mats[vi], index))
        continue;

      hits[vi] += virtual_frames[IndexToI(index)].hits;
      miss[vi] += virtual_frames[IndexToI(index)].miss;
    }

    return true;
  });

  for (uint64 vi = 0; vi < orientations.size(); vi++)
  {
    const float h = hits[vi];
    const float m = miss[vi];
    const float gain = (h + m > 0.0f) ? (h / (h + m)) : 0.0f;
    gains.push_back(gain);

    if (vi == 0 || gain > best_gain)
    {
      best = orientations[vi];
      best_gain = gain;
    }
  }

  return best;
}

OriginVisibility::Vector3iVector OriginVisibility::GetVoxelNeighborhood(const Eigen::Vector3i & center) const
{
  Vector3iVector result;

  Eigen::Vector3i di;
  for (di.z() = -1; di.z() <= 1; di.z()++)
    for (di.y() = -1; di.y() <= 1; di.y()++)
      for (di.x() = -1; di.x() <= 1; di.x()++)
      {
        if ((di.array() == 0).all())
          continue;

        const Eigen::Vector3i ni = di + center;
        if (IsValidIndex(ni))
          result.push_back(ni);
      }

  return result;
}

bool OriginVisibility::IsIndexWithinOrientationMatrix(const Eigen::Vector2f & tan_hfov,
                                                      const Eigen::Matrix3f & mat,
                                                      const Eigen::Vector3i & index) const
{
  const Eigen::Vector3f fwd = mat.col(2);
  const Eigen::Vector3f right = mat.col(0);
  const Eigen::Vector3f down = mat.col(1);

  const Eigen::Vector3f nb = IndexToBearing(index);

  if (nb.dot(fwd) <= 0.0f)
    return false; // behind sensor

  if (std::abs(nb.dot(right)) > tan_hfov.x() * nb.dot(fwd))
    return false; // out of sensor horizontal fov

  if (std::abs(nb.dot(down)) > tan_hfov.y() * nb.dot(fwd))
    return false; // out of sensor vertical fov

  return true;
}

bool OriginVisibility::IsIndexWithinOrientation(const Eigen::Vector2f & tan_hfov,
                                                const Eigen::Quaternionf & orientation,
                                                const Eigen::Vector3i & index) const
{
  const Eigen::Vector3f fwd = orientation * Eigen::Vector3f(0, 0, 1);
  const Eigen::Vector3f right = orientation * Eigen::Vector3f(1, 0, 0);
  const Eigen::Vector3f down = orientation * Eigen::Vector3f(0, 1, 0);
  Eigen::Matrix3f mat;
  mat.col(2) = fwd;
  mat.col(0) = right;
  mat.col(1) = down;

  return IsIndexWithinOrientationMatrix(tan_hfov, mat, index);
}

float OriginVisibility::GetGainAtSensorOrientation(const Eigen::Quaternionf & orientation,
                                                   const Eigen::Vector2f & hfov,
                                                   double & hits, double & miss) const
{
  hits = 0.0;
  miss = 0.0;

  BoolVector visited;
  if (is_3d)
    visited.resize(resolution * resolution * resolution, false);
  else
    visited.resize(resolution * resolution, false);

  const Eigen::Vector2f tan_hfov = hfov.array().tan();

  const Eigen::Vector3f initial_bearing = orientation * Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  const Eigen::Vector3i initial_i = BearingToIndex(initial_bearing);

  hits += virtual_frames[IndexToI(initial_i)].hits;
  miss += virtual_frames[IndexToI(initial_i)].miss;
  visited[IndexToI(initial_i)] = true;

  Vector3iDeque queue;
  queue.push_back(initial_i);

  // expand in the neighborhood and sum everything in the sensor FOV
  while (!queue.empty())
  {
    const Eigen::Vector3i ci = queue.front();
    queue.pop_front();

    const Vector3iVector neighbors = GetVoxelNeighborhood(ci);
    for (const Eigen::Vector3i & ni : neighbors)
    {
      if (visited[IndexToI(ni)])
        continue;
      visited[IndexToI(ni)] = true;

      if (!IsIndexWithinOrientation(tan_hfov, orientation, ni))
        continue;

      hits += virtual_frames[IndexToI(ni)].hits;
      miss += virtual_frames[IndexToI(ni)].miss;

      queue.push_back(ni);
    }
  }

  if (hits + miss == 0.0)
    return 0.0;
  return hits / (hits + miss);
}

OriginVisibility::QuaternionfVector OriginVisibility::GenerateStandardOrientationSet(const uint64 num_angles)
{
  const uint64 horizontal_resolution = num_angles;
  const int64 vertical_resolution = num_angles / 2 - 1;
  const uint64 zrot_resolution = num_angles / 4;

  QuaternionfVector orientations;

  for (int64 zrot = 0; zrot < zrot_resolution; zrot++)
  {
    for (uint64 i = 0; i < horizontal_resolution; i++)
      for (int64 h = -(vertical_resolution/2); h <= (vertical_resolution/2); h++)
      {
        const Eigen::Quaternionf orient(
              Eigen::AngleAxisf(float(i) / horizontal_resolution * 2.0f * M_PI, Eigen::Vector3f::UnitZ()) *
              Eigen::AngleAxisf(float(h) / (vertical_resolution/2) * (M_PI / 2.0f), Eigen::Vector3f::UnitY()) *
              Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitY()) *
              Eigen::AngleAxisf(float(zrot) / (zrot_resolution) * (M_PI), Eigen::Vector3f::UnitZ())
              );
        orientations.push_back(orient);
      }
    const Eigen::Quaternionf orient_top(
          Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(float(zrot) / (zrot_resolution) * (M_PI), Eigen::Vector3f::UnitZ())
          );
    orientations.push_back(orient_top);
    const Eigen::Quaternionf orient_bottom(
          Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()) *
          Eigen::AngleAxisf(float(zrot) / (zrot_resolution) * (M_PI), Eigen::Vector3f::UnitZ())
          );
    orientations.push_back(orient_bottom);
  }

  return orientations;
}
