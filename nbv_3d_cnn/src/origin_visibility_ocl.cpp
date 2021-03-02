#include <nbv_3d_cnn/origin_visibility.h>
#include <nbv_3d_cnn/generate_test_dataset_opencl.h>

Eigen::Quaternionf OriginVisibility::GetBestSensorOrientationOCL(GenerateTestDatasetOpenCL *opencl,
                                                                 const QuaternionfVector & orientations,
                                                                 const Eigen::Vector2f & hfov,
                                                                 float & best_gain,
                                                                 FloatVector & gains) const
{
  Eigen::Quaternionf best;
  best_gain = 0.0f;
  gains.resize(orientations.size());

  const Eigen::Vector2f tan_hfov = hfov.array().tan();

  FloatVector hits(orientations.size(), 0.0f);
  FloatVector miss(orientations.size(), 0.0f);

  Voxelgrid view_cube_hits(resolution, resolution, resolution);
  Voxelgrid view_cube_miss(resolution, resolution, resolution);

  ForeachIndex([&](const Eigen::Vector3i & index) -> bool
  {
    view_cube_hits.at(index) = virtual_frames[IndexToI(index)].hits;
    view_cube_miss.at(index) = virtual_frames[IndexToI(index)].miss;
    return true;
  });

  opencl->EvaluateSensorOrientationsOnViewCube(view_cube_hits, view_cube_miss,
                                               hfov, orientations,
                                               hits, miss);

  for (uint64 vi = 0; vi < orientations.size(); vi++)
  {
    const float h = hits[vi];
    const float m = miss[vi];
    const float gain = (h + m > 0.0f) ? (h / (h + m)) : 0.0f;
    gains[vi] = gain;

    if (vi == 0 || gain > best_gain)
    {
      best = orientations[vi];
      best_gain = gain;
    }
  }

  return best;
}
