#ifndef __OPENCL_VERSION__
  #define global
  #define __global
  #define kernel
#endif

#define SQR(x) ((x)*(x))

#ifndef NULL
  #define NULL (0)
#endif

bool Equal2(int2 a, int2 b)
{
  return a.x == b.x && a.y == b.y;
}

bool Equal3(int3 a, int3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

bool Equal3f(float3 a, float3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

int2 FloatToInt2(float2 f)
{
  return (int2)(f.x, f.y);
}

int2 UintToInt2(uint2 f)
{
  return (int2)(f.x, f.y);
}

float3 UintToFloat3(uint3 f)
{
  return (float3)(f.x, f.y, f.z);
}

int3 FloatToInt3(float3 f)
{
  return (int3)(f.x, f.y, f.z);
}

void SetElem3f(float3 * v, int index, float value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

float GetElem3f(const float3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

void SetElem3i(int3 * v, int index, int value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

int GetElem3i(const int3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

void SetElem3u(uint3 * v, int index, uint value)
{
  switch (index)
  {
    case 0: v->x = value; break;
    case 1: v->y = value; break;
    case 2: v->z = value; break;
  }
}

uint GetElem3u(const uint3 v, int index)
{
  switch (index)
  {
    case 0: return v.x;
    case 1: return v.y;
    case 2: return v.z;
  }

  return v.x;
}

float4 quat_Inverse(const float4 q)
{
  return (float4)(-q.xyz, q.w);
}

float4 quat_Mult(const float4 q1, const float4 q2)
{
  float4 q;
  q.x = (q1.w * q2.x) + (q1.x * q2.w) + (q1.y * q2.z) - (q1.z * q2.y);
  q.y = (q1.w * q2.y) - (q1.x * q2.z) + (q1.y * q2.w) + (q1.z * q2.x);
  q.z = (q1.w * q2.z) + (q1.x * q2.y) - (q1.y * q2.x) + (q1.z * q2.w);
  q.w = (q1.w * q2.w) - (q1.x * q2.x) - (q1.y * q2.y) - (q1.z * q2.z);

  return q;
}

float3 quat_ApplyRotation(const float4 q, const float3 v)
{
  const float4 ev = (float4)(v.xyz, 0.0f);
  const float4 iq = quat_Inverse(q);
  const float4 result = quat_Mult(quat_Mult(q, ev), iq);
  return result.xyz;
}

int3 BearingToCubeCoord(const uint3 view_cube_resolution, const float3 bearing)
{
  int max_coord;
  const float3 norm_diff = normalize(bearing);
  #pragma unroll
  for (int i = 0; i < 3; i++)
    if (i == 0 || fabs(GetElem3f(norm_diff, i)) > fabs(GetElem3f(norm_diff, max_coord)))
      max_coord = i;
  const float3 max_c_norm_diff = norm_diff / fabs(GetElem3f(norm_diff, max_coord)) + (float3)(1.0f, 1.0f, 1.0f);
  const float3 cube_f_coord = max_c_norm_diff * UintToFloat3(view_cube_resolution) / 2.0f;
  int3 cube_coord = FloatToInt3(floor(cube_f_coord));
  #pragma unroll
  for (int i = 0; i < 3; i++)
    if (GetElem3i(cube_coord, i) < 0)
      SetElem3i(&cube_coord, i, 0);
  #pragma unroll
  for (int i = 0; i < 3; i++)
    if (GetElem3i(cube_coord, i) >= GetElem3u(view_cube_resolution, i))
      SetElem3i(&cube_coord, i, GetElem3u(view_cube_resolution, i) - 1);

  return cube_coord;
}

void kernel SimulateMultiRay(global const ushort * environment, const uint width, const uint height, const uint depth,
                             const uint rays_size,
                             global const float * origins, global const float * orientations, const float max_range,
                             global float * distances, global int3 * observed_points)
{
  const int ray_id = get_global_id(0);

  const float3 origin = (float3)(origins[ray_id], origins[ray_id + rays_size],
                                 origins[ray_id + 2*rays_size]);

  const float3 ray_bearing = (float3)(orientations[ray_id], orientations[ray_id + rays_size],
                                      orientations[ray_id + 2*rays_size]);

  if (Equal3f(ray_bearing, (float3)(0, 0, 0)))
  {
    distances[ray_id] = 0.0f;
    observed_points[ray_id] = (int3)(-1, -1, -1);
    return; // invalid
  }

  distances[ray_id] = max_range;
  observed_points[ray_id] = (int3)(-1, -1, -1);

  uint max_range_i = (uint)max_range;
  for (uint z = 0; z < max_range_i; z++)
  {
    float3 pt = ray_bearing * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);
    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      distances[ray_id] = length(rpt - origin) + 0.5f;
      return;
    }

    if (environment[ipt.z * width * height + ipt.y * width + ipt.x])
    {
      distances[ray_id] = length(rpt - origin) + 0.5f;
      observed_points[ray_id] = ipt;
      return;
    }
  }
}

void kernel SimulateMultiRayWithInformationGain(global const float * occupied_environment,
                                                global const float * empty_environment,
                                                const uint width,
                                                const uint height,
                                                const uint depth,
                                                const uint rays_size,
                                                const float sensor_focal_length,
                                                global const float * origins,
                                                global const float * orientations,
                                                const float max_range,
                                                const float min_range,
                                                const uchar stop_at_first_hit,
                                                const float a_priori_occupied_prob,
                                                global float * hits,
                                                global float * miss
                                                )
{
  const int ray_id = get_global_id(0);

  const float3 origin = (float3)(origins[ray_id], origins[ray_id + rays_size],
                                 origins[ray_id + 2 * rays_size]);

  const float3 ray_bearing = (float3)(orientations[ray_id], orientations[ray_id + rays_size],
                                      orientations[ray_id + 2 * rays_size]);

  if (Equal3f(ray_bearing, (float3)(0, 0, 0)))
  {
    hits[ray_id] = 0.0f;
    miss[ray_id] = 1.0f;
    return; // invalid
  }

  hits[ray_id] = 0.0f;
  miss[ray_id] = 0.0f;

  uint max_range_i = (uint)max_range;
  float occluded = 0.0f;
  for (uint z = 0; z < max_range_i; z++)
  {
    float3 pt = ray_bearing * (float)(z) + origin;
    float3 rpt = round(pt);
    int3 ipt = FloatToInt3(rpt);

    float dimensions = (depth <= 1) ? 1.0 : 2.0;
    float pw = pow((float)(z) / sensor_focal_length, dimensions);
    float distance_weight = min(pw, 1.0f);

    float oe, ee;
    if (ipt.x < 0 || ipt.y < 0 || ipt.z < 0 ||
        ipt.x >= width || ipt.y >= height || ipt.z >= depth)
    {
      oe = 0.0;
      ee = 1.0;
    }
    else
    {
      oe = occupied_environment[ipt.z * width * height + ipt.y * width + ipt.x];
      ee = empty_environment[ipt.z * width * height + ipt.y * width + ipt.x];
    }

    if (stop_at_first_hit && ee < 0.5)
    {
      if (oe > 0.5 )
        miss[ray_id] += 1.0;
      else if (z < min_range)
      {
        //hits[ray_id] += distance_weight * z / min_range;
        hits[ray_id] += 0.01f;
        //miss[ray_id] += 1.0f;
        miss[ray_id] += 0.99f;
      }
      else
      {
        hits[ray_id] += distance_weight;
        miss[ray_id] += 1.0f - distance_weight;
      }

      return;
    }

    float prob_unknown = (1.0f - oe - ee);
    float prob_unknown_and_reachable = prob_unknown * (1.0 - occluded);
    hits[ray_id] += distance_weight * prob_unknown_and_reachable;
    miss[ray_id] += distance_weight * (1.0f - prob_unknown_and_reachable);

    float prob_occluding_if_unknown = prob_unknown * a_priori_occupied_prob;
    float prob_occluding_if_occupied = oe;
    // new_reachable = old_reachable * empty_prob
    // ->
    // (1 - new_occluded) = (1 - old_occluded) * (1 - occupied_prob)
    occluded = 1.0f - (1.0f - occluded) * (1.0f - prob_occluding_if_occupied - prob_occluding_if_unknown);
  }
}

void kernel FillEnvironmentFromView(const uint width, const uint height, const uint depth,
                                    const float3 origin, const float4 orientation,
                                    const uint2 sensor_resolution, const float sensor_f,
                                    global const float * nearest_dist,
                                    global const int3 * observed_points,
                                    global ushort * filled_environment
                                    )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  if (filled_environment[x + y * width + z * width * height])
    return; // already set

  const float3 cell = (float3)(x, y, z);
  const float3 diff = cell - origin;

  const float3 fwd = quat_ApplyRotation(orientation, (float3)(0, 0, 1));
  const float3 down = quat_ApplyRotation(orientation, (float3)(0, 1, 0));
  const float3 right = quat_ApplyRotation(orientation, (float3)(1, 0, 0));

  if (dot(diff, fwd) < 0.0f)
    return; // wrong orientation

  const float cell_sqr_distance = dot(diff, diff);
  const float cell_fwd_distance = dot(diff, fwd);
  const float cell_right_distance = dot(diff, right);
  const float cell_down_distance = dot(diff, down);

  const int2 sensor_pixel = (int2)(floor(cell_right_distance / cell_fwd_distance * sensor_f +
                                   (float)(sensor_resolution.x) / 2.0f),
                                   floor(cell_down_distance / cell_fwd_distance * sensor_f +
                                   (float)(sensor_resolution.y) / 2.0f)
                                  );
  if (any(sensor_pixel < 0) || any(sensor_pixel >= UintToInt2(sensor_resolution)))
    return;

  uint pixel_i = sensor_pixel.x + sensor_pixel.y * sensor_resolution.x;

  if (nearest_dist[pixel_i] < 0.0f)
    return;

  if (cell_sqr_distance > pow(nearest_dist[pixel_i], 2.0f))
    return;

  const int3 observed_point = observed_points[pixel_i];

  if (Equal3(observed_point, (int3)(x, y, z)))
    return; // occupied is not empty

  filled_environment[x + y * width + z * width * height] = 255;
}

void kernel FillEnvironmentFromViewCube(const uint width,
                                        const uint height,
                                        const uint depth,
                                        const float3 origin,
                                        const float4 orientation,
                                        const float2 sensor_tan_hfov,
                                        const uint3 view_cube_resolution,
                                        global const float * nearest_dist,
                                        global const int2 * observed_points,
                                        global ushort * filled_environment)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  if (filled_environment[x + y * width + z * width * height])
    return; // already set

  const float3 cell = (float3)(x, y, z);
  const float3 diff = cell - origin;
  const float3 norm_diff = normalize(diff);

  if (length(diff) < 1.0f)
  {
    filled_environment[x + y * width + z * width * height] = 255;
    return;
  }

  const float3 fwd = quat_ApplyRotation(orientation, (float3)(0, 0, 1));
  const float3 right = quat_ApplyRotation(orientation, (float3)(1, 0, 0));
  const float3 down = quat_ApplyRotation(orientation, (float3)(0, 1, 0));

  if (dot(norm_diff, fwd) <= 0.0f)
    return; // behind sensor

  if (fabs(dot(diff, right)) > sensor_tan_hfov.x * dot(diff, fwd))
    return; // out of sensor horizontal fov

  if (fabs(dot(diff, down)) > sensor_tan_hfov.y * dot(diff, fwd))
    return; // out of sensor vertical fov

  const float cell_sqr_distance = dot(diff, diff);

  int3 cube_coord = BearingToCubeCoord(view_cube_resolution, norm_diff);

  const uint pixel_i = cube_coord.x + cube_coord.y * view_cube_resolution.x +
                       cube_coord.z * view_cube_resolution.x * view_cube_resolution.y;
  if (nearest_dist[pixel_i] < 0.0f)
    return;

  if (cell_sqr_distance > pow(nearest_dist[pixel_i], 2.0f))
    return;

  filled_environment[x + y * width + z * width * height] = 255;
}

void kernel ComputeGainFromViewCube(global const ushort * environment,
                                    const uint width, const uint height, const uint depth,
                                    const float3 origin,
                                    const uint3 view_cube_resolution,
                                    const float max_range,
                                    global const float * nearest_dist,
                                    global uint * hits,
                                    global uint * miss
                                    )
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);
  const int z = get_global_id(2);

  const float3 cell = (float3)(x, y, z);
  const float3 diff = cell - origin;

  const float3 norm_diff = normalize(diff);

  const int3 cube_coord = BearingToCubeCoord(view_cube_resolution, norm_diff);

  const float cell_sqr_distance = dot(diff, diff);

  if (cell_sqr_distance > SQR(max_range))
    return;

  const uint pixel_i = cube_coord.x + cube_coord.y * view_cube_resolution.x +
                       cube_coord.z * view_cube_resolution.x * view_cube_resolution.y;

  if (nearest_dist[pixel_i] < 0.0f)
    return;

  if (cell_sqr_distance >= SQR(nearest_dist[pixel_i]))
  {
    atomic_inc(&(miss[pixel_i]));
    return; // occluded
  }

  if (environment[x + y * width + z * width * height])
  {
    atomic_inc(&(miss[pixel_i]));
    return; // already set
  }

  atomic_inc(&(hits[pixel_i]));
}

void kernel EvaluateSensorOrientationOnViewCube(global const uint2 * view_cube,
                                                const uint width,
                                                const uint height,
                                                const uint depth,
                                                const float2 tan_hfov,
                                                const uint orientations_size,
                                                global const float4 * orientations,
                                                global uint * hits,
                                                global uint * miss)
{
  const uint x = get_global_id(0);
  const uint y = get_global_id(1);
  const uint z = get_global_id(2) % depth;
  const uint cube_id = get_global_id(2) / depth;

  if (x != 0 && y != 0 && z != 0 &&
      x + 1 != width && y + 1 != height && z + 1 != depth)
    return;

  uint i = x + y * width + z * width * height +
           cube_id * width * height * depth;

  float3 voxel_orient = normalize((float3)(x - width / 2.0f + 0.5, y - height / 2.0f + 0.5, z - depth / 2.0f + 0.5));

  const uint view_cube_hits = view_cube[i].x;
  const uint view_cube_miss = view_cube[i].y;

  for (uint vi = 0; vi < orientations_size; vi++)
  {
    const float4 orientation = orientations[vi];

    const float3 fwd = quat_ApplyRotation(orientation, (float3)(0, 0, 1));
    const float3 right = quat_ApplyRotation(orientation, (float3)(1, 0, 0));
    const float3 down = quat_ApplyRotation(orientation, (float3)(0, 1, 0));

    if (dot(voxel_orient, fwd) <= 0.0f)
      continue; // behind sensor

    if (fabs(dot(voxel_orient, right)) > tan_hfov.x * dot(voxel_orient, fwd))
      continue; // out of sensor horizontal fov

    if (fabs(dot(voxel_orient, down)) > tan_hfov.y * dot(voxel_orient, fwd))
      continue; // out of sensor vertical fov

    atomic_add(&(hits[vi + orientations_size * cube_id]), view_cube_hits);
    atomic_add(&(miss[vi + orientations_size * cube_id]), view_cube_miss);
  }
}

void kernel FillUint(
                     const uint c,
                     global uint * to_be_filled
                     )
{
  const int x = get_global_id(0);
  to_be_filled[x] = c;
}

void kernel FillFloat(
                      const float c,
                      global float * to_be_filled
                      )
{
  const int x = get_global_id(0);
  to_be_filled[x] = c;
}
