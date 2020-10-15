#include "generate_test_dataset_opencl.h"

#include "generate_test_dataset.cl.h"
#include "generate_test_dataset.h"

static cl_float2 EToCL(const Eigen::Vector2f & v)
{
  cl_float2 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  return result;
}

static cl_int2 EToCL(const Eigen::Vector2i & v)
{
  cl_int2 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  return result;
}

static cl_float3 EToCL(const Eigen::Vector3f & v)
{
  cl_float3 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  result.s[2] = v[2];
  return result;
}

static cl_int3 EToCL(const Eigen::Vector3i & v)
{
  cl_int3 result;
  result.s[0] = v[0];
  result.s[1] = v[1];
  result.s[2] = v[2];
  return result;
}

static cl_float4 EToCL(const Eigen::Quaternionf & q)
{
  cl_float4 result;
  result.s[0] = q.x();
  result.s[1] = q.y();
  result.s[2] = q.z();
  result.s[3] = q.w();
  return result;
}

static Eigen::Vector2i CLToE(const cl_int2 & v)
{
  return Eigen::Vector2i(v.s[0], v.s[1]);
}

static Eigen::Vector3i CLToE(const cl_int3 & v)
{
  return Eigen::Vector3i(v.s[0], v.s[1], v.s[2]);
}

static Eigen::Vector2f CLToE(const cl_float2 & v)
{
  return Eigen::Vector2f(v.s[0], v.s[1]);
}

static Eigen::Vector3f CLToE(const cl_float3 & v)
{
  return Eigen::Vector3f(v.s[0], v.s[1], v.s[2]);
}

GenerateTestDatasetOpenCL::GenerateTestDatasetOpenCL(ros::NodeHandle & nh): m_nh(nh)
{
  m_last_multi_ray_size = 0;
  m_last_compute_gain_from_view_size = 0;
  m_last_fill_environment_from_view_size = 0;
  m_last_sensor_resolution = 0;
  m_last_environment_size = Eigen::Vector3i::Zero();
  m_last_environment_ig_size = Eigen::Vector3i::Zero();
  m_evaluate_sensor_orientation_orientations_size = 0;
  m_evaluate_sensor_orientation_view_cube_size = 0;
  m_evaluate_sensor_orientation_hits_miss_size = 0;

  InitOpenCL();

  m_fill_uint_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "FillUint"));
  m_fill_float_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "FillFloat"));
}

void GenerateTestDatasetOpenCL::InitOpenCL()
{
  std::string param_string;

  std::string platform_name;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_PLATFORM_NAME, platform_name, PARAM_DEFAULT_OPENCL_PLATFORM_NAME);
  std::string device_name;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_DEVICE_NAME, device_name, PARAM_DEFAULT_OPENCL_DEVICE_NAME);

  cl_device_type device_type;
  m_nh.param<std::string>(PARAM_NAME_OPENCL_DEVICE_TYPE, param_string, PARAM_DEFAULT_OPENCL_DEVICE_TYPE);
  if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL)
    device_type = CL_DEVICE_TYPE_ALL;
  else if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_CPU)
    device_type = CL_DEVICE_TYPE_CPU;
  else if (param_string == PARAM_VALUE_OPENCL_DEVICE_TYPE_GPU)
    device_type = CL_DEVICE_TYPE_GPU;
  else
  {
    ROS_ERROR("generate_test_dataset_opencl: invalid parameter opencl_device_type, value '%s', using '%s' instead.",
              param_string.c_str(), PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL);
    device_type = CL_DEVICE_TYPE_ALL;
  }

  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);

  if (all_platforms.empty())
  {
    ROS_ERROR("generate_test_dataset_opencl: opencl: no platforms found.");
    exit(1);
  }

  {
    std::string all_platform_names;
    for (uint64 i = 0; i < all_platforms.size(); i++)
      all_platform_names += "\n  -- " + all_platforms[i].getInfo<CL_PLATFORM_NAME>();
    ROS_INFO_STREAM("generate_test_dataset_opencl: opencl: found platforms:" << all_platform_names);
  }
  uint64 platform_id = 0;
  if (!platform_name.empty())
  {
    ROS_INFO("generate_test_dataset_opencl: looking for matching platform: %s", platform_name.c_str());
    for (uint64 i = 0; i < all_platforms.size(); i++)
    {
      const std::string plat = all_platforms[i].getInfo<CL_PLATFORM_NAME>();
      if (plat.find(platform_name) != std::string::npos)
      {
        ROS_INFO("generate_test_dataset_opencl: found matching platform: %s", plat.c_str());
        platform_id = i;
        break;
      }
    }
  }

  cl::Platform default_platform = all_platforms[platform_id];
  ROS_INFO_STREAM("generate_test_dataset_opencl: using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>());

  std::vector<cl::Device> all_devices;
  default_platform.getDevices(device_type, &all_devices);
  if (all_devices.empty())
  {
      ROS_INFO("generate_test_dataset_opencl: no devices found.");
      exit(1);
  }
  {
    std::string all_device_names;
    for (uint64 i = 0; i < all_devices.size(); i++)
      all_device_names += "\n  -- " + all_devices[i].getInfo<CL_DEVICE_NAME>();
    ROS_INFO_STREAM("generate_test_dataset_opencl: found devices:" << all_device_names);
  }
  uint64 device_id = 0;
  if (!device_name.empty())
  {
    ROS_INFO("generate_test_dataset_opencl: looking for matching device: %s", device_name.c_str());
    for (uint64 i = 0; i < all_devices.size(); i++)
    {
      const std::string dev = all_devices[i].getInfo<CL_DEVICE_NAME>();
      if (dev.find(device_name) != std::string::npos)
      {
        ROS_INFO("generate_test_dataset_opencl: found matching device: %s", dev.c_str());
        device_id = i;
        break;
      }
    }
  }

  cl::Device default_device = all_devices[device_id];
  m_opencl_device = CLDevicePtr(new cl::Device(default_device));
  ROS_INFO_STREAM("generate_test_dataset_opencl: using device: " << default_device.getInfo<CL_DEVICE_NAME>());

  m_opencl_context = CLContextPtr(new cl::Context({*m_opencl_device}));

  m_opencl_command_queue = CLCommandQueuePtr(new cl::CommandQueue(*m_opencl_context,*m_opencl_device));

  std::string source = GENERATE_TEST_DATASET_CL;

  cl::Program::Sources sources;
  sources.push_back({source.c_str(),source.length()});

  ROS_INFO("generate_test_dataset_opencl: building program... ");
  m_opencl_program = CLProgramPtr(new cl::Program(*m_opencl_context,sources));
  if (m_opencl_program->build({*m_opencl_device}) != CL_SUCCESS)
  {
    ROS_ERROR_STREAM("generate_test_dataset_opencl: error building opencl_program: " <<
                     m_opencl_program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(*m_opencl_device));
    exit(1);
  }
  ROS_INFO("generate_test_dataset_opencl: initialized.");
}

void GenerateTestDatasetOpenCL::SimulateMultiRayWithInformationGainBatched(const Voxelgrid &known_empty,
                                                                           const Voxelgrid &known_occupied,
                                                                           const Vector3fVector &origins,
                                                                           const Vector3fVector &bearings,
                                                                           const float sensor_f,
                                                                           const float max_range,
                                                                           const float min_range,
                                                                           const bool stop_at_first_hit,
                                                                           const float a_priori_occupied_prob,
                                                                           FloatVector & hits,
                                                                           FloatVector & miss)
{
  const uint64 rays_size = origins.size();
  const uint64 BATCH_SIZE = 1000 * 1000;

  const uint64 batches = (rays_size / BATCH_SIZE) + !!(rays_size % BATCH_SIZE);

  hits.resize(rays_size);
  miss.resize(rays_size);

  for (uint64 b = 0; b < batches; b++)
  {
    FloatVector b_hits;
    FloatVector b_miss;

    const uint64 start_index = b * BATCH_SIZE;
    const uint64 end_index = std::min<uint64>(start_index + BATCH_SIZE, rays_size);
    const uint64 operations = end_index - start_index;

    const Vector3fVector b_origins(origins.begin() + start_index, origins.begin() + end_index);
    const Vector3fVector b_orientations(bearings.begin() + start_index, bearings.begin() + end_index);

    SimulateMultiRayWithInformationGain(known_empty, known_occupied, b_origins, b_orientations,
                                        sensor_f, max_range, min_range, stop_at_first_hit, a_priori_occupied_prob,
                                        b_hits, b_miss);

    for (uint64 i = 0; i < operations; i++)
    {
      hits[i + start_index] = b_hits[i];
      miss[i + start_index] = b_miss[i];
    }
  }
}

void GenerateTestDatasetOpenCL::SimulateMultiRayWithInformationGain(const Voxelgrid &known_empty,
                                                                    const Voxelgrid &known_occupied,
                                                                    const Vector3fVector &origins,
                                                                    const Vector3fVector &bearings,
                                                                    const float sensor_f,
                                                                    const float max_range,
                                                                    const float min_range,
                                                                    const bool stop_at_first_hit,
                                                                    const float a_priori_occupied_prob,
                                                                    FloatVector & hits,
                                                                    FloatVector & miss)
{
  const uint64 rays_size = origins.size();

  if (bearings.size() != origins.size())
  {
    ROS_ERROR("SimulateMultiRayWithInformationGain: orientations and origins size mismatch.");
    return;
  }

  if (!m_multi_ray_ig_origins || !m_multi_ray_ig_hits || !m_multi_ray_ig_miss ||
      !m_multi_ray_ig_orientations || m_last_multi_ray_ig_size < rays_size)
  {
    m_multi_ray_ig_origins = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float), "m_multi_ray_ig_origins");
    m_multi_ray_ig_orientations = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float),
                                               "m_multi_ray_ig_orientations");
    m_multi_ray_ig_hits = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float), "m_multi_ray_ig_hits");
    m_multi_ray_ig_miss = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float), "m_multi_ray_ig_miss");
    m_last_multi_ray_ig_size = rays_size;
  }

  if (!m_multi_ray_ig_kernel)
  {
    m_multi_ray_ig_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "SimulateMultiRayWithInformationGain"));
  }

  const Eigen::Vector3i environment_size = known_empty.GetSize();
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_multi_ray_ig_known_occupied || !m_multi_ray_ig_known_empty || environment_size != m_last_environment_ig_size)
  {
    m_multi_ray_ig_known_occupied = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_float),
                                                 "m_multi_ray_ig_known_occupied");
    m_multi_ray_ig_known_empty = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_float),
                                              "m_multi_ray_ig_known_empty");
    m_last_environment_ig_size = environment_size;
  }

  CLFloatVector cl_known_occupied(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        cl_known_occupied[z * width * height + y * width + x] = known_occupied.at(x, y, z);

  CLFloatVector cl_known_empty(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        cl_known_empty[z * width * height + y * width + x] = known_empty.at(x, y, z);

  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_ig_known_occupied, CL_TRUE, 0,
                                             environment_size.prod() * sizeof(cl_float),
                                             cl_known_occupied.data());

  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_ig_known_empty, CL_TRUE, 0,
                                             environment_size.prod() * sizeof(cl_float),
                                             cl_known_empty.data());

  CLFloatVector cl_origins(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_origins[i + 0] = origins[i].x();
    cl_origins[i + rays_size] = origins[i].y();
    cl_origins[i + 2 * rays_size] = origins[i].z();
  }
  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_ig_origins, CL_TRUE, 0,
                                             cl_origins.size() * sizeof(cl_float),
                                             cl_origins.data());

  CLFloatVector cl_orientations(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_orientations[i + 0 * rays_size] = bearings[i].x();
    cl_orientations[i + 1 * rays_size] = bearings[i].y();
    cl_orientations[i + 2 * rays_size] = bearings[i].z();
  }
  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_ig_orientations, CL_TRUE, 0,
                                             cl_orientations.size() * sizeof(cl_float),
                                             cl_orientations.data());

  {
    uint64 c = 0;
    m_multi_ray_ig_kernel->setArg(c++, *m_multi_ray_ig_known_occupied);
    m_multi_ray_ig_kernel->setArg(c++, *m_multi_ray_ig_known_empty);
    m_multi_ray_ig_kernel->setArg(c++, cl_uint(width));
    m_multi_ray_ig_kernel->setArg(c++, cl_uint(height));
    m_multi_ray_ig_kernel->setArg(c++, cl_uint(depth));
    m_multi_ray_ig_kernel->setArg(c++, cl_uint(rays_size));
    m_multi_ray_ig_kernel->setArg(c++, cl_float(sensor_f));
    m_multi_ray_ig_kernel->setArg(c++, *m_multi_ray_ig_origins);
    m_multi_ray_ig_kernel->setArg(c++, *m_multi_ray_ig_orientations);
    m_multi_ray_ig_kernel->setArg(c++, cl_float(max_range));
    m_multi_ray_ig_kernel->setArg(c++, cl_float(min_range));
    m_multi_ray_ig_kernel->setArg(c++, cl_uchar(stop_at_first_hit));
    m_multi_ray_ig_kernel->setArg(c++, cl_float(a_priori_occupied_prob));
    m_multi_ray_ig_kernel->setArg(c++, *m_multi_ray_ig_hits);
    m_multi_ray_ig_kernel->setArg(c++, *m_multi_ray_ig_miss);
  }
  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_multi_ray_ig_kernel, cl::NullRange,
                                    cl::NDRange(rays_size), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  SimulateMultiRayWithInformationGain: error m_multi_ray_ig_kernel: %d!", ret);
    exit(1);
  }

  CLFloatVector cl_hits(rays_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_multi_ray_ig_hits, CL_TRUE, 0,
                                            rays_size * sizeof(cl_float),
                                            cl_hits.data());

  CLFloatVector cl_miss(rays_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_multi_ray_ig_miss, CL_TRUE, 0,
                                            rays_size * sizeof(cl_float),
                                            cl_miss.data());

  hits.resize(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    hits[i] = cl_hits[i];

  miss.resize(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    miss[i] = cl_miss[i];
}

void GenerateTestDatasetOpenCL::SimulateMultiRayWI(
  const Voxelgrid &environment, const Vector3fVector &origins,
  const Vector3fVector &bearings, const float max_range,
  FloatVector & nearest_dist, Vector3iVector &observed_points)
{
  const uint64 size = origins.size();

  Uint64Vector mapping;
  mapping.reserve(size);
  Vector3fVector bearings_out;
  bearings_out.reserve(size);
  Vector3fVector origins_out;
  origins_out.reserve(size);

  nearest_dist.resize(size, 0.0f);
  observed_points.resize(size, -Eigen::Vector3i::Ones());

  FloatVector nearest_dist_out;
  Vector3iVector observed_points_out;

  for (uint64 i = 0; i < size; i++)
  {
    if (bearings[i] != Eigen::Vector3f::Zero())
    {
      mapping.push_back(i);
      origins_out.push_back(origins[i]);
      bearings_out.push_back(bearings[i]);
    }
  }

  SimulateMultiRay(environment, origins_out, bearings_out, max_range, nearest_dist_out, observed_points_out);

  for (uint64 i = 0; i < mapping.size(); i++)
  {
    nearest_dist[mapping[i]] = nearest_dist_out[i];
    observed_points[mapping[i]] = observed_points_out[i];
  }
}

void GenerateTestDatasetOpenCL::SimulateMultiRay(
  const Voxelgrid & environment,
  const Vector3fVector & origins, const Vector3fVector & bearings,
  const float max_range,
  FloatVector & nearest_dist, Vector3iVector & observed_points)
{
  uint64 rays_size = origins.size();

  if (bearings.size() != origins.size())
  {
    ROS_ERROR("generate_test_dataset_opencl: orientations and origins size mismatch.");
    return;
  }

  if (!m_multi_ray_origins || !m_multi_ray_distances || !m_multi_ray_observed_points ||
      !m_multi_ray_orientations || m_last_multi_ray_size < rays_size)
  {
    m_multi_ray_origins = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float), "m_multi_ray_origins");
    m_multi_ray_orientations = CreateBuffer(m_opencl_context, rays_size * 3 * sizeof(cl_float), "m_multi_ray_orientations");
    m_multi_ray_distances = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float), "m_multi_ray_distances");
    m_multi_ray_observed_points = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_int3), "m_multi_ray_observed_points");
    m_last_multi_ray_size = rays_size;
  }

  if (!m_simulate_multi_ray_kernel)
  {
    m_simulate_multi_ray_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "SimulateMultiRay"));
  }

  const Eigen::Vector3i environment_size = environment.GetSize();
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();

  if (!m_environment || environment_size != m_last_environment_size)
  {
    m_environment = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_ushort), "m_environment");
    m_last_environment_size = environment_size;
  }

  CLUInt16Vector cl_environment(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        cl_environment[z * width * height + y * width + x] = environment.at(x, y, z) * 255;

  m_opencl_command_queue->enqueueWriteBuffer(*m_environment, CL_TRUE, 0,
                                             cl_environment.size() * sizeof(cl_ushort),
                                             cl_environment.data());

  CLFloatVector cl_origins(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_origins[i + 0] = origins[i].x();
    cl_origins[i + rays_size] = origins[i].y();
    cl_origins[i + 2 * rays_size] = origins[i].z();
  }
  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_origins, CL_TRUE, 0,
                                             rays_size * 3 * sizeof(cl_float),
                                             cl_origins.data());

  CLFloatVector cl_bearings(rays_size * 3);
  for (uint64 i = 0; i < rays_size; i++)
  {
    cl_bearings[i + 0] = bearings[i].x();
    cl_bearings[i + rays_size] = bearings[i].y();
    cl_bearings[i + 2 * rays_size] = bearings[i].z();
  }
  m_opencl_command_queue->enqueueWriteBuffer(*m_multi_ray_orientations, CL_TRUE, 0,
                                             rays_size * 3 * sizeof(cl_float),
                                             cl_bearings.data());

  {
    uint64 c = 0;
    m_simulate_multi_ray_kernel->setArg(c++, *m_environment);
    m_simulate_multi_ray_kernel->setArg(c++, cl_uint(width));
    m_simulate_multi_ray_kernel->setArg(c++, cl_uint(height));
    m_simulate_multi_ray_kernel->setArg(c++, cl_uint(depth));
    m_simulate_multi_ray_kernel->setArg(c++, cl_uint(rays_size));
    m_simulate_multi_ray_kernel->setArg(c++, *m_multi_ray_origins);
    m_simulate_multi_ray_kernel->setArg(c++, *m_multi_ray_orientations);
    m_simulate_multi_ray_kernel->setArg(c++, cl_float(max_range));
    m_simulate_multi_ray_kernel->setArg(c++, *m_multi_ray_distances);
    m_simulate_multi_ray_kernel->setArg(c++, *m_multi_ray_observed_points);
  }
  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_simulate_multi_ray_kernel, cl::NullRange,
                                    cl::NDRange(rays_size), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  SimulateMultiRay: error m_simulate_multi_ray_kernel: %d!", ret);
    exit(1);
  }

  CLFloatVector cl_nearest_dist(rays_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_multi_ray_distances, CL_TRUE, 0,
                                            rays_size * sizeof(cl_float),
                                            cl_nearest_dist.data());

  CLInt3Vector cl_observed_points(rays_size);
  m_opencl_command_queue->enqueueReadBuffer(*m_multi_ray_observed_points, CL_TRUE, 0,
                                            rays_size * sizeof(cl_int3),
                                            cl_observed_points.data());

  nearest_dist.resize(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    nearest_dist[i] = cl_nearest_dist[i];

  observed_points.resize(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    observed_points[i] = CLToE(cl_observed_points[i]);
}

void GenerateTestDatasetOpenCL::FillEnvironmentFromView(const Voxelgrid & input_empty,
                                                        const Eigen::Vector3f & origin,
                                                        const Eigen::Quaternionf & orientation,
                                                        const float sensor_f,
                                                        const Eigen::Vector2i & sensor_resolution,
                                                        const FloatVector & nearest_dist,
                                                        const Vector3iVector & observed_points,
                                                        Voxelgrid & filled_empty
                                                        )
{
  uint64 rays_size = nearest_dist.size();

  if (!m_fill_environment_from_view_nearest_dist || !m_fill_environment_from_view_observed_points ||
      m_last_fill_environment_from_view_size < rays_size)
  {
    m_fill_environment_from_view_nearest_dist = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_float),
                                                             "m_fill_environment_from_view_nearest_dist");
    m_fill_environment_from_view_observed_points = CreateBuffer(m_opencl_context, rays_size * sizeof(cl_int3),
                                                                "m_fill_environment_from_view_observed_points");
    m_last_fill_environment_from_view_size = rays_size;
  }

  if (!m_fill_environment_from_view_kernel)
  {
    m_fill_environment_from_view_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "FillEnvironmentFromView"));
  }

  const Eigen::Vector3i environment_size = input_empty.GetSize();
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();
  if (!m_environment || environment_size != m_last_environment_size)
  {
    m_environment = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_ushort), "m_environment");
    m_last_environment_size = environment_size;
  }

  CLUInt16Vector cl_environment(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      cl_environment[z * width * height + y * width + x] = input_empty.at(x, y, z) * 255;

  m_opencl_command_queue->enqueueWriteBuffer(*m_environment, CL_TRUE, 0,
                                             cl_environment.size() * sizeof(cl_ushort),
                                             cl_environment.data());

  CLFloatVector cl_distances(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    cl_distances[i] = nearest_dist[i];
  m_opencl_command_queue->enqueueWriteBuffer(*m_fill_environment_from_view_nearest_dist, CL_TRUE, 0,
                                             rays_size * sizeof(cl_float),
                                             cl_distances.data());

  CLInt3Vector cl_observed_points(rays_size);
  for (uint64 i = 0; i < rays_size; i++)
    cl_observed_points[i] = EToCL(observed_points[i]);
  m_opencl_command_queue->enqueueWriteBuffer(*m_fill_environment_from_view_observed_points, CL_TRUE, 0,
                                             rays_size * sizeof(cl_int3),
                                             cl_observed_points.data());

  /*
   *                                const uint width, const uint height, const uint depth,
                                    const float3 origin, const float4 orientation,
                                    const uint2 sensor_resolution, const float sensor_f,
                                    global const float * nearest_dist,
                                    global const int3 * observed_points,
                                    global ushort * filled_environment*/
  {
    uint64 c = 0;

    m_fill_environment_from_view_kernel->setArg(c++, cl_uint(width));
    m_fill_environment_from_view_kernel->setArg(c++, cl_uint(height));
    m_fill_environment_from_view_kernel->setArg(c++, cl_uint(depth));
    m_fill_environment_from_view_kernel->setArg(c++, EToCL(origin));
    m_fill_environment_from_view_kernel->setArg(c++, EToCL(orientation));
    m_fill_environment_from_view_kernel->setArg(c++, EToCL(sensor_resolution));
    m_fill_environment_from_view_kernel->setArg(c++, cl_float(sensor_f));
    m_fill_environment_from_view_kernel->setArg(c++, *m_fill_environment_from_view_nearest_dist);
    m_fill_environment_from_view_kernel->setArg(c++, *m_fill_environment_from_view_observed_points);
    m_fill_environment_from_view_kernel->setArg(c++, *m_environment);
  }
  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_environment_from_view_kernel, cl::NullRange,
                                    cl::NDRange(width, height, depth), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  FillEnvironmentFromView: error m_fill_environment_from_view_kernel: %d!", ret);
    exit(1);
  }

  m_opencl_command_queue->enqueueReadBuffer(*m_environment, CL_TRUE, 0,
                                            cl_environment.size() * sizeof(cl_ushort),
                                            cl_environment.data());

  filled_empty = Voxelgrid(width, height, depth);
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        filled_empty.at(x, y, z) = (cl_environment[x + y * width + z * width * height] ? 1.0f : 0.0f);
      }
}

void GenerateTestDatasetOpenCL::FillEnvironmentFromViewCube(const Voxelgrid &input_empty,
                                                            const Eigen::Vector3f &origin,
                                                            const Eigen::Quaternionf &orientation,
                                                            const Eigen::Vector2f & sensor_hfov,
                                                            const Eigen::Vector3i &view_cube_resolution,
                                                            const FloatVector & nearest_dist,
                                                            const Vector3iVector &observed_points,
                                                            Voxelgrid &filled_empty
                                                            )
{
  if (!m_fill_environment_from_view_cube_nearest_dist || !m_fill_environment_from_view_cube_observed_points ||
      m_last_fill_environment_from_view_cube_size < view_cube_resolution.prod())
  {
    m_fill_environment_from_view_cube_nearest_dist = CreateBuffer(m_opencl_context,
                                                                  view_cube_resolution.prod() * sizeof(cl_float),
                                                                  "m_fill_environment_from_view_cube_nearest_dist");
    m_fill_environment_from_view_cube_observed_points = CreateBuffer(m_opencl_context, view_cube_resolution.prod() *
                                                                     sizeof(cl_int3),
                                                                     "m_fill_environment_from_view_cube_observed_"
                                                                     "points");
    m_last_fill_environment_from_view_cube_size = view_cube_resolution.prod();
  }

  if (!m_fill_environment_from_view_cube_kernel)
  {
    m_fill_environment_from_view_cube_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program,
                                                                          "FillEnvironmentFromViewCube"));
  }

  const Eigen::Vector2f sensor_tan_hfov = sensor_hfov.array().tan();

  const Eigen::Vector3i environment_size = input_empty.GetSize();
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();
  if (!m_environment || environment_size != m_last_environment_size)
  {
    m_environment = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_ushort), "m_environment");
    m_last_environment_size = environment_size;
  }

  CLUInt16Vector cl_environment(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        cl_environment[y * width + x] = input_empty.at(x, y, z) * 255;

  m_opencl_command_queue->enqueueWriteBuffer(*m_environment, CL_TRUE, 0,
                                             cl_environment.size() * sizeof(cl_ushort),
                                             cl_environment.data());

  CLFloatVector cl_distances(view_cube_resolution.prod());
  for (uint64 i = 0; i < view_cube_resolution.prod(); i++)
    cl_distances[i] = nearest_dist[i];
  m_opencl_command_queue->enqueueWriteBuffer(*m_fill_environment_from_view_cube_nearest_dist, CL_TRUE, 0,
                                             cl_distances.size() * sizeof(cl_float),
                                             cl_distances.data());

  CLInt3Vector cl_observed_points(view_cube_resolution.prod());
  for (uint64 i = 0; i < view_cube_resolution.prod(); i++)
    cl_observed_points[i] = EToCL(observed_points[i]);
  m_opencl_command_queue->enqueueWriteBuffer(*m_fill_environment_from_view_cube_observed_points, CL_TRUE, 0,
                                             cl_observed_points.size() * sizeof(cl_int3),
                                             cl_observed_points.data());

  /*
   *                                    const uint width,
                                        const uint height,
                                        const uint depth,
                                        const float3 origin,
                                        const float4 orientation,
                                        const float sensor_f,
                                        const int3 view_cube_resolution,
                                        global const float * nearest_dist,
                                        global const int2 * observed_points,
                                        global ushort * filled_environment*/
  {
    uint64 c = 0;

    m_fill_environment_from_view_cube_kernel->setArg(c++, cl_uint(width));
    m_fill_environment_from_view_cube_kernel->setArg(c++, cl_uint(height));
    m_fill_environment_from_view_cube_kernel->setArg(c++, cl_uint(depth));
    m_fill_environment_from_view_cube_kernel->setArg(c++, EToCL(origin));
    m_fill_environment_from_view_cube_kernel->setArg(c++, EToCL(orientation));
    m_fill_environment_from_view_cube_kernel->setArg(c++, EToCL(sensor_tan_hfov));
    m_fill_environment_from_view_cube_kernel->setArg(c++, EToCL(view_cube_resolution));
    m_fill_environment_from_view_cube_kernel->setArg(c++, *m_fill_environment_from_view_cube_nearest_dist);
    m_fill_environment_from_view_cube_kernel->setArg(c++, *m_fill_environment_from_view_cube_observed_points);
    m_fill_environment_from_view_cube_kernel->setArg(c++, *m_environment);
  }
  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_environment_from_view_cube_kernel, cl::NullRange,
                                    cl::NDRange(width, height, depth), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  FillEnvironmentFromViewCube: error m_fill_environment_from_view_cube_kernel: %d!", ret);
    exit(1);
  }

  m_opencl_command_queue->enqueueReadBuffer(*m_environment, CL_TRUE, 0,
                                            cl_environment.size() * sizeof(cl_ushort),
                                            cl_environment.data());

  filled_empty = Voxelgrid(width, height, depth);
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        filled_empty.at(x, y, z) = cl_environment[x + y * width + z * width * height];
      }
}

void GenerateTestDatasetOpenCL::EvaluateSensorOrientationsOnViewCube(const Voxelgrid & view_cube_hits,
                                                                     const Voxelgrid & view_cube_miss,
                                                                     const Eigen::Vector2f & hfov,
                                                                     const QuaternionfVector & orientations,
                                                                     FloatVector & hits,
                                                                     FloatVector & miss
                                                                     )
{
  const float orientations_size = orientations.size();
  const uint64 num_view_cubes = 1;
  hits.assign(orientations_size * num_view_cubes, 0.0f);
  miss.assign(orientations_size * num_view_cubes, 0.0f);
  if (!num_view_cubes)
    return;

  const Eigen::Vector3i view_cube_size = view_cube_hits.GetSize();

  if (!m_evaluate_sensor_orientation_hits || !m_evaluate_sensor_orientation_miss ||
      m_evaluate_sensor_orientation_hits_miss_size != orientations_size * num_view_cubes)
  {
    m_evaluate_sensor_orientation_hits = CreateBuffer(m_opencl_context, orientations_size * num_view_cubes *
                                                      sizeof(cl_uint),
                                                      "m_evaluate_sensor_orientation_hits");
    m_evaluate_sensor_orientation_miss = CreateBuffer(m_opencl_context, orientations_size * num_view_cubes *
                                                      sizeof(cl_uint),
                                                      "m_evaluate_sensor_orientation_miss");
    m_evaluate_sensor_orientation_hits_miss_size = orientations_size * num_view_cubes;
  }

  if (!m_evaluate_sensor_orientation_orientations ||
      m_evaluate_sensor_orientation_orientations_size != orientations_size)
  {
    m_evaluate_sensor_orientation_orientations = CreateBuffer(m_opencl_context, orientations_size *
                                                              sizeof(cl_float4),
                                                              "m_evaluate_sensor_orientation_orientations");
    m_evaluate_sensor_orientation_orientations_size = orientations_size;
  }

  if (!m_evaluate_sensor_orientation_view_cube ||
      m_evaluate_sensor_orientation_view_cube_size != view_cube_size.prod())
  {
    m_evaluate_sensor_orientation_view_cube = CreateBuffer(m_opencl_context, view_cube_size.prod() * num_view_cubes *
                                                           sizeof(cl_int2),
                                                           "m_evaluate_sensor_orientation_view_cube");
    m_evaluate_sensor_orientation_view_cube_size = view_cube_size.prod() * num_view_cubes;
  }

  if (!m_evaluate_sensor_orientation_kernel)
  {
    m_evaluate_sensor_orientation_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "EvaluateSensorOrientationOnViewCube"));
  }

  CLFloat4Vector cl_orientations(orientations_size);
  for (uint64 i = 0; i < orientations_size; i++)
    cl_orientations[i] = EToCL(orientations[i]);
  m_opencl_command_queue->enqueueWriteBuffer(*m_evaluate_sensor_orientation_orientations, CL_TRUE, 0,
                                             cl_orientations.size() * sizeof(cl_float4),
                                             cl_orientations.data());

  {
    uint64 c = 0;
    m_fill_uint_kernel->setArg(c++, cl_float(0));
    m_fill_uint_kernel->setArg(c++, *m_evaluate_sensor_orientation_hits);
    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_uint_kernel, cl::NullRange,
                                      cl::NDRange(orientations_size), cl::NullRange);
    if (ret != CL_SUCCESS)
    {
      ROS_ERROR("  FillUintKernel: error m_fill_uint_kernel hits: %d!", ret);
      exit(1);
    }
  }

  {
    uint64 c = 0;
    m_fill_uint_kernel->setArg(c++, cl_uint(0));
    m_fill_uint_kernel->setArg(c++, *m_evaluate_sensor_orientation_miss);
    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_uint_kernel, cl::NullRange,
                                      cl::NDRange(orientations_size), cl::NullRange);
    if (ret != CL_SUCCESS)
    {
      ROS_ERROR("  FillUintKernel: error m_fill_uint_kernel hits: %d!", ret);
      exit(1);
    }
  }

  CLInt2Vector cl_view_cube(view_cube_size.prod() * num_view_cubes);
  for (uint64 view_cube_id = 0; view_cube_id < num_view_cubes; view_cube_id++)
    for (uint64 z = 0; z < view_cube_size.z(); z++)
      for (uint64 y = 0; y < view_cube_size.y(); y++)
        for (uint64 x = 0; x < view_cube_size.x(); x++)
        {
          const uint64 i = x + y * view_cube_size.x() + z * view_cube_size.x() * view_cube_size.y();
          const Voxelgrid & view_cube_h = view_cube_hits;
          const Voxelgrid & view_cube_m = view_cube_miss;
          cl_view_cube[i] = EToCL(Eigen::Vector2i(view_cube_h.at(x, y, z) * 255, view_cube_m.at(x, y, z) * 255));
        }
  m_opencl_command_queue->enqueueWriteBuffer(*m_evaluate_sensor_orientation_view_cube, CL_TRUE, 0,
                                             cl_view_cube.size() * sizeof(cl_uint2),
                                             cl_view_cube.data());

  const Eigen::Vector2f tan_hfov = hfov.array().tan();

  {
    uint64 c = 0;
    m_evaluate_sensor_orientation_kernel->setArg(c++, *m_evaluate_sensor_orientation_view_cube);
    m_evaluate_sensor_orientation_kernel->setArg(c++, cl_uint(view_cube_size.x()));
    m_evaluate_sensor_orientation_kernel->setArg(c++, cl_uint(view_cube_size.y()));
    m_evaluate_sensor_orientation_kernel->setArg(c++, cl_uint(view_cube_size.z()));
    m_evaluate_sensor_orientation_kernel->setArg(c++, EToCL(tan_hfov));
    m_evaluate_sensor_orientation_kernel->setArg(c++, cl_uint(orientations_size));
    m_evaluate_sensor_orientation_kernel->setArg(c++, *m_evaluate_sensor_orientation_orientations);
    m_evaluate_sensor_orientation_kernel->setArg(c++, *m_evaluate_sensor_orientation_hits);
    m_evaluate_sensor_orientation_kernel->setArg(c++, *m_evaluate_sensor_orientation_miss);
  }
  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_evaluate_sensor_orientation_kernel, cl::NullRange,
                                    cl::NDRange(view_cube_size.x(), view_cube_size.y(), view_cube_size.z()
                                                * num_view_cubes), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  EvaluateSensorOrientationKernel: error m_compute_gain_from_view_kernel: %d!", ret);
    exit(1);
  }

  CLUInt32Vector cl_hits(orientations_size * num_view_cubes);
  m_opencl_command_queue->enqueueReadBuffer(*m_evaluate_sensor_orientation_hits, CL_TRUE, 0,
                                            cl_hits.size() * sizeof(cl_uint),
                                            cl_hits.data());
  CLUInt32Vector cl_miss(orientations_size * num_view_cubes);
  m_opencl_command_queue->enqueueReadBuffer(*m_evaluate_sensor_orientation_miss, CL_TRUE, 0,
                                            cl_miss.size() * sizeof(cl_uint),
                                            cl_miss.data());

  for (uint64 i = 0; i < orientations_size; i++)
    hits[i] = cl_hits[i] / 255.0f;
  for (uint64 i = 0; i < orientations_size; i++)
    miss[i] = cl_miss[i] / 255.0f;
}

void GenerateTestDatasetOpenCL::ComputeGainFromViewCube(const Voxelgrid & input_empty,
                                                        const Eigen::Vector3f & origin,
                                                        const Eigen::Vector3i & view_cube_resolution,
                                                        const float max_range,
                                                        const FloatVector & nearest_dist,
                                                        Uint64Vector & hits,
                                                        Uint64Vector & miss)
{
  if (!m_compute_gain_from_view_nearest_dist || !m_compute_gain_from_view_hits ||
      m_last_compute_gain_from_view_size < view_cube_resolution.prod())
  {
    m_compute_gain_from_view_nearest_dist = CreateBuffer(m_opencl_context, view_cube_resolution.prod() *
                                                         sizeof(cl_float),
                                                         "m_compute_gain_from_view_nearest_dist");
    m_compute_gain_from_view_hits = CreateBuffer(m_opencl_context, view_cube_resolution.prod() *
                                                 sizeof(cl_uint),
                                                 "m_compute_gain_from_view_hits");
    m_compute_gain_from_view_miss = CreateBuffer(m_opencl_context, view_cube_resolution.prod() *
                                                 sizeof(cl_uint),
                                                 "m_compute_gain_from_view_miss");
    m_last_compute_gain_from_view_size = view_cube_resolution.prod();
  }

  if (!m_compute_gain_from_view_kernel)
  {
    m_compute_gain_from_view_kernel = CLKernelPtr(new cl::Kernel(*m_opencl_program, "ComputeGainFromViewCube"));
  }

  const Eigen::Vector3i environment_size = input_empty.GetSize();
  const uint64 width = environment_size.x();
  const uint64 height = environment_size.y();
  const uint64 depth = environment_size.z();
  if (!m_environment || environment_size != m_last_environment_size)
  {
    m_environment = CreateBuffer(m_opencl_context, environment_size.prod() * sizeof(cl_ushort), "m_environment");
    m_last_environment_size = environment_size;
  }

  CLUInt16Vector cl_environment(environment_size.prod());
  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        cl_environment[z * width * height + y * width + x] = input_empty.at(x, y, z);

  m_opencl_command_queue->enqueueWriteBuffer(*m_environment, CL_TRUE, 0,
                                             cl_environment.size() * sizeof(cl_ushort),
                                             cl_environment.data());

  CLFloatVector cl_distances(view_cube_resolution.prod());
  for (uint64 i = 0; i < cl_distances.size(); i++)
    cl_distances[i] = nearest_dist[i];
  m_opencl_command_queue->enqueueWriteBuffer(*m_compute_gain_from_view_nearest_dist, CL_TRUE, 0,
                                             view_cube_resolution.prod() * sizeof(cl_float),
                                             cl_distances.data());

  {
    uint64 c = 0;
    m_fill_uint_kernel->setArg(c++, cl_uint(0));
    m_fill_uint_kernel->setArg(c++, *m_compute_gain_from_view_hits);
    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_uint_kernel, cl::NullRange,
                                      cl::NDRange(view_cube_resolution.prod()), cl::NullRange);
    if (ret != CL_SUCCESS)
    {
      ROS_ERROR("  FillUintKernel: error m_fill_uint_kernel hits: %d!", ret);
      exit(1);
    }
  }

  {
    uint64 c = 0;
    m_fill_uint_kernel->setArg(c++, cl_uint(0));
    m_fill_uint_kernel->setArg(c++, *m_compute_gain_from_view_miss);
    int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_fill_uint_kernel, cl::NullRange,
                                      cl::NDRange(view_cube_resolution.prod()), cl::NullRange);
    if (ret != CL_SUCCESS)
    {
      ROS_ERROR("  FillUintKernel: error m_fill_uint_kernel miss: %d!", ret);
      exit(1);
    }
  }

  {
    uint64 c = 0;

    /*                          global const ushort * environment,
                                const uint width, const uint height, const uint depth,
                                const float3 origin,
                                const uint3 view_cube_resolution,
                                const float max_range,
                                global const float * nearest_dist,
                                global uint * hits,
                                global uint * miss
*/

    m_compute_gain_from_view_kernel->setArg(c++, *m_environment);
    m_compute_gain_from_view_kernel->setArg(c++, cl_uint(width));
    m_compute_gain_from_view_kernel->setArg(c++, cl_uint(height));
    m_compute_gain_from_view_kernel->setArg(c++, cl_uint(depth));
    m_compute_gain_from_view_kernel->setArg(c++, EToCL(origin));
    m_compute_gain_from_view_kernel->setArg(c++, EToCL(view_cube_resolution));
    m_compute_gain_from_view_kernel->setArg(c++, cl_float(max_range));
    m_compute_gain_from_view_kernel->setArg(c++, *m_compute_gain_from_view_nearest_dist);
    m_compute_gain_from_view_kernel->setArg(c++, *m_compute_gain_from_view_hits);
    m_compute_gain_from_view_kernel->setArg(c++, *m_compute_gain_from_view_miss);
  }
  int ret = m_opencl_command_queue->enqueueNDRangeKernel(*m_compute_gain_from_view_kernel, cl::NullRange,
                                    cl::NDRange(width, height, depth), cl::NullRange);
  if (ret != CL_SUCCESS)
  {
    ROS_ERROR("  ComputeGainFromView: error m_compute_gain_from_view_kernel: %d!", ret);
    exit(1);
  }

  CLUInt32Vector cl_hits(view_cube_resolution.prod());
  m_opencl_command_queue->enqueueReadBuffer(*m_compute_gain_from_view_hits, CL_TRUE, 0,
                                            view_cube_resolution.prod() * sizeof(cl_uint),
                                            cl_hits.data());

  CLUInt32Vector cl_miss(view_cube_resolution.prod());
  m_opencl_command_queue->enqueueReadBuffer(*m_compute_gain_from_view_miss, CL_TRUE, 0,
                                            view_cube_resolution.prod() * sizeof(cl_uint),
                                            cl_miss.data());

  hits.resize(view_cube_resolution.prod());
  for (uint64 i = 0; i < hits.size(); i++)
    hits[i] = cl_hits[i];

  miss.resize(view_cube_resolution.prod());
  for (uint64 i = 0; i < miss.size(); i++)
    miss[i] = cl_miss[i];
}

void GenerateTestDatasetOpenCL::SimulateView(const Voxelgrid & environment, const Eigen::Vector3f & origin,
                                             const Vector3fVector ray_orientations,
                                             const float max_range,
                                             FloatVector & nearest_dist, Vector3iVector & observed_points)
{
  const Vector3fVector origins(ray_orientations.size(), origin);
  SimulateMultiRay(environment, origins, ray_orientations, max_range, nearest_dist, observed_points);
}

GenerateTestDatasetOpenCL::CLBufferPtr GenerateTestDatasetOpenCL::CreateBuffer(const CLContextPtr context,
                                                                               const size_t size,
                                                                               const std::string name) const
{
  cl_int err;
  CLBufferPtr buf = CLBufferPtr(new cl::Buffer(*context,CL_MEM_READ_WRITE,
                                    size, NULL, &err));
  if (err != CL_SUCCESS)
  {
    ROS_ERROR("could not allocate buffer '%s' of size %u, error %d", name.c_str(), unsigned(size), int(err));
  }
  return buf;
}
