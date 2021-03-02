#include <nbv_3d_cnn/simulate_nbv_cycle_adapter.h>

#include <nbv_3d_cnn/origin_visibility.h>
#include "simulate_nbv_cycle.h"
#include <nbv_3d_cnn/generate_test_dataset_opencl.h>
#include <nbv_3d_cnn/generate_single_image.h>

typedef uint64_t uint64;
typedef int64_t int64;
typedef uint8_t uint8;

bool RandomNBVAdapter::GetNextBestView(const Voxelgrid & environment,
                                       const Voxelgrid & empty,
                                       const Voxelgrid & occupied,
                                       const Voxelgrid & frontier,
                                       const Vector3fVector & skip_origins,
                                       const QuaternionfVector &skip_orentations,
                                       Eigen::Vector3f & origin,
                                       Eigen::Quaternionf &orientation,
                                       ViewWithScoreVector * const all_views_with_scores)
{
  const float orientation_angle = (float(rand()) / RAND_MAX) * 2.0 * M_PI;
  if (!m_is_3d)
  {
    orientation = Eigen::Quaternionf(Eigen::AngleAxisf(orientation_angle, Eigen::Vector3f::UnitZ()) *
                                     Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitX()));
  }
  else
  {
    const Eigen::Vector3f random = Eigen::Vector3f::Random();
    const Eigen::Vector3f axis = random.normalized();
    orientation = Eigen::Quaternionf(Eigen::AngleAxisf(orientation_angle, axis));
  }

  m_last_scores = *empty.FilledWith(0.0f);

  uint64 empty_counter = 0;

  const uint64 height = empty.GetHeight();
  const uint64 width = empty.GetWidth();
  const uint64 depth = empty.GetDepth();

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (empty.at(x, y, z))
          empty_counter++;
      }

  const uint64 selected = rand() % empty_counter;
  uint64 selected_counter = 0;

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (!empty.at(x, y, z))
          continue;

        if (selected == selected_counter)
        {
          origin = Eigen::Vector3f(x, y, z);
          m_last_scores.at(x, y, z) = 255;
          return true;
        }

        selected_counter++;
      }

  return true;
}

void CNNDirectionalNBVAdapter::onRawData(const nbv_3d_cnn::FloatsConstPtr raw_data)
{
  m_raw_data = raw_data;
  ROS_INFO("CNNOriginVisibilityNBVAdapter: got raw data.");
}

CNNDirectionalNBVAdapter::CNNDirectionalNBVAdapter(ros::NodeHandle & nh,
                                                   GenerateTestDatasetOpenCL &opencl,
                                                   const bool is_3d,
                                                   const Eigen::Vector2f &sensor_hfov,
                                                   const Mode mode,
                                                   const uint64_t accuracy_skip,
                                                   const uint64_t cnn_accuracy_skip):
  m_nh(nh), m_opencl(opencl), m_is_3d(is_3d), m_mode(mode), m_private_nh("~")
{
  std::string param_string;

  m_sensor_hfov = sensor_hfov;
  m_accuracy_skip = accuracy_skip;
  m_cnn_accuracy_skip = cnn_accuracy_skip;

  if (m_mode == MODE_OV || m_mode == MODE_OV_DIRECT)
    m_nh.param<std::string>(PARAM_NAME_PREDICT_ACTION_NAME, param_string, PARAM_DEFAULT_PREDICT_ACTION_NAME);
  else if (m_mode == MODE_FLAT)
    m_nh.param<std::string>(PARAM_NAME_PREDICT_FLAT_ACTION_NAME, param_string,
                            PARAM_DEFAULT_PREDICT_FLAT_ACTION_NAME);
  if (!m_is_3d)
    m_predict_action_client.reset(new PredictActionClient(param_string, true));
  else
    m_predict_3d_action_client.reset(new Predict3dActionClient(param_string, true));

  m_private_nh.setCallbackQueue(&m_raw_data_callback_queue);
  m_raw_data_subscriber = m_private_nh.subscribe(param_string + "raw_data", 1, &CNNDirectionalNBVAdapter::onRawData, this);

  ROS_INFO("simulate_nbv_cycle: CNNOriginVisibilityNBVAdapter: waiting for prediction server on topic %s",
           param_string.c_str());
  if (!m_is_3d)
    m_predict_action_client->waitForServer();
  else
    m_predict_3d_action_client->waitForServer();
  ROS_INFO("simulate_nbv_cycle: CNNOriginVisibilityNBVAdapter: prediction server ok.");
}

bool CNNDirectionalNBVAdapter::GetNextBestView3d(const Voxelgrid & environment,
                                                 const Voxelgrid & empty,
                                                 const Voxelgrid & occupied,
                                                 const Voxelgrid & frontier,
                                                 const Vector3fVector & skip_origins,
                                                 const QuaternionfVector & skip_orentations,
                                                 Eigen::Vector3f & origin,
                                                 Eigen::Quaternionf & orientation,
                                                 ViewWithScoreVector * const all_views_with_scores)
{
  nbv_3d_cnn::Predict3dGoal goal;
  goal.frontier = frontier.ToFloat32MultiArray();
  goal.empty = empty.ToFloat32MultiArray();

  ROS_INFO("simulate_nbv_cycle: sending goal...");
  m_raw_data.reset();
  m_predict_3d_action_client->sendGoal(goal);

  ROS_INFO("simulate_nbv_cycle: waiting for result...");
  bool finished_before_timeout = m_predict_3d_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout || m_predict_3d_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("simulate_nbv_cycle: action did not succeed.");
    return false;
  }

  ROS_INFO("simulate_nbv_cycle: got result.");
  nbv_3d_cnn::Predict3dResult result = *(m_predict_3d_action_client->getResult());
  if (result.scores.data.empty())
  {
    ROS_INFO("simulate_nbv_cycle: waiting for raw data.");
    ros::Rate rate(100);
    while (!m_raw_data)
    {
      m_raw_data_callback_queue.callAvailable(ros::WallDuration());
      rate.sleep();
    }
    result.scores.data = m_raw_data->data;
    m_raw_data.reset();
    ROS_INFO("simulate_nbv_cycle: got raw data.");
  }
  m_last_scores = *Voxelgrid::FromFloat32MultiArray(result.scores);

  Eigen::Vector3f max_origin = Eigen::Vector3f::Zero();
  Eigen::Quaternionf max_orientation = Eigen::Quaternionf::Identity();

  const uint64 submatrix_side = (m_mode == MODE_OV || m_mode == MODE_OV_DIRECT) ? 4 : 1;

  const uint64 width = m_last_scores.GetWidth() / submatrix_side;
  const uint64 height = m_last_scores.GetHeight() / submatrix_side;
  const uint64 depth = m_last_scores.GetDepth() / submatrix_side;

  ROS_INFO("simulate_nbv_cycle: finding best score.");
  FloatVector gains;
  float max_score = -1.0f;
  for (uint64 z = 0; z < depth; z += m_accuracy_skip)
    for (uint64 y = 0; y < height; y += m_accuracy_skip)
      for (uint64 x = 0; x < width; x += m_accuracy_skip)
      {
        const Eigen::Vector3i origin_i = Eigen::Vector3i(x, y, z) * int(m_cnn_accuracy_skip);
        if (!empty.at(origin_i))
          continue; // place sensor only in known empty

        ViewWithScore vws;

        Eigen::Quaternionf orient = Eigen::Quaternionf::Identity();
        float score = 0;
        const Eigen::Vector3f origin = origin_i.cast<float>();
        vws.origin = origin;

        if (m_mode == MODE_OV || m_mode == MODE_OV_DIRECT)
        {
          Voxelgrid submatrix = *m_last_scores.GetSubmatrix(
              Eigen::Vector3i(x * submatrix_side, y * submatrix_side, z * submatrix_side),
              Eigen::Vector3i(submatrix_side, submatrix_side, submatrix_side));

          OriginVisibility ov = OriginVisibility::FromVoxelgrid(Eigen::Vector3f(x, y, z),
                                                                submatrix_side, submatrix);
          const QuaternionfVector orientations = OriginVisibility::GenerateStandardOrientationSet(8);

          orient = ov.GetBestSensorOrientationOCL(&m_opencl, orientations, m_sensor_hfov, score, gains);

          if (all_views_with_scores)
          {
            for (uint64 orient_i = 0; orient_i < orientations.size(); orient_i++)
            {
              vws.orientation = orientations[orient_i];
              vws.score = gains[orient_i];
              all_views_with_scores->push_back(vws);
            }
          }
        }
        else if (m_mode == MODE_FLAT)
        {
          const QuaternionfVector orientations = OriginVisibility::GenerateStandardOrientationSet(8);

          const Voxelgrid submatrix = *m_last_scores.GetSubmatrix(
            Eigen::Vector3i(x * orientations.size(), y, z), Eigen::Vector3i(orientations.size(), 1, 1));

          float ms = 0;
          uint64 mi = 0;
          for (uint64 i = 0; i < orientations.size(); i++)
          {
            const float maybe_ms = submatrix.at(i, 1, 1);
            if (mi == 0 || maybe_ms > ms)
            {
              ms = maybe_ms;
              mi = i;
            }

            if (all_views_with_scores)
            {
              vws.orientation = orientations[i];
              vws.score = maybe_ms;
              all_views_with_scores->push_back(vws);
            }
          }

          score = ms;
          orient = orientations[mi];
        }
        else
          ROS_ERROR("CNNOriginVisibilityNBVAdapter::GetNextBestView3d: unsupported mode.");

        if (max_score >= score)
          continue;

        {
          bool should_be_skipped = false;
          for (uint64 i = 0; i < skip_origins.size() && !should_be_skipped; i++)
          {
            if (skip_orentations[i].vec() == orient.vec() &&
                skip_orentations[i].w() == orient.w() && skip_origins[i] == origin)
              should_be_skipped = true;
          }
          if (should_be_skipped)
            continue;
        }

        max_score = score;
        max_origin = origin;
        max_orientation = orient;
      }

  origin = max_origin;
  orientation = max_orientation;

  return max_score > 0.0f;
}

bool CNNDirectionalNBVAdapter::GetNextBestView2d(const Voxelgrid & environment,
                                                 const Voxelgrid & empty,
                                                 const Voxelgrid & occupied,
                                                 const Voxelgrid & frontier,
                                                 const Vector3fVector & skip_origins,
                                                 const QuaternionfVector & skip_orentations,
                                                 Eigen::Vector3f & origin,
                                                 Eigen::Quaternionf & orientation,
                                                 ViewWithScoreVector * const all_views_with_scores)
{
  const uint64 width = empty.GetWidth();
  const uint64 height = empty.GetHeight();

  nbv_3d_cnn::PredictGoal goal;

  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->encoding = "mono8";
    cv_ptr->header.stamp = ros::Time::now();
    cv_ptr->image = empty.ToOpenCVImage2D();
    goal.empty = *cv_ptr->toImageMsg();
    cv_ptr->image = frontier.ToOpenCVImage2D();
    goal.frontier = *cv_ptr->toImageMsg();
  }

  ROS_INFO("simulate_nbv_cycle: sending goal...");
  m_predict_action_client->sendGoal(goal);

  ROS_INFO("simulate_nbv_cycle: waiting for result...");
  bool finished_before_timeout = m_predict_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout || m_predict_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("simulate_nbv_cycle: action did not succeed.");
    return false;
  }

  ROS_INFO("simulate_nbv_cycle: got result.");
  const nbv_3d_cnn::PredictResult result = *(m_predict_action_client->getResult());

  const sensor_msgs::Image & scores_msg = result.scores;

  cv::Mat cv_scores;
  {
    cv_bridge::CvImagePtr bridge = cv_bridge::toCvCopy(scores_msg);
    cv_scores = bridge->image;
  }
  m_last_scores = *Voxelgrid::FromOpenCVImage2DFloat(cv_scores);

  Eigen::Vector3f max_origin;
  Eigen::Quaternionf max_orientation;

  ROS_INFO("simulate_nbv_cycle: finding best score.");
  const float submatrix_side = m_last_scores.GetWidth() / empty.GetWidth();
  float max_score = -1.0f;
  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (!empty.at(x, y, 0))
        continue; // place sensor only in known empty

      Voxelgrid submatrix = *m_last_scores.GetSubmatrix(Eigen::Vector3i(x * submatrix_side, y * submatrix_side, 0),
                                                 Eigen::Vector3i(submatrix_side, submatrix_side, 1));
      OriginVisibility ov = OriginVisibility::FromVisibilityMatrix(Eigen::Vector3f(x, y, 0),
                                                                   submatrix_side, submatrix.ToOpenCVImage2D());
      if (m_mode == MODE_OV)
        ov = ov.SmoothByHFOV(m_sensor_hfov.x());

      Eigen::Vector3i index;
      Eigen::Quaternionf orient;
      const float score = ov.MaxVisibility(index, orient);
      const Eigen::Vector3f origin = Eigen::Vector3f(x, y, 0);

      if (max_score >= score)
        continue;

      {
        bool should_be_skipped = false;
        for (uint64 i = 0; i < skip_origins.size() && !should_be_skipped; i++)
        {
          if (skip_orentations[i].dot(orient) > 0.99f && skip_origins[i] == origin)
            should_be_skipped = true;
        }
        if (should_be_skipped)
          continue;
      }

      max_score = score;
      max_origin = origin;
      max_orientation = orient;
    }

  origin = max_origin;
  orientation = max_orientation;

  return max_score > 0.0f;
}

bool CNNDirectionalNBVAdapter::GetNextBestView(const Voxelgrid & environment,
                                               const Voxelgrid & empty,
                                               const Voxelgrid & occupied,
                                               const Voxelgrid & frontier,
                                               const Vector3fVector & skip_origins,
                                               const QuaternionfVector & skip_orentations,
                                               Eigen::Vector3f & origin,
                                               Eigen::Quaternionf &orientation,
                                               ViewWithScoreVector * const all_views_with_scores)
{
  if (m_is_3d)
    return GetNextBestView3d(environment, empty, occupied, frontier, skip_origins, skip_orentations, origin, orientation,
                             all_views_with_scores);
  else
    return GetNextBestView2d(environment, empty, occupied, frontier, skip_origins, skip_orentations, origin, orientation,
                             all_views_with_scores);
}

void CNNQuatNBVAdapter::onRawData(const nbv_3d_cnn::FloatsConstPtr raw_data)
{
  m_raw_data = raw_data;
  ROS_INFO("CNNScoreAngleNBVAdapter: got raw data.");
}

CNNQuatNBVAdapter::CNNQuatNBVAdapter(ros::NodeHandle & nh, const bool is_3d,
                                     const uint64_t accuracy_skip,
                                     const uint64_t cnn_accuracy_skip):
  m_nh(nh), m_private_nh("~")
{
  std::string param_string;

  m_is_3d = is_3d;
  m_accuracy_skip = accuracy_skip;
  m_cnn_accuracy_skip = cnn_accuracy_skip;

  m_nh.param<std::string>(PARAM_NAME_PREDICT_QUAT_ACTION_NAME, param_string,
                          PARAM_DEFAULT_PREDICT_QUAT_ACTION_NAME);
  if (!m_is_3d)
    m_predict_action_client.reset(new PredictActionClient(param_string, true));
  else
    m_predict_3d_action_client.reset(new Predict3dActionClient(param_string, true));

  m_private_nh.setCallbackQueue(&m_raw_data_callback_queue);
  m_raw_data_subscriber = m_private_nh.subscribe(param_string + "raw_data", 1, &CNNQuatNBVAdapter::onRawData, this);

  ROS_INFO("simulate_nbv_cycle: CNNScoreAngleNBVAdapter: waiting for prediction server...");
  if (!m_is_3d)
    m_predict_action_client->waitForServer();
  else
    m_predict_3d_action_client->waitForServer();
  ROS_INFO("simulate_nbv_cycle: CNNScoreAngleNBVAdapter: prediction server ok.");
}

bool CNNQuatNBVAdapter::GetNextBestView3d(const Voxelgrid & environment,
                                          const Voxelgrid & empty,
                                          const Voxelgrid & occupied,
                                          const Voxelgrid & frontier,
                                          const Vector3fVector & skip_origins,
                                          const QuaternionfVector &skip_orentations,
                                          Eigen::Vector3f & origin,
                                          Eigen::Quaternionf &orientation,
                                          ViewWithScoreVector * const all_views_with_scores)
{
  nbv_3d_cnn::Predict3dGoal goal;

  goal.empty = empty.ToFloat32MultiArray();
  goal.frontier = frontier.ToFloat32MultiArray();

  ROS_INFO("simulate_nbv_cycle: sending goal...");
  m_raw_data.reset();
  m_predict_3d_action_client->sendGoal(goal);

  ROS_INFO("simulate_nbv_cycle: waiting for result...");
  bool finished_before_timeout = m_predict_3d_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout || m_predict_3d_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("simulate_nbv_cycle: action did not succeed.");
    return false;
  }

  ROS_INFO("simulate_nbv_cycle: got result.");
  nbv_3d_cnn::Predict3dResult result = *(m_predict_3d_action_client->getResult());
  if (result.scores.data.empty())
  {
    ROS_INFO("simulate_nbv_cycle: waiting for raw data.");
    ros::Rate rate(100);
    while (!m_raw_data)
    {
      m_raw_data_callback_queue.callAvailable(ros::WallDuration());
      rate.sleep();
    }
    result.scores.data = m_raw_data->data;
    m_raw_data.reset();
    ROS_INFO("simulate_nbv_cycle: got raw data.");
  }

  m_last_scores = *Voxelgrid4::FromFloat32MultiArray(result.scores);

  const uint64 width = m_last_scores.GetWidth();
  const uint64 height = m_last_scores.GetHeight();
  const uint64 depth = m_last_scores.GetDepth();

  Eigen::Vector3f max_origin = Eigen::Vector3f::Zero();
  Eigen::Quaternionf max_orientation = Eigen::Quaternionf::Identity();

  ROS_INFO("simulate_nbv_cycle: finding best score.");
  float max_score = 0.0f;
  for (uint64 z = 0; z < depth; z += m_accuracy_skip)
    for (uint64 y = 0; y < height; y += m_accuracy_skip)
      for (uint64 x = 0; x < width; x += m_accuracy_skip)
      {
        const Eigen::Vector3i origin_i(x * m_cnn_accuracy_skip, y * m_cnn_accuracy_skip, z * m_cnn_accuracy_skip);
        if (!empty.at(origin_i))
          continue; // place sensor only in known empty

        const float qx = m_last_scores.at(0).at(x, y, z);
        const float qy = m_last_scores.at(1).at(x, y, z);
        const float qz = m_last_scores.at(2).at(x, y, z);
        const float qw = m_last_scores.at(3).at(x, y, z);
        const Eigen::Vector4f xyzw(qx, qy, qz, qw);

        const float score = xyzw.norm();

        if (max_score >= score)
          continue;

        const Eigen::Quaternionf q(xyzw.normalized());

        {
          bool should_be_skipped = false;
          for (uint64 i = 0; i < skip_origins.size() && !should_be_skipped; i++)
          {
            const float EPSILON = 0.0001f;
            if ((skip_orentations[i].vec() - q.vec()).norm() < (3.0f * EPSILON) &&
                std::abs(skip_orentations[i].w() - q.w()) < EPSILON &&
                skip_origins[i] == origin_i.cast<float>())
              should_be_skipped = true;
          }
          if (should_be_skipped)
            continue;
        }

        max_score = score;
        max_origin = origin_i.cast<float>();
        max_orientation = q;
      }

  origin = max_origin;
  orientation = max_orientation;

  return max_score > 0.0f;
}

bool CNNQuatNBVAdapter::GetNextBestView(const Voxelgrid & environment,
                                        const Voxelgrid & empty,
                                        const Voxelgrid & occupied,
                                        const Voxelgrid & frontier,
                                        const Vector3fVector & skip_origins,
                                        const QuaternionfVector &skip_orentations,
                                        Eigen::Vector3f & origin,
                                        Eigen::Quaternionf &orientation,
                                        ViewWithScoreVector * const all_views_with_scores)
{
  if (m_is_3d)
    return GetNextBestView3d(environment, empty, occupied, frontier, skip_origins, skip_orentations, origin, orientation,
                             all_views_with_scores);
  else
    return GetNextBestView2d(environment, empty, occupied, frontier, skip_origins, skip_orentations, origin, orientation,
                             all_views_with_scores);
}

bool CNNQuatNBVAdapter::GetNextBestView2d(const Voxelgrid & environment,
                                          const Voxelgrid & empty,
                                          const Voxelgrid & occupied,
                                          const Voxelgrid & frontier,
                                          const Vector3fVector & skip_origins,
                                          const QuaternionfVector &skip_orentations,
                                          Eigen::Vector3f & origin,
                                          Eigen::Quaternionf &orientation,
                                          ViewWithScoreVector * const all_views_with_scores)
{

  const uint64 width = empty.GetWidth();
  const uint64 height = empty.GetHeight();

  nbv_3d_cnn::PredictGoal goal;

  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->encoding = "mono8";
    cv_ptr->header.stamp = ros::Time::now();
    cv_ptr->image = empty.ToOpenCVImage2D();
    goal.empty = *cv_ptr->toImageMsg();
    cv_ptr->image = frontier.ToOpenCVImage2D();
    goal.frontier = *cv_ptr->toImageMsg();
  }

  ROS_INFO("simulate_nbv_cycle: sending goal...");
  m_predict_action_client->sendGoal(goal);

  ROS_INFO("simulate_nbv_cycle: waiting for result...");
  bool finished_before_timeout = m_predict_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout || m_predict_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("simulate_nbv_cycle: action did not succeed.");
    return false;
  }

  ROS_INFO("simulate_nbv_cycle: got result.");
  const nbv_3d_cnn::PredictResult result = *(m_predict_action_client->getResult());

  const sensor_msgs::Image & scores_msg = result.scores;

  cv::Mat cv_scores;
  {
    cv_bridge::CvImagePtr bridge = cv_bridge::toCvCopy(scores_msg);
    cv_scores = bridge->image;
  }
  m_last_scores = *Voxelgrid4::FromOpenCVImage2D(cv_scores);

  Eigen::Vector3f max_origin, max_orientation;

  ROS_INFO("simulate_nbv_cycle: finding best score.");
  float max_score = -1.0f;
  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
    {
      if (!empty.at(x, y, 0))
        continue; // place sensor only in known empty

      const float fx = (cv_scores.at<cv::Vec3f>(y, x)[0] - 0.5f) * 2.0f;
      const float fy = (cv_scores.at<cv::Vec3f>(y, x)[1] - 0.5f) * 2.0f;
      const Eigen::Vector3f xyz(fx, fy, 0);

      const float score = xyz.norm();
      const Eigen::Vector3f bearing = xyz.normalized();
      const Eigen::Vector3f origin = Eigen::Vector3f(x, y, 0.0f);

      if (max_score >= score)
        continue;

      {
        bool should_be_skipped = false;
        for (uint64 i = 0; i < skip_origins.size() && !should_be_skipped; i++)
        {
          if ((skip_orentations[i] * Eigen::Vector3f::UnitZ()).dot(bearing) > 0.99f &&
              skip_origins[i] == origin)
            should_be_skipped = true;
        }
        if (should_be_skipped)
          continue;
      }

      max_score = score;
      max_origin = origin;
      max_orientation = bearing;
    }

  origin = max_origin;
  orientation = GenerateSingleImage::Bearing2DToQuat(max_orientation);

  return max_score > 0.0f;
}

InformationGainNBVAdapter::InformationGainNBVAdapter(ros::NodeHandle & nh,
                                                     GenerateTestDatasetOpenCL & opencl,
                                                     GenerateSingleImage & generate_single_image,
                                                     const float max_range,
                                                     const float min_range,
                                                     const bool stop_at_first_hit,
                                                     const float a_priori_occupied_prob,
                                                     uint64_t view_cube_resolution,
                                                     const Eigen::Vector2f &sensor_hfov,
                                                     const bool is_omniscient,
                                                     const bool is_3d,
                                                     const uint64_t accuracy_skip,
                                                     const uint64_t sample_fixed_number_of_views):
  m_nh(nh), m_opencl(opencl), m_generate_single_image(generate_single_image)
{
  m_max_range = max_range;
  m_min_range = min_range;
  m_stop_at_first_hit = stop_at_first_hit;
  m_a_priori_occupied_prob = a_priori_occupied_prob;
  m_sensor_hfov = sensor_hfov;

  m_view_cube_resolution = view_cube_resolution;

  m_is_omniscent = is_omniscient;
  m_is_3d = is_3d;
  m_accuracy_skip = accuracy_skip;
  m_sample_fixed_number_of_views = sample_fixed_number_of_views;
}

void InformationGainNBVAdapter::ForEachEmpty(const Voxelgrid & empty, const uint64 skip_accuracy,
                                             const std::function<void(const uint64 index, const Eigen::Vector3i &)> &f) const
{
  const uint64 width = empty.GetWidth();
  const uint64 height = empty.GetHeight();
  const uint64 depth = empty.GetDepth();

  uint64 i = 0;
  for (uint64 z = 0; z < depth; z += skip_accuracy)
    for (uint64 y = 0; y < height; y += skip_accuracy)
      for (uint64 x = 0; x < width; x += skip_accuracy)
      {
        const Eigen::Vector3i xyz(x, y, z);
        if (!empty.at(xyz))
          continue;

        f(i, xyz);
        i++;
      }
}

void InformationGainNBVAdapter::GetNextBestView3DHelper(const Eigen::Vector3i & xyz,
                                                        const OriginVisibilityVector & ovv,
                                                        uint64 & counter,
                                                        Voxelgrid & not_smoothed_scores,
                                                        const Eigen::Vector3i & subrect_origin,
                                                        const Eigen::Vector3i & subrect_size,
                                                        Eigen::Quaternionf & this_orientation,
                                                        float & this_score,
                                                        const QuaternionfVector & orientations,
                                                        FloatVector & gains
                                                        ) const
{
  const OriginVisibility & nsov = ovv[counter];

  not_smoothed_scores.SetSubmatrix(subrect_origin, nsov.GetVisibilityVoxelgrid(m_view_cube_resolution));
  this_orientation = nsov.GetBestSensorOrientationOCL(&m_opencl,
                                                      orientations,
                                                      m_sensor_hfov,
                                                      this_score,
                                                      gains);

  counter++;
}

void InformationGainNBVAdapter::GetNextBestView2DHelper(const Eigen::Vector3i & xyz,
                                                        const OriginVisibilityVector & ovv,
                                                        uint64 & counter,
                                                        Voxelgrid & not_smoothed_scores,
                                                        Voxelgrid & scores,
                                                        const Eigen::Vector3i & subrect_origin,
                                                        const Eigen::Vector3i & subrect_size,
                                                        Eigen::Quaternionf & this_orientation,
                                                        float & this_score,
                                                        const QuaternionfVector & orientations
                                                        ) const
{
  const OriginVisibility & nsov = ovv[counter];

  {
    const OriginVisibility ov = nsov.SmoothByHFOV(m_sensor_hfov.x());

    cv::Mat ns_score_matrix = nsov.GetVisibilityMatrix(m_view_cube_resolution);
    not_smoothed_scores.SetSubmatrix(subrect_origin, *Voxelgrid::FromOpenCVImage2DFloat(ns_score_matrix));

    cv::Mat score_matrix = ov.GetVisibilityMatrix(m_view_cube_resolution);
    scores.SetSubmatrix(subrect_origin, *Voxelgrid::FromOpenCVImage2DFloat(score_matrix));

    Eigen::Vector3i index;
    Eigen::Vector3f bearing;
    this_score = ov.MaxVisibility(index, bearing);
    this_orientation = GenerateSingleImage::Bearing2DToQuat(bearing);
  }

  counter++;
}

bool InformationGainNBVAdapter::GetNextBestView(const Voxelgrid & environment,
                                                const Voxelgrid & empty,
                                                const Voxelgrid & occupied,
                                                const Voxelgrid & frontier,
                                                const Vector3fVector & skip_origins,
                                                const QuaternionfVector & skip_orentations,
                                                Eigen::Vector3f & origin,
                                                Eigen::Quaternionf &orientation,
                                                ViewWithScoreVector * const all_views_with_scores)
{
  const uint64 width = empty.GetWidth();
  const uint64 height = empty.GetHeight();
  const uint64 depth = empty.GetDepth();

  // evaluate using a slightly shorter max range, to prevent approximation error
  const float reduced_max_range = (1.0f - 2.0f * M_PI / (16 * 4)) * m_max_range;

  Voxelgrid known_occupied = occupied;
  if (m_is_omniscent)
  {
    Voxelgrid environment_shrink = *environment.ErodeCross(Eigen::Vector3i::Ones());
    known_occupied = *known_occupied.Or(environment_shrink);
  }

  const uint64 view_cube_resolution_if_3d = m_is_3d ? m_view_cube_resolution : uint64(1);
  const uint64 accuracy_skip_if_3d = m_is_3d ? m_accuracy_skip : uint64(1);

  Vector3fVector origins;

  ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: computing origins");

  Eigen::Vector3i roi_min = Eigen::Vector3i::Zero();
  Eigen::Vector3i roi_max = Eigen::Vector3i(width, height, depth);
  ROS_INFO_STREAM("nbv_3d_cnn: InformationGainNBVAdapter: ROI is: "
                  << roi_min.transpose() << " - " << roi_max.transpose());

  uint64 total_possible_origins_count = 0;
  ForEachEmpty(empty, m_accuracy_skip, [&](const uint64 i, const Eigen::Vector3i & xyz)
  {
    total_possible_origins_count = i;
  });
  total_possible_origins_count++;

  BoolVector valid_origins(total_possible_origins_count, true);
  if (m_sample_fixed_number_of_views)
  {
    valid_origins.assign(valid_origins.size(), false);

    for (uint64 vi = 0; vi < m_sample_fixed_number_of_views; vi++)
    {
      Uint64Vector selectable_indices;
      selectable_indices.reserve(valid_origins.size());
      for (uint64 i = 0; i < valid_origins.size(); i++)
        if (!valid_origins[i])
          selectable_indices.push_back(i);
      if (selectable_indices.empty())
        break; // all poses selected
      const uint64 selected_i = rand() % selectable_indices.size();
      const uint64 selected = selectable_indices[selected_i];
      valid_origins[selected] = true;
    }

    ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: sampled %u viewpoints of %u total viewpoints.",
             unsigned(m_sample_fixed_number_of_views), unsigned(valid_origins.size()));
  }

  ForEachEmpty(empty, m_accuracy_skip, [&](const uint64 index, const Eigen::Vector3i & xyz)
  {
    if (!valid_origins[index])
      return;
    origins.push_back(xyz.cast<float>());
  });

  if (m_last_scores.IsEmpty())
    m_last_scores = Voxelgrid(width * m_view_cube_resolution / m_accuracy_skip,
                              height * m_view_cube_resolution / m_accuracy_skip,
                              depth * view_cube_resolution_if_3d / accuracy_skip_if_3d);

  Voxelgrid not_smoothed_scores;
  Voxelgrid scores = m_last_scores;
  if (!m_last_not_smoothed_scores.IsEmpty())
    not_smoothed_scores = m_last_not_smoothed_scores;
  else
    not_smoothed_scores = Voxelgrid(width * m_view_cube_resolution / m_accuracy_skip,
                                    height * m_view_cube_resolution / m_accuracy_skip,
                                    depth * view_cube_resolution_if_3d / accuracy_skip_if_3d);

  {
    const Eigen::Vector3i sub_origin(roi_min.x() * m_view_cube_resolution / m_accuracy_skip,
                                     roi_min.y() * m_view_cube_resolution / m_accuracy_skip,
                                     roi_min.z() * view_cube_resolution_if_3d / accuracy_skip_if_3d);
    const Eigen::Vector3i sub_size((roi_max.x() - roi_min.x()) * m_view_cube_resolution / m_accuracy_skip,
                                   (roi_max.y() - roi_min.y()) * m_view_cube_resolution / m_accuracy_skip,
                                   (roi_max.z() - roi_min.z()) * view_cube_resolution_if_3d / accuracy_skip_if_3d);
    not_smoothed_scores.FillSubmatrix(sub_origin, sub_size, 0.0f);
    scores.FillSubmatrix(sub_origin, sub_size, 0.0f);
  }

  ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: evaluatemultiviewcubes");

  OriginVisibilityVector ovv;
  ovv = m_generate_single_image.InformationGainMultiViewCubes(environment, known_occupied, empty,
                                                              origins, reduced_max_range,
                                                              m_min_range,
                                                              m_stop_at_first_hit,
                                                              m_a_priori_occupied_prob);

  Eigen::Vector3f max_origin = -Eigen::Vector3f::Ones();
  Eigen::Quaternionf max_orientation = Eigen::Quaternionf::Identity();

  ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: computing scores and visibility matrix");

  QuaternionfVector orientations;
  if (m_is_3d)
  {
    orientations = OriginVisibility::GenerateStandardOrientationSet(8);
    ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: orientations: %u", unsigned(orientations.size()));
  }

  uint64 counter = 0;
  float max_score = -1.0f;
  ForEachEmpty(empty, m_accuracy_skip, [&](const uint64 index, const Eigen::Vector3i & xyz)
  {
    const uint64 x = xyz.x(), y = xyz.y(), z = xyz.z();

    if (!valid_origins[index])
      return;

    float this_score;
    Eigen::Quaternionf this_orientation;
    const Eigen::Vector3i subrect_origin(x * m_view_cube_resolution / m_accuracy_skip,
                                         y * m_view_cube_resolution / m_accuracy_skip,
                                         z * view_cube_resolution_if_3d / accuracy_skip_if_3d);
    const Eigen::Vector3i subrect_size(m_view_cube_resolution / m_accuracy_skip,
                                       m_view_cube_resolution / m_accuracy_skip,
                                       view_cube_resolution_if_3d / accuracy_skip_if_3d);

    FloatVector gains;

    if (m_is_3d)
      GetNextBestView3DHelper(xyz, ovv, counter, not_smoothed_scores, subrect_origin,
                              subrect_size, this_orientation, this_score, orientations, gains);
    else
      GetNextBestView2DHelper(xyz, ovv, counter, scores, not_smoothed_scores, subrect_origin,
                              subrect_size, this_orientation, this_score, orientations);

    const Eigen::Vector3f this_origin = Eigen::Vector3f(x, y, z);
    {
      bool should_be_skipped = false;
      for (uint64 i = 0; i < skip_origins.size() && !should_be_skipped; i++)
      {
        const float EPSILON = 0.0001f;
        if ((skip_orentations[i].vec() - this_orientation.vec()).norm() < (3.0f * EPSILON) &&
            std::abs(skip_orentations[i].w() - this_orientation.w()) < EPSILON &&
            skip_origins[i] == this_origin)
          should_be_skipped = true;
      }
      if (should_be_skipped)
        return;
    }

    if (all_views_with_scores)
    {
      ViewWithScore vws;
      vws.origin = xyz.cast<float>();
      for (uint64 i = 0; i < orientations.size(); i++)
      {
        vws.score = gains[i];
        vws.orientation = orientations[i];
        all_views_with_scores->push_back(vws);
      }
    }

    if (max_score < this_score)
    {
      max_score = this_score;
      max_origin = this_origin;
      max_orientation = this_orientation;
    }
  });

  m_last_not_smoothed_scores = not_smoothed_scores;
  m_last_scores = scores;

  ROS_INFO("nbv_3d_cnn: Generating debug image");

  origin = max_origin;
  orientation = max_orientation;

  Voxelgrid expected_observation = *environment.FilledWith(0.0f);

  if (!m_is_3d)
  {

    Eigen::Quaternionf axisq;
    {
      Eigen::Vector3f axis;
      OriginVisibility temp_ov(Eigen::Vector3f(0, 0, 0), 16, m_is_3d);
      axis = temp_ov.GetAxisFromFrame(temp_ov.BearingToFrame(max_orientation * Eigen::Vector3f::UnitZ())).col(2);
      axisq = GenerateSingleImage::Bearing2DToQuat(axis);
    }

    FloatVector dists;
    Vector3fVector ray_bearings;
    const Eigen::Vector2i res(16, m_is_3d ? 16 : 1);
    const uint64 focal = 8;
    Vector3iVector nearest = m_generate_single_image.SimulateView(environment, origin, axisq, focal, res,
                                                                  m_max_range, m_min_range, dists, ray_bearings);
    m_opencl.FillEnvironmentFromView(expected_observation, origin, axisq, focal, res, dists,
                                     nearest, expected_observation);
  }

  m_last_expected_observation = expected_observation;
  m_last_orientation = orientation;
  m_last_origin = max_origin;

  ROS_INFO("nbv_3d_cnn: InformationGainNBVAdapter: best score %f", float(max_score));

  return max_score > 0.0f;
}

cv::Mat InformationGainNBVAdapter::GetDebugImage(const Voxelgrid & environment) const
{
  const Voxelgrid leo = *(this->GetLastExpectedObservation());
  const Eigen::Quaternionf & axis = this->GetLastOrientation();
  const Eigen::Vector3f ax = axis * Eigen::Vector3f(0.0f, 0.0f, 1.0f);

  const uint64 width = environment.GetWidth();
  const uint64 height = environment.GetHeight();

  cv::Mat cv_color_leo;
  cv::Mat cv_leo = leo.ToOpenCVImage2D();
  cv::cvtColor(cv_leo, cv_color_leo, CV_GRAY2RGB);
  {
    const cv::Point origin(m_last_origin.x(), m_last_origin.y());
    const Eigen::Vector3f o2 = m_last_origin + ax * 15;
    const cv::Point dir(o2.x(), o2.y());
    cv::circle(cv_color_leo, origin,
               10, cv::Scalar(255, 0, 0), 1, cv::LINE_8);
    cv::line(cv_color_leo, origin, dir, cv::Scalar(255, 0, 0));

    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        if (environment.at(x, y, 0))
        {
          if (leo.at(x, y, 0))
            cv_color_leo.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
          else
            cv_color_leo.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 127);
        }
      }
  }

  return cv_color_leo;
}

void AutocompleteIGainNBVAdapter::onRawData(const nbv_3d_cnn::FloatsConstPtr raw_data)
{
  m_raw_data = raw_data;
  ROS_INFO("AutocompleteIGainNBVAdapter: got raw data.");
}

AutocompleteIGainNBVAdapter::AutocompleteIGainNBVAdapter(ros::NodeHandle & nh,
                                                         GenerateTestDatasetOpenCL &opencl,
                                                         GenerateSingleImage &generate_single_image,
                                                         const float max_range,
                                                         uint64_t directional_view_cube_resolution,
                                                         const Eigen::Vector2f & sensor_hfov,
                                                         const bool is_3d,
                                                         const uint64_t accuracy_skip,
                                                         const uint64_t sample_fixed_number_of_views):
  m_nh(nh), m_opencl(opencl), m_generate_single_image(generate_single_image), m_private_nh("~")
{
  m_is_3d = is_3d;

  std::string param_string;

  m_nh.param<std::string>(PARAM_NAME_PREDICT_AUTOCOMPLETE_ACTION_NAME, param_string,
                          PARAM_DEFAULT_PREDICT_AUTOCOMPLETE_ACTION_NAME);
  if (!m_is_3d)
    m_predict_action_client.reset(new PredictActionClient(param_string, true));
  else
    m_predict3d_action_client.reset(new Predict3dActionClient(param_string, true));

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: waiting for prediction server...");
  if (!m_is_3d)
    m_predict_action_client->waitForServer();
  else
    m_predict3d_action_client->waitForServer();
  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: prediction server ok.");

  m_private_nh.setCallbackQueue(&m_raw_data_callback_queue);
  m_raw_data_subscriber = m_private_nh.subscribe(param_string + "raw_data", 1, &AutocompleteIGainNBVAdapter::onRawData, this);

  m_information_gain.reset(new InformationGainNBVAdapter(nh,
                                                         opencl,
                                                         generate_single_image,
                                                         max_range,
                                                         0.0f,
                                                         false,
                                                         0.0f,
                                                         directional_view_cube_resolution,
                                                         sensor_hfov,
                                                         false,
                                                         m_is_3d,
                                                         accuracy_skip,
                                                         sample_fixed_number_of_views
                                                         ));
}

bool AutocompleteIGainNBVAdapter::Predict3d(const Voxelgrid &empty, const Voxelgrid &frontier, Voxelgrid &autocompleted)
{
  nbv_3d_cnn::Predict3dGoal goal;

  goal.empty = empty.ToFloat32MultiArray();
  goal.frontier = frontier.ToFloat32MultiArray();

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: sending goal...");
  m_raw_data.reset();
  m_predict3d_action_client->sendGoal(goal);

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: waiting for result...");
  bool finished_before_timeout = m_predict3d_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout ||
      m_predict3d_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: action did not succeed.");
    return false;
  }

  nbv_3d_cnn::Predict3dResult result = *(m_predict3d_action_client->getResult());
  if (result.scores.data.empty())
  {
    ROS_INFO("simulate_nbv_cycle: waiting for raw data.");
    ros::Rate rate(100);
    while (!m_raw_data)
    {
      m_raw_data_callback_queue.callAvailable(ros::WallDuration());
      rate.sleep();
    }
    result.scores.data = m_raw_data->data;
    m_raw_data.reset();
    ROS_INFO("simulate_nbv_cycle: got raw data.");
  }

  Voxelgrid::Ptr maybe_autocompleted = Voxelgrid::FromFloat32MultiArray(result.scores);
  if (!maybe_autocompleted)
  {
    ROS_ERROR("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: could not convert result to voxelgrid.");
    return false;
  }
  autocompleted = *maybe_autocompleted;
  return true;
}

bool AutocompleteIGainNBVAdapter::Predict(const Voxelgrid & empty, const Voxelgrid & frontier, Voxelgrid & autocompleted)
{
  nbv_3d_cnn::PredictGoal goal;

  {
    cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
    cv_ptr->encoding = "mono8";
    cv_ptr->header.stamp = ros::Time::now();
    cv_ptr->image = empty.ToOpenCVImage2D();
    goal.empty = *cv_ptr->toImageMsg();
    cv_ptr->image = frontier.ToOpenCVImage2D();
    goal.frontier = *cv_ptr->toImageMsg();
  }

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: sending goal...");
  m_predict_action_client->sendGoal(goal);

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: waiting for result...");
  bool finished_before_timeout = m_predict_action_client->waitForResult(ros::Duration(30.0));
  if (!finished_before_timeout ||
      m_predict_action_client->getState() != actionlib::SimpleClientGoalState::SUCCEEDED)
  {
    ROS_ERROR("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: action did not succeed.");
    return false;
  }

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: got result.");
  const nbv_3d_cnn::PredictResult result = *(m_predict_action_client->getResult());

  const sensor_msgs::Image & autocompleted_msg = result.scores;

  cv::Mat cv_autocompleted;
  {
    cv_bridge::CvImagePtr bridge = cv_bridge::toCvCopy(autocompleted_msg);
    cv_autocompleted = bridge->image;
  }

  autocompleted = *Voxelgrid::FromOpenCVImage2DFloat(cv_autocompleted);

  return true;
}

bool AutocompleteIGainNBVAdapter::GetNextBestView(const Voxelgrid & environment,
                                                  const Voxelgrid & empty,
                                                  const Voxelgrid & occupied,
                                                  const Voxelgrid & frontier,
                                                  const Vector3fVector & skip_origins,
                                                  const QuaternionfVector &skip_orentations,
                                                  Eigen::Vector3f & origin,
                                                  Eigen::Quaternionf &orientation,
                                                  ViewWithScoreVector * const all_views_with_scores)
{


  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: processing result.");

  Voxelgrid autocompleted;
  if (m_is_3d)
  {
    if (!Predict3d(empty, occupied, autocompleted))
      return false;
  }
  else
  {
    if (!Predict(empty, occupied, autocompleted))
      return false;
  }
  autocompleted.Clamp(0.0f, 1.0f); // clamp [0,1]
  m_last_autocompleted_image = autocompleted;

//  autocompleted.Multiply(1.0f - m_probability_cnn_prediction_wrong);
//  autocompleted.Add(m_probability_cnn_prediction_wrong * m_a_priori_occupied_prob);
  autocompleted = *autocompleted.Or(occupied);
  autocompleted = *autocompleted.AndNot(empty);

  ROS_INFO("simulate_nbv_cycle: AutocompleteIGainNBVAdapter: computing information gain.");

  return m_information_gain->GetNextBestView(environment, empty, autocompleted, frontier,
                                             skip_origins, skip_orentations, origin, orientation,
                                             all_views_with_scores);
}
