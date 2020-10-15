#include "simulate_nbv_cycle.h"

// ROS
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <actionlib/client/simple_action_client.h>

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
#include <memory>
#include <fstream>

// custom
#include "generate_test_dataset_opencl.h"
#include "origin_visibility.h"
#include "generate_single_image.h"
#include "simulate_nbv_cycle_adapter.h"
#include "voxelgrid.h"
#include <nbv_3d_cnn/PredictAction.h>

class SimulateNBVCycle
{
  public:
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<float> FloatVector;
  typedef uint8_t uint8;
  typedef uint64_t uint64;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn::PredictAction> PredictActionClient;
  typedef std::shared_ptr<PredictActionClient> PredictActionClientPtr;

  SimulateNBVCycle(ros::NodeHandle & nh):
    m_nh(nh), m_opencl(m_nh)
  {
    std::string param_string;
    double param_double;
    int param_int;

    m_nh.param<std::string>(PARAM_NAME_3D_MODE, param_string, PARAM_DEFAULT_3D_MODE);
    if (param_string == PARAM_VALUE_3D_MODE_2D)
      m_is_3d = false;
    else if (param_string == PARAM_VALUE_3D_MODE_3D)
      m_is_3d = true;
    else
    {
      ROS_ERROR("simulate_nbv_cycle: invalid value for parameter %s: %s", PARAM_NAME_3D_MODE, param_string.c_str());
      m_is_3d = false;
    }

    m_nh.param<std::string>(PARAM_NAME_IMAGE_FILE_NAME, m_image_file_name, PARAM_DEFAULT_IMAGE_FILE_NAME);

    m_nh.param<bool>(PARAM_NAME_SAVE_IMAGES, m_save_images, PARAM_DEFAULT_SAVE_IMAGES);

    m_nh.param<std::string>(PARAM_NAME_DEBUG_OUTPUT_FOLDER, m_debug_output_folder, PARAM_DEFAULT_DEBUG_OUTPUT_FOLDER);

    m_nh.param<double>(PARAM_NAME_SENSOR_FOCAL_LENGTH, param_double, PARAM_DEFAULT_SENSOR_FOCAL_LENGTH);
    m_sensor_focal_length = param_double;

    m_nh.param<int>(PARAM_NAME_SENSOR_RESOLUTION_X, param_int, PARAM_DEFAULT_SENSOR_RESOLUTION_X);
    m_sensor_resolution_x = param_int;

    m_nh.param<int>(PARAM_NAME_SENSOR_RESOLUTION_Y, param_int, PARAM_DEFAULT_SENSOR_RESOLUTION_Y);
    m_sensor_resolution_y = param_int;

    m_nh.param<double>(PARAM_NAME_SENSOR_RANGE_VOXELS, param_double, PARAM_DEFAULT_SENSOR_RANGE_VOXELS);
    m_sensor_range_voxels = param_double;

    m_nh.param<double>(PARAM_NAME_A_PRIORI_OCCUPIED_PROB, param_double, PARAM_DEFAULT_A_PRIORI_OCCUPIED_PROB);
    m_a_priori_occupied_prob = param_double;

    m_nh.param<std::string>(PARAM_NAME_LOG_FILE_NAME, param_string, PARAM_DEFAULT_LOG_FILE_NAME);
    m_log_file_name = param_string;

    m_nh.param<int>(PARAM_NAME_MAX_ITERATIONS, param_int, PARAM_DEFAULT_MAX_ITERATIONS);
    m_max_iterations = param_int;

    m_nh.param<double>(PARAM_NAME_AUTOCOMPLETE_IGAIN_RESCALE_MARGIN, param_double,
                       PARAM_DEFAULT_AUTOCOMPLETE_IGAIN_RESCALE_MARGIN);
    m_autocomplete_igain_rescale_margin = param_double;

    m_nh.param<double>(PARAM_NAME_IGAIN_MIN_RANGE, param_double, PARAM_DEFAULT_IGAIN_MIN_RANGE);
    m_igain_min_range_voxels = param_double;

    m_nh.param<int>(PARAM_NAME_ACCURACY_SKIP_VOXELS, param_int, PARAM_DEFAULT_ACCURACY_SKIP_VOXELS);
    m_accuracy_skip_voxels = param_int;

    m_nh.param<std::string>(PARAM_NAME_KNOWN_EMPTY_FILE_NAME, m_known_empty_file_name, PARAM_DEFAULT_KNOWN_EMPTY_FILE_NAME);
    m_nh.param<std::string>(PARAM_NAME_KNOWN_OCCUPIED_FILE_NAME, m_known_occupied_file_name, PARAM_DEFAULT_KNOWN_OCCUPIED_FILE_NAME);

    {
      int view_cube_resolution;
      m_nh.param<int>(PARAM_NAME_VIEW_CUBE_RESOLUTION, view_cube_resolution, PARAM_DEFAULT_VIEW_CUBE_RESOLUTION);

      int submatrix_resolution;
      m_nh.param<int>(PARAM_NAME_SUBMATRIX_RESOLUTION, submatrix_resolution,
                      PARAM_DEFAULT_SUBMATRIX_RESOLUTION);

      m_generate_single_image.reset(new GenerateSingleImage(m_nh, m_opencl, m_is_3d, m_sensor_range_voxels,
                                                            m_sensor_resolution_x, m_sensor_resolution_y,
                                                            m_sensor_focal_length, view_cube_resolution,
                                                            submatrix_resolution));
    }

    m_first_loop = true;
    m_loop_counter = 0;

    const Eigen::Vector2f sensor_hfov(std::atan2(float(m_sensor_resolution_x) / 2.0f, float(m_sensor_focal_length)),
                                      std::atan2(float(m_sensor_resolution_y) / 2.0f, float(m_sensor_focal_length)));

    m_nh.param<std::string>(PARAM_NAME_NBV_ALGORITHM, param_string, PARAM_DEFAULT_NBV_ALGORITHM);
    if (param_string == PARAM_VALUE_NBV_ALGORITHM_RANDOM)
      m_current_adapter.reset(new RandomNBVAdapter(m_nh, m_is_3d));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_CNNDirectional)
      m_current_adapter.reset(new CNNDirectionalNBVAdapter(m_nh, m_opencl, m_is_3d, sensor_hfov,
                                                                CNNDirectionalNBVAdapter::MODE_OV,
                                                                m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_CNNDirectDirectional)
      m_current_adapter.reset(new CNNDirectionalNBVAdapter(m_nh, m_opencl, m_is_3d, sensor_hfov,
                                                                CNNDirectionalNBVAdapter::MODE_OV_DIRECT,
                                                                m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_CNNFlat)
      m_current_adapter.reset(new CNNDirectionalNBVAdapter(m_nh, m_opencl, m_is_3d,  sensor_hfov,
                                                                CNNDirectionalNBVAdapter::MODE_FLAT,
                                                                m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_CNNQuat)
      m_current_adapter.reset(new CNNQuatNBVAdapter(m_nh, m_is_3d, m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_InformationGain)
      m_current_adapter.reset(new InformationGainNBVAdapter(m_nh, m_opencl, *m_generate_single_image,
                                                            m_sensor_range_voxels, m_igain_min_range_voxels, true, 0.0,
                                                            4, sensor_hfov, false, m_is_3d, m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_InformationGainProb)
      m_current_adapter.reset(new InformationGainNBVAdapter(m_nh, m_opencl, *m_generate_single_image,
                                                            m_sensor_range_voxels, 0.0f, false, m_a_priori_occupied_prob,
                                                            4, sensor_hfov, false, m_is_3d, m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_AutocompleteIGain)
      m_current_adapter.reset(new AutocompleteIGainNBVAdapter(m_nh, m_opencl, *m_generate_single_image,
                                                              m_sensor_range_voxels,
                                                              4, sensor_hfov, m_is_3d, m_accuracy_skip_voxels));
    else if (param_string == PARAM_VALUE_NBV_ALGORITHM_OmniscientGain)
      m_current_adapter.reset(new InformationGainNBVAdapter(m_nh, m_opencl, *m_generate_single_image,
                                                            m_sensor_range_voxels, 0.0f, false, 0.0,
                                                            4, sensor_hfov, true, m_is_3d, m_accuracy_skip_voxels));
    else
    {
      ROS_FATAL("simulate_nbv_cycle: unknown algorithm %s", param_string.c_str());
      exit(1);
    }

    m_timer = m_nh.createTimer(ros::Duration(0.0), &SimulateNBVCycle::onTimer, this, true);
  }

  Eigen::Vector3f FindAvailableOrigin(const Voxelgrid & environment)
  {
    const uint64 width = environment.GetWidth();
    const uint64 height = environment.GetHeight();
    const uint64 depth = environment.GetDepth();

    uint64 count = 0;
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          if (!environment.at(x, y, z))
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
          if (!environment.at(x, y, z))
          {
            if (count == selected)
              return Eigen::Vector3f(x, y, z);
            count++;
          }
        }

    return Eigen::Vector3f(NAN, NAN, NAN);
  }

  void onTimer(const ros::TimerEvent &)
  {
    const std::string test_folder = m_debug_output_folder;

    Eigen::Vector3f max_origin;
    Eigen::Quaternionf max_orientation;

    Voxelgrid cv_grayscale_scores = *m_environment_image.FilledWith(0.0f);
    Voxelgrid4 cv_color_scores(m_environment_image.GetSize());

    bool success = true;
    double computation_time;

    if (m_first_loop)
    {
      const std::string environment_filename = m_image_file_name;
      ROS_INFO("simulate_nbv_cycle: loading image: \"%s\"", environment_filename.c_str());
      Voxelgrid::Ptr env_ptr;
      if (!m_is_3d)
        env_ptr = Voxelgrid::Load2DOpenCV(environment_filename);
      else
        env_ptr = Voxelgrid::Load3DOctomap(environment_filename);
      if (!env_ptr)
      {
        ROS_ERROR("simulate_nbv_cycle: error while loading image.");
        return;
      }
      m_environment_image = *env_ptr;

    }

    bool preloaded = false;
    if (m_first_loop && m_known_empty_file_name != "" && m_known_occupied_file_name != "")
    {
      ROS_INFO("simulate_nbv_cycle: initializing from empty file name: %s", m_known_empty_file_name.c_str());
      ROS_INFO("simulate_nbv_cycle: initializing from occupied file name: %s", m_known_occupied_file_name.c_str());
      if (!m_is_3d)
      {
        m_current_observed_empty = *Voxelgrid::Load2DOpenCV(m_known_empty_file_name);
        m_current_observed_occupied = *Voxelgrid::Load2DOpenCV(m_known_occupied_file_name);
      }
      else
      {
        m_current_observed_empty = *Voxelgrid::Load3DOctomap(m_known_empty_file_name);
        m_current_observed_occupied = *Voxelgrid::Load3DOctomap(m_known_occupied_file_name);
      }

      m_current_frontier = *m_generate_single_image->FrontierFromObservedAndEmpty(m_current_observed_occupied,
                                                                                  m_current_observed_empty);
      preloaded = true;
    }

    if (m_first_loop && !preloaded)
    {
      m_current_observed_empty = *m_environment_image.FilledWith(0.0f);
      m_current_observed_occupied = *m_environment_image.FilledWith(0.0f);
      m_current_frontier = *m_environment_image.FilledWith(0.0f);

      Vector3fVector origin_v;
      QuaternionfVector orientation_v;
      for (uint64 i = 0; i < 1; i++)
      {
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
        }
        max_origin = FindAvailableOrigin(m_environment_image);

        origin_v.push_back(max_origin);
        orientation_v.push_back(orientation);

        max_orientation = orientation;
      }

      cv_grayscale_scores = m_current_observed_empty;

      m_first_loop = false;
      success = true;
      computation_time = 0;
    }
    else
    {
      if (dynamic_cast<InformationGainNBVAdapter *>(m_current_adapter.get()))
      {
        InformationGainNBVAdapter * const ogn = dynamic_cast<InformationGainNBVAdapter *>(m_current_adapter.get());
        if (ogn->IsOmniscient() && !m_prev_origins.empty())
          ogn->SetROIByLastOrigin(m_prev_origins.back(), m_prev_orientations.back());
      }

      //if (dynamic_cast<AutocompleteIGainNBVAdapter *>(m_current_adapter.get()))
      //{
      //  AutocompleteIGainNBVAdapter * const ogn = dynamic_cast<AutocompleteIGainNBVAdapter *>(m_current_adapter.get());
      //  if (!m_prev_origins.empty())
      //    ogn->SetROIByLastOrigin(m_prev_origins.back(), m_prev_orientations.back());
      //}


      ros::Time start = ros::Time::now();
      success = m_current_adapter->GetNextBestView(m_environment_image,
                                                   m_current_observed_empty,
                                                   m_current_observed_occupied,
                                                   m_current_frontier,
                                                   m_prev_origins,
                                                   m_prev_orientations,
                                                   max_origin,
                                                   max_orientation);
      ros::Duration elapsed = ros::Time::now() - start;
      computation_time = elapsed.toSec();

      cv_grayscale_scores = m_current_adapter->GetScores();
      cv_color_scores = m_current_adapter->GetColorScores();

      if (dynamic_cast<InformationGainNBVAdapter *>(m_current_adapter.get()))
      {
        InformationGainNBVAdapter * const ogn = dynamic_cast<InformationGainNBVAdapter *>(m_current_adapter.get());
        cv::Mat cv_color_leo = ogn->GetDebugImage(m_environment_image);
        cv::imwrite(test_folder + "leo_" + std::to_string(m_loop_counter) + "_leo.png", cv_color_leo);

        Voxelgrid::ConstPtr not_smoothed_scores = ogn->GetLastNotSmoothedScores();
        not_smoothed_scores->Save2D3D(test_folder + std::to_string(m_loop_counter) +
                                               "_last_not_smoothed", m_is_3d);
      }

      if (dynamic_cast<AutocompleteIGainNBVAdapter *>(m_current_adapter.get()))
      {
        AutocompleteIGainNBVAdapter * const ogn = dynamic_cast<AutocompleteIGainNBVAdapter *>(m_current_adapter.get());

        Voxelgrid autocompleted_scores = ogn->GetLastAutocompletedImage();
        autocompleted_scores.Save2D3D(test_folder + std::to_string(m_loop_counter) + "_autocompleted", m_is_3d);

        cv::Mat debug_image = ogn->GetDebugImage(m_environment_image);
        cv::imwrite(test_folder + "leo_" + std::to_string(m_loop_counter) + "_leo.png", debug_image);

        Voxelgrid::ConstPtr not_smoothed_scores = ogn->GetLastNotSmoothedScores();
        not_smoothed_scores->Save2D3D(test_folder + "" + std::to_string(m_loop_counter) +
                                      "_last_not_smoothed", m_is_3d);
      }
    }

    const uint64 width = m_environment_image.GetWidth();
    const uint64 height = m_environment_image.GetHeight();
    const uint64 depth = m_environment_image.GetDepth();

    ROS_INFO_STREAM("simulate_nbv_cycle: best score at " << max_origin.transpose() <<
                    " orientation " << max_orientation.vec().transpose() << " " << max_orientation.w() <<
                    " success: " << (success ? "TRUE" : "FALSE"));

    m_current_frontier = *m_generate_single_image->FrontierFromObservedAndEmpty(m_current_observed_occupied,
                                                                                m_current_observed_empty);

    ROS_INFO("simulate_nbv_cycle: saving images");

    if (m_save_images)
    {
      cv::Mat cv_color_empty = m_current_observed_empty.ToOpenCVImage2D();
      cv::cvtColor(cv_color_empty, cv_color_empty, CV_GRAY2RGB);
      {
        for (uint64 y = 0; y < height; y++)
          for (uint64 x = 0; x < width; x++)
          {
            if (m_current_observed_occupied.at(x, y, 0))
              cv_color_empty.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            else if (m_environment_image.at(x, y, 0))
              cv_color_empty.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 127);
          }

        for (uint64 i = 0; i <= m_prev_origins.size(); i++)
        {
          Eigen::Vector3f symbol_origin;
          Eigen::Quaternionf symbol_orientation;

          cv::Scalar color = cv::Scalar(127, 127, 127);

          if (i == m_prev_origins.size())
          {
            symbol_origin = max_origin;
            symbol_orientation = max_orientation;
            color = cv::Scalar(255, 0, 0);
          }
          else
          {
            symbol_orientation = m_prev_orientations[i];
            symbol_origin = m_prev_origins[i];
          }

          const int THICKNESS = 2;

          const cv::Point origin(symbol_origin.x(), symbol_origin.y());
          const Eigen::Vector3f o2 = symbol_origin +
                                     Eigen::Vector3f(symbol_orientation * Eigen::Vector3f(0.0f, 0.0f, 1.0f)) * 15.0f;
          const cv::Point dir(o2.x(), o2.y());
          cv::circle(cv_color_empty, origin, 10, color, THICKNESS, cv::LINE_8);
          cv::line(cv_color_empty, origin, dir, color, THICKNESS);
        }
      }

      if (!cv_grayscale_scores.IsEmpty())
        cv_grayscale_scores.Save2D3D(test_folder + std::to_string(m_loop_counter) + "_scores", m_is_3d);
      else if (!cv_color_scores.IsEmpty())
        cv_color_scores.Save2D3D(test_folder + std::to_string(m_loop_counter) + "_scores", m_is_3d);
      cv::imwrite(test_folder + "views_" + std::to_string(m_loop_counter) + "_view.png", cv_color_empty);
      m_current_observed_empty.Save2D3D(test_folder + std::to_string(m_loop_counter) + "_empty", m_is_3d);
      m_current_frontier.Save2D3D(test_folder + std::to_string(m_loop_counter) + "_frontier", m_is_3d);
      //cv::imwrite(test_folder + "gt.png", view_cube_evaluation);
      m_current_observed_occupied.Save2D3D(test_folder + std::to_string(m_loop_counter) + "_occupied", m_is_3d);
    }

    ROS_INFO("simulate_nbv_cycle: saved image %u", unsigned(m_loop_counter));

    ROS_INFO("simulate_nbv_cycle: simulating view");
    Voxelgrid occupied;
    Voxelgrid empty = *m_generate_single_image->FillViewKnown(m_environment_image, max_origin, max_orientation,
                                                              m_sensor_focal_length,
                                                              Eigen::Vector2i(m_sensor_resolution_x, m_sensor_resolution_y),
                                                              m_sensor_range_voxels,
                                                              occupied);
//    Voxelgrid empty = *m_generate_single_image->FillViewKnownUsingCubeResolution(m_environment_image,
//                                                                                 max_origin,
//                                                                                 max_orientation,
//                                                                                 m_sensor_range_voxels,
//                                                                                 occupied);
    m_current_observed_empty = *m_current_observed_empty.Or(empty);
    m_current_observed_occupied = *m_current_observed_occupied.Or(occupied);
    m_current_frontier = *m_generate_single_image->FrontierFromObservedAndEmpty(m_current_observed_occupied,
                                                                                m_current_observed_empty);

    AppendToLog(m_loop_counter, m_environment_image, m_current_observed_empty, computation_time,
                max_origin, max_orientation);

    ROS_INFO("simulate_nbv_cycle: computing termination condition");
    bool terminated = false;

    if (!success)
    {
      ROS_INFO("simulate_nbv_cycle: zero predicted gain, reconstruction terminated.");
      terminated = true;
    }

    if (!m_current_adapter->IsRandom()) // random NBV may duplicate position
    {
      bool duplicate_found = false;
      for (uint64 i = 0; i < m_prev_orientations.size(); i++)
        if (m_prev_orientations[i].vec() == max_orientation.vec() &&
            m_prev_orientations[i].w() == max_orientation.w() &&
            m_prev_origins[i] == max_origin)
          duplicate_found = true;

      if (duplicate_found)
      {
        ROS_INFO("simulate_nbv_cycle: duplicate position, reconstruction terminated.");
        terminated = true;
      }
    }

    bool frontier_found = false;
    for (uint64 z = 0; z < depth && !frontier_found; z++)
      for (uint64 y = 0; y < height && !frontier_found; y++)
        for (uint64 x = 0; x < width && !frontier_found; x++)
        {
          if (m_current_frontier.at(x, y, z))
            frontier_found = true;
        }

    if (!frontier_found)
    {
      ROS_INFO("simulate_nbv_cycle: frontier not found, reconstruction terminated.");
      terminated = true;
    }

    if (m_loop_counter >= m_max_iterations)
    {
      ROS_INFO("simulate_nbv_cycle: %u iterations performed, reconstruction terminated.", unsigned(m_loop_counter));
      terminated = true;
    }

    m_prev_orientations.push_back(max_orientation);
    m_prev_origins.push_back(max_origin);

    m_timer.stop();
    m_loop_counter++;
    if (!terminated)
    {
      ROS_INFO("simulate_nbv_cycle: re-scheduling.");
      m_timer.start();
    }
    else
    {
      ros::shutdown();
    }
  }

  void AppendToLog(const uint64 loop_counter, const Voxelgrid & environment,
                   const Voxelgrid & current_observed_empty, const double computation_time,
                   const Eigen::Vector3f & position, const Eigen::Quaternionf & bearing)
  {
    if (m_log_file_name.empty())
      return;

    const uint64 width = environment.GetWidth();
    const uint64 height = environment.GetHeight();
    const uint64 depth = environment.GetDepth();

    uint64 current_unknown_count = 0;
    uint64 total_unknown_count = 0;
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          if (environment.at(x, y, z))
            continue;

          total_unknown_count++;

          if (current_observed_empty.at(x, y, z))
            continue;

          current_unknown_count++;
        }

    if (loop_counter == 0) // clear the file
    {
      std::ofstream ofile(m_log_file_name);
      ofile << "Iteration" << "\t" << "\"Current unknown\"" << "\t" << "\"Total unknown\"" << "\t" <<
               "\"Computation time\"" << "\n";
    }

    std::ofstream ofile(m_log_file_name, std::ios_base::app);
    ofile << loop_counter << "\t" << current_unknown_count << "\t" << total_unknown_count << "\t"
          << computation_time << "\t";
    ofile << position.x() << "\t" << position.y() << "\t" << position.z() << "\t";
    ofile << bearing.x() << "\t" << bearing.y() << "\t" << bearing.z() << "\t" << bearing.w() << "\n";

    if (!ofile)
      ROS_ERROR("simulate_nbv_cycle: could not write to file: %s", m_log_file_name.c_str());
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  ros::NodeHandle & m_nh;
  ros::Timer m_timer;

  bool m_first_loop;
  uint64 m_loop_counter;
  uint64 m_max_iterations;

  bool m_is_3d;
  bool m_save_images;

  std::string m_image_file_name;
  std::string m_debug_output_folder;
  std::string m_log_file_name;

  std::string m_known_empty_file_name;
  std::string m_known_occupied_file_name;

  float m_sensor_focal_length;
  uint64 m_sensor_resolution_x;
  uint64 m_sensor_resolution_y;
  float m_sensor_range_voxels;
  float m_igain_min_range_voxels;
  float m_a_priori_occupied_prob;
  float m_autocomplete_igain_rescale_margin;
  uint64 m_accuracy_skip_voxels;

  Vector3fVector m_prev_origins;
  QuaternionfVector m_prev_orientations;

  Voxelgrid m_current_observed_empty;
  Voxelgrid m_current_observed_occupied;
  Voxelgrid m_current_frontier;
  Voxelgrid m_environment_image;

  INBVAdapterPtr m_current_adapter;

  GenerateTestDatasetOpenCL m_opencl;
  std::shared_ptr<GenerateSingleImage> m_generate_single_image;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "simulate_nbv_cycle");

  ros::NodeHandle nh("~");

  int param_int;
  nh.param<int>(PARAM_NAME_RANDOM_SEED, param_int, PARAM_DEFAULT_RANDOM_SEED);
  srand(param_int);

  SimulateNBVCycle snc(nh);

  ros::spin();

  return 0;
}
