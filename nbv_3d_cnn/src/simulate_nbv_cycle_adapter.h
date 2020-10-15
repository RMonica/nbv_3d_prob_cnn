#ifndef SIMULATE_NBV_CYCLE_ADAPTER_H
#define SIMULATE_NBV_CYCLE_ADAPTER_H

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

#include <nbv_3d_cnn/PredictAction.h>
#include <nbv_3d_cnn/Predict3dAction.h>
#include <nbv_3d_cnn/Floats.h>
#include "generate_single_image.h"

class INBVAdapter
{
  public:
  ~INBVAdapter() {}

  typedef std::vector<float> FloatVector;

  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;

  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;

  virtual bool GetNextBestView(const Voxelgrid & environment,
                               const Voxelgrid & empty,
                               const Voxelgrid & occupied,
                               const Voxelgrid & frontier,
                               const Vector3fVector & skip_origins,
                               const QuaternionfVector & skip_orentations,
                               Eigen::Vector3f & origin,
                               Eigen::Quaternionf & orientation
                               ) = 0;

  virtual Voxelgrid GetScores() const = 0;
  virtual Voxelgrid4 GetColorScores() const = 0;

  virtual bool IsRandom() const = 0;

};
typedef std::shared_ptr<INBVAdapter> INBVAdapterPtr;

class RandomNBVAdapter: public INBVAdapter
{
  public:
  explicit RandomNBVAdapter(ros::NodeHandle & nh, const bool is_3d): m_nh(nh), m_is_3d(is_3d) {}

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation) override;

  Voxelgrid GetScores() const override {return m_last_scores; }
  Voxelgrid4 GetColorScores() const override {return Voxelgrid4(); }

  virtual bool IsRandom() const override {return true; }

  private:
  ros::NodeHandle & m_nh;

  bool m_is_3d;

  Voxelgrid m_last_scores;
};

class CNNDirectionalNBVAdapter: public INBVAdapter
{
  public:
  typedef actionlib::SimpleActionClient<nbv_3d_cnn::PredictAction> PredictActionClient;
  typedef std::shared_ptr<PredictActionClient> PredictActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn::Predict3dAction> Predict3dActionClient;
  typedef std::shared_ptr<Predict3dActionClient> Predict3dActionClientPtr;


  enum Mode
  {
    MODE_OV,
    MODE_OV_DIRECT,
    MODE_FLAT,
  };
  explicit CNNDirectionalNBVAdapter(ros::NodeHandle & nh,
                                         GenerateTestDatasetOpenCL & opencl,
                                         const bool is_3d,
                                         const Eigen::Vector2f & sensor_hfov,
                                         const Mode mode,
                                         const uint64_t accuracy_skip);

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation) override;

  bool GetNextBestView3d(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation);

  bool GetNextBestView2d(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation);

  virtual bool IsRandom() const override {return false; }

  Voxelgrid GetScores() const override {return m_last_scores; }
  Voxelgrid4 GetColorScores() const override {return Voxelgrid4(); }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  void onRawData(const nbv_3d_cnn::FloatsConstPtr raw_data);

  ros::NodeHandle & m_nh;

  PredictActionClientPtr m_predict_action_client;
  Predict3dActionClientPtr m_predict_3d_action_client;

  ros::Subscriber m_raw_data_subscriber;
  nbv_3d_cnn::FloatsConstPtr m_raw_data;
  ros::CallbackQueue m_raw_data_callback_queue;
  ros::NodeHandle m_private_nh;

  GenerateTestDatasetOpenCL & m_opencl;

  uint64_t m_accuracy_skip;

  bool m_is_3d;
  Eigen::Vector2f m_sensor_hfov;
  Mode m_mode;

  Voxelgrid m_last_scores;
};

class CNNQuatNBVAdapter: public INBVAdapter
{
  public:
  typedef actionlib::SimpleActionClient<nbv_3d_cnn::PredictAction> PredictActionClient;
  typedef std::shared_ptr<PredictActionClient> PredictActionClientPtr;
  typedef actionlib::SimpleActionClient<nbv_3d_cnn::Predict3dAction> Predict3dActionClient;
  typedef std::shared_ptr<Predict3dActionClient> Predict3dActionClientPtr;

  explicit CNNQuatNBVAdapter(ros::NodeHandle & nh, const bool is_3d,
                                   const uint64_t accuracy_skip);

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation) override;

  bool GetNextBestView3d(const Voxelgrid & environment,
                         const Voxelgrid & empty,
                         const Voxelgrid & occupied,
                         const Voxelgrid & frontier,
                         const Vector3fVector & skip_origins,
                         const QuaternionfVector & skip_orentations,
                         Eigen::Vector3f & origin,
                         Eigen::Quaternionf & orientation);

  bool GetNextBestView2d(const Voxelgrid & environment,
                         const Voxelgrid & empty,
                         const Voxelgrid & occupied,
                         const Voxelgrid & frontier,
                         const Vector3fVector & skip_origins,
                         const QuaternionfVector & skip_orentations,
                         Eigen::Vector3f & origin,
                         Eigen::Quaternionf & orientation);

  virtual bool IsRandom() const override {return false; }

  Voxelgrid GetScores() const override {return Voxelgrid(); }
  Voxelgrid4 GetColorScores() const override {return m_last_scores; }

  private:
  void onRawData(const nbv_3d_cnn::FloatsConstPtr raw_data);

  ros::NodeHandle & m_nh;

  bool m_is_3d;

  uint64 m_accuracy_skip;

  PredictActionClientPtr m_predict_action_client;
  Predict3dActionClientPtr m_predict_3d_action_client;

  ros::Subscriber m_raw_data_subscriber;
  nbv_3d_cnn::FloatsConstPtr m_raw_data;
  ros::CallbackQueue m_raw_data_callback_queue;
  ros::NodeHandle m_private_nh;

  Voxelgrid4 m_last_scores;
};

class InformationGainNBVAdapter: public INBVAdapter
{
  public:
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;

  explicit InformationGainNBVAdapter(ros::NodeHandle & nh,
                                     GenerateTestDatasetOpenCL &opencl,
                                     GenerateSingleImage &generate_single_image,
                                     const float max_range,
                                     const float min_range,
                                     const bool stop_at_first_hit,
                                     const float a_priori_occupied_prob,
                                     uint64_t directional_view_cube_resolution,
                                     const Eigen::Vector2f & sensor_hfov,
                                     const bool is_omniscient,
                                     const bool is_3d,
                                     const uint64_t accuracy_skip);

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector &skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation) override;

  virtual bool IsRandom() const override {return false; }

  void SetRegionOfInterest(const Eigen::Vector3i & min, const Eigen::Vector3i & max);
  void SetROIByLastOrigin(const Eigen::Vector3f & origin, const Eigen::Quaternionf & orientation);

  const Voxelgrid::Ptr GetLastExpectedObservation() const
    {return Voxelgrid::Ptr(new Voxelgrid(m_last_expected_observation)); }
  const Eigen::Quaternionf & GetLastOrientation() const {return m_last_orientation; }
  const Voxelgrid::Ptr GetLastNotSmoothedScores() const
    {return Voxelgrid::Ptr(new Voxelgrid(m_last_not_smoothed_scores)); }

  cv::Mat GetDebugImage(const Voxelgrid &environment) const;

  bool IsOmniscient() const {return m_is_omniscent; }

  Voxelgrid GetScores() const override {return m_last_scores; }
  Voxelgrid4 GetColorScores() const override {return Voxelgrid4(); }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  void ForEachEmpty(const Voxelgrid & empty, const uint64 skip_accuracy,
                    const Eigen::Vector3i & roi_min, const Eigen::Vector3i & roi_max,
                    const std::function<void(const Eigen::Vector3i &)> &f) const;

  void GetNextBestView3DHelper(const bool in_roi,
                               const Eigen::Vector3i & xyz,
                               const OriginVisibilityVector & ovv,
                               uint64_t & counter,
                               Voxelgrid & not_smoothed_scores,
                               const Eigen::Vector3i & subrect_origin,
                               const Eigen::Vector3i & subrect_size,
                               Eigen::Quaternionf & this_orientation,
                               float & this_score,
                               const QuaternionfVector & orientations
                               ) const;

  void GetNextBestView2DHelper(const bool in_roi,
                               const Eigen::Vector3i & xyz,
                               const OriginVisibilityVector & ovv,
                               uint64_t & counter,
                               Voxelgrid & not_smoothed_scores,
                               Voxelgrid & scores,
                               const Eigen::Vector3i & subrect_origin,
                               const Eigen::Vector3i & subrect_size,
                               Eigen::Quaternionf & this_orientation,
                               float & this_score,
                               const QuaternionfVector & orientations
                               ) const;

  ros::NodeHandle & m_nh;

  bool m_is_omniscent;

  bool m_is_3d;
  uint64 m_accuracy_skip;

  Voxelgrid m_last_expected_observation;
  Eigen::Quaternionf m_last_orientation;
  Eigen::Vector3f m_last_origin;
  Voxelgrid m_last_not_smoothed_scores;
  Voxelgrid m_last_scores;

  float m_max_range;
  float m_min_range;
  bool m_stop_at_first_hit;
  float m_a_priori_occupied_prob;
  uint64_t m_view_cube_resolution;
  Eigen::Vector2f m_sensor_hfov;

  Eigen::Vector3i m_region_of_interest_min;
  Eigen::Vector3i m_region_of_interest_max;
  Voxelgrid m_prev_scores;

  GenerateTestDatasetOpenCL & m_opencl;
  GenerateSingleImage & m_generate_single_image;
};

class AutocompleteIGainNBVAdapter: public INBVAdapter
{
  public:
  typedef std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f> > Vector2fVector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn::PredictAction> PredictActionClient;
  typedef std::shared_ptr<PredictActionClient> PredictActionClientPtr;

  typedef actionlib::SimpleActionClient<nbv_3d_cnn::Predict3dAction> Predict3dActionClient;
  typedef std::shared_ptr<Predict3dActionClient> Predict3dActionClientPtr;

  explicit AutocompleteIGainNBVAdapter(ros::NodeHandle & nh,
                                       GenerateTestDatasetOpenCL &opencl,
                                       GenerateSingleImage &generate_single_image,
                                       const float max_range,
                                       uint64_t directional_view_cube_resolution,
                                       const Eigen::Vector2f & sensor_hfov,
                                       const bool is_3d,
                                       const uint64_t accuracy_skip);

  bool GetNextBestView(const Voxelgrid & environment,
                       const Voxelgrid & empty,
                       const Voxelgrid & occupied,
                       const Voxelgrid & frontier,
                       const Vector3fVector & skip_origins,
                       const QuaternionfVector & skip_orentations,
                       Eigen::Vector3f & origin,
                       Eigen::Quaternionf & orientation) override;

  bool Predict3d(const Voxelgrid &empty, const Voxelgrid &frontier, Voxelgrid &autocompleted);
  bool Predict(const Voxelgrid &empty, const Voxelgrid &frontier, Voxelgrid &autocompleted);

  virtual bool IsRandom() const override {return false; }

  void SetRegionOfInterest(const Eigen::Vector3i & min, const Eigen::Vector3i & max)
    {m_information_gain->SetRegionOfInterest(min, max); }
  void SetROIByLastOrigin(const Eigen::Vector3f & origin, const Eigen::Quaternionf & orientation)
    {m_information_gain->SetROIByLastOrigin(origin, orientation); }

  const Voxelgrid::Ptr GetLastExpectedObservation() const {return m_information_gain->GetLastExpectedObservation(); }
  const Eigen::Quaternionf GetLastOrientation() const {return m_information_gain->GetLastOrientation(); }
  const Voxelgrid::Ptr GetLastNotSmoothedScores() const {return m_information_gain->GetLastNotSmoothedScores(); }

  cv::Mat GetDebugImage(const Voxelgrid &environment) const {return m_information_gain->GetDebugImage(environment); }

  Voxelgrid GetLastAutocompletedImage() const {return m_last_autocompleted_image; }

  Voxelgrid GetScores() const override {return m_information_gain->GetScores(); }
  Voxelgrid4 GetColorScores() const override {return m_information_gain->GetColorScores(); }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  void onRawData(const nbv_3d_cnn::FloatsConstPtr raw_data);

  ros::NodeHandle & m_nh;

  bool m_is_3d;

  Voxelgrid m_last_autocompleted_image;

  PredictActionClientPtr m_predict_action_client;
  Predict3dActionClientPtr m_predict3d_action_client;

  ros::Subscriber m_raw_data_subscriber;
  nbv_3d_cnn::FloatsConstPtr m_raw_data;
  ros::CallbackQueue m_raw_data_callback_queue;
  ros::NodeHandle m_private_nh;

  GenerateTestDatasetOpenCL & m_opencl;
  GenerateSingleImage & m_generate_single_image;

  std::shared_ptr<InformationGainNBVAdapter> m_information_gain;
};

#endif // SIMULATE_NBV_CYCLE_ADAPTER_H
