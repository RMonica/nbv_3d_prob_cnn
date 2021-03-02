#ifndef ORIGIN_VISIBILITY_H
#define ORIGIN_VISIBILITY_H

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/StdDeque>

#include <ros/ros.h>

#include <cmath>
#include <stdint.h>

// OpenCV
#include <opencv2/core/core.hpp>

#include <nbv_3d_cnn/voxelgrid.h>

class GenerateTestDatasetOpenCL;

struct OriginVisibility
{
  typedef uint64_t uint64;
  typedef int64_t int64;
  typedef uint8_t uint8;

  enum TObsType
  {
    OBSTYPE_OCCUPIED = 0,
    OBSTYPE_EMPTY,
    OBSTYPE_UNKNOWN,
    OBSTYPE_MAX
  };

  enum TFrame
  {
    FRAME_TOP = 0,
    FRAME_LEFT = 1,
    FRAME_BOTTOM = 2,
    FRAME_RIGHT = 3,
    FRAME_MAX_2D = 4,
    FRAME_FAR = 4,
    FRAME_NEAR = 5,
    FRAME_MAX = 6,
  };

  struct PixelInfo
  {
    double hits = 0;
    double miss = 0;

    float GetVisibility() const {return (hits + miss) > 0.0 ? ((hits) / (hits + miss)) : 0.0; }
  };

  typedef std::vector<PixelInfo> PixelInfoVector;
  typedef std::vector<Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > Vector2iVector;
  typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iVector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<bool> BoolVector;
  typedef std::deque<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> > Vector3iDeque;
  typedef std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > QuaternionfVector;
  typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > Matrix3fVector;
  typedef std::vector<float> FloatVector;

  OriginVisibility(const Eigen::Vector3f & p_pos, const uint64 res,
                   const bool is_3d);

  Eigen::Matrix3f GetAxisFromFrame(const TFrame frame) const;

  TFrame BearingToFrame(const Eigen::Vector3f & bearing) const;

  Eigen::Vector3i BearingToIndex(const Eigen::Vector3f & bearing) const;

  Eigen::Vector3f IndexToBearing(const Eigen::Vector3i index) const;

  void IntegrateObservation(const Eigen::Quaternionf &orientation, const TObsType type, const double count = 1.0);
  void IntegrateObservation(const Eigen::Vector3f &bearing, const TObsType type, const double count = 1.0);

  float AverageVisibility() const;
  float MaxVisibility(Eigen::Vector3i &index, Eigen::Quaternionf &orientation) const;
    // finds max visibility, with frame
  float MaxVisibility(Eigen::Vector3i &index, Eigen::Vector3f &bearing) const;

  cv::Mat GetVisibilityMatrix(const uint64 resolution) const;

  Voxelgrid GetVisibilityVoxelgrid(const uint64 res_out) const;

  static OriginVisibility FromVisibilityMatrix(const Eigen::Vector3f & p_pos,
                                               const uint64 &res, const cv::Mat matrix);
  static OriginVisibility FromVoxelgrid(const Eigen::Vector3f & p_pos,
                                        const uint64 &res, const Voxelgrid &voxelgrid);

  bool IsValidIndex(const Eigen::Vector3i & index) const;
  void ForeachIndex(const std::function<bool(const Eigen::Vector3i &)> &f) const;

  std::string ToString() const;

  // 2D only
  Eigen::Vector3i GetClockwiseIndexFrame(const Eigen::Vector3i & index) const;
  Eigen::Vector3i GetCounterClockwiseIndexFrame(const Eigen::Vector3i & index) const;
  OriginVisibility SmoothByHFOV(const float hfov) const;

  // 3D only
  OriginVisibility SmoothByHFOV3D(const Eigen::Vector2f & hfov) const;
  // GetBestSensorOrientation* should do exactly the same thing, but they are optimized for
  // High Resolution of the internal frame or High Sampling
  Eigen::Quaternionf GetBestSensorOrientationHighResolution(const QuaternionfVector &orientations,
                                                            const Eigen::Vector2f & hfov,
                                                            float &best_gain,
                                                            FloatVector & gains) const;
  Eigen::Quaternionf GetBestSensorOrientationManyViews(const QuaternionfVector &orientations,
                                                       const Eigen::Vector2f & hfov,
                                                       float &best_gain,
                                                       FloatVector & gains) const;
  Eigen::Quaternionf GetBestSensorOrientationOCL(GenerateTestDatasetOpenCL * opencl,
                                                 const QuaternionfVector &orientations,
                                                 const Eigen::Vector2f & hfov,
                                                 float &best_gain,
                                                 FloatVector &gains) const;
  Vector3iVector GetVoxelNeighborhood(const Eigen::Vector3i &center) const;
  float GetGainAtSensorOrientation(const Eigen::Quaternionf & orientation, const Eigen::Vector2f & hfov,
                                   double &hits, double &miss) const;
  bool IsIndexWithinOrientation(const Eigen::Vector2f & tan_hfov,
                                const Eigen::Quaternionf & orientation,
                                const Eigen::Vector3i & index) const;
  bool IsIndexWithinOrientationMatrix(const Eigen::Vector2f & tan_hfov,
                                      const Eigen::Matrix3f & mat,
                                      const Eigen::Vector3i & index) const;

  static QuaternionfVector GenerateStandardOrientationSet(const uint64 num_angles);

  uint64 IndexToI(const Eigen::Vector3i & index) const {return index.x() + index.y() * resolution +
                                                               index.z() * resolution * resolution; }

  Eigen::Vector3f GetPos() const {return pos; }


  private:
  Eigen::Vector3f pos;

  bool is_3d;

  PixelInfoVector virtual_frames;
  uint64 resolution;
  float focal_length;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

typedef std::vector<OriginVisibility> OriginVisibilityVector;

#endif // ORIGIN_VISIBILITY_H
