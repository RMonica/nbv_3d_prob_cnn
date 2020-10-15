#ifndef VOXELGRID_H
#define VOXELGRID_H

// STL
#include <stdint.h>
#include <vector>
#include <memory>
#include <string>

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace octomap
{
  class OcTree;
}

class Voxelgrid
{
  public:
  typedef std::vector<float> FloatVector;
  typedef uint64_t uint64;
  typedef uint8_t uint8;
  typedef uint32_t uint32;

  typedef std::shared_ptr<Voxelgrid> Ptr;
  typedef std::shared_ptr<const Voxelgrid> ConstPtr;

  Voxelgrid(): Voxelgrid(Eigen::Vector3i::Zero()) {}
  Voxelgrid(const uint64 width, const uint64 height, const uint64 depth);
  explicit Voxelgrid(const Eigen::Vector3i & size);

  // LOAD/SAVE/CONVERSIONS
  static Ptr Load2DOpenCV(const std::string & filename);
  static Ptr Load3DOctomap(const std::string & filename);
  bool Save2D3D(const std::string & filename_prefix, const bool is_3d) const;

  cv::Mat ToOpenCVImage2D() const;
  static Ptr FromOpenCVImage2DUint8(const cv::Mat &image);
  static Ptr FromOpenCVImage2DFloat(const cv::Mat &image);
  bool SaveOpenCVImage2D(const std::string & filename) const;

  std::shared_ptr<octomap::OcTree> ToOctomapOctree() const;
  static Ptr FromOctomapOctree(octomap::OcTree & octree);
  bool SaveOctomapOctree(const std::string & filename) const;

  std_msgs::Float32MultiArray ToFloat32MultiArray() const;
  static Ptr FromFloat32MultiArray(const std_msgs::Float32MultiArray & arr);

  // 3D versions
  float & at(const uint64 x, const uint64 y, const uint64 z) {return m_data[x + y * m_width + z * m_width * m_height]; }
  float & at(const Eigen::Vector3i & index) {return at(index.x(), index.y(), index.z()); }
  float & operator[](const Eigen::Vector3i & index) {return at(index); }

  const float & at(const uint64 x, const uint64 y, const uint64 z) const
    {return m_data[x + y * m_width + z * m_width * m_height]; }
  const float & at(const Eigen::Vector3i & index) const {return at(index.x(), index.y(), index.z()); }
  const float & operator[](const Eigen::Vector3i & index) const {return at(index); }

  // Get size
  const Eigen::Vector3i GetSize() const {return Eigen::Vector3i(m_width, m_height, m_depth); }
  const uint64 GetWidth() const {return m_width; }
  const uint64 GetHeight() const {return m_height; }
  const uint64 GetDepth() const {return m_depth; }
  const bool IsEmpty() const {return GetSize() == Eigen::Vector3i::Zero(); }

  // RESIZE
  Voxelgrid::Ptr Resize(const Eigen::Vector3f & scale) const;
  Voxelgrid::Ptr Resize(const Eigen::Vector2f & scale) const {return Resize(Eigen::Vector3f(scale.x(), scale.y(), 1.0f)); }
  Voxelgrid::Ptr Resize(const float & scale) const {return Resize(Eigen::Vector3f(scale, scale, scale)); }

  // FILL
  void Fill(const float value);
  Voxelgrid::Ptr FilledWith(const float value) const;

  // BOOLEAN / MAX
  Voxelgrid::Ptr Or(const Voxelgrid & other) const {return Max(other); }
  Voxelgrid::Ptr Max(const Voxelgrid & other) const;

  // AND
  Voxelgrid::Ptr And(const Voxelgrid & other) const {return Min(other); }
  Voxelgrid::Ptr Min(const Voxelgrid & other) const;
  Voxelgrid::Ptr AndNot(const Voxelgrid & other) const;
  Voxelgrid::Ptr Not() const;

  // DILATE
  Voxelgrid::Ptr DilateRect(const Eigen::Vector3i & kernel_size) const;
  Voxelgrid::Ptr DilateCross(const Eigen::Vector3i & kernel_size) const;

  // ERODE
  Voxelgrid::Ptr ErodeRect(const Eigen::Vector3i & kernel_size) const;
  Voxelgrid::Ptr ErodeCross(const Eigen::Vector3i & kernel_size) const;

  // SUBMATRIX
  // copies other in this, starting at origin
  void SetSubmatrix(const Eigen::Vector3i & origin, const Voxelgrid & other);
  // get submatrix
  Voxelgrid::Ptr GetSubmatrix(const Eigen::Vector3i & origin, const Eigen::Vector3i & size) const;
  // set submatrix to uniform value
  void FillSubmatrix(const Eigen::Vector3i & origin, const Eigen::Vector3i & size, const float value);

  // CLAMP
  Voxelgrid::Ptr Clamped(const float min, const float max) const;
  void Clamp(const float min, const float max);

  void Min(const float min);
  void Max(const float max);

  // arithmetics for scalars
  void Add(const float value);
  void Multiply(const float value);

  void DivideBy(const Voxelgrid & other);

  void ToStream(std::ostream & ostr) const;
  static Ptr FromStream(std::istream & istr);
  std::string ToString() const;
  static Ptr FromString(const std::string & str);
  static Ptr FromFile(const std::string & filename);
  bool ToFile(const std::string & filename) const;
  bool ToFileBinary(const std::string & filename) const;
  static Ptr FromFileBinary(const std::string &filename);

  // transpose/reflect
  Voxelgrid::Ptr Transpose(const uint64 axis0, const uint64 axis1, const uint64 axis2);
  Voxelgrid::Ptr Reflect(const uint64 axis);
  Voxelgrid::Ptr Rotate90(const uint64 axis1, const uint64 axis2); // counter_clockwise
  Voxelgrid::Ptr Rotate90n(const uint64 axis1, const uint64 axis2, const uint64 n);

  private:
  Eigen::VectorXf m_data;
  uint64 m_width, m_height, m_depth;
};

class Voxelgrid4
{
  public:
  typedef uint64_t uint64;
  typedef uint8_t uint8;

  typedef std::shared_ptr<Voxelgrid4> Ptr;
  typedef std::shared_ptr<const Voxelgrid4> ConstPtr;

  Voxelgrid4(const uint64 width, const uint64 height, const uint64 depth);
  Voxelgrid4(const uint64 width, const uint64 height);
  explicit Voxelgrid4(const Eigen::Vector3i & size);
  Voxelgrid4(): Voxelgrid4(Eigen::Vector3i::Zero()) {}

  Voxelgrid & at(const uint64 c) {return m_grids[c]; }
  Voxelgrid & operator[](const uint64 c) {return at(c); }

  const Voxelgrid & at(const uint64 c) const {return m_grids[c]; }
  const Voxelgrid & operator[](const uint64 c) const {return at(c); }

  void SetAt(const Eigen::Vector3i & coord, const Eigen::Vector3f & data)
    { at(0)[coord] = data[0]; at(1)[coord] = data[1]; at(2)[coord] = data[2]; at(3)[coord] = 1.0f; }
  void SetAt(const Eigen::Vector3i & coord, const Eigen::Vector4f & data)
    { at(0)[coord] = data[0]; at(1)[coord] = data[1]; at(2)[coord] = data[2]; at(3)[coord] = data[3]; }
  Eigen::Vector4f GetAt4(const Eigen::Vector3i & coord) const
    { return Eigen::Vector4f(at(0)[coord], at(1)[coord], at(2)[coord], at(3)[coord]); }
  Eigen::Vector3f GetAt3(const Eigen::Vector3i & coord) const
    { return Eigen::Vector3f(at(0)[coord], at(1)[coord], at(2)[coord]); }

  bool Save2D3D(const std::string & filename_prefix, const bool is_3d) const;
  bool SaveOctomapOctree(const std::string & filename_prefix) const;
  bool ToFile(const std::string & filename_prefix) const;
  bool ToFileBinary(const std::string & filename_prefix) const;

  static Voxelgrid4::Ptr FromOpenCVImage2D(const cv::Mat & cv_mat);
  static Voxelgrid4::Ptr FromFloat32MultiArray(const std_msgs::Float32MultiArray & array);
  cv::Mat ToOpenCVImage2D() const;
  bool SaveOpenCVImage2D(const std::string & filename) const;

  const Eigen::Vector3i GetSize() const {return Eigen::Vector3i(m_width, m_height, m_depth); }
  const uint64 GetWidth() const {return m_width; }
  const uint64 GetHeight() const {return m_height; }
  const uint64 GetDepth() const {return m_depth; }
  const bool IsEmpty() const {return GetSize() == Eigen::Vector3i::Zero(); }

  void Fill(const Eigen::Vector4f & value)
    {m_grids[0].Fill(value[0]); m_grids[1].Fill(value[1]); m_grids[2].Fill(value[2]); m_grids[3].Fill(value[3]);}

  private:
  Voxelgrid m_grids[4];
  uint64 m_width, m_height, m_depth;
};

#endif // UNIVERSAL_VOXELGRID_H
