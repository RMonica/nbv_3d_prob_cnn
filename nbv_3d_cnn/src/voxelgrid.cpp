#include <nbv_3d_cnn/voxelgrid.h>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

#include <sstream>

#define MAGIC_TERNARY_BINARY "VXGT"

Voxelgrid::Voxelgrid(const uint64 width, const uint64 height, const uint64 depth)
{
  m_width = width;
  m_height = height;
  m_depth = depth;
  m_data = Eigen::VectorXf::Zero(m_width * m_height * m_depth);
}

Voxelgrid::Voxelgrid(const Eigen::Vector3i & size):
  Voxelgrid(size.x(), size.y(), size.z())
{}

Voxelgrid::Ptr Voxelgrid::Load2DOpenCV(const std::string & filename)
{
  cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

  if (!image.data)
  {
    ROS_ERROR("Voxelgrid: unable to load image %s using OpenCV.", filename.c_str());
    return Ptr();
  }

  image.convertTo(image, CV_32FC1, 1.0f/255.0f);

  Voxelgrid::Ptr result(new Voxelgrid(image.cols, image.rows, 1));

  for (uint64 y = 0; y < image.rows; y++)
    for (uint64 x = 0; x < image.cols; x++)
      result->at(x, y, 0) = image.at<float>(y, x);

  return result;
}

Voxelgrid::Ptr Voxelgrid::FromOpenCVImage2DUint8(const cv::Mat & image)
{
  Voxelgrid::Ptr result(new Voxelgrid(image.cols, image.rows, 1));

  for (uint64 y = 0; y < image.rows; y++)
    for (uint64 x = 0; x < image.cols; x++)
      result->at(x, y, 0) = image.at<uint8>(y, x) / 255.0f;

  return result;
}

Voxelgrid::Ptr Voxelgrid::FromOpenCVImage2DFloat(const cv::Mat & image)
{
  Voxelgrid::Ptr result(new Voxelgrid(image.cols, image.rows, 1));

  for (uint64 y = 0; y < image.rows; y++)
    for (uint64 x = 0; x < image.cols; x++)
      result->at(x, y, 0) = image.at<float>(y, x);

  return result;
}

std::shared_ptr<octomap::OcTree> Voxelgrid::ToOctomapOctree(const float resolution) const
{
  std::shared_ptr<octomap::OcTree> result(new octomap::OcTree(resolution));

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < m_depth; i.z()++)
    for (i.y() = 0; i.y() < m_height; i.y()++)
      for (i.x() = 0; i.x() < m_width; i.x()++)
      {
        const Eigen::Vector3f ecoords = (i.cast<float>() + 0.5f * Eigen::Vector3f::Ones()) * resolution -
                                         0.5f * Eigen::Vector3f::Ones();
        const float value = at(i);

        if (value < 0.0001f)
          continue; // almost zero

        const octomap::point3d new_point(ecoords.x(), ecoords.y(), ecoords.z());
        octomap::OcTreeNode * scene_node = result->search(new_point);
        if (!scene_node)
          scene_node = result->updateNode(new_point, true);
        if (value > 0.5f)
          scene_node->setLogOdds(octomap::logodds(0.99));
        else
          scene_node->setLogOdds(octomap::logodds(0.01f));
      }

  return result;
}

std::shared_ptr<octomap::OcTree> Voxelgrid::ToOctomapOctree() const
{
  const uint64 max_coord = std::max(std::max(m_depth, m_height), m_width);
  const float resolution = 1.0f / max_coord;

  std::shared_ptr<octomap::OcTree> result(new octomap::OcTree(resolution));

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < m_depth; i.z()++)
    for (i.y() = 0; i.y() < m_height; i.y()++)
      for (i.x() = 0; i.x() < m_width; i.x()++)
      {
        const Eigen::Vector3f ecoords = (i.cast<float>() + 0.5f * Eigen::Vector3f::Ones()) * resolution -
                                         0.5f * Eigen::Vector3f::Ones();
        const float value = at(i);

        if (value < 0.0001f)
          continue; // almost zero

        const octomap::point3d new_point(ecoords.x(), ecoords.y(), ecoords.z());
        octomap::OcTreeNode * scene_node = result->search(new_point);
        if (!scene_node)
          scene_node = result->updateNode(new_point, true);
        if (value > 0.5f)
          scene_node->setLogOdds(octomap::logodds(0.99));
        else
          scene_node->setLogOdds(octomap::logodds(0.01f));
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::FromOctomapOctree(octomap::OcTree & octree,
                                            const Eigen::Vector3i & bbx_isize)
{
  octree.expand();
  const float resolution = octree.getResolution();

  const Eigen::Vector3f e_bbx_min = -resolution * bbx_isize.cast<float>() / 2.0f;

  Voxelgrid::Ptr result(new Voxelgrid(bbx_isize.x(), bbx_isize.y(), bbx_isize.z()));

  for (uint64 z = 0; z < bbx_isize.z(); z++)
    for (uint64 y = 0; y < bbx_isize.y(); y++)
      for (uint64 x = 0; x < bbx_isize.x(); x++)
      {
        result->at(x, y, z) = 0.0f;

        const Eigen::Vector3f new_epoint = (Eigen::Vector3i(x, y, z).cast<float>() +
                                            0.5f * Eigen::Vector3f::Ones()) * resolution + e_bbx_min;
        const octomap::point3d new_point(new_epoint.x(), new_epoint.y(), new_epoint.z());
        octomap::OcTreeNode * node = octree.search(new_point);
        if (node)
          result->at(x, y, z) = (node->getOccupancy() > 0.5f ? 1.0f : 0.0f);
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::FromOctomapOctree(octomap::OcTree & octree)
{
  octree.expand();
  const float resolution = octree.getResolution();
  const float bbox_side = 1.0;

  const Eigen::Vector3f e_bbx_min = -Eigen::Vector3f::Ones() * bbox_side / 2.0f;

  const Eigen::Vector3f bbx_size = bbox_side * Eigen::Vector3f::Ones();
  const Eigen::Vector3i bbx_isize = (bbx_size / resolution).array().round().cast<int>();

  Voxelgrid::Ptr result(new Voxelgrid(bbx_isize.x(), bbx_isize.y(), bbx_isize.z()));

  for (uint64 z = 0; z < bbx_isize.z(); z++)
    for (uint64 y = 0; y < bbx_isize.y(); y++)
      for (uint64 x = 0; x < bbx_isize.x(); x++)
      {
        result->at(x, y, z) = 0.0f;

        const Eigen::Vector3f new_epoint = (Eigen::Vector3i(x, y, z).cast<float>() +
                                            0.5f * Eigen::Vector3f::Ones()) * resolution + e_bbx_min;
        const octomap::point3d new_point(new_epoint.x(), new_epoint.y(), new_epoint.z());
        octomap::OcTreeNode * node = octree.search(new_point);
        if (node)
          result->at(x, y, z) = (node->getOccupancy() > 0.5f ? 1.0f : 0.0f);
      }

  return result;
}

bool Voxelgrid::SaveOctomapOctree(const std::string & filename) const
{
  std::shared_ptr<octomap::OcTree> octree = ToOctomapOctree();
  return octree->writeBinary(filename);
}

bool Voxelgrid::SaveOctomapOctree(const std::string & filename, const float resolution) const
{
  std::shared_ptr<octomap::OcTree> octree = ToOctomapOctree(resolution);
  return octree->writeBinary(filename);
}

Voxelgrid::Ptr Voxelgrid::Load3DOctomap(const std::string & filename)
{
  octomap::OcTree octree(0.1);
  if (!octree.readBinary(filename))
  {
    ROS_ERROR("Voxelgrid: unable to load octree %s using OctoMap.", filename.c_str());
    return Ptr();
  }

  return FromOctomapOctree(octree);
}

Voxelgrid::Ptr Voxelgrid::Load3DOctomapWithISize(const std::string & filename,
                                                 const Eigen::Vector3i & isize)
{
  octomap::OcTree octree(0.1);
  if (!octree.readBinary(filename))
  {
    ROS_ERROR("Voxelgrid: unable to load octree %s using OctoMap.", filename.c_str());
    return Ptr();
  }

  return FromOctomapOctree(octree, isize);
}

Voxelgrid::Ptr Voxelgrid::HalveSize(const Eigen::Vector3i & iterations) const
{
  const Eigen::Vector3i scale(1ul << (iterations.x() - 1),
                              1ul << (iterations.y() - 1),
                              1ul << (iterations.z() - 1));
  const Eigen::Vector3i my_size = GetSize();
  const Eigen::Vector3i new_size = my_size.array() / scale.array();

  Voxelgrid::Ptr result(new Voxelgrid(new_size));

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < new_size.z(); i.z()++)
    for (i.y() = 0; i.y() < new_size.y(); i.y()++)
      for (i.x() = 0; i.x() < new_size.x(); i.x()++)
      {
        const Eigen::Vector3i source_i = i.array() * scale.array();

        if ((source_i.array() < 0).any())
          continue;
        if ((source_i.array() >= my_size.array()).any())
          continue;

        result->at(i) = at(source_i);
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::Resize(const Eigen::Vector3f & scale) const
{
  const Eigen::Vector3i my_size = GetSize();
  const Eigen::Vector3i new_size = (my_size.cast<float>().array() * scale.array()).round().cast<int>();

  Voxelgrid::Ptr result(new Voxelgrid(new_size));

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < new_size.z(); i.z()++)
    for (i.y() = 0; i.y() < new_size.y(); i.y()++)
      for (i.x() = 0; i.x() < new_size.x(); i.x()++)
      {
        const Eigen::Vector3i source_i = (i.cast<float>().array() / scale.array()).round().cast<int>();
        if ((source_i.array() < 0).any())
          continue;
        if ((source_i.array() >= my_size.array()).any())
          continue;
        result->at(i) = at(source_i);
      }

  return result;
}

cv::Mat Voxelgrid::ToOpenCVImage2D() const
{
  const uint64 width = GetWidth();
  const uint64 height = GetHeight();

  cv::Mat result(width, height, CV_8UC1);

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      result.at<uint8>(y, x) = uint8(at(x, y, 0) * 255.0f);

  return result;
}

bool Voxelgrid::SaveOpenCVImage2D(const std::string & filename) const
{
  cv::Mat image = ToOpenCVImage2D();
  return cv::imwrite(filename, image);
}

bool Voxelgrid::Save2D3D(const std::string & filename_prefix, const bool is_3d) const
{
  if (!is_3d)
    return SaveOpenCVImage2D(filename_prefix + ".png");
  else
  {
    return SaveOctomapOctree(filename_prefix + ".bt") &&
           ToFileBinary(filename_prefix + ".binvoxelgrid");
  }
}

bool Voxelgrid::Save2D3DR(const std::string & filename_prefix, const bool is_3d, const float resolution) const
{
  if (!is_3d)
    return SaveOpenCVImage2D(filename_prefix + ".png");
  else
  {
    return SaveOctomapOctree(filename_prefix + ".bt", resolution) &&
           ToFileBinary(filename_prefix + ".binvoxelgrid");
  }
}

std_msgs::Float32MultiArray Voxelgrid::ToFloat32MultiArray() const
{
  std_msgs::Float32MultiArray result;

  result.data.resize(m_width * m_height * m_depth);

  for (uint64 z = 0; z < m_depth; z++)
    for (uint64 y = 0; y < m_height; y++)
      for (uint64 x = 0; x < m_width; x++)
        result.data[x + y * m_width + z * m_width * m_height] = at(x, y, z);

  std_msgs::MultiArrayLayout & layout = result.layout;
  std_msgs::MultiArrayDimension dim;
  dim.label = "z";
  dim.size = m_depth;
  dim.stride = m_height * m_width;
  layout.dim.push_back(dim);
  dim.label = "y";
  dim.size = m_height;
  dim.stride = m_width;
  layout.dim.push_back(dim);
  dim.label = "x";
  dim.size = m_width;
  dim.stride = 1;
  layout.dim.push_back(dim);

  return result;
}

Voxelgrid::Ptr Voxelgrid::FromFloat32MultiArray(const std_msgs::Float32MultiArray & arr)
{
  const std_msgs::MultiArrayLayout & layout = arr.layout;

  auto dim_x = std::find_if(layout.dim.begin(), layout.dim.end(),
                            [](const std_msgs::MultiArrayDimension & dim) -> bool
                            {return dim.label == "x"; });
  auto dim_y = std::find_if(layout.dim.begin(), layout.dim.end(),
                            [](const std_msgs::MultiArrayDimension & dim) -> bool
                            {return dim.label == "y"; });
  auto dim_z = std::find_if(layout.dim.begin(), layout.dim.end(),
                            [](const std_msgs::MultiArrayDimension & dim) -> bool
                            {return dim.label == "z"; });

  if (dim_x == layout.dim.end() || dim_y == layout.dim.end() || dim_z == layout.dim.end())
  {
    ROS_ERROR("Voxelgrid: could not find x, y, z, in Float32MultiArray");
    return Ptr();
  }

  if (dim_x->size && dim_y->size && dim_z->size)
  {
    const uint64 expected_size = (dim_x->size - 1) * dim_x->stride +
      (dim_y->size - 1) * dim_y->stride + (dim_z->size - 1) * dim_z->stride + 1;
    if (expected_size > arr.data.size())
    {
      ROS_ERROR("Voxelgrid: array size does not match in Float32MultiArray, it is %u, it should be at least %u",
                unsigned(arr.data.size()), unsigned(expected_size));
      return Ptr();
    }
  }

  Voxelgrid::Ptr result(new Voxelgrid(dim_x->size, dim_y->size, dim_z->size));

  for (uint64 z = 0; z < dim_z->size; z++)
    for (uint64 y = 0; y < dim_y->size; y++)
      for (uint64 x = 0; x < dim_x->size; x++)
      {
        const float v = arr.data[x * dim_x->stride + y * dim_y->stride + z * dim_z->stride];
        result->at(x, y, z) = v;
      }

  return result;
}

void Voxelgrid::Fill(const float value)
{
  m_data = Eigen::VectorXf::Ones(m_width * m_height * m_depth) * value;
}

Voxelgrid::Ptr Voxelgrid::FilledWith(const float value) const
{
  Voxelgrid::Ptr result(new Voxelgrid(m_width, m_height, m_depth));
  result->Fill(value);
  return result;
}

Voxelgrid::Ptr Voxelgrid::Max(const Voxelgrid & other) const
{
  if (other.GetSize() != GetSize())
    throw std::string("Voxelgrid::Max: SIZE DOES NOT MATCH.");

  Voxelgrid::Ptr result(new Voxelgrid(m_width, m_height, m_depth));
  result->m_data = m_data.array().max(other.m_data.array());
  return result;
}

Voxelgrid::Ptr Voxelgrid::Min(const Voxelgrid & other) const
{
  if (other.GetSize() != GetSize())
    throw std::string("Voxelgrid::Mask: SIZE DOES NOT MATCH.");

  Voxelgrid::Ptr result(new Voxelgrid(m_width, m_height, m_depth));
  result->m_data = m_data.array().min(other.m_data.array());
  return result;
}

Voxelgrid::Ptr Voxelgrid::Not() const
{
  Voxelgrid::Ptr neg_voxelgrid(new Voxelgrid(m_width, m_height, m_depth));
  neg_voxelgrid->m_data = Eigen::VectorXf::Ones(m_width * m_height * m_depth) - m_data;
  return neg_voxelgrid;
}

Voxelgrid::Ptr Voxelgrid::AndNot(const Voxelgrid & other) const
{
  if (other.GetSize() != GetSize())
    throw std::string("Voxelgrid::MaskNot: SIZE DOES NOT MATCH.");

  Voxelgrid::Ptr neg_voxelgrid(new Voxelgrid(m_width, m_height, m_depth));
  neg_voxelgrid->m_data = Eigen::VectorXf::Ones(m_width * m_height * m_depth) - other.m_data;
  return And(*neg_voxelgrid);
}

Voxelgrid::Ptr Voxelgrid::DilateRect(const Eigen::Vector3i & kernel_size) const
{
  Voxelgrid::Ptr result = FilledWith(0.0f);

  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < size.z(); i.z()++)
    for (i.y() = 0; i.y() < size.y(); i.y()++)
      for (i.x() = 0; i.x() < size.x(); i.x()++)
      {
        const float value = at(i);
        if (value < 0.0001f)
          continue; // already zero

        Eigen::Vector3i di;
        for (di.z() = -kernel_size.z(); di.z() <= kernel_size.z(); di.z()++)
          for (di.y() = -kernel_size.y(); di.y() <= kernel_size.y(); di.y()++)
            for (di.x() = -kernel_size.x(); di.x() <= kernel_size.x(); di.x()++)
            {
              const Eigen::Vector3i ni = i + di;

              if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
                continue;

              result->at(ni) = value;
            }
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::DilateCross(const Eigen::Vector3i & kernel_size) const
{
  Voxelgrid::Ptr result = FilledWith(0.0f);

  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < size.z(); i.z()++)
    for (i.y() = 0; i.y() < size.y(); i.y()++)
      for (i.x() = 0; i.x() < size.x(); i.x()++)
      {
        const float value = at(i);
        if (value < 0.0001f)
          continue; // already zero

        result->at(i) = value; // cross center

        for (uint64 coord = 0; coord < 3; coord++)
          for (int sign = -1; sign <= 1; sign += 2)
          {
            Eigen::Vector3i di = Eigen::Vector3i::Zero();
            for (di[coord] = 1; di[coord] <= kernel_size[coord]; di[coord]++)
            {
              const Eigen::Vector3i ni = i + di * sign;

              if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
                continue;

              result->at(ni) = value;
            }
          }
      }

  return result;
}

void Voxelgrid::SetSubmatrix(const Eigen::Vector3i & origin, const Voxelgrid & other)
{
  const Eigen::Vector3i other_size = other.GetSize();
  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < other_size.z(); i.z()++)
    for (i.y() = 0; i.y() < other_size.y(); i.y()++)
      for (i.x() = 0; i.x() < other_size.x(); i.x()++)
      {
        const Eigen::Vector3i ni = origin + i;
        if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
          continue;

        at(ni) = other.at(i);
      }
}

void Voxelgrid::FillSubmatrix(const Eigen::Vector3i & origin, const Eigen::Vector3i & other_size,
                              const float value)
{
  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < other_size.z(); i.z()++)
    for (i.y() = 0; i.y() < other_size.y(); i.y()++)
      for (i.x() = 0; i.x() < other_size.x(); i.x()++)
      {
        const Eigen::Vector3i ni = origin + i;
        if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
          continue;

        at(ni) = value;
      }
}

Voxelgrid::Ptr Voxelgrid::GetSubmatrix(const Eigen::Vector3i & origin, const Eigen::Vector3i & result_size) const
{
  Voxelgrid::Ptr result_ptr(new Voxelgrid(result_size));
  Voxelgrid & result = *result_ptr;
  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < result_size.z(); i.z()++)
    for (i.y() = 0; i.y() < result_size.y(); i.y()++)
      for (i.x() = 0; i.x() < result_size.x(); i.x()++)
      {
        const Eigen::Vector3i ni = origin + i;
        if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
          continue;

        result.at(i) = at(ni);
      }

  return result_ptr;
}

Voxelgrid::Ptr Voxelgrid::ErodeRect(const Eigen::Vector3i & kernel_size) const
{
  Voxelgrid::Ptr result = FilledWith(1.0f);

  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < size.z(); i.z()++)
    for (i.y() = 0; i.y() < size.y(); i.y()++)
      for (i.x() = 0; i.x() < size.x(); i.x()++)
      {
        const float value = at(i);
        if (value > 0.0001f)
          continue; // already one

        Eigen::Vector3i di;
        for (di.z() = -kernel_size.z(); di.z() <= kernel_size.z(); di.z()++)
          for (di.y() = -kernel_size.y(); di.y() <= kernel_size.y(); di.y()++)
            for (di.x() = -kernel_size.x(); di.x() <= kernel_size.x(); di.x()++)
            {
              const Eigen::Vector3i ni = i + di;

              if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
                continue;

              result->at(ni) = 0.0f;
            }
      }

  return result;
}
Voxelgrid::Ptr Voxelgrid::ErodeCross(const Eigen::Vector3i & kernel_size) const
{
  Voxelgrid::Ptr result = FilledWith(1.0f);

  const Eigen::Vector3i size = GetSize();

  Eigen::Vector3i i;
  for (i.z() = 0; i.z() < size.z(); i.z()++)
    for (i.y() = 0; i.y() < size.y(); i.y()++)
      for (i.x() = 0; i.x() < size.x(); i.x()++)
      {
        const float value = at(i);
        if (value > 0.0001f)
          continue; // already one

        result->at(i) = value; // cross center

        for (uint64 coord = 0; coord < 3; coord++)
          for (int sign = -1; sign <= 1; sign += 2)
          {
            Eigen::Vector3i di = Eigen::Vector3i::Zero();
            for (di[coord] = 1; di[coord] <= kernel_size[coord]; di[coord]++)
            {
              const Eigen::Vector3i ni = i + di * sign;

              if ((ni.array() < 0).any() || (ni.array() >= size.array()).any())
                continue;

              result->at(ni) = 0.0f;
            }
          }
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::Clamped(const float min, const float max) const
{
  Voxelgrid::Ptr result(new Voxelgrid(*this));
  result->Clamp(min, max);
  return result;
}

void Voxelgrid::Clamp(const float min, const float max)
{
  m_data = m_data.array().max(min).min(max);
}

void Voxelgrid::Min(const float min)
{
  m_data = m_data.array().min(min);
}

void Voxelgrid::Max(const float max)
{
  m_data = m_data.array().max(max);
}

void Voxelgrid::Add(const float value)
{
  m_data = m_data.array() + value;
}

void Voxelgrid::Multiply(const float value)
{
  m_data = m_data * value;
}

void Voxelgrid::DivideBy(const Voxelgrid & other)
{
  const Eigen::Vector3i size = GetSize();
  const Eigen::Vector3i other_size = other.GetSize();
  if (size != other_size)
  {
    std::cerr << "Voxelgrid::Divide: size mismatch: " << size.transpose() <<
                 " != " << other_size.transpose() << std::endl;
    exit(1);
  }

  m_data = m_data.array() / other.m_data.array();
}

bool Voxelgrid::ToFileTernaryBinary(const std::string & filename) const
{
  std::ofstream ofile(filename, std::ios::binary);

  const std::string magic = MAGIC_TERNARY_BINARY;

  const uint32 version = 1;
  const uint32 width = GetWidth();
  const uint32 height = GetHeight();
  const uint32 depth = GetDepth();

  ofile.write(magic.c_str(), 4);
  ofile.write((const char *)&version, sizeof(version));
  ofile.write((const char *)&width, sizeof(width));
  ofile.write((const char *)&height, sizeof(height));
  ofile.write((const char *)&depth, sizeof(depth));

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const float v = at(x, y, z);
        const int8 bv = ((v > 0.5) ? 1 : ((v < -0.5) ? -1 : 0));
        ofile.write((const char *)&bv, sizeof(bv));
      }

  if (!ofile)
    return false;
  return true;
}

Voxelgrid::Ptr Voxelgrid::FromFileTernaryBinary(const std::string &filename)
{
  std::ifstream ifile(filename, std::ios::binary);

  if (!ifile)
  {
    ROS_ERROR("Voxelgrid::FromFileTernaryBinary: cannot open file %s", filename.c_str());
    return Voxelgrid::Ptr();
  }

  const std::string magic = MAGIC_TERNARY_BINARY;

  char maybe_magic[5];
  ifile.read(maybe_magic, 4);
  maybe_magic[4] = 0;
  if (magic != maybe_magic)
  {
    ROS_ERROR("Voxelgrid::FromFileTernaryBinary: file must start with %s, '%s' found instead", magic.c_str(),
              maybe_magic);
    return Voxelgrid::Ptr();
  }

  uint32 version, width, height, depth;
  ifile.read((char *)&version, sizeof(version));
  ifile.read((char *)&width, sizeof(version));
  ifile.read((char *)&height, sizeof(version));
  ifile.read((char *)&depth, sizeof(version));

  if (!ifile || version != 1)
  {
    ROS_ERROR("Voxelgrid::FromFileTernaryBinary: invalid version %u", unsigned(version));
    return Voxelgrid::Ptr();
  }

  Voxelgrid::Ptr result(new Voxelgrid(width, height, depth));

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        int8 v;
        ifile.read((char *)&v, sizeof(v));
        result->at(x, y, z) = float(v);
      }

  if (!ifile)
    return Voxelgrid::Ptr();
  return result;
}

bool Voxelgrid::ToFileBinary(const std::string &filename) const
{
  std::ofstream ofile(filename, std::ios::binary);

  const std::string magic = "VXGR";

  const uint32 version = 1;
  const uint32 width = GetWidth();
  const uint32 height = GetHeight();
  const uint32 depth = GetDepth();

  ofile.write(magic.c_str(), 4);
  ofile.write((const char *)&version, sizeof(version));
  ofile.write((const char *)&width, sizeof(width));
  ofile.write((const char *)&height, sizeof(height));
  ofile.write((const char *)&depth, sizeof(depth));

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        const float v = at(x, y, z);
        ofile.write((const char *)&v, sizeof(v));
      }

  if (!ofile)
    return false;
  return true;
}

Voxelgrid::Ptr Voxelgrid::FromFileBinary(const std::string &filename)
{
  std::ifstream ifile(filename, std::ios::binary);

  if (!ifile)
  {
    ROS_ERROR("Voxelgrid::FromFileBinary: cannot open file %s", filename.c_str());
    return Voxelgrid::Ptr();
  }

  const std::string magic = "VXGR";

  char maybe_magic[5];
  ifile.read(maybe_magic, 4);
  maybe_magic[4] = 0;
  if (magic != maybe_magic)
  {
    ROS_ERROR("Voxelgrid::FromFileBinary: file must start with %s, '%s' found instead", magic.c_str(), maybe_magic);
    return Voxelgrid::Ptr();
  }

  uint32 version, width, height, depth;
  ifile.read((char *)&version, sizeof(version));
  ifile.read((char *)&width, sizeof(version));
  ifile.read((char *)&height, sizeof(version));
  ifile.read((char *)&depth, sizeof(version));

  if (!ifile || version != 1)
  {
    ROS_ERROR("Voxelgrid::FromFileBinary: invalid version %u", unsigned(version));
    return Voxelgrid::Ptr();
  }

  Voxelgrid::Ptr result(new Voxelgrid(width, height, depth));

  for (uint64 z = 0; z < depth; z++)
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        float v;
        ifile.read((char *)&v, sizeof(v));
        result->at(x, y, z) = v;
      }

  if (!ifile)
    return Voxelgrid::Ptr();
  return result;
}

void Voxelgrid::ToStream(std::ostream & ostr) const
{
  const uint64 width = GetWidth();
  const uint64 height = GetHeight();
  const uint64 depth = GetDepth();

  ostr << "VOXELGRID " << m_width << " " << m_height << " " << m_depth << "\n";
  for (uint64 z = 0; z < depth; z++)
  {
    ostr << "Depth " << z << "\n";
    for (uint64 y = 0; y < height; y++)
    {
      for (uint64 x = 0; x < width; x++)
      {
        ostr << at(x, y, z) << " ";
      }
      ostr << "\n";
    }
  }

  ostr << "DIRGLEXOV\n";
}

std::string Voxelgrid::ToString() const
{
  std::ostringstream ostr;
  ToStream(ostr);
  return ostr.str();
}

Voxelgrid::Ptr Voxelgrid::FromStream(std::istream & istr)
{
  std::string magic;
  istr >> magic;
  if (magic != "VOXELGRID")
  {
    ROS_ERROR("Voxelgrid::FromString: file must start with VOXELGRID, '%s' found instead", magic.c_str());
    return Voxelgrid::Ptr();
  }

  uint64 width, height, depth;
  istr >> width >> height >> depth;
  if (!istr)
  {
    ROS_ERROR("Voxelgrid::FromString: width, height, depth expected.");
    return Voxelgrid::Ptr();
  }

  Voxelgrid::Ptr result(new Voxelgrid(width, height, depth));
  while (istr >> magic)
  {
    if (magic == "Depth")
    {
      uint64 z;
      if (!(istr >> z))
      {
        ROS_ERROR("Voxelgrid::FromString: z coordinate expected after Depth.");
        return Voxelgrid::Ptr();
      }

      if (z >= depth)
      {
        ROS_ERROR("Voxelgrid::FromString: z coordinate %u >= maximum %u.",
                  unsigned(z), unsigned(depth));
        return Voxelgrid::Ptr();
      }

      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          float val;
          if (!(istr >> val))
          {
            ROS_ERROR("Voxelgrid::FromString: Float value expected at coords %u %u %u.",
                      unsigned(x), unsigned(y), unsigned(z));
            return Voxelgrid::Ptr();
          }

          result->at(x, y, z) = val;
        }
    }
    else if (magic == "DIRGLEXOV")
    {
      return result;
    }
    else
    {
      ROS_ERROR("Voxelgrid::FromString: expected magic or depth, got %s.", magic.c_str());
      return Voxelgrid::Ptr();
    }
  }

  ROS_ERROR("Voxelgrid::FromString: unexpected end of file.");
  return Voxelgrid::Ptr();
}

Voxelgrid::Ptr Voxelgrid::FromString(const std::string & str)
{
  std::istringstream istr(str);
  return FromStream(istr);
}

Voxelgrid::Ptr Voxelgrid::FromFile(const std::string & filename)
{
  std::ifstream ifile(filename.c_str());
  return FromStream(ifile);
}

bool Voxelgrid::ToFile(const std::string & filename) const
{
  std::ofstream ofile(filename.c_str());
  ToStream(ofile);
  if (!ofile)
  {
    ROS_ERROR("Voxelgrid::Tofile: could not save file %s", filename.c_str());
    return false;
  }
  return true;
}

Voxelgrid::Ptr Voxelgrid::Transpose(const uint64 axis0, const uint64 axis1, const uint64 axis2)
{
  const Eigen::Vector3i size = GetSize();
  Voxelgrid::Ptr result(new Voxelgrid(size[axis0], size[axis1], size[axis2]));

  for (uint64 z = 0; z < m_depth; z++)
    for (uint64 y = 0; y < m_height; y++)
      for (uint64 x = 0; x < m_width; x++)
      {
        const Eigen::Vector3i xyz(x, y, z);
        result->at(xyz[axis0], xyz[axis1], xyz[axis2]) = at(xyz);
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::Reflect(const uint64 axis)
{
  const Eigen::Vector3i size = GetSize();
  Voxelgrid::Ptr result(new Voxelgrid(size));

  for (uint64 z = 0; z < m_depth; z++)
    for (uint64 y = 0; y < m_height; y++)
      for (uint64 x = 0; x < m_width; x++)
      {
        Eigen::Vector3i xyz(x, y, z);
        xyz[axis] = size[axis] - xyz[axis] - 1;
        result->at(x, y, z) = at(xyz);
      }

  return result;
}

Voxelgrid::Ptr Voxelgrid::Rotate90(const uint64 axis1, const uint64 axis2)
{
  uint64 xyz[3] = {0, 1, 2};
  xyz[axis1] = axis2;
  xyz[axis2] = axis1;

  return Reflect(axis1)->Transpose(xyz[0], xyz[1], xyz[2]);
}

Voxelgrid::Ptr Voxelgrid::Rotate90n(const uint64 axis1, const uint64 axis2, const uint64 n)
{
  if (n == 0)
    return Voxelgrid::Ptr(new Voxelgrid(*this));

  Voxelgrid::Ptr result = Rotate90(axis1, axis2);
  for (uint64 i = 1; i < n; i++)
    result = result->Rotate90(axis1, axis2);

  return result;
}

Voxelgrid::Ptr Voxelgrid::Threshold(const float th, const float value_if_above, const float value_if_below) const
{
  Voxelgrid::Ptr result(new Voxelgrid(GetSize()));

  for (uint64 z = 0; z < m_depth; z++)
    for (uint64 y = 0; y < m_height; y++)
      for (uint64 x = 0; x < m_width; x++)
      {
        const Eigen::Vector3i xyz(x, y, z);
        result->at(xyz) = (at(xyz) > th) ? value_if_above : value_if_below;
      }

  return result;
}

// ------------------- VOXELGRID4 ------------------
Voxelgrid4::Voxelgrid4(const uint64 width, const uint64 height, const uint64 depth):
  Voxelgrid4(Eigen::Vector3i(width, height, depth))
{}

Voxelgrid4::Voxelgrid4(const uint64 width, const uint64 height):
  Voxelgrid4(Eigen::Vector3i(width, height, 0))
{}

Voxelgrid4::Voxelgrid4(const Eigen::Vector3i & size):
  m_grids{Voxelgrid(size), Voxelgrid(size), Voxelgrid(size), Voxelgrid(size)}
{
  m_depth = size.z();
  m_height = size.y();
  m_width = size.x();
}

cv::Mat Voxelgrid4::ToOpenCVImage2D() const
{
  const uint64 width = GetWidth();
  const uint64 height = GetHeight();

  cv::Mat result(width, height, CV_8UC4);

  cv::Mat ocv[4];
  for (uint64 i = 0; i < 4; i++)
    ocv[i] = m_grids[i].ToOpenCVImage2D();

  for (uint64 y = 0; y < height; y++)
    for (uint64 x = 0; x < width; x++)
      for (uint64 i = 0; i < 4; i++)
        result.at<cv::Vec4b>(y, x)[i] = ocv[i].at<uint8>(y, x);

  return result;
}
bool Voxelgrid4::SaveOpenCVImage2D(const std::string & filename) const
{
  cv::Mat image = ToOpenCVImage2D();
  return cv::imwrite(filename, image);
}

bool Voxelgrid4::SaveOctomapOctree(const std::string & filename_prefix) const
{
  for (uint64 i = 0; i < 4; i++)
    if (!m_grids[i].SaveOctomapOctree(filename_prefix + std::to_string(i) + ".bt"))
      return false;
  return true;
}

bool Voxelgrid4::ToFile(const std::string & filename_prefix) const
{
  for (uint64 i = 0; i < 4; i++)
    if (!m_grids[i].ToFile(filename_prefix + std::to_string(i) + ".voxelgrid"))
      return false;
  return true;
}

bool Voxelgrid4::ToFileBinary(const std::string & filename_prefix) const
{
  for (uint64 i = 0; i < 4; i++)
    if (!m_grids[i].ToFileBinary(filename_prefix + std::to_string(i) + ".binvoxelgrid"))
      return false;
  return true;
}

Voxelgrid4::Ptr Voxelgrid4::FromOpenCVImage2D(const cv::Mat & cv_mat)
{
  const uint64 height = cv_mat.rows;
  const uint64 width = cv_mat.cols;

  Voxelgrid4::Ptr result_ptr(new Voxelgrid4(width, height, 1));
  Voxelgrid4 & result = *result_ptr;

  if (cv_mat.type() == CV_8UC3)
  {
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        for (uint64 i = 0; i < 3; i++)
          result.at(i).at(x, y, 0) = cv_mat.at<cv::Vec3b>(y, x)[i] / 255.0f;
        result.at(3).at(x, y, 0) = 1.0f;
      }
  }
  else if (cv_mat.type() == CV_32FC3)
  {
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
      {
        for (uint64 i = 0; i < 3; i++)
          result.at(i).at(x, y, 0) = cv_mat.at<cv::Vec3f>(y, x)[i];
        result.at(3).at(x, y, 0) = 1.0f;
      }
  }
  else if (cv_mat.type() == CV_32FC4)
  {
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        for (uint64 i = 0; i < 4; i++)
          result.at(i).at(x, y, 0) = cv_mat.at<cv::Vec4f>(y, x)[i];
  }
  else if (cv_mat.type() == CV_8UC4)
  {
    for (uint64 y = 0; y < height; y++)
      for (uint64 x = 0; x < width; x++)
        for (uint64 i = 0; i < 4; i++)
          result.at(i).at(x, y, 0) = cv_mat.at<cv::Vec4b>(y, x)[i] / 255.0f;
  }
  else
  {
    ROS_ERROR("FromOpenCVImage2D: invalid matrix type: %u", unsigned(cv_mat.type()));
  }

  return result_ptr;
}

Voxelgrid4::Ptr Voxelgrid4::FromFloat32MultiArray(const std_msgs::Float32MultiArray & arr)
{
  const std_msgs::MultiArrayLayout & layout = arr.layout;

  auto dim_x = std::find_if(layout.dim.begin(), layout.dim.end(),
                            [](const std_msgs::MultiArrayDimension & dim) -> bool
                            {return dim.label == "x"; });
  auto dim_y = std::find_if(layout.dim.begin(), layout.dim.end(),
                            [](const std_msgs::MultiArrayDimension & dim) -> bool
                            {return dim.label == "y"; });
  auto dim_z = std::find_if(layout.dim.begin(), layout.dim.end(),
                            [](const std_msgs::MultiArrayDimension & dim) -> bool
                            {return dim.label == "z"; });
  auto dim_channels = std::find_if(layout.dim.begin(), layout.dim.end(),
                                   [](const std_msgs::MultiArrayDimension & dim) -> bool
                                   {return dim.label == "channels"; });

  if (dim_x == layout.dim.end() || dim_y == layout.dim.end() || dim_z == layout.dim.end() ||
      dim_channels == layout.dim.end())
  {
    ROS_ERROR("Voxelgrid4: could not find x, y, z, channels in Float32MultiArray");
    return Ptr();
  }

  if (dim_x->size && dim_y->size && dim_z->size && dim_channels->size)
  {
    const uint64 expected_size = (dim_x->size - 1) * dim_x->stride +
      (dim_y->size - 1) * dim_y->stride + (dim_z->size - 1) * dim_z->stride +
      (dim_channels->size - 1) * dim_channels->stride + 1;
    if (expected_size > arr.data.size())
    {
      ROS_ERROR("Voxelgrid4: array size does not match in Float32MultiArray, it is %u, it should be at least %u",
                unsigned(arr.data.size()), unsigned(expected_size));
      return Ptr();
    }
  }

  if (dim_channels->size != 4)
  {
    ROS_ERROR("Voxelgrid4: expected exactly 4 channels in Float32MultiArray, it is %u", unsigned(dim_channels->size));
    return Ptr();
  }

  Voxelgrid4::Ptr result(new Voxelgrid4(dim_x->size, dim_y->size, dim_z->size));
  for (uint64 channel = 0; channel < 4; channel++)
    for (uint64 z = 0; z < dim_z->size; z++)
      for (uint64 y = 0; y < dim_y->size; y++)
        for (uint64 x = 0; x < dim_x->size; x++)
        {
          result->at(channel).at(x, y, z) = arr.data[x * dim_x->stride + y * dim_y->stride +
                                                     z * dim_z->stride + channel * dim_channels->stride];
        }

  return result;
}

bool Voxelgrid4::Save2D3D(const std::string & filename_prefix, const bool is_3d) const
{
  if (!is_3d)
    return SaveOpenCVImage2D(filename_prefix + ".png");
  else
    return SaveOctomapOctree(filename_prefix) &&
           ToFileBinary(filename_prefix);
}
