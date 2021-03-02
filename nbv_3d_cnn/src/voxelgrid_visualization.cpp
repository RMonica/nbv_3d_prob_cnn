#include "voxelgrid_visualization.h"


#include <stdint.h>
#include <string>
#include <vector>
#include <sstream>

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>

#include <nbv_3d_cnn/voxelgrid.h>

#define FRAME_ID "map"

class VoxelgridVisualization
{
  public:
  typedef uint64_t uint64;

  typedef std::vector<Voxelgrid::ConstPtr> VoxelgridConstPtrVector;
  typedef std::vector<std_msgs::ColorRGBA> ColorRGBAVector;
  typedef pcl::PointCloud<pcl::PointXYZRGBA> PointXYZRGBACloud;

  enum class SequenceMode
  {
    ONE_SHOT,
    VIDEO,
  };

  VoxelgridVisualization(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    double param_double;

    m_nh.param<std::string>(PARAM_NAME_OCTOMAP_NAME, m_filename, PARAM_DEFAULT_OCTOMAP_NAME);

    m_nh.param<std::string>(PARAM_NAME_VOXELGRID_NAME_PREFIX, m_voxelgrid_filename_prefix, PARAM_DEFAULT_VOXELGRID_NAME_PREFIX);
    m_nh.param<std::string>(PARAM_NAME_EMPTY_VOXELGRID_NAME, m_voxelgrid_empty_filename, PARAM_DEFAULT_EMPTY_VOXELGRID_NAME);
    m_nh.param<std::string>(PARAM_NAME_EMPTY_VOXELGRID_SUFFIX, m_voxelgrid_empty_suffix, PARAM_DEFAULT_EMPTY_VOXELGRID_SUFFIX);

    m_nh.param<std::string>(PARAM_NAME_OCCUPIED_VOXELGRID_NAME, m_voxelgrid_occupied_filename, PARAM_DEFAULT_OCCUPIED_VOXELGRID_NAME);
    m_nh.param<std::string>(PARAM_NAME_OCCUPIED_VOXELGRID_SUFFIX, m_voxelgrid_occupied_suffix, PARAM_DEFAULT_OCCUPIED_VOXELGRID_SUFFIX);

    m_nh.param<double>(PARAM_NAME_INITIAL_DELAY, m_initial_delay, PARAM_DEFAULT_INITIAL_DELAY);
    m_nh.param<double>(PARAM_NAME_FRAME_DELAY, m_frame_delay, PARAM_DEFAULT_FRAME_DELAY);

    m_nh.param<std::string>(PARAM_NAME_COLORS, param_string, PARAM_DEFAULT_COLORS);
    {
      std::istringstream istr(param_string);
      float r, g, b;
      while (istr >> r >> g >> b)
      {
        std_msgs::ColorRGBA color;
        color.r = r;
        color.g = g;
        color.b = b;
        color.a = 1.0f;
        m_colors.push_back(color);
      }

      ROS_INFO("voxelgrid_visualization: loaded %u colors from '%s'", unsigned(m_colors.size()), param_string.c_str());
    }

    m_nh.param<std::string>(PARAM_NAME_SEQUENCE_MODE, param_string, PARAM_DEFAULT_SEQUENCE_MODE);
    if (param_string == PARAM_VALUE_SEQUENCE_ONE_SHOT)
      m_sequence_mode = SequenceMode::ONE_SHOT;
    else if (param_string == PARAM_VALUE_SEQUENCE_VIDEO)
      m_sequence_mode = SequenceMode::VIDEO;
    else
    {
      ROS_FATAL("voxelgrid_visualization: unknown sequence mode: %s", param_string.c_str());
      exit(1);
    }

    m_nh.param<std::string>(PARAM_NAME_MARKER_OUT_TOPIC, param_string, PARAM_DEFAULT_MARKER_OUT_TOPIC);
    m_marker_publisher = m_nh.advertise<visualization_msgs::MarkerArray>(param_string, 1);

    m_nh.param<std::string>(PARAM_NAME_CLOUD_OUT_TOPIC, param_string, PARAM_DEFAULT_CLOUD_OUT_TOPIC);
    m_cloud_publisher = m_nh.advertise<sensor_msgs::PointCloud2>(param_string, 1);

    m_nh.param<std::string>(PARAM_NAME_NAMESPACE, m_namespace, PARAM_DEFAULT_NAMESPACE);

    m_nh.param<bool>(PARAM_NAME_USE_SEQUENCE_COUNTER, m_use_sequence_counter, PARAM_DEFAULT_USE_SEQUENCE_COUNTER);

    m_nh.param<bool>(PARAM_NAME_USE_RAINBOW, m_use_rainbow, PARAM_DEFAULT_USE_RAINBOW);

    m_nh.param<double>(PARAM_NAME_OCCUPANCY_TH, param_double, PARAM_DEFAULT_OCCUPANCY_TH);
    m_occupancy_th = param_double;

    m_nh.param<std::string>(PARAM_NAME_VOXELGRID_SIZE, param_string, PARAM_DEFAULT_VOXELGRID_SIZE);
    m_has_voxelgrid_size = false;
    if (param_string != "")
    {
      std::istringstream istr(param_string);
      istr >> m_voxelgrid_size.x() >> m_voxelgrid_size.y() >> m_voxelgrid_size.z();
      if (!istr)
      {
        ROS_ERROR("voxelgrid_visualization: could not parse voxelgrid size string %s", param_string.c_str());
      }
      else
      {
        m_has_voxelgrid_size = true;
        ROS_INFO_STREAM("voxelgrid_visualization: voxelgrid size is " << m_voxelgrid_size.transpose());
      }
    }

    m_sequence_counter = 0;

    m_terminated = false;

    ROS_INFO("voxelgrid_visualization: waiting %f seconds.", m_frame_delay);
    m_timer = m_nh.createTimer(ros::Duration(m_frame_delay), &VoxelgridVisualization::onTimer, this, true);
  }

  visualization_msgs::Marker GetDeleteAllMarker(const std::string ns)
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = FRAME_ID;
    marker.action = marker.DELETEALL;
    marker.id = 0;
    marker.ns = ns;
    return marker;
  }

  PointXYZRGBACloud VoxelGridToCloud(const Voxelgrid::ConstPtr vgp, const std_msgs::ColorRGBA & color_in)
  {
    ROS_INFO("voxelgrid_visualization: building cloud.");

    const Voxelgrid & vg = *vgp;

    const uint64 depth = vg.GetDepth();
    const uint64 height = vg.GetHeight();
    const uint64 width = vg.GetWidth();

    PointXYZRGBACloud cloud;

    std_msgs::ColorRGBA color = color_in;
    color.a = 1.0f;

    const float SCALE = 0.1;

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const Eigen::Vector3f gpt(x - width/2.0f, y - height/2.0f, z - depth/2.0f);
          const Eigen::Vector3f pt = gpt * SCALE;

          pcl::PointXYZRGBA ppt;
          ppt.x = pt.x();
          ppt.y = pt.y();
          ppt.z = pt.z();
          ppt.r = std::round(color.r * 255.0f);
          ppt.g = std::round(color.g * 255.0f);
          ppt.b = std::round(color.b * 255.0f);
          ppt.a = std::round(color.a * 255.0f);

          const float v = vg.at(x, y, z);

          if (v < m_occupancy_th)
            continue;

          cloud.push_back(ppt);
        }

    cloud.is_dense = true;
    cloud.height = 1;
    cloud.width = cloud.size();

    return cloud;
  }

  Eigen::Vector3i Rainbow(double ratio)
  {
    pcl::PointXYZRGB rgb;
    pcl::PointXYZHSV hsv;
    hsv.h = std::min<double>(ratio * 360.0, 359.0);
    hsv.s = 1.0;
    hsv.v = 1.0;

    pcl::PointXYZHSVtoXYZRGB(hsv, rgb);

    Eigen::Vector3i rgb_v;
    rgb_v.x() = rgb.r;
    rgb_v.y() = rgb.g;
    rgb_v.z() = rgb.b;
    return rgb_v;
  }

  visualization_msgs::Marker VoxelGridToMsg(const uint64 id, const Voxelgrid::ConstPtr vgp, const std_msgs::ColorRGBA & color_in)
  {
    ROS_INFO("voxelgrid_visualization: building voxelgrid.");

    const Voxelgrid & vg = *vgp;

    const uint64 depth = vg.GetDepth();
    const uint64 height = vg.GetHeight();
    const uint64 width = vg.GetWidth();

    visualization_msgs::Marker cubes_marker;
    cubes_marker.header.frame_id = "map";
    cubes_marker.header.stamp = ros::Time::now();
    cubes_marker.type = cubes_marker.CUBE_LIST;
    cubes_marker.action = cubes_marker.ADD;
    cubes_marker.id = id;
    cubes_marker.ns = m_namespace;

    tf::poseEigenToMsg(Eigen::Affine3d::Identity(), cubes_marker.pose);

    std_msgs::ColorRGBA color = color_in;
    color.a = 1.0f;

    const float SCALE = 0.1;

    Eigen::Vector3i maxes(Eigen::Vector3i::Zero());
    Eigen::Vector3i mines(Eigen::Vector3i(width, height, depth));
    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const float v = vg.at(x, y, z);
          if (v < m_occupancy_th)
            continue;
          maxes = maxes.array().max(Eigen::Vector3i(x, y, z).array());
          mines = mines.array().min(Eigen::Vector3i(x, y, z).array());
        }

    for (uint64 z = 0; z < depth; z++)
      for (uint64 y = 0; y < height; y++)
        for (uint64 x = 0; x < width; x++)
        {
          const Eigen::Vector3f gpt(x - width/2.0f, y - height/2.0f, z - depth/2.0f);
          const Eigen::Vector3f pt = gpt * SCALE;

          cubes_marker.scale.x = SCALE;
          cubes_marker.scale.y = SCALE;
          cubes_marker.scale.z = SCALE;

          const float v = vg.at(x, y, z);

          if (v < m_occupancy_th)
            continue;

          geometry_msgs::Point msg_pt;
          msg_pt.x = pt.x();
          msg_pt.y = pt.y();
          msg_pt.z = pt.z();
          cubes_marker.points.push_back(msg_pt);

          if (m_use_rainbow)
          {
            const Eigen::Vector3i rainbow = Rainbow(float(z - mines.z()) / (maxes.z() - mines.z()));
            color.r = rainbow.x() / 255.0f;
            color.g = rainbow.y() / 255.0f;
            color.b = rainbow.z() / 255.0f;
          }

          cubes_marker.colors.push_back(color);
        }

    return cubes_marker;
  }

  void onTimer(const ros::TimerEvent &)
  {
    if (m_terminated)
      return;

    if (m_sequence_mode == SequenceMode::VIDEO)
    {
      m_timer.stop();
      m_timer.setPeriod(ros::Duration(m_frame_delay));
      m_sequence_counter++;
      ROS_INFO("voxelgrid_visualization: re-scheduling.");
      m_timer.start();
    }

    ROS_INFO("voxelgrid_visualization: loading.");

    visualization_msgs::MarkerArray markers;
    markers.markers.push_back(GetDeleteAllMarker(m_namespace));
    uint64 color_counter = 0;

    PointXYZRGBACloud cloud;

    Voxelgrid::ConstPtr vg;
    std_msgs::ColorRGBA vg_color;
    if (!m_filename.empty())
    {
      vg_color = m_colors[color_counter++];
      if (m_has_voxelgrid_size)
        vg = Voxelgrid::Load3DOctomapWithISize(m_filename, m_voxelgrid_size);
      else
        vg = Voxelgrid::Load3DOctomap(m_filename);
      if (!vg)
        m_terminated = true;
    }

    const std::string sequence_counter_string = m_use_sequence_counter ? std::to_string(m_sequence_counter) : std::string();

    if (!m_voxelgrid_occupied_filename.empty() || !m_voxelgrid_occupied_suffix.empty())
    {
      const std::string occupied_filename = m_voxelgrid_filename_prefix + m_voxelgrid_occupied_filename +
                                            sequence_counter_string + m_voxelgrid_occupied_suffix;

      const Voxelgrid::ConstPtr vg_occupied = Voxelgrid::FromFileBinary(occupied_filename);

      if (!vg_occupied)
        m_terminated = true;

      if (!m_terminated)
      {
        markers.markers.push_back(VoxelGridToMsg(1, vg_occupied, m_colors[color_counter]));
        cloud += VoxelGridToCloud(vg_occupied, m_colors[color_counter]);
        color_counter++;
      }

      if (!m_voxelgrid_empty_filename.empty() || !m_voxelgrid_empty_suffix.empty())
      {
        const std::string empty_filename = m_voxelgrid_filename_prefix + m_voxelgrid_empty_filename +
                                           sequence_counter_string + m_voxelgrid_empty_suffix;
        const Voxelgrid::ConstPtr vg_empty = Voxelgrid::FromFileBinary(empty_filename);
        if (vg_empty)
        {
          Voxelgrid::ConstPtr vg_unknown = vg_empty->Or(*vg_occupied)->Not();
          if (vg)
          {
            vg_unknown = vg_unknown->AndNot(*vg);
            vg = vg->AndNot(*vg_occupied);
          }
          markers.markers.push_back(VoxelGridToMsg(2, vg_unknown, m_colors[color_counter]));
          cloud += VoxelGridToCloud(vg_unknown, m_colors[color_counter]);
          color_counter++;
        }
      }
    }

    if (vg)
    {
      markers.markers.push_back(VoxelGridToMsg(0, vg, vg_color));
      cloud += VoxelGridToCloud(vg, vg_color);
    }

    ROS_INFO("voxelgrid_visualization: publishing.");

    if (!m_terminated)
    {
      m_marker_publisher.publish(markers);

      sensor_msgs::PointCloud2 cloud_msg;
      pcl::toROSMsg(cloud, cloud_msg);
      cloud_msg.header.frame_id = FRAME_ID;
      m_cloud_publisher.publish(cloud_msg);
    }

    if (m_terminated)
    {
      m_timer.stop();
      m_terminated = true;
      ROS_INFO("voxelgrid_visualization: stopping.");
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  private:
  ros::NodeHandle & m_nh;

  ros::Timer m_timer;
  ros::Publisher m_marker_publisher;
  ros::Publisher m_cloud_publisher;

  std::string m_filename;

  std::string m_voxelgrid_filename_prefix;
  std::string m_voxelgrid_occupied_filename;
  std::string m_voxelgrid_occupied_suffix;
  std::string m_voxelgrid_empty_filename;
  std::string m_voxelgrid_empty_suffix;

  double m_initial_delay;
  double m_frame_delay;

  bool m_terminated;

  bool m_use_sequence_counter;
  float m_occupancy_th;

  ColorRGBAVector m_colors;
  std::string m_namespace;

  SequenceMode m_sequence_mode;
  uint64 m_sequence_counter;

  bool m_has_voxelgrid_size;
  Eigen::Vector3i m_voxelgrid_size;

  bool m_use_rainbow;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "voxelgrid_visualization");

  ros::NodeHandle nh("~");

  VoxelgridVisualization vv(nh);

  ros::spin();

  return 0;
}


