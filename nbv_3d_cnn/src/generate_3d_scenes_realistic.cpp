#include "generate_3d_scenes_realistic.h"

#include <ros/ros.h>

#include <string>
#include <stdint.h>
#include <sstream>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

class Generate3DScenesRealistic
{
  public:
  typedef std::vector<std::string> StringVector;
  typedef uint64_t uint64;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Vector3f> > Affine3fVector;
  typedef pcl::PointCloud<pcl::PointXYZ> PointXYZCloud;

  typedef std::vector<PointXYZCloud, Eigen::aligned_allocator<PointXYZCloud> > PointXYZCloudVector;

  typedef octomap::OcTree OcTree;
  typedef std::shared_ptr<OcTree> OcTreePtr;
  typedef std::shared_ptr<const OcTree> OcTreeConstPtr;

  static double Rand01d() { return double(rand()) / double(RAND_MAX); }

  static float Rand01f() { return float(rand()) / float(RAND_MAX); }
  static float Rand0505f() { return float(rand()) / float(RAND_MAX) - 0.5f; }

  Generate3DScenesRealistic(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;
    double param_double;

    m_nh.param<std::string>(PARAM_NAME_OBJECT_FOLDER_PREFIX, param_string, PARAM_DEFAULT_OBJECT_FOLDER_PREFIX);
    m_object_folder_prefix = param_string;

    m_nh.param<std::string>(PARAM_NAME_OBJECT_FOLDER_LIST, param_string, PARAM_DEFAULT_OBJECT_FOLDER_LIST);
    {
      std::istringstream istr(param_string);
      std::string str;
      while (istr >> str)
        m_object_folder_list.push_back(str);
    }

    m_nh.param<std::string>(PARAM_NAME_OBJECT_SUFFIX_LIST, param_string, PARAM_DEFAULT_OBJECT_SUFFIX_LIST);
    {
      std::istringstream istr(param_string);
      std::string str;
      while (istr >> str)
        m_object_suffix_list.push_back(str);
    }

    m_nh.param<std::string>(PARAM_NAME_OUTPUT_SCENE_PREFIX, param_string, PARAM_DEFAULT_OUTPUT_SCENE_PREFIX);
    m_output_scene_prefix = param_string;

    m_nh.param<int>(PARAM_NAME_MAX_SCENES, param_int, PARAM_DEFAULT_MAX_SCENES);
    m_max_scenes = param_int;

    m_nh.param<int>(PARAM_NAME_SCENE_SIZE, param_int, PARAM_DEFAULT_SCENE_SIZE);
    m_scene_size = param_int;

    m_nh.param<int>(PARAM_NAME_SCENE_HEIGHT, param_int, PARAM_DEFAULT_SCENE_HEIGHT);
    m_scene_height = param_int;

    m_nh.param<double>(PARAM_NAME_SCENE_RESOLUTION, param_double, PARAM_DEFAULT_SCENE_RESOLUTION);
    m_scene_resolution = param_double;

    m_nh.param<double>(PARAM_NAME_SCENE_SPREAD, param_double, PARAM_DEFAULT_SCENE_SPREAD);
    m_scene_spread = param_double;

    m_nh.param<int>(PARAM_NAME_OBJECTS_PER_SCENE_MIN, param_int, PARAM_DEFAULT_OBJECTS_PER_SCENE_MIN);
    m_objects_per_scene_min = param_int;

    m_nh.param<int>(PARAM_NAME_OBJECTS_PER_SCENE_MAX, param_int, PARAM_DEFAULT_OBJECTS_PER_SCENE_MAX);
    m_objects_per_scene_max = param_int;

    LoadObjects();

    m_scene_counter = 1;
    m_timer = m_nh.createTimer(ros::Duration(0.0), &Generate3DScenesRealistic::onTimer, this, true);
  }

  void LoadObjects()
  {
    ROS_INFO("generate_3d_scenes: loading objects.");
    for (uint64 i = 0; i < m_object_folder_list.size(); i++)
    {
      const std::string folder_name = m_object_folder_list[i];
      const std::string suffix = m_object_suffix_list[i];
      const std::string prefix = m_object_folder_prefix + folder_name;
      ROS_INFO("generate_3d_scenes: loading prefix %s", prefix.c_str());

      uint64 file_counter = 1;
      while (true)
      {
        std::ostringstream ofilename;
        ofilename << std::setfill('0') << std::setw(2) << file_counter << std::flush;
        const std::string filename = prefix + ofilename.str() + suffix;
        ROS_INFO("generate_3d_scenes: loading file %s", filename.c_str());

        PointXYZCloud cloud;
        bool success = !pcl::io::loadPCDFile(filename, cloud);
        if (!success)
          break;
        m_object_point_clouds.push_back(cloud);
        file_counter++;
      }
    }

    ROS_INFO("generate_3d_scenes: loaded %u objects.", unsigned(m_object_point_clouds.size()));
  }

  int RareNoise()
  {
    const int rarity = 20;
    int noise = rand() % rarity;
    if (noise == 0)
      noise = -1;
    else if (noise == (rarity - 1))
      noise = 1;
    else
      noise = 0;
    return noise;
  }

  int HalfNoise()
  {
    return rand() % 2;
  }

  void onTimer(const ros::TimerEvent &)
  {
    ROS_INFO("generate_3d_scenes: processing scene %u.", unsigned(m_scene_counter));

    const float scene_side_mt = m_scene_resolution * m_scene_size;
    const float scene_height_mt = m_scene_resolution * m_scene_height;

    const uint64 objects_per_this_scene = m_objects_per_scene_min +
      (rand() % (1 + m_objects_per_scene_max - m_objects_per_scene_min));
    ROS_INFO("generate_3d_scenes: this scene contains %u objects.", unsigned(objects_per_this_scene));

    Uint64Vector objects_in_this_scene;
    for (uint64 i = 0; i < objects_per_this_scene; i++)
      objects_in_this_scene.push_back(rand() % m_object_point_clouds.size());

    OcTreePtr scene(new OcTree(m_scene_resolution));

    const Eigen::Vector3f bbox_min(-scene_side_mt / 2.0f, -scene_side_mt / 2.0f, -scene_height_mt / 2.0f);
    const Eigen::Vector3f bbox_max(scene_side_mt / 2.0f, scene_side_mt / 2.0f, scene_height_mt / 2.0f);

    {
      octomap::point3d bmax(bbox_max.x(), bbox_max.y(), bbox_max.z());
      octomap::point3d bmin(bbox_min.x(), bbox_min.y(), bbox_min.z());
      scene->setBBXMax(bmax);
      scene->setBBXMin(bmin);
      scene->useBBXLimit(true);
    }

    const int half_scene_size = int(m_scene_size) / 2;
    const int half_scene_height = int(m_scene_height) / 2;

    // main plane
    const int PLANE_AVG_THICKNESS = 6.0f * 0.0058f / m_scene_resolution;
    const float plane_height = (PLANE_AVG_THICKNESS + HalfNoise() - half_scene_height) * m_scene_resolution;

    for (uint64 obj_i = 0; obj_i < objects_per_this_scene; obj_i++)
    {
      const PointXYZCloud & object_cloud = m_object_point_clouds[objects_in_this_scene[obj_i]];
      pcl::PointXYZ cloud_min;
      pcl::PointXYZ cloud_max;
      pcl::getMinMax3D(object_cloud, cloud_min, cloud_max);

      Eigen::Vector3f avg_center = (Eigen::Vector3f(cloud_min.x, cloud_min.y, cloud_min.z) +
                                    Eigen::Vector3f(cloud_max.x, cloud_max.y, cloud_max.z)) / 2.0f;
      avg_center.z() = cloud_min.z;
      Eigen::Vector3f plane_translation(Eigen::Vector3f(0.0f, 0.0f, plane_height + m_scene_resolution * 2.0f));

      for (uint64 attempt_i = 0; attempt_i < 12; attempt_i++)
      {
        const float angle = M_PI * 2.0f * Rand01d();
        Eigen::Affine3f transform;
        transform.linear() = Eigen::AngleAxisf(angle, Eigen::Vector3f(0.0f, 0.0f, 1.0f)).matrix();

        const Eigen::Vector3f translation(scene_side_mt * Rand0505f() * m_scene_spread,
                                          scene_side_mt * Rand0505f() * m_scene_spread, 0.0f);
        transform.translation() = translation;

        transform = transform * Eigen::Translation3f(plane_translation) * Eigen::Translation3f(-avg_center);

        PointXYZCloud transformed_point_cloud;
        pcl::transformPointCloud(object_cloud, transformed_point_cloud, transform);

        // check free space
        bool found_occupied = false;
        for (uint64 point_i = 0; point_i < transformed_point_cloud.size(); point_i++)
        {
          const pcl::PointXYZ & pt = transformed_point_cloud[point_i];
          const Eigen::Vector3f ept(pt.x, pt.y, pt.z);

          if ((ept.array() <= bbox_min.array()).any())
            continue;
          if ((ept.array() >= bbox_max.array()).any())
            continue;

          const octomap::point3d new_point(ept.x(), ept.y(), ept.z());
          octomap::OcTreeNode * scene_node = scene->search(new_point);
          if (scene_node)
            found_occupied = true;
        }

        if (found_occupied)
          continue;

        // if free space, place object
        for (uint64 point_i = 0; point_i < transformed_point_cloud.size(); point_i++)
        {
          const pcl::PointXYZ & pt = transformed_point_cloud[point_i];
          const Eigen::Vector3f ept(pt.x, pt.y, pt.z);

          if ((ept.array() <= bbox_min.array()).any())
            continue;
          if ((ept.array() >= bbox_max.array()).any())
            continue;

          const octomap::point3d new_point(ept.x(), ept.y(), ept.z());
          octomap::OcTreeNode * scene_node = scene->search(new_point);
          if (!scene_node)
            scene_node = scene->updateNode(new_point, true);
          scene_node->setLogOdds(octomap::logodds(0.99f));
        }

        break;
      }

    }

    // main plane
    {
      for (uint64 y = 0; y < m_scene_size; y++)
        for (uint64 x = 0; x < m_scene_size; x++)
        {
          const int noise = RareNoise();
          const int noise2 = RareNoise();
          //const int noise = 0, noise2 = 0;
          const int max_z_i = int(std::floor(plane_height / m_scene_resolution)) + noise + 1;
          const int min_z_i = std::max<int>(max_z_i - PLANE_AVG_THICKNESS + noise2, -int(m_scene_size / 2 - 1));
          Eigen::Vector3f ept((x + 0.5f) * m_scene_resolution - scene_side_mt / 2.0f,
                              (y + 0.5f) * m_scene_resolution - scene_side_mt / 2.0f,
                              0.0f);

          for (int z = -half_scene_height; z <= max_z_i; z++)
          {
            const octomap::point3d new_point(ept.x(), ept.y(), (z + 0.5f) * m_scene_resolution);
            octomap::OcTreeNode * scene_node = scene->search(new_point);
            if (!scene_node)
              scene_node = scene->updateNode(new_point, true);
            scene_node->setLogOdds(octomap::logodds(0.99f));
          }
        }
    }

    const std::string out_file = m_output_scene_prefix + std::to_string(m_scene_counter) + ".bt";
    ROS_INFO("generate_3d_scenes: writing file %s", out_file.c_str());
    if (!scene->writeBinary(out_file))
      ROS_ERROR("generate_3d_scenes: error while writing file!");

    m_scene_counter++;
    if (m_scene_counter <= m_max_scenes)
    {
      m_timer.stop();
      ROS_INFO("generate_3d_scenes: re-scheduling.");
      m_timer.start();
    }
    else
    {
      ROS_INFO("generate_3d_scenes: end.");
      ros::shutdown();
    }
  }

  private:
  std::string m_object_folder_prefix;
  StringVector m_object_folder_list;
  StringVector m_object_suffix_list;

  PointXYZCloudVector m_object_point_clouds;

  std::string m_output_scene_prefix;

  uint64 m_scene_counter;
  uint64 m_max_scenes;

  uint64 m_scene_size;
  uint64 m_scene_height;
  float m_scene_resolution;
  float m_scene_spread;

  uint64 m_objects_per_scene_min;
  uint64 m_objects_per_scene_max;

  ros::Timer m_timer;

  ros::NodeHandle & m_nh;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "generate_3d_scenes_realistic");

  ros::NodeHandle nh("~");

  int random_seed;
  nh.param<int>(PARAM_NAME_RANDOM_SEED, random_seed, PARAM_DEFAULT_RANDOM_SEED);
  srand(random_seed);

  Generate3DScenesRealistic g3s(nh);

  ros::spin();

  return 0;
}
