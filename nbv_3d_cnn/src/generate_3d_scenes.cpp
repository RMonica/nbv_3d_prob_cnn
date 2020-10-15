#include "generate_3d_scenes.h"

#include <ros/ros.h>

#include <string>
#include <stdint.h>
#include <sstream>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <octomap/octomap.h>
#include <octomap/OcTree.h>

class Generate3DScenes
{
  public:
  typedef std::vector<std::string> StringVector;
  typedef uint64_t uint64;
  typedef std::vector<uint64> Uint64Vector;
  typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vector3fVector;
  typedef std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Vector3f> > Affine3fVector;

  typedef octomap::OcTree OcTree;
  typedef std::shared_ptr<OcTree> OcTreePtr;
  typedef std::shared_ptr<const OcTree> OcTreeConstPtr;

  static double Rand01d() { return double(rand()) / double(RAND_MAX); }

  static float Rand01f() { return float(rand()) / float(RAND_MAX); }
  static float Rand0505f() { return float(rand()) / float(RAND_MAX) - 0.5f; }

  Generate3DScenes(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;
    double param_double;

    m_nh.param<std::string>(PARAM_NAME_OBJECT_FILE_PREFIX, param_string, PARAM_DEFAULT_OBJECT_FILE_PREFIX);
    m_object_file_prefix = param_string;

    m_nh.param<std::string>(PARAM_NAME_OBJECT_FILE_LIST, param_string, PARAM_DEFAULT_OBJECT_FILE_LIST);
    {
      std::istringstream istr(param_string);
      std::string str;
      while (istr >> str)
        m_object_file_list.push_back(str);
    }

    m_nh.param<std::string>(PARAM_NAME_OUTPUT_SCENE_PREFIX, param_string, PARAM_DEFAULT_OUTPUT_SCENE_PREFIX);
    m_output_scene_prefix = param_string;

    m_nh.param<int>(PARAM_NAME_MAX_SCENES, param_int, PARAM_DEFAULT_MAX_SCENES);
    m_max_scenes = param_int;

    m_nh.param<int>(PARAM_NAME_SCENE_SIZE, param_int, PARAM_DEFAULT_SCENE_SIZE);
    m_scene_size = param_int;

    m_scene_resolution = 1.0f / m_scene_size;

    m_nh.param<int>(PARAM_NAME_OBJECTS_PER_SCENE_MIN, param_int, PARAM_DEFAULT_OBJECTS_PER_SCENE_MIN);
    m_objects_per_scene_min = param_int;

    m_nh.param<int>(PARAM_NAME_OBJECTS_PER_SCENE_MAX, param_int, PARAM_DEFAULT_OBJECTS_PER_SCENE_MAX);
    m_objects_per_scene_max = param_int;

    m_nh.param<double>(PARAM_NAME_OBJECT_SCALE, param_double, PARAM_DEFAULT_OBJECT_SCALE);
    m_object_scale = param_double;

    m_scene_counter = 1;
    m_timer = m_nh.createTimer(ros::Duration(0.0), &Generate3DScenes::onTimer, this, true);
  }

  Eigen::Vector3f ComputeOctreeCenter(const OcTreePtr & octree)
  {
    bool first = true;
    Eigen::Vector3f min, max;

    octree->expand();
    for (OcTree::leaf_iterator it = octree->begin_leafs(); it != octree->end_leafs(); it++)
    {
      if (it->getOccupancy() < 0.5f)
        continue; // ignore empty

      const octomap::point3d point = it.getCoordinate();
      const Eigen::Vector3f epoint(point.x(), point.y(), point.z());
      if (first)
      {
        min = max = epoint;
        first = false;
      }
      else
      {
        min = min.array().min(epoint.array());
        max = max.array().max(epoint.array());
      }
    }

    return (max + min) / 2.0f;
  }

  void onTimer(const ros::TimerEvent &)
  {
    ROS_INFO("generate_3d_scenes: processing scene %u.", unsigned(m_scene_counter));

    const float scene_side = m_scene_resolution * m_scene_size;

    const uint64 objects_per_this_scene = m_objects_per_scene_min +
      (rand() % (1 + m_objects_per_scene_max - m_objects_per_scene_min));
    ROS_INFO("generate_3d_scenes: this scene contains %u objects.", unsigned(objects_per_this_scene));

    Uint64Vector objects_in_this_scene;
    for (uint64 i = 0; i < objects_per_this_scene; i++)
      objects_in_this_scene.push_back(rand() % m_object_file_list.size());

    Affine3fVector object_poses;
    for (uint64 i = 0; i < objects_per_this_scene; i++)
    {
      const float angle = M_PI * 2.0f * Rand01d();
      const Eigen::Vector3f axis = Eigen::Vector3f(Rand01f() - 0.5f, Rand01f() - 0.5f, Rand01f() - 0.5f).normalized();
      Eigen::Affine3f pose;
      pose.linear() = Eigen::AngleAxisf(angle, axis).matrix();

      const Eigen::Vector3f translation(scene_side * Rand0505f(), scene_side * Rand0505f(), scene_side * Rand0505f());
      pose.translation() = translation;

      object_poses.push_back(pose);
    }

    OcTreePtr scene(new OcTree(m_scene_resolution));

    for (uint64 obj_i = 0; obj_i < objects_per_this_scene; obj_i++)
    {
      const std::string obj_file = m_object_file_prefix + m_object_file_list[objects_in_this_scene[obj_i]];
      ROS_INFO("generate_3d_scenes: loading object %s", obj_file.c_str());

      const OcTreePtr octree(new OcTree(0.1));
      if (!octree->readBinary(obj_file))
      {
        ROS_ERROR("generate_3d_scenes: file %s could not be loaded!", obj_file.c_str());
        continue;
      }

      const float input_resolution = octree->getResolution();
      const float resolution_scale = m_scene_resolution / input_resolution;

      const Eigen::Vector3f bbox_min = -Eigen::Vector3f::Ones() * scene_side / 2.0f;
      const Eigen::Vector3f bbox_max = Eigen::Vector3f::Ones() * scene_side / 2.0f;

      {
        octomap::point3d bmax(bbox_max.x(), bbox_max.y(), bbox_max.z());
        octomap::point3d bmin(bbox_min.x(), bbox_min.y(), bbox_min.z());
        scene->setBBXMax(bmax);
        scene->setBBXMin(bmin);
        scene->useBBXLimit(true);
      }

      const Eigen::Vector3f octree_center = ComputeOctreeCenter(octree);

      octree->expand();
      for (OcTree::leaf_iterator it = octree->begin_leafs(); it != octree->end_leafs(); it++)
      {
        if (it->getOccupancy() < 0.5f)
          continue; // save only occupied

        const octomap::point3d point = it.getCoordinate();
        const Eigen::Vector3f epoint(point.x(), point.y(), point.z());
        const Eigen::Vector3f new_epoint = object_poses[obj_i] *
            ((epoint - octree_center) * m_object_scale * resolution_scale);

        if ((new_epoint.array() < bbox_min.array()).any())
          continue;
        if ((new_epoint.array() >= bbox_max.array()).any())
          continue;

        const octomap::point3d new_point(new_epoint.x(), new_epoint.y(), new_epoint.z());
        octomap::OcTreeNode * scene_node = scene->search(new_point);
        if (!scene_node)
          scene_node = scene->updateNode(new_point, true);
        scene_node->setLogOdds(octomap::logodds(0.99f));
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
  std::string m_object_file_prefix;
  StringVector m_object_file_list;
  std::string m_output_scene_prefix;

  uint64 m_scene_counter;
  uint64 m_max_scenes;

  uint64 m_scene_size;
  float m_scene_resolution;

  float m_object_scale;

  uint64 m_objects_per_scene_min;
  uint64 m_objects_per_scene_max;

  ros::Timer m_timer;

  ros::NodeHandle & m_nh;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "generate_3d_scenes");

  ros::NodeHandle nh("~");

  int random_seed;
  nh.param<int>(PARAM_NAME_RANDOM_SEED, random_seed, PARAM_DEFAULT_RANDOM_SEED);
  srand(random_seed);

  Generate3DScenes g3s(nh);

  ros::spin();

  return 0;
}
