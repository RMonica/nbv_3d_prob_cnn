#include <ros/ros.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/StdVector>

// STL
#include <string>
#include <stdint.h>
#include <vector>
#include <cmath>
#include <sstream>

// custom
#include <nbv_3d_cnn/voxelgrid.h>

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "voxelgrid_to_octomap");

  if (argc < 3)
  {
    ROS_INFO("usage: voxelgrid_to_octomap src.voxelgrid dst.bt");
    return 0;
  }

  ROS_INFO("voxelgrid_to_octomap: loading file %s", argv[1]);
  Voxelgrid::Ptr vx = Voxelgrid::FromFile(argv[1]);
  if (!vx)
  {
    ROS_FATAL("voxelgrid_to_octomap: could not load file.");
    return 1;
  }

  ROS_INFO("voxelgrid_to_octomap: saving file %s", argv[2]);
  vx->SaveOctomapOctree(argv[2]);

  return 0;
}
