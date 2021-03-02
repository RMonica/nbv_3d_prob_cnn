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
#include <iostream>

// custom
#include <nbv_3d_cnn/voxelgrid.h>

typedef uint64_t uint64;

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "voxelgrid_to_binvoxelgrid");

  if (argc < 3)
  {
    std::cout << "usage: voxelgrid_to_binvoxelgrid [src.voxelgrid dst.binvoxelgrid]+" << std::endl;
    return 0;
  }

  uint64 count = (argc - 1) / 2;

  for (uint64 i = 0; i < count; i++)
  {
    const std::string load_filename = argv[1 + i * 2];
    const std::string save_filename = argv[1 + i * 2 + 1];
    std::cout << "voxelgrid_to_binvoxelgrid: loading " << i << " file " << load_filename << std::endl;
    Voxelgrid::Ptr vx = Voxelgrid::FromFile(load_filename);
    if (!vx)
    {
      std::cout << "voxelgrid_to_binvoxelgrid: could not load file." << std::endl;
      return 1;
    }

    std::cout << "voxelgrid_to_binvoxelgrid: saving file " << save_filename << std::endl;
    vx->ToFileBinary(save_filename);
  }

  std::cout << "voxelgrid_to_binvoxelgrid: done." << std::endl;

  return 0;
}
