#ifndef GENERATE_TEST_DATASET_H
#define GENERATE_TEST_DATASET_H

#define PARAM_NAME_SOURCE_IMAGES_PREFIX     "source_images_prefix"
#define PARAM_DEFAULT_SOURCE_IMAGES_PREFIX  ""

#define PARAM_NAME_DEST_IMAGES_PREFIX     "dest_images_prefix"
#define PARAM_DEFAULT_DEST_IMAGES_PREFIX  ""

#define PARAM_NAME_SOURCE_IMAGES_SUFFIX    "source_image_suffix"
#define PARAM_DEFAULT_SOURCE_IMAGES_SUFFIX ".tif"

#define PARAM_NAME_ENVIRONMENT_RESIZE     "environment_resize"
#define PARAM_DEFAULT_ENVIRONMENT_RESIZE  "1 1" // format: div_x div_y

#define PARAM_NAME_SENSOR_RANGE_VOXELS    "sensor_range_voxels"
#define PARAM_DEFAULT_SENSOR_RANGE_VOXELS (double(150))

#define PARAM_NAME_RANDOM_SEED            "random_seed"
#define PARAM_DEFAULT_RANDOM_SEED         (int(0))

#define PARAM_NAME_SENSOR_FOCAL_LENGTH    "sensor_focal_length"
#define PARAM_DEFAULT_SENSOR_FOCAL_LENGTH (float(256.0))

#define PARAM_NAME_SENSOR_RESOLUTION_X    "sensor_resolution_x"
#define PARAM_DEFAULT_SENSOR_RESOLUTION_X (int(512))

#define PARAM_NAME_SENSOR_RESOLUTION_Y     "sensor_resolution_y"
#define PARAM_DEFAULT_SENSOR_RESOLUTION_Y  (int(1))

#define PARAM_NAME_VIEW_CUBE_RESOLUTION   "view_cube_resolution"
#define PARAM_DEFAULT_VIEW_CUBE_RESOLUTION (int(4))

#define PARAM_NAME_SUBMATRIX_RESOLUTION    "submatrix_resolution"
#define PARAM_DEFAULT_SUBMATRIX_RESOLUTION (int(4))

#define PARAM_NAME_NUM_VIEW_POSES_MIN     "num_view_poses_min"
#define PARAM_DEFAULT_NUM_VIEW_POSES_MIN  (int(6))

#define PARAM_NAME_NUM_VIEW_POSES_MAX     "num_view_poses_max"
#define PARAM_DEFAULT_NUM_VIEW_POSES_MAX  (int(10))

#define PARAM_NAME_A_PRIORI_OCCUPIED_PROB "a_priori_occupied_prob"
#define PARAM_DEFAULT_A_PRIORI_OCCUPIED_PROB (double(0.1))

#define PARAM_NAME_PREFIX_LIST            "prefix_list"
#define PARAM_DEFAULT_PREFIX_LIST         "austin"

#define PARAM_NAME_3D_MODE                "mode_3d"
#define PARAM_VALUE_3D_MODE_2D            "2d"
#define PARAM_VALUE_3D_MODE_3D            "3d"
#define PARAM_DEFAULT_3D_MODE             PARAM_VALUE_3D_MODE_2D

// ** OpenCL **
#define PARAM_NAME_OPENCL_PLATFORM_NAME        "opencl_platform_name"
#define PARAM_DEFAULT_OPENCL_PLATFORM_NAME     ""           // a part of the name is enough, empty = default

#define PARAM_NAME_OPENCL_DEVICE_NAME          "opencl_device_name"
#define PARAM_DEFAULT_OPENCL_DEVICE_NAME       ""           // a part of the name is enough, empty = default

#define PARAM_NAME_OPENCL_DEVICE_TYPE          "opencl_device_type"
#define PARAM_VALUE_OPENCL_DEVICE_TYPE_GPU     "GPU"
#define PARAM_VALUE_OPENCL_DEVICE_TYPE_CPU     "CPU"
#define PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL     "ALL"
#define PARAM_DEFAULT_OPENCL_DEVICE_TYPE       PARAM_VALUE_OPENCL_DEVICE_TYPE_ALL

#endif // GENERATE_TEST_DATASET_H
