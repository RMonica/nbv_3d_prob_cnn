#ifndef VOXELGRID_VISUALIZATION_H
#define VOXELGRID_VISUALIZATION_H

#define PARAM_NAME_VOXELGRID_NAME_PREFIX         "voxelgrid_name_prefix"
#define PARAM_DEFAULT_VOXELGRID_NAME_PREFIX      ""

#define PARAM_NAME_OCCUPIED_VOXELGRID_NAME       "occupied_voxelgrid_name"
#define PARAM_DEFAULT_OCCUPIED_VOXELGRID_NAME    ""

#define PARAM_NAME_OCCUPIED_VOXELGRID_SUFFIX     "occupied_voxelgrid_suffix"
#define PARAM_DEFAULT_OCCUPIED_VOXELGRID_SUFFIX  ""

#define PARAM_NAME_EMPTY_VOXELGRID_NAME          "empty_voxelgrid_name"
#define PARAM_DEFAULT_EMPTY_VOXELGRID_NAME       ""

#define PARAM_NAME_USE_SEQUENCE_COUNTER          "use_sequence_counter"
#define PARAM_DEFAULT_USE_SEQUENCE_COUNTER       (bool(true))

#define PARAM_NAME_VOXELGRID_SIZE                "voxelgrid_size"
#define PARAM_DEFAULT_VOXELGRID_SIZE             "" // x y z (voxels)

#define PARAM_NAME_OCCUPANCY_TH                  "occupancy_th"
#define PARAM_DEFAULT_OCCUPANCY_TH               (float(0.5))

#define PARAM_NAME_EMPTY_VOXELGRID_SUFFIX        "empty_voxelgrid_suffix"
#define PARAM_DEFAULT_EMPTY_VOXELGRID_SUFFIX     ""

#define PARAM_NAME_OCTOMAP_NAME       "octomap_name"
#define PARAM_DEFAULT_OCTOMAP_NAME    ""

#define PARAM_NAME_MARKER_OUT_TOPIC    "marker_out_topic"
#define PARAM_DEFAULT_MARKER_OUT_TOPIC "marker"

#define PARAM_NAME_CLOUD_OUT_TOPIC    "cloud_out_topic"
#define PARAM_DEFAULT_CLOUD_OUT_TOPIC "cloud"

#define PARAM_NAME_USE_RAINBOW        "use_rainbow"
#define PARAM_DEFAULT_USE_RAINBOW     (bool(false))

#define PARAM_NAME_SEQUENCE_MODE      "sequence_mode"
#define PARAM_VALUE_SEQUENCE_ONE_SHOT "one_shot"
#define PARAM_VALUE_SEQUENCE_VIDEO    "video"
#define PARAM_DEFAULT_SEQUENCE_MODE   PARAM_VALUE_SEQUENCE_ONE_SHOT

#define PARAM_NAME_INITIAL_DELAY      "initial_delay"
#define PARAM_DEFAULT_INITIAL_DELAY   (double(2.0))

#define PARAM_NAME_FRAME_DELAY        "frame_delay"
#define PARAM_DEFAULT_FRAME_DELAY     (double(1.0))

#define PARAM_NAME_NAMESPACE          "namespace"
#define PARAM_DEFAULT_NAMESPACE       ""

#define PARAM_NAME_COLORS             "colors"
#define PARAM_DEFAULT_COLORS          "0.8 0 0"

#endif // VOXELGRID_VISUALIZATION_H
