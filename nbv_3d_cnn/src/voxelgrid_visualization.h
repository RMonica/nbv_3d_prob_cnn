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

#define PARAM_NAME_EMPTY_VOXELGRID_SUFFIX        "empty_voxelgrid_suffix"
#define PARAM_DEFAULT_EMPTY_VOXELGRID_SUFFIX     ""

#define PARAM_NAME_OCTOMAP_NAME       "octomap_name"
#define PARAM_DEFAULT_OCTOMAP_NAME    ""

#define PARAM_NAME_MARKER_OUT_TOPIC    "marker_out_topic"
#define PARAM_DEFAULT_MARKER_OUT_TOPIC "marker"

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
