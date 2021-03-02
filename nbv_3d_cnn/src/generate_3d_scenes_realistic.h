#ifndef GENERATE_3D_SCENES_H
#define GENERATE_3D_SCENES_H

#define PARAM_NAME_RANDOM_SEED    "random_seed"
#define PARAM_DEFAULT_RANDOM_SEED (int(12345))

#define PARAM_NAME_OBJECT_FOLDER_PREFIX   "object_folder_prefix"
#define PARAM_DEFAULT_OBJECT_FOLDER_PREFIX ""

#define PARAM_NAME_OBJECT_FOLDER_LIST     "object_folder_list"
#define PARAM_DEFAULT_OBJECT_FOLDER_LIST  ""

#define PARAM_NAME_OBJECT_SUFFIX_LIST     "object_suffix_list"
#define PARAM_DEFAULT_OBJECT_SUFFIX_LIST  ""

#define PARAM_NAME_OUTPUT_SCENE_PREFIX  "output_scene_prefix"
#define PARAM_DEFAULT_OUTPUT_SCENE_PREFIX ""

#define PARAM_NAME_MAX_SCENES           "max_scenes"
#define PARAM_DEFAULT_MAX_SCENES        (int(10))

#define PARAM_NAME_SCENE_SIZE           "scene_size"
#define PARAM_DEFAULT_SCENE_SIZE        (int(64))

#define PARAM_NAME_SCENE_HEIGHT         "scene_height"
#define PARAM_DEFAULT_SCENE_HEIGHT      (int(64))

#define PARAM_NAME_SCENE_SPREAD         "scene_spread"
#define PARAM_DEFAULT_SCENE_SPREAD      (double(1.0))

#define PARAM_NAME_SCENE_RESOLUTION     "scene_resolution"
#define PARAM_DEFAULT_SCENE_RESOLUTION  (double(0.025))

#define PARAM_NAME_OBJECTS_PER_SCENE_MIN "objects_per_scene_min"
#define PARAM_DEFAULT_OBJECTS_PER_SCENE_MIN (int(5))

#define PARAM_NAME_OBJECTS_PER_SCENE_MAX "objects_per_scene_max"
#define PARAM_DEFAULT_OBJECTS_PER_SCENE_MAX (int(10))

#define PARAM_NAME_OBJECT_SCALE          "object_scale"
#define PARAM_DEFAULT_OBJECT_SCALE       (double(1.0))

#endif // GENERATE_3D_SCENES_H
