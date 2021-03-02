#ifndef AUGMENT_TEST_DATASET_H
#define AUGMENT_TEST_DATASET_H

#define PARAM_NAME_SOURCE_IMAGES_PREFIX     "source_images_prefix"
#define PARAM_DEFAULT_SOURCE_IMAGES_PREFIX  ""

#define PARAM_NAME_DEST_IMAGES_PREFIX       "dest_images_prefix"
#define PARAM_DEFAULT_DEST_IMAGES_PREFIX    ""

#define PARAM_NAME_SUBMATRIX_RESOLUTION     "submatrix_resolution"
#define PARAM_DEFAULT_SUBMATRIX_RESOLUTION  (int(4))

#define PARAM_NAME_SENSOR_FOCAL_LENGTH    "sensor_focal_length"
#define PARAM_DEFAULT_SENSOR_FOCAL_LENGTH (float(28))

#define PARAM_NAME_SENSOR_RESOLUTION_X    "sensor_resolution_x"
#define PARAM_DEFAULT_SENSOR_RESOLUTION_X (int(32))

#define PARAM_NAME_SENSOR_RESOLUTION_Y     "sensor_resolution_y"
#define PARAM_DEFAULT_SENSOR_RESOLUTION_Y  (int(24))

#define PARAM_NAME_AUGMENT_ORIENTATIONS    "augment_orientations"
#define PARAM_DEFAULT_AUGMENT_ORIENTATIONS "1 2 4" // x y z

#endif // AUGMENT_TEST_DATASET_H
