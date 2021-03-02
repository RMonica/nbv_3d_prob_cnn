#ifndef SIMULATE_NBV_CYCLE_H
#define SIMULATE_NBV_CYCLE_H

#define PARAM_NAME_IMAGE_FILE_NAME         "image_file_name"
#define PARAM_DEFAULT_IMAGE_FILE_NAME      "test.png"

// INITIALIZE WITH IMAGE {
#define PARAM_NAME_KNOWN_OCCUPIED_FILE_NAME  "known_occupied_file_name"
#define PARAM_DEFAULT_KNOWN_OCCUPIED_FILE_NAME ""

#define PARAM_NAME_KNOWN_EMPTY_FILE_NAME   "known_empty_file_name"
#define PARAM_DEFAULT_KNOWN_EMPTY_FILE_NAME ""
// }

#define PARAM_NAME_SAVE_IMAGES             "save_images"
#define PARAM_DEFAULT_SAVE_IMAGES          (bool(true))

#define PARAM_NAME_DEBUG_OUTPUT_FOLDER     "debug_output_folder"
#define PARAM_DEFAULT_DEBUG_OUTPUT_FOLDER  ""

#define PARAM_NAME_INITIAL_VIEW            "initial_view"
#define PARAM_DEFAULT_INITIAL_VIEW         "" // empty for random

#define PARAM_NAME_PREDICT_ACTION_NAME     "predict_action_name"
#define PARAM_DEFAULT_PREDICT_ACTION_NAME  "predict"

#define PARAM_NAME_PREDICT_FLAT_ACTION_NAME "predict_flat_action_name"
#define PARAM_DEFAULT_PREDICT_FLAT_ACTION_NAME "predict"

#define PARAM_NAME_PREDICT_QUAT_ACTION_NAME    "predict_quat_action_name"
#define PARAM_DEFAULT_PREDICT_QUAT_ACTION_NAME "predict"

#define PARAM_NAME_PREDICT_AUTOCOMPLETE_ACTION_NAME    "predict_autocomplete_action_name"
#define PARAM_DEFAULT_PREDICT_AUTOCOMPLETE_ACTION_NAME "predict"

#define PARAM_NAME_AUTOCOMPLETE_IGAIN_RESCALE_MARGIN    "autocomplete_igain_rescale_margin"
#define PARAM_DEFAULT_AUTOCOMPLETE_IGAIN_RESCALE_MARGIN (double(0.1))

#define PARAM_NAME_IGAIN_MIN_RANGE                      "igain_min_range"
#define PARAM_DEFAULT_IGAIN_MIN_RANGE                   (double(10.0))

#define PARAM_NAME_ACCURACY_SKIP_VOXELS                 "accuracy_skip_voxels"
#define PARAM_DEFAULT_ACCURACY_SKIP_VOXELS              (int(1)) // 1: no skip, >1 faster, but decrease accuracy

#define PARAM_NAME_LOG_FILE_NAME           "log_file"
#define PARAM_DEFAULT_LOG_FILE_NAME        ""

#define PARAM_NAME_MAX_ITERATIONS          "max_iterations"
#define PARAM_DEFAULT_MAX_ITERATIONS       (int(100))

#define PARAM_NAME_SAMPLE_FIXED_NUMBER_OF_VIEWS    "sample_fixed_number_of_views" // for AutocompleteFixedNumberIGain
#define PARAM_DEFAULT_SAMPLE_FIXED_NUMBER_OF_VIEWS (int(100))

#define PARAM_NAME_NBV_ALGORITHM           "nbv_algorithm"
#define PARAM_VALUE_NBV_ALGORITHM_RANDOM   "Random"
#define PARAM_VALUE_NBV_ALGORITHM_CNNDirectional "CNNDirectional"
#define PARAM_VALUE_NBV_ALGORITHM_CNNDirectDirectional "CNNDirectDirectional"
#define PARAM_VALUE_NBV_ALGORITHM_CNNFlat "CNNFlat"
#define PARAM_VALUE_NBV_ALGORITHM_CNNQuat "CNNQuat"
#define PARAM_VALUE_NBV_ALGORITHM_InformationGain "InformationGain"
#define PARAM_VALUE_NBV_ALGORITHM_InformationGainProb "InformationGainProb"
#define PARAM_VALUE_NBV_ALGORITHM_AutocompleteIGain "AutocompleteIGain"
#define PARAM_VALUE_NBV_ALGORITHM_AutocompleteFixedNumberIGain "AutocompleteFixedNumberIGain"
#define PARAM_VALUE_NBV_ALGORITHM_OmniscientGain "OmniscientGain"
#define PARAM_DEFAULT_NBV_ALGORITHM        PARAM_VALUE_NBV_ALGORITHM_RANDOM

#endif // SIMULATE_NBV_CYCLE_H
