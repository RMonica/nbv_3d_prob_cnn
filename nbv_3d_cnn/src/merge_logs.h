#ifndef MERGE_LOGS_H
#define MERGE_LOGS_H

#define PARAM_NAME_LOG_FILES      "log_files"
#define PARAM_DEFAULT_LOG_FILES   "file1 file2" // list of files

#define PARAM_NAME_OUTPUT_FILE    "output_file"
#define PARAM_DEFAULT_OUTPUT_FILE "file3"

#define PARAM_NAME_FORCE_COUNT    "force_count"
#define PARAM_DEFAULT_FORCE_COUNT (int(0)) // if 0, all will be processed, otherwise only the first force_count
                                           // if less than force_count, last entry will be replicated

#define PARAM_NAME_SKIP_FIRST_LINES "skip_first_lines"
#define PARAM_DEFAULT_SKIP_FIRST_LINES (int(1))

#define PARAM_NAME_TIME_SKIP_FIRST_LINES "time_skip_first_lines"
#define PARAM_DEFAULT_TIME_SKIP_FIRST_LINES (int(0))

#endif // MERGE_LOGS_H
