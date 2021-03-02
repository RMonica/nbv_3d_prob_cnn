#include "merge_logs.h"

#include <ros/ros.h>

#include <string>
#include <vector>
#include <sstream>
#include <stdint.h>
#include <fstream>

class MergeLogs
{
  public:
  typedef uint64_t uint64;

  typedef std::vector<std::string> StringVector;
  typedef std::vector<double> DoubleVector;
  typedef std::vector<uint64> Uint64Vector;

  MergeLogs(ros::NodeHandle & nh): m_nh(nh)
  {
    std::string param_string;
    int param_int;

    m_nh.param<std::string>(PARAM_NAME_LOG_FILES, param_string, PARAM_DEFAULT_LOG_FILES);
    {
      std::istringstream istr(param_string);
      std::string f;
      while (istr >> f)
      {
        m_files.push_back(f);
        ROS_INFO("merge_logs: log file %s", f.c_str());
      }
    }

    m_nh.param<std::string>(PARAM_NAME_OUTPUT_FILE, m_output_file, PARAM_DEFAULT_OUTPUT_FILE);

    m_nh.param<int>(PARAM_NAME_FORCE_COUNT, param_int, PARAM_DEFAULT_FORCE_COUNT);
    m_force_count = param_int;

    m_nh.param<int>(PARAM_NAME_SKIP_FIRST_LINES, param_int, PARAM_DEFAULT_SKIP_FIRST_LINES);
    m_skip_first_lines = param_int;

    m_nh.param<int>(PARAM_NAME_TIME_SKIP_FIRST_LINES, param_int, PARAM_DEFAULT_TIME_SKIP_FIRST_LINES);
    m_time_skip_first_lines = param_int;

    DoMerge(m_files, m_output_file);
  }

  void DoMerge(const StringVector & input_files, const std::string & output_file)
  {
    Uint64Vector counters;
    DoubleVector unknown_sum;
    DoubleVector time_sum;

    for (const std::string & s : input_files)
    {
      ROS_INFO("merge_logs: reading file %s", s.c_str());

      std::ifstream ifile(s);
      if (!ifile)
      {
        ROS_ERROR("merge_logs: couldn't find file %s", s.c_str());
        continue;
      }

      uint64 last_current_unknown, last_total_unknown;
      double last_computation_time;

      std::string line;
      uint64 first_line = m_skip_first_lines;
      uint64 time_first_line = m_time_skip_first_lines;
      uint64 skipped_time_first_lines = 0;
      uint64 counter = 0;
      while (std::getline(ifile, line))
      {
        if (first_line)
        {
          first_line--;
          continue;
        }

        std::istringstream ssline(line);
        uint64 iteration, current_unknown, total_unknown;
        double computation_time;
        ssline >> iteration >> current_unknown >> total_unknown >> computation_time;

        if (!ssline)
        {
          ROS_ERROR("merge_logs: invalid line %s", line.c_str());
          continue;
        }

        if (unknown_sum.size() <= counter)
        {
          unknown_sum.resize(counter + 1, 0.0);
          time_sum.resize(counter + 1, 0.0);
          counters.resize(counter + 1, 0);
        }

        unknown_sum[counter] += double(current_unknown) / double(total_unknown);

        if (time_first_line)
        {
          time_first_line--;
          skipped_time_first_lines++;
        }

        if (!time_first_line && skipped_time_first_lines)
        {
          for (uint64 i = 0; i < skipped_time_first_lines - 1; i++)
            time_sum[i] += computation_time;
          skipped_time_first_lines = 0;
        }

        if (!time_first_line)
          time_sum[counter] += computation_time;

        counters[counter]++;

        last_computation_time = computation_time;
        last_current_unknown = current_unknown;
        last_total_unknown = total_unknown;

        counter++;
        if (m_force_count > 0 && counter >= m_force_count)
          break;
      }

      ROS_INFO("merge_logs: processed %u lines.", unsigned(counter));

      while (counter < m_force_count)
      {
        ROS_INFO("merge_logs: padding line %u.", unsigned(counter));

        if (unknown_sum.size() <= counter)
        {
          unknown_sum.resize(counter + 1, 0.0);
          time_sum.resize(counter + 1, 0.0);
          counters.resize(counter + 1, 0);
        }

        unknown_sum[counter] += double(last_current_unknown) / double(last_total_unknown);
        time_sum[counter] += last_computation_time;
        counters[counter]++;

        counter++;
      }
    }

    std::ofstream ofile(output_file);
    ROS_INFO("merge_logs: saving file %s", output_file.c_str());
    ofile << "Iteration\t\"Current unknown\"\t\"Total unknown\"\t\"Computation time\"\n";

    const uint64 size = unknown_sum.size();
    for (uint64 i = 0; i < size; i++)
    {
      const uint64 iteration = i;
      const uint64 total_unknown = 1562500;
      const uint64 current_unknown = std::round(total_unknown * unknown_sum[i] / counters[i]);
      const double computation_time = time_sum[i] / counters[i];

      ofile << iteration << "\t" << current_unknown << "\t" << total_unknown << "\t" << computation_time << "\n";
    }

    if (!ofile)
      ROS_ERROR("merge_logs: could not save file %s", output_file.c_str());

    ros::shutdown();
  }

  private:
  ros::NodeHandle & m_nh;

  StringVector m_files;
  std::string m_output_file;
  uint64 m_force_count;
  uint64 m_skip_first_lines;
  uint64 m_time_skip_first_lines;
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "merge_logs");

  ros::NodeHandle nh("~");

  MergeLogs ml(nh);

  ros::spin();

  return 0;
}
