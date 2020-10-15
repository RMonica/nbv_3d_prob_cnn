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

      std::string line;
      bool first_line = true;
      uint64 counter = 0;
      while (std::getline(ifile, line))
      {
        if (first_line)
        {
          first_line = false;
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
        time_sum[counter] += computation_time;
        counters[counter]++;

        counter++;
      }

      ROS_INFO("merge_logs: processed %u lines.", unsigned(counter));
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
};

int main(int argc, char ** argv)
{
  ros::init(argc, argv, "merge_logs");

  ros::NodeHandle nh("~");

  MergeLogs ml(nh);

  ros::spin();

  return 0;
}
