#include <stdio.h>
#include <array>
#include <execution>
#include <iostream>
#include <vector>
#include "include/avx.hpp"
#include "include/constants.hpp"
#include "include/data_preprocessing.hpp"
#include "include/errors.hpp"
#include "include/gpu.hpp"
#include "include/logger.hpp"
#include "include/svg.hpp"
#include "include/warnings.hpp"

constexpr size_t NO_SUBJECTS = 16;
constexpr size_t FILE_NAME_PADDING = 3;
const std::string OUT_FOLDER_PATH = "out";

Logging::Logger& logger = Logging::Logger::get_instance();

/**
 * Vector of pairs of strings of source files of the patients. 
 * Contains only valid ones (only those for which both ACC and HR resource files exist)
 * (Position + 1) corresponds the patient's number
 * First item of the pair corresponds to the ACC source file path
 * Second item of the pair corresponds to the HR source file path
 */
std::vector<std::pair<std::string, std::string>> valid_subject_ids{};

/**
 * Validate all needed resources
 *
 * @return RETURN_OK if at least some resources found, RETURN_ERR 
 */
int8_t validate_resources() {
  //Validate that the resource folder exists
  if (!std::filesystem::exists(RESOURCE_FOLDER_PATH)) {
    logger.log_message(
        Logging::LOG_LEVEL::ERROR,
        errors::ERRORS_MAP.at(errors::ERRORS::RESOURCE_FOLDER_NOT_FOUND) +
            RESOURCE_FOLDER_PATH);
    return errors::ERRORS::RESOURCE_FOLDER_NOT_FOUND;
  }

  std::string acc_file_path{}, hr_file_path{}, curr_file_number{};

  constexpr size_t SOURCE_FILE_NAME_LEN =
      32;  //Path len inside the resources folder
  acc_file_path.reserve(SOURCE_FILE_NAME_LEN);
  hr_file_path.reserve(SOURCE_FILE_NAME_LEN);

  //Validate all necessary
  for (size_t i = 0; i < NO_SUBJECTS; ++i) {
    curr_file_number = std::to_string((i + 1));

    //Note: This could be implemented using std::format for compilers supporting C++20
    curr_file_number.insert(curr_file_number.begin(),
                            FILE_NAME_PADDING - curr_file_number.size(),
                            '0');  //File number padding

    acc_file_path.append(RESOURCE_FOLDER_PATH)
        .append(FILE_PATH_SEPARATOR)
        .append(curr_file_number)
        .append(FILE_PATH_SEPARATOR)
        .append("ACC_")
        .append(curr_file_number)
        .append(SOURCE_FILE_FORMAT);

    hr_file_path.append(RESOURCE_FOLDER_PATH)
        .append(FILE_PATH_SEPARATOR)
        .append(curr_file_number)
        .append(FILE_PATH_SEPARATOR)
        .append("HR_")
        .append(curr_file_number)
        .append(SOURCE_FILE_FORMAT);

    if (!std::filesystem::exists(acc_file_path) ||
        !std::filesystem::exists(hr_file_path)) {
      logger.log_warning(warnings::FILE_NOT_FOUND);
      logger.log_message(Logging::LOG_LEVEL::WARNING,
                         "Subject " + curr_file_number +
                             " will not be processed due to missing file(s)");
      continue;
    }

    if (std::filesystem::is_empty(acc_file_path) ||
        std::filesystem::is_empty(hr_file_path)) {
      logger.log_warning(
          warnings::WARNINGS::FILE_IS_EMPTY,
          "Subject's " + curr_file_number + " file(s) are empty");
      logger.log_message(Logging::LOG_LEVEL::WARNING,
                         "Subject " + curr_file_number +
                             " will not be processed due to missing file(s)");
      continue;
    }

    valid_subject_ids.insert(valid_subject_ids.end(),
                             std::pair(acc_file_path, hr_file_path));
    logger.log_info("Subject's " + curr_file_number + " resource files found");

    acc_file_path.clear();
    hr_file_path.clear();
  }

  return RETURN_OK;
}

int main(int argc, char* argv[]) {
  std::cout << "\n-------------------------" << std::endl;
  std::cout << "Welcome to the PPR Correlation Finder" << std::endl;
  std::cout << "Copyright @2023 Jan Heres" << std::endl;
  std::cout << "-------------------------\n" << std::endl;

  Logging::APP_LOGGING_LEVEL = Logging::LOG_LEVEL::INFO;

  uint8_t period_size = 1;
  if (argc > 1) {
    try {
      period_size = std::stoi(argv[1]);
      if (period_size > MAX_SUPPORTED_PERIOD_SIZE || period_size < 1) {
        logger.log_warning(warnings::WARNINGS::INVALID_PERIOD_SIZE,
                           "Falling back to the default period size (=1)");
        period_size = 1;
      }
    } catch (const std::invalid_argument& err) {
      logger.log_warning(warnings::WARNINGS::COULD_NOT_PARSE_CMD_ARGS,
                         "(" + (std::string)argv[1] +
                             "). The period size will fall back to \"1\"");
      period_size = 1;
    }
  }

  logger.log_info("Period size: " + std::to_string(period_size));

  logger.log_info("Validating resource files...");
  int8_t rv = validate_resources();
  if (rv != RETURN_OK) {
    exit(rv);
  }
  logger.log_info("All resource files have been found");

  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);

  std::vector<cl::Device> devices;
  for (cl::Platform platform : platforms) {
    platform.getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
  }

  if (devices.size() < 1) {
    logger.log_error(errors::ERRORS::OPENCL_NO_DEVICE_FOUND);
    exit(errors::ERRORS::OPENCL_NO_DEVICE_FOUND);
  }

  size_t user_dev_id = 0;
  if (devices.size() > 1) {
    while (true) {
      std::cout << "Available OpenCL devices: " << std::endl;
      for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << "[" << i << "]: " << devices[i].getInfo<CL_DEVICE_NAME>()
                  << std::endl;
      }

      std::cout << "Choose a device: ";

      std::string user_input;
      std::cin >> user_input;

      std::size_t tmp = std::stoi(user_input);

      if (tmp > 0 && tmp < devices.size()) {
        user_dev_id = tmp;
        break;
      }

      std::cout << "Invalid choice. Please, try again." << std::endl
                << std::endl;
    }
  }

  cl::Device opencl_device = cl::Device{devices[user_dev_id]};
  opencl::Gpu gpu = opencl::Gpu(opencl_device);

  logger.log_info("Chosen OpenCL device: " +
                  opencl_device.getInfo<CL_DEVICE_NAME>());

  logger.log_info("Beginning data preprocessing...");

  for (size_t i = 0; i < valid_subject_ids.size(); ++i) {

    DataPreprocessing::SubjectDataProcessor data_processor(
        valid_subject_ids[i].first, valid_subject_ids[i].second);
    const size_t NO_VALUES_ACC = 3;

    std::array<std::vector<float_t>, NO_VALUES_ACC> acc_values;
    std::vector<float_t> hr_values;
    std::pair<uint8_t, long long> timestamp_diff;
    long long time_diff;

    timestamp_diff = data_processor.validate_timestamps(period_size);
    if (timestamp_diff.first == RETURN_OK) {
      time_diff = timestamp_diff.second;
      logger.log_info("Timestamp difference calculated: " +
                      std::to_string(time_diff));
    } else {
      logger.log_warning(warnings::WARNINGS::TIMESTAMP_CALCULATION_WARNING);
    }

    time_diff = timestamp_diff.first == RETURN_OK ? timestamp_diff.second : 0;
    logger.log_debug("Timestamp diff: " + std::to_string(time_diff));

    std::int8_t tmp_sign = time_diff > 0 ? 1 : -1;

    std::optional tmp_acc = data_processor.preprocess_acc_file(
        period_size, time_diff > 0 ? time_diff * tmp_sign : 0);

    std::optional tmp_hr = data_processor.preprocess_hr_file(
        period_size, time_diff < 0 ? time_diff * tmp_sign : 0);

    if (tmp_acc == std::nullopt || tmp_hr == std::nullopt) {
      logger.log_error(errors::ERRORS::COULD_NOT_PREPROCESS_VALUES);
      return EXIT_FAILURE;
    }

    hr_values = tmp_hr.value();
    acc_values = tmp_acc.value();

    // Linear interpolation + AVX2 proper padding
    {
      int64_t len_diff = acc_values[0].size() - hr_values.size();
      size_t padding = 0;
      if (len_diff < 0) {
        len_diff = std::abs(len_diff);

        for (size_t j = 0; j < acc_values.size(); ++j) {
          padding = data_processor.interpolate_vector_linear(
              acc_values[j], len_diff + padding);
        }

        data_processor.interpolate_vector_linear(hr_values, padding);
      } else if (len_diff > 0) {
        padding = data_processor.interpolate_vector_linear(hr_values, len_diff);

        for (size_t j = 0; j < acc_values.size(); ++j) {
          data_processor.interpolate_vector_linear(acc_values[j], padding);
        }
      }
    }

    const std::optional<float_t> tmp = avx::vector_sum_avx2(hr_values);

    if (tmp == std::nullopt) {
      logger.log_error(errors::ERRORS::COULD_NOT_PREPROCESS_VALUES);
      return EXIT_FAILURE;
    }

    const float_t hr_avg = tmp.value() / hr_values.size();

    // Precalculate HR value statistics needed for the correlation calculation
    // These need to be calculated just once, the will not change during the following computations
    std::vector<float_t> hr_values_diffs = hr_values;
    std::float_t hr_values_squared_diffs = 0.0;
    for (size_t j = 0; j < hr_values_diffs.size(); ++j) {
      hr_values_diffs[j] -= hr_avg;
      hr_values_squared_diffs += hr_values_diffs[j] * hr_values_diffs[j];
    }

    const float_t hr_values_squared_root = sqrtf(hr_values_squared_diffs);

    float_t initial_correlation = 0.0f;
    std::vector<float_t> result(0.0, acc_values.size());
    cl::Buffer acc_buffer, hr_buffer, result_buffer;
    cl::Context current_context;

    for (size_t j = 0; j < NO_VALUES_ACC; ++j) {
      std::vector<float_t> curr_acc_values = acc_values[j];
      // As an example, calculate the initial correlation on CPU using AVX2 registers, since we need to calculate it just once
      std::optional tmp = avx::calculate_pearsons_correlation(
          curr_acc_values, hr_values_diffs, hr_values_squared_root);
      if (tmp != std::nullopt) {
        initial_correlation = tmp.value();
        logger.log_info("Initial correlation (axis " + std::to_string(j) +
                        ") is " + std::to_string(initial_correlation));
      }

      auto desc = opencl_device.getInfo<CL_DEVICE_NAME>();
      logger.log_info("Starting correlation formula generation on device: " +
                      desc);

      std::pair<std::vector<float_t>, std::vector<float_t>> best_fit =
          gpu.compute_correlation_formula(curr_acc_values, hr_values_diffs,
                                          hr_values_squared_root);

      std::string tree_string;
      tree_string.reserve(GENERATION_INDIVIDUAL_SIZE *
                          20);  // 20 chars per node should be enough

      // Convert the syntax tree to a string
      for (size_t k = 0; k < GENERATION_INDIVIDUAL_SIZE;
           k += GENERATION_TREE_NODE_SIZE) {
        std::string op = "?";

        tree_string.append(" + (");

        float_t curr = best_fit.second[k + 1];
        if (curr == X_FLOAT_REPRESENTATION) {
          tree_string.append("x ");
        } else {
          tree_string.append(std::to_string(curr) + " ");
        }

        curr = best_fit.second[k];
        if (curr == ADD_FLOAT_REPRESENTATION) {
          op = "+";
        } else if (curr == SUB_FLOAT_REPRESENTATION) {
          op = "-";
        } else if (curr == MUL_FLOAT_REPRESENTATION) {
          op = "*";
        } else if (curr == DIV_FLOAT_REPRESENTATION) {
          op = "/";
        }

        tree_string.append(op + " " + std::to_string(best_fit.second[k + 2]) +
                           ")");
      }

      logger.log_info("Tree corresponding to the calculated correlation: " +
                      tree_string);

      // Plot the values
      std::string axis = "";
      switch (j) {
        case 0:
          axis = "X";
          break;
        case 1:
          axis = "Y";
          break;
        case 2:
          axis = "Z";
          break;
        default:
          axis = "unknown";
      }
      std::string filename = OUT_FOLDER_PATH + "/patient_" +
                             std::to_string(i + 1) + "_axis_" + axis + "_" +
                             opencl_device.getInfo<CL_DEVICE_NAME>() + ".svg";
      logger.log_info("Exporting results into " + filename);
      svg::plot_correlation_values(filename, best_fit.first, hr_values,
                                   tree_string);
      logger.log_info("Results exported");
    }
  }
}
