#include "include/data_preprocessing.hpp"
#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <execution>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include "include/constants.hpp"
#include "include/errors.hpp"
#include "include/logger.hpp"

namespace DataPreprocessing {

Logging::Logger& logger = Logging::Logger::get_instance();

SubjectDataProcessor::SubjectDataProcessor(const std::string& _acc_file_path,
                                           const std::string& _hr_file_path) {
  if (_acc_file_path.empty() || _hr_file_path.empty()) {
    throw std::invalid_argument("File path argument must not be empty");
  }

  this->_acc_file_stream.open(_acc_file_path, std::ios::in);
  this->_hr_file_stream.open(_hr_file_path, std::ios::in);
  if (!this->_acc_file_stream.is_open() || !this->_hr_file_stream.is_open()) {
    throw new std::ifstream::failure(
        "Either ACC file or HR file could not have been opened.");
  }

  this->_acc_file_path = _acc_file_path;
  this->_hr_file_path = _hr_file_path;
}

SubjectDataProcessor::~SubjectDataProcessor() {
  if (this->_acc_file_stream.is_open()) {
    this->_acc_file_stream.close();
  }

  if (this->_hr_file_stream.is_open()) {
    this->_hr_file_stream.close();
  }
}

// PRIVATE METHODS //

const std::optional<std::array<float_t, ACC_NO_VALUES>>
SubjectDataProcessor::parse_acc_value(
    std::string& value_string) const noexcept {
  std::array<float_t, ACC_NO_VALUES> rv = {0, 0, 0};  // X, Y, Z

  if (value_string.empty()) {
    return std::nullopt;
  }

  u_long pos;
  std::string tmp;

  // Parse X and Y
  for (size_t i = 0; i < ACC_NO_VALUES - 1; ++i) {
    pos = value_string.find(DATA_DELIMITER);

    if (pos == std::string::npos) {
      logger.log_error(
          errors::ERRORS::INVALID_FILE_STRUCTURE,
          "Accelerometer values are in a wrong format. Valid format: acc_x" +
              std::to_string(DATA_DELIMITER) + "acc_y" +
              std::to_string(DATA_DELIMITER) + "acc_z");
      return std::nullopt;
    }

    tmp = value_string.substr(0, pos);

    try {
      rv[i] = std::stoi(tmp);
    } catch (std::invalid_argument& e) {
      logger.log_error(errors::ERRORS::COULD_NOT_PARSE_VALUE,
                       "(Value: " + tmp + ")");
      return std::nullopt;
    };
    value_string.erase(0, pos + 1);  // +1 to delete the delimiter as well
  }

  // Now, in the copy the Z value remains
  try {
    rv[ACC_NO_VALUES - 1] = std::stoi(value_string);
  } catch (std::invalid_argument& e) {
    logger.log_error(errors::ERRORS::COULD_NOT_PARSE_VALUE,
                     "(Value: " + tmp + ")");
    return std::nullopt;
  };

  return rv;
}

const std::optional<std::array<std::vector<float_t>, ACC_NO_VALUES>>
SubjectDataProcessor::parse_acc_file(const uint8_t period_size,
                                     const u_long timestamp_diff) noexcept {
  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "(Provided value: " + std::to_string(period_size) + ")");
    return std::nullopt;
  }
  const uint8_t MAX_LINE_LEN = 64;
  const size_t LOGGING_THRESHOLD = 1000000;

  std::vector<float_t> values_x, values_y, values_z;

  std::string curr_line;
  curr_line.reserve(MAX_LINE_LEN);

  u_long pos, lines = 0;
  std::array<float_t, ACC_NO_VALUES> curr_vals = {0, 0, 0};  // X, Y, Z
  std::optional<std::array<float_t, ACC_NO_VALUES>> parsed;

  logger.log_info("Beginning parsing file " + _acc_file_path);

  _acc_file_stream.seekg(_acc_file_stream.beg);
  std::getline(_acc_file_stream, curr_line);  //Skip the first line

  // Sync up with HR measurements
  u_long diff = (long)((timestamp_diff * HR_SAMPLE_FREQ) / period_size);
  if (diff > 0) {
    logger.log_info("ACC measurements are \"ahead\" by " +
                    std::to_string(diff) + " lines. Skipping...");
  }

  for (u_long i = 0; i < diff; ++i) {
    std::getline(_acc_file_stream, curr_line);
  }

  while (std::getline(_acc_file_stream, curr_line)) {
    pos = curr_line.find(DATA_DELIMITER);
    if (pos == std::string::npos) {
      logger.log_error(errors::ERRORS::INVALID_FILE_STRUCTURE,
                       "ACC files must have the following structure: datetime" +
                           std::to_string(DATA_DELIMITER) + "acc_x" +
                           std::to_string(DATA_DELIMITER) + "acc_y" +
                           std::to_string(DATA_DELIMITER) + "acc_z");
      return std::nullopt;
    }

    /* auto timestamp = parse_datetime(curr_line.substr(0, pos), DATETIME_FORMAT); */  // We do not care about parsing the timestamps anymore
    curr_line.erase(0, pos + 1);  // +1 to delete the delimiter as well

    parsed = parse_acc_value(curr_line);

    if (parsed == std::nullopt) {
      logger.log_warning(warnings::WARNINGS::ACC_VALUE_NOT_PARSED);
      return std::nullopt;
    }

    curr_vals = parsed.value();

    values_x.push_back(curr_vals[0]);
    values_y.push_back(curr_vals[1]);
    values_z.push_back(curr_vals[2]);

    if (++lines % LOGGING_THRESHOLD == 0) {
      logger.log_info("Parsed " + std::to_string(lines) +
                      " lines in current ACC file");
    };
  }

  logger.log_info("Parsed " + std::to_string(lines) + " lines from " +
                  _acc_file_path);

  std::array<std::vector<float_t>, ACC_NO_VALUES> results = {values_x, values_y,
                                                             values_z};

  logger.log_debug("Values X size: " + std::to_string(values_x.size()));
  logger.log_debug("Values Y size: " + std::to_string(values_y.size()));
  logger.log_debug("Values Z size: " + std::to_string(values_z.size()));

  return results;
}

const std::optional<std::vector<float_t>> SubjectDataProcessor::parse_hr_file(
    const uint8_t period_size, const u_long timestamp_diff) noexcept {
  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "(Provided value: " + std::to_string(period_size) + ")");
    return std::nullopt;
  }
  const size_t MAX_LINE_LEN = 32, LOGGING_THRESHOLD = 100000,
               MAX_VAL_STR_LEN = 5;
  std::vector<float_t> values;
  std::string curr_line, tmp;
  u_long pos, lines = 0;
  uint8_t curr_val;

  curr_line.reserve(MAX_LINE_LEN);
  tmp.reserve(MAX_VAL_STR_LEN);
  _hr_file_stream.seekg(_hr_file_stream.beg);
  std::getline(_hr_file_stream, curr_line);  // Skip the first csv header line

  // Sync up with accelerometer measurements
  u_long diff = (long)((timestamp_diff * HR_SAMPLE_FREQ) / period_size);
  if (diff > 0) {
    logger.log_info("HR measurements are \"ahead\" by " + std::to_string(diff) +
                    " lines. Skipping...");
  }
  for (u_long i = 0; i < diff; ++i) {
    std::getline(_hr_file_stream, curr_line);
  }

  logger.log_info("Beginning parsing file " + _hr_file_path);

  while (std::getline(_hr_file_stream, curr_line)) {
    pos = curr_line.find(DATA_DELIMITER);
    if (pos == std::string::npos) {
      logger.log_error(
          errors::ERRORS::INVALID_FILE_STRUCTURE,
          "HR files need to have the following structure: <datetime" +
              std::to_string(DATA_DELIMITER) + " hr>");
      return std::nullopt;
    }

    tmp = curr_line.substr(pos + 1, curr_line.length() - 1);

    try {
      curr_val = std::stoi(tmp);
    } catch (std::invalid_argument& e) {
      logger.log_error(errors::COULD_NOT_PARSE_VALUE, "(Value: " + tmp + ")");
      return std::nullopt;
    }
    values.insert(values.end(), curr_val);

    if (++lines % LOGGING_THRESHOLD == 0) {
      logger.log_info("Parsed " + std::to_string(lines) +
                      " in the current HR file");
    }
  }

  return values;
}

std::vector<float_t> SubjectDataProcessor::normalize_acc_values(
    const std::uint8_t period_size,
    const std::vector<float_t>& values) const noexcept {
  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "(Provided value: " + std::to_string(period_size) + ")");
    return std::vector<float_t>();
  }

  const size_t NORMALIZATION_PERIOD = ACC_SAMPLE_FREQ * period_size;

  std::vector<float_t> rv(values.size() / NORMALIZATION_PERIOD + 1, 0.0f);

  logger.log_info("Beginning normalizing ACC values... ");

  std::for_each(std::execution::par, values.begin(), values.end(),
                [&rv, &values, NORMALIZATION_PERIOD](float_t const& value) {
                  size_t index = (&value - &values[0]) / NORMALIZATION_PERIOD;
                  rv[index] += value;
                });

  const uint8_t ACC_MAX_VALUE = 127;
  std::for_each(std::execution::par_unseq, rv.begin(), rv.end(),
                [NORMALIZATION_PERIOD](float_t& value) {
                  value /= NORMALIZATION_PERIOD * ACC_MAX_VALUE;
                });

  logger.log_debug("Normalized ACC values count: " + std::to_string(rv.size()));

  return rv;
}

std::vector<float_t> SubjectDataProcessor::normalize_hr_values(
    const uint8_t period_size,
    const std::vector<float_t>& values) const noexcept {

  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "(Provided value: " + std::to_string(period_size) + ")");
    return std::vector<float_t>();
  }

  std::vector<float_t> rv(values.size() / period_size, 0.0f);

  std::for_each(std::execution::par, values.begin(), values.end(),
                [&rv, &values, period_size](float_t const& value) {
                  size_t index = (&value - &values[0]) / period_size;
                  rv[index] += value;
                });

  std::for_each(
      std::execution::par_unseq, rv.begin(), rv.end(),
      [period_size](float_t& value) { value /= (period_size * HR_MAX_VALUE); });

  logger.log_debug("Normalized HR values count: " + std::to_string(rv.size()));

  return rv;
}

size_t SubjectDataProcessor::interpolate_vector_linear(
    std::vector<float_t>& vector, const size_t count) noexcept {
  size_t old_size = vector.size();

  size_t new_size = old_size + count;
  size_t padding = count;

  if ((new_size & (new_size - 1)) != 0) {  // Not a power of 2
    double exp = std::floor(std::log2(new_size));
    new_size = (size_t)pow(2, exp + 1);
    padding = new_size - old_size;
  }

  vector.resize(new_size);

  for (size_t i = 0; i < padding; ++i) {
    float_t t = (float_t)i / (padding - 1);
    size_t index = (size_t)(t * (old_size - 1));  // Index in the smaller vector
    float_t fraction =
        t * (old_size - 1) - index;  // Fractional part for interpolation
    vector[old_size + i] =
        vector[index] + fraction * (vector[index + 1] -
                                    vector[index]);  // Lerp = a + f * (b - a)
  }

  return padding - count;
}

// PUBLIC METHODS //

const std::chrono::system_clock::time_point
SubjectDataProcessor::parse_datetime(const std::string& timestamp,
                                     const std::string& format) noexcept {
  std::tm tm{};
  std::istringstream ss(timestamp);
  ss >> std::get_time(&tm, format.c_str());

  return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

std::pair<uint8_t, long long> SubjectDataProcessor::validate_timestamps(
    const uint8_t period_size) noexcept {
  if (!_acc_file_stream.is_open() || !_hr_file_stream.is_open()) {
    logger.log_error(errors::ERRORS::COULD_NOT_OPEN_FILE_HANDLE,
                     "(" + _hr_file_path + " or " + _acc_file_path + ")");
    return std::pair(EXIT_FAILURE, LONG_MIN);
  }

  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "Provided value: (" + std::to_string(period_size) + ")");
    return std::pair(RETURN_NOK, LONG_MIN);
  }

  _acc_file_stream.seekg(_acc_file_stream.beg);
  _hr_file_stream.seekg(_hr_file_stream.beg);

  std::string curr_line;
  u_long pos;

  // Skip the first lines
  std::getline(_acc_file_stream, curr_line);
  std::getline(_hr_file_stream, curr_line);

  std::getline(_acc_file_stream, curr_line);
  pos = curr_line.find(DATA_DELIMITER);
  if (pos == std::string::npos) {
    logger.log_error(errors::ERRORS::INVALID_FILE_STRUCTURE,
                     "ACC files must have the following structure: "
                     "datetime" +
                         std::to_string(DATA_DELIMITER) + "acc_x" +
                         std::to_string(DATA_DELIMITER) + "acc_y" +
                         std::to_string(DATA_DELIMITER) + "acc_z");
    return std::pair(EXIT_FAILURE, LONG_MIN);
  }

  curr_line.erase(pos + 1, curr_line.length() - 1);

  const std::chrono::time_point acc_timestamp =
      parse_datetime(curr_line, DATETIME_FORMAT);

  std::getline(_hr_file_stream, curr_line);

  pos = curr_line.find(DATA_DELIMITER);
  if (pos == std::string::npos) {
    logger.log_error(errors::ERRORS::INVALID_FILE_STRUCTURE,
                     "HR files must have the following structure: "
                     "datetime" +
                         std::to_string(DATA_DELIMITER) + "hr");
    return std::pair(EXIT_FAILURE, LONG_MIN);
  }

  curr_line.erase(pos + 1, curr_line.length() - 1);

  const std::chrono::time_point hr_timestamp =
      parse_datetime(curr_line, DATETIME_FORMAT);

  std::chrono::seconds acc_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(
          acc_timestamp.time_since_epoch());

  logger.log_debug("Calculated ACC time since epoch: " +
                   std::to_string(acc_seconds.count()));

  std::chrono::seconds hr_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(
          hr_timestamp.time_since_epoch());

  logger.log_debug("Calculated HR time since epoch: " +
                   std::to_string(hr_seconds.count()));

  long long timestamp_diff = acc_seconds.count() - hr_seconds.count();
  if (period_size == 1) {
    return std::pair(EXIT_SUCCESS, timestamp_diff);
  }

  int8_t tmp = timestamp_diff > 0 ? 1 : -1;

  timestamp_diff = timestamp_diff * tmp;
  uint8_t remainder = timestamp_diff % period_size;

  if (remainder != 0) {
    timestamp_diff += period_size - remainder + 1;
  }

  return std::pair(RETURN_OK, timestamp_diff * tmp);
}

const std::optional<std::array<std::vector<float_t>, ACC_NO_VALUES>>
SubjectDataProcessor::preprocess_acc_file(
    const std::uint8_t period_size, const u_long timestamp_diff) noexcept {
  logger.log_debug("Period size: " + std::to_string(period_size));
  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "(Provided value: " + std::to_string(period_size) + ")");
    return std::nullopt;
  }

  std::optional parsed_optional = parse_acc_file(period_size, timestamp_diff);

  if (parsed_optional == std::nullopt) {
    return std::nullopt;
  }

  std::array<std::vector<float_t>, ACC_NO_VALUES> parsed_values =
      parsed_optional.value();

  logger.log_debug("Parsed values size: " +
                   std::to_string(parsed_values.size()));

  std::array<std::vector<float_t>, ACC_NO_VALUES> normalized_values;
  size_t count = 0;
  std::for_each(std::execution::seq, parsed_values.begin(), parsed_values.end(),
                [period_size, &normalized_values, &count,
                 this](std::vector<float_t>& curr_vals) {
                  normalized_values[count++] =
                      normalize_acc_values(period_size, curr_vals);
                });

  if (normalized_values.size() != ACC_NO_VALUES) {
    return std::nullopt;
  }

  for (size_t i = 0; i < normalized_values.size(); ++i) {
    if (normalized_values.empty()) {
      return std::nullopt;
    }
  }

  return normalized_values;
}

const std::optional<std::vector<float_t>>
SubjectDataProcessor::preprocess_hr_file(const uint8_t period_size,
                                         const u_long timestamp_diff) noexcept {
  if (period_size == 0) {
    logger.log_error(errors::ERRORS::INVALID_PERIOD_SIZE,
                     "(Provided value: " + std::to_string(period_size) + ")");
    return std::nullopt;
  }

  std::optional parsed_optional = parse_hr_file(period_size, timestamp_diff);
  if (parsed_optional == std::nullopt) {
    return std::nullopt;
  }

  std::vector<float_t> parsed_values = parsed_optional.value();

  if (parsed_values.empty()) {
    return std::nullopt;
  }

  return normalize_hr_values(period_size, parsed_values);
}

};  // namespace DataPreprocessing
