#pragma once

#include <cmath>
#include <optional>
#include <vector>
#include "logger.hpp"

namespace DataPreprocessing {

using u_long = unsigned long;
/** Internal logger instance */
extern Logging::Logger& logger;

/** Character delimiter of values in the source files */
constexpr static char DATA_DELIMITER = ',';

/** Number of different values (axis). In our case is 3 because we have 3 axis (X,Y,Z) */
constexpr size_t ACC_NO_VALUES = 3;

/** Format of the timestamps in the source files */
const std::string DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S";

/** Class used for subject data preprocessing methods */
class SubjectDataProcessor {
 private:
  std::string _acc_file_path;
  std::ifstream _acc_file_stream;

  std::string _hr_file_path;
  std::ifstream _hr_file_stream;

  /**
   * Parse values from the string. Example: "-1, 0, 1" would parse into std::vector{-1, 0, 1}
   *
   * @param values String of values to be parsed. WILL BE MODIFIED!
   *
   * @return An array representing the values in order: X,Y,Z, std::nullopt if the values could not have been parsed
   */
  const std::optional<std::array<float_t, ACC_NO_VALUES>> parse_acc_value(
      std::string& value_string) const noexcept;

  /**
   * Parse out the whole ACC source file
   *
   * @param period_size Selected size of watched period (e.g. 1s, 10s, 20s, ...)
   * @param timestamp_diff Time difference by which are the accelerometer measurements "ahead"
   *
   * @return An array of X,Y,Z vectors representing the parsed values
   */
  const std::optional<std::array<std::vector<float_t>, ACC_NO_VALUES>>
  parse_acc_file(const uint8_t period_size = 1,
                 const u_long timestamp_diff = 0) noexcept;

  /**
   * Parse out the whole HR source file
   *
   * @param period_size Selected size of watched period (e.g. 1s, 10s, 20s, ...)
   * @param timestamp_diff Time difference by which are the heart rate monitor measurements "ahead"
   *
   * @return A vector of parsed HR values
   */
  const std::optional<std::vector<float_t>> parse_hr_file(
      const uint8_t period_size = 1, const u_long timestamp_diff = 0) noexcept;

  /**
   * Normalize values from the accelerometer.
   *
   * @param period_size Selected size of watched period (e.g. 1s, 10s, 20s, ...). 
   * @param values Vector of measured values (only one axis at a time).
   *
   * @return Vector of normalized values -> moving averages mapped onto (0;1) interval
   */
  std::vector<float_t> normalize_acc_values(
      const uint8_t period_size,
      const std::vector<float_t>& values) const noexcept;

  /**
   * Normalize values from the HR monitor
   *
   * @param period_size Selected size of watched period (e.g. 1s, 10s, 20s, ...). 
   * @param Values vector of measured values
   *
   * @return Vector of normalized values -> moving averages mapped onto (0;1) interval
   */
  std::vector<float_t> normalize_hr_values(
      const uint8_t period_size,
      const std::vector<float_t>& values) const noexcept;

 public:
  /**
   * Class Constructor
   *
   * @param acc_file_path Path to the accelerometer source file 
   * @param hr_file_path Path to the heart rate source file 
   */
  SubjectDataProcessor(const std::string& acc_file_path,
                       const std::string& hr_file_path);

  virtual ~SubjectDataProcessor();

  /**
   * Parse string timestamp into a datetime value 
   *
   * @param timestamp Timestamp to be parsed
   * @param format Datetime format to be used in the parsing
   *
   * @return New value representation of the @param timestamp
   */
  static const std::chrono::system_clock::time_point parse_datetime(
      const std::string& timestamp, const std::string& format) noexcept;

  /**
   * Method to validate if timestamps in both @code acc_file_path and @code hr_file_path are in sync
   * 
   * @param period_size Size of the normalization period. 
   *
   * @return Pair of return code and the difference. If all went well, the first value will be RETURN_OK, RETURN_NOK if otherwise. If you get RETURN_OK, the second value will represent the measured difference in seconds between the timestamps. If it is positive, ACC file is "ahead", otherwise HR file is "ahead". 0 means there is no difference
   */
  std::pair<uint8_t, long long> validate_timestamps(
      const uint8_t period_size = 1) noexcept;

  /**
   * Preprocess the ACC source file
   *
   * @param period_size Size of the period over which the values should be averaged (e.g. 1=1s, 10=10s, ...). Default is 1s
   * @param timestamp_diff Signalizes that the accelerometer is out of sync with the heart rate monitor. If other than zero, the number of measurements in this time interval will be skipped
   *
   * @return An array of X,Y,Z respective vectors of moving average values for each axis
   */
  const std::optional<std::array<std::vector<float_t>, ACC_NO_VALUES>>
  preprocess_acc_file(const uint8_t period_size = 1,
                      const u_long timestamp_diff = 0) noexcept;

  /**
   * Preprocess the HR source file
   *
   * @param period_size Size of the period over which the values should be averaged (e.g. 1=1s, 10=10s, ...)
   * @param timestamp_diff Signalizes that the heart rate monitor is out of sync with the accelerometer. If other than zero, the number of measurements in this time interval will be skipped
   *
   * @return A vector of heart rates that are already preprocessed
   */
  const std::optional<std::vector<float_t>> preprocess_hr_file(
      const uint8_t period_size = 1, const u_long timestamp_diff = 0) noexcept;

  /**
   * Perform a linear interpolation on a vector and optional padding, if the new length is not padded for AVX2
   *
   * @param vector Vector to be appended the new interpolated values
   * @param count The number of newly interpolated values
   *
   * @return the number of elements that were added due to AVX2 padding
   */
  size_t interpolate_vector_linear(std::vector<float_t>& vector,
                                   const size_t count) noexcept;
};
}  // namespace DataPreprocessing
