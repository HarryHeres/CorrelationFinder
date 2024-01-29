#include <filesystem>
#include <iostream>
#include <string>

#include "include/constants.hpp"
#include "include/errors.hpp"
#include "include/logger.hpp"
#include "include/warnings.hpp"

namespace Logging {

enum LOG_LEVEL APP_LOGGING_LEVEL = Logging::LOG_LEVEL::DEBUG;

Logger::Logger() noexcept {
  const std::string LOG_FOLDER_PATH = std::filesystem::path("log");

  const time_t time = std::time(nullptr);
  const std::tm* timestamp = std::localtime(&time);
  const size_t MAX_FILE_NAME_LEN = 32;

  this->_log_file_path = LOG_FOLDER_PATH;

  std::error_code err{};
  std::filesystem::create_directories(this->_log_file_path, err);

  if (!std::filesystem::exists(this->_log_file_path)) {
    std::cout << "Logs directory " << this->_log_file_path
              << " could not have been created. Reason: " << err.message()
              << std::endl;
    return;
  }

  //Efficient string concatenation - minimizing the number of copies and allocations
  this->_log_file_path.reserve(MAX_FILE_NAME_LEN);
  this->_log_file_path.append("/")
      .append("log_")
      .append(std::to_string(timestamp->tm_mday))
      .append("-")
      .append(std::to_string(timestamp->tm_mon + 1))
      .append("_")
      .append(std::to_string(1900 + timestamp->tm_year))
      .append("_")
      .append(std::to_string(timestamp->tm_hour))
      .append("-")
      .append(std::to_string(timestamp->tm_min))
      .append("-")
      .append(std::to_string(timestamp->tm_sec))
      .append(".txt");

  this->_log_file_stream.open(this->_log_file_path,
                              std::fstream::out | std::fstream::app);
}

Logger::~Logger() {
  if (this->_log_file_stream.is_open()) {
    this->_log_file_stream.close();
  }
}

void Logger::log_message(enum LOG_LEVEL level,
                         const std::string& message) noexcept {

  if (level < Logging::APP_LOGGING_LEVEL) {
    return;
  }

  const time_t time = std::time(nullptr);
  const std::tm* timestamp = std::localtime(&time);

  std::string full_message;
  full_message.reserve(MAX_LOG_MSG_LEN);
  full_message.append("[")
      .append(std::to_string(timestamp->tm_hour))
      .append(":")
      .append(std::to_string(timestamp->tm_min))
      .append(":")
      .append(std::to_string(timestamp->tm_sec))
      .append("]: ")
      .append(LOG_LEVEL[level])
      .append(": ")
      .append(message);

  if (level == LOG_LEVEL::ERROR) {
    std::cerr << full_message << std::endl;
  } else {
    std::cout << full_message << std::endl;
  }

  if (this->_log_file_stream.is_open()) {
    this->_log_file_stream << full_message << std::endl;
  }
}

void Logger::log_debug(const std::string& message) noexcept {
  if (message.empty()) {
    return;
  }

  this->log_message(LOG_LEVEL::DEBUG, message);
}

void Logger::log_info(const std::string& message) noexcept {
  if (message.empty()) {
    return;
  }

  this->log_message(LOG_LEVEL::INFO, message);
}

void Logger::log_warning(enum warnings::WARNINGS warning_code,
                         const std::string& to_append) noexcept {
  std::string message = warnings::WARNINGS_MAP.at(warning_code);

  if (!to_append.empty()) {
    message.append(" ").append(to_append);
  }

  this->log_message(LOG_LEVEL::WARNING, message);
}

void Logger::log_error(enum errors::ERRORS error_code,
                       const std::string& to_append) noexcept {
  std::string message = errors::ERRORS_MAP.at(error_code);

  if (!to_append.empty()) {
    message.append(" ").append(to_append);
  }

  this->log_message(LOG_LEVEL::ERROR, message);
}

};  // namespace Logging
