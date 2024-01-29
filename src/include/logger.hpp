#pragma once

#include <filesystem>
#include <fstream>
#include <string>

#include "errors.hpp"
#include "warnings.hpp"

namespace Logging {

/** Logger available logging levels */
enum LOG_LEVEL { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };

/** Logging level string representation */
const std::string LOG_LEVEL[] = {"DEBUG", "INFO", "WARNING", "ERROR"};

/** Default log files folder path */
extern std::string LOG_FOLDER_PATH;

/** Maximum length of a log message (including the timestamps) */
constexpr std::size_t MAX_LOG_MSG_LEN = 512;

/** App log level. Cannot log messages that have lower priority than this level */
extern enum LOG_LEVEL APP_LOGGING_LEVEL;

/**
 * Custom logger class.
 * All info is logged to BOTH stdout and a log file specified in the constructor
 */
class Logger {
 private:
  std::string _log_file_path;
  std::ofstream _log_file_stream;

  Logger() noexcept;

 public:
  Logger(Logger const&) = delete;
  void operator=(Logger const&) = delete;

  /**
   * Destructor for this class.
   * Closes the log file if still open
   */
  virtual ~Logger();

  /**
   * Log a general message 
   *
   * @param level Logging level be used for logging
   * @param message Message to be logged
   */
  void log_message(const enum LOG_LEVEL level,
                   const std::string& message) noexcept;

  /**
   * Log a debug message 
   *
   * @param message Message to be logged
   */
  void log_debug(const std::string& message) noexcept;

  /**
   * Log an informational message 
   *
   * @param message Message to be logged
   */
  void log_info(const std::string& message) noexcept;

  /**
   * Log a predefined WARNING message
   *
   * @param warning_number Warning code constant from the Warnings namespace
   * @param append Optional string to be appended to the end of the message
   */
  void log_warning(const warnings::WARNINGS warning_number,
                   const std::string& append = "") noexcept;

  /**
   * Log a predefined ERROR message
   *
   * @param error_number Error code constant from the Errors namespace
   * @param append Optional string to be appended to the end of the message
   */
  void log_error(const errors::ERRORS error_number,
                 const std::string& append = "") noexcept;

  /**
   * Method to get a concrete Singleton instance of the Logger class
   *
   * @return Singleton instance 
   */
  static Logger& get_instance() {
    static Logger instance;
    return instance;
  }
};

};  // namespace Logging
