#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include "constants.hpp"

namespace warnings {

enum WARNINGS {
  FILE_NOT_FOUND = 0,
  FILE_IS_EMPTY = 1,
  FILE_HANDLE_NOT_OPEN = 2,
  TIMESTAMP_CALCULATION_WARNING = 3,
  ACC_VALUE_NOT_PARSED = 3,
  PARAMETER_WAS_EMPTY = 4,
  COULD_NOT_PARSE_CMD_ARGS = 5,
  INVALID_PERIOD_SIZE = 6,
};

/** Map of all available warnings and their respective messages */
const std::unordered_map<size_t, std::string> WARNINGS_MAP = {
    {FILE_NOT_FOUND, "File has not been found"},
    {FILE_IS_EMPTY, "File has no contents"},
    {FILE_HANDLE_NOT_OPEN, "Tried to read a closed file"},
    {TIMESTAMP_CALCULATION_WARNING,
     "Timestamp difference calculation has encountered a problem. The "
     "difference will not be taken into account"},
    {ACC_VALUE_NOT_PARSED, "Following value could not have been parsed"},
    {PARAMETER_WAS_EMPTY, "Function parameter was empty"},
    {COULD_NOT_PARSE_CMD_ARGS,
     "Could not parse one or more command line arguments"},
    {INVALID_PERIOD_SIZE,
     "Invalid period size specified. Supported period size is between 1 and " +
         std::to_string(MAX_SUPPORTED_PERIOD_SIZE)},

};
}  // namespace warnings
