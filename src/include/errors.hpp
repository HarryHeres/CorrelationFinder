#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

namespace errors {

enum ERRORS {
  RESOURCE_FOLDER_NOT_FOUND = 1,
  FILE_NOT_FOUND = 2,
  FILE_IS_EMPTY = 3,
  COULD_NOT_OPEN_FILE_HANDLE = 4,
  INVALID_FILE_STRUCTURE = 5,
  COULD_NOT_PARSE_VALUE = 6,
  INVALID_PERIOD_SIZE = 7,
  PARAMETER_WAS_EMPTY = 8,
  INVALID_AVX_VECTOR_SIZE = 9,
  INVALID_ARGUMENT = 10,
  COULD_NOT_PREPROCESS_VALUES = 11,
  OPENCL_BUILD_ERROR = 12,
  OPENCL_BUFFER_ALLOC_ERROR = 13,
  OPENCL_NO_DEVICE_FOUND = 14,
};

/** Map of all available errors and their respective messages */
const std::unordered_map<size_t, std::string> ERRORS_MAP = {
    {RESOURCE_FOLDER_NOT_FOUND, "Resource folder has not been found"},
    {FILE_NOT_FOUND, "File has not been found"},
    {FILE_IS_EMPTY, "File is empty"},
    {COULD_NOT_OPEN_FILE_HANDLE, "File could not have been open"},
    {INVALID_FILE_STRUCTURE, "Invalid file structure"},
    {COULD_NOT_PARSE_VALUE, "Could not parse value"},
    {INVALID_PERIOD_SIZE, "Period size must be greater than 1"},
    {PARAMETER_WAS_EMPTY, "Input parameter was empty"},
    {INVALID_AVX_VECTOR_SIZE,
     "Vector size for AVX instructions needs to be atleast twice the size of "
     "AVX registers and padded correctly"},
    {INVALID_ARGUMENT, "Invalid argument"},
    {COULD_NOT_PREPROCESS_VALUES, "Could not preprocess HR or ACC values"},
    {OPENCL_BUILD_ERROR, "OpenCL kernel build has encountered an error"},
    {OPENCL_BUFFER_ALLOC_ERROR, "OpenCL could not allocate a buffer"},
    {OPENCL_NO_DEVICE_FOUND,
     "No OpenCL computing device found. Cannot proceed further."},

};
}  // namespace errors
