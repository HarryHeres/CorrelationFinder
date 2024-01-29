#pragma once

#include <cmath>
#include <cstdint>
#include <string>

extern const uint8_t MAX_SUPPORTED_PERIOD_SIZE;

extern const std::string FILE_PATH_SEPARATOR;
extern const std::string RESOURCE_FOLDER_PATH;
extern const std::string SOURCE_FILE_FORMAT;
extern const std::string OPENCL_KERNEL_FILE_PATH;
extern const uint8_t RETURN_OK;
extern const uint8_t RETURN_NOK;
extern const uint8_t ACC_SAMPLE_FREQ;
extern const uint8_t HR_SAMPLE_FREQ;
extern const uint8_t HR_MAX_VALUE;
extern const uint8_t ACC_NO_VALUES;
extern const uint8_t FLOATS_PER_AVX2;
extern const uint8_t MIN_VEC_SIZE_AVX2;

extern const float_t X_FLOAT_REPRESENTATION;
extern const float_t ADD_FLOAT_REPRESENTATION;
extern const float_t SUB_FLOAT_REPRESENTATION;
extern const float_t MUL_FLOAT_REPRESENTATION;
extern const float_t DIV_FLOAT_REPRESENTATION;

extern const size_t GENERATION_SIZE;
extern const size_t GENERATION_INDIVIDUAL_SIZE;
extern const size_t GENERATION_ITERATION_COUNT;
extern const size_t GENERATION_TREE_NODE_SIZE;
