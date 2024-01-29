#include "include/constants.hpp"

const std::string FILE_PATH_SEPARATOR =
#if defined(WIN32) || defined(_WIN32)
    "\\"
#else
    "/"
#endif
    ;

const uint8_t MAX_SUPPORTED_PERIOD_SIZE = 1;

const std::string RESOURCE_FOLDER_PATH = "resources";
const std::string SOURCE_FILE_FORMAT = ".csv";
const std::string OPENCL_KERNEL_FILE_PATH = "src/kernel.cl";
const uint8_t RETURN_OK = 0;
const uint8_t RETURN_NOK = -1;
const uint8_t ACC_SAMPLE_FREQ = 32;
const uint8_t HR_SAMPLE_FREQ = 1;
const uint8_t HR_MAX_VALUE = 255;
const uint8_t ACC_NO_VALUES = 3;
const uint8_t FLOATS_PER_AVX2 = 8;
const uint8_t MIN_VEC_SIZE_AVX2 = 16;

const float_t X_FLOAT_REPRESENTATION = 11.0f;
const float_t ADD_FLOAT_REPRESENTATION = 1.0f;
const float_t SUB_FLOAT_REPRESENTATION = 2.0f;
const float_t MUL_FLOAT_REPRESENTATION = 3.0f;
const float_t DIV_FLOAT_REPRESENTATION = 4.0f;

const size_t GENERATION_SIZE = 100;
const size_t GENERATION_INDIVIDUAL_SIZE = 30;
const size_t GENERATION_ITERATION_COUNT = 100;
const size_t GENERATION_TREE_NODE_SIZE = 3;
