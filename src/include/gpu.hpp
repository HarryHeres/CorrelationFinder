#pragma once

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 100

#ifdef __APPLE__
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#else
#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_TARGET_OPENCL_VERSION 200
#endif

#include <CL/opencl.hpp>
#include <optional>
#include <vector>
#include "logger.hpp"
#include "math.h"

namespace opencl {

/**
   * Class representing a GPU OpenCL device
   */
class Gpu {
 private:
  /** OpenCL device representation */
  const cl::Device device;

  /** Compiled OpenCL source code */
  cl::Program program;

  /** OpenCL queue for this device */
  cl::CommandQueue device_queue;

  /**
   * Load source code (kernel) for the OpenCL device from a .cl file
   *
   * @param filename Path to the source .cl file
   *
   * @return String representing the whole source code (not compiled yet)
   */
  static const std::string load_kernel_source_from_file(
      const std::string& filepath) noexcept;

  /**
   * Calculate the ACC values sum and differences squared (the parts of the Pearson's correlation formula)
   *
   * @param device_queue OpenCL event queue
   * @param nominator_buffer OpenCL buffer for the nominator of the formula
   * @param acc_diff_squared OpenCL buffer for the squared differences of the ACC values (denominator)
   * @param hr_values_diffs_buffer OpenCL buffer with the squared differences (of each value and the average) of the HR values (denominator)
   * @param buffer_size Size of the OpenCL buffers
   * @param acc_avg Average of the ACC values
   */
  void calculate_correlation_acc_values(
      const cl::CommandQueue& device_queue, const cl::Buffer& nominator_buffer,
      const cl::Buffer& acc_diff_squared_buffer,
      const cl::Buffer& hr_values_diffs_buffer, const size_t buffer_size,
      const float_t acc_avg) const noexcept;

 public:
  /** OpenCL context of this OpenCL device */
  cl::Context device_context;

  Gpu(cl::Device device);

  /** Return this device's OpenCL event queue */
  const cl::CommandQueue get_device_queue() const noexcept;

  /** 
   * Fill the OpenCL buffer with a value
   *
   * @param device_queue OpenCL queue 
   * @param value Value to fill the buffer with
   * @param buffer_size Size of the buffer
   * @param buffer Buffer to be filled with the value
   */
  void fill_float_buffer(const cl::CommandQueue& device_queue,
                         const float_t value, const size_t buffer_size,
                         const cl::Buffer& buffer) const noexcept;

  /**
   * Copy values from one buffer to another
   *
   * @param device_queue OpenCL queue
   * @param buffer_size Size of the buffers
   * @param from Source buffer
   * @param from_offset Source buffer offset
   * @param to Destination buffer
   * @param to_offset Destination buffer offset
   *
   */
  void copy_float_buffer(const cl::CommandQueue& device_queue,
                         const size_t buffer_size, const cl::Buffer& from,
                         const size_t from_offset, const cl::Buffer& to,
                         const size_t to_offset) const noexcept;

  /**
   * Parallel sum of a vector represented inside the OpenCL buffer
   *
   * @param device_queue OpenCL queue
   * @param vector_len Count of the elements inside the vector
   * @param vector_buffer Buffer representing the vector
   *
   * @return Pointer to the parallel sums inside the buffer. The LAST element will represent the whole sum
   */
  float_t* sum_vector(const cl::CommandQueue& device_queue,
                      const size_t vector_len,
                      const cl::Buffer& vector_buffer) const noexcept;

  /**
   * Perform a generation crossover between every two individuals of the generation
   *
   * @param device_queue OpenCL queue
   * @param generation_buffer Buffer with the whole generation
   * @param crossover_point Position from which will the individuals be "crossed" (swapped)
   */
  void perform_crossover(const cl::CommandQueue& device_queue,
                         const cl::Buffer& generation_buffer,
                         const size_t crossover_point) const noexcept;

  /** 
   * Compute Pearson's correlation coefficient between two vectors (ACC/HR_generated and initial HR)
   *
   * @param acc_buffer Buffer of the initial ACC values
   * @param hr_values_diffs_buffer Buffer with the initial HR values differences (of each value and their global average)
   * @param hr_values_diff_squared_root Square root of the square of differences (of each value and their global average)
   * @param buffer_size Size of the ACC buffer
   * @param vector_len Number of the ACC values
   */
  float_t compute_pearsons_correlation(
      const cl::Buffer& acc_buffer, const cl::Buffer& hr_values_diffs_buffer,
      const float_t hr_values_diff_squared_root, const size_t buffer_size,
      const size_t vector_len) const noexcept;

  /**
   * Compute the correlation formula of the initial ACC and HR values using a genetic algorithm 
   *
   * @param acc_values initial ACC values
   * @param hr_values_diffs Vector where each element represents a difference between the initial HR value and their global average
   * @param hr_values_diff_squared_root Square root of the square of differences (of each value and their global average)
   *
   * @return Pair of values. First represents the newly generated HR values and the second represents a syntax tree of nodes with the following structure: 
   * [0, root (operation), left_child (operand), right_child (operand), root, ...]
   */
  std::pair<std::vector<float_t>, std::vector<float_t>>
  compute_correlation_formula(
      std::vector<float_t>& acc_values, std::vector<float_t>& hr_values_diffs,
      const float_t hr_values_diff_squared_root) const noexcept;

  /**
   * Print out the results of compilation of the OpenCL source code
   *
   * @param program OpenCL source code 
   */
  void dump_opencl_build_log(const cl::Program& program) const noexcept;
};

}  // namespace opencl
