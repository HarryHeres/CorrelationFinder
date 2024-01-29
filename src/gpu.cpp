#include "include/gpu.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include "include/avx.hpp"
#include "include/constants.hpp"

namespace opencl {

Logging::Logger& logger = Logging::Logger::get_instance();

const std::string Gpu::load_kernel_source_from_file(
    const std::string& filepath) noexcept {
  if (filepath.empty()) {
    logger.log_error(errors::ERRORS::COULD_NOT_OPEN_FILE_HANDLE,
                     "Provided filename was empty");
  }
  std::ifstream input_stream;
  input_stream.open(filepath);

  if (!input_stream.is_open()) {
    logger.log_error(errors::ERRORS::FILE_NOT_FOUND,
                     "(Filename: " + OPENCL_KERNEL_FILE_PATH + ")");
  }

  std::stringstream kernel_string_stream;
  kernel_string_stream << input_stream.rdbuf();
  std::string kernel_string = kernel_string_stream.str();

  if (input_stream.is_open()) {
    input_stream.close();
  }

  return kernel_string;
}

Gpu::Gpu(cl::Device device) : device(device) {
  std::vector<std::string> source_codes{
      load_kernel_source_from_file(OPENCL_KERNEL_FILE_PATH)};
  cl::Program::Sources sources(source_codes);
  cl::Context device_context = {device};
  cl::Program program(device_context, sources);

  try {
    program.build(device, "-cl-std=CL2.0");
  } catch (cl::Error& err) {
    std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
    logger.log_error(errors::ERRORS::OPENCL_BUILD_ERROR, log);
    return;
  }

  this->device_context = device_context;
  this->program = program;
  this->device_queue = cl::CommandQueue(device_context, device, 0);
}

const cl::CommandQueue Gpu::get_device_queue() const noexcept {
  return this->device_queue;
}

void Gpu::fill_float_buffer(const cl::CommandQueue& device_queue,
                            const float_t value, const size_t buffer_size,
                            const cl::Buffer& buffer) const noexcept {
  try {
    device_queue.enqueueFillBuffer(buffer, value, 0.0f, buffer_size);
  } catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUILD_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
  }
}

void Gpu::copy_float_buffer(const cl::CommandQueue& device_queue,
                            const size_t buffer_size, const cl::Buffer& from,
                            const size_t from_offset, const cl::Buffer& to,
                            const size_t to_offset) const noexcept {
  try {
    device_queue.enqueueCopyBuffer(from, to, from_offset, to_offset,
                                   buffer_size);
  } catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUILD_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
  }
}

float_t* Gpu::sum_vector(const cl::CommandQueue& device_queue,
                         const size_t vector_len,
                         const cl::Buffer& vector_buffer) const noexcept {
  try {
    cl::Kernel kernel(program, "parallel_prefix_sum");
    this->dump_opencl_build_log(program);

    for (size_t i = 0; i < log2(vector_len); ++i) {
      int idx_offset = (int)std::pow(2, i + 1.0);
      size_t nd_range = (size_t)(vector_len / idx_offset);

      kernel.setArg(0, vector_buffer);
      kernel.setArg(1, sizeof(int), &idx_offset);

      device_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                        cl::NDRange(nd_range), cl::NullRange);
    }

    float_t* result = reinterpret_cast<float*>(device_queue.enqueueMapBuffer(
        vector_buffer,
        CL_TRUE,  // blokovat
        CL_MAP_READ, 0,
        vector_len * sizeof(float_t)));  // The sum is at the last position

    return result;
  } catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUILD_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
  }
  return nullptr;
}

void Gpu::calculate_correlation_acc_values(
    const cl::CommandQueue& device_queue, const cl::Buffer& nominator_buffer,
    const cl::Buffer& acc_diff_squared_buffer,
    const cl::Buffer& hr_values_diffs_buffer, const size_t values_count,
    const float_t acc_avg) const noexcept {

  try {
    cl::Kernel kernel(program, "calculate_correlation_acc_values");
    this->dump_opencl_build_log(program);

    kernel.setArg(0, nominator_buffer);
    kernel.setArg(1, acc_diff_squared_buffer);
    kernel.setArg(2, hr_values_diffs_buffer);
    kernel.setArg(3, acc_avg);

    device_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange(values_count), cl::NullRange);

  } catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUILD_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
  }
}

void Gpu::perform_crossover(const cl::CommandQueue& device_queue,
                            const cl::Buffer& generation_buffer,
                            const size_t crossover_point) const noexcept {
  try {
    cl::Kernel kernel(program, "perform_crossover");
    this->dump_opencl_build_log(program);

    kernel.setArg(0, generation_buffer);
    kernel.setArg(1, (int)crossover_point);
    kernel.setArg(2, (int)GENERATION_INDIVIDUAL_SIZE);

    device_queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                      cl::NDRange((int)(GENERATION_SIZE / 2)),
                                      cl::NullRange);

  } catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUILD_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
  }
}

float_t Gpu::compute_pearsons_correlation(
    const cl::Buffer& acc_buffer, const cl::Buffer& hr_values_diffs_buffer,
    const float_t hr_values_diff_squared_root, const size_t acc_buffer_size,
    const size_t acc_vector_len) const noexcept {

  const cl::Buffer working_buffer = cl::Buffer(
      this->device_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      acc_buffer_size, nullptr);

  const cl::Buffer acc_diff_squared_buffer = cl::Buffer(
      this->device_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      acc_buffer_size, nullptr);

  cl::CommandQueue queue = this->get_device_queue();
  try {

    // Calculate ACC average
    this->copy_float_buffer(queue, acc_buffer_size, acc_buffer, 0,
                            working_buffer, 0);

    float_t vec_sum = this->sum_vector(queue, acc_vector_len,
                                       working_buffer)[acc_vector_len - 1];
    /* logger.log_debug("ACC VALUES VEC SUM (OPENCL): " + std::to_string(vec_sum)); */

    float_t acc_avg = vec_sum / acc_vector_len;
    /* logger.log_debug("ACC VALUES AVG (OPENCL): " + std::to_string(acc_avg)); */

    // Prepare buffers
    this->copy_float_buffer(queue, acc_buffer_size, acc_buffer, 0,
                            working_buffer, 0);
    this->fill_float_buffer(queue, 0.0, acc_buffer_size,
                            acc_diff_squared_buffer);

    // Calculate correlation nominator and ACC squared diff
    this->calculate_correlation_acc_values(
        queue, working_buffer, acc_diff_squared_buffer, hr_values_diffs_buffer,
        acc_vector_len, acc_avg);

    float* result = this->sum_vector(queue, acc_vector_len, working_buffer);
    float_t nominator = result[acc_vector_len - 1];

    queue.enqueueUnmapMemObject(working_buffer, result);

    result = this->sum_vector(queue, acc_vector_len, acc_diff_squared_buffer);
    float_t acc_diff_squared_sum = result[acc_vector_len - 1];

    queue.enqueueUnmapMemObject(acc_diff_squared_buffer, result);

    /* logger.log_debug("NOMINATOR: " + std::to_string(nominator)); */
    /* logger.log_debug("ACC VALUES DIFF SUM: " + */
    /* std::to_string(acc_diff_squared_sum)); */

    float_t denominator =
        sqrtf(acc_diff_squared_sum) * hr_values_diff_squared_root;

    return nominator / denominator;
  } catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUFFER_ALLOC_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
    return 0.0;
  }

  return 0.0;
}

std::pair<std::vector<float_t>, std::vector<float_t>>
Gpu::compute_correlation_formula(
    std::vector<float_t>& acc_values, std::vector<float_t>& hr_values_diffs,
    const float_t hr_values_diff_squared_root) const noexcept {

  const cl::CommandQueue queue = this->device_queue;

  // Generate the initial generation
  std::vector<float_t> generation(GENERATION_SIZE * GENERATION_INDIVIDUAL_SIZE,
                                  0.0f);

  // CPU initial generation initialization
  std::random_device rd;
  std::mt19937 gen(rd());  // Standard Mersenne Twister

  std::uniform_real_distribution<float_t> operand_distr(0.0f, 0.5f);
  std::uniform_int_distribution<size_t> op_distr(1, 4);  // 1, 2, 3 or 4
  std::uniform_int_distribution<size_t> x_distr(0, 1);   // Either 0 or 1

  // Initialize the first generation
  for (size_t i = 0; i < GENERATION_SIZE; i += 2) {
    generation[i * GENERATION_INDIVIDUAL_SIZE] = ADD_FLOAT_REPRESENTATION;
    generation[i * GENERATION_INDIVIDUAL_SIZE + 1] = X_FLOAT_REPRESENTATION;
    generation[i * GENERATION_INDIVIDUAL_SIZE + 2] = operand_distr(gen);
  }

  for (size_t i = 1; i < GENERATION_SIZE; i += 2) {
    generation[i * GENERATION_INDIVIDUAL_SIZE] = SUB_FLOAT_REPRESENTATION;
    generation[i * GENERATION_INDIVIDUAL_SIZE + 1] = X_FLOAT_REPRESENTATION;
    generation[i * GENERATION_INDIVIDUAL_SIZE + 2] = operand_distr(gen);
  }

  const cl::Buffer generation_buffer =
      cl::Buffer{this->device_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                 GENERATION_SIZE * GENERATION_INDIVIDUAL_SIZE * sizeof(float_t),
                 generation.data()};

  // Create necessary buffers
  const cl::Buffer acc_buffer =
      cl::Buffer(this->device_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                 acc_values.size() * sizeof(float), acc_values.data());

  const cl::Buffer hr_values_diffs_buffer = cl::Buffer(
      this->device_context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      hr_values_diffs.size() * sizeof(float), hr_values_diffs.data());

  const float_t initial_correlation = this->compute_pearsons_correlation(
      acc_buffer, hr_values_diffs_buffer, hr_values_diff_squared_root,
      acc_values.size() * sizeof(float_t), acc_values.size());

  const cl::Buffer best_fit_buffer = cl::Buffer(
      this->device_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      GENERATION_INDIVIDUAL_SIZE * sizeof(float_t), nullptr);

  size_t crossover_idx =
      GENERATION_TREE_NODE_SIZE;  // Offset because the "root" node already initialized

  const float_t correlation_not_found = 2.0f;
  float_t best_found_correlation = correlation_not_found;

  const size_t generated_values_count = hr_values_diffs.size();
  const cl::Buffer generated_values_buffer = cl::Buffer(
      this->device_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      generated_values_count * sizeof(float_t), nullptr);

  const cl::Buffer best_fit_values_buffer = cl::Buffer(
      this->device_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      generated_values_count * sizeof(float_t), nullptr);

  // Begin the genetic generation
  for (size_t i = 0; i < GENERATION_ITERATION_COUNT; ++i) {
    const float_t prev_correlation = best_found_correlation;
    for (size_t j = 0; j < GENERATION_SIZE; ++j) {
      // Calculate the random matrix
      for (size_t k = crossover_idx; k < GENERATION_INDIVIDUAL_SIZE;
           k += GENERATION_TREE_NODE_SIZE) {
        size_t idx = j * GENERATION_INDIVIDUAL_SIZE + k;
        generation[idx] = (float_t)op_distr(gen);

        // Decide whenever use X or not
        generation[idx + 1] = x_distr(gen) == 0 ? X_FLOAT_REPRESENTATION
                                                : (float_t)operand_distr(gen);

        generation[idx + 2] = (float_t)operand_distr(gen);
      }

      // Perform the generation
      try {
        this->fill_float_buffer(queue, 0.0f,
                                generated_values_count * sizeof(float_t),
                                generated_values_buffer);

        cl::Kernel kernel(program, "generate_hr_values");
        this->dump_opencl_build_log(program);

        kernel.setArg(0, generation_buffer);
        kernel.setArg(1, j);
        kernel.setArg(2, (int)GENERATION_INDIVIDUAL_SIZE);
        kernel.setArg(3, acc_buffer);
        kernel.setArg(4, generated_values_buffer);

        queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                   cl::NDRange(generated_values_count),
                                   cl::NullRange);

        float_t new_correlation = this->compute_pearsons_correlation(
            generated_values_buffer, hr_values_diffs_buffer,
            hr_values_diff_squared_root,
            generated_values_count * sizeof(float_t), generated_values_count);

        if (std::fabs(initial_correlation - new_correlation) <
            std::fabs(initial_correlation -
                      best_found_correlation)) {  // Found a better fit
          logger.log_info(
              "Found correlation: " + std::to_string(new_correlation) + " in " +
              std::to_string(i + 1) + ". iteration");
          best_found_correlation = new_correlation;
          this->copy_float_buffer(
              queue, GENERATION_INDIVIDUAL_SIZE * sizeof(float_t),
              generation_buffer,
              (j * GENERATION_INDIVIDUAL_SIZE) * sizeof(float_t),
              best_fit_buffer, 0);

          this->copy_float_buffer(
              queue, generated_values_count * sizeof(float_t),
              generated_values_buffer, 0, best_fit_values_buffer, 0);
        }

        // Perform crossover
        /* this->perform_crossover(queue, generation_buffer, crossover_idx); */

      } catch (cl::Error& err) {
        logger.log_error(errors::ERRORS::OPENCL_BUILD_ERROR,
                         "(" + (std::string)err.what() + ", " +
                             std::to_string(err.err()) + ")");
      }
    }

    if (best_found_correlation ==
        prev_correlation) {  // Haven't found a better fit in this iteration
      crossover_idx = crossover_idx - GENERATION_TREE_NODE_SIZE == 0
                          ? crossover_idx
                          : crossover_idx - GENERATION_TREE_NODE_SIZE;
    } else {
      crossover_idx = crossover_idx + GENERATION_TREE_NODE_SIZE >=
                              GENERATION_INDIVIDUAL_SIZE - 1
                          ? crossover_idx
                          : crossover_idx + GENERATION_TREE_NODE_SIZE;
    }

    if (i > 0 && i % 10 == 0) {
      logger.log_info("Finished [" + std::to_string(i) + "/" +
                      std::to_string(GENERATION_ITERATION_COUNT) +
                      "] iterations");  // Realistically it's i-1 th iteration
    }
  }

  try {
    // Unmap memory segments
    std::vector<float_t> allocated =
        std::vector<float_t>(generated_values_count, 0.0f);

    queue.enqueueReadBuffer(best_fit_values_buffer, CL_TRUE, 0,
                            generated_values_count * sizeof(float_t),
                            allocated.data());

    std::vector<float_t> best_fit_values = allocated;

    queue.enqueueUnmapMemObject(generated_values_buffer, allocated.data());

    queue.enqueueReadBuffer(best_fit_buffer, CL_TRUE, 0,
                            GENERATION_INDIVIDUAL_SIZE * sizeof(float_t),
                            allocated.data());

    std::vector<float_t> best_fit = allocated;

    queue.enqueueUnmapMemObject(best_fit_buffer, allocated.data());

    logger.log_info("Best found correlation: " +
                    std::to_string(best_found_correlation));
    return std::pair<std::vector<float_t>, std::vector<float_t>>(
        best_fit_values, best_fit);
  }

  catch (cl::Error& err) {
    logger.log_error(
        errors::ERRORS::OPENCL_BUILD_ERROR,
        "(" + (std::string)err.what() + ", " + std::to_string(err.err()) + ")");
  }

  return std::pair<std::vector<float_t>, std::vector<float_t>>(
      std::vector<float_t>(), std::vector<float_t>());
}

void Gpu::dump_opencl_build_log(const cl::Program& program) const noexcept {
  cl_int buildErr = CL_SUCCESS;
  auto buildInfo = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&buildErr);

  if (!buildInfo.empty()) {
    for (auto& pair : buildInfo) {
      const auto msg = pair.second;
      if (!msg.empty()) {
        logger.log_error(errors::ERRORS::OPENCL_BUILD_ERROR, msg);
      }
    }
  }
}

}  // namespace opencl
