#include "include/avx.hpp"
#include <numeric>
#include <optional>
#include "include/constants.hpp"

namespace avx {

Logging::Logger& logger = Logging::Logger::get_instance();

std::optional<float_t> vector_sum_avx2(
    const std::vector<float_t>& values) noexcept {
  float_t sum1 = 0.0f;
  float_t sum2 = 0.0f;
  float_t sum3 = 0.0f;
  float_t sum4 = 0.0f;
  float_t sum5 = 0.0f;
  float_t sum6 = 0.0f;
  float_t sum7 = 0.0f;
  float_t sum8 = 0.0f;

  if (values.empty()) {
    logger.log_error(errors::ERRORS::PARAMETER_WAS_EMPTY);
    return std::nullopt;
  }

  if (values.size() < MIN_VEC_SIZE_AVX2) {
    logger.log_error(errors::ERRORS::INVALID_AVX_VECTOR_SIZE);
    return std::nullopt;
  }

  size_t diff = values.size() % FLOATS_PER_AVX2;
  if (diff > 0) {
    logger.log_error(errors::ERRORS::INVALID_AVX_VECTOR_SIZE,
                     "(Padding difference: " + std::to_string(diff) + ")");
    return std::nullopt;
  }

  // Loop unwrapping just to make sure we make the best effort for vectorization
  for (size_t i = 0; i < values.size(); i += FLOATS_PER_AVX2) {
    sum1 += values[i];
    sum2 += values[i + 1];
    sum3 += values[i + 2];
    sum4 += values[i + 3];
    sum5 += values[i + 4];
    sum6 += values[i + 5];
    sum7 += values[i + 6];
    sum8 += values[i + 7];
  }

  return sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7 + sum8;
}

std::optional<float_t> calculate_pearsons_correlation(
    const std::vector<float_t>& acc_values,
    const std::vector<float_t>& hr_values_diffs,
    const float_t hr_diff_square_root) noexcept {
  float_t correlation = 0;

  if (acc_values.empty() || hr_values_diffs.empty()) {
    logger.log_error(
        errors::ERRORS::PARAMETER_WAS_EMPTY,
        "(input vector of values for the Pearson's correlation calculation)");
    return std::nullopt;
  }

  int32_t diff = acc_values.size() - hr_values_diffs.size();
  if (diff != 0) {
    logger.log_error(
        errors::ERRORS::INVALID_ARGUMENT,
        "(vectors for Pearson's correlation are not the same size (diff = " +
            std::to_string(diff) + ")");
    return std::nullopt;
  }

  // First pass - get averages
  std::optional<float_t> tmp_sum = vector_sum_avx2(acc_values);
  if (tmp_sum == std::nullopt) {
    return std::nullopt;
  }

  /* logger.log_debug("ACC VALUES VEC SUM (AVX): " + */
  /*                  std::to_string(tmp_sum.value())); */

  float_t avg_acc = tmp_sum.value() / acc_values.size();
  /* logger.log_debug("ACC VALUES AVG (AVX): " + std::to_string(avg_acc)); */

  std::vector<float_t> acc_diff_squared(acc_values.size());
  std::vector<float_t> nominator(acc_values.size());
  for (size_t i = 0; i < acc_values.size(); ++i) {
    nominator[i] = (acc_values[i] - avg_acc) * (hr_values_diffs[i]);
    acc_diff_squared[i] = (acc_values[i] - avg_acc) * (acc_values[i] - avg_acc);
  }

  std::optional tmp = vector_sum_avx2(nominator);
  if (tmp == std::nullopt) {
    return std::nullopt;
  }

  correlation = tmp.value();
  /* logger.log_debug("Nominator (AVX): " + std::to_string(correlation)); */
  tmp = vector_sum_avx2(acc_diff_squared);
  if (tmp == std::nullopt) {
    return std::nullopt;
  }

  /* logger.log_debug("ACC DIFF SQUARED SUM (AVX): " + */
  /*                  std::to_string(tmp.value())); */
  correlation /= sqrtf(tmp.value()) * hr_diff_square_root;

  return correlation;
}

}  // namespace avx
