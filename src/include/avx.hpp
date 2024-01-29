#pragma once

#include <math.h>
#include <optional>
#include <vector>
#include "logger.hpp"

namespace avx {

/**
   * Calculate sum over a vector of @type float_t using AVX2
   *
   * @param values Vector to be "summed". Needs to be at least twice the length of the AVX2 register size. 
   * IMPORTANT: The vector needs to be padded correctly for AVX2
   *
   * @return Pair of number of elements processed (due to possible padding) and sum of all of the elements of the input vector or std::nullopt, if any requirements were not fulfilled.
   */
std::optional<float_t> vector_sum_avx2(
    const std::vector<float_t>& values) noexcept;

/**
   * Calculate the Pearson's correlation coefficient of the ACC and HR measured values
   *
   * @param acc_values Vector of measured ACC values (only one axis)
   * @param hr_values Vector of measure HR value_string
   *
   * @return Pearson's Correlation coefficient between the two input measurements or std::nullopt if some of the requirements were not met
   */
std::optional<float_t> calculate_pearsons_correlation(
    const std::vector<float_t>& acc_values,
    const std::vector<float_t>& hr_values_diffs,
    const float_t hr_diff_square_root) noexcept;
}  // namespace avx
