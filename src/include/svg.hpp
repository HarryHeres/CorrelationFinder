#pragma once

#include <string>
#include <vector>
#include "math.h"

namespace svg {

/**
   * Plot the correlaction vectors (initial and generated) into an SVG output
   *
   * @param filepath Path to the file, including the filename
   * @param generated_values Vector of the generated values
   * @param initial_values Vector of the initial values
   * @param correlation_formula Correlation formula that was found in a string format
   */
void plot_correlation_values(const std::string& filepath,
                             const std::vector<float_t>& generated_values,
                             const std::vector<float_t>& initial_values,
                             const std::string& correlation_formula) noexcept;
}  // namespace svg
