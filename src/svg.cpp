#include "include/svg.hpp"
#include <fstream>
#include "include/constants.hpp"
#include "include/logger.hpp"

namespace svg {
Logging::Logger& logger = Logging::Logger::get_instance();

void plot_correlation_values(const std::string& filename,
                             const std::vector<float_t>& generated_values,
                             const std::vector<float_t>& initial_values,
                             const std::string& correlation_formula) noexcept {
  std::ofstream plot_file;

  plot_file.open(filename);

  if (!plot_file.is_open()) {
    logger.log_error(errors::ERRORS::COULD_NOT_OPEN_FILE_HANDLE,
                     "Could open file for writing: " + filename +
                         ".\nPlot will not be created");
    return;
  }

  const size_t x_offset = 50;
  const size_t y_offset = 100;
  const size_t correlation_text_offset = 1500;
  const size_t downscale = 10;
  const size_t plot_height = 2000;
  const size_t plot_width = initial_values.size() / downscale;

  // Header
  plot_file << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>"
            << std::endl;
  plot_file << "<svg width=\"" << plot_width + x_offset << "\" height=\""
            << plot_height + y_offset
            << "\" xmlns=\"http://www.w3.org/2000/svg\">" << std::endl;

  // Correlation formula
  plot_file << "<text x=\"" << correlation_text_offset << "\" y=\""
            << y_offset / 3 << "\" font-size=\"20\" text-anchor=\"middle\">"
            << "Correlation formula: " << correlation_formula << "</text>"
            << std::endl;

  // Plot beginning
  plot_file << "<text x=\"" << x_offset / 2 << "\" y=\""
            << plot_height / 2 + y_offset
            << "\" font-size=\"20\" text-anchor=\"middle\">0</text>"
            << std::endl;

  // X axis
  plot_file << "<line x1=\"" << x_offset << "\" y1=\""
            << plot_height / 2 + y_offset << "\" x2=\"" << plot_width + x_offset
            << "\" y2=\"" << plot_height / 2 + y_offset
            << "\" "
               "stroke=\"black\"/>"
            << std::endl;

  // Y axis
  plot_file << "<line x1=\"" << x_offset << "\" y1=\"" << y_offset << "\" x2=\""
            << x_offset << "\" y2=\"" << plot_height / 2 + y_offset
            << "\" "
               "stroke=\"black\"/>"
            << std::endl;

  const size_t circle_size = 3;
  for (size_t i = 0; i < initial_values.size() / downscale; i += downscale) {
    plot_file << "<circle cx=\"" << x_offset + (i / downscale) - circle_size / 2
              << "\" cy=\""
              << y_offset + size_t(plot_height / 2) -
                     fabs(generated_values[i] * HR_MAX_VALUE)
              << "\" r=\"" << circle_size << "\" fill=\"red\"/>" << std::endl;
    plot_file << "<circle cx=\"" << x_offset + (i / downscale) - circle_size / 2
              << "\" cy=\""
              << y_offset + size_t(plot_height / 2) -
                     (initial_values[i] * HR_MAX_VALUE)
              << "\" r=\"" << circle_size << "\" fill=\"blue\"/>" << std::endl;
  }

  // Closing out
  plot_file << "</svg>" << std::endl;
  plot_file.close();
}
}  // namespace svg
