#pragma once
#include "common/fishdef.h"
#include "core/mat.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace fish {
namespace image_proc {
namespace statistic {
using namespace fish::core::mat;

template<class T> struct SumValueTypeTraits { using type = void; };

// with sign!
template<> struct SumValueTypeTraits<int8_t> { using type = int32_t; };

template<> struct SumValueTypeTraits<uint8_t> { using type = uint32_t; };

template<> struct SumValueTypeTraits<int16_t> { using type = int32_t; };

template<> struct SumValueTypeTraits<uint16_t> { using type = uint32_t; };

template<> struct SumValueTypeTraits<int32_t> { using type = int64_t; };

template<> struct SumValueTypeTraits<uint32_t> { using type = uint64_t; };


template<> struct SumValueTypeTraits<int64_t> { using type = int64_t; };

template<> struct SumValueTypeTraits<float> { using type = double; };

template<> struct SumValueTypeTraits<double> { using type = double; };


// the pixel value is 255,use 256 to restore the historgram!s
using histogram_t = std::array<int, 256>;
FISH_ALWAYS_INLINE histogram_t get_image_histogram(const ImageMat<uint8_t>& image) {
    int                  height    = image.get_height();
    int                  width     = image.get_width();
    const unsigned char* image_ptr = image.get_data_ptr();
    histogram_t          histogram;
    std::fill(histogram.begin(), histogram.end(), 0);
    size_t data_size = static_cast<size_t>(height) * static_cast<size_t>(width);
    for (size_t i = 0; i < height * width; ++i) {
        ++histogram[image_ptr[i]];
    }
    return histogram;
}

// should add a trait to got the sum value type!
template<class T>
float compute_roi_mean(const ImageMat<T>& image, const ImageMat<uint8_t>& mask, int original_x,
                       int original_y) {

    // just for debug!
    // auto   mask_view = mask.get_image_view();
    int    rect_h = mask.get_height();
    int    rect_w = mask.get_width();
    double sum    = 0;
    int    count  = 0;
    for (int y = 0; y < rect_h; ++y) {
        for (int x = 0; x < rect_w; ++x) {
            if (mask(y, x) != 0) {
                T value = image(y + original_y, x + original_x);
                sum += value;
                ++count;
            }
        }
    }
    float mean_value = sum / (static_cast<double>(count) + 1e-9);
    return mean_value;
}

struct StatResult {
    double sum;
    int    pixel_count;
};
template<class T>
StatResult compute_roi_stat(const ImageMat<T>& image, ImageMat<uint8_t>& mask, int x1, int y1) {
    int    mask_h = mask.get_height();
    int    mask_w = mask.get_width();
    double sum    = 0.0;
    int    count  = 0;
    for (int y = 0; y < mask_h; ++y) {
        for (int x = 0; x < mask_w; ++x) {
            if (mask(y, x) != 0) {
                sum += image(y + y1, x + x1);
                ++count;
            }
        }
    }
    StatResult stat;
    stat.sum         = sum;
    stat.pixel_count = count;
    return stat;
}
}   // namespace statistic
}   // namespace image_proc
}   // namespace fish