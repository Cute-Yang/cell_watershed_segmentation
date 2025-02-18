#include "image_proc/rank_filter.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "image_proc/rank_filter_helper.h"
#include "utils/logging.h"
#include <limits>
#include <vector>

namespace fish {
namespace image_proc {
namespace rank_filter {
namespace internal {

enum class FilterKernelKind : uint8_t { SmallKernel = 0, BigKernel = 1 };

enum class PixelValueKind : uint8_t { NormalValue = 0, OppositeValue = 1 };

FISH_ALWAYS_INLINE double clip_radius(double radius) {
    if (radius >= 1.5 && radius < 1.75) {
        radius = 1.75;
    } else if (radius >= 2.5 && radius < 2.85) {
        radius = 2.85;
    }
    return radius;
}

FISH_ALWAYS_INLINE int compute_k_radius(double radius) {
    radius       = clip_radius(radius);
    int square_r = static_cast<int>(radius * radius) + 1;
    int k_radius = static_cast<int>(std::sqrt(square_r + 1e-10));
    return k_radius;
}


FISH_ALWAYS_INLINE int compute_height(double radius) {
    int k_radius = compute_k_radius(radius);
    int k_height = 2 * k_radius + 1;
    return k_height;
}

FISH_ALWAYS_INLINE int compute_height(int k_radius) {
    return 2 * k_radius + 1;
}

FISH_ALWAYS_INLINE int compute_npoints(double radius) {
    radius       = clip_radius(radius);
    int square_r = static_cast<int>(radius * radius) + 1;
    int k_radius = static_cast<int>(std::sqrt(square_r + 1e-10));
    int n_points = 2 * k_radius + 1;
    for (int i = 1; i <= k_radius; ++i) {
        int dx = static_cast<int>(std::sqrt(square_r - i * i + 1e-10));
        n_points += 4 * dx + 2;
    }
    return n_points;
}

// FISH_ALWAYS_INLINE constexpr bool is_multi_step(FilterType filter_type) {
//     return (filter_type >= FilterType::OPEN && filter_type <= FilterType::TOP_HAT);
// }

FISH_ALWAYS_INLINE constexpr bool is_multi_step(FilterType filter_type) {
    if (filter_type == FilterType::OPEN || filter_type == FilterType::CLOSE ||
        filter_type == FilterType::TOP_HAT) {
        return true;
    }
    return false;
}



std::vector<int> compute_line_radius_offsets(double radius) {
    radius                    = clip_radius(radius);
    int              square_r = static_cast<int>(radius * radius) + 1;
    int              k_radius = static_cast<int>(std::sqrt(square_r + 1e-10));
    int              k_height = 2 * k_radius + 1;
    std::vector<int> line_radius_offsets;
    line_radius_offsets.resize(2 * k_height + 2);
    line_radius_offsets[2 * k_radius]     = -k_radius;
    line_radius_offsets[2 * k_radius + 1] = k_radius;
    int n_points                          = 2 * k_radius + 1;
    for (int y = 1; y <= k_radius; ++y) {
        int dx = static_cast<int>(std::sqrt(square_r - y * y + 1e-10));
        line_radius_offsets[2 * (k_radius - y)]     = -dx;
        line_radius_offsets[2 * (k_radius - y) + 1] = dx;
        line_radius_offsets[2 * (k_radius + y)]     = -dx;
        line_radius_offsets[2 * (k_radius + y) + 1] = dx;
        n_points += 4 * dx + 2;
    }
    line_radius_offsets[line_radius_offsets.size() - 2] = n_points;
    line_radius_offsets[line_radius_offsets.size() - 1] = k_radius;
    return line_radius_offsets;
}


std::vector<int> compute_cache_points(const std::vector<int>& line_radius_offsets, int cache_width,
                                      int k_radius) {
    int              k_height = 2 * k_radius + 1;
    std::vector<int> cache_points(k_height);
    // std::vector<int> cache_points(2 * k_height);
    cache_points.resize(2 * k_height);
    for (int i = 0; i < k_height; ++i) {
        cache_points[2 * i]     = i * cache_width + k_radius + line_radius_offsets[2 * i];
        cache_points[2 * i + 1] = i * cache_width + k_radius + line_radius_offsets[2 * i + 1];
    }
    return cache_points;
}

FISH_ALWAYS_INLINE void update_cache_points(std::vector<int>& cache_points, int delta_y,
                                            int cache_width, int cache_element_size) {
    for (size_t i = 0; i < cache_points.size(); ++i) {
        cache_points[i] = (cache_points[i] + cache_width * delta_y) % cache_element_size;
    }
}

template<class T, FilterKernelKind kernel_kind, typename = dtype_limit_t<T>>
void rank_filter_max_impl(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                          int cache_points_size, int y, int channel) {
    constexpr bool is_small_kernel = kernel_kind == FilterKernelKind::SmallKernel;
    constexpr T    type_min_value  = std::numeric_limits<T>::lowest();
    T              max_value{0};
    int            width = output_mat.get_width();
    if constexpr (is_small_kernel) {
        for (int x = 0; x < width; ++x) {
            max_value = compute_area_max_value<T>(
                cache, cache_points, cache_points_size, x, 0, type_min_value);
            output_mat(y, x, channel) = max_value;
        }
    } else {
        max_value =
            compute_area_max_value<T>(cache, cache_points, cache_points_size, 0, 0, type_min_value);
        output_mat(y, 0, channel) = max_value;
        for (int x = 1; x < width; ++x) {
            T new_max_value =
                compute_side_max_value<T, true>(cache, cache_points, cache_points_size, x);
            if (new_max_value >= max_value) {
                max_value = new_max_value;
            } else {
                T remove_max_value =
                    compute_side_max_value<T, false>(cache, cache_points, cache_points_size, x);
                if (remove_max_value >= max_value) {
                    max_value = compute_area_max_value<T>(
                        cache, cache_points, cache_points_size, x, 1, new_max_value);
                }
            }
            output_mat(y, x, channel) = max_value;
        }
    }
}



template<class T, FilterKernelKind kernel_kind, typename = dtype_limit_t<T>>
void rank_filter_min_impl(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                          int cache_points_size, int y, int channel) {
    constexpr bool is_small_kernel = (kernel_kind == FilterKernelKind::SmallKernel);
    T              min_value{0};
    constexpr T    type_max_value = std::numeric_limits<T>::max();
    int            width          = output_mat.get_width();
    if constexpr (is_small_kernel) {
        for (int x = 0; x < width; ++x) {
            min_value = compute_area_min_value<T>(
                cache, cache_points, cache_points_size, x, 0, type_max_value);
            output_mat(y, x, channel) = min_value;
        }
    } else {
        min_value =
            compute_area_min_value<T>(cache, cache_points, cache_points_size, 0, 0, type_max_value);
        output_mat(y, 0, channel) = min_value;
        for (int x = 1; x < width; ++x) {
            T new_min_value =
                compute_side_min_value<T, true>(cache, cache_points, cache_points_size, x);
            if (new_min_value <= min_value) {
                min_value = new_min_value;
            } else {
                T remove_min_value =
                    compute_side_min_value<T, false>(cache, cache_points, cache_points_size, x);
                if (remove_min_value <= min_value) {
                    min_value = compute_area_min_value<T>(
                        cache, cache_points, cache_points_size, x, 1, new_min_value);
                }
            }
            output_mat(y, x, channel) = min_value;
        }
    }
}

template<class T, FilterKernelKind kernel_kind, typename = dtype_limit_t<T>>
void rank_filter_mean_impl(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                           int cache_points_size, int y, int channel, int k_npoints) {
    constexpr bool is_small_kernel = (kernel_kind == FilterKernelKind::SmallKernel);
    double         k_npoints_f     = 1.0 / static_cast<double>(k_npoints);
    int            width           = output_mat.get_width();
    if constexpr (is_small_kernel) {
        for (int x = 0; x < width; ++x) {
            double mean_value =
                compute_area_sum_value<T>(cache, cache_points, cache_points_size, x) * k_npoints_f;
            output_mat.set_value_f(y, x, channel, mean_value);
        }
    } else {
        double sum_value = compute_area_sum_value<T>(cache, cache_points, cache_points_size, 0);
        output_mat.set_value_f(y, 0, channel, sum_value * k_npoints_f);
        for (int x = 1; x < width; ++x) {
            sum_value += compute_side_sum_value<T>(cache, cache_points, cache_points_size, x);
            output_mat.set_value_f(y, x, channel, sum_value * k_npoints_f);
        }
    }
}

template<class T, FilterKernelKind kernel_kind, typename = dtype_limit_t<T>>
void rank_filter_variance_impl(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                               int cache_points_size, int y, int channel, int k_npoints) {
    constexpr bool is_small_kernel = (kernel_kind == FilterKernelKind::SmallKernel);
    double         sum_values[2];
    double         k_npoints_f = 1.0 / static_cast<double>(k_npoints);
    int            width       = output_mat.get_width();
    if constexpr (is_small_kernel) {
        for (int x = 0; x < width; ++x) {
            compute_area_sum_value(cache, cache_points, cache_points_size, x, sum_values);
            double value =
                (sum_values[1] - sum_values[0] * sum_values[0] * k_npoints_f) * k_npoints_f;
            if (value < 0.0) {
                value = 0.0;
            }
            output_mat.set_value_f(y, x, channel, value);
        }
    } else {
        compute_area_sum_value(cache, cache_points, cache_points_size, 0, sum_values);
        double value = (sum_values[1] - sum_values[0] * sum_values[0] * k_npoints_f) * k_npoints_f;
        if (value < 0.0) [[unlikely]] {
            value = 0.0;
        }
        output_mat.set_value_f(y, 0, channel, value);
        for (int x = 1; x < width; ++x) {
            compute_side_sum_value(cache, cache_points, cache_points_size, x, sum_values);
            double value =
                (sum_values[1] - sum_values[0] * sum_values[0] * k_npoints_f) * k_npoints_f;
            if (value < 0.0) {
                value = 0.0;
            }
            output_mat.set_value_f(y, x, channel, value);
        }
    }
}

template<class T, typename = dtype_limit_t<T>>
void rank_filter_median_impl(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                             int cache_points_size, int y, int channel, T* temp_sort_buffer,
                             int k_npoints) {
    int width = output_mat.get_width();
    for (int x = 0; x < width; ++x) {
        T median_value = compute_area_median_value(
            cache, cache_points, cache_points_size, x, temp_sort_buffer, k_npoints);
        output_mat(y, x, channel) = median_value;
    }
}


template<class T, PixelValueKind value_kind, FilterKernelKind kernel_kind,
         typename = dtype_limit_t<T>>
void rank_filter_outlier_impl(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                              int cache_points_size, int y, int channel, int y_in_cache,
                              int cache_width, T threshold, int k_radius, T* temp_sort_buffer,
                              int k_npoints) {
    constexpr bool is_small_kernel = (kernel_kind == FilterKernelKind::SmallKernel);
    // while 0 is dark,255 is white which is reversed!
    constexpr bool is_opposite    = (value_kind == PixelValueKind::OppositeValue);
    int            width          = output_mat.get_width();
    constexpr T    type_max_value = std::numeric_limits<T>::max();
    constexpr T    type_min_value = std::numeric_limits<T>::lowest();
    int            cache_start    = y_in_cache * cache_width + k_radius;
    if constexpr (is_small_kernel) {
        if constexpr (is_opposite) {
            T min_value{0};
            for (int x = 0; x < width; ++x) {
                min_value = compute_area_min_value(
                    cache, cache_points, cache_points_size, x, 0, type_max_value);
                T value = cache[cache_start + x];
                if (value - min_value > threshold) {
                    T median_value = compute_area_median_value(
                        cache, cache_points, cache_points_size, x, temp_sort_buffer, k_npoints);
                    if (value > median_value) {
                        value = median_value;
                    }
                }
                output_mat(y, x, channel) = value;
            }
        } else {
            T max_value{0};
            for (int x = 0; x < width; ++x) {
                max_value = compute_area_max_value(
                    cache, cache_points, cache_points_size, x, 0, type_min_value);
                T value = cache[cache_start + x];
                if (max_value - value > threshold) {
                    T median_value = compute_area_median_value(
                        cache, cache_points, cache_points_size, x, temp_sort_buffer, k_npoints);
                    if (value < median_value) {
                        value = median_value;
                    }
                }
                output_mat(y, x, channel) = value;
            }
        }
    } else {
        if constexpr (is_opposite) {
            T min_value = 0;
            min_value   = compute_area_min_value(
                cache, cache_points, cache_points_size, 0, 0, type_max_value);
            T value = cache[cache_start];
            if (value - min_value > threshold) {
                T median_value = compute_area_median_value(
                    cache, cache_points, cache_points_size, 0, temp_sort_buffer, k_npoints);
                if (median_value > value) {
                    value = median_value;
                }
            }
            output_mat(y, 0, channel) = value;

            for (int x = 1; x < width; ++x) {
                T new_min_value =
                    compute_side_min_value<T, true>(cache, cache_points, cache_points_size, x);
                if (new_min_value <= min_value) {
                    min_value = new_min_value;
                } else {
                    T remove_min_value =
                        compute_side_min_value<T, false>(cache, cache_points, cache_points_size, x);
                    if (remove_min_value <= min_value) {
                        min_value = compute_area_min_value(
                            cache, cache_points, cache_points_size, x, 1, new_min_value);
                    }
                }
                T value = cache[cache_start + x];
                if (value - min_value > threshold) {
                    T median_value = compute_area_median_value(
                        cache, cache_points, cache_points_size, x, temp_sort_buffer, k_npoints);
                    if (value >= median_value) {
                        value = median_value;
                    }
                }
                output_mat(y, x, channel) = value;
            }
        } else {
            T max_value = 0;
            max_value   = compute_area_max_value(
                cache, cache_points, cache_points_size, 0, 0, type_min_value);
            T value = cache[cache_start];
            if (max_value - value > threshold) {
                T median_value = compute_area_median_value(
                    cache, cache_points, cache_points_size, 0, temp_sort_buffer, k_npoints);
                if (value < median_value) {
                    value = median_value;
                }
            }
            output_mat(y, 0, channel) = value;

            for (int x = 1; x < width; ++x) {
                T new_max_value =
                    compute_side_max_value<T, true>(cache, cache_points, cache_points_size, x);
                if (new_max_value >= max_value) {
                    max_value = new_max_value;
                } else {
                    T remove_max_value =
                        compute_side_max_value<T, false>(cache, cache_points, cache_points_size, x);
                    if (remove_max_value >= max_value) {
                        max_value = compute_area_max_value(
                            cache, cache_points, cache_points_size, x, 1, type_min_value);
                    }
                }
                T value = cache[cache_start + x];
                if (max_value - value > threshold) {
                    T median_value = compute_area_median_value(
                        cache, cache_points, cache_points_size, x, temp_sort_buffer, k_npoints);
                    if (value < median_value) {
                        value = median_value;
                    }
                }
                output_mat(y, x, channel) = value;
            }
        }
    }
}


template<class T, FilterType rank_filter_type, FilterKernelKind kernel_kind,
         typename = image_dtype_limit_t<T>>
void rank_filter_detail(ImageMat<T>& output_mat, const T* cache, const int* cache_points,
                        int cache_points_size, int y, int channel, int y_in_cache, int cache_width,
                        T threshold, int k_radius, T* temp_sort_buffer, int k_npoints,
                        bool is_opposite) {
    FISH_StaticAssert(is_implemented(rank_filter_type), "rank filter detail is not implemented!");
    if constexpr (rank_filter_type == FilterType::MAX) {
        rank_filter_max_impl<T, kernel_kind>(
            output_mat, cache, cache_points, cache_points_size, y, channel);
    } else if constexpr (rank_filter_type == FilterType::MIN) {
        rank_filter_min_impl<T, kernel_kind>(
            output_mat, cache, cache_points, cache_points_size, y, channel);
    } else if constexpr (rank_filter_type == FilterType::MEAN) {
        rank_filter_mean_impl<T, kernel_kind>(
            output_mat, cache, cache_points, cache_points_size, y, channel, k_npoints);
    } else if constexpr (rank_filter_type == FilterType::VARIANCE) {
        rank_filter_variance_impl<T, kernel_kind>(
            output_mat, cache, cache_points, cache_points_size, y, channel, k_npoints);
    } else if constexpr (rank_filter_type == FilterType::MEDIAN) {
        rank_filter_median_impl<T>(output_mat,
                                   cache,
                                   cache_points,
                                   cache_points_size,
                                   y,
                                   channel,
                                   temp_sort_buffer,
                                   k_npoints);
    } else if constexpr (rank_filter_type == FilterType::OUTLIER) {
        if (is_opposite) {
            rank_filter_outlier_impl<T, PixelValueKind::OppositeValue, kernel_kind>(
                output_mat,
                cache,
                cache_points,
                cache_points_size,
                y,
                channel,
                y_in_cache,
                cache_width,
                threshold,
                k_radius,
                temp_sort_buffer,
                k_npoints);
        } else {
            rank_filter_outlier_impl<T, PixelValueKind::NormalValue, kernel_kind>(output_mat,
                                                                                  cache,
                                                                                  cache_points,
                                                                                  cache_points_size,
                                                                                  y,
                                                                                  channel,
                                                                                  y_in_cache,
                                                                                  cache_width,
                                                                                  threshold,
                                                                                  k_radius,
                                                                                  temp_sort_buffer,
                                                                                  k_npoints);
        }
    } else {
        LOG_ERROR("rank detail for {} is not implemented!", FilterTypeStr[rank_filter_type]);
    }
}

template<class T, FilterType rank_filter_type, typename = image_dtype_limit_t<T>>
void _rank_filter_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat, T threshold,
                       bool is_opposite, int channel, double radius) {
    std::vector<int> line_radius_offsets = compute_line_radius_offsets(radius);
    int              height              = input_mat.get_height();
    int              width               = input_mat.get_width();
    int              k_radius            = compute_k_radius(radius);
    int              k_height            = compute_height(radius);
    int              k_npoints           = compute_npoints(radius);

    LOG_INFO("k_radius:{} k_height:{} k_npoints:{}", k_radius, k_height, k_npoints);

    std::vector<T> temp_sort_datas(k_npoints);
    T*             temp_sort_buffer = temp_sort_datas.data();

    int              cache_width = width + 2 * k_radius;
    std::vector<int> cache_points =
        compute_cache_points(line_radius_offsets, cache_width, k_radius);

    const int* cache_points_ptr  = cache_points.data();
    int        cache_points_size = cache_points.size();

    int            cache_height = k_height;
    std::vector<T> cache(cache_width * cache_height);
    int            cache_element_size = cache_height * cache_width;

    bool is_small_kernel = (k_radius < 2);

    int previous_y = k_height / 2 - cache_height;

    int y_end = (k_radius + 1) < height ? k_radius + 1 : height;

    int y_read = 0;

    T* cache_ptr = cache.data();
    for (; y_read < y_end; ++y_read) {
        int y_in_cache = y_read % cache_height;
        copy_to_cache(input_mat, cache_ptr, y_in_cache, cache_width, y_read, channel, k_radius);
    }

    // tail
    for (; y_read < k_radius + 1; ++y_read) {
        int y_in_cache     = y_read % cache_height;
        int pad_y_in_cache = (height - 1) % cache_height;

        int copy_start = cache_width * pad_y_in_cache;
        int copy_end   = cache_width * (pad_y_in_cache + 1);
        int copy_to    = cache_width * y_in_cache;
        std::copy(cache_ptr + copy_start, cache_ptr + copy_end, cache_ptr + copy_to);
    }

    // head
    for (int y_in_cache = cache_height - k_height / 2; y_in_cache < cache_height; ++y_in_cache) {
        int copy_start = 0;
        int copy_end   = cache_width;
        int copy_to    = y_in_cache * cache_width;
        std::copy(cache_ptr + copy_start, cache_ptr + copy_end, cache_ptr + copy_to);
    }

    // this stage we do not need to condisder do any padding!
    // y_end = height - k_radius;

    update_cache_points(cache_points, cache_height - k_height / 2, cache_width, cache_element_size);
    if (is_small_kernel) {
        rank_filter_detail<T, rank_filter_type, FilterKernelKind::SmallKernel>(output_mat,
                                                                               cache_ptr,
                                                                               cache_points_ptr,
                                                                               cache_points_size,
                                                                               0,
                                                                               channel,
                                                                               0,
                                                                               cache_width,
                                                                               threshold,
                                                                               k_radius,
                                                                               temp_sort_buffer,
                                                                               k_npoints,
                                                                               is_opposite);
    } else {
        rank_filter_detail<T, rank_filter_type, FilterKernelKind::BigKernel>(output_mat,
                                                                             cache_ptr,
                                                                             cache_points_ptr,
                                                                             cache_points_size,
                                                                             0,
                                                                             channel,
                                                                             0,
                                                                             cache_width,
                                                                             threshold,
                                                                             k_radius,
                                                                             temp_sort_buffer,
                                                                             k_npoints,
                                                                             is_opposite);
    }

    y_end        = height - k_radius;
    int y_center = 1;
    if (is_small_kernel) {
        for (; y_center < y_end; ++y_center) {
            update_cache_points(cache_points, 1, cache_width, cache_element_size);
            // the append row in cache!
            int copy_y_in_cache = (y_center + k_radius) % cache_height;
            // the center row in cache
            int y_in_cache = y_center % cache_height;
            copy_to_cache(input_mat,
                          cache_ptr,
                          copy_y_in_cache,
                          cache_width,
                          y_center + k_radius,
                          channel,
                          k_radius);
            rank_filter_detail<T, rank_filter_type, FilterKernelKind::SmallKernel>(
                output_mat,
                cache_ptr,
                cache_points_ptr,
                cache_points_size,
                y_center,
                channel,
                y_in_cache,
                cache_width,
                threshold,
                k_radius,
                temp_sort_buffer,
                k_npoints,
                is_opposite);
        }
        // using the last rows
        int pad_y_in_cache = (height - 1) % cache_height;
        int copy_start     = pad_y_in_cache * cache_width;
        int copy_end       = (pad_y_in_cache + 1) * cache_width;
        for (; y_center < height; ++y_center) {
            update_cache_points(cache_points, 1, cache_width, cache_element_size);
            int copy_y_in_cache = (y_center + k_radius) % cache_height;
            int copy_to         = copy_y_in_cache * cache_width;
            int y_in_cache      = y_center % cache_height;
            std::copy(cache_ptr + copy_start, cache_ptr + copy_end, cache_ptr + copy_to);
            rank_filter_detail<T, rank_filter_type, FilterKernelKind::SmallKernel>(
                output_mat,
                cache_ptr,
                cache_points_ptr,
                cache_points_size,
                y_center,
                channel,
                y_in_cache,
                cache_width,
                threshold,
                k_radius,
                temp_sort_buffer,
                k_npoints,
                is_opposite);
        }
    } else {
        for (; y_center < y_end; ++y_center) {
            update_cache_points(cache_points, 1, cache_width, cache_element_size);
            // the append row in cache!
            int copy_y_in_cache = (y_center + k_radius) % cache_height;
            // the center row in cache
            int y_in_cache = y_center % cache_height;
            copy_to_cache(input_mat,
                          cache_ptr,
                          copy_y_in_cache,
                          cache_width,
                          y_center + k_radius,
                          channel,
                          k_radius);
            rank_filter_detail<T, rank_filter_type, FilterKernelKind::BigKernel>(output_mat,
                                                                                 cache_ptr,
                                                                                 cache_points_ptr,
                                                                                 cache_points_size,
                                                                                 y_center,
                                                                                 channel,
                                                                                 y_in_cache,
                                                                                 cache_width,
                                                                                 threshold,
                                                                                 k_radius,
                                                                                 temp_sort_buffer,
                                                                                 k_npoints,
                                                                                 is_opposite);
        }
        // using the last rows
        int pad_y_in_cache = (height - 1) % cache_height;
        int copy_start     = pad_y_in_cache * cache_width;
        int copy_end       = (pad_y_in_cache + 1) * cache_width;
        for (; y_center < height; ++y_center) {
            update_cache_points(cache_points, 1, cache_width, cache_element_size);
            int copy_y_in_cache = (y_center + k_radius) % cache_height;
            int copy_to         = copy_y_in_cache * cache_width;
            int y_in_cache      = y_center % cache_height;
            std::copy(cache_ptr + copy_start, cache_ptr + copy_end, cache_ptr + copy_to);
            rank_filter_detail<T, rank_filter_type, FilterKernelKind::BigKernel>(output_mat,
                                                                                 cache_ptr,
                                                                                 cache_points_ptr,
                                                                                 cache_points_size,
                                                                                 y_center,
                                                                                 channel,
                                                                                 y_in_cache,
                                                                                 cache_width,
                                                                                 threshold,
                                                                                 k_radius,
                                                                                 temp_sort_buffer,
                                                                                 k_npoints,
                                                                                 is_opposite);
        }
    }
}

template<class T, FilterType rank_filter_type, typename = image_dtype_limit_t<T>>
void rank_filter_impl(const ImageMat<T>& input_mat, ImageMat<T>& output_mat, T threshold,
                      bool light_background, bool substract, bool is_inverted_lut,
                      OutlierValueKind outlier_kind, int channel, double radius) {
    bool is_opposite{false};
    if constexpr (rank_filter_type == FilterType::OUTLIER) {
        is_opposite = (is_inverted_lut == (outlier_kind == OutlierValueKind::DarkOutlier));
    }

    if constexpr (rank_filter_type == FilterType::TOP_HAT) {
        is_opposite =
            (is_inverted_lut && !light_background) || (!is_inverted_lut && light_background);
    }

    int            height     = input_mat.get_height();
    int            width      = input_mat.get_width();
    constexpr bool multi_step = is_multi_step(rank_filter_type);

    if constexpr (multi_step) {
        if (is_opposite) {
            _rank_filter_impl<T, FilterType::MIN>(
                input_mat, output_mat, threshold, is_opposite, channel, radius);
        } else {
            _rank_filter_impl<T, FilterType::MAX>(
                input_mat, output_mat, threshold, is_opposite, channel, radius);
        }
    } else {
        _rank_filter_impl<T, rank_filter_type>(
            input_mat, output_mat, threshold, is_opposite, channel, radius);
    }

    if constexpr (multi_step) {
        if (is_opposite) {
            _rank_filter_impl<T, FilterType::MAX>(
                input_mat, output_mat, threshold, is_opposite, channel, radius);
        } else {
            _rank_filter_impl<T, FilterType::MIN>(
                input_mat, output_mat, threshold, is_opposite, channel, radius);
        }

        if constexpr (rank_filter_type == FilterType::TOP_HAT) {
            if (substract) {
                T offset = 0;
                if constexpr (IntegerTypeRequire<T>::value) {
                    // 255 or 65536
                    offset = std::numeric_limits<T>::max();
                }
                int      data_size  = input_mat.get_element_num();
                const T* input_ptr  = input_mat.get_data_ptr();
                T*       output_ptr = output_mat.get_data_ptr();
                for (int i = 0; i < data_size; ++i) {
                    // make the data not over flow! 0 ~ max_value
                    T value       = input_ptr[i] - output_ptr[i] + offset;
                    output_ptr[i] = value;
                }
            }
        }
    }
}

}   // namespace internal
template<class T, typename>
Status::ErrorCode rank_filter(const ImageMat<T>& input_mat, ImageMat<T>& output_mat,
                              FilterType rank_filter_type, double radius) {
    if (radius <= 0.0) {
        LOG_ERROR("got invalid radius {} for rank filter...", radius);
        return Status::ErrorCode::InvalidRankFilterRadius;
    }

    if (input_mat.empty()) {
        LOG_ERROR("input mat is invalid!");
        return Status::ErrorCode::InvalidMatShape;
    }

    if (output_mat.get_layout() != input_mat.get_layout()) {
        LOG_ERROR("input and output mat have different layout!");
        return Status::ErrorCode::MatLayoutMismath;
    }

    int height   = input_mat.get_height();
    int width    = input_mat.get_width();
    int channels = input_mat.get_channels();

    if (!input_mat.compare_shape(output_mat)) {
        LOG_INFO("the output mat have different shape with input mat,we will reshape it!");
        output_mat.resize(height, width, channels, true);
    }

    const char* rank_filter_type_str = FilterTypeStr[rank_filter_type];
    LOG_INFO("apply rank filter with type {}", rank_filter_type_str);

    constexpr bool             light_backgroup = false;
    constexpr bool             substract       = true;
    constexpr bool             is_inverted_lut = false;
    constexpr OutlierValueKind outlier_kind    = OutlierValueKind::WhiteOutlier;
    constexpr T                threshold       = 50;
    for (int channel = 0; channel < channels; ++channel) {
        if (rank_filter_type == FilterType::MAX) {
            internal::rank_filter_impl<T, FilterType::MAX>(input_mat,
                                                           output_mat,
                                                           threshold,
                                                           light_backgroup,
                                                           substract,
                                                           is_inverted_lut,
                                                           outlier_kind,
                                                           channel,
                                                           radius);
        } else if (rank_filter_type == FilterType::MIN) {
            internal::rank_filter_impl<T, FilterType::MIN>(input_mat,
                                                           output_mat,
                                                           threshold,
                                                           light_backgroup,
                                                           substract,
                                                           is_inverted_lut,
                                                           outlier_kind,
                                                           channel,
                                                           radius);
        } else if (rank_filter_type == FilterType::MEAN) {
            internal::rank_filter_impl<T, FilterType::MEAN>(input_mat,
                                                            output_mat,
                                                            threshold,
                                                            light_backgroup,
                                                            substract,
                                                            is_inverted_lut,
                                                            outlier_kind,
                                                            channel,
                                                            radius);
        } else if (rank_filter_type == FilterType::VARIANCE) {
            internal::rank_filter_impl<T, FilterType::VARIANCE>(input_mat,
                                                                output_mat,
                                                                threshold,
                                                                light_backgroup,
                                                                substract,
                                                                is_inverted_lut,
                                                                outlier_kind,
                                                                channel,
                                                                radius);
        } else if (rank_filter_type == FilterType::MEDIAN) {
            internal::rank_filter_impl<T, FilterType::MEDIAN>(input_mat,
                                                              output_mat,
                                                              threshold,
                                                              light_backgroup,
                                                              substract,
                                                              is_inverted_lut,
                                                              outlier_kind,
                                                              channel,
                                                              radius);
        } else if (rank_filter_type == FilterType::OUTLIER) {
            internal::rank_filter_impl<T, FilterType::OUTLIER>(input_mat,
                                                               output_mat,
                                                               threshold,
                                                               light_backgroup,
                                                               substract,
                                                               is_inverted_lut,
                                                               outlier_kind,
                                                               channel,
                                                               radius);
        } else if (rank_filter_type == FilterType::TOP_HAT) {
            internal::rank_filter_impl<T, FilterType::TOP_HAT>(input_mat,
                                                               output_mat,
                                                               threshold,
                                                               light_backgroup,
                                                               substract,
                                                               is_inverted_lut,
                                                               outlier_kind,
                                                               channel,
                                                               radius);
        } else if (rank_filter_type == FilterType::OPEN) {
            internal::rank_filter_impl<T, FilterType::OPEN>(input_mat,
                                                            output_mat,
                                                            threshold,
                                                            light_backgroup,
                                                            substract,
                                                            is_inverted_lut,
                                                            outlier_kind,
                                                            channel,
                                                            radius);
        } else if (rank_filter_type == FilterType::CLOSE) {
            internal::rank_filter_impl<T, FilterType::TOP_HAT>(input_mat,
                                                               output_mat,
                                                               threshold,
                                                               light_backgroup,
                                                               substract,
                                                               is_inverted_lut,
                                                               outlier_kind,
                                                               channel,
                                                               radius);
        } else {
            LOG_WARN("sorry,we do not support rank_filter with type {}",
                     FilterTypeStr[rank_filter_type]);
            return Status::ErrorCode::UnsupportedRankFilterType;
        }
    }
    return Status::ErrorCode::Ok;
}

template Status::ErrorCode rank_filter<uint8_t>(const ImageMat<uint8_t>& input_mat,
                                                ImageMat<uint8_t>&       output_mat,
                                                FilterType rank_filter_type, double radius);


template Status::ErrorCode rank_filter<uint16_t>(const ImageMat<uint16_t>& input_mat,
                                                 ImageMat<uint16_t>&       output_mat,
                                                 FilterType rank_filter_type, double radius);


template Status::ErrorCode rank_filter<float>(const ImageMat<float>& input_mat,
                                              ImageMat<float>&       output_mat,
                                              FilterType rank_filter_type, double radius);

}   // namespace rank_filter
}   // namespace image_proc
}   // namespace fish