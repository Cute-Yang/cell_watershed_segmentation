#include "segmentation/watershed_cell_detection.h"
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "core/mat_ops.h"
#include "image_proc/convolution.h"
#include "image_proc/distance_transform.h"
#include "image_proc/fill_mask.h"
#include "image_proc/find_contour.h"
#include "image_proc/find_maximum.h"
#include "image_proc/guassian_blur.h"
#include "image_proc/img_stat.h"
#include "image_proc/neighbor_filter.h"
#include "image_proc/polygon.h"
#include "image_proc/rank_filter.h"
#include "image_proc/roi_labeling.h"
#include "segmentation/estimate_backgroud.h"
#include "segmentation/morphological_transform.h"
#include "segmentation/shape_simplifier.h"
#include "segmentation/tile_util.h"
#include "segmentation/watershed.h"
#include "utils/logging.h"
#include "utils/thread_pool.h"
#include <cmath>
#include <future>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>
// just for write the image to opencv,do nothing...

namespace fish {
namespace segmentation {
namespace watershed_cell_detection {
using namespace fish::core;
using namespace fish::core::mat;
using namespace fish::image_proc::rank_filter;
using namespace fish::image_proc::guassian_blur;
using namespace fish::image_proc::convolution;
using namespace fish::image_proc::find_maximum;
using namespace fish::core::mat_ops;
using namespace fish::segmentation::estimate;
using namespace fish::segmentation::morphological;
using namespace fish::image_proc::roi_labeling;
using namespace fish::segmentation::watershed;
using namespace fish::image_proc::contour;
using namespace fish::image_proc::statistic;
using namespace fish::image_proc::fill_mask;
using namespace fish::image_proc::neighbor_filter;
using namespace fish::image_proc::distance_transform;
using namespace fish::segmentation::shape_simplier;
using namespace fish::utils::parallel;

constexpr uint32_t max_parallel_size = 12;
static uint32_t    parallel_size = std::min(max_parallel_size, std::thread::hardware_concurrency());
static ThreadPool  pool(parallel_size);
/**
 * @brief
 *
 * @param original_image the mat of image!
 * @param detect_channel,the channel you want to use
 * @param Hematoxylin_channel
 * @param DAB_channel
 * @param background_radius
 * @param max_background
 * @param median_radius
 * @param sigma
 * @param threshold
 * @param min_area
 * @param max_area
 * @param merge_all
 * @param watershed_postprocess
 * @param exclude_DAB
 * @param cell_expansion
 * @param smooth_boundaries
 * @param make_measurements
 * @param background_by_reconstruction
 * @param downsample_factor
 * @return void
 */
namespace internal {

enum class MemoryUnitKind {
    Byte  = 0,
    Kbyte = 0,
    // we useually use it!
    Mbyte = 1,
    Gbyte = 2,
    Tbyte = 3
};

template<MemoryUnitKind unit_kind> struct UnitDivdeValueTraits {
    static constexpr float value = 1.0f;
};

template<> struct UnitDivdeValueTraits<MemoryUnitKind::Kbyte> {
    static constexpr float value = 1024.0f;
};

template<> struct UnitDivdeValueTraits<MemoryUnitKind::Mbyte> {
    static constexpr float value = 1024.0f * 1024.0f;
};
template<> struct UnitDivdeValueTraits<MemoryUnitKind::Gbyte> {
    static constexpr float value = 1024.0f * 1024.0f * 1024.0f;
};

template<> struct UnitDivdeValueTraits<MemoryUnitKind::Tbyte> {
    static constexpr float value = 1024.0f * 1024.0f * 1024.0f * 1024.0f;
};



template<class T, MemoryUnitKind unit_kind = MemoryUnitKind::Mbyte,
         typename = image_dtype_limit<T>::type>
float compute_mat_memory_size(const ImageMat<T>& image) {
    size_t          element_nbytes = image.get_nbytes();
    constexpr float unit_divide    = UnitDivdeValueTraits<unit_kind>::value;
    return static_cast<float>(element_nbytes) / unit_divide;
}


// return the max of requested size and real size as perfered size!
FISH_ALWAYS_INLINE double compute_averaged_pixel_size_microns(double pixel_size_microns_h,
                                                              double pixel_size_microns_w) {
    return (pixel_size_microns_h + pixel_size_microns_w) * 0.5;
}

double compute_preferred_pixel_size_microns(double pixel_size_microns_h,
                                            double pixel_size_microns_w,
                                            double requested_pixel_size) {
    double averaged_pixel_size =
        compute_averaged_pixel_size_microns(pixel_size_microns_h, pixel_size_microns_w);
    if (requested_pixel_size < 0.0) {
        LOG_INFO("the given requested pixel size microns < 0.0 which is unexpected!so we will "
                 "multiply -1.0");
        requested_pixel_size = averaged_pixel_size * (-requested_pixel_size);
    }
    // use the max value as final prefered pixel size!
    requested_pixel_size = FISH_MAX(requested_pixel_size, averaged_pixel_size);
    return requested_pixel_size;
}

double compute_preferred_donwsample_macrons(double pixel_size_microns,
                                            double requested_pixel_size_microns, bool apply_log2) {
    double downsample_factor;
    if (apply_log2) {
        downsample_factor =
            std::pow(2.0,
                     std::round(std::log(requested_pixel_size_microns / pixel_size_microns) /
                                std::log(2.0)));
    } else {
        downsample_factor = requested_pixel_size_microns / pixel_size_microns;
    }
    return downsample_factor;
}

double compute_downsample_factor(double pixel_size_microns_h, double pixel_size_microns_w,
                                 double preferred_pixel_size_microns, bool apply_log2) {
    double pixel_size_microns =
        compute_averaged_pixel_size_microns(pixel_size_microns_h, pixel_size_microns_w);
    double downsample_factor = compute_preferred_donwsample_macrons(
        pixel_size_microns, preferred_pixel_size_microns, false);
    if (downsample_factor < 1.0) {
        downsample_factor = 1.0;
    }
    return downsample_factor;
}

template<class T1, class T2, typename = dtype_limit<T1>, typename = dtype_limit<T2>>
ImageMat<T2> convert_mat_dtype(const ImageMat<T1>& mat) {
    if constexpr (std::is_same_v<T1, T2>) {
        return mat;
    }
    int          height   = mat.get_height();
    int          width    = mat.get_width();
    int          channels = mat.get_channels();
    ImageMat<T2> converted_mat(height, width, channels, MatMemLayout::LayoutRight);
    T2*          converted_mat_ptr = converted_mat.get_data_ptr();
    const T1*    mat_ptr           = mat.get_data_ptr();
    size_t       data_size         = height * width * channels;
    for (size_t i = 0; i < data_size; ++i) {
        converted_mat_ptr[i] = static_cast<T2>(mat_ptr[i]);
    }
    return converted_mat;
}



void clip_polygons(std::vector<PolygonTypef32>& polygons, int width, int height) {
    constexpr float xmin = 0.0f;
    float           xmax = static_cast<float>(width - 1);

    constexpr float ymin = 0.0f;
    float           ymax = static_cast<float>(height - 1);

    size_t polygon_size = polygons.size();
    for (size_t i = 0; i < polygon_size; ++i) {
        auto&  polygon     = polygons[i];
        size_t vertex_size = polygon.size();
        for (size_t j = 0; j < vertex_size; ++j) {
            if (polygon[j].x < xmin) {
                LOG_INFO("find small x {}", polygon[j].x);
                polygon[j].x = xmin;
            } else if (polygon[j].x > xmax) {
                polygon[j].x = xmax;
            }

            if (polygon[j].y < ymin) {
                polygon[j].y = ymin;
            } else if (polygon[j].y > ymax) {
                polygon[j].y = ymax;
            }
        }
    }
}

Status::ErrorCode cell_detection_impl(
    const ImageMat<float>& original_image, int detect_channel, int Hematoxylin_channel,
    int DAB_channel, double background_radius, double max_background, double median_radius,
    double sigma, double threshold, double min_area, double max_area, bool merge_all,
    bool watershed_postprocess, bool exclude_DAB, double cell_expansion, bool smooth_boundaries,
    bool make_measurements, bool background_by_reconstruction, bool refine_boundary,
    double downsample_factor, std::vector<PolygonTypef32>& out_nuclei_rois,
    std::vector<PolygonTypef32>& out_cell_rois) {
    if (original_image.empty()) {
        LOG_ERROR("the origianl image is empty,so noting to do....");
        return Status::ErrorCode::InvalidMatShape;
    }
    int height   = original_image.get_height();
    int width    = original_image.get_width();
    int channels = original_image.get_channels();
    if (detect_channel < 0 || detect_channel >= channels) {
        LOG_ERROR("out original image has channels {},but you speicfy channel {} to detecit,this "
                  "is invalid!",
                  channels,
                  detect_channel);
        return Status::ErrorCode::InvalidMatChannle;
    }
    LOG_INFO("we will use channel {} to  detect...", detect_channel);
    // for the fill func,we pass the mask to reuse the memory!
    // how to reuse the memory!
    LOG_INFO("generate two placeholder to reuse the memory...");
    ImageMat<uint8_t> image_u8_placeholder(height, width, 1, MatMemLayout::LayoutRight);
    ImageMat<float>   image_f32_placeholder(height, width, 1, MatMemLayout::LayoutRight);
    // for memory report!
    float image_f32_memory_size =
        static_cast<float>(height * width * sizeof(float)) / 1024.0f / 1024.0f;
    float image_u8_memory_size = static_cast<float>(height * width) / 1024.0f / 1024.0f;
    // the background mask need to allocate a buffer!
    ImageMat<uint8_t> background_mask;
    // the image to compute the measurements!
    ImageMat<float> measurement_image;

    // avoid large image out of range!
    size_t data_size = static_cast<size_t>(height) * static_cast<size_t>(width);
    // apply copy from detect_image!
    LOG_INFO("copying the detect channel data to a new mat!");
    ImageMat<float> transform_image(height, width, 1, MatMemLayout::LayoutRight);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            transform_image(y, x) = original_image(y, x, detect_channel);
        }
    }

    Status::ErrorCode invoke_status;
    if (median_radius > 0) {
        LOG_INFO("apply median filter with radius {}", median_radius);
        // attention,here our func can invoke inplace!
        // add nodiscard to
        invoke_status =
            rank_filter(transform_image, transform_image, FilterType::MEDIAN, median_radius);
        if (invoke_status != Status::ErrorCode::Ok) {
            const char* error_msg = Status::get_error_msg(invoke_status);
            LOG_ERROR("apply rank filter fail,the error msg is {}", error_msg);
            return invoke_status;
        }
    }
    if (exclude_DAB) {
        bool Hematoxylin_valid = is_valid_channel(Hematoxylin_channel, channels);
        bool DAB_valid         = is_valid_channel(DAB_channel, channels);
        // pass by ref!
        auto& DAB_mask = image_u8_placeholder;
        if (Hematoxylin_valid && DAB_valid && Hematoxylin_channel != DAB_channel) {
            LOG_INFO("exclude the DAB...");
            constexpr uint8_t DAB_fill_value = 1;
            simple_threshold::greater_equal_than(
                original_image, DAB_mask, Hematoxylin_channel, DAB_channel, DAB_fill_value);
            constexpr double DAB_rank_radius = 2.5;
            rank_filter(DAB_mask, DAB_mask, FilterType::MEDIAN, DAB_rank_radius);
            rank_filter(DAB_mask, DAB_mask, FilterType::MAX, DAB_rank_radius);
            // if the mask == 0,set the pixel value to zero!
            uint8_t* DAB_mask_ptr        = DAB_mask.get_data_ptr();
            float*   transform_image_ptr = transform_image.get_data_ptr();
            // make sure all of our data have same layout!
            // while use the pointer will be a little faster than access with our index...
            for (size_t i = 0; i < data_size; ++i) {
                if (DAB_mask_ptr[i] == 0) {
                    transform_image_ptr[i] = 0.0f;
                }
            }
        }
    }

    // allocate memory for measurement image!while measurement_image can not use
    // image_f32_placeholder!
    LOG_INFO("allocate memory {}Mb for measurments image,also for filter...",
             image_f32_memory_size);
    measurement_image.resize(height, width, 1, true);

    if (background_radius > 0) {
        auto&  background_image     = image_f32_placeholder;
        float* background_image_ptr = background_image.get_data_ptr();
        float* transform_image_ptr  = transform_image.get_data_ptr();
        // copy the transfomr image value to background...
        for (size_t i = 0; i < data_size; ++i) {
            background_image_ptr[i] = transform_image_ptr[i];
        }

        // allocate memory for background mask...,while this can not use the
        // image_u8_placeholder!
        estimate_background(transform_image,
                            background_image,
                            background_mask,
                            background_radius,
                            max_background,
                            background_by_reconstruction);
        copy_image_mat(background_image, transform_image, ValueOpKind::SUBSTRACT);

        LOG_INFO("using the image after background estimate as the image to make measurements!");
        float* measurement_image_ptr = measurement_image.get_data_ptr();
        // now the buffer
        for (size_t i = 0; i < data_size; ++i) {
            measurement_image_ptr[i] = transform_image_ptr[i];
        }
    } else {
        // we can copy the original buffer to temp image buffer!
        //  use the original image to make measuremetns..
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                measurement_image(y, x) = original_image(y, x);
            }
        }
    }

    // first step,generate the rois...
    auto& guassian_blur_result = image_f32_placeholder;
    invoke_status              = guassian_blur_2d(transform_image, guassian_blur_result, sigma);
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_ERROR("apply guassian transform occur error {}", Status::get_error_msg(invoke_status));
        return invoke_status;
    }
    // auto guassian_blur_result_view = guassian_blur_result.get_image_view();

    // apply conv!
    constexpr int conv_kh                               = 3;
    constexpr int conv_kw                               = 3;
    float         convolution_kernel[conv_kh * conv_kw] = {
        0.0f, -1.0f, 0.0f, -1.0f, 4.0f, -1.0f, 0.0f, -1.0f, 0.0f};
    // now need to allocate a buffer to restore the conv result1;
    invoke_status =
        convolution_2d(guassian_blur_result, transform_image, convolution_kernel, conv_kh, conv_kw);
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_INFO("apply convolution failed...");
        return invoke_status;
    }

    // auto conv_result_view = transform_image.get_image_view();


    LOG_INFO("apply binarize with thresold 0.0f");
    ImageMat<uint8_t> transform_image_mask(height, width, 1, MatMemLayout::LayoutRight);
    // compare a matrix with scalr
    threshold_above(transform_image, transform_image_mask, 0.0f);

    // auto transform_image_mask_view = transform_image_mask.get_image_view();

    // reuse memory
    LOG_INFO("binding image placeholder to morphological image...");
    ImageMat<float>& morphological_image = image_f32_placeholder;

    LOG_INFO("apply morphological transform...");
    find_regional_maxima(transform_image, morphological_image, 0.001f);

    LOG_INFO("compute the image label....");

    // support 2^32 -1 polygons...
    using image_label_t = uint32_t;
    ImageMat<image_label_t> label_image(height, width, 1, MatMemLayout::LayoutRight);


    invoke_status = compute_image_label(morphological_image, label_image, 0.0f, false);
    // just for debug!
    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_ERROR("fail to compute image label...");
        return invoke_status;
    }

    LOG_INFO("apply watershed transform...");

    invoke_status = watershed_transform(transform_image, label_image, 0.0f, false);

    // watch the label after watershed iteration!
    // auto label_image_view = label_image.get_image_view();
    // just for debug!
    // auto transform_image_view = transform_image.get_image_view();


    if (invoke_status != Status::ErrorCode::Ok) {
        LOG_ERROR("fail to apply watershed transform...");
        return invoke_status;
    }

    // generate the rois...
    std::vector<PolygonType> rois;
    std::vector<PolyMask>    roi_masks;

    // in qupath,they set min value to 0.5,but image type is short,will apply int(0.5 + 0.5)
    // -> 1.0
    constexpr image_label_t lower_thresh  = 1;
    constexpr image_label_t higher_thresh = std::numeric_limits<image_label_t>::max();
    LOG_INFO("find filled rois...");
    {
        LOG_INFO("binding image_u8_palceholder to temp_fill_mask...");
        auto& temp_fill_mask = image_u8_placeholder;
        invoke_status        = get_filled_polygon(label_image,
                                           temp_fill_mask,
                                           WandMode::FOUR_CONNECTED,
                                           rois,
                                           roi_masks,
                                           lower_thresh,
                                           higher_thresh,
                                           false);
        if (invoke_status != Status::ErrorCode::Ok) {
            LOG_INFO("fill poly fail...");
            return invoke_status;
        }
        LOG_INFO("find {} polygon...", rois.size());
    }

    // filter the rois by mean value,fill the image and use this image to apply transform!
    ImageMat<uint8_t> filled_image(height, width, 1, MatMemLayout::LayoutRight);
    filled_image.set_zero();
    PolygonFiller poly_filler;

    if (background_mask.empty()) {
        LOG_INFO("filter the poly with speicfy threshold %f,only use the measure image...",
                 threshold);
        // compute the mean value of poly,then fitler the mean value which less than our
        // threshold
        size_t remove_rois_by_mean_pixel = 0;
        for (size_t i = 0; i < rois.size(); ++i) {
            auto& roi        = rois[i];
            auto& roi_mask   = roi_masks[i].mask;
            int   original_x = roi_masks[i].original_x;
            int   original_y = roi_masks[i].original_y;
            // write a template to compute the statndard!
            float poly_mean_value =
                compute_roi_mean(measurement_image, roi_mask, original_x, original_y);

            // if the poly mean value less than the threshold,discard it!
            if (poly_mean_value <= threshold) {
                ++remove_rois_by_mean_pixel;
                continue;
            }
            fill_image_with_mask<uint8_t>(filled_image, roi_mask, original_x, original_y, 255);
        }
        LOG_INFO("the filter roi size by pixel mean value is {},the remain rois is {}",
                 remove_rois_by_mean_pixel,
                 rois.size() - remove_rois_by_mean_pixel);
    } else {
        // check the background has same shape with image
        if (!background_mask.compare_shape(measurement_image)) {
            LOG_ERROR("the background have different shape with measurement image!");
            return Status::ErrorCode::MatShapeMismatch;
        }

        size_t remove_rois_by_mean_pixel            = 0;
        size_t remove_rois_by_mean_background_pixel = 0;
        // also check the background!
        for (size_t i = 0; i < rois.size(); ++i) {
            auto& roi        = rois[i];
            auto& roi_mask   = roi_masks[i].mask;
            int   original_x = roi_masks[i].original_x;
            int   original_y = roi_masks[i].original_y;

            float poly_mean_value =
                compute_roi_mean(measurement_image, roi_mask, original_x, original_y);
            if (poly_mean_value <= threshold) {
                ++remove_rois_by_mean_pixel;
                continue;
            }
            // also use the background image!
            float background_poly_mean =
                compute_roi_mean(background_mask, roi_mask, original_x, original_y);
            if (background_poly_mean > 0) {
                ++remove_rois_by_mean_background_pixel;
                continue;
            }
            fill_image_with_mask<uint8_t>(filled_image, roi_mask, original_x, original_y, 255);
        }

        LOG_INFO("the filtered rois by mean pixel is {},by background mean pixel is {},the remain "
                 "rois is {}",
                 remove_rois_by_mean_pixel,
                 remove_rois_by_mean_background_pixel,
                 rois.size() - remove_rois_by_mean_pixel - remove_rois_by_mean_background_pixel);
    }


    if (merge_all) {
        LOG_INFO("binding image_u8_palceholder to neigh_filter_3x3_result...");
        ImageMat<uint8_t>& neigh_filter_3x3_result = image_u8_placeholder;
        bool               pad_edges               = true;
        int                binary_count            = 3;
        neighbor_filter_with_3x3_window(filled_image,
                                        neigh_filter_3x3_result,
                                        NeighborFilterType::MAX,
                                        pad_edges,
                                        binary_count);
        // debug the filled rois
        LOG_INFO("swap the neight_filter_3x3_result to filled_image...");
        filled_image.swap(neigh_filter_3x3_result);

        copy_image_mat(transform_image_mask, filled_image, ValueOpKind::AND);

        // auto filled_image_view = filled_image.get_image_view();
        if (watershed_postprocess) {
            std::vector<PolygonType> postprocess_rois;
            std::vector<PolyMask>    postprocess_roi_masks;
            constexpr uint8_t        lower_thresh   = 127;
            constexpr uint8_t        higher_thresh  = 255;
            ImageMat<uint8_t>&       temp_fill_mask = image_u8_placeholder;
            invoke_status                           = get_filled_polygon(filled_image,
                                               temp_fill_mask,
                                               WandMode::FOUR_CONNECTED,
                                               postprocess_rois,
                                               postprocess_roi_masks,
                                               lower_thresh,
                                               higher_thresh,
                                               false);
            if (invoke_status != Status::ErrorCode::Ok) {
                LOG_ERROR("invoke fill polygon fail");
                return invoke_status;
            }
            for (size_t i = 0; i < postprocess_roi_masks.size(); ++i) {
                auto& poly_mask = postprocess_roi_masks[i].mask;
                // got the left upper coor of the poly!
                int original_x = postprocess_roi_masks[i].original_x;
                int original_y = postprocess_roi_masks[i].original_y;
                int rh         = poly_mask.get_height();
                int rw         = poly_mask.get_width();
                fill_image_with_mask<uint8_t>(filled_image, poly_mask, original_x, original_y, 255);
            }

            // only for debug
            // auto filled_image_view = filled_image.get_image_view();
            LOG_INFO("the brownfox jumps over the lazydog!");

            {
                LOG_INFO("binding image_f32_placeholder to distance image...");
                ImageMat<float>& distance_image = image_f32_placeholder;
                distance_transform<uint8_t>(filled_image, distance_image, false, 0);

                // just for debug
                // auto               distance_image_view = distance_image.get_image_view();
                ImageMat<uint8_t>  distance_mask;   // if empty,we think all value is valid!
                constexpr float    MAXFINDER_TOLERANCE = 0.5f;
                ImageMat<uint8_t>& maximum_mask        = image_u8_placeholder;
                bool               strict              = false;
                bool               exclude_on_edges    = false;
                bool               is_EDM              = true;
                invoke_status                          = find_maxima(distance_image,
                                            distance_mask,
                                            maximum_mask,
                                            strict,
                                            MAXFINDER_TOLERANCE,
                                            NO_THRESHOLD,
                                            EDMOutputType::SEGMENTED,
                                            exclude_on_edges,
                                            true);
                if (invoke_status != Status::ErrorCode::Ok) {
                    LOG_ERROR("find maximum mask fail....");
                    return invoke_status;
                }
                copy_image_mat(maximum_mask, filled_image, ValueOpKind::AND);
            }
        }
    }
    // if only the part of image
    if (refine_boundary && sigma > 1.5) {
        LOG_INFO("refine boundary....");
        // copy the original image value to transform image
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                transform_image(y, x) = original_image(y, x, detect_channel);
            }
        }
        LOG_INFO("binding the image_f32_placeholder to guassian_blur_result");
        ImageMat<float>& guassian_blur_result = image_f32_placeholder;
        constexpr float  refine_sigma         = 1.0f;
        invoke_status = guassian_blur_2d(transform_image, guassian_blur_result, refine_sigma);

        if (invoke_status != Status::ErrorCode::Ok) {
            LOG_ERROR("refine guassian blur fail...");
            return invoke_status;
        }
        // swap the data to transform image...
        // conv
        int   conv_kh        = 3;
        int   conv_kw        = 3;
        float conv_kernel[9] = {0, -1, 0, -1, 4, -1, 0, -1, 0};
        invoke_status =
            convolution_2d(guassian_blur_result, transform_image, conv_kernel, conv_kh, conv_kw);
        if (invoke_status != Status::ErrorCode::Ok) {
            LOG_ERROR("refine conv_kernel blur fail...");
            return invoke_status;
        }

        // here we can reuse the transform image mask!
        threshold_above(transform_image, transform_image_mask, 0.0f);

        copy_image_mat(filled_image, transform_image_mask, ValueOpKind::MIN);

        // apply 3x3 filter!
        auto& neigh_filter_3x3_result = image_u8_placeholder;
        int   binary_count            = 0;
        bool  pad_edges               = false;
        neighbor_filter_with_3x3_window(filled_image,
                                        neigh_filter_3x3_result,
                                        NeighborFilterType::MIN,
                                        pad_edges,
                                        binary_count);
        filled_image.swap(neigh_filter_3x3_result);
        copy_image_mat(transform_image_mask, filled_image, ValueOpKind::MAX);
    }

    std::vector<PolygonType> nuclei_rois;
    std::vector<PolyMask>    nuclei_roi_masks;
    int                      remove_roi_size = 0;
    {
        LOG_INFO("binding image_u8_placeholder to temp_fill_mask...");
        ImageMat<uint8_t>& temp_fill_mask = image_u8_placeholder;
        constexpr uint8_t  thresh_lower   = 127;
        constexpr uint8_t  thresh_higher  = 255;
        get_filled_polygon(filled_image,
                           temp_fill_mask,
                           WandMode::FOUR_CONNECTED,
                           nuclei_rois,
                           nuclei_roi_masks,
                           thresh_lower,
                           thresh_higher,
                           false);
        LOG_INFO("find {} polygon....", nuclei_rois.size());
    }
    // filter the poly by pixel count and pixel value
    std::vector<uint8_t> nuclei_roi_keep_flags(nuclei_rois.size(), 1);

    if (min_area > 0.0 || max_area > 0.0) {
        // convert the area to int,the min area use floor,and the max area use ceil,because we
        // use the open interal!
        int min_area_cast = static_cast<int>(std::ceil(min_area));
        int max_area_cast = static_cast<int>(max_area);
        if (min_area_cast < 0) {
            min_area_cast = 0;
        }

        if (max_area_cast < 0) {
            max_area_cast = std::numeric_limits<int>::max();
        }

        LOG_INFO("we just keep the polygon which have pixel count in range({},{})",
                 min_area_cast,
                 max_area_cast);
        // specify it in prev!
        constexpr uint8_t current_fill_value = 0;
        for (size_t i = 0; i < nuclei_roi_masks.size(); ++i) {
            auto& poly_mask  = nuclei_roi_masks[i].mask;
            int   original_x = nuclei_roi_masks[i].original_x;
            int   original_y = nuclei_roi_masks[i].original_y;
            // compute the pixel count and pixel value!
            StatResult poly_stat =
                compute_roi_stat(measurement_image, poly_mask, original_x, original_y);
            int poly_pixel_count = poly_stat.pixel_count;
            // avoid get inf value!^_^
            double poly_mean_pixel = (poly_stat.sum) / (poly_pixel_count + 1e-9);
            // we should split the condition...
            if (poly_mean_pixel < threshold || poly_pixel_count < min_area_cast ||
                poly_pixel_count > max_area_cast) {
                // fill it to zero!
                fill_image_with_mask(
                    filled_image, poly_mask, original_x, original_y, current_fill_value);
                nuclei_roi_keep_flags[i] = 0;
                ++remove_roi_size;
            }
        }
        LOG_INFO("the remove polygon size is {},the remain polygon size is {}",
                 remove_roi_size,
                 nuclei_rois.size() - remove_roi_size);
        if (remove_roi_size > 0) {
            LOG_INFO(
                "we will remvoe the rois which is not wanted,after filter by area,we remove {} "
                "rois...",
                remove_roi_size);
            // we can adjust the memory to align it!
            std::vector<PolygonType> temp_nuclei_rois;
            temp_nuclei_rois.reserve(nuclei_rois.size() - remove_roi_size);
            std::vector<PolyMask> temp_nuclei_roi_masks;
            temp_nuclei_roi_masks.reserve(nuclei_rois.size() - remove_roi_size);
            // remove the invalid
            for (size_t i = 0; i < nuclei_rois.size(); ++i) {
                if (nuclei_roi_keep_flags[i] != 0) {
                    temp_nuclei_rois.push_back(std::move(nuclei_rois[i]));
                    temp_nuclei_roi_masks.push_back(std::move(nuclei_roi_masks[i]));
                }
            }
            // then swap the buffer
            nuclei_rois.swap(temp_nuclei_rois);
            nuclei_roi_masks.swap(temp_nuclei_roi_masks);
        }
    }

    // rois label to image...
    // only when make measurements...
    label_image.set_zero();
    for (size_t i = 0; i < nuclei_rois.size(); ++i) {
        int   fill_value = i + 1;
        auto& poly_mask  = nuclei_roi_masks[i].mask;
        int   original_x = nuclei_roi_masks[i].original_x;
        int   original_y = nuclei_roi_masks[i].original_y;
        fill_image_with_mask<uint32_t>(label_image, poly_mask, original_x, original_y, fill_value);
    }

    // just for debug!
    // auto   _label_image_view = label_image.get_image_view();
    double downsample_sqrt = std::sqrt(downsample_factor);

    std::vector<PolygonTypef32> smoothed_nuclei_rois;
    smoothed_nuclei_rois.reserve(nuclei_rois.size());
    LOG_INFO("apply smooth for nuclei_rois...");
    if (smooth_boundaries) {
        constexpr size_t    estimate_smooth_size = 128;
        PolygonInterpolator poly_interpolator(estimate_smooth_size);
        PolygonSimplifier   poly_simplifier(estimate_smooth_size);


        bool interpolate_apply_smooth = false;

        // to reuse the memory!
        // PolygonTypef32 smooth_middle_polygon;
        // smooth_middle_polygon.reserve(estimate_smooth_size);

        // we only use these two polygon,we can got the result!
        PolygonTypef32 interpolate_middle_polygon;
        interpolate_middle_polygon.reserve(estimate_smooth_size);

        PolygonTypef32 f32_middle_polygon;
        f32_middle_polygon.reserve(estimate_smooth_size);

        // reserve to specify size!

        for (size_t i = 0; i < nuclei_rois.size(); ++i) {
            // a better usage!
            PolygonType& roi = nuclei_rois[i];
            convert_polygon_to_float(roi, f32_middle_polygon);

            float interpolate_interval = 1.0f;
            poly_interpolator.get_interpolated_polygon_impl(f32_middle_polygon,
                                                            interpolate_middle_polygon,
                                                            interpolate_interval,
                                                            interpolate_apply_smooth,
                                                            RoiType::POLYGON);
            smooth_polygon_roi(interpolate_middle_polygon, f32_middle_polygon);

            interpolate_interval =
                FISH_MIN(2.0f, static_cast<float>(f32_middle_polygon.size() * 0.1));
            poly_interpolator.get_interpolated_polygon_impl(f32_middle_polygon,
                                                            interpolate_middle_polygon,
                                                            interpolate_interval,
                                                            interpolate_apply_smooth,
                                                            RoiType::POLYGON);
            poly_simplifier.simplify_impl(
                interpolate_middle_polygon, f32_middle_polygon, downsample_sqrt / 2.0);
            smoothed_nuclei_rois.push_back(f32_middle_polygon);

            // PolygonTypef32 roi_f32     = convert_polygon_to_float(roi);
            // float          interval_s1 = 1.0f;
            // PolygonTypef32 roi_f32_s1 =
            //     get_interpolated_polygon(roi_f32, interval_s1, false, RoiType::POLYGON);

            // PolygonTypef32 roi_f32_s2  = smooth_polygon_roi(roi_f32_s1);
            // float          interval_s3 = FISH_MIN(2.0, roi_f32_s2.size() * 0.1);
            // PolygonTypef32 roi_f32_s3 =
            //     get_interpolated_polygon(roi_f32_s2, interval_s3, false, RoiType::POLYGON);

            // PolygonTypef32 simplified_polygon =
            //     simplify_polygon_points_better(roi_f32_s3, downsample_sqrt / 2.0);
            // smoothed_nuclei_rois.push_back(std::move(simplified_polygon));
        }
    } else {
        // just convert to float!
        for (size_t i = 0; i < nuclei_rois.size(); ++i) {
            PolygonType&   roi     = nuclei_rois[i];
            PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);
            smoothed_nuclei_rois.push_back(std::move(roi_f32));
        }
    }

    LOG_INFO("clip the smoothed nuclei rois....");
    clip_polygons(smoothed_nuclei_rois, width, height);
    out_nuclei_rois.swap(smoothed_nuclei_rois);
    // firstly,compute the nuclei! if not apply expansion,just use nuclei
    if (cell_expansion > 0.0) {
        LOG_INFO("apply cell expansion!");
        ImageMat<float>& distance_image = image_f32_placeholder;
        distance_transform<uint8_t>(filled_image, distance_image, false, 255);
        float* distance_image_ptr = distance_image.get_data_ptr();
        for (size_t i = 0; i < data_size; ++i) {
            distance_image_ptr[i] *= 1.0f;
        }
        double             cell_expansion_threshold = -1.0 * cell_expansion;
        ImageMat<uint32_t> cell_label_image         = label_image;
        watershed_transform<float, uint32_t>(
            distance_image, cell_label_image, cell_expansion_threshold, false);
        ImageMat<uint8_t>&       temp_fill_mask = image_u8_placeholder;
        std::vector<PolygonType> cell_rois;
        std::vector<PolyMask>    cell_roi_masks;
        labels_to_filled_polygon(
            cell_label_image, temp_fill_mask, nuclei_rois.size(), cell_rois, cell_roi_masks, true);

        std::vector<PolygonTypef32> smoothed_cell_rois;
        smoothed_cell_rois.reserve(cell_rois.size());
        if (smooth_boundaries) {
            for (size_t i = 0; i < cell_rois.size(); ++i) {
                PolygonType&   roi     = cell_rois[i];
                PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);
                if (smooth_boundaries) {
                    constexpr float interval_s1 = 1.0f;
                    PolygonTypef32  roi_f32_s1 =
                        get_interpolated_polygon(roi_f32, interval_s1, false, RoiType::POLYGON);
                    PolygonTypef32 roi_f32_s2  = smooth_polygon_roi(roi_f32_s1);
                    float          interval_s3 = FISH_MIN(2.0, roi_f32_s2.size());
                    PolygonTypef32 roi_f32_s3 =
                        get_interpolated_polygon(roi_f32_s2, interval_s3, false, RoiType::POLYGON);
                    PolygonTypef32 roi_f32_s4 =
                        simplify_polygon_points_better(roi_f32_s3, downsample_sqrt / 2.0);
                    smoothed_cell_rois.push_back(std::move(roi_f32_s4));
                }
            }
        } else {
            // also convert to float!
            for (size_t i = 0; i < cell_rois.size(); ++i) {
                PolygonType&   roi     = cell_rois[i];
                PolygonTypef32 roi_f32 = convert_polygon_to_float(roi);
                smoothed_cell_rois.push_back(std::move(roi_f32));
            }
        }
        out_cell_rois.swap(smoothed_cell_rois);
    }
    return Status::ErrorCode::Ok;
}   // namespace internal
}   // namespace internal


void WatershedCellDetector::print_watershed_params() {
    LOG_INFO("*******watershed segmentaiton params*******");
    LOG_INFO("background_radius:{}", background_radius);
    LOG_INFO("median_radius:{}", median_radius);
    LOG_INFO("sigma:{}", sigma);
    LOG_INFO("min_area:{}", min_area);
    LOG_INFO("max_area:{}", max_area);
    LOG_INFO("cell_expansion:{}", cell_expansion);
    LOG_INFO("max_background:{}", max_background);
    LOG_INFO("merge_all:{}", merge_all);
    LOG_INFO("watershed_postprocess:{}", apply_watershed_postprocess);
    LOG_INFO("exclude_DAB:{}", exclude_DAB);
    LOG_INFO("smooth_boundaries:{}", smooth_boundaries);
    LOG_INFO("make_measurements:{}", make_measurements);
}


size_t WatershedCellDetector::auto_compute_tile_overlap() {
    size_t overlap = 0;
    if (!have_pixel_size_microns) {
        overlap = (cell_expansion > 0.0) ? 25 : 10;
    } else {
        constexpr double nucleus_radius_microns  = 10.0;
        double           _cell_expansion_microns = nucleus_radius_microns;
        if (cell_expansion > 0.0) {
            _cell_expansion_microns += cell_expansion;
        }
        double pixel_size = internal::compute_averaged_pixel_size_microns(pixel_size_microns_h,
                                                                          pixel_size_microns_w);
        overlap           = static_cast<size_t>(_cell_expansion_microns / pixel_size * 2.0);
    }
    constexpr size_t MAX_OVERLAP = 30;
    overlap                      = FISH_MIN(MAX_OVERLAP, overlap);
    LOG_INFO("the auto compute overlap is {}", overlap);
    return overlap;
}

bool WatershedCellDetector::cell_detection(const ImageMat<float>& original_image,
                                           int detect_channel, int Hematoxylin_channel,
                                           int DAB_channel) {
    print_watershed_params();
    double downsample;
    if (have_pixel_size_microns) {
        LOG_INFO("compute the downsample factor wiht pixel size macrons");
        double preferred_pixel_size_microns = internal::compute_preferred_pixel_size_microns(
            pixel_size_microns_h, pixel_size_microns_w, requested_pixel_size);
        downsample = internal::compute_downsample_factor(
            pixel_size_microns_h, pixel_size_microns_w, preferred_pixel_size_microns, false);
    } else {
        // if do not have the macrons,just use 1.0 as the downsample...
        downsample = 1.0;
        LOG_INFO(
            "apply cell detection without pixel size microns,so we will set the downsample factor "
            "to 1.0 as default...");
    }
    LOG_INFO("the downsample factor is {}", downsample);
    Status::ErrorCode status;
    if (!have_pixel_size_microns) {
        status = internal::cell_detection_impl(original_image,
                                               detect_channel,
                                               Hematoxylin_channel,
                                               DAB_channel,
                                               background_radius,
                                               max_background,
                                               median_radius,
                                               sigma,
                                               threshold,
                                               min_area,
                                               max_area,
                                               merge_all,
                                               apply_watershed_postprocess,
                                               exclude_DAB,
                                               cell_expansion,
                                               smooth_boundaries,
                                               make_measurements,
                                               background_by_reconstruction,
                                               refine_boundary,
                                               downsample,
                                               nuclei_rois,
                                               cell_rois);
    } else {
        // transform some datas!
        double pixel_size_microns = internal::compute_averaged_pixel_size_microns(
            pixel_size_microns_h, pixel_size_microns_w);
        LOG_INFO("transform the image proc parmas by divide pixe size microns {}",
                 pixel_size_microns);

        // the precision is not same with qupath,so we just write the param by self!
        //  the radius param...
        double new_background_radius = background_radius / pixel_size_microns;
        double new_median_radius     = median_radius / pixel_size_microns;
        double new_sigma             = sigma / pixel_size_microns;
        double new_min_area          = min_area / (pixel_size_microns * pixel_size_microns);
        double new_max_area          = max_area / (pixel_size_microns * pixel_size_microns);
        double new_cell_expansion    = cell_expansion / pixel_size_microns;

        // change the new sigma to 2.88 to get the same result with quapth!
        // just for debug! our c++ literial have error float....
        new_sigma             = 2.880000;
        new_min_area          = 9.216000;
        new_max_area          = 460.800000;
        new_background_radius = 15.36;


        status = internal::cell_detection_impl(original_image,
                                               detect_channel,
                                               Hematoxylin_channel,
                                               DAB_channel,
                                               new_background_radius,
                                               max_background,
                                               new_median_radius,
                                               new_sigma,
                                               threshold,
                                               new_min_area,
                                               new_max_area,
                                               merge_all,
                                               apply_watershed_postprocess,
                                               exclude_DAB,
                                               new_cell_expansion,
                                               smooth_boundaries,
                                               make_measurements,
                                               background_by_reconstruction,
                                               refine_boundary,
                                               downsample,
                                               nuclei_rois,
                                               cell_rois);
    }
    bool ret = status == Status::ErrorCode::Ok;
    if (!ret) {
        LOG_ERROR("fail to apply watershed segmentation....");
    }
    return ret;
}

bool WatershedCellDetector::cell_detection(const ImageMat<float>& original_image,
                                           int                    detect_channel) {
    if (exclude_DAB) {
        LOG_ERROR("can not exclude DAB because you did not specify Hematoxylin channel and DAB "
                  "channel...");
        return false;
    }
    constexpr int Hematoxylin_channel = -1;
    constexpr int DAB_channel         = -1;

    bool ret = cell_detection(original_image, detect_channel, Hematoxylin_channel, DAB_channel);
    return ret;
}

bool WatershedCellDetector::cell_detection(const ImageMat<uint8_t>& original_image,
                                           int detect_channel, int Hematoxylin_channel,
                                           int DAB_channel) {
    ImageMat<float> original_image_f32 =
        internal::convert_mat_dtype<uint8_t, float>(original_image);
    bool ret = cell_detection(original_image_f32, detect_channel, Hematoxylin_channel, DAB_channel);
    return ret;
}

bool WatershedCellDetector::cell_detection(const ImageMat<uint8_t>& original_image,
                                           int                      detect_channel) {
    if (exclude_DAB) {
        LOG_ERROR(
            "can not exclude DAB because you did not specify Hematoxylin channel and DAB channel");
        return false;
    }
    constexpr int Hematoxylin_channel = -1;
    constexpr int DAB_channel         = -1;
    bool ret = cell_detection(original_image, detect_channel, Hematoxylin_channel, DAB_channel);
    return ret;
}

bool WatershedCellDetector::cell_detection_by_tiling(const ImageMat<float>& original_image,
                                                     int detect_channel, int Hematoxylin_channel,
                                                     int DAB_channel) {
    print_watershed_params();
    double downsample;
    if (have_pixel_size_microns) {
        LOG_INFO("compute the downsample factor wiht pixel size macrons");
        double preferred_pixel_size_microns = internal::compute_preferred_pixel_size_microns(
            pixel_size_microns_h, pixel_size_microns_w, requested_pixel_size);
        downsample = internal::compute_downsample_factor(
            pixel_size_microns_h, pixel_size_microns_w, preferred_pixel_size_microns, false);
    } else {
        // if do not have the macrons,just use 1.0 as the downsample...
        downsample = 1.0;
        LOG_INFO(
            "apply cell detection without pixel size microns,so we will set the downsample factor "
            "to 1.0 as default...");
    }
    LOG_INFO("the downsample factor is {}", downsample);

    // set the detection params...
    double pixel_size_microns =
        internal::compute_averaged_pixel_size_microns(pixel_size_microns_h, pixel_size_microns_w);
    LOG_INFO("transform the image proc parmas by divide pixe size microns {}", pixel_size_microns);
    double new_background_radius = background_radius / pixel_size_microns;
    double new_median_radius     = median_radius / pixel_size_microns;
    double new_sigma             = sigma / pixel_size_microns;
    double new_min_area          = min_area / (pixel_size_microns * pixel_size_microns);
    double new_max_area          = max_area / (pixel_size_microns * pixel_size_microns);
    double new_cell_expansion    = cell_expansion / pixel_size_microns;

    // change the new sigma to 2.88 to get the same result with quapth!
    // just for debug! our c++ literial have error float....
    // new_sigma             = 2.880000;
    // new_min_area          = 9.216000;     // the minimum area of polygon
    // new_max_area          = 460.800000;   // the maximum area of polygon
    // new_background_radius = 15.36;

    // for larget image!
    new_sigma             = 3.0;
    new_min_area          = 10.0;   // the minimum area of polygon
    new_max_area          = 400;    // the maximum area of polygon
    new_background_radius = 4.0;
    int height            = original_image.get_height();
    int width             = original_image.get_width();
    int channels          = original_image.get_channels();
    if (channels != 1) {
        LOG_INFO("we only support single channel image now for large image...");
        return false;
    }
    int overlap = auto_compute_tile_overlap();

    // all of this can removed to the class member to reuse the memory!
    std::vector<tile::RectRange>                           tile_rois;
    std::vector<tile::RectRange>                           intersect_rects;
    std::vector<std::vector<tile::TileRelationInfoToRect>> tile_relations;

    constexpr bool use_fixed_tile_size = false;
    tile::TileInfo tile_info;
    bool           tile_ret = tile::compute_tile_infos(height,
                                             width,
                                             TileParam::PREFERED_TILE_SIZE,
                                             overlap,
                                             use_fixed_tile_size,
                                             tile_rois,
                                             intersect_rects,
                                             tile_relations,
                                             tile_info);
    if (!tile_ret) {
        LOG_ERROR("unable to compute the tile roi info,we will not apply watershed segmentation!");
        return false;
    }
    if (tile_rois.size() == 0) {
        LOG_ERROR("can not get expected tile rois...");
        return false;
    }

    std::vector<Status::ErrorCode> tile_running_status;
    size_t                         tile_num = tile_rois.size();
    LOG_INFO("split current image to  {} tiles...", tile_num);
    tile_nuclei_rois.resize(tile_num);

    LOG_INFO("apply expansion to get the polygon of cell...");
    tile_cell_rois.resize(tile_num);

    tile_running_status.resize(tile_num, Status::ErrorCode::Unknown);


    size_t pool_size       = pool.get_pool_size();
    size_t _tile_num       = tile_num / pool_size * pool_size;
    size_t remain_tile_num = tile_num - _tile_num;

    using CellDetctionRetType = Status::ErrorCode;
    std::vector<std::future<CellDetctionRetType>> cell_detection_futures;
    cell_detection_futures.reserve(std::min(tile_num, pool_size));

    // process the divde part!
    for (size_t i = 0; i < _tile_num; i += pool_size) {
        // avoid allocate too many memory!
        for (size_t j = 0; j < pool_size; ++j) {
            size_t tile_index = i + j;
            int    tile_x1    = tile_rois[tile_index].x1;
            int    tile_y1    = tile_rois[tile_index].y1;
            int    tile_x2    = tile_rois[tile_index].x2;
            int    tile_y2    = tile_rois[tile_index].y2;
            LOG_INFO("create watershed segmentation with x1:{} y1:{} x2:{} y2:{}",
                     tile_x1,
                     tile_y1,
                     tile_x2,
                     tile_y2);
            int             tile_width  = tile_x2 - tile_x1;
            int             tile_height = tile_y2 - tile_y1;
            ImageMat<float> tile_image(tile_height, tile_width, 1, MatMemLayout::LayoutRight);
            // copying the tile image data to current buffer!
            for (int y = 0; y < tile_height; ++y) {
                for (int x = 0; x < tile_width; ++x) {
                    tile_image(y, x) = original_image(y + tile_y1, x + tile_x1);
                }
            }
            // we'd bettter use pmr to avoid fragment!
            auto& single_tile_nuclei_rois = tile_nuclei_rois[tile_index];
            auto& single_tile_cell_rois   = tile_cell_rois[tile_index];
            auto  detection_future =
                pool.enqueue(internal::cell_detection_impl,
                             tile_image,
                             detect_channel,
                             Hematoxylin_channel,
                             DAB_channel,
                             new_background_radius,
                             max_background,
                             new_median_radius,
                             new_sigma,
                             threshold,
                             new_min_area,
                             new_max_area,
                             merge_all,
                             apply_watershed_postprocess,
                             exclude_DAB,
                             new_cell_expansion,
                             smooth_boundaries,
                             make_measurements,
                             background_by_reconstruction,
                             refine_boundary,
                             downsample,
                             std::ref(single_tile_nuclei_rois),
                             std::ref(single_tile_cell_rois));   // cpp will convert ref to copy,so
                                                                 // we should wrap it with std::ref!
            cell_detection_futures.push_back(std::move(detection_future));
        }
        // sync the result!
        for (size_t j = 0; j < pool_size; ++j) {
            size_t tile_index               = i + j;
            auto   status                   = cell_detection_futures[j].get();
            tile_running_status[tile_index] = status;
            if (status != Status::ErrorCode::Ok) {
                LOG_ERROR("tile image {} running failed...", tile_index);
            } else {
                LOG_INFO("tile image {} finish...", tile_index);
            }
        }
        cell_detection_futures.clear();
    }

    // process the tail...
    for (size_t tile_index = _tile_num; tile_index < tile_num; ++tile_index) {
        int tile_x1 = tile_rois[tile_index].x1;
        int tile_y1 = tile_rois[tile_index].y1;
        int tile_x2 = tile_rois[tile_index].x2;
        int tile_y2 = tile_rois[tile_index].y2;

        // the x2 and y2 is the concept of end!
        int tile_width  = tile_x2 - tile_x1;
        int tile_height = tile_y2 - tile_y1;
        LOG_INFO("create watershed segmentation with x1:{} y1:{} x2:{} y2:{}",
                 tile_x1,
                 tile_y1,
                 tile_x2,
                 tile_y2);
        // because the life time of the tile image < thread lefe time,if we pass by ref,will get
        ImageMat<float> tile_image(tile_height, tile_width, 1, MatMemLayout::LayoutRight);
        // copying the tile image data to current buffer!
        for (int y = 0; y < tile_height; ++y) {
            for (int x = 0; x < tile_width; ++x) {
                tile_image(y, x) = original_image(y + tile_y1, x + tile_x1);
            }
        }

        // we can pass it by ref,the life time of the polygon beglong to our detection object!
        auto& single_tile_nuclei_rois = tile_nuclei_rois[tile_index];
        auto& single_tile_cell_rois   = tile_cell_rois[tile_index];
        // large data should use reference to pass!
        // if we pass the tile image by param,will get some unexpected data...
        auto detection_future = pool.enqueue(internal::cell_detection_impl,
                                             tile_image,
                                             detect_channel,
                                             Hematoxylin_channel,
                                             DAB_channel,
                                             new_background_radius,
                                             max_background,
                                             new_median_radius,
                                             new_sigma,
                                             threshold,
                                             new_min_area,
                                             new_max_area,
                                             merge_all,
                                             apply_watershed_postprocess,
                                             exclude_DAB,
                                             new_cell_expansion,
                                             smooth_boundaries,
                                             make_measurements,
                                             background_by_reconstruction,
                                             refine_boundary,
                                             downsample,
                                             std::ref(single_tile_nuclei_rois),
                                             std::ref(single_tile_cell_rois));
        cell_detection_futures.push_back(std::move(detection_future));
    }

    for (size_t i = 0; i < remain_tile_num; ++i) {
        size_t tile_index               = _tile_num + i;
        auto   status                   = cell_detection_futures[i].get();
        tile_running_status[tile_index] = status;
        if (status != Status::ErrorCode::Ok) {
            LOG_ERROR("tile image {} running failed...", tile_index);
        } else {
            LOG_INFO("tile image {} finish...", tile_index);
        }
    }

    bool ret = true;
    for (size_t i = 0; i < tile_num; ++i) {
        if (tile_running_status[i] != Status::ErrorCode::Ok) {
            ret = false;
            break;
        }
    }

    // transform the coordinates of polygon from tile to original
    LOG_INFO("transfomr the coordinates of polygon from tile to original image!");

    // for the nuclei!
    for (size_t i = 0; i < tile_nuclei_rois.size(); ++i) {
        auto& single_tile_nuclei_rois = tile_nuclei_rois[i];

        int x1 = tile_rois[i].x1;
        int y1 = tile_rois[i].y1;
        tile::transform_polygon_tile_to_original(single_tile_nuclei_rois, x1, y1);
    }

    // solving the overlap of tiles...
    LOG_INFO("solving the overlaps....");
    tile::solve_overlap_polygons(tile_nuclei_rois,
                                 tile_rois,
                                 tile_relations,
                                 tile_info.tile_nx,
                                 tile_info.tile_ny,
                                 width,
                                 height,
                                 tile_polygon_status);
    return ret;
}

}   // namespace watershed_cell_detection
}   // namespace segmentation
}   // namespace fish