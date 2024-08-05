#include "image_proc/roi_labeling.h"
#include "core/base.h"
#include "core/mat.h"
#include "image_proc/fill_mask.h"
#include "image_proc/find_contour.h"
#include "image_proc/flood_filler.h"
#include "image_proc/polygon.h"
#include "utils/logging.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace fish {
namespace image_proc {
namespace roi_labeling {
using namespace fish::image_proc::flood_filler;
using namespace fish::image_proc::fill_mask;
using namespace fish::image_proc::contour;

template<class T1, class T2, FloodNeighConnType conn_type, typename = dtype_limit_t<T1>,
         typename = dtype_limit_t<T2>>
Status::ErrorCode compute_image_label_impl(const ImageMat<T1>& image, ImageMat<T2>& label_image,
                                           T1 threshold) {
    int height   = image.get_height();
    int width    = image.get_width();
    int channels = image.get_channels();
    if (channels != 1) {
        LOG_ERROR("the input mat should have 1 channel,but got {}", channels);
        return Status::ErrorCode::InvalidMatShape;
    }
    if (image.get_layout() != label_image.get_layout()) {
        LOG_ERROR("the input and output have diff layout which is not supported!");
        return Status::ErrorCode::MatLayoutMismath;
    }
    if (!label_image.shape_equal(height, width, 1)) {
        label_image.resize(height, width, 1, true);
    }
    constexpr T2 max_label = std::numeric_limits<T2>::max();
    const T1*    image_ptr = image.get_data_ptr();
    T2*          label_ptr = label_image.get_data_ptr();
    size_t       data_size = static_cast<size_t>(height) * static_cast<size_t>(width);
    label_image.set_zero();
    for (size_t i = 0; i < data_size; ++i) {
        if (image_ptr[i] > threshold) {
            label_ptr[i] = max_label;
        }
    }
    // maybe float?
    T2          pixel_label = 0;
    FloodFiller flood_filler;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (label_image(y, x) == max_label) {
                ++pixel_label;
                if (pixel_label == 65) {
                    LOG_INFO("the brownfox jumps over the lazydog!");
                }
                if constexpr (conn_type == FloodNeighConnType::Conn4) {
                    flood_filler.fill(label_image, x, y, pixel_label);
                } else {
                    flood_filler.fill_eight(label_image, x, y, pixel_label);
                }
            }
        }
    }
    return Status::Ok;
}

Status::ErrorCode compute_image_label(const ImageMat<float>& image, ImageMat<uint16_t>& label_image,
                                      float threshold, bool conn_8) {
    if (conn_8) {
        return compute_image_label_impl<float, uint16_t, FloodNeighConnType::Conn8>(
            image, label_image, threshold);
    } else {
        return compute_image_label_impl<float, uint16_t, FloodNeighConnType::Conn4>(
            image, label_image, threshold);
    }
}

Status::ErrorCode compute_image_label(const ImageMat<float>& image, ImageMat<uint32_t>& label_image,
                                      float threshold, bool conn_8) {
    if (conn_8) {
        return compute_image_label_impl<float, uint32_t, FloodNeighConnType::Conn8>(
            image, label_image, threshold);
    } else {
        return compute_image_label_impl<float, uint32_t, FloodNeighConnType::Conn4>(
            image, label_image, threshold);
    }
}

Status::ErrorCode compute_image_label(const ImageMat<uint8_t>& image,
                                      ImageMat<uint16_t>& label_image, uint8_t threshold,
                                      bool conn_8) {
    if (conn_8) {
        return compute_image_label_impl<uint8_t, uint16_t, FloodNeighConnType::Conn8>(
            image, label_image, threshold);
    } else {
        return compute_image_label_impl<uint8_t, uint16_t, FloodNeighConnType::Conn4>(
            image, label_image, threshold);
    }
}

Status::ErrorCode compute_image_label(const ImageMat<uint8_t>& image,
                                      ImageMat<uint32_t>& label_image, float threshold,
                                      bool conn_8) {
    if (conn_8) {
        return compute_image_label_impl<uint8_t, uint32_t, FloodNeighConnType::Conn8>(
            image, label_image, threshold);
    } else {
        return compute_image_label_impl<uint8_t, uint32_t, FloodNeighConnType::Conn4>(
            image, label_image, threshold);
    }
}

// we will always set zero while init our mask!
template<class T, bool only_poly>
Status::ErrorCode get_filled_polygon_impl(const ImageMat<T>& image, ImageMat<uint8_t>& image_mask,
                                          int wand_mode, std::vector<PolygonType>& filled_rois,
                                          std::vector<PolyMask>& roi_masks, T thresh_lower,
                                          T thresh_higher) {
    // simple show the image data!
    std::vector<T> image_view(image.get_data_ptr(), image.get_data_ptr() + image.get_element_num());
    constexpr size_t estimate_poly_size = 512;
    filled_rois.clear();
    filled_rois.reserve(estimate_poly_size);

    roi_masks.clear();
    filled_rois.reserve(estimate_poly_size);
    constexpr size_t estimae_polygon_size = 4;
    filled_rois.reserve(estimae_polygon_size);
    roi_masks.reserve(estimae_polygon_size);
    if constexpr (std::is_same<T, uint8_t>::value) {
        if (&image == &image_mask) {
            LOG_ERROR("the image and image mask can not be same one!");
            return Status::ErrorCode::InvokeInplace;
        }
    }
    int height   = image.get_height();
    int width    = image.get_width();
    int channels = image.get_channels();
    if (channels != 1) {
        return Status::ErrorCode::InvalidMatChannle;
    }
    if (!image_mask.shape_equal(height, width, 1)) {
        image_mask.resize(height, width, 1, true);
    }
    image_mask.set_zero();

    uint8_t*          image_mask_ptr = image_mask.get_data_ptr();
    constexpr uint8_t fill_value     = 255;
    Wand<T>           wand_runner;

    PolygonFiller poly_filler;
    wand_runner.set_lower_threshold(thresh_lower);
    wand_runner.set_upper_threshold(thresh_higher);

    size_t find_poly_size = 0;

    // just for debug!
    // std::vector<std::vector<int>> all_xs;
    // all_xs.reserve(estimae_polygon_size);

    // std::vector<std::vector<int>> all_ys;
    // all_ys.reserve(estimae_polygon_size);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image_mask(y, x) == fill_value) {
                continue;
            }
            T value = image(y, x);
            if (value >= thresh_lower && value <= thresh_higher) {
                // continue;
                if (!wand_runner.auto_outline(image, x, y, 0.0, wand_mode)) {
                    continue;
                }
                ++find_poly_size;

                // got the ref!
                std::vector<Coordinate2d>& poly = wand_runner.get_points_ref();
                if constexpr (only_poly) {
                    poly_filler.fill_polygon(poly.data(), poly.size(), image_mask, fill_value);
                    // transform the coor,make the left upper as original point!
                    auto bound_rect = get_bounding_box(poly);
                    // std::vector<Coordinate2d> _poly      = poly;
                    int original_x = bound_rect.x;
                    int original_y = bound_rect.y;
                    // only for debug! ^_^
                    // std::vector<int> xs(poly.size());
                    // std::vector<int> ys(poly.size());
                    // for (size_t i = 0; i < poly.size(); ++i) {
                    //     xs[i] = poly[i].x - original_x;
                    //     ys[i] = poly[i].y - original_y;
                    // }

                    filled_rois.push_back(poly);
                    // all_xs.push_back(std::move(xs));
                    // all_ys.push_back(std::move(ys));
                } else {
                    Rectangle bound_rect = get_bounding_box(poly);
                    int       original_x = bound_rect.x;
                    int       original_y = bound_rect.y;
                    // substract 1 to get the right fill result!
                    int rect_h = bound_rect.height;
                    int rect_w = bound_rect.width;

                    // there have some erro while finding the polygon!
                    // do not any initialize for the mask placeholder!
                    ImageMat<uint8_t> poly_mask(rect_h, rect_w, 1, MatMemLayout::LayoutRight);
                    poly_mask.set_zero();

                    for (size_t i = 0; i < poly.size(); ++i) {
                        poly[i].x -= original_x;
                        poly[i].y -= original_y;
                    }
                    // we should add a function to fill the gloal mask and poly mask together!
                    poly_filler.fill_polygon(poly.data(), poly.size(), poly_mask, fill_value);
                    // then,recover the coordinae...
                    // here fill mask maybe failed... so got the repeated polygon!
                    // should merge this!
                    // assume the layout is LayoutRight!
                    for (int yy = 0; yy < rect_h; ++yy) {
                        for (int xx = 0; xx < rect_w; ++xx) {
                            // this is error!,0 | 0 is also,will make other value bad,only set value
                            // with 0xff!
                            // this is not need!
                            if (poly_mask(yy, xx) == fill_value) {
                                // the yy+y is the data ptr of mask!
                                // can not use the fill value!
                                image_mask(yy + original_y, xx + original_x) = fill_value;
                            }
                        }
                    }
                    // the quapth save the poly substract the original point!
                    for (size_t i = 0; i < poly.size(); ++i) {
                        poly[i].x += original_x;
                        poly[i].y += original_y;
                    }

                    // just for debug!
                    auto poly_mask_view = poly_mask.get_image_view();
                    filled_rois.push_back(poly);
                    // maybe allocate memory for many times!
                    roi_masks.emplace_back(original_x, original_y, std::move(poly_mask));

                    // split the x and y to compare the middle result... ^_^
                    // only for debug!
                    size_t poly_size = poly.size();
                    // std::vector<int> xs(poly_size);
                    // std::vector<int> ys(poly_size);
                    // for (size_t i = 0; i < poly_size; ++i) {
                    //     xs[i] = poly[i].x - original_x;
                    //     ys[i] = poly[i].y - original_y;
                    // }

                    // all_xs.push_back(std::move(xs));
                    // all_ys.push_back(std::move(ys));
                }
            }
        }
    }
    LOG_INFO("we find {} polys...", find_poly_size);

    return Status::ErrorCode::Ok;
}

// define te lower and upper is prety hard!
Status::ErrorCode get_filled_polygon(const ImageMat<uint8_t>& image, ImageMat<uint8_t>& image_mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, uint8_t thresh_lower,
                                     uint8_t thresh_higher, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = get_filled_polygon_impl<uint8_t, true>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    } else {
        run_status = get_filled_polygon_impl<uint8_t, false>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    }
    return run_status;
}


Status::ErrorCode get_filled_polygon(const ImageMat<uint16_t>& image, ImageMat<uint8_t>& image_mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, uint16_t thresh_lower,
                                     uint16_t thresh_higher, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = get_filled_polygon_impl<uint16_t, true>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    } else {
        run_status = get_filled_polygon_impl<uint16_t, false>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    }
    return run_status;
}


// if given image_mask is empty,we will allocate the buffer!
Status::ErrorCode get_filled_polygon(const ImageMat<uint32_t>& image, ImageMat<uint8_t>& image_mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, uint32_t thresh_lower,
                                     uint32_t thresh_higher, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = get_filled_polygon_impl<uint32_t, true>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    } else {
        run_status = get_filled_polygon_impl<uint32_t, false>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    }
    return run_status;
}

Status::ErrorCode get_filled_polygon(const ImageMat<float>& image, ImageMat<uint8_t>& image_mask,
                                     int wand_mode, std::vector<PolygonType>& filled_rois,
                                     std::vector<PolyMask>& roi_masks, float thresh_lower,
                                     float thresh_higher, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = get_filled_polygon_impl<float, true>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    } else {
        run_status = get_filled_polygon_impl<float, false>(
            image, image_mask, wand_mode, filled_rois, roi_masks, thresh_lower, thresh_higher);
    }
    return run_status;
}


template<class T, bool only_poly>
Status::ErrorCode labels_to_filled_polygon_impl(const ImageMat<T>& label_image,
                                                ImageMat<uint8_t>& image_mask, int n,
                                                std::vector<PolygonType>& filled_rois,
                                                std::vector<PolyMask>&    roi_masks) {
    if (label_image.empty()) {
        return Status::ErrorCode::InvalidMatShape;
    }
    int height   = label_image.get_height();
    int width    = label_image.get_width();
    int channels = label_image.get_channels();
    if (channels != 1) {
        return Status::ErrorCode::InvalidMatChannle;
    }

    if (!image_mask.shape_equal(height, width, 1)) {
        image_mask.resize(height, width, 1, true);
    }

    if (image_mask.get_layout() != MatMemLayout::LayoutRight) {
        image_mask.set_layout(MatMemLayout::LayoutRight);
    }

    image_mask.set_zero();
    constexpr uint8_t fill_value = 255;
    Wand<T>           wand_runner;
    PolygonFiller     poly_filler;
    uint8_t*          image_mask_ptr = image_mask.get_data_ptr();
    // make sure your mat layout is layout right,otherwise,access by width will very low
    // performence!
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (image_mask(y, x) == fill_value) {
                continue;
            }
            T value = label_image(y, x);
            // here the 0 is background!
            if (value > 0 && value <= n) {
                // only find the polygon with specify value!
                // only fill the polygon with same pixel value!
                wand_runner.set_lower_threshold(value);
                wand_runner.set_upper_threshold(value);
                // like the hook!
                // means that we did not found a valid polygon!
                // maybe run faile,not find the polygon!
                if (!wand_runner.auto_outline(label_image, x, y, 0, WandMode::EIGHT_CONNECTAED)) {
                    continue;
                }
                std::vector<Coordinate2d>& poly = wand_runner.get_points_ref();
                if constexpr (only_poly) {
                    poly_filler.fill_polygon(poly.data(), poly.size(), image_mask, fill_value);
                    filled_rois.push_back(poly);
                } else {
                    Rectangle bound_rect = get_bounding_box(poly);
                    int       x1         = bound_rect.x;
                    int       y1         = bound_rect.y;
                    int       rh         = bound_rect.height;
                    int       rw         = bound_rect.width;

                    ImageMat<uint8_t> poly_mask(rh, rw, 1, MatMemLayout::LayoutRight);
                    poly_mask.set_zero();
                    std::vector<Coordinate2d> _poly(poly.size());
                    for (size_t i = 0; i < _poly.size(); ++i) {
                        _poly[i].x -= x1;
                        _poly[i].y -= y1;
                    }
                    poly_filler.fill_polygon(_poly.data(), _poly.size(), poly_mask, fill_value);
                    for (size_t i = 0; i < _poly.size(); ++i) {
                        _poly[i].x += x1;
                        _poly[i].y += y1;
                    }
                    uint8_t* poly_mask_ptr = poly_mask.get_data_ptr();
                    for (int yy = 0; yy < rh; ++yy) {
                        for (int xx = 0; xx < rw; ++xx) {
                            if (poly_mask(yy, xx) == fill_value) {
                                image_mask(yy + y1, xx + x1) = fill_value;
                            }
                        }
                    }

                    filled_rois.push_back(std::move(_poly));
                    roi_masks.emplace_back(x1, y1, std::move(poly_mask));
                }
            }
        }
    }
    return Status::ErrorCode::Ok;
}


Status::ErrorCode labels_to_filled_polygon(const ImageMat<uint16_t>& compute_label_image,
                                           ImageMat<uint8_t>& image_mask, int n,
                                           std::vector<PolygonType>& filled_rois,
                                           std::vector<PolyMask>& roi_masks, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = labels_to_filled_polygon_impl<uint16_t, true>(
            compute_label_image, image_mask, n, filled_rois, roi_masks);
    } else {
        run_status = labels_to_filled_polygon_impl<uint16_t, false>(
            compute_label_image, image_mask, n, filled_rois, roi_masks);
    }
    return run_status;
}


Status::ErrorCode labels_to_filled_polygon(const ImageMat<uint32_t>& compute_label_image,
                                           ImageMat<uint8_t>& image_mask, int n,
                                           std::vector<PolygonType>& filled_rois,
                                           std::vector<PolyMask>& roi_masks, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = labels_to_filled_polygon_impl<uint32_t, true>(
            compute_label_image, image_mask, n, filled_rois, roi_masks);
    } else {
        run_status = labels_to_filled_polygon_impl<uint32_t, false>(
            compute_label_image, image_mask, n, filled_rois, roi_masks);
    }
    return run_status;
}

Status::ErrorCode labels_to_filled_polygon(const ImageMat<float>& compute_label_image,
                                           ImageMat<uint8_t>& image_mask, int n,
                                           std::vector<PolygonType>& filled_rois,
                                           std::vector<PolyMask>& roi_masks, bool only_poly) {
    Status::ErrorCode run_status;
    if (only_poly) {
        run_status = labels_to_filled_polygon_impl<float, true>(
            compute_label_image, image_mask, n, filled_rois, roi_masks);
    } else {
        run_status = labels_to_filled_polygon_impl<float, false>(
            compute_label_image, image_mask, n, filled_rois, roi_masks);
    }
    return run_status;
}

// maybe not use this function forever!
template<class T, typename = dtype_limit<T>>
void clear_outside_impl(ImageMat<T>& image, ImageMat<uint8_t>& image_mask,
                        const PolygonType& polygon, float value) {
    int height   = image.get_height();
    int width    = image.get_width();
    int channels = image.get_channels();
    if (channels != 1) {
        LOG_ERROR("only support single channel mat....");
        return;
    }
    if (polygon.size() == 4) {
        // maybe a rectagnle!
        Rectangle bound = get_bounding_box(polygon);
        if (bound.x == 0 && bound.y == 0 && bound.height == height && bound.width == width) {
            LOG_INFO("the given polygon is a rectagnle and have the same size with out image,so we "
                     "do not clear any...");
            return;
        }
    }
    constexpr uint8_t fill_value = 255;
    if (!image_mask.shape_equal(height, width, 1)) {
        image_mask.resize(height, width, 1, true);
    }
    image_mask.set_zero();
    PolygonFiller poly_filler;
    poly_filler.fill_polygon(polygon.data(), polygon.size(), image_mask, fill_value);
    if (value == 0) {
        // copy_image_mat(image_mask, image, ValueOpKind::MULTIPLY);
        // need to write a new function...
    } else {
        // clip the value
        T        clip_value     = compute_clip_value<T>(value);
        T*       data_ptr       = image.get_data_ptr();
        uint8_t* image_mask_ptr = image_mask.get_data_ptr();
        int      data_size      = height * width;
        for (int i = 0; i < data_size; ++i) {
            if (image_mask_ptr[i] == 0) {
                data_ptr[i] = clip_value;
            }
        }
    }
}

void clear_outside(ImageMat<uint8_t>& image, ImageMat<uint8_t>& image_mask,
                   const PolygonType& polygon) {
    clear_outside_impl(image, image_mask, polygon, 0.0f);
}


void clear_outside(ImageMat<uint16_t>& image, ImageMat<uint8_t>& image_mask,
                   const PolygonType& polygon) {
    clear_outside_impl(image, image_mask, polygon, 0.0f);
}

void clear_outside(ImageMat<uint32_t>& image, ImageMat<uint8_t>& image_mask,
                   const PolygonType& polygon) {
    clear_outside_impl(image, image_mask, polygon, 0.0f);
}

void clear_outside(ImageMat<float>& image, ImageMat<uint8_t>& image_mask,
                   const PolygonType& polygon) {
    clear_outside_impl(image, image_mask, polygon, 0.0f);
}



}   // namespace roi_labeling
}   // namespace image_proc
}   // namespace fish