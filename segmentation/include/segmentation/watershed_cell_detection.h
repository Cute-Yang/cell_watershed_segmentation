#pragma once
#include "core/mat.h"
#include "image_proc/polygon.h"
#include "utils/logging.h"
#include <cstdint>
#include <vector>

// define a simple macro to check the param!
#define CHECK_POSTIVE_PARAM(CHECK_VALUE, CHECK_VAR_NAME)                          \
    if (CHECK_VALUE < 0) {                                                        \
        LOG_WARN("got unexpected {} for param {}", CHECK_VALUE, #CHECK_VAR_NAME); \
        return;                                                                   \
    }

#define SHOW_UPDATE_PARAM(PREV_VALUE, VALUE, VAR_NAME) \
    LOG_INFO("update param {} from {} -> {}", #VAR_NAME, PREV_VALUE, VALUE);

namespace fish {
namespace segmentation {
namespace watershed_cell_detection {
using namespace fish::core::mat;
using namespace fish::image_proc::polygon;


namespace WatershedCellDetectionParam {
// if the simga is too big,we will use a small value to refine it!
constexpr bool   REFINE_BOUNDARY              = true;
constexpr double BACKGROUND_RADIUS            = 15;
constexpr double MAX_BACKGROUND               = 0.3;
constexpr int    Z_DIMENSION                  = 0;
constexpr int    T_DIMENSION                  = 0;
constexpr double CELL_EXPANSION               = 0;
constexpr double MIN_AREA                     = 0.0;
constexpr double MAX_AREA                     = 0.0;
constexpr double MEDIAN_RADIUS                = 2;
constexpr double SIGMA                        = 2.5;
constexpr double THRESHOLD                    = 0.3;
constexpr bool   MERGE_ALL                    = true;
constexpr bool   APPLY_WATERSHED_POSTPROCESS  = true;
constexpr bool   EXCLUDE_DAB                  = false;
constexpr bool   SMOOTH_BOUNDARIES            = true;
constexpr bool   MAKE_MEASUREMENTS            = true;
constexpr bool   BACKGROUND_BY_RECONSTRUCTION = true;
constexpr double REQUESTED_PIXEL_SIZE         = 0.5;
}   // namespace WatershedCellDetectionParam

namespace TileParam {
// the tile size! min/optimize/max
constexpr size_t PREFERED_TILE_SIZE = 2048;
constexpr size_t TILE_SIZE          = 3072;
constexpr size_t MAX_TILE_SIZE      = 4096;
}   // namespace TileParam


struct PolygonRange {
    // the start idx of vertext
    int vertex_start;
    // the size of vertex of this polygon!
    int vertex_size;
};

template<class T> struct FlattenPolygons {
    std::vector<GenericCoordinate2d<T>> flat_vertices;
    std::vector<PolygonRange>           polygon_ranges;
};

using FlattenPolygonsf32 = FlattenPolygons<float>;
using FlattenPolygonsf64 = FlattenPolygons<double>;
using FlattenPolygonsi32 = FlattenPolygons<int32_t>;
using FlattenPolygonsi64 = FlattenPolygons<int64_t>;

using TilePolygonsf32 = std::vector<PolygonTypef32>;
// reused the memory!
using PmrTilePolygonsf32 = std::vector<PmrPolygonTypef32>;
using PmrTilePolygonsf64 = std::vector<PmrPolygonTypef64>;

class WatershedCellDetector {
private:
    // should use other type,the memory may got fragment!
    //  this is the polygon of nuclei
    std::vector<PolygonTypef32> nuclei_rois;
    // this is the polygon of cell!
    std::vector<PolygonTypef32> cell_rois;

    std::vector<TilePolygonsf32> tile_nuclei_rois;
    std::vector<TilePolygonsf32> tile_cell_rois;
    // the detect params!
    double background_radius;
    double max_background;
    double median_radius;
    double sigma;
    double threshold;
    double min_area;
    double max_area;
    double merge_all;
    double cell_expansion;

    // if the image have micron info,we should record it!
    double pixel_size_microns_h;
    double pixel_size_microns_w;
    double requested_pixel_size;

    // the overlap for tiles,we should clear the polygons which have overlap which process large
    // image!
    // some options...
    bool apply_watershed_postprocess;
    bool exclude_DAB;
    bool smooth_boundaries;

    // this is not used now!
    bool make_measurements;
    bool background_by_reconstruction;
    bool refine_boundary;


    // the pysical options of image!
    bool have_pixel_size_microns;

    // just for temp!
    std::vector<std::vector<int>> tile_polygon_status;


    void transform_params_by_microns();
    // should add 3 statics nuclei_stat/cell_stat/Cytoplasm stat(the area between nuclei and cell)
    // ^_^
    size_t auto_compute_tile_overlap();
    void   print_watershed_params();

public:
    WatershedCellDetector() {
        // here just set the default value...
        LOG_INFO("All the detection params are initialized with default value,if you want to use "
                 "specify value,plese update them...");
        refine_boundary              = WatershedCellDetectionParam::REFINE_BOUNDARY;
        background_radius            = WatershedCellDetectionParam::BACKGROUND_RADIUS;
        max_background               = WatershedCellDetectionParam::MAX_BACKGROUND;
        cell_expansion               = WatershedCellDetectionParam::CELL_EXPANSION;
        min_area                     = WatershedCellDetectionParam::MIN_AREA;
        max_area                     = WatershedCellDetectionParam::MAX_AREA;
        median_radius                = WatershedCellDetectionParam::MEDIAN_RADIUS;
        sigma                        = WatershedCellDetectionParam::SIGMA;
        threshold                    = WatershedCellDetectionParam::THRESHOLD;
        merge_all                    = WatershedCellDetectionParam::MERGE_ALL;
        apply_watershed_postprocess  = WatershedCellDetectionParam::APPLY_WATERSHED_POSTPROCESS;
        exclude_DAB                  = WatershedCellDetectionParam::EXCLUDE_DAB;
        smooth_boundaries            = WatershedCellDetectionParam::SMOOTH_BOUNDARIES;
        make_measurements            = WatershedCellDetectionParam::MAKE_MEASUREMENTS;
        requested_pixel_size         = WatershedCellDetectionParam::REQUESTED_PIXEL_SIZE;
        background_by_reconstruction = WatershedCellDetectionParam::BACKGROUND_BY_RECONSTRUCTION;

        have_pixel_size_microns = false;
        pixel_size_microns_h    = 0.0;
        pixel_size_microns_w    = 0.0;
    }


    // the function to set the detect params...
    void set_background_radius(double background_radius_) {
        CHECK_POSTIVE_PARAM(background_radius_, BACKGROUND_RADIUS);
        SHOW_UPDATE_PARAM(background_radius, background_radius, BACKGROUND_RADIUS);
        background_radius = background_radius_;
    }

    void set_median_radius(double median_radius_) {
        CHECK_POSTIVE_PARAM(median_radius_, MEDIAN_RADIUS);
        SHOW_UPDATE_PARAM(median_radius, median_radius_, MEDIAN_RADIUS);
        median_radius = median_radius_;
    }

    void set_max_background(double max_background_) {
        CHECK_POSTIVE_PARAM(max_background_, MAX_BACKGROUND);
        SHOW_UPDATE_PARAM(max_background, max_background_, MAX_BACKGROUND);
        max_background = max_background_;
    }

    void set_sigma(double sigma_) {
        CHECK_POSTIVE_PARAM(sigma_, SIGMA);
        SHOW_UPDATE_PARAM(sigma, sigma_, SIGMA);
        sigma = sigma_;
    }

    void set_threshold(double threshold_) {
        CHECK_POSTIVE_PARAM(threshold_, BRIGHTNESS_THRESHOLD);
        SHOW_UPDATE_PARAM(threshold, threshold_, THRESHOLD);
        threshold = threshold_;
    }

    void set_min_area(double min_area_) {
        CHECK_POSTIVE_PARAM(min_area_, MIN_AREA);
        SHOW_UPDATE_PARAM(min_area, min_area_, MIN_AREA);
        min_area = min_area_;
    }

    void set_max_area(double max_area_) {
        CHECK_POSTIVE_PARAM(max_area_, MAX_AREA);
        SHOW_UPDATE_PARAM(max_area, max_area_, MAX_AREA);
        max_area = max_area_;
    }

    void set_merge_all(bool merge_all_) {
        CHECK_POSTIVE_PARAM(merge_all_, MERGE_ALL);
        SHOW_UPDATE_PARAM(merge_all, merge_all_, MERGE_ALL);
        merge_all = merge_all_;
    }

    void set_watershed_postprocess(bool apply_watershed_postprocess_) {
        SHOW_UPDATE_PARAM(
            apply_watershed_postprocess, apply_watershed_postprocess_, APPLY_WATERSHED_POSTPROCESS);
        apply_watershed_postprocess = apply_watershed_postprocess_;
    }

    // the options params....
    void set_exclude_DAB(bool exclude_DAB_) {
        SHOW_UPDATE_PARAM(exclude_DAB, exclude_DAB_, EXCLUDE_DAB);
        exclude_DAB = exclude_DAB_;
    }

    void set_cell_expansion(double cell_expansion_) {
        CHECK_POSTIVE_PARAM(cell_expansion_, CELL_EXPANSION);
        SHOW_UPDATE_PARAM(cell_expansion, cell_expansion_, CELL_EXPANSION);

        cell_expansion = cell_expansion_;
    }

    void set_smooth_boundaries(bool smooth_boundaries_) {
        SHOW_UPDATE_PARAM(smooth_boundaries, smooth_boundaries_, SMOOTH_BOUNDARIES);
        smooth_boundaries = smooth_boundaries_;
    }

    void set_make_measurements(bool make_measurements_) {
        SHOW_UPDATE_PARAM(make_measurements, make_measurements_, MAKE_MEASUREMENTS);
        make_measurements = make_measurements_;
    }

    // for morophical!
    void set_background_by_reconstruction(bool background_by_reconstruction_) {
        SHOW_UPDATE_PARAM(background_by_reconstruction,
                          background_by_reconstruction_,
                          BACKGROUND_BY_RECONSTRUCTION);
        background_by_reconstruction = background_by_reconstruction_;
    }

    // adjust the segmention with small param ^_^
    void set_refine_boundary(bool refine_boundary_) {
        SHOW_UPDATE_PARAM(refine_boundary, refine_boundary_, REFINE_BOUNDARY);
        refine_boundary = refine_boundary_;
    }

    void set_have_pixle_size_microns(bool have_pixel_size_microns_) {
        SHOW_UPDATE_PARAM(
            have_pixel_size_microns, have_pixel_size_microns_, HAVE_PIXEL_SIZE_MICRONS);
        have_pixel_size_microns = have_pixel_size_microns_;
    }

    // the physical info of image!
    void set_pixel_size_microns(double pixel_size_microns_h_, double pixel_size_microns_w_) {
        CHECK_POSTIVE_PARAM(pixel_size_microns_h_, PIXEL_SIZE_MICRONS_H);
        SHOW_UPDATE_PARAM(pixel_size_microns_h, pixel_size_microns_h_, PIXEL_SIZE_MICRONS_H);
        pixel_size_microns_h = pixel_size_microns_h_;

        CHECK_POSTIVE_PARAM(pixel_size_microns_w_, PIXEL_SIZE_MICRONS_W);
        SHOW_UPDATE_PARAM(pixel_size_microns_w, pixel_size_microns_w_, PIXEL_SIZE_MICRONS_W);
        pixel_size_microns_w = pixel_size_microns_w_;
    }

    // this param will be effective only set have pixel size microns
    void set_requested_pixel_size(double requested_pixel_size_) {
        CHECK_POSTIVE_PARAM(requested_pixel_size_, REQUESTED_PIXEL_SIZE);
        SHOW_UPDATE_PARAM(requested_pixel_size, requested_pixel_size_, REQUESTED_PIXEL_SIZE);
        requested_pixel_size = requested_pixel_size_;
    }

    // this function for image which have microns,be sure your image is sampled with speicfy
    // ratio!
    bool cell_detection(const ImageMat<float>& original_image, int detect_channel,
                        int Hematoxylin_channel, int DAB_channel);
    // for single channel image maybe!
    bool cell_detection(const ImageMat<float>& original_image, int detect_channel);

    // this function for image which do not have micron info..

    bool cell_detection(const ImageMat<uint8_t>& original_image, int detect_channel,
                        int Hematoxylin_channel, int DAB_channel);

    bool cell_detection(const ImageMat<uint8_t>& original_image, int detect_channel);


    bool cell_detection_by_tiling(const ImageMat<float>& original_image, int detect_channel,
                                  int Hematoxylin_channel, int DAB_channel);


    std::vector<PolygonTypef32>& get_nuclei_rois_ref() { return nuclei_rois; }

    std::vector<std::vector<PolygonTypef32>>& get_tile_nuclei_rois_ref() {
        return tile_nuclei_rois;
    }

    std::vector<std::vector<int>>& get_tile_polygon_status_ref() { return tile_polygon_status; }

    const std::vector<PolygonTypef32>& get_nuclei_rois_cref() const { return nuclei_rois; }

    std::vector<PolygonTypef32> get_nuclei_rois() const { return nuclei_rois; }

    std::vector<PolygonTypef32>& get_cell_rois_ref() { return cell_rois; }

    const std::vector<PolygonTypef32>& get_cell_rois_cref() const { return cell_rois; }

    std::vector<PolygonTypef32> get_cell_rois() const { return cell_rois; }
};
}   // namespace watershed_cell_detection
}   // namespace segmentation
}   // namespace fish