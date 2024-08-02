#include "segmentation/tile_util.h"
#include "common/fishdef.h"
#include "image_proc/clipper.h"
#include "image_proc/polygon.h"
#include "utils/logging.h"
#include <array>
#include <cstddef>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <vector>



// define a simple macro to remove overlap,because we know the number of invokers...
#define REMOVE_OVERLAP_IMPL_INVOKE(i, j)                                                          \
    {                                                                                             \
        auto&  first_tile                  = relations[i];                                        \
        auto&  second_tile                 = relations[j];                                        \
        size_t first_tile_index            = first_tile.tile_yi * tile_nx + first_tile.tile_xi;   \
        size_t first_tile_intersect_index  = static_cast<size_t>(first_tile.intersection_type);   \
        size_t second_tile_index           = second_tile.tile_yi * tile_nx + second_tile.tile_xi; \
        size_t second_tile_intersect_index = static_cast<size_t>(second_tile.intersection_type);  \
        auto&  first_bound_infos =                                                                \
            tile_related_polygon_infos[first_tile_index][first_tile_intersect_index];             \
        auto& second_bound_infos =                                                                \
            tile_related_polygon_infos[second_tile_index][second_tile_intersect_index];           \
        auto& first_original_polygons  = tile_polygons[first_tile_index];                         \
        auto& second_original_polygons = tile_polygons[second_tile_index];                        \
        solve_overlap_impl(first_bound_infos,                                                     \
                           first_original_polygons,                                               \
                           second_bound_infos,                                                    \
                           second_original_polygons,                                              \
                           area_solver);                                                          \
    }
namespace fish {
namespace segmentation {
namespace tile {

enum class InitializeRelatedPolygonKind : int {
    InitNull           = 0,
    InitOnlyHorizontal = 1,
    InitOnlyVertical   = 2,
    InitAll            = 3
};

constexpr size_t SMALL_RELATION_SIZE                                       = 2;
constexpr size_t LARGE_RELATION_SIZE                                       = 4;
constexpr size_t NORMAL_VERTICAL_RELATION_SIZE                             = SMALL_RELATION_SIZE;
constexpr size_t NORMAL_HORIZONAL_RELATION_SIZE                            = SMALL_RELATION_SIZE;
constexpr size_t NORMAL_INTERSECTION_RELATION_SIZE                         = LARGE_RELATION_SIZE;
constexpr std::array<const char*, layout_size> TILE_INTERSECT_TYPE_STRINGS = {"LEFT_ONLY",
                                                                              "RIGHT_ONLY",
                                                                              "TOP_ONLY",
                                                                              "BOTTOM_ONLY",
                                                                              "LEFT_TOP",
                                                                              "LEFT_BOTTOM",
                                                                              "RIGHT_TOP",
                                                                              "RIGHT_BOTTOM"};
class OverlapAreaSolver {
    constexpr static float  PRECISION             = 100.0;
    constexpr static double AREA_PRECISION        = PRECISION * PRECISION;
    constexpr static double AREA_PRECISION_SCALED = 1.0 / AREA_PRECISION;
    constexpr static size_t ESTIMATE_VERTEX_SIZE  = 64;
    // constexpr static size_t ESTIMATE_INTERSECT_SIZE = 32;

private:
    using VertexValueType = ClipperLib::cInt;
    std::vector<ClipperLib::IntPoint> reused_path;
    // std::vector<std::vector<ClipperLib::IntPoint>> intersect_path;
    ClipperLib::Clipper solver;

    void clear_source() {
        solver.Clear();
        reused_path.clear();
    }

public:
    OverlapAreaSolver() { reused_path.reserve(ESTIMATE_VERTEX_SIZE); };
    OverlapAreaSolver(size_t estimate_vertext_size, size_t estimate_intersect_size) {
        reused_path.reserve(estimate_vertext_size);
    }

    // return a flat means whether the two polygon intersect!
    bool compute(const PolygonTypef32& p1, const PolygonTypef32 p2, double& overlap_area) {
        // cover the polygon to i32
        for (size_t i = 0; i < p1.size(); ++i) {
            reused_path.emplace_back(static_cast<VertexValueType>(p1[i].x * PRECISION),
                                     static_cast<VertexValueType>(p1[i].y * PRECISION));
        }

        // add the polygon...
        solver.AddPath(reused_path, ClipperLib::PolyType::ptSubject, true);
        for (size_t i = 0; i < p2.size(); ++i) {
            reused_path.emplace_back(static_cast<VertexValueType>(p2[i].x * PRECISION),
                                     static_cast<VertexValueType>(p2[i].y * PRECISION));
        }
        solver.AddPath(reused_path, ClipperLib::PolyType::ptClip, true);

        // we should reuse it!
        ClipperLib::Paths res;
        bool              ret = solver.Execute(
            ClipperLib::ClipType::ctIntersection, res, ClipperLib::PolyFillType::pftNonZero);
        if (!ret) FISH_UNLIKELY_STD {
                LOG_ERROR("fail to solve overlap for specify polygons....");
                return false;
            }

        if (res.size() > 0) {
            overlap_area = ClipperLib::Area(res[0]) * AREA_PRECISION_SCALED;
        } else {
            return false;
        }

        // clear the source!
        clear_source();
        return true;
    }
};

const char* convert_tile_intersection_type(TileIntersectRectType rect_type) {
    size_t index = static_cast<size_t>(rect_type);
    return TILE_INTERSECT_TYPE_STRINGS[index];
}

// compute how many intersection rects we have with spcify tile info!
int compute_intersection_rects(int tile_nx, int tile_ny) {
    if (tile_nx <= 0 || tile_ny <= 0) {
        LOG_WARN(
            "the tile_nx:{} tile_ny:{} contains invalid value,so we don't know how to compute it!",
            tile_nx,
            tile_ny);
        return 0;
    }
    return (tile_ny - 1) * (tile_nx - 1) * 3 + tile_nx - 1 + tile_ny - 1;
}

bool compute_tile_infos(int height, int width, int tile_size, int overlap, bool is_fixed_size,
                        std::vector<RectRange>&                           tile_ranges,
                        std::vector<RectRange>&                           intersect_rects,
                        std::vector<std::vector<TileRelationInfoToRect>>& tile_relations,
                        TileInfo&                                         tile_info) {
    //  firstly,check the param!
    if (height <= 0 || width <= 0 || tile_size <= 0 || overlap <= 0) {
        LOG_ERROR("the image size info got invalid value which is not expected,height:{} width:{} "
                  "tile_size:{} overlap:{}",
                  height,
                  width,
                  tile_size,
                  overlap);
        return false;
    }


    float width_f32  = static_cast<float>(width);
    float height_f32 = static_cast<float>(height);

    int tile_size_f32 = static_cast<float>(tile_size);
    int tile_nx       = static_cast<int>(std::ceil(width_f32 / tile_size_f32));
    int tile_ny       = static_cast<int>(std::ceil(height_f32 / tile_size_f32));

    // whether to adjust the tile_w,not make the last too small!
    // for example,if the image size is 3000,we set tile to 2048,if not fix size,will got 2048 +
    // 52,which is not good!
    // but if we adjust it,just got 1500 x 1500,which can control the memory!
    int tile_w = is_fixed_size ? tile_size : static_cast<int>(std::ceil(width_f32 / tile_nx));
    int tile_h = is_fixed_size ? tile_size : static_cast<int>(std::ceil(height_f32 / tile_ny));

    // just return false,the overlap is illogical!
    if (overlap > tile_w || overlap > tile_h) {
        LOG_ERROR(
            "the overlap is not correct,overlap:{} tile_w:{} tile_h:{}", overlap, tile_w, tile_h);
        return false;
    }

    // compute the adjust xmin/ymin
    float x_center = width_f32 * 0.5f;
    float y_center = height_f32 * 0.5f;

    int adjusted_xmin = static_cast<int>(x_center - static_cast<float>(tile_nx * tile_w) * 0.5f);
    int adjusted_ymin = static_cast<int>(y_center - static_cast<float>(tile_ny * tile_h) * 0.5f);
    LOG_INFO("the adjusted xmin is {},adjusted ymin is {} with height:{} width:{}",
             adjusted_xmin,
             adjusted_ymin,
             height,
             width);

    // save the tile info...
    tile_info.tile_w        = tile_w;
    tile_info.tile_h        = tile_h;
    tile_info.tile_nx       = tile_nx;
    tile_info.tile_ny       = tile_ny;
    tile_info.adjusted_xmin = adjusted_xmin;
    tile_info.adjusted_ymin = adjusted_ymin;

    // compute the overlap ranges...
    size_t tile_roi_num = tile_nx * tile_ny;
    // clear is very important,otherwise,we will got unexpected data!
    tile_ranges.clear();
    tile_ranges.reserve(tile_roi_num);

    // this is universal!
    for (int tile_yi = 0; tile_yi < tile_ny; ++tile_yi) {
        // so each tile will expand with overlap * 2,exclude the edge tile!
        // maybe overflow,if the overlap is too large,we should
        int y1 = std::max(0, adjusted_ymin + tile_yi * tile_h - overlap);
        int y2 = std::min(height, adjusted_ymin + (tile_yi + 1) * tile_h + overlap);

        for (int tile_xi = 0; tile_xi < tile_nx; ++tile_xi) {
            int x1 = std::max(0, adjusted_xmin + tile_xi * tile_w - overlap);
            int x2 = std::min(width, adjusted_xmin + (tile_xi + 1) * tile_w + overlap);
            tile_ranges.emplace_back(x1, y1, x2, y2);
        }
    }


    // now create the intersect rects and relations!
    size_t intersection_rect_size = compute_intersection_rects(tile_nx, tile_ny);
    tile_relations.clear();
    tile_relations.reserve(intersection_rect_size);

    intersect_rects.clear();
    intersect_rects.reserve(intersection_rect_size);

    if (tile_nx == 1 && tile_ny == 1) {
        LOG_INFO("tile nx = 1 && tile_ny == 1,no need to compute overlap relation info!");
    } else if (tile_nx > 1 && tile_ny == 1) {
        LOG_INFO("tile nx = 1,tile_ny = {},only compute the simple vertical relations!", tile_nx);
        int vertical_y1 = 0;
        int vertical_y2 = height;
        int tile_yi     = 0;
        for (size_t tile_xi = 0; tile_xi < tile_nx; ++tile_xi) {
            // start of next tile,never < 0!
            int vertical_x1 = adjusted_xmin + (tile_xi + 1) * tile_w - overlap;
            // the end of current tile!
            int vertical_x2 = adjusted_xmin + (tile_xi + 1) * tile_w + overlap;
            intersect_rects.emplace_back(vertical_x1, vertical_y1, vertical_x2, vertical_y2);

            std::vector<TileRelationInfoToRect> vertical_relations;
            vertical_relations.reserve(NORMAL_VERTICAL_RELATION_SIZE);
            vertical_relations.emplace_back(
                tile_xi, tile_yi, TileIntersectRectType::LAYOUT_RIGHT_ONLY);
            vertical_relations.emplace_back(
                tile_xi + 1, tile_yi, TileIntersectRectType::LAYOUT_LEFT_ONLY);
            tile_relations.push_back(std::move(vertical_relations));
        }
    } else if (tile_nx == 1 && tile_ny > 1) {
        LOG_INFO("tile_nx = {},tile_ny = 1,only compute the simple horizontal relations...");
        int horizontal_x1 = 0;
        int horizontal_x2 = width;
        int tile_xi       = 0;
        for (size_t tile_yi = 0; tile_yi < tile_ny - 1; ++tile_yi) {
            int horizontal_y1 = adjusted_ymin + (tile_yi + 1) * tile_h - overlap;
            int horizontal_y2 = adjusted_ymin + (tile_yi + 1) * tile_h + overlap;
            intersect_rects.emplace_back(
                horizontal_x1, horizontal_y1, horizontal_x2, horizontal_y2);

            std::vector<TileRelationInfoToRect> horizontal_relations;
            horizontal_relations.reserve(NORMAL_HORIZONAL_RELATION_SIZE);
            horizontal_relations.emplace_back(
                tile_xi, tile_yi, TileIntersectRectType::LAYOUT_BOTTOM_ONLY);
            horizontal_relations.emplace_back(
                tile_xi, tile_yi + 1, TileIntersectRectType::LAYOUT_TOP_ONLY);
            tile_relations.push_back(std::move(horizontal_relations));
        }
    } else if (tile_nx > 1 && tile_ny > 1) {
        LOG_INFO("tile_nx = {} tile_ny = {},the relations is complex....");
        for (int tile_yi = 0; tile_yi < tile_ny - 1; ++tile_yi) {
            // the vertical y range euqal current tile
            int vertical_y1 = std::max(0, adjusted_ymin + tile_yi * tile_h - overlap);
            int vertical_y2 = adjusted_ymin + (tile_yi + 1) * tile_h + overlap;

            // horizontal
            // the y1 is start y of next tile
            int horizontal_y1 = adjusted_xmin + (tile_yi + 1) * tile_h - overlap;
            // the y2 is end y of current tile!
            int horizontal_y2 = adjusted_ymin + (tile_yi + 1) * tile_h + overlap;

            for (int tile_xi = 0; tile_xi < tile_nx - 1; ++tile_xi) {
                // the vertical x range
                // x1 is the start x of next tile
                int vertical_x1 = adjusted_xmin + (tile_xi + 1) * tile_w - overlap;
                // x2 is the end x of current tile
                int vertical_x2 = adjusted_xmin + (tile_xi + 1) * tile_w + overlap;
                intersect_rects.emplace_back(vertical_x1, vertical_y1, vertical_x2, vertical_y2);
                std::vector<TileRelationInfoToRect> vertical_relations;
                vertical_relations.reserve(NORMAL_HORIZONAL_RELATION_SIZE);
                // xi,yi
                vertical_relations.emplace_back(
                    tile_xi, tile_yi, TileIntersectRectType::LAYOUT_RIGHT_ONLY);
                // xi + 1,yi
                vertical_relations.emplace_back(
                    tile_xi + 1, tile_yi, TileIntersectRectType::LAYOUT_LEFT_ONLY);
                tile_relations.push_back(std::move(vertical_relations));

                // the horizontal y range equals current tile
                int horizontal_x1 = std::max(0, adjusted_xmin + tile_xi * tile_w - overlap);
                int horizontal_x2 = adjusted_xmin + (tile_xi + 1) * tile_w + overlap;
                intersect_rects.emplace_back(
                    horizontal_x1, horizontal_y1, horizontal_x2, horizontal_y2);
                std::vector<TileRelationInfoToRect> horizontal_relations;
                horizontal_relations.reserve(NORMAL_HORIZONAL_RELATION_SIZE);
                // xi,yi
                horizontal_relations.emplace_back(
                    tile_xi, tile_yi, TileIntersectRectType::LAYOUT_BOTTOM_ONLY);
                // xi,yi + 1
                horizontal_relations.emplace_back(
                    tile_xi, tile_yi + 1, TileIntersectRectType::LAYOUT_TOP_ONLY);
                tile_relations.push_back(std::move(horizontal_relations));

                // the maxmimum of left top!
                int intersect_x1 = vertical_x1;
                int intersect_y1 = horizontal_y1;

                // the minimum of right bottom!
                int intersect_x2 = vertical_x2;
                int intersect_y2 = horizontal_y2;
                intersect_rects.emplace_back(
                    intersect_x1, intersect_y1, intersect_x2, intersect_y2);

                std::vector<TileRelationInfoToRect> intersect_relations;
                intersect_relations.reserve(NORMAL_INTERSECTION_RELATION_SIZE);
                // xi,yi
                intersect_relations.emplace_back(
                    tile_xi, tile_yi, TileIntersectRectType::LAYOUT_RIGHT_BOTTOM);
                // xi+1,yi
                intersect_relations.emplace_back(
                    tile_xi + 1, tile_yi, TileIntersectRectType::LAYOUT_LEFT_BOTTOM);
                // xi,yi+1
                intersect_relations.emplace_back(
                    tile_xi, tile_yi + 1, TileIntersectRectType::LAYOUT_RIGHT_TOP);
                // xi+1,yi+1
                intersect_relations.emplace_back(
                    tile_xi + 1, tile_yi + 1, TileIntersectRectType::LAYOUT_LEFT_TOP);
                tile_relations.push_back(std::move(intersect_relations));
            }

            // the last column,only have the horizontal intersection!
            int tile_xi       = tile_nx - 1;
            int horizontal_x1 = adjusted_xmin + tile_xi * tile_w - overlap;
            int horizontal_x2 = std::min(width, adjusted_xmin + (tile_xi + 1) * tile_w + overlap);
            intersect_rects.emplace_back(
                horizontal_x1, horizontal_y1, horizontal_x2, horizontal_y2);

            std::vector<TileRelationInfoToRect> horizontal_relations;
            horizontal_relations.reserve(NORMAL_HORIZONAL_RELATION_SIZE);
            // tile_nx -1,tile_yi
            horizontal_relations.emplace_back(
                tile_xi, tile_yi, TileIntersectRectType::LAYOUT_BOTTOM_ONLY);
            // tile_nx-1,tile_yi + 1
            horizontal_relations.emplace_back(
                tile_xi, tile_yi + 1, TileIntersectRectType::LAYOUT_TOP_ONLY);
            tile_relations.push_back(std::move(horizontal_relations));
        }

        // the last row,only have the vertical column
        int tile_yi = tile_ny - 1;
        // the vertical y range equals tile!
        // the value maybe negative!
        int vertical_y1 = std::max(0, adjusted_ymin + tile_yi * tile_h - overlap);
        int vertical_y2 = adjusted_ymin + (tile_yi + 1) * tile_h + overlap;

        // do not consider the tile_nx -1,tile_ny -1!!!
        for (int tile_xi = 0; tile_xi < tile_nx - 1; ++tile_xi) {
            // the start of next tile
            int vertical_x1 = adjusted_xmin + (tile_xi + 1) * tile_w - overlap;
            // the end of current tile
            int vertical_x2 = adjusted_xmin + (tile_xi + 1) * tile_w + overlap;
            // so the diff of them is 2 * overlap!
            intersect_rects.emplace_back(vertical_x1, vertical_y1, vertical_x2, vertical_y2);
            std::vector<TileRelationInfoToRect> vertical_relations;
            vertical_relations.reserve(NORMAL_VERTICAL_RELATION_SIZE);
            // xi,yi right only
            vertical_relations.emplace_back(
                tile_xi, tile_yi, TileIntersectRectType::LAYOUT_RIGHT_ONLY);
            // xi + 1,yi left only!
            vertical_relations.emplace_back(
                tile_xi + 1, tile_yi, TileIntersectRectType::LAYOUT_LEFT_ONLY);
            tile_relations.push_back(std::move(vertical_relations));
        }
    }
    return true;
}


/*
if tile_xi == 1 && tile_yi == 1,no need to apply any init,because there is no any overlap!
if tile_xi > 1 && tile_yi==1,only need to init the vertical
if tile_xi == 1 && tile_yi > 1,only need to init the horizontal
if tile_xi > 1 && tile_yi > 1,only need
so we only init what we need,do not take any unneccessary things...
*/
bool initialize_tile_related_polygon_infos_better(
    std::vector<TileRelatedPolygonType>& tile_related_polygon_infos, int tile_nx, int tile_ny) {
    // here we reseve 16 for samll overlap,and 128 for large overlap!
    constexpr size_t SMALL_SIZE = 16;
    constexpr size_t LARGE_SIZE = 128;
    if (tile_nx <= 0 || tile_ny <= 0) {
        LOG_ERROR("the tile numeric params are invalid where tile_nx = {},tile_ny = {}",
                  tile_nx,
                  tile_ny);
        return false;
    }

    if (tile_nx * tile_ny != tile_related_polygon_infos.size()) {
        LOG_ERROR(
            "the tile_nx:{} multiply tile_ny:{} not equal to tile size:{} which is unexpected!",
            tile_nx,
            tile_ny,
            tile_related_polygon_infos.size());
        return false;
    }

    if (tile_nx == 1 && tile_ny == 1) {
        LOG_INFO("the tile_nx == 1 && tile_ny = 1,so do not need apply any overlap initialize!");
        return true;
    }

    // firstly,clear all source,be sure there is no any dirty data!
    LOG_INFO("clear all dirty datas for tile related polygon infos...");
    for (size_t i = 0; i < tile_related_polygon_infos.size(); ++i) {
        auto& related_polygon_infos = tile_related_polygon_infos[i];
        for (size_t j = 0; j < related_polygon_infos.size(); ++j) {
            related_polygon_infos[j].clear_source();
        }
    }

    if (tile_nx > 1 && tile_ny == 1) {
        LOG_INFO("get tile_nx = {} tile_ny = {},the overlap only has vertical kind...",
                 tile_nx,
                 tile_ny);
        // the first one only has right kind!
        tile_related_polygon_infos[0][RelatedIndex::RIGHT_ONLY_INDEX].reserve_memory(LARGE_SIZE);
        // middle have left and right both!
        for (size_t tile_xi = 1; tile_xi < tile_nx - 1; ++tile_xi) {
            tile_related_polygon_infos[tile_xi][RelatedIndex::LEFT_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
            tile_related_polygon_infos[tile_xi][RelatedIndex::RIGHT_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
        }
        // the last have left
        tile_related_polygon_infos[tile_nx - 1][0].reserve_memory(LARGE_SIZE);
    } else if (tile_nx == 1 && tile_ny > 1) {
        LOG_INFO("get tile_nx = {} tile_ny = {},the overlap only have horizontal kind...");
        // same as above case!
        tile_related_polygon_infos[0][RelatedIndex::BOTTOM_ONLY_INDEX].reserve_memory(LARGE_SIZE);
        for (size_t tile_yi = 1; tile_yi < tile_ny - 1; ++tile_yi) {
            tile_related_polygon_infos[tile_yi][RelatedIndex::TOP_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
            tile_related_polygon_infos[tile_yi][RelatedIndex::BOTTOM_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
        }
        tile_related_polygon_infos[tile_ny - 1][RelatedIndex::TOP_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
    } else if (tile_nx > 1 && tile_ny > 1) {
        LOG_INFO("get tile_nx = {} tile_ny = {} ,the overlap compute is complext,for some "
                 "tile,have 8 overlaps....");
        // process the first row
        // auto& related_polygon_infos = tile_related_polygon_infos[0];
        // the first case has right and bottom right bottom 3 cases!
        // exlude top left
        tile_related_polygon_infos[0][RelatedIndex::RIGHT_ONLY_INDEX].reserve_memory(LARGE_SIZE);
        tile_related_polygon_infos[0][RelatedIndex::BOTTOM_ONLY_INDEX].reserve_memory(LARGE_SIZE);
        tile_related_polygon_infos[0][RelatedIndex::RIGHT_BOTTOM_INDEX].reserve_memory(SMALL_SIZE);

        // the actually index is 0 * tile_nx + tile_xi
        for (size_t tile_xi = 1; tile_xi < tile_nx - 1; ++tile_xi) {
            // exclude top!
            tile_related_polygon_infos[tile_xi][RelatedIndex::LEFT_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
            tile_related_polygon_infos[tile_xi][RelatedIndex::LEFT_BOTTOM_INDEX].reserve_memory(
                SMALL_SIZE);
            tile_related_polygon_infos[tile_xi][RelatedIndex::RIGHT_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
            tile_related_polygon_infos[tile_xi][RelatedIndex::RIGHT_BOTTOM_INDEX].reserve_memory(
                SMALL_SIZE);
            tile_related_polygon_infos[tile_xi][RelatedIndex::BOTTOM_ONLY_INDEX].reserve_memory(
                LARGE_SIZE);
        }
        // the last one,exluce right case!
        tile_related_polygon_infos[tile_nx - 1][RelatedIndex::LEFT_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
        tile_related_polygon_infos[tile_nx - 1][RelatedIndex::BOTTOM_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
        tile_related_polygon_infos[tile_nx - 1][RelatedIndex::LEFT_BOTTOM_INDEX].reserve_memory(
            SMALL_SIZE);


        for (size_t tile_yi = 1; tile_yi < tile_ny - 1; ++tile_yi) {
            // the first of current row
            // exluce left!
            size_t flat_tile_index = tile_yi * tile_nx;
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::TOP_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_TOP_INDEX]
                .reserve_memory(SMALL_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::BOTTOM_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_BOTTOM_INDEX]
                .reserve_memory(SMALL_SIZE);

            for (size_t tile_xi = 1; tile_xi < tile_nx - 1; ++tile_xi) {
                // include 8
                size_t flat_tile_index = tile_yi * tile_nx + tile_xi;
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_ONLY_INDEX]
                    .reserve_memory(LARGE_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_ONLY_INDEX]
                    .reserve_memory(LARGE_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::TOP_ONLY_INDEX]
                    .reserve_memory(LARGE_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::BOTTOM_ONLY_INDEX]
                    .reserve_memory(LARGE_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_TOP_INDEX]
                    .reserve_memory(SMALL_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_BOTTOM_INDEX]
                    .reserve_memory(SMALL_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_TOP_INDEX]
                    .reserve_memory(SMALL_SIZE);
                tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_BOTTOM_INDEX]
                    .reserve_memory(SMALL_SIZE);
            }
            // the last one,exclude right
            flat_tile_index = tile_yi * tile_nx + tile_nx - 1;
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::TOP_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_TOP_INDEX]
                .reserve_memory(SMALL_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::BOTTOM_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_BOTTOM_INDEX]
                .reserve_memory(SMALL_SIZE);
        }

        // the last row tile_ny-1,0
        size_t tile_yi         = tile_ny - 1;
        size_t flat_tile_index = tile_nx * tile_yi;
        // exlucde left and bottom!
        tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
        tile_related_polygon_infos[flat_tile_index][RelatedIndex::TOP_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
        tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_TOP_INDEX].reserve_memory(
            SMALL_SIZE);
        for (size_t tile_xi = 1; tile_xi < tile_nx - 1; ++tile_xi) {
            // exclude bottom!
            size_t flat_tile_index = tile_yi * tile_nx + tile_xi;
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_TOP_INDEX]
                .reserve_memory(SMALL_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::RIGHT_TOP_INDEX]
                .reserve_memory(SMALL_SIZE);
            tile_related_polygon_infos[flat_tile_index][RelatedIndex::TOP_ONLY_INDEX]
                .reserve_memory(LARGE_SIZE);
        }
        // the last one!
        size_t tile_xi  = tile_nx - 1;
        flat_tile_index = tile_yi * tile_nx + tile_xi;
        // exclude right and bottom!
        tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
        tile_related_polygon_infos[flat_tile_index][RelatedIndex::LEFT_TOP_INDEX].reserve_memory(
            SMALL_SIZE);
        tile_related_polygon_infos[flat_tile_index][RelatedIndex::TOP_ONLY_INDEX].reserve_memory(
            LARGE_SIZE);
    }
    return true;
}

// avoid vector expansion...
// this is not effective!
void initialize_related_polygon_infos(TileRelatedPolygonType& related_polygon_infos) {
    // you can adjust it to get better performence!
    constexpr size_t LARGE_SIZE = 128;
    constexpr size_t SMALL_SIZE = 16;

    constexpr std::array<size_t, 4> SMALL_INDEXES = {RelatedIndex::LEFT_TOP_INDEX,
                                                     RelatedIndex::LEFT_BOTTOM_INDEX,
                                                     RelatedIndex::RIGHT_TOP_INDEX,
                                                     RelatedIndex::RIGHT_BOTTOM_INDEX};
    for (size_t i = 0; i < SMALL_INDEXES.size(); ++i) {
        size_t index       = SMALL_INDEXES[i];
        auto&  bound_infos = related_polygon_infos[index].bound_infos;
        bound_infos.clear();
        bound_infos.reserve(SMALL_SIZE);
    }
    constexpr std::array<size_t, 4> LARGE_INDEXES = {RelatedIndex::LEFT_ONLY_INDEX,
                                                     RelatedIndex::RIGHT_ONLY_INDEX,
                                                     RelatedIndex::TOP_ONLY_INDEX,
                                                     RelatedIndex::BOTTOM_ONLY_INDEX};
    for (size_t i = 0; i < LARGE_INDEXES.size(); ++i) {
        size_t index       = LARGE_INDEXES[i];
        auto&  bound_infos = related_polygon_infos[i].bound_infos;
        bound_infos.clear();
        bound_infos.reserve(LARGE_SIZE);
    }
}

// the implemention to
bool compute_tile_related_polygons_impl(const std::vector<PolygonTypef32>& tile_polygons,
                                        const std::vector<RectRange>& tile_infos, int tile_xi,
                                        int tile_yi, int tile_nx, int tile_ny, int width,
                                        int height, TileRelatedPolygonType& related_polygon_infos) {

    size_t tile_num = tile_infos.size();
    if (tile_nx <= 0 || tile_ny <= 0 || tile_num == 0) {
        LOG_ERROR("the tile size is invalid where tile_nx = {} tile_ny = {} tile_num = {}",
                  tile_nx,
                  tile_ny,
                  tile_num);
        return false;
    }

    if (tile_num != tile_nx * tile_ny) {
        LOG_ERROR("the tile nx:{} x ny:{} not equal to tile_infos size:{},you should give me right "
                  "infos...",
                  tile_nx,
                  tile_ny,
                  tile_infos.size());
        return false;
    }

    // if tile_nx = 1 && tile_ny = 1,no need to compute
    if (tile_nx == 1 && tile_ny == 1) {
        LOG_INFO("the tile nx and tile ny = 1,no need to compute any tile info...");
        return true;
    }

    // initialize the polygon infos... initialize it outside!
    // initialize_related_polygon_infos(related_polygon_infos);

    // before run,clear the boundings...
    size_t polygon_size = tile_polygons.size();
    // compute the intersection bounding of each tile!

    // the x range of this tile to filter the polgyon!
    int left_vertical_x;
    int right_vertical_x;

    // solve the x range!

    // the y range of this tile to filter the polygon!
    int top_horizontal_y;
    int bottom_horizontal_y;

    // if the tile is the first along the width
    if (tile_xi == 0) {
        // because there is none tile overlap with our first for the left!
        left_vertical_x = 0;
    } else {
        // the left vertical equals to the xi-1,yi's end
        size_t overlap_tile_index = tile_yi * tile_nx + (tile_xi - 1);
        left_vertical_x           = tile_infos[overlap_tile_index].x2;
    }

    // if the tile is the last along the width
    if (tile_xi == tile_nx - 1) {
        // because there is none tile overlap with last for the right!
        right_vertical_x = width;
    } else {
        // the right vertical equals to the xi+1,yi's first!
        size_t overlap_tile_index = tile_yi * tile_nx + (tile_xi + 1);
        right_vertical_x          = tile_infos[overlap_tile_index].x1;
    }

    // solve the y as same as x!
    // the first along the height
    if (tile_yi == 0) {
        top_horizontal_y = 0;
    } else {
        // the top horizontal y equals to xi,yi-1's end
        size_t overlap_tile_index = (tile_yi - 1) * tile_nx + tile_xi;
        top_horizontal_y          = tile_infos[overlap_tile_index].y2;
    }

    // the last along the height
    if (tile_yi == tile_ny - 1) {
        bottom_horizontal_y = height;
    } else {
        // the bottom horizontal y equals to xi,yi+1's first!
        size_t overlap_tile_index = (tile_yi + 1) * tile_nx + tile_xi;
        bottom_horizontal_y       = tile_infos[overlap_tile_index].y1;
    }

    LOG_INFO("we will generate candidate polygons,left_vertical_x:{} right_vertical_x:{} "
             "top_horizontal_y:{} bottom_horizontal_y:{}",
             left_vertical_x,
             right_vertical_x,
             top_horizontal_y,
             bottom_horizontal_y);

    for (size_t i = 0; i < polygon_size; ++i) {
        auto&  polygon     = tile_polygons[i];
        size_t vertex_size = polygon.size();
        float  x1          = std::numeric_limits<float>::max();
        float  y1          = std::numeric_limits<float>::max();

        float x2 = 0.0f;
        float y2 = 0.0f;
        // find the bounding box of the polygon!
        for (size_t j = 0; j < vertex_size; ++j) {
            x1 = FISH_MIN(x1, polygon[j].x);
            x2 = FISH_MAX(x2, polygon[j].x);
            y1 = FISH_MIN(y1, polygon[j].y);
            y2 = FISH_MAX(y2, polygon[j].y);
        }
        // fast filte
        bool inner_flag = (x1 >= left_vertical_x && x2 <= right_vertical_x &&
                           y1 >= top_horizontal_y && y2 <= bottom_horizontal_y);
        if (inner_flag) {
            // if the polygon is only located at inner,it is not the candidate polygon...
            continue;
        }

        // all polygon initialize to keep status!
        PolygonBoundInfo bound_info(x1, x2, y1, y2, i, OverlapPolygonStatus::Keep);

        // the trace conditions...
        bool left_x_flag = (x1 < left_vertical_x);
        bool top_y_flag  = (y1 < top_horizontal_y);

        bool right_x_flag  = (x2 > right_vertical_x);
        bool bottom_y_flag = (y2 > bottom_horizontal_y);

        // allow a polygon cross different overlap areas!

        // for the left only! if x less than x1
        if (left_x_flag) {
            related_polygon_infos[RelatedIndex::LEFT_ONLY_INDEX].push_back(bound_info);
            if (top_y_flag) {
                related_polygon_infos[RelatedIndex::LEFT_TOP_INDEX].bound_infos.push_back(
                    bound_info);
            }

            if (bottom_y_flag) {
                related_polygon_infos[RelatedIndex::LEFT_BOTTOM_INDEX].push_back(bound_info);
            }
        }

        // if here exists a polygon which x1 < left and x2 > right,we just split it to left!
        if (right_x_flag) {
            related_polygon_infos[RelatedIndex::RIGHT_ONLY_INDEX].push_back(bound_info);
            if (top_y_flag) {
                related_polygon_infos[RelatedIndex::RIGHT_TOP_INDEX].push_back(bound_info);
            }

            if (bottom_y_flag) {
                related_polygon_infos[RelatedIndex::RIGHT_BOTTOM_INDEX].push_back(bound_info);
            }
        }

        // only consider y here!
        if (top_y_flag) {
            related_polygon_infos[RelatedIndex::TOP_ONLY_INDEX].push_back(bound_info);
        } else {
            if (bottom_y_flag) {
                related_polygon_infos[RelatedIndex::BOTTOM_ONLY_INDEX].push_back(bound_info);
            }
        }
    }

    // print the info!
    LOG_INFO("the info of tile x:{} y:{}", tile_xi, tile_yi);
    for (size_t i = 0; i < related_polygon_infos.size(); ++i) {
        const char* type_string            = TILE_INTERSECT_TYPE_STRINGS[i];
        size_t      candidate_polygon_size = related_polygon_infos[i].get_polygon_size();
        LOG_INFO("related type:{} candidate polygon size:{}", type_string, candidate_polygon_size);
    }
    return true;
}


// defined as class to reuse the memory!
TileRelatedPolygonType compute_tile_related_polygons(std::vector<PolygonTypef32>& tile_polygons,
                                                     int tile_xi, int tile_yi, int tile_nx,
                                                     int tile_ny, int overlap) {
    // 24 x 8
    TileRelatedPolygonType tile_related_polygon_indexes;
    // compute left related!
    if (tile_xi > 0) {}
    // compute the right relaed!
    if (tile_xi < tile_nx - 1) {}

    size_t polygon_size = tile_polygons.size();
    // using as out!
    std::vector<Coordinate2df32> min_vertices;
    std::vector<Coordinate2df32> max_vertices;

    for (size_t i = 0; i < polygon_size; ++i) {}

    return tile_related_polygon_indexes;
}



// the impl of two solver!
void solve_overlap_impl(TilePolygonBoundInfo&              first_bound_infos,
                        const std::vector<PolygonTypef32>& first_original_polygons,
                        TilePolygonBoundInfo&              second_bound_infos,
                        const std::vector<PolygonTypef32>& second_original_polygons,
                        OverlapAreaSolver&                 area_solver) {
    constexpr double overlap_threshold = 0.1;
    if (first_bound_infos.all_dead() || second_bound_infos.all_dead()) {
        LOG_INFO("the given polygons is all dead....");
        return;
    }

    size_t first_candidate_num  = first_bound_infos.get_polygon_size();
    size_t second_candidate_num = second_bound_infos.get_polygon_size();

    for (size_t i = 0; i < first_candidate_num; ++i) {
        if (first_bound_infos.is_removed(i)) {
            continue;
        }
        auto& first_bound = first_bound_infos.get_bound_info(i);
        for (size_t j = 0; j < second_candidate_num; ++j) {
            if (second_bound_infos.is_removed(j)) {
                continue;
            }
            auto& first_polygon = first_original_polygons[i];
            auto& second_bound  = second_bound_infos.get_bound_info(j);

            // should mark it const as even as you can!
            if (!second_bound.is_intersect(first_bound)) {
                continue;
            }
            auto&  second_polygon = second_original_polygons[j];
            double overlap_area;
            bool   succeeded = area_solver.compute(first_polygon, second_polygon, overlap_area);
            if (!succeeded) {
                continue;
            }

            double first_area  = compute_simple_polygon_area<float>(first_polygon);
            double second_area = compute_simple_polygon_area<float>(second_polygon);
            // keep the minimum..
            if (first_area >= second_area) {
                if (overlap_area > second_area * overlap_threshold) {
                    // remove first,and finish current loop!
                    second_bound_infos.mark_removed(j);
                }
            } else {
                if (overlap_area > first_area * overlap_threshold) {
                    first_bound_infos.mark_removed(i);
                    break;
                }
            }
        }

        if (first_bound_infos.all_dead() || second_bound_infos.all_dead()) {
            break;
        }
    }
}

bool solve_overlap_polygons(const std::vector<std::vector<PolygonTypef32>>&         tile_polygons,
                            const std::vector<RectRange>&                           tile_infos,
                            const std::vector<std::vector<TileRelationInfoToRect>>& tile_relations,
                            int tile_nx, int tile_ny, int width, int height,
                            std::vector<std::vector<int>>& tile_polygon_status) {
    // firstly,compute each,const have xxx
    if (tile_nx <= 0 || tile_ny <= 0 || height <= 0 || width <= 0) {
        LOG_ERROR("unexpected none positive values,tile_nx:{} tile_ny:{} height:{} width:{}",
                  tile_nx,
                  tile_ny,
                  height,
                  width);
        return false;
    }
    size_t tile_num = tile_nx * tile_ny;
    if (tile_polygons.size() != tile_num || tile_infos.size() != tile_num) {
        LOG_INFO("got invalid tile sizes,tiles_polygons have size {},tiles_info have size {} "
                 "tile_nx:{} tile_ny: {}",
                 tile_polygons.size(),
                 tile_infos.size(),
                 tile_nx,
                 tile_ny);
        return false;
    }

    std::vector<TileRelatedPolygonType> tile_related_polygon_infos;
    tile_related_polygon_infos.resize(tile_num);

    for (int tile_yi = 0; tile_yi < tile_ny; ++tile_yi) {
        for (int tile_xi = 0; tile_xi < tile_nx; ++tile_xi) {
            size_t flat_tile_index       = tile_yi * tile_nx + tile_xi;
            auto&  related_polygon_infos = tile_related_polygon_infos[flat_tile_index];
            auto&  single_tile_polygons  = tile_polygons[flat_tile_index];
            bool   ret                   = compute_tile_related_polygons_impl(single_tile_polygons,
                                                          tile_infos,
                                                          tile_xi,
                                                          tile_yi,
                                                          tile_nx,
                                                          tile_ny,
                                                          width,
                                                          height,
                                                          related_polygon_infos);
            if (!ret) {
                LOG_ERROR(
                    "can not compute the tile related infos for tile x:{} y:{}", tile_nx, tile_ny);
                return false;
            }
        }
    }

    // find the overlap polygons with rect!
    // the size should be 2 or 4!
    OverlapAreaSolver area_solver;
    for (size_t i = 0; i < tile_relations.size(); ++i) {
        auto&  relations        = tile_relations[i];
        size_t related_tile_num = relations.size();
        // only for debug!
        if (related_tile_num != SMALL_RELATION_SIZE && related_tile_num != LARGE_RELATION_SIZE) {
            LOG_ERROR("got unexpected related tile num {},we only want {} and {}",
                      related_tile_num,
                      SMALL_RELATION_SIZE,
                      LARGE_RELATION_SIZE);
            return false;
        }
        if (related_tile_num == SMALL_RELATION_SIZE) {
            REMOVE_OVERLAP_IMPL_INVOKE(0, 1);
        } else {
            // can expand with 0,1 0,2 0,3 1,2 1,3 2,3
            // for (size_t k1 = 0; k1 < related_tile_num; ++k1) {
            //     for (size_t k2 = 0; k2 < related_tile_num; ++k2) {
            //         REMOVE_OVERLAP_IMPL_INVOKE(k1, k2);
            //     }
            // }
            REMOVE_OVERLAP_IMPL_INVOKE(0, 1);
            REMOVE_OVERLAP_IMPL_INVOKE(0, 2);
            REMOVE_OVERLAP_IMPL_INVOKE(0, 3);
            REMOVE_OVERLAP_IMPL_INVOKE(1, 2);
            REMOVE_OVERLAP_IMPL_INVOKE(1, 3);
            REMOVE_OVERLAP_IMPL_INVOKE(2, 3);
        }
    }
    tile_polygon_status.resize(tile_num);
    for (size_t i = 0; i < tile_num; ++i) {
        std::vector<int>& single_tile_polygon_status = tile_polygon_status[i];
        single_tile_polygon_status.clear();
        single_tile_polygon_status.resize(tile_polygons[i].size(), 1);
        auto& related_polygon_infos = tile_related_polygon_infos[i];

        for (size_t j = 0; j < related_polygon_infos.size(); ++j) {
            auto& candidates = related_polygon_infos[j];
            for (size_t k = 0; k < candidates.get_polygon_size(); ++k) {
                if (candidates.is_removed(k)) {
                    single_tile_polygon_status[k] = 0;
                }
            }
        }
    }
    return true;
}

// transform our polygon to orginal!
// claim the coordinate in valid range!
void transform_polygon_tile_to_original(std::vector<PolygonTypef32>& tile_polygons, int x1,
                                        int y1) {
    // check whether the x1/y1 is valid!
    if (x1 < 0 || y1 < 0) {
        LOG_ERROR("the tile left top coordinate got unexpected negatie value ({},{}),so we will "
                  "not do any transform!",
                  x1,
                  y1);
        return;
    }
    size_t polygon_size = tile_polygons.size();
    // no need to papply any transform for the first tile!
    if (x1 == 0 && y1 == 0) {
        LOG_INFO("the tile left top coordinate is (0,0),so do not need to do any transform for the "
                 "first tile!",
                 x1,
                 y1);
    } else {
        LOG_INFO("transform the coordinates by plus x:{} y:{}", x1, y1);
        float x1_f32 = static_cast<float>(x1);
        float y1_f32 = static_cast<float>(y1);
        for (size_t i = 0; i < polygon_size; ++i) {
            auto&  polygon     = tile_polygons[i];
            size_t vertex_size = polygon.size();
            for (size_t j = 0; j < vertex_size; ++j) {
                polygon[j].x += x1_f32;
                polygon[j].y += y1_f32;
            }
        }
    }
}


}   // namespace tile
}   // namespace segmentation
}   // namespace fish