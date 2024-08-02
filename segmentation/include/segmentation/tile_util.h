#pragma once
#include "image_proc/polygon.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>
namespace fish {
namespace segmentation {
namespace tile {
using namespace fish::image_proc::polygon;
// a rect have different type with different tiles
enum class TileIntersectRectType : int {
    LAYOUT_LEFT_ONLY    = 0,
    LAYOUT_RIGHT_ONLY   = 1,
    LAYOUT_TOP_ONLY     = 2,
    LAYOUT_BOTTOM_ONLY  = 3,
    LAYOUT_LEFT_TOP     = 4,
    LAYOUT_LEFT_BOTTOM  = 5,
    LAYOUT_RIGHT_TOP    = 6,
    LAYOUT_RIGHT_BOTTOM = 7,
    LAYOUT_COUNT
};
constexpr size_t layout_size = static_cast<size_t>(TileIntersectRectType::LAYOUT_COUNT);

namespace RelatedIndex {
constexpr size_t LEFT_ONLY_INDEX   = static_cast<size_t>(TileIntersectRectType::LAYOUT_LEFT_ONLY);
constexpr size_t RIGHT_ONLY_INDEX  = static_cast<size_t>(TileIntersectRectType::LAYOUT_RIGHT_ONLY);
constexpr size_t TOP_ONLY_INDEX    = static_cast<size_t>(TileIntersectRectType::LAYOUT_TOP_ONLY);
constexpr size_t BOTTOM_ONLY_INDEX = static_cast<size_t>(TileIntersectRectType::LAYOUT_BOTTOM_ONLY);

constexpr size_t LEFT_TOP_INDEX    = static_cast<size_t>(TileIntersectRectType::LAYOUT_LEFT_TOP);
constexpr size_t LEFT_BOTTOM_INDEX = static_cast<size_t>(TileIntersectRectType::LAYOUT_LEFT_BOTTOM);
constexpr size_t RIGHT_TOP_INDEX   = static_cast<size_t>(TileIntersectRectType::LAYOUT_RIGHT_TOP);
constexpr size_t RIGHT_BOTTOM_INDEX =
    static_cast<size_t>(TileIntersectRectType::LAYOUT_RIGHT_BOTTOM);

}   // namespace RelatedIndex

// record the polygon index and it's bounding box!
// using TileRelatedPolygonType = std::array<std::vector<int>, layout_size>;
int compute_intersection_rects(int tile_nx, int tile_ny);

struct RectRange {
    int x1;
    int y1;
    int x2;
    int y2;
};

enum class OverlapPolygonStatus : int { Keep = 0, Removed = 1, Unknown = 2 };
struct PolygonBoundInfo {
    // the x1,x2,y1,y2 is th
    float x1;
    float x2;
    float y1;
    float y2;
    int   polygon_index;
    // if the status is removed,we will discard it!
    OverlapPolygonStatus status;

    PolygonBoundInfo() = delete;
    PolygonBoundInfo(float x1_, float x2_, float y1_, float y2_, int polygon_index_)
        : x1(x1_)
        , x2(x2_)
        , y1(y1_)
        , y2(y2_)
        , polygon_index(polygon_index_)
        , status(OverlapPolygonStatus::Keep) {}

    PolygonBoundInfo(float x1_, float x2_, float y1_, float y2_, int polygon_index_,
                     OverlapPolygonStatus status_)
        : x1(x1_)
        , x2(x2_)
        , y1(y1_)
        , y2(y2_)
        , polygon_index(polygon_index_)
        , status(status_) {}

    bool is_intersect(const PolygonBoundInfo& rhs) const {
        int inter_x1 = std::max(x1, rhs.x1);
        int inter_x2 = std::min(x2, rhs.x2);

        int  inter_y1       = std::max(y1, rhs.y1);
        int  inter_y2       = std::min(y2, rhs.y2);
        bool intersect_flag = (inter_x1 < inter_x2) && (inter_y1 < inter_y2);
        return intersect_flag;
    }
};


// the info to compute the tile!
struct TileInfo {
    // the tile infos...
    int tile_w;
    int tile_h;

    int tile_nx;
    int tile_ny;

    // maybe negative!
    int adjusted_xmin;
    int adjusted_ymin;
};


struct TilePolygonBoundInfo {
    std::vector<PolygonBoundInfo> bound_infos;
    size_t                        active_polygon_num;

    TilePolygonBoundInfo()
        : bound_infos()
        , active_polygon_num(0) {}

    // void reserve(size_t n) { bound_infos.reserve(n); }
    // compat with stl's vector.push_back ^_^
    void push_back(const PolygonBoundInfo& x) {
        bound_infos.push_back(x);
        ++active_polygon_num;
    }

    bool is_active(size_t index) const {
        return bound_infos[index].status == OverlapPolygonStatus::Keep;
    }


    bool is_removed(size_t index) const {
        return bound_infos[index].status == OverlapPolygonStatus::Removed;
    }

    PolygonBoundInfo& get_bound_info(size_t index) { return bound_infos[index]; }

    const PolygonBoundInfo& get_bound_info(size_t index) const { return bound_infos[index]; }

    void mark_removed(size_t index) {
        bound_infos[index].status = OverlapPolygonStatus::Removed;
        --active_polygon_num;
    }

    bool all_dead() const { return active_polygon_num == 0; }

    void mark_keep(size_t index) {
        bound_infos[index].status = OverlapPolygonStatus::Keep;
        ++active_polygon_num;
    }

    void reserve_memory(size_t n) { bound_infos.reserve(n); }

    void reset_activate_polygon_num() {
        size_t num = 0;
        for (size_t i = 0; i < bound_infos.size(); ++i) {
            if (bound_infos[i].status == OverlapPolygonStatus::Keep) {
                ++num;
            }
        }
        active_polygon_num = num;
    }

    // this is not good!
    // only can invoke this for the first time...
    // void init_activate_polygon_num_fast() { active_polygon_num = bound_infos.size(); }
    void init_activate_polygon_num_slow() { reset_activate_polygon_num(); }

    size_t get_activate_num() const { return active_polygon_num; }

    size_t get_polygon_size() const { return bound_infos.size(); }

    size_t get_candidate_size() const { return bound_infos.size(); }

    void clear_source() {
        bound_infos.clear();
        active_polygon_num = 0;
    }
};


using TileRelatedPolygonType = std::array<TilePolygonBoundInfo, layout_size>;


const char* convert_tile_intersection_type(TileIntersectRectType rect_type);

struct TileRelationInfoToRect {
    // the xi/yi of current  tile!
    int tile_xi;
    int tile_yi;
    // the intersection type of tile to current rect!
    TileIntersectRectType intersection_type;

    TileRelationInfoToRect(int tile_xi_, int tile_yi_, TileIntersectRectType intersection_type_)
        : tile_xi(tile_xi_)
        , tile_yi(tile_yi_)
        , intersection_type(intersection_type_) {}

    void swap(TileRelationInfoToRect& rhs) {
        int                   temp_tile_xi           = tile_xi;
        int                   temp_tile_yi           = tile_yi;
        TileIntersectRectType temp_intersection_type = intersection_type;

        tile_xi           = rhs.tile_xi;
        tile_yi           = rhs.tile_yi;
        intersection_type = rhs.intersection_type;

        rhs.tile_xi           = temp_tile_xi;
        rhs.tile_yi           = temp_tile_yi;
        rhs.intersection_type = temp_intersection_type;
    }
};



// compute the roi infos!
bool compute_tile_infos(int height, int width, int tile_size, int overlap, bool is_fixed_size,
                        std::vector<RectRange>& tile_rois, std::vector<RectRange>& intersect_rects,
                        std::vector<std::vector<TileRelationInfoToRect>>& tile_relations,
                        TileInfo&                                         tile_info);
class TileRelatedPolygonsSelector {
private:
    // use this to restore the min vertices and max vertices of current polygon!
    std::vector<RectRange> polygon_boundings;
    // only clear the source,buf not free the buffer!
    void clear_source() { polygon_boundings.clear(); }

    // clear and free the buffer!
    void clear_and_release_source() {
        polygon_boundings.clear();
        polygon_boundings.shrink_to_fit();
    }

public:
    TileRelatedPolygonsSelector() {}

    // you can specify the initialize size!

    TileRelatedPolygonsSelector(size_t bounding_size) { polygon_boundings.reserve(bounding_size); }
    // to support pmr!
};


// lazyout right!ny is at axis_0 nx is at axis 1
// we will save the polygon indexes...

bool compute_tile_related_polygons_impl(const PolygonTypef32*         single_tile_polygons,
                                        size_t                        single_tile_polygon_len,
                                        const std::vector<RectRange>& tiles_info, int tile_xi,
                                        int tile_yi, int tile_nx, int tile_ny, int width,
                                        int height, TileRelatedPolygonType& related_polygon_infos);

bool compute_tile_related_polygons_impl(const std::vector<PolygonTypef32>& single_tile_polygons,
                                        const std::vector<RectRange>& tile_infos, int tile_xi,
                                        int tile_yi, int tile_nx, int tile_ny, int width,
                                        int height, TileRelatedPolygonType& related_polygon_infos);

// the element is the ref of polygons...
using TilePolygon = std::vector<PolygonTypef32>;
// maybe use std::vector<PmrPolygonTypef32>
using TilesPolygonsRef = std::vector<TilePolygon>;

bool solve_overlap_polygons(const std::vector<std::vector<PolygonTypef32>>&         tile_polygons,
                            const std::vector<RectRange>&                           tile_infos,
                            const std::vector<std::vector<TileRelationInfoToRect>>& tile_relations,
                            int tile_nx, int tile_ny, int width, int height,
                            std::vector<std::vector<int>>& tile_polygon_status);

// plus the coordinate with left coor and right coor!

void transform_polygon_tile_to_original(std::vector<PolygonTypef32>& tile_polygons, int x1, int y1);

}   // namespace tile
}   // namespace segmentation
}   // namespace fish