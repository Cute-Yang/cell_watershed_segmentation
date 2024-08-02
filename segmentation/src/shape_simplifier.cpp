#include "segmentation/shape_simplifier.h"
#include "common/fishdef.h"
#include "core/mat.h"
#include "image_proc/polygon.h"
#include <cmath>
#include <queue>
#include <set>
#include <vector>
namespace fish {
namespace segmentation {
namespace shape_simplier {
using namespace fish::core;
using namespace fish::image_proc::polygon;

// min heap!
template<class T> struct PwaCompare {
    bool operator()(const PointWithArea<T>* lhs, const PointWithArea<T>* rhs) {
        return lhs->area > rhs->area;
    }
};

PolygonTypef32 simplify_polygon_points(const PolygonTypef32& polygon, float altitude_threshold) {
    if (polygon.size() <= 1) {
        return polygon;
    }
    // make the uinque...
    PolygonTypef32 removed_adjacent_polygon;
    removed_adjacent_polygon.reserve(polygon.size());
    Coordinate2df32 last_point = polygon[0];
    removed_adjacent_polygon.push_back(polygon[0]);
    for (size_t i = 1; i < polygon.size(); ++i) {
        if (polygon[i] != last_point) {
            removed_adjacent_polygon.push_back(polygon[i]);
            last_point = polygon[i];
        }
    }
    if (last_point == polygon[0]) {
        // delete the last!
        removed_adjacent_polygon.resize(removed_adjacent_polygon.size() - 1);
    }
    if (removed_adjacent_polygon.size() <= 3) {
        return removed_adjacent_polygon;
    }

    int n = removed_adjacent_polygon.size();

    using QueueType =
        std::priority_queue<PointWithAreaf32*, std::vector<PointWithAreaf32*>, PwaCompare<float>>;

    QueueType point_queue;

    Coordinate2df32               prev_point    = removed_adjacent_polygon[n - 1];
    Coordinate2df32               current_point = removed_adjacent_polygon[0];
    PointWithAreaf32*             prev_pwa      = nullptr;
    PointWithAreaf32*             first_pwa     = nullptr;
    std::vector<PointWithAreaf32> pwa_pool;
    pwa_pool.reserve(n);

    // building the queue
    for (int i = 0; i < n; ++i) {
        Coordinate2df32 next_point = removed_adjacent_polygon[(i + 1) % n];
        double          area       = calculate_triple_area(prev_point, current_point, next_point);
        pwa_pool.emplace_back(current_point, area);
        // get the data ptr of current element1
        PointWithAreaf32* pwa = pwa_pool.data() + pwa_pool.size();
        pwa->set_prev(prev_pwa);
        if (prev_pwa != nullptr) {
            prev_pwa->set_next(pwa);
        }
        point_queue.push(pwa);
        prev_pwa   = pwa;
        prev_point = current_point;
        if (i == n - 1) {
            pwa->set_next(first_pwa);
            first_pwa->set_prev(pwa);
        } else if (i == 0) {
            first_pwa = pwa;
        }
    }

    double max_area = 0;
    int    min_size = FISH_MIN(n / 100, 3);

    // do not use the set! which will cost many memory!
    std::set<Coordinate2df32> remove_points;

    while (point_queue.size() > min_size) {
        PointWithAreaf32* pwa = point_queue.top();
        point_queue.pop();
        auto&  _next_p = pwa->get_next()->p;
        auto&  _prev_p = pwa->get_prev()->p;
        float  dx      = _next_p.x - _prev_p.x;
        float  dy      = _next_p.y - _prev_p.y;
        double dist    = std::sqrt(dx * dx + dy * dy) + 1e-9;
        double altitue = pwa->get_area() * 2 / dist;
        if (altitue > altitude_threshold) {
            break;
        }
        if (pwa->get_area() < max_area) {
            pwa->set_area(max_area);
        } else {
            max_area = pwa->get_area();
        }
        remove_points.insert(pwa->get_point_ref());
        prev_pwa                   = pwa->get_prev();
        PointWithAreaf32* next_pwa = pwa->get_next();
        prev_pwa->set_next(next_pwa);
        next_pwa->set_prev(prev_pwa);

        // this is unsafe!
        prev_pwa->update_area();
        next_pwa->update_area();
        // remvoe current and rebuild
        // maybe slow,but I don't have a better method...
        for (size_t i = 0; i < point_queue.size(); ++i) {
            auto* element_ptr = point_queue.top();
            point_queue.pop();
            point_queue.push(element_ptr);
        }
        // adjust the priority!
    }
    if (remove_points.size() == 0) {
        return removed_adjacent_polygon;
    }
    PolygonTypef32 ret_polygon;
    ret_polygon.reserve(removed_adjacent_polygon.size() - remove_points.size());
    for (size_t i = 0; i < removed_adjacent_polygon.size(); ++i) {
        if (remove_points.find(removed_adjacent_polygon[i]) != remove_points.end()) {
            ret_polygon.push_back(removed_adjacent_polygon[i]);
        }
    }
    return ret_polygon;
}

// consider wrap it to a class,and keep a pwa pool attribute to avoid memory allocate!
PolygonTypef32 simplify_polygon_points_better(const PolygonTypef32& polygon,
                                              float                 altitude_threshold) {
    if (polygon.size() <= 1) {
        return polygon;
    }
    // make the uinque...
    PolygonTypef32 removed_adjacent_polygon;
    removed_adjacent_polygon.reserve(polygon.size());
    Coordinate2df32 last_point = polygon[0];
    removed_adjacent_polygon.push_back(polygon[0]);
    for (size_t i = 1; i < polygon.size(); ++i) {
        if (polygon[i] != last_point) {
            removed_adjacent_polygon.push_back(polygon[i]);
            last_point = polygon[i];
        }
    }
    if (last_point == polygon[0]) {
        // delete the last!
        removed_adjacent_polygon.resize(removed_adjacent_polygon.size() - 1);
    }
    if (removed_adjacent_polygon.size() <= 3) {
        return removed_adjacent_polygon;
    }

    int                           n             = removed_adjacent_polygon.size();
    Coordinate2df32               prev_point    = removed_adjacent_polygon[n - 1];
    Coordinate2df32               current_point = removed_adjacent_polygon[0];
    PointWithAreaf32*             prev_pwa      = nullptr;
    PointWithAreaf32*             first_pwa     = nullptr;
    std::vector<PointWithAreaf32> pwa_pool;
    pwa_pool.reserve(n);
    PointWithAreaf32* _start_ptr = pwa_pool.data();

    // building the queue
    for (int i = 0; i < n; ++i) {
        Coordinate2df32 next_point = removed_adjacent_polygon[(i + 1) % n];
        double          area       = calculate_triple_area(prev_point, current_point, next_point);
        pwa_pool.emplace_back(current_point, area);
        PointWithAreaf32* pwa = _start_ptr + i;
        pwa->set_prev(prev_pwa);
        if (prev_pwa != nullptr) {
            prev_pwa->set_next(pwa);
        }
        prev_pwa      = pwa;
        prev_point    = current_point;
        current_point = next_point;
        if (i == n - 1) {
            pwa->set_next(first_pwa);
            first_pwa->set_prev(pwa);
        } else if (i == 0) {
            first_pwa = pwa;
        }
    }

    double max_area = 0;
    int    min_size = FISH_MAX(n / 100, 3);

    std::set<Coordinate2df32>      remove_points;
    int                            remain_size = n;
    int                            search_idx  = 0;
    std::vector<PointWithAreaf32*> pwa_ptrs;
    pwa_ptrs.reserve(pwa_pool.size());
    for (size_t i = 0; i < pwa_pool.size(); ++i) {
        pwa_ptrs.push_back(&pwa_pool[i]);
    }
    // this code is error!
    while (remain_size > min_size) {
        int min_area_idx = search_idx;
        for (int i = search_idx + 1; i < pwa_ptrs.size(); ++i) {
            if (pwa_ptrs[i]->get_area() < pwa_ptrs[min_area_idx]->get_area()) {
                min_area_idx = i;
            }
        }
        // means that the search idx only access once,we will access search idx + 1 at last time!
        //  swap the value of min_area_idx and search idx!
        PointWithAreaf32* pwa  = pwa_ptrs[min_area_idx];
        pwa_ptrs[min_area_idx] = pwa_ptrs[search_idx];
        pwa_ptrs[search_idx]   = pwa;

        auto& _next_p = pwa->get_next()->p;
        auto& _prev_p = pwa->get_prev()->p;
        float dx      = _next_p.x - _prev_p.x;
        float dy      = _next_p.y - _prev_p.y;
        // avoid divide zero!
        double dist     = std::sqrt(dx * dx + dy * dy) + 1e-9;
        double altitude = pwa->get_area() * 2 / dist;
        if (altitude > altitude_threshold) {
            break;
        }
        if (pwa->get_area() < max_area) {
            pwa->set_area(max_area);
        } else {
            max_area = pwa->get_area();
        }
        remove_points.insert(pwa->get_point_ref());
        prev_pwa                   = pwa->get_prev();
        PointWithAreaf32* next_pwa = pwa->get_next();
        prev_pwa->set_next(next_pwa);
        next_pwa->set_prev(prev_pwa);

        // this is unsafe!
        prev_pwa->update_area();
        next_pwa->update_area();
        // adjust the priority!
        ++search_idx;
        --remain_size;
    }
    if (remove_points.size() == 0) {
        return removed_adjacent_polygon;
    }
    PolygonTypef32 ret_polygon;
    size_t         simplified_size = removed_adjacent_polygon.size() - remove_points.size();
    ret_polygon.reserve(simplified_size);
    for (size_t i = 0; i < removed_adjacent_polygon.size(); ++i) {
        if (remove_points.find(removed_adjacent_polygon[i]) == remove_points.end()) {
            ret_polygon.push_back(removed_adjacent_polygon[i]);
        }
    }
    return ret_polygon;
}


void PolygonSimplifier::simplify_impl(const PolygonTypef32& original_polygon,
                                      PolygonTypef32& out_polygon, float altitude_threshold) {
    // clear the data maybe invoked from last time!
    clear_resource();
    out_polygon.clear();
    size_t vertex_size = original_polygon.size();
    if (vertex_size == 1) {
        // maybe the same in/out
        if (&out_polygon == &original_polygon) {
            return;
        }
        out_polygon.resize(vertex_size);
        out_polygon[0] = original_polygon[0];
        return;
    }
    // remove the point which have same coordinate!
    unique_vertices.reserve(vertex_size);
    // Coordinate2df32 prev_vertex = original_polygon[0];
    unique_vertices.push_back(original_polygon[0]);
    for (size_t i = 1; i < vertex_size; ++i) {
        if (original_polygon[i] != original_polygon[i - 1]) {
            unique_vertices.push_back(original_polygon[i]);
        }
    }
    if (original_polygon[vertex_size - 1] == original_polygon[0]) {
        // just ignore the last vertex!
        unique_vertices.resize(unique_vertices.size() - 1);
    }
    size_t unique_vertex_size = unique_vertices.size();
    // the point is too samll!
    if (unique_vertex_size <= 3) {
        out_polygon.assign(unique_vertices.begin(), unique_vertices.end());
        return;
    }

    pwa_pool.reserve(unique_vertex_size);
    PointWithAreaf32* _start_ptr = pwa_pool.data();

    // the first vertex!
    float area = calculate_triple_area(
        unique_vertices[unique_vertex_size - 1], unique_vertices[0], unique_vertices[1]);
    pwa_pool.emplace_back(unique_vertices[0], area);
    // the memory is contingous...
    pwa_pool[0].set_next(_start_ptr + 1);
    pwa_pool[0].set_prev(_start_ptr + unique_vertex_size - 1);

    // the normal vertex!
    for (size_t i = 1; i < vertex_size - 1; ++i) {
        // but the divide is very slow!
        float area = calculate_triple_area(
            unique_vertices[i - 1], unique_vertices[i], unique_vertices[i + 1]);
        pwa_pool.emplace_back(unique_vertices[i], area);
        // this is unsafe!
        PointWithAreaf32* pwa = _start_ptr + i;
        pwa->set_prev(_start_ptr + i - 1);
        pwa->set_next(_start_ptr + i + 1);
    }

    // the last vertex
    area = calculate_triple_area(unique_vertices[unique_vertex_size - 2],
                                 unique_vertices[unique_vertex_size - 1],
                                 unique_vertices[0]);
    pwa_pool.emplace_back(unique_vertices[unique_vertex_size - 1], area);
    pwa_pool[unique_vertex_size - 1].set_prev(_start_ptr + unique_vertex_size - 2);
    pwa_pool[unique_vertex_size - 1].set_next(_start_ptr);

    float max_area = 0.0f;

    size_t min_size     = FISH_MIN(unique_vertex_size / 100, 3);
    size_t remain_size  = unique_vertex_size;
    size_t search_index = 0;

    // construct the linked list!
    pwa_linked_list.reserve(unique_vertex_size);
    for (size_t i = 0; i < unique_vertex_size; ++i) {
        pwa_linked_list.push_back(_start_ptr + i);
    }

    // simplify the polygon until the vertex size is small enough!
    // size_t remove_vertex_size = 0;
    while (remain_size > min_size) {
        // find the polygon which have min area!
        size_t min_area_index = search_index;
        for (size_t i = search_index + 1; i < unique_vertex_size; ++i) {
            if (pwa_linked_list[i]->get_area() < pwa_linked_list[min_area_index]->get_area()) {
                min_area_index = i;
            }
        }
        // swap the two nodes!
        PointWithAreaf32* pwa           = pwa_linked_list[min_area_index];
        pwa_linked_list[min_area_index] = pwa_linked_list[search_index];
        pwa_linked_list[search_index]   = pwa;

        auto& next_vertex = pwa->get_next()->p;
        auto& prev_vertex = pwa->get_prev()->p;

        float dx = next_vertex.x - prev_vertex.x;
        float dy = next_vertex.y - prev_vertex.y;

        float dist     = std::sqrt(dx * dx + dy * dy) + 1e-9;
        float altitude = pwa->get_area() * 2 / dist;
        if (altitude > altitude_threshold) {
            break;
        }

        // this code is useful?
        if (pwa->get_area() < max_area) {
            pwa->set_area(max_area);
        } else {
            max_area = pwa->get_area();
        }

        // update the linked list!
        pwa->flag = REMOVE_VERTEX;

        PointWithAreaf32* prev_pwa = pwa->get_prev();
        PointWithAreaf32* next_pwa = pwa->get_next();
        prev_pwa->set_next(next_pwa);
        next_pwa->set_prev(prev_pwa);

        // update the area unsafe?
        prev_pwa->update_area();
        next_pwa->update_area();
        ++search_index;
        --remain_size;
    }

    size_t simplified_size = unique_vertex_size - search_index;
    out_polygon.reserve(simplified_size);

    for (size_t i = 0; i < unique_vertex_size; ++i) {
        if (pwa_pool[i].flag == KEEP_VERTEX) {
            out_polygon.push_back(pwa_pool[i].get_point_ref());
        }
    }
    // clear the simplifier resource!
}



PolygonTypef32 PolygonSimplifier::simplify_impl(const PolygonTypef32& original_polygon,
                                                float                 altitude_threshold) {
    PolygonTypef32 out_polygon;
    simplify_impl(original_polygon, out_polygon, altitude_threshold);
    return out_polygon;
}


}   // namespace shape_simplier
}   // namespace segmentation
}   // namespace fish