#pragma once
#include "core/mat.h"
#include "image_proc/polygon.h"
#include <cstdlib>
#include <type_traits>
#include <vector>
namespace fish {
namespace segmentation {
namespace shape_simplier {
using namespace fish::image_proc::polygon;

// the return type of area only can be float/double!
template<class T>
using TripleAreaLimit =
    typename std::enable_if<std::is_same<float, T>::value || std::is_same<T, double>::value,
                            T>::type;

// using shoelace to compute the area of triangle!
template<class T1, class T2 = float, typename = TripleAreaLimit<T2>>
T2 calculate_triple_area(const GenericCoordinate2d<T1>& p1, const GenericCoordinate2d<T1>& p2,
                         const GenericCoordinate2d<T1>& p3) {
    T1 value = std::abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)));
    T2 area  = 0.5 * static_cast<T2>(value);
    return area;
}


PolygonTypef32 simplify_polygon_points(const PolygonTypef32& polygon, float altitude_threshold);

PolygonTypef32 simplify_polygon_points_better(const PolygonTypef32& polygon,
                                              float                 altitude_threshold);

constexpr float area_epsilon_f32 = 1e-7f;

// for double,we set a smaller epsilon!
constexpr double area_epsilon_f64 = 1e-9;

template<class T> struct AreaEpsilonTraits {
    static constexpr T area_epsilon = 0;
};

template<> struct AreaEpsilonTraits<float> {
    static constexpr float area_epsilon = area_epsilon_f32;
};

template<> struct AreaEpsilonTraits<double> {
    static constexpr double area_epsilon = area_epsilon_f64;
};

constexpr int KEEP_VERTEX   = 1;
constexpr int REMOVE_VERTEX = 0;

template<class T> struct PointWithArea {
    using PointType = GenericCoordinate2d<T>;
    PointWithArea* prev;
    PointWithArea* next;
    PointType      p;
    float          area;
    // whether to keep
    int flag;

    PointWithArea(const GenericCoordinate2d<T>& p_, float area_)
        : p(p_)
        , area(area_)
        , flag(KEEP_VERTEX) {}

    PointWithArea(T x, T y, float area_)
        : p(x, y)
        , area(area_)
        , flag(KEEP_VERTEX) {}

    void set_area(float area_) { area = area_; }

    GenericCoordinate2d<T>& get_point_ref() { return p; }
    GenericCoordinate2d<T>& get_point_cref() const { return p; }

    void update_area() {
        area = calculate_triple_area<float, float>(prev->get_point_ref(), p, next->get_point_ref());
    }

    T get_x() const { return p.x; }

    T get_y() const { return p.y; }

    double get_area() const { return area; }

    void set_prev(PointWithArea<T>* prev_) { prev = prev_; }

    void set_next(PointWithArea<T>* next_) { next = next_; }

    // get the pointer of previous node!
    PointWithArea<T>* get_prev() { return prev; }

    // get the pointer of next node!
    PointWithArea<T>* get_next() { return next; }

    // we may need to add a precision like 1e-7?
    // small enough?
    bool operator<(const PointWithArea<T>& rhs) { return (rhs.area - area) > area_epsilon_f32; }

    bool operator==(const PointWithArea<T>& rhs) {
        return std::abs(area - rhs.area) < area_epsilon_f32;
    }

    bool operator>(const PointWithArea<T>& rhs) { return (area - rhs.area) > area_epsilon_f32; }
};

using PointWithAreaf32 = PointWithArea<float>;
using PointWithAreaf64 = PointWithArea<double>;
using PointWithAreai32 = PointWithArea<int32_t>;

// we want to reuse the buffer!
class PolygonSimplifier {
private:
    // the data of each node!
    std::vector<PointWithAreaf32> pwa_pool;
    // the node ptrs....
    std::vector<PointWithAreaf32*> pwa_linked_list;
    // use this to storde the unique vertex...
    PolygonTypef32 unique_vertices;

    void clear_resource() {
        pwa_pool.clear();
        pwa_linked_list.clear();
        unique_vertices.clear();
    }

public:
    PolygonSimplifier() {}

    // you can specify the size to get better performence!
    PolygonSimplifier(size_t pool_size) {
        pwa_pool.reserve(pool_size);
        pwa_linked_list.reserve(pool_size);
        unique_vertices.reserve(pool_size);
    }

    PolygonSimplifier(const PolygonSimplifier& rhs)
        : pwa_pool(rhs.pwa_pool)
        , pwa_linked_list(rhs.pwa_linked_list)
        , unique_vertices(rhs.unique_vertices) {}

    PolygonSimplifier(PolygonSimplifier&& rhs) noexcept
        : pwa_pool(std::move(rhs.pwa_pool))
        , pwa_linked_list(std::move(rhs.pwa_linked_list))
        , unique_vertices(std::move(rhs.unique_vertices)) {}

    PolygonSimplifier& operator=(const PolygonSimplifier& rhs) = delete;

    PolygonSimplifier& operator=(PolygonSimplifier&& rhs) noexcept = delete;

    PolygonTypef32 simplify_impl(const PolygonTypef32& original_polygon, float altitude_threshold);

    void simplify_impl(const PolygonTypef32& original_polygon, PolygonTypef32& out_polygon,
                       float altitude_threshold);
    // write the simplified result to original polygon!
    void simplify_impl_inplace(PolygonTypef32& original_polygon, float altitude_threshold);
};
}   // namespace shape_simplier
}   // namespace segmentation
}   // namespace fish