#pragma once
#include "common/fishdef.h"
#include "core/base.h"
#include "core/mat.h"
#include "utils/logging.h"
#include <array>
#include <cstddef>
#include <functional>
#include <queue>
#include <stdexcept>
#include <vector>

namespace fish {
namespace segmentation {
namespace watershed {
enum class NeighborConnectiveType : uint8_t { Conn8 = 0, Conn4 = 1 };
using namespace fish::core::mat;
constexpr uint8_t IS_QUEUED  = 1;
constexpr uint8_t NOT_QUEUED = 0;
template<class T> struct PixelWithValue {
    int    x;
    int    y;
    T      value;
    size_t count;
    PixelWithValue(int x_, int y_, T value_, size_t count_)
        : x(x_)
        , y(y_)
        , value(value_)
        , count(count_) {}
    PixelWithValue() = delete;
    PixelWithValue(const PixelWithValue<T>& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , value(rhs.value)
        , count(rhs.count) {}

    PixelWithValue(PixelWithValue<T>&& rhs)
        : x(rhs.x)
        , y(rhs.y)
        , value(rhs.value)
        , count(rhs.count) {}
    PixelWithValue<T>& operator=(const PixelWithValue<T>& rhs) {
        x     = rhs.x;
        y     = rhs.y;
        value = rhs.value;
        count = rhs.count;
        return *this;
    }

    PixelWithValue<T>& operator=(PixelWithValue<T>&& rhs) {
        x     = rhs.x;
        y     = rhs.y;
        value = rhs.value;
        count = rhs.count;
        return *this;
    }

    // 在相同的像素值下,前面添加的元素表示越高优先级(count越小表示越大)
    // the std::less requires the lhs and rhs is const!
    bool less(const PixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            // 所以这里比较是相反的(在我们实际的程序中,不可能出现两个count值一样的数据)
            return count > rhs.count;
        }
        return value < rhs.value;
    }


    // 后面添加的元素,表示具有较低优先级(count越大表示越小)
    bool greater(const PixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            return count < rhs.count;
        }
        return value > rhs.value;
    }

    bool equal(const PixelWithValue<T>& rhs) const {
        return (value == rhs.value && count == rhs.count && x == rhs.x && y == rhs.x);
    }
    bool operator<(const PixelWithValue<T>& rhs) const { return less(rhs); }
    bool operator>(const PixelWithValue<T>& rhs) const { return greater(rhs); }
    bool operator==(const PixelWithValue<T>& rhs) const { return equal(rhs); }
};
using FloatPixelWithValue       = PixelWithValue<float>;
using UCharPixelWithValue       = PixelWithValue<unsigned char>;
using UIntPixelWithValue        = PixelWithValue<unsigned int>;
using UShortPixelWithValue      = PixelWithValue<unsigned short>;
using float_pixel_with_value_t  = FloatPixelWithValue;
using uchar_pixel_with_value_t  = UCharPixelWithValue;
using uint_pixel_with_value_t   = UIntPixelWithValue;
using ushort_pixel_with_value_t = UShortPixelWithValue;


// if T==float,the sizeof pxiel is 4 + 4+8 = 16 bytes
// if restore x,y,will pad to 24 bytes
template<class T> struct BetterPixelWithValue {
    uint32_t index;
    T        value;
    size_t   count;
    BetterPixelWithValue(uint32_t index_, T value_, size_t count)
        : index(index_)
        , value(value_)
        , count(count) {}
    BetterPixelWithValue()                                   = delete;
    BetterPixelWithValue(const BetterPixelWithValue<T>& rhs) = default;
    BetterPixelWithValue(BetterPixelWithValue<T>&& rhs)      = default;
    bool less(const BetterPixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            // the count never equal!
            return count > rhs.count;
        }
        return value < rhs.value;
    }
    bool equal(const BetterPixelWithValue<T>& rhs) const {
        return (value == rhs.value && count == rhs.count && index == rhs.index);
    }
    bool greater(const BetterPixelWithValue<T>& rhs) const {
        if (value == rhs.value) {
            return count < rhs.count;
        }
        return value > rhs.value;
    }

    // 一定保证左操作数 op 右操作数时 返回true
    bool operator<(const BetterPixelWithValue<T>& rhs) const { return less(rhs); }
    bool operator>(const BetterPixelWithValue<T>& rhs) const { return greater(rhs); }
    bool operator==(const BetterPixelWithValue<T>& rhs) const { return equal(rhs); }
};

using BetterFloatPixelWithValue        = BetterPixelWithValue<float>;
using BetterUCharPixelWithValue        = BetterPixelWithValue<unsigned char>;
using BetterUIntPixelWithValue         = BetterPixelWithValue<unsigned int>;
using BetterUShortPixelWithValue       = BetterPixelWithValue<unsigned short>;
using better_float_pixel_with_value_t  = BetterFloatPixelWithValue;
using better_uchar_pixel_with_value_t  = BetterUCharPixelWithValue;
using better_uint_pixel_with_value_t   = BetterUIntPixelWithValue;
using better_ushort_pixel_with_value_t = BetterUShortPixelWithValue;

// 理论上只需要1/8内存
class SaveMemoryLogicalMask {
private:
    int                  height;
    int                  width;
    std::vector<uint8_t> mask;

public:
    SaveMemoryLogicalMask(int height_, int width_)
        : height(height_)
        , width(width_)
        // 向上取整,防止越界
        , mask((height_ * width_ + 7) / 8, 0) {}

    template<bool value> void set_value(int x, int y) {
        int block_idx = (y * width + x) / 8;
        int bit_idx   = y * width + x - block_idx * 8;
        // 将对应的bit位设置成1,其他位置不关心,所以|运算
        if constexpr (value) {
            mask[block_idx] |= bit_flags[bit_idx];
        } else {
            // 将对应的bit位射0,同时其他位置不关心
            mask[block_idx] &= (~bit_flags[bit_idx]);
        }
    }

    bool get_value(int x, int y) {
        int block_idx = (y * width + x) / 8;
        int bit_idx   = y * width + x - block_idx * 8;
        // 获取对应的bit位,其余设置成0,如果为0,表示该bit位位0,否则为1
        return (mask[block_idx] & bit_flags[bit_idx]) != 0;
    }

public:
    // static constexpr std::array<uint8_t, 8> bit_flags = {
    //     0b1, 0b10, 0b100, 0b1000, 0b10000, 0b100000, 0b1000000, 0b1000000};
    static constexpr std::array<uint8_t, 8> bit_flags = {1, 2, 4, 8, 16, 32, 64, 128};
};


template<class T> class WatershedQueueWrapper {
public:
    using pixel_ref_t       = PixelWithValue<T>&;
    using const_pixel_ref_t = const PixelWithValue<T>&;
    using pixel_ptr_t       = PixelWithValue<T>*;
    using const_pixel_ptr_t = const PixelWithValue<T>*;

private:
    std::priority_queue<PixelWithValue<T>, std::vector<PixelWithValue<T>>,
                        std::less<PixelWithValue<T>>>
        pixel_queue;
    // 用来指示(x,y)处的元素是否已经进入过队列
    //  std::vector<uint8_t> queued;
    ImageMat<uint8_t> queued;
    // 计数器,用来表示像素的优先级
    size_t counter;

public:
    // construct deafult,the queded matrix allocate the new buffer!
    WatershedQueueWrapper()
        : pixel_queue()   // 这种构造方式可能会带来严重扩容问题
        , queued()
        , counter(0) {
        LOG_INFO("you must invoke the intialize function to initialize it!");
    }

    // construct with given buffer for queued,to save the buffer!
    WatershedQueueWrapper(uint8_t* queued_buf, int height, int width)
        : pixel_queue()
        , queued(height, width, 1, queued_buf, MatMemLayout::LayoutRight, false)
        , counter(0) {
        if (height <= 0 || width <= 0 || queued_buf == nullptr) {
            // raise an exception!
            throw std::runtime_error("fail to initialize the watershed queue with given buffer!");
        }
        LOG_INFO(
            "initialize the queued matrix with given buffer,be sure it is valid before we use it!");
    }

    void set_shared_queued_buffer(uint8_t* shared_queued_buffer, int height, int width) {
        if (!queued.empty()) {
            LOG_INFO("remove the onwership of queued mat...");
        }
        queued.set_shared_buffer(height, width, 1, shared_queued_buffer, MatMemLayout::LayoutRight);
    }
    // 指定底层容器的容量,避免扩容
    //  delete copy
    WatershedQueueWrapper(const WatershedQueueWrapper<T>& rhs) = delete;
    WatershedQueueWrapper(WatershedQueueWrapper<T>&& rhs)      = delete;

    WatershedQueueWrapper<T>& operator=(const WatershedQueueWrapper<T>& rhs) = delete;
    WatershedQueueWrapper<T>& operator=(WatershedQueueWrapper<T>&& rhs)      = delete;

    template<class MarkerType>
    void initialize(const ImageMat<T>& image, const ImageMat<MarkerType>& marker, T min_threshold,
                    float estimate_enqueue_rate) {
        constexpr MarkerType marker_zero = static_cast<MarkerType>(0);
        // 可以获取到其 container
        int height   = image.get_height();
        int width    = image.get_width();
        int channels = image.get_channels();
        // check the channles whether equal to 1!
        if (channels != 1) {
            LOG_ERROR("the watershed only suppot single channel image now...");
            return;
        }
        // allocate 0.2 x element size as init space
        std::vector<PixelWithValue<T>> queue_container;
        if (estimate_enqueue_rate <= 0.0f || estimate_enqueue_rate >= 1.0f) {
            LOG_WARN("get invalid estimate_enqueue_rate %f,we will set 0.2 as default!",
                     estimate_enqueue_rate);
            estimate_enqueue_rate = 0.2f;
        }
        size_t estimate_enqueue_size = static_cast<float>(height * width) * estimate_enqueue_rate;
        queue_container.reserve(estimate_enqueue_size);
        std::priority_queue<PixelWithValue<T>> temp_queue(std::less<PixelWithValue<T>>(),
                                                          std::move(queue_container));
        LOG_INFO("allocate {} elements for our quque container!", estimate_enqueue_size);

        // the image maybe have same shape with input image if we give speciyf data ptr!
        if (queued.compare_shape(image)) {
            LOG_INFO("the queued matrix already have same shape with input image,maybe you give me "
                     "a shared memory^_^");
        } else {
            LOG_INFO("resize the queued matrix to shape ({},{})", height, width);
            int queued_height = queued.get_height();
            int queued_width  = queued.get_width();
            if (queued_height > 0 && queued_width > 0) {
                LOG_INFO("the queued have shape({},{}) which is not equal to given image,so we "
                         "will resize it and allocate new memory!",
                         queued_height,
                         queued_width);
            }
            queued.resize(height, width, 1, true);
        }
        // fill with not queued!
        queued.fill_with_value(NOT_QUEUED);
        // this case should never happend!
        if (height == 1 && width == 1) [[unlikely]] {
            LOG_INFO("got image with shape(1,1) which is unexpected!");
            if (image(0, 0) <= min_threshold || marker(0, 0) != marker_zero) {
                queued(0, 0) = IS_QUEUED;
            } else {
                queued(0, 0) = IS_QUEUED;
                temp_queue.emplace(0, 0, image(0, 0), counter);
                ++counter;
            }
        }
        if (height == 1) [[unlikely]] {
            // if miss neigh,and the value > threshold,we will push it always
            LOG_INFO("enqueue with shape(1,{})", width);
            for (int x = 0; x < width - 1; ++x) {
                if (image(0, x) <= min_threshold) {
                    queued(0, x) = IS_QUEUED;
                } else {
                    if (marker(0, x) != marker_zero) {
                        queued(0, 0) = IS_QUEUED;
                    } else {
                        queued(0, 0) = IS_QUEUED;
                        temp_queue.emplace(x, 0, image(0, x), counter);
                        ++counter;
                    }
                }
            }
        } else if (width == 1) [[unlikely]] {
            // process the first point
            for (int y = 0; y < height; ++y) {
                if (image(y, 0) <= min_threshold) {
                    queued(y, 0) = IS_QUEUED;
                } else {
                    if (marker(y, 0) != marker_zero) {
                        queued(y, 0) = IS_QUEUED;
                    } else {
                        queued(y, 0) = IS_QUEUED;
                        temp_queue.emplace(0, y, image(y, 0), counter);
                        ++counter;
                    }
                }
            }
        } else {
            LOG_INFO("enqueue with image shape ({},{})", height, width);
            // the first row...
            for (int x = 0; x < width; ++x) {
                if (image(0, x) <= min_threshold) {
                    queued(0, x) = IS_QUEUED;
                } else {
                    if (marker(0, x) != marker_zero) {
                        queued(0, x) = IS_QUEUED;
                    } else {
                        queued(0, x) = IS_QUEUED;
                        temp_queue.emplace(x, 0, image(0, x), counter);
                        ++counter;
                    }
                }
            }

            for (int y = 1; y < height - 1; ++y) {
                // handle special value y,0
                if (image(y, 0) <= min_threshold) {
                    queued(y, 0) = IS_QUEUED;
                } else {
                    if (marker(y, 0) != marker_zero) {
                        queued(y, 0) = IS_QUEUED;
                    } else {
                        queued(y, 0) = IS_QUEUED;
                        temp_queue.emplace(0, y, image(y, 0), counter);
                        ++counter;
                    }
                }

                for (int x = 1; x < width - 1; ++x) {
                    if (image(y, x) <= min_threshold) {
                        queued(y, x) = IS_QUEUED;
                    } else {
                        if (marker(y, x) != marker_zero) {
                            queued(y, x) = IS_QUEUED;
                            // this case have left,right,top,bottom neigh!
                        } else if (marker(y, x + 1) != marker_zero ||
                                   marker(y, x - 1) != marker_zero ||
                                   marker(y - 1, x) != marker_zero ||
                                   marker(y + 1, x) != marker_zero) {
                            queued(y, x) = IS_QUEUED;
                            temp_queue.emplace(x, y, image(y, x), counter);
                            ++counter;
                        }
                    }
                }
                // process y,width -1
                if (image(y, width - 1) <= min_threshold) {
                    queued(y, width - 1) = IS_QUEUED;
                } else {
                    if (marker(y, width - 1) != marker_zero) {
                        queued(y, width - 1) = IS_QUEUED;
                    } else {
                        queued(y, width - 1) = IS_QUEUED;
                        temp_queue.emplace(width - 1, y, image(y, width - 1), counter);
                        ++counter;
                    }
                }
            }
            // here if the point miss neigh,will return float.NAN,it is not equal to the zero!
            for (int x = 0; x < width; ++x) {
                if (image(height - 1, x) <= min_threshold) {
                    queued(height - 1, x) = IS_QUEUED;
                } else {
                    if (marker(height - 1, x) != marker_zero) {
                        queued(height - 1, x) = IS_QUEUED;
                    } else {
                        queued(height - 1, x) = IS_QUEUED;
                        temp_queue.emplace(x, height - 1, image(height - 1, x), counter);
                        ++counter;
                    }
                }
            }
        }
        // swap the queue and temp queue
        LOG_INFO("the eneuque point num is {}", counter);
        // swap pixel_queue and temp queue!
        pixel_queue.swap(temp_queue);
    }

    FISH_ALWAYS_INLINE void add(int x, int y, T value) {
        if (queued(y, x) == NOT_QUEUED) {
            pixel_queue.emplace(x, y, value, counter);
            ++counter;
            queued(y, x) = IS_QUEUED;
        }
    }

    FISH_ALWAYS_INLINE bool can_add_to_queue(int x, int y) { return queued(y, x) == NOT_QUEUED; }

    // this function will return the ref of our value!
    FISH_ALWAYS_INLINE const PixelWithValue<T>& get_top_pixel() { return pixel_queue.top(); }
    // this function return the clone value,is safe!
    FISH_ALWAYS_INLINE PixelWithValue<T> get_top_pixel_clone() { return pixel_queue.top(); }
    FISH_ALWAYS_INLINE void              remove_top_pixel() { pixel_queue.pop(); }

    // will copy....
    PixelWithValue<T> get_top_pixel_safe() {
        PixelWithValue<T> pixel = pixel_queue.top();
        pixel_queue.pop();
    }

    bool is_empty() { return pixel_queue.empty(); }
};


template<class T1, class T2, typename = image_dtype_limit_t<T1>, typename = image_dtype_limit_t<T2>>
Status::ErrorCode watershed_transform(const ImageMat<T1>& image, ImageMat<T2>& marker,
                                      T1 min_threshold, bool conn8, uint8_t* shared_queued_buffer);

// if false,means that all neightbors are background value or neighbors are different!
// if ture,means,the none background values are same
// return the fg_same?
}   // namespace watershed
}   // namespace segmentation
}   // namespace fish