#include "core/base.h"
#include "utils/logging.h"
#include <array>

namespace fish {
namespace core {
namespace base {
namespace Status {

constexpr std::array<const char*, ErrorCode::CodeNum> ErrorCodeStr = {
    "Ok",
    "InvalidMatDimension",
    "InvalidMatIndex",
    "InvalidGuassianParam",
    "InvalidMatShape",
    "MatShapeMismatch",
    "MatLayoutMismatch",
    "InvalidConvKernel",
    "InvalidRankFilterRadius",
    "UnsupportedRankFilterType",
    " UnsupportedNeighborFilterType",
    " UnsupportedValueOp",
    "InvalidMatChannle",
    "InvokeInplace",
    "WatershedSegmentationError",
    "Unknown",
    "UnexpectedMatLayout"};

const char* get_error_msg(Status::ErrorCode err) {
    // try to cast the err
    size_t error_index = static_cast<size_t>(err);
    return ErrorCodeStr[error_index];
}

// check the index!
const char* get_error_msg_safe(Status::ErrorCode err) {
    constexpr size_t array_size  = ErrorCodeStr.size();
    size_t           error_index = static_cast<size_t>(err);
    if (error_index >= array_size) {
        LOG_INFO("out of range...");
        return "out of range";
    }
    return ErrorCodeStr[error_index];
}
}   // namespace Status
}   // namespace base
}   // namespace core
}   // namespace fish