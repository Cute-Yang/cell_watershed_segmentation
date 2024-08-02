#include "core/base.h"
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
    "Unknown"};

const char* get_error_msg(Status::ErrorCode err) {
    // try to cast the err
    size_t error_index = static_cast<size_t>(err);
    return ErrorCodeStr[error_index];
}
}   // namespace Status
}   // namespace base
}   // namespace core
}   // namespace fish