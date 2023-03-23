#include <gtest/gtest.h>
#include <arrow/chunked_array.h>
#include <vector>
#include <arrow/util/logging.h>

using namespace DB;

TEST(ChunkedArray, ChunkedArrayWithZeroChunksShouldNotAbort)
{
    std::vector<std::shared_ptr<::arrow::Array>> empty_chunks_vector;

    EXPECT_ANY_THROW(::arrow::ChunkedArray{empty_chunks_vector});

    ::arrow::util::ArrowLog(__FILE__, __LINE__, ::arrow::util::ArrowLogLevel::ARROW_FATAL);
}
