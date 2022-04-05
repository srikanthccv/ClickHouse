#pragma once

#include <mutex>
#include <vector>
#include <IO/ReadBuffer.h>
#include <Common/PODArray.h>
#include "Interpreters/IJoin.h"
#include <Core/SortDescription.h>
#include <Processors/Chunk.h>
#include <Processors/Merges/Algorithms/IMergingAlgorithm.h>
#include <Processors/Merges/IMergingTransform.h>
#include <base/logger_useful.h>
#include <Core/SortCursor.h>

namespace Poco { class Logger; }

namespace DB
{

class IJoin;
using JoinPtr = std::shared_ptr<IJoin>;


/*
 * Wrapper for SortCursorImpl
 * It is used to store information about the current state of the cursor.
 */
class FullMergeJoinCursor
{
public:

    FullMergeJoinCursor(const Columns & columns, const SortDescription & desc)
        : impl(columns, desc)
    {
    }

    SortCursor getCursor()
    {
        return SortCursor(&impl);
    }

    /*
    /// Expects !atEnd()
    size_t getEqualLength() const
    {
        size_t pos = impl.getRow() + 1;
        for (; pos < impl.rows; ++pos)
            if (!samePrev(pos))
                break;
        return pos - impl.getRow();
    }

    /// Expects lhs_pos > 0
    bool ALWAYS_INLINE samePrev(size_t lhs_pos) const
    {
        for (size_t i = 0; i < impl.sort_columns_size; ++i)
            if (impl.sort_columns[i]->compareAt(lhs_pos - 1, lhs_pos, *(impl.sort_columns[i]), 1) != 0)
                return false;
        return true;
    }
    */

    SortCursorImpl & getImpl()
    {
        return impl;
    }

private:
    SortCursorImpl impl;
    // bool has_left_nullable = false;
    // bool has_right_nullable = false;
};

/*
 * This class is used to join chunks from two sorted streams.
 * It is used in MergeJoinTransform.
 */
class MergeJoinAlgorithm final : public IMergingAlgorithm
{
public:
    explicit MergeJoinAlgorithm(JoinPtr table_join, const Blocks & input_headers);

    virtual void initialize(Inputs inputs) override;
    virtual void consume(Input & input, size_t source_num) override;
    virtual Status merge() override;

private:
    SortDescription left_desc;
    SortDescription right_desc;

    std::vector<Input> current_inputs;
    std::vector<FullMergeJoinCursor> cursors;

    std::vector<Chunk> sample_chunks;

    bool left_stream_finished = false;
    bool right_stream_finished = false;

    JoinPtr table_join;
    Poco::Logger * log;
};

class MergeJoinTransform final : public IMergingTransform<MergeJoinAlgorithm>
{
public:
    MergeJoinTransform(
        JoinPtr table_join,
        const Blocks & input_headers,
        const Block & output_header,
        UInt64 limit_hint = 0);

    String getName() const override { return "MergeJoinTransform"; }

protected:
    void onFinish() override;
    UInt64 elapsed_ns = 0;

    Poco::Logger * log;
};

}
