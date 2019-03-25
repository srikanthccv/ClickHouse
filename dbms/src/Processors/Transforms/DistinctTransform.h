#pragma once
#include <Processors/ISimpleTransform.h>
#include <DataStreams/SizeLimits.h>
#include <Core/ColumnNumbers.h>
#include <Interpreters/SetVariants.h>

namespace DB
{

class DistinctTransform : ISimpleTransform
{
public:
    DistinctTransform(
        const Block & header,
        const SizeLimits & set_size_limits,
        UInt64 limit_hint,
        const Names & columns);

    String getName() const override { return "DistinctTransform"; }

protected:
    void transform(Chunk & chunk) override;

private:
    ColumnNumbers key_columns_pos;
    SetVariants data;
    Sizes key_sizes;
    UInt64 limit_hint;

    bool no_more_rows = false;

    /// Restrictions on the maximum size of the output data.
    SizeLimits set_size_limits;

    template <typename Method>
    void buildFilter(
        Method & method,
        const ColumnRawPtrs & key_columns,
        IColumn::Filter & filter,
        size_t rows,
        SetVariants & variants) const;
};

}
