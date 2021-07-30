#include <DataStreams/TTLCalcInputStream.h>
#include <DataTypes/DataTypeDate.h>
#include <Interpreters/inplaceBlockConversions.h>
#include <Interpreters/TreeRewriter.h>
#include <Interpreters/ExpressionAnalyzer.h>
#include <Columns/ColumnConst.h>
#include <Interpreters/addTypeConversionToAST.h>
#include <Storages/TTLMode.h>
#include <Interpreters/Context.h>

#include <DataStreams/TTLDeleteAlgorithm.h>
#include <DataStreams/TTLColumnAlgorithm.h>
#include <DataStreams/TTLAggregationAlgorithm.h>
#include <DataStreams/TTLUpdateInfoAlgorithm.h>

namespace DB
{

TTLCalcInputStream::TTLCalcInputStream(
    const BlockInputStreamPtr & input_,
    const MergeTreeData & storage_,
    const StorageMetadataPtr & metadata_snapshot_,
    const MergeTreeData::MutableDataPartPtr & data_part_,
    time_t current_time_,
    bool force_)
    : data_part(data_part_)
    , log(&Poco::Logger::get(storage_.getLogName() + " (TTLCalcInputStream)"))
{
    children.push_back(input_);
    header = children.at(0)->getHeader();
    auto old_ttl_infos = data_part->ttl_infos;

    if (metadata_snapshot_->hasRowsTTL())
    {
        const auto & rows_ttl = metadata_snapshot_->getRowsTTL();
        algorithms.emplace_back(std::make_unique<TTLUpdateInfoAlgorithm>(
            rows_ttl, old_ttl_infos.table_ttl, current_time_, force_));
    }

    for (const auto & where_ttl : metadata_snapshot_->getRowsWhereTTLs())
        algorithms.emplace_back(std::make_unique<TTLUpdateInfoAlgorithm>(
            where_ttl, old_ttl_infos.rows_where_ttl[where_ttl.result_column], current_time_, force_));

    for (const auto & group_by_ttl : metadata_snapshot_->getGroupByTTLs())
        algorithms.emplace_back(std::make_unique<TTLUpdateInfoAlgorithm>(
            group_by_ttl, old_ttl_infos.group_by_ttl[group_by_ttl.result_column], current_time_, force_));

    if (metadata_snapshot_->hasAnyColumnTTL())
    {
        for (const auto & [name, description] : metadata_snapshot_->getColumnTTLs())
        {
            algorithms.emplace_back(std::make_unique<TTLUpdateInfoAlgorithm>(
                description, old_ttl_infos.columns_ttl[name], current_time_, force_));
        }
    }

    for (const auto & move_ttl : metadata_snapshot_->getMoveTTLs())
        algorithms.emplace_back(std::make_unique<TTLUpdateInfoAlgorithm>(
            move_ttl, old_ttl_infos.moves_ttl[move_ttl.result_column], current_time_, force_));

    for (const auto & recompression_ttl : metadata_snapshot_->getRecompressionTTLs())
        algorithms.emplace_back(std::make_unique<TTLUpdateInfoAlgorithm>(
            recompression_ttl, old_ttl_infos.recompression_ttl[recompression_ttl.result_column], current_time_, force_));
}

Block reorderColumns(Block block, const Block & header)
{
    Block res;
    for (const auto & col : header)
        res.insert(block.getByName(col.name));

    return res;
}

Block TTLCalcInputStream::readImpl()
{
    auto block = children.at(0)->read();
    for (const auto & algorithm : algorithms)
        algorithm->execute(block);

    if (!block)
        return block;

    return reorderColumns(std::move(block), header);
}

void TTLCalcInputStream::readSuffixImpl()
{
    data_part->ttl_infos = {};
    for (const auto & algorithm : algorithms)
        algorithm->finalize(data_part);
}

}
