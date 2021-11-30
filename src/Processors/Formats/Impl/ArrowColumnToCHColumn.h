#pragma once

#include "config_formats.h"

#if USE_ARROW || USE_ORC || USE_PARQUET

#include <DataTypes/IDataType.h>
#include <Core/ColumnWithTypeAndName.h>
#include <Core/Block.h>
#include <arrow/table.h>


namespace DB
{

class Block;
class Chunk;

class ArrowColumnToCHColumn
{
public:
    using NameToColumnPtr = std::unordered_map<std::string, std::shared_ptr<arrow::ChunkedArray>>;

    ArrowColumnToCHColumn(const Block & header_, const std::string & format_name_, bool import_nested_, bool defaults_for_omitted_fields_);

    /// Constructor that create header by arrow schema. It will be useful for inserting
    /// data from file without knowing table structure.
    ArrowColumnToCHColumn(const arrow::Schema & schema, const std::string & format_name, bool import_nested_, bool defaults_for_omitted_fields_);

    void arrowTableToCHChunk(Chunk & res, std::shared_ptr<arrow::Table> & table);

    void arrowColumnsToCHChunk(Chunk & res, NameToColumnPtr & name_to_column_ptr);

private:
    const Block header;
    const std::string format_name;
    bool import_nested;
    bool defaults_for_omitted_fields;

    /// Map {column name : dictionary column}.
    /// To avoid converting dictionary from Arrow Dictionary
    /// to LowCardinality every chunk we save it and reuse.
    std::unordered_map<std::string, std::shared_ptr<ColumnWithTypeAndName>> dictionary_values;
};

}

#endif
