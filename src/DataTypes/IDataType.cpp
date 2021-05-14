#include <Columns/IColumn.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnSparse.h>

#include <Common/Exception.h>
#include <Common/escapeForFileName.h>
#include <Common/SipHash.h>

#include <IO/WriteHelpers.h>
#include <IO/Operators.h>

#include <DataTypes/IDataType.h>
#include <DataTypes/DataTypeCustom.h>
#include <DataTypes/NestedUtils.h>
#include <DataTypes/Serializations/SerializationSparse.h>
#include <DataTypes/Serializations/SerializationInfo.h>
#include <DataTypes/Serializations/SerializationTupleElement.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
    extern const int DATA_TYPE_CANNOT_BE_PROMOTED;
    extern const int ILLEGAL_COLUMN;
}

IDataType::~IDataType() = default;

String IDataType::getName() const
{
    if (custom_name)
    {
        return custom_name->getName();
    }
    else
    {
        return doGetName();
    }
}

String IDataType::doGetName() const
{
    return getFamilyName();
}

void IDataType::updateAvgValueSizeHint(const IColumn & column, double & avg_value_size_hint)
{
    /// Update the average value size hint if amount of read rows isn't too small
    size_t column_size = column.size();
    if (column_size > 10)
    {
        double current_avg_value_size = static_cast<double>(column.byteSize()) / column_size;

        /// Heuristic is chosen so that avg_value_size_hint increases rapidly but decreases slowly.
        if (current_avg_value_size > avg_value_size_hint)
            avg_value_size_hint = std::min(1024., current_avg_value_size); /// avoid overestimation
        else if (current_avg_value_size * 2 < avg_value_size_hint)
            avg_value_size_hint = (current_avg_value_size + avg_value_size_hint * 3) / 4;
    }
}

MutableColumnPtr IDataType::createColumn(const ISerialization & serialization) const
{
    auto column = createColumn();
    if (serialization.getKind() == ISerialization::Kind::SPARSE)
        return ColumnSparse::create(std::move(column));

    return column;
}

ColumnPtr IDataType::createColumnConst(size_t size, const Field & field) const
{
    auto column = createColumn();
    column->insert(field);
    return ColumnConst::create(std::move(column), size);
}


ColumnPtr IDataType::createColumnConstWithDefaultValue(size_t size) const
{
    return createColumnConst(size, getDefault());
}

DataTypePtr IDataType::promoteNumericType() const
{
    throw Exception("Data type " + getName() + " can't be promoted.", ErrorCodes::DATA_TYPE_CANNOT_BE_PROMOTED);
}

size_t IDataType::getSizeOfValueInMemory() const
{
    throw Exception("Value of type " + getName() + " in memory is not of fixed size.", ErrorCodes::LOGICAL_ERROR);
}

DataTypePtr IDataType::tryGetSubcolumnType(const String & subcolumn_name) const
{
    if (subcolumn_name == MAIN_SUBCOLUMN_NAME)
        return shared_from_this();

    return nullptr;
}

DataTypePtr IDataType::getSubcolumnType(const String & subcolumn_name) const
{
    auto subcolumn_type = tryGetSubcolumnType(subcolumn_name);
    if (subcolumn_type)
        return subcolumn_type;

    throw Exception(ErrorCodes::ILLEGAL_COLUMN, "There is no subcolumn {} in type {}", subcolumn_name, getName());
}

ColumnPtr IDataType::getSubcolumn(const String & subcolumn_name, const IColumn &) const
{
    throw Exception(ErrorCodes::ILLEGAL_COLUMN, "There is no subcolumn {} in type {}", subcolumn_name, getName());
}

Names IDataType::getSubcolumnNames() const
{
    NameSet res;
    getDefaultSerialization()->enumerateStreams([&res, this](const ISerialization::SubstreamPath & substream_path)
    {
        ISerialization::SubstreamPath new_path;
        /// Iterate over path to try to get intermediate subcolumns for complex nested types.
        for (const auto & elem : substream_path)
        {
            new_path.push_back(elem);
            auto subcolumn_name = ISerialization::getSubcolumnNameForStream(new_path);
            if (!subcolumn_name.empty() && tryGetSubcolumnType(subcolumn_name))
                res.insert(subcolumn_name);
        }
    });

    return Names(std::make_move_iterator(res.begin()), std::make_move_iterator(res.end()));
}

void IDataType::insertDefaultInto(IColumn & column) const
{
    column.insertDefault();
}

void IDataType::setCustomization(DataTypeCustomDescPtr custom_desc_) const
{
    /// replace only if not null
    if (custom_desc_->name)
        custom_name = std::move(custom_desc_->name);

    if (custom_desc_->serialization)
        custom_serialization = std::move(custom_desc_->serialization);
}

SerializationPtr IDataType::getDefaultSerialization() const
{
    if (custom_serialization)
        return custom_serialization;

    return doGetDefaultSerialization();
}

SerializationPtr IDataType::getSparseSerialization() const
{
    return std::make_shared<SerializationSparse>(getDefaultSerialization());
}

SerializationPtr IDataType::getSubcolumnSerialization(const String & subcolumn_name, const BaseSerializationGetter &) const
{
    throw Exception(ErrorCodes::ILLEGAL_COLUMN, "There is no subcolumn {} in type {}", subcolumn_name, getName());
}

SerializationPtr IDataType::getSerialization(const String & column_name, const SerializationInfo & info) const
{
    ISerialization::Settings settings =
    {
        .num_rows = info.getNumberOfRows(),
        .num_default_rows = info.getNumberOfDefaultRows(column_name),
        .ratio_for_sparse_serialization = info.getRatioForSparseSerialization()
    };

    return getSerialization(settings);
}

SerializationPtr IDataType::getSerialization(const IColumn & column) const
{
    if (column.isSparse())
        return getSparseSerialization();

    ISerialization::Settings settings =
    {
        .num_rows = column.size(),
        .num_default_rows = column.getNumberOfDefaultRows(IColumn::DEFAULT_ROWS_SEARCH_STEP),
        .ratio_for_sparse_serialization = IColumn::DEFAULT_RATIO_FOR_SPARSE_SERIALIZATION
    };

    return getSerialization(settings);
}

SerializationPtr IDataType::getSerialization(const ISerialization::Kinds & kinds) const
{
    if (!kinds.subcolumns.empty())
        throw Exception(ErrorCodes::LOGICAL_ERROR,"Data type {} doesn't support "
            "custom kinds of serialization for subcolumns ot doesn't have subcolumns at all.", getName());

    if (kinds.main == ISerialization::Kind::SPARSE)
        return getSparseSerialization();

    return getDefaultSerialization();
}

SerializationPtr IDataType::getSerialization(const ISerialization::Settings & settings) const
{
    if (supportsSparseSerialization())
    {
        double ratio = settings.num_rows ? std::min(static_cast<double>(settings.num_default_rows) / settings.num_rows, 1.0) : 0.0;
        if (ratio > settings.ratio_for_sparse_serialization)
            return getSparseSerialization();
    }

    return getDefaultSerialization();
}

// static
SerializationPtr IDataType::getSerialization(const NameAndTypePair & column, const IDataType::StreamExistenceCallback & callback)
{
    if (column.isSubcolumn())
    {
        auto base_serialization_getter = [&](const IDataType & subcolumn_type)
        {
            return subcolumn_type.getSerialization(column.name, callback);
        };

        auto type_in_storage = column.getTypeInStorage();
        return type_in_storage->getSubcolumnSerialization(column.getSubcolumnName(), base_serialization_getter);
    }

    return column.type->getSerialization(column.name, callback);
}

SerializationPtr IDataType::getSerialization(const String & column_name, const StreamExistenceCallback & callback) const
{
    if (supportsSparseSerialization())
    {
        auto sparse_idx_name = escapeForFileName(column_name) + ".sparse.idx";
        if (callback(sparse_idx_name))
            return getSparseSerialization();
    }

    return getDefaultSerialization();
}

DataTypePtr IDataType::getTypeForSubstream(const ISerialization::SubstreamPath & substream_path) const
{
    auto type = tryGetSubcolumnType(ISerialization::getSubcolumnNameForStream(substream_path));
    if (type)
        return type->getSubcolumnType(MAIN_SUBCOLUMN_NAME);

    return getSubcolumnType(MAIN_SUBCOLUMN_NAME);
}

void IDataType::enumerateStreams(const SerializationPtr & serialization, const StreamCallbackWithType & callback, ISerialization::SubstreamPath & path) const
{
    serialization->enumerateStreams([&](const ISerialization::SubstreamPath & substream_path)
    {
        callback(substream_path, *getTypeForSubstream(substream_path));
    }, path);
}

}
