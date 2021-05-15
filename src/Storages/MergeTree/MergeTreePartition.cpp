#include <Storages/MergeTree/MergeTreePartition.h>
#include <Storages/MergeTree/MergeTreeData.h>
#include <Storages/MergeTree/IMergeTreeDataPart.h>
#include <IO/HashingWriteBuffer.h>
#include <Common/FieldVisitors.h>
#include <DataTypes/DataTypeDate.h>
#include <DataTypes/DataTypeTuple.h>
#include <Columns/ColumnTuple.h>
#include <Common/SipHash.h>
#include <Common/typeid_cast.h>
#include <Common/hex.h>
#include <Core/Block.h>


namespace DB
{
namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

static std::unique_ptr<ReadBufferFromFileBase> openForReading(const DiskPtr & disk, const String & path)
{
    return disk->readFile(path, std::min(size_t(DBMS_DEFAULT_BUFFER_SIZE), disk->getFileSize(path)));
}

String MergeTreePartition::getID(const MergeTreeData & storage) const
{
    return getID(storage.getInMemoryMetadataPtr()->getPartitionKey().sample_block);
}

/// NOTE: This ID is used to create part names which are then persisted in ZK and as directory names on the file system.
/// So if you want to change this method, be sure to guarantee compatibility with existing table data.
String MergeTreePartition::getID(const Block & partition_key_sample) const
{
    if (value.size() != partition_key_sample.columns())
        throw Exception("Invalid partition key size: " + toString(value.size()), ErrorCodes::LOGICAL_ERROR);

    if (value.empty())
        return "all"; /// It is tempting to use an empty string here. But that would break directory structure in ZK.

    /// In case all partition fields are represented by integral types, try to produce a human-readable ID.
    /// Otherwise use a hex-encoded hash.
    bool are_all_integral = true;
    for (const Field & field : value)
    {
        if (field.getType() != Field::Types::UInt64 && field.getType() != Field::Types::Int64)
        {
            are_all_integral = false;
            break;
        }
    }

    String result;

    if (are_all_integral)
    {
        FieldVisitorToString to_string_visitor;
        for (size_t i = 0; i < value.size(); ++i)
        {
            if (i > 0)
                result += '-';

            if (typeid_cast<const DataTypeDate *>(partition_key_sample.getByPosition(i).type.get()))
                result += toString(DateLUT::instance().toNumYYYYMMDD(DayNum(value[i].safeGet<UInt64>())));
            else
                result += applyVisitor(to_string_visitor, value[i]);

            /// It is tempting to output DateTime as YYYYMMDDhhmmss, but that would make partition ID
            /// timezone-dependent.
        }

        return result;
    }

    SipHash hash;
    FieldVisitorHash hashing_visitor(hash);
    for (const Field & field : value)
        applyVisitor(hashing_visitor, field);

    char hash_data[16];
    hash.get128(hash_data);
    result.resize(32);
    for (size_t i = 0; i < 16; ++i)
        writeHexByteLowercase(hash_data[i], &result[2 * i]);

    return result;
}

void MergeTreePartition::serializeText(const MergeTreeData & storage, WriteBuffer & out, const FormatSettings & format_settings) const
{
    auto metadata_snapshot = storage.getInMemoryMetadataPtr();
    const auto & partition_key_sample = metadata_snapshot->getPartitionKey().sample_block;
    size_t key_size = partition_key_sample.columns();

    if (key_size == 0)
    {
        writeCString("tuple()", out);
    }
    else if (key_size == 1)
    {
        const DataTypePtr & type = partition_key_sample.getByPosition(0).type;
        auto column = type->createColumn();
        column->insert(value[0]);
        type->getDefaultSerialization()->serializeText(*column, 0, out, format_settings);
    }
    else
    {
        DataTypes types;
        Columns columns;
        for (size_t i = 0; i < key_size; ++i)
        {
            const auto & type = partition_key_sample.getByPosition(i).type;
            types.push_back(type);
            auto column = type->createColumn();
            column->insert(value[i]);
            columns.push_back(std::move(column));
        }

        auto tuple_serialization = DataTypeTuple(types).getDefaultSerialization();
        auto tuple_column = ColumnTuple::create(columns);
        tuple_serialization->serializeText(*tuple_column, 0, out, format_settings);
    }
}

void MergeTreePartition::load(const MergeTreeData & storage, const DiskPtr & disk, const String & part_path)
{
    auto metadata_snapshot = storage.getInMemoryMetadataPtr();
    if (!metadata_snapshot->hasPartitionKey())
        return;

    const auto & partition_key_sample = metadata_snapshot->getPartitionKey().sample_block;
    auto partition_file_path = part_path + "partition.dat";
    auto file = openForReading(disk, partition_file_path);
    value.resize(partition_key_sample.columns());
    for (size_t i = 0; i < partition_key_sample.columns(); ++i)
    {
        if (partition_key_sample.getByPosition(i).name.starts_with("modulo"))
        {
            SerializationNumber<Int8>().deserializeBinary(value[i], *file);
        }
        else
        {
            partition_key_sample.getByPosition(i).type->getDefaultSerialization()->deserializeBinary(value[i], *file);
        }
    }
}

void MergeTreePartition::store(const MergeTreeData & storage, const DiskPtr & disk, const String & part_path, MergeTreeDataPartChecksums & checksums) const
{
    auto metadata_snapshot = storage.getInMemoryMetadataPtr();
    const auto & partition_key_sample = metadata_snapshot->getPartitionKey().sample_block;
    store(partition_key_sample, disk, part_path, checksums);
}

void MergeTreePartition::store(const Block & partition_key_sample, const DiskPtr & disk, const String & part_path, MergeTreeDataPartChecksums & checksums) const
{
    if (!partition_key_sample)
        return;

    auto out = disk->writeFile(part_path + "partition.dat");
    HashingWriteBuffer out_hashing(*out);
    for (size_t i = 0; i < value.size(); ++i)
    {
        if (partition_key_sample.getByPosition(i).name.starts_with("modulo"))
        {
            SerializationNumber<Int8>().serializeBinary(value[i], out_hashing);
        }
        else
        {
            partition_key_sample.getByPosition(i).type->getDefaultSerialization()->serializeBinary(value[i], out_hashing);
        }
    }

    out_hashing.next();
    checksums.files["partition.dat"].file_size = out_hashing.count();
    checksums.files["partition.dat"].file_hash = out_hashing.getHash();
    out->finalize();
}

void MergeTreePartition::create(const StorageMetadataPtr & metadata_snapshot, Block block, size_t row, ContextPtr context)
{
    if (!metadata_snapshot->hasPartitionKey())
        return;

    auto partition_key_sample_block = executePartitionByExpression(metadata_snapshot, block, context);
    size_t partition_columns_num = partition_key_sample_block.columns();
    value.resize(partition_columns_num);

    for (size_t i = 0; i < partition_columns_num; ++i)
    {
        const auto & column_name = partition_key_sample_block.getByPosition(i).name;
        const auto & partition_column = block.getByName(column_name).column;
        partition_column->get(row, value[i]);
    }
}

Block MergeTreePartition::executePartitionByExpression(const StorageMetadataPtr & metadata_snapshot, Block & block, ContextPtr context)
{
    const auto & partition_key = metadata_snapshot->getPartitionKey();
    bool adjusted_expression = false;
    auto modulo_to_modulo_legacy = [&](ASTFunction * function_ast)
    {
        if (function_ast->name == "modulo")
        {
            function_ast->name = "moduloLegacy";
            adjusted_expression = true;
        }
    };
    ASTPtr new_ast = partition_key.definition_ast->clone();
    if (auto * function_ast = typeid_cast<ASTFunction *>(new_ast.get()))
    {
        if (function_ast->name == "tuple")
        {
            auto children = function_ast->arguments->children;
            for (auto child : children)
            {
                if (auto * child_function_ast = typeid_cast<ASTFunction *>(child.get()))
                    modulo_to_modulo_legacy(child_function_ast);
            }
        }
        else
            modulo_to_modulo_legacy(function_ast);

        if (adjusted_expression)
        {
            auto adjusted_partition_key = KeyDescription::getKeyFromAST(new_ast, metadata_snapshot->columns, context);
            adjusted_partition_key.expression->execute(block);
            return adjusted_partition_key.sample_block;
        }
    }
    partition_key.expression->execute(block);
    return partition_key.sample_block;
}
}
