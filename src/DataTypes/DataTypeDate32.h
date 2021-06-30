#pragma once

#include <DataTypes/DataTypeNumberBase.h>

namespace DB
{
class DataTypeDate32 final : public DataTypeNumberBase<Int32>
{
public:
    static constexpr auto family_name = "Date32";

    TypeIndex getTypeId() const override { return TypeIndex::Date32; }
    const char * getFamilyName() const override { return family_name; }

    bool canBeUsedAsVersion() const override { return true; }
    bool canBeInsideNullable() const override { return true; }

    bool equals(const IDataType & rhs) const override;

protected:
    SerializationPtr doGetDefaultSerialization() const override;
};
}
