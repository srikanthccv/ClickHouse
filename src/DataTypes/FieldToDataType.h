#pragma once

#include <memory>
#include <Core/Types.h>
#include <Core/Field.h>
#include <Common/FieldVisitors.h>


namespace DB
{

class IDataType;
using DataTypePtr = std::shared_ptr<const IDataType>;


/** For a given Field returns the minimum data type that allows this value to be stored.
  * Note that you still have to convert Field to corresponding data type before inserting to columns
  *  (for example, this is necessary to convert elements of Array to common type).
  */
class FieldToDataType : public StaticVisitor<DataTypePtr>
{
public:
    FieldToDataType(bool allow_convertion_to_string_ = false)
      : allow_convertion_to_string(allow_convertion_to_string_)
    {
    }

    DataTypePtr operator() (const Null & x) const;
    DataTypePtr operator() (const UInt64 & x) const;
    DataTypePtr operator() (const UInt128 & x) const;
    DataTypePtr operator() (const Int64 & x) const;
    DataTypePtr operator() (const Int128 & x) const;
    DataTypePtr operator() (const UUID & x) const;
    DataTypePtr operator() (const Float64 & x) const;
    DataTypePtr operator() (const String & x) const;
    DataTypePtr operator() (const Array & x) const;
    DataTypePtr operator() (const Tuple & tuple) const;
    DataTypePtr operator() (const Map & map) const;
    DataTypePtr operator() (const Object & map) const;
    DataTypePtr operator() (const DecimalField<Decimal32> & x) const;
    DataTypePtr operator() (const DecimalField<Decimal64> & x) const;
    DataTypePtr operator() (const DecimalField<Decimal128> & x) const;
    DataTypePtr operator() (const DecimalField<Decimal256> & x) const;
    DataTypePtr operator() (const AggregateFunctionStateData & x) const;
    DataTypePtr operator() (const UInt256 & x) const;
    DataTypePtr operator() (const Int256 & x) const;
    DataTypePtr operator() (const bool & x) const;

private:
    bool allow_convertion_to_string;
};

}
