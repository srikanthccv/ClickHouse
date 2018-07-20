#pragma once

#include <string>
#include <vector>
#include <cstdint>


namespace DB
{

/// Data types for representing elementary values from a database in RAM.

struct Null {};

using UInt8 = uint8_t;
using UInt16 = uint16_t;
using UInt32 = uint32_t;
using UInt64 = uint64_t;

using Int8 = int8_t;
using Int16 = int16_t;
using Int32 = int32_t;
using Int64 = int64_t;
using Int128 = __int128;

using Float32 = float;
using Float64 = double;

using String = std::string;


/** Note that for types not used in DB, IsNumber is false.
  */
template <typename T> constexpr bool IsNumber = false;

template <> constexpr bool IsNumber<UInt8> = true;
template <> constexpr bool IsNumber<UInt16> = true;
template <> constexpr bool IsNumber<UInt32> = true;
template <> constexpr bool IsNumber<UInt64> = true;
template <> constexpr bool IsNumber<Int8> = true;
template <> constexpr bool IsNumber<Int16> = true;
template <> constexpr bool IsNumber<Int32> = true;
template <> constexpr bool IsNumber<Int64> = true;
template <> constexpr bool IsNumber<Float32> = true;
template <> constexpr bool IsNumber<Float64> = true;

template <typename T> struct TypeName;

template <> struct TypeName<UInt8>   { static const char * get() { return "UInt8";   } };
template <> struct TypeName<UInt16>  { static const char * get() { return "UInt16";  } };
template <> struct TypeName<UInt32>  { static const char * get() { return "UInt32";  } };
template <> struct TypeName<UInt64>  { static const char * get() { return "UInt64";  } };
template <> struct TypeName<Int8>    { static const char * get() { return "Int8";    } };
template <> struct TypeName<Int16>   { static const char * get() { return "Int16";   } };
template <> struct TypeName<Int32>   { static const char * get() { return "Int32";   } };
template <> struct TypeName<Int64>   { static const char * get() { return "Int64";   } };
template <> struct TypeName<Int128>  { static const char * get() { return "Int128";   } };
template <> struct TypeName<Float32> { static const char * get() { return "Float32"; } };
template <> struct TypeName<Float64> { static const char * get() { return "Float64"; } };
template <> struct TypeName<String>  { static const char * get() { return "String";  } };


/// Not a data type in database, defined just for convenience.
using Strings = std::vector<String>;

}
