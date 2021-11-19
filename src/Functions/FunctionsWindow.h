#pragma once

#include <base/DateLUT.h>
#include <DataTypes/DataTypeInterval.h>
#include <Functions/IFunction.h>

namespace DB
{

/** Window functions:
  *
  * TUMBLE(time_attr, interval [, timezone])
  *
  * TUMBLE_START(window_id)
  *
  * TUMBLE_START(time_attr, interval [, timezone])
  *
  * TUMBLE_END(window_id)
  *
  * TUMBLE_END(time_attr, interval [, timezone])
  *
  * HOP(time_attr, hop_interval, window_interval [, timezone])
  *
  * HOP_START(window_id)
  *
  * HOP_START(time_attr, hop_interval, window_interval [, timezone])
  *
  * HOP_END(window_id)
  *
  * HOP_END(time_attr, hop_interval, window_interval [, timezone])
  *
  */
enum WindowFunctionName
{
    TUMBLE,
    TUMBLE_START,
    TUMBLE_END,
    HOP,
    HOP_START,
    HOP_END,
    WINDOW_ID
};

template <IntervalKind::Kind unit>
struct ToStartOfTransform;

#define TRANSFORM_DATE(INTERVAL_KIND) \
    template <> \
    struct ToStartOfTransform<IntervalKind::INTERVAL_KIND> \
    { \
        static ExtendedDayNum execute(UInt32 t, UInt64 delta, const DateLUTImpl & time_zone) \
        { \
            return time_zone.toStartOf##INTERVAL_KIND##Interval(time_zone.toDayNum(t), delta); \
        } \
    };
    TRANSFORM_DATE(Year)
    TRANSFORM_DATE(Quarter)
    TRANSFORM_DATE(Month)
    TRANSFORM_DATE(Week)
#undef TRANSFORM_DATE

    template <>
    struct ToStartOfTransform<IntervalKind::Day>
    {
        static UInt32 execute(UInt32 t, UInt64 delta, const DateLUTImpl & time_zone)
        {
            return time_zone.toStartOfDayInterval(time_zone.toDayNum(t), delta);
        }
    };

#define TRANSFORM_TIME(INTERVAL_KIND) \
    template <> \
    struct ToStartOfTransform<IntervalKind::INTERVAL_KIND> \
    { \
        static UInt32 execute(UInt32 t, UInt64 delta, const DateLUTImpl & time_zone) \
        { \
            return time_zone.toStartOf##INTERVAL_KIND##Interval(t, delta); \
        } \
    };
    TRANSFORM_TIME(Hour)
    TRANSFORM_TIME(Minute)
    TRANSFORM_TIME(Second)
#undef TRANSFORM_DATE

    template <IntervalKind::Kind unit>
    struct AddTime;

#define ADD_DATE(INTERVAL_KIND) \
    template <> \
    struct AddTime<IntervalKind::INTERVAL_KIND> \
    { \
        static ExtendedDayNum execute(UInt16 d, UInt64 delta, const DateLUTImpl & time_zone) \
        { \
            return time_zone.add##INTERVAL_KIND##s(ExtendedDayNum(d), delta); \
        } \
    };
    ADD_DATE(Year)
    ADD_DATE(Quarter)
    ADD_DATE(Month)
#undef ADD_DATE

    template <>
    struct AddTime<IntervalKind::Week>
    {
        static ExtendedDayNum execute(UInt16 d, UInt64 delta, const DateLUTImpl &) { return ExtendedDayNum(d + 7 * delta);}
    };

#define ADD_TIME(INTERVAL_KIND, INTERVAL) \
    template <> \
    struct AddTime<IntervalKind::INTERVAL_KIND> \
    { \
        static UInt32 execute(UInt32 t, Int64 delta, const DateLUTImpl &) { return t + INTERVAL * delta; } \
    };
    ADD_TIME(Day, 86400)
    ADD_TIME(Hour, 3600)
    ADD_TIME(Minute, 60)
    ADD_TIME(Second, 1)
#undef ADD_TIME

template <WindowFunctionName type>
struct WindowImpl
{
    static constexpr auto name = "UNKNOWN";

    static DataTypePtr getReturnType(const ColumnsWithTypeAndName & arguments, const String & function_name);

    static ColumnPtr dispatchForColumns(const ColumnsWithTypeAndName & arguments, const String & function_name);
};

template <WindowFunctionName type>
class FunctionWindow : public IFunction
{
public:
    static constexpr auto name = WindowImpl<type>::name;
    static FunctionPtr create(ContextPtr) { return std::make_shared<FunctionWindow>(); }
    String getName() const override { return name; }
    bool isVariadic() const override { return true; }
    size_t getNumberOfArguments() const override { return 0; }
    bool useDefaultImplementationForConstants() const override { return true; }
    ColumnNumbers getArgumentsThatAreAlwaysConstant() const override { return {1, 2, 3}; }
    bool isSuitableForShortCircuitArgumentsExecution(const DataTypesWithConstInfo &) const override { return true; }

    DataTypePtr getReturnTypeImpl(const ColumnsWithTypeAndName & arguments) const override;

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & /*result_type*/, size_t /*input_rows_count*/) const override;
};

using FunctionTumble = FunctionWindow<TUMBLE>;
using FunctionTumbleStart = FunctionWindow<TUMBLE_START>;
using FunctionTumbleEnd = FunctionWindow<TUMBLE_END>;
using FunctionHop = FunctionWindow<HOP>;
using FunctionWindowId = FunctionWindow<WINDOW_ID>;
using FunctionHopStart = FunctionWindow<HOP_START>;
using FunctionHopEnd = FunctionWindow<HOP_END>;
}
