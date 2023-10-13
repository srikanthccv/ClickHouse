#include <DataTypes/DataTypesNumber.h>
#include <Functions/DateTimeTransforms.h>
#include <Functions/FunctionDateOrDateTimeToSomething.h>
#include <Functions/FunctionFactory.h>


namespace DB
{

using FunctionToDaysSinceYearZero = FunctionDateOrDateTimeToSomething<DataTypeUInt32, ToDaysSinceYearZeroImpl>;

REGISTER_FUNCTION(ToDaysSinceYearZero)
{
    factory.registerFunction<FunctionToDaysSinceYearZero>(FunctionDocumentation{
        .description = R"(
Returns for the number of days passed since 1 January 0000 in the proleptic Gregorian calendar defined by ISO 8601, a date or date with time.
The calculation is opposite to MySQL's TO_DAYS() function.
)",
        .examples{{"typical", "SELECT toDaysSinceYearZero(toDate('2023-09-08'))", "713569"}},
        .categories{"Dates and Times"}});

    /// MySQL compatibility alias.
    // factory.registerAlias("TO_DAYS", FunctionToDaysSinceYearZero::name, FunctionFactory::CaseInsensitive);
}

}