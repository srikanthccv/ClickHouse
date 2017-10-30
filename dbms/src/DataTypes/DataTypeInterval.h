#pragma once

#include <DataTypes/DataTypeNumberBase.h>


namespace DB
{

/** Data type to deal with INTERVAL in SQL (arithmetic on time intervals).
  *
  * Mostly the same as Int64.
  * But also tagged with interval kind.
  *
  * Intended isage is for temporary elements in expressions,
  *  not for storing values in tables.
  */
class DataTypeInterval final : public DataTypeNumberBase<Int64>
{
public:
    enum Kind
    {
        Second,
        Minute,
        Hour,
        Day,
        Week,
        Month,
        Year
    };

private:
    Kind kind;

public:
    static constexpr bool is_parametric = true;

    Kind getKind() const { return kind; }

    const char * kindToString() const
    {
        switch (kind)
        {
            case Second: return "Second";
            case Minute: return "Minute";
            case Hour: return "Hour";
            case Day: return "Day";
            case Week: return "Week";
            case Month: return "Month";
            case Year: return "Year";
        }
    }

    DataTypeInterval(Kind kind) : kind(kind) {};

    std::string getName() const override { return std::string("Interval") + kindToString(); }
    const char * getFamilyName() const override { return "Interval"; }

    bool behavesAsNumber() const override { return false; }
    bool notForTables() const override { return true; }

    DataTypePtr clone() const override { return std::make_shared<DataTypeInterval>(kind); }
};

}

