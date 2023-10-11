#pragma once

#include <Parsers/IAST.h>
#include <optional>

namespace DB
{

/// Either a (possibly compound) expression representing a partition value or a partition ID.
class ASTPartition : public IAST
{
public:
    ASTPtr value;
    std::optional<size_t> fields_count;

    String id;
    bool all = false;

    String getID(char) const override;
    ASTPtr clone() const override;

protected:
    void formatImpl(const FormatSettings & settings, FormatState & state, FormatStateStacked frame) const override;
};

}
