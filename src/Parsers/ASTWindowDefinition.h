#pragma once

#include <Parsers/IAST.h>


namespace DB
{

struct ASTWindowDefinition : public IAST
{
    ASTPtr partition_by;

    ASTPtr order_by;

    ASTPtr clone() const override;

    String getID(char delimiter) const override;
};

struct ASTWindowListElement : public IAST
{
    String name;

    // ASTWindowDefinition
    ASTPtr definition;

    ASTPtr clone() const override;

    String getID(char delimiter) const override;
};

}
