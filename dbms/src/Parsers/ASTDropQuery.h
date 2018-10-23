#pragma once

#include <Parsers/ASTQueryWithTableAndOutput.h>
#include <Parsers/ASTQueryWithOnCluster.h>


namespace DB
{

/** DROP query
  */
class ASTDropQuery : public ASTQueryWithTableAndOutput, public ASTQueryWithOnCluster
{
public:
    enum Kind
    {
        Drop,
        Detach,
        Truncate,
    };

    Kind kind;
    bool if_exists{false};

    /** Get the text that identifies this element. */
    String getID() const override;
    ASTPtr clone() const override;

    ASTPtr getRewrittenASTWithoutOnCluster(const std::string & new_database) const override;

protected:
    void formatQueryImpl(const FormatSettings & settings, FormatState &, FormatStateStacked) const override;
};

}
