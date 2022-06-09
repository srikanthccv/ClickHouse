#pragma once

#include <Parsers/IParserBase.h>
#include <Parsers/Kusto/ParserKQLQuery.h>

namespace DB
{
class ParserKQLSummarize : public ParserKQLBase
{
public:
    ASTPtr group_expression_list;

protected:
    const char * getName() const override { return "KQL summarize"; }
    bool parseImpl(Pos & pos, ASTPtr & node, Expected & expected) override;
    std::pair<String, String> removeLastWord(String input);
    String getBinGroupbyString(String exprBin);
};

}
