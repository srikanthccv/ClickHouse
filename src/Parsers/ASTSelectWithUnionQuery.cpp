#include <Parsers/ASTSelectWithUnionQuery.h>
#include <Parsers/ASTSelectQuery.h>
#include <Common/typeid_cast.h>

#include <iostream>

namespace DB
{

ASTPtr ASTSelectWithUnionQuery::clone() const
{
    auto res = std::make_shared<ASTSelectWithUnionQuery>(*this);
    res->children.clear();

    res->list_of_selects = list_of_selects->clone();
    res->children.push_back(res->list_of_selects);

    res->union_modes.insert(res->union_modes.begin(), union_modes.begin(), union_modes.end());
    res->flatten_nodes_list = flatten_nodes_list->clone();

    cloneOutputOptions(*res);
    return res;
}


void ASTSelectWithUnionQuery::formatQueryImpl(const FormatSettings & settings, FormatState & state, FormatStateStacked frame) const
{
    std::cout << "\n\nin format \n\n";
    std::string indent_str = settings.one_line ? "" : std::string(4 * frame.indent, ' ');

#if 0
    auto mode_to_str = [&](auto mode)
    {
        if (mode == Mode::Unspecified)
            return "";
        else if (mode == Mode::ALL)
            return "ALL";
        else
            return "DISTINCT";
    };
#endif

    for (ASTs::const_iterator it = flatten_nodes_list->children.begin(); it != flatten_nodes_list->children.end(); ++it)
    {
        if (it != list_of_selects->children.begin())
            settings.ostr << settings.nl_or_ws << indent_str << (settings.hilite ? hilite_keyword : "") << "UNION "
                          // << mode_to_str(union_modes[it - list_of_selects->children.begin() - 1]) << (settings.hilite ? hilite_none : "")
                          << settings.nl_or_ws;

        (*it)->formatImpl(settings, state, frame);
    }
    std::cout << "\n\nafter format \n\n";
}

}
