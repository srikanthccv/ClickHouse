#include <Interpreters/ExpressionActions.h>
#include <DataStreams/ExpressionBlockInputStream.h>


namespace DB
{

ExpressionBlockInputStream::ExpressionBlockInputStream(const BlockInputStreamPtr & input, const ExpressionActionsPtr & expression_)
    : expression(expression_)
{
    children.push_back(input);
    cached_header = children.back()->getHeader();
    expression->execute(cached_header, true);
}

String ExpressionBlockInputStream::getName() const { return "Expression"; }

Block ExpressionBlockInputStream::getTotals()
{
    totals = children.back()->getTotals();
    expression->executeOnTotals(totals);

    return totals;
}

Block ExpressionBlockInputStream::getHeader() const
{
    return cached_header.cloneEmpty();
}

Block ExpressionBlockInputStream::readImpl()
{
    if (!initialized)
    {
        if (expression->resultIsAlwaysEmpty())
            return {};

        initialized = true;
    }

    Block res = children.back()->read();
    if (res)
        expression->execute(res);
    return res;
}

Block SplittingExpressionBlockInputStream::readImpl()
{
    if (!initialized)
    {
        if (expression->resultIsAlwaysEmpty())
            return {};

        initialized = true;
    }

    Block res;
    if (likely(!not_processed))
    {
        res = children.back()->read();
        if (!res)
            return res;
    }
    else
        res.swap(not_processed);

    action_number = expression->execute(res, action_number, not_processed);
    return res;
}

}
