#include <Parsers/ASTTransactionControl.h>
#include <IO/Operators.h>
#include <Common/SipHash.h>

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

void ASTTransactionControl::formatImpl(const FormatSettings & format /*state*/, FormatState &, FormatStateStacked /*frame*/) const
{
    switch (action)
    {
        case BEGIN:
            format.ostr << (format.hilite ? hilite_keyword : "") << "BEGIN TRANSACTION" << (format.hilite ? hilite_none : "");
            break;
        case COMMIT:
            format.ostr << (format.hilite ? hilite_keyword : "") << "COMMIT" << (format.hilite ? hilite_none : "");
            break;
        case ROLLBACK:
            format.ostr << (format.hilite ? hilite_keyword : "") << "ROLLBACK" << (format.hilite ? hilite_none : "");
            break;
    }
}

void ASTTransactionControl::updateTreeHashImpl(SipHash & hash_state) const
{
    hash_state.update(action);
}

}
