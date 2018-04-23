#pragma once

#include <Functions/IFunction.h>

#include <Interpreters/ExpressionActions.h>

namespace DB
{

struct LLVMSharedData;

struct LLVMSharedDataPtr : std::shared_ptr<LLVMSharedData>
{
    // just like `IFunctionBase::compile` accepting `llvm::IRBuilderBase`, this weird wrapper exists to allow
    // other code not to depend on LLVM headers.
    LLVMSharedDataPtr();

    // also, this is not a destructor because it's probably not `noexcept`.
    void finalize();
};

// second array is of `char` because `LLVMPreparedFunction::executeImpl` can't use a `std::vector<bool>` for this
using LLVMCompiledFunction = void(const void ** inputs, const char * is_constant, void * output, size_t block_size);

class LLVMPreparedFunction : public PreparedFunctionImpl
{
    std::shared_ptr<const IFunctionBase> parent;
    LLVMSharedDataPtr context;
    LLVMCompiledFunction * function;

public:
    LLVMPreparedFunction(LLVMSharedDataPtr context, std::shared_ptr<const IFunctionBase> parent);

    String getName() const override { return parent->getName(); }

    // TODO: more efficient implementation for constants
    bool useDefaultImplementationForConstants() const override { return true; }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result) override
    {
        size_t block_size = 0;
        std::vector<const void *> columns(arguments.size());
        std::vector<char> is_const(arguments.size());
        for (size_t i = 0; i < arguments.size(); i++)
        {
            auto * column = block.getByPosition(arguments[i]).column.get();
            if (column->size())
                // assume the column is a `ColumnVector<T>`. there's probably no good way to actually
                // check that at runtime, so let's just hope it's always true for columns containing types
                // for which `LLVMSharedData::toNativeType` returns non-null.
                columns[i] = column->getDataAt(0).data;
            is_const[i] = column->isColumnConst();
            block_size = column->size();
        }
        auto col_res = parent->createResultColumn(block_size);
        if (!col_res->isColumnConst() && !col_res->isDummy() && block_size)
            function(columns.data(), is_const.data(), const_cast<char *>(col_res->getDataAt(0).data), block_size);
        block.getByPosition(result).column = std::move(col_res);
    };
};

class LLVMFunction : public IFunctionBase, std::enable_shared_from_this<LLVMFunction>
{
    ExpressionActions::Actions actions; // all of them must have type APPLY_FUNCTION
    Names arg_names;
    DataTypes arg_types;
    LLVMSharedDataPtr context;

    LLVMFunction(ExpressionActions::Actions actions, Names arg_names, DataTypes arg_types, LLVMSharedDataPtr context)
        : actions(std::move(actions)), arg_names(std::move(arg_names)), arg_types(std::move(arg_types)), context(context)
    {}

public:
    static std::shared_ptr<LLVMFunction> create(ExpressionActions::Actions actions, LLVMSharedDataPtr context);

    String getName() const override { return actions.back().result_name; }

    const Names & getArgumentNames() const { return arg_names; }

    const DataTypes & getArgumentTypes() const override { return arg_types; }

    const DataTypePtr & getReturnType() const override { return actions.back().function->getReturnType(); }

    PreparedFunctionPtr prepare(const Block &) const override { return std::make_shared<LLVMPreparedFunction>(context, shared_from_this()); }

    IColumn::Ptr createResultColumn(size_t size) const override { return actions.back().function->createResultColumn(size); }

    bool isDeterministic() override
    {
        for (const auto & action : actions)
            if (!action.function->isDeterministic())
                return false;
        return true;
    }

    bool isDeterministicInScopeOfQuery() override
    {
        for (const auto & action : actions)
            if (!action.function->isDeterministicInScopeOfQuery())
                return false;
        return true;
    }

    // TODO: these methods require reconstructing the call tree:
    // bool isSuitableForConstantFolding() const;
    // bool isInjective(const Block & sample_block);
    // bool hasInformationAboutMonotonicity() const;
    // Monotonicity getMonotonicityForRange(const IDataType & type, const Field & left, const Field & right) const;
};

}
