#pragma once

#include <Functions/IFunction.h>

#include <Interpreters/ExpressionActions.h>

namespace DB
{

class LLVMContext
{
    struct Data;
    std::shared_ptr<Data> shared;

public:
    LLVMContext();

    void finalize();

    bool isCompilable(const IFunctionBase& function) const;

    Data * operator->() const {
        return shared.get();
    }
};

/// second array is of `char` because `LLVMPreparedFunction::executeImpl` can't use a `std::vector<bool>` for this
using LLVMCompiledFunction = void(const void ** inputs, const char * is_constant, void * output, size_t block_size);

class LLVMPreparedFunction : public PreparedFunctionImpl
{
    std::shared_ptr<const IFunctionBase> parent;
    LLVMContext context;
    LLVMCompiledFunction * function;

public:
    LLVMPreparedFunction(LLVMContext context, std::shared_ptr<const IFunctionBase> parent);

    String getName() const override { return parent->getName(); }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result) override;
};

class LLVMFunction : public std::enable_shared_from_this<LLVMFunction>, public IFunctionBase
{
    /// all actions must have type APPLY_FUNCTION
    ExpressionActions::Actions actions;
    Names arg_names;
    DataTypes arg_types;
    LLVMContext context;

public:
    LLVMFunction(ExpressionActions::Actions actions, LLVMContext context);

    String getName() const override { return actions.back().result_name; }

    const Names & getArgumentNames() const { return arg_names; }

    const DataTypes & getArgumentTypes() const override { return arg_types; }

    const DataTypePtr & getReturnType() const override { return actions.back().function->getReturnType(); }

    PreparedFunctionPtr prepare(const Block &) const override { return std::make_shared<LLVMPreparedFunction>(context, shared_from_this()); }

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

    /// TODO: these methods require reconstructing the call tree:
    /*
    bool isSuitableForConstantFolding() const;
    bool isInjective(const Block & sample_block);
    bool hasInformationAboutMonotonicity() const;
    Monotonicity getMonotonicityForRange(const IDataType & type, const Field & left, const Field & right) const;
    */
};

}
