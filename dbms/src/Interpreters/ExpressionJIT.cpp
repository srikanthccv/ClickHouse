#include <Interpreters/ExpressionJIT.h>

#if USE_EMBEDDED_COMPILER

#include <optional>

#include <Columns/ColumnConst.h>
#include <Columns/ColumnNullable.h>
#include <Columns/ColumnVector.h>
#include <Common/typeid_cast.h>
#include <DataTypes/DataTypeNullable.h>
#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/Native.h>
#include <Functions/IFunction.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

#include <llvm/Config/llvm-config.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/SectionMemoryManager.h>
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
#include <llvm/ExecutionEngine/Orc/IRCompileLayer.h>
#include <llvm/ExecutionEngine/Orc/NullResolver.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>

#pragma GCC diagnostic pop


/** HACK
  * Allow to link with LLVM that was compiled without RTTI.
  * This is the default option when you build LLVM from sources.
  * We define fake symbols for RTTI to help linker.
  * This assumes that enabling/disabling RTTI doesn't change memory layout of objects
  *  in any significant way and it doesn't affect the code that isn't actually using RTTI.
  * Proper solution: recompile LLVM with enabled RTTI.
  */
extern "C"
{

__attribute__((__weak__)) int _ZTIN4llvm13ErrorInfoBaseE = 0;
__attribute__((__weak__)) int _ZTIN4llvm12MemoryBufferE = 0;

}


namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

namespace
{
    struct ColumnData
    {
        const char * data = nullptr;
        const char * null = nullptr;
        size_t stride;
    };

    struct ColumnDataPlaceholder
    {
        llvm::Value * data_init; /// first row
        llvm::Value * null_init;
        llvm::Value * stride;
        llvm::PHINode * data; /// current row
        llvm::PHINode * null;
    };
}

static ColumnData getColumnData(const IColumn * column)
{
    ColumnData result;
    const bool is_const = column->isColumnConst();
    if (is_const)
        column = &reinterpret_cast<const ColumnConst *>(column)->getDataColumn();
    if (auto * nullable = typeid_cast<const ColumnNullable *>(column))
    {
        result.null = nullable->getNullMapColumn().getRawData().data;
        column = &nullable->getNestedColumn();
    }
    result.data = column->getRawData().data;
    result.stride = is_const ? 0 : column->sizeOfValueIfFixed();
    return result;
}

static void applyFunction(IFunctionBase & function, Field & value)
{
    const auto & type = function.getArgumentTypes().at(0);
    Block block = {{ type->createColumnConst(1, value), type, "x" }, { nullptr, function.getReturnType(), "y" }};
    function.execute(block, {0}, 1, 1);
    block.safeGetByPosition(1).column->get(0, value);
}

struct LLVMContext
{
    llvm::LLVMContext context;
#if LLVM_VERSION_MAJOR >= 7
    llvm::orc::ExecutionSession execution_session;
    std::unique_ptr<llvm::Module> module;
#else
    std::shared_ptr<llvm::Module> module;
#endif
    std::unique_ptr<llvm::TargetMachine> machine;
    llvm::orc::RTDyldObjectLinkingLayer objectLayer;
    llvm::orc::IRCompileLayer<decltype(objectLayer), llvm::orc::SimpleCompiler> compileLayer;
    llvm::DataLayout layout;
    llvm::IRBuilder<> builder;
    std::unordered_map<std::string, void *> symbols;

    LLVMContext()
#if LLVM_VERSION_MAJOR >= 7
        : module(std::make_unique<llvm::Module>("jit", context))
#else
        : module(std::make_shared<llvm::Module>("jit", context))
#endif
        , machine(llvm::EngineBuilder().selectTarget())
#if LLVM_VERSION_MAJOR >= 7
        , objectLayer(execution_session, [](llvm::orc::VModuleKey)
        {
            return llvm::orc::RTDyldObjectLinkingLayer::Resources
            {
                std::make_shared<llvm::SectionMemoryManager>(),
                std::make_shared<llvm::orc::NullResolver>()
            };
        })
#else
        , objectLayer([]() { return std::make_shared<llvm::SectionMemoryManager>(); })
#endif
        , compileLayer(objectLayer, llvm::orc::SimpleCompiler(*machine))
        , layout(machine->createDataLayout())
        , builder(context)
    {
        module->setDataLayout(layout);
        module->setTargetTriple(machine->getTargetTriple().getTriple());
    }

    void finalize()
    {
        if (!module->size())
            return;
        llvm::PassManagerBuilder builder;
        llvm::legacy::FunctionPassManager fpm(module.get());
        builder.OptLevel = 3;
        builder.SLPVectorize = true;
        builder.LoopVectorize = true;
        builder.RerollLoops = true;
        builder.VerifyInput = true;
        builder.VerifyOutput = true;
        builder.populateFunctionPassManager(fpm);
        for (auto & function : *module)
            fpm.run(function);

        /// name, mangled name
        std::vector<std::pair<std::string, std::string>> function_names;
        function_names.reserve(module->size());
        for (const auto & function : *module)
        {
            std::string mangled_name;
            llvm::raw_string_ostream mangled_name_stream(mangled_name);
            llvm::Mangler::getNameWithPrefix(mangled_name_stream, function.getName(), layout);
            function_names.emplace_back(function.getName(), mangled_name);
        }

#if LLVM_VERSION_MAJOR >= 7
        llvm::orc::VModuleKey module_key = execution_session.allocateVModule();
        llvm::cantFail(compileLayer.addModule(module_key, std::move(module)));
#else
        llvm::cantFail(compileLayer.addModule(module, std::make_shared<llvm::orc::NullResolver>()));
#endif

        for (const auto & names : function_names)
            if (auto symbol = compileLayer.findSymbol(names.second, false).getAddress())
                symbols[names.first] = reinterpret_cast<void *>(*symbol);
    }
};

class LLVMPreparedFunction : public PreparedFunctionImpl
{
    std::string name;
    std::shared_ptr<LLVMContext> context;
    void * function;

public:
    LLVMPreparedFunction(std::string name_, std::shared_ptr<LLVMContext> context)
        : name(std::move(name_)), context(context), function(context->symbols.at(name))
    {}

    String getName() const override { return name; }

    bool useDefaultImplementationForNulls() const override { return false; }

    bool useDefaultImplementationForConstants() const override { return true; }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t block_size) override
    {
        auto col_res = block.getByPosition(result).type->createColumn()->cloneResized(block_size);
        if (block_size)
        {
            std::vector<ColumnData> columns(arguments.size() + 1);
            for (size_t i = 0; i < arguments.size(); ++i)
            {
                auto * column = block.getByPosition(arguments[i]).column.get();
                if (!column)
                    throw Exception("Column " + block.getByPosition(arguments[i]).name + " is missing", ErrorCodes::LOGICAL_ERROR);
                columns[i] = getColumnData(column);
            }
            columns[arguments.size()] = getColumnData(col_res.get());
            reinterpret_cast<void (*) (size_t, ColumnData *)>(function)(block_size, columns.data());
        }
        block.getByPosition(result).column = std::move(col_res);
    };
};

static void compileFunction(std::shared_ptr<LLVMContext> & context, const IFunctionBase & f)
{
    auto & arg_types = f.getArgumentTypes();
    auto & b = context->builder;
    auto * size_type = b.getIntNTy(sizeof(size_t) * 8);
    auto * data_type = llvm::StructType::get(b.getInt8PtrTy(), b.getInt8PtrTy(), size_type);
    auto * func_type = llvm::FunctionType::get(b.getVoidTy(), { size_type, data_type->getPointerTo() }, /*isVarArg=*/false);
    auto * func = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, f.getName(), context->module.get());
    auto args = func->args().begin();
    llvm::Value * counter_arg = &*args++;
    llvm::Value * columns_arg = &*args++;

    auto * entry = llvm::BasicBlock::Create(b.getContext(), "entry", func);
    b.SetInsertPoint(entry);
    std::vector<ColumnDataPlaceholder> columns(arg_types.size() + 1);
    for (size_t i = 0; i <= arg_types.size(); ++i)
    {
        auto & type = i == arg_types.size() ? f.getReturnType() : arg_types[i];
        auto * data = b.CreateLoad(b.CreateConstInBoundsGEP1_32(data_type, columns_arg, i));
        columns[i].data_init = b.CreatePointerCast(b.CreateExtractValue(data, {0}), toNativeType(b, removeNullable(type))->getPointerTo());
        columns[i].null_init = type->isNullable() ? b.CreateExtractValue(data, {1}) : nullptr;
        columns[i].stride = b.CreateExtractValue(data, {2});
    }

    /// assume nonzero initial value in `counter_arg`
    auto * loop = llvm::BasicBlock::Create(b.getContext(), "loop", func);
    b.CreateBr(loop);
    b.SetInsertPoint(loop);
    auto * counter_phi = b.CreatePHI(counter_arg->getType(), 2);
    counter_phi->addIncoming(counter_arg, entry);
    for (auto & col : columns)
    {
        col.data = b.CreatePHI(col.data_init->getType(), 2);
        col.data->addIncoming(col.data_init, entry);
        if (col.null_init)
        {
            col.null = b.CreatePHI(col.null_init->getType(), 2);
            col.null->addIncoming(col.null_init, entry);
        }
    }
    ValuePlaceholders arguments(arg_types.size());
    for (size_t i = 0; i < arguments.size(); ++i)
    {
        arguments[i] = [&b, &col = columns[i], &type = arg_types[i]]() -> llvm::Value *
        {
            auto * value = b.CreateLoad(col.data);
            if (!col.null)
                return value;
            auto * is_null = b.CreateICmpNE(b.CreateLoad(col.null), b.getInt8(0));
            auto * nullable = llvm::Constant::getNullValue(toNativeType(b, type));
            return b.CreateInsertValue(b.CreateInsertValue(nullable, value, {0}), is_null, {1});
        };
    }
    auto * result = f.compile(b, std::move(arguments));
    if (columns.back().null)
    {
        b.CreateStore(b.CreateExtractValue(result, {0}), columns.back().data);
        b.CreateStore(b.CreateSelect(b.CreateExtractValue(result, {1}), b.getInt8(1), b.getInt8(0)), columns.back().null);
    }
    else
    {
        b.CreateStore(result, columns.back().data);
    }
    auto * cur_block = b.GetInsertBlock();
    for (auto & col : columns)
    {
        /// currently, stride is either 0 or size of native type
        auto * is_const = b.CreateICmpEQ(col.stride, llvm::ConstantInt::get(size_type, 0));
        col.data->addIncoming(b.CreateSelect(is_const, col.data, b.CreateConstInBoundsGEP1_32(nullptr, col.data, 1)), cur_block);
        if (col.null)
            col.null->addIncoming(b.CreateSelect(is_const, col.null, b.CreateConstInBoundsGEP1_32(nullptr, col.null, 1)), cur_block);
    }
    counter_phi->addIncoming(b.CreateSub(counter_phi, llvm::ConstantInt::get(size_type, 1)), cur_block);

    auto * end = llvm::BasicBlock::Create(b.getContext(), "end", func);
    b.CreateCondBr(b.CreateICmpNE(counter_phi, llvm::ConstantInt::get(size_type, 1)), loop, end);
    b.SetInsertPoint(end);
    b.CreateRetVoid();
}

static llvm::Constant * getNativeValue(llvm::Type * type, const IColumn & column, size_t i)
{
    if (!type)
        return nullptr;
    if (auto * constant = typeid_cast<const ColumnConst *>(&column))
        return getNativeValue(type, constant->getDataColumn(), 0);
    if (auto * nullable = typeid_cast<const ColumnNullable *>(&column))
    {
        auto * value = getNativeValue(type->getContainedType(0), nullable->getNestedColumn(), i);
        auto * is_null = llvm::ConstantInt::get(type->getContainedType(1), nullable->isNullAt(i));
        return value ? llvm::ConstantStruct::get(static_cast<llvm::StructType *>(type), value, is_null) : nullptr;
    }
    if (type->isFloatTy())
        return llvm::ConstantFP::get(type, static_cast<const ColumnVector<Float32> &>(column).getElement(i));
    if (type->isDoubleTy())
        return llvm::ConstantFP::get(type, static_cast<const ColumnVector<Float64> &>(column).getElement(i));
    if (type->isIntegerTy())
        return llvm::ConstantInt::get(type, column.getUInt(i));
    /// TODO: if (type->isVectorTy())
    return nullptr;
}

/// Same as IFunctionBase::compile, but also for constants and input columns.
using CompilableExpression = std::function<llvm::Value * (llvm::IRBuilderBase &, const ValuePlaceholders &)>;

static CompilableExpression subexpression(ColumnPtr c, DataTypePtr type)
{
    return [=](llvm::IRBuilderBase & b, const ValuePlaceholders &) { return getNativeValue(toNativeType(b, type), *c, 0); };
}

static CompilableExpression subexpression(size_t i)
{
    return [=](llvm::IRBuilderBase &, const ValuePlaceholders & inputs) { return inputs[i](); };
}

static CompilableExpression subexpression(const IFunctionBase & f, std::vector<CompilableExpression> args)
{
    return [&, args = std::move(args)](llvm::IRBuilderBase & builder, const ValuePlaceholders & inputs)
    {
        ValuePlaceholders input;
        for (const auto & arg : args)
            input.push_back([&]() { return arg(builder, inputs); });
        auto * result = f.compile(builder, input);
        if (result->getType() != toNativeType(builder, f.getReturnType()))
            throw Exception("Function " + f.getName() + " generated an llvm::Value of invalid type", ErrorCodes::LOGICAL_ERROR);
        return result;
    };
}

class LLVMFunction : public IFunctionBase
{
    std::string name;
    Names arg_names;
    DataTypes arg_types;
    std::shared_ptr<LLVMContext> context;
    std::vector<FunctionBasePtr> originals;
    std::unordered_map<StringRef, CompilableExpression> subexpressions;

public:
    LLVMFunction(const ExpressionActions::Actions & actions, std::shared_ptr<LLVMContext> context, const Block & sample_block)
        : name(actions.back().result_name), context(context)
    {
        for (const auto & c : sample_block)
            /// TODO: implement `getNativeValue` for all types & replace the check with `c.column && toNativeType(...)`
            if (c.column && getNativeValue(toNativeType(context->builder, c.type), *c.column, 0))
                subexpressions[c.name] = subexpression(c.column, c.type);
        for (const auto & action : actions)
        {
            const auto & names = action.argument_names;
            const auto & types = action.function->getArgumentTypes();
            std::vector<CompilableExpression> args;
            for (size_t i = 0; i < names.size(); ++i)
            {
                auto inserted = subexpressions.emplace(names[i], subexpression(arg_names.size()));
                if (inserted.second)
                {
                    arg_names.push_back(names[i]);
                    arg_types.push_back(types[i]);
                }
                args.push_back(inserted.first->second);
            }
            subexpressions[action.result_name] = subexpression(*action.function, std::move(args));
            originals.push_back(action.function);
        }
        compileFunction(context, *this);
    }

    bool isCompilable() const override { return true; }

    llvm::Value * compile(llvm::IRBuilderBase & builder, ValuePlaceholders values) const override { return subexpressions.at(name)(builder, values); }

    String getName() const override { return name; }

    const Names & getArgumentNames() const { return arg_names; }

    const DataTypes & getArgumentTypes() const override { return arg_types; }

    const DataTypePtr & getReturnType() const override { return originals.back()->getReturnType(); }

    PreparedFunctionPtr prepare(const Block &) const override { return std::make_shared<LLVMPreparedFunction>(name, context); }

    bool isDeterministic() override
    {
        for (const auto & f : originals)
            if (!f->isDeterministic())
                return false;
        return true;
    }

    bool isDeterministicInScopeOfQuery() override
    {
        for (const auto & f : originals)
            if (!f->isDeterministicInScopeOfQuery())
                return false;
        return true;
    }

    bool isSuitableForConstantFolding() const override
    {
        for (const auto & f : originals)
            if (!f->isSuitableForConstantFolding())
                return false;
        return true;
    }

    bool isInjective(const Block & sample_block) override
    {
        for (const auto & f : originals)
            if (!f->isInjective(sample_block))
                return false;
        return true;
    }

    bool hasInformationAboutMonotonicity() const override
    {
        for (const auto & f : originals)
            if (!f->hasInformationAboutMonotonicity())
                return false;
        return true;
    }

    Monotonicity getMonotonicityForRange(const IDataType & type, const Field & left, const Field & right) const override
    {
        const IDataType * type_ = &type;
        Field left_ = left;
        Field right_ = right;
        Monotonicity result(true, true, true);
        /// monotonicity is only defined for unary functions, so the chain must describe a sequence of nested calls
        for (size_t i = 0; i < originals.size(); ++i)
        {
            Monotonicity m = originals[i]->getMonotonicityForRange(*type_, left_, right_);
            if (!m.is_monotonic)
                return m;
            result.is_positive ^= !m.is_positive;
            result.is_always_monotonic &= m.is_always_monotonic;
            if (i + 1 < originals.size())
            {
                if (left_ != Field())
                    applyFunction(*originals[i], left_);
                if (right_ != Field())
                    applyFunction(*originals[i], right_);
                if (!m.is_positive)
                    std::swap(left_, right_);
                type_ = originals[i]->getReturnType().get();
            }
        }
        return result;
    }
};

static bool isCompilable(llvm::IRBuilderBase & builder, const IFunctionBase& function)
{
    if (!toNativeType(builder, function.getReturnType()))
        return false;
    for (const auto & type : function.getArgumentTypes())
        if (!toNativeType(builder, type))
            return false;
    return function.isCompilable();
}

void compileFunctions(ExpressionActions::Actions & actions, const Names & output_columns, const Block & sample_block)
{
    auto context = std::make_shared<LLVMContext>();
    /// an empty optional is a poisoned value prohibiting the column's producer from being removed
    /// (which it could be, if it was inlined into every dependent function).
    std::unordered_map<std::string, std::unordered_set<std::optional<size_t>>> current_dependents;
    for (const auto & name : output_columns)
        current_dependents[name].emplace();
    /// a snapshot of each compilable function's dependents at the time of its execution.
    std::vector<std::unordered_set<std::optional<size_t>>> dependents(actions.size());
    for (size_t i = actions.size(); i--;)
    {
        switch (actions[i].type)
        {
            case ExpressionAction::REMOVE_COLUMN:
                current_dependents.erase(actions[i].source_name);
                /// poison every other column used after this point so that inlining chains do not cross it.
                for (auto & dep : current_dependents)
                    dep.second.emplace();
                break;

            case ExpressionAction::PROJECT:
                current_dependents.clear();
                for (const auto & proj : actions[i].projection)
                    current_dependents[proj.first].emplace();
                break;

            case ExpressionAction::ADD_COLUMN:
            case ExpressionAction::COPY_COLUMN:
            case ExpressionAction::ARRAY_JOIN:
            case ExpressionAction::JOIN:
            {
                Names columns = actions[i].getNeededColumns();
                for (const auto & column : columns)
                    current_dependents[column].emplace();
                break;
            }

            case ExpressionAction::APPLY_FUNCTION:
            {
                dependents[i] = current_dependents[actions[i].result_name];
                const bool compilable = isCompilable(context->builder, *actions[i].function);
                for (const auto & name : actions[i].argument_names)
                {
                    if (compilable)
                        current_dependents[name].emplace(i);
                    else
                        current_dependents[name].emplace();
                }
                break;
            }
        }
    }

    std::vector<ExpressionActions::Actions> fused(actions.size());
    for (size_t i = 0; i < actions.size(); ++i)
    {
        if (actions[i].type != ExpressionAction::APPLY_FUNCTION || !isCompilable(context->builder, *actions[i].function))
            continue;
        fused[i].push_back(actions[i]);
        if (dependents[i].find({}) != dependents[i].end())
        {
            /// the result of compiling one function in isolation is pretty much the same as its `execute` method.
            if (fused[i].size() == 1)
                continue;
            auto fn = std::make_shared<LLVMFunction>(std::move(fused[i]), context, sample_block);
            actions[i].function = fn;
            actions[i].argument_names = fn->getArgumentNames();
            continue;
        }
        /// TODO: determine whether it's profitable to inline the function if there's more than one dependent.
        for (const auto & dep : dependents[i])
            fused[*dep].insert(fused[*dep].end(), fused[i].begin(), fused[i].end());
    }
    context->finalize();
}

}


namespace
{
    struct LLVMTargetInitializer
    {
        LLVMTargetInitializer()
        {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();
        }
    } llvmInitializer;
}

#endif
