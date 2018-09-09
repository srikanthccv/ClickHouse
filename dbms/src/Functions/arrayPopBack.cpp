#include <Functions/arrayPop.h>
#include <Functions/FunctionFactory.h>


namespace DB
{

class FunctionArrayPopBack : public FunctionArrayPop
{
public:
    static constexpr auto name = "arrayPopBack";
    static FunctionPtr create(const Context & context) { return std::make_shared<FunctionArrayPopBack>(context); }
    FunctionArrayPopBack() : FunctionArrayPop(false, name) {}
};

void registerFunctionArrayPopBack(FunctionFactory & factory)
{
    factory.registerFunction<FunctionArrayPopBack>();
}

}
