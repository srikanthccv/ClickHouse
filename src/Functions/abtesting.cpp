#if !defined(ARCADIA_BUILD) 

#include <math.h>
#include <sstream>

#include <DataTypes/DataTypeString.h>
#include <Columns/ColumnString.h>
#include <Columns/ColumnConst.h>
#include <Columns/ColumnsNumber.h>
#include <Functions/FunctionFactory.h>
#include <Functions/FunctionHelpers.h>
#include <Functions/abtesting.h>
#include <IO/WriteHelpers.h>
#include <IO/WriteBufferFromOStream.h>

#define STATS_ENABLE_STDVEC_WRAPPERS
#include <stats.hpp>

namespace DB
{

namespace ErrorCodes
{
    extern const int ILLEGAL_TYPE_OF_ARGUMENT;
    extern const int BAD_ARGUMENTS;
}

static const String BETA = "beta";
static const String GAMMA = "gamma";

template <bool higher_is_better>
Variants bayesian_ab_test(String distribution, std::vector<double> xs, std::vector<double> ys)
{
    const size_t r = 1000, c = 100;

    Variants variants(xs.size());
    std::vector<std::vector<double>> samples_matrix;

    if (distribution == BETA)
    {
        double alpha, beta;

        for (size_t i = 0; i < xs.size(); ++i)
            if (xs[i] < ys[i])
                throw Exception("Conversions cannot be larger than trials", ErrorCodes::BAD_ARGUMENTS);

        for (size_t i = 0; i < xs.size(); ++i)
        {
            alpha = 1.0 + ys[i];
            beta = 1.0 + xs[i] - ys[i];

            samples_matrix.push_back(stats::rbeta<std::vector<double>>(r, c, alpha, beta));
        }
    }
    else if (distribution == GAMMA)
    {
        double shape, scale;

        for (size_t i = 0; i < xs.size(); ++i)
        {
            shape = 1.0 + xs[i];
            scale = 250.0 / (1 + 250.0 * ys[i]);
            std::vector<double> samples = stats::rgamma<std::vector<double>>(r, c, shape, scale);
            for (size_t j = 0; j < samples.size(); ++j)
                samples[j] = 1 / samples[j];
            samples_matrix.push_back(samples);
        }
    }

    std::vector<double> means;
    for (size_t i = 0; i < xs.size(); ++i)
    {
        auto mean = accumulate(samples_matrix[i].begin(), samples_matrix[i].end(), 0.0) / samples_matrix[i].size();
        means.push_back(mean);
    }

    // Beats control
    for (size_t i = 1; i < xs.size(); ++i)
    {
        for (size_t n = 0; n < r * c; ++n)
        {
            if (higher_is_better)
            {
                if (samples_matrix[i][n] > samples_matrix[0][n])
                    ++variants[i].beats_control;
            }
            else
            {
                if (samples_matrix[i][n] < samples_matrix[0][n])
                    ++variants[i].beats_control;
            }
        }
    }

    for (size_t i = 1; i < xs.size(); ++i)
        variants[i].beats_control = static_cast<double>(variants[i].beats_control) / r / c;

    // To be best
    std::vector<size_t> count_m(xs.size(), 0);
    std::vector<double> row(xs.size(), 0);

    for (size_t n = 0; n < r * c; ++n)
    {
        for (size_t i = 0; i < xs.size(); ++i)
            row[i] = samples_matrix[i][n];

        double m;
        if (higher_is_better)
            m = *std::max_element(row.begin(), row.end());
        else
            m = *std::min_element(row.begin(), row.end());

        for (size_t i = 0; i < xs.size(); ++i)
        {
            if (m == samples_matrix[i][n])
            {
                ++variants[i].best;
                break;
            }
        }
    }

    for (size_t i = 0; i < xs.size(); ++i)
        variants[i].best = static_cast<double>(variants[i].best) / r / c;

    return variants;
}

class FunctionBayesAB : public IFunction
{
public:
    static constexpr auto name = "bayesAB";

    static FunctionPtr create(const Context &)
    {
        return std::make_shared<FunctionBayesAB>();
    }

    String getName() const override
    {
        return name;
    }

    bool isDeterministic() const override { return false; }
    bool isDeterministicInScopeOfQuery() const override { return false; }

    size_t getNumberOfArguments() const override { return 5; }

    DataTypePtr getReturnTypeImpl(const DataTypes &) const override
    {
        return std::make_shared<DataTypeString>();
    }

    void executeImpl(Block & block, const ColumnNumbers & arguments, size_t result, size_t input_rows_count) override
    {
        if (input_rows_count == 0)
        {
            block.getByPosition(result).column = std::move(ColumnString::create());
            return;
        }

        std::vector<double> xs, ys;
        std::vector<std::string> variant_names;
        String dist;
        bool higher_is_better;

        if (const ColumnConst * col_dist = checkAndGetColumnConst<ColumnString>(block.getByPosition(arguments[0]).column.get()))
        {
            dist = col_dist->getDataAt(0).data;
            dist = Poco::toLower(dist);
            if (dist != BETA && dist != GAMMA)
                throw Exception("First argument for function " + getName() + " cannot be " + dist, ErrorCodes::BAD_ARGUMENTS);
        }
        else
            throw Exception("First argument for function " + getName() + " must be Constant string", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        if (const ColumnConst * col_higher_is_better = checkAndGetColumnConst<ColumnUInt8>(block.getByPosition(arguments[1]).column.get()))
            higher_is_better = col_higher_is_better->getBool(0);
        else
            throw Exception("Second argument for function " + getName() + " must be Constatnt boolean", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        if (const ColumnConst * col_const_arr = checkAndGetColumnConst<ColumnArray>(block.getByPosition(arguments[2]).column.get()))
        {
            if (!col_const_arr)
                throw Exception("Thrid argument for function " + getName() + " must be Array of constant strings", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

            Array src_arr = col_const_arr->getValue<Array>();
            for (size_t i = 0; i < src_arr.size(); ++i)
                variant_names.push_back(src_arr[i].get<const String &>());
        }

        if (const ColumnConst * col_const_arr = checkAndGetColumnConst<ColumnArray>(block.getByPosition(arguments[3]).column.get()))
        {
            if (!col_const_arr)
                throw Exception("Forth argument for function " + getName() + " must be Array of constant doubles", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

            Array src_arr = col_const_arr->getValue<Array>();

            for (size_t i = 0, size = src_arr.size(); i < size; ++i)
                xs.push_back(src_arr[i].get<const Float64 &>());
        }

        if (const ColumnConst * col_const_arr = checkAndGetColumnConst<ColumnArray>(block.getByPosition(arguments[4]).column.get()))
        {
            if (!col_const_arr)
                throw Exception("Fifth argument for function " + getName() + " must be Array of constant doubles", ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

            Array src_arr = col_const_arr->getValue<Array>();

            for (size_t i = 0, size = src_arr.size(); i < size; ++i)
                ys.push_back(src_arr[i].get<const Float64 &>());
        }

        if (variant_names.size() != xs.size() || xs.size() != ys.size())
            throw Exception(ErrorCodes::BAD_ARGUMENTS, "Sizes of arguments doen't match: variant_names: {}, xs: {}, ys: {}", variant_names.size(), xs.size(), ys.size());

        if (std::count_if(xs.begin(), xs.end(), [](double v) { return v < 0; }) > 0 ||
            std::count_if(ys.begin(), ys.end(), [](double v) { return v < 0; }) > 0)
            throw Exception("Negative values don't allowed", ErrorCodes::BAD_ARGUMENTS);

        Variants variants;

        if (higher_is_better)
            variants = bayesian_ab_test<true>(dist, xs, ys);
        else
            variants = bayesian_ab_test<false>(dist, xs, ys);

        FormatSettings settings;
        std::stringstream s;

        {
            WriteBufferFromOStream buf(s);

            writeCString("{\"data\":[", buf);
            for (size_t i = 0; i < variants.size(); ++i)
            {
                writeCString("{\"variant_name\":", buf);
                writeJSONString(variant_names[i], buf, settings);
                writeCString(",\"beats_control\":", buf);
                writeText(variants[i].beats_control, buf);
                writeCString(",\"to_be_best\":", buf);
                writeText(variants[i].best, buf);
                writeCString("}", buf);
                if (i != xs.size() -1) writeCString(",", buf);
            }
            writeCString("]}", buf);
        }

        auto dst = ColumnString::create();
        std::string result_str = s.str();
        dst->insertData(result_str.c_str(), result_str.length());
        block.getByPosition(result).column = std::move(dst);
    }
};

void registerFunctionBayesAB(FunctionFactory & factory)
{
    factory.registerFunction<FunctionBayesAB>();
}

}

#endif
