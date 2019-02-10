#pragma once

#include <type_traits>

#include <IO/WriteHelpers.h>
#include <IO/ReadHelpers.h>

#include <DataTypes/DataTypesNumber.h>
#include <DataTypes/DataTypesDecimal.h>
#include <Columns/ColumnVector.h>

#include <AggregateFunctions/IAggregateFunction.h>

#include <cmath>
#include <exception>

#include <Columns/ColumnsCommon.h>
#include <Columns/ColumnsNumber.h>
#include <Functions/FunctionHelpers.h>
#include <Common/FieldVisitors.h>


namespace DB
{

namespace ErrorCodes
{
    extern const int NUMBER_OF_ARGUMENTS_DOESNT_MATCH;
    extern const int BAD_ARGUMENTS;
}

class IGradientComputer
{
public:
    IGradientComputer(UInt32 sz)
    : batch_gradient(sz, 0)
    {}
    virtual ~IGradientComputer() = default;

    virtual void compute(const std::vector<Float64> & weights, Float64 bias, Float64 learning_rate,
            Float64 target, const IColumn ** columns, size_t row_num) = 0;

    void reset()
    {
        batch_gradient.assign(batch_gradient.size(), 0);
    }

    void write(WriteBuffer & buf) const
    {
        writeBinary(batch_gradient, buf);
    }

    void read(ReadBuffer & buf)
    {
        readBinary(batch_gradient, buf);
    }

    const std::vector<Float64> & get() const
    {
        return batch_gradient;
    }
    virtual Float64 predict(const std::vector<Float64> & predict_feature, const std::vector<Float64> & weights, Float64 bias) const = 0;
    virtual void predict_for_all(ColumnVector<Float64>::Container & container, Block & block, const ColumnNumbers & arguments, const std::vector<Float64> & weights, Float64 bias) const = 0;

protected:
    std::vector<Float64> batch_gradient;  // gradient for bias lies in batch_gradient[batch_gradient.size() - 1]
};

class LinearRegression : public IGradientComputer
{
public:
    LinearRegression(UInt32 sz)
    : IGradientComputer(sz)
    {}

    void compute(const std::vector<Float64> & weights, Float64 bias, Float64 learning_rate,
            Float64 target, const IColumn ** columns, size_t row_num) override
    {
        Float64 derivative = (target - bias);
        for (size_t i = 0; i < weights.size(); ++i)
        {
            derivative -= weights[i] * static_cast<const ColumnVector<Float64> &>(*columns[i + 1]).getData()[row_num];
        }
        derivative *= (2 * learning_rate);

        batch_gradient[weights.size()] += derivative;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            batch_gradient[i] += derivative * static_cast<const ColumnVector<Float64> &>(*columns[i + 1]).getData()[row_num];
        }
    }
    Float64 predict(const std::vector<Float64> & predict_feature, const std::vector<Float64> & weights, Float64 bias) const override
    {
        /// не обновляем веса при предикте, т.к. это может замедлить предсказание
        /// однако можно например обновлять их при каждом мердже не зависимо от того, сколько элементнов в батче
//        if (cur_batch)
//        {
//            update_weights();
//        }

        Float64 res{0.0};
        for (size_t i = 0; i < predict_feature.size(); ++i)
        {
            res += predict_feature[i] * weights[i];
        }
        res += bias;

        return res;
    }
    void predict_for_all(ColumnVector<Float64>::Container & container, Block & block, const ColumnNumbers & arguments, const std::vector<Float64> & weights, Float64 bias) const override
    {
        size_t rows_num = block.rows();
        std::cout << "\n\nROWS NUM: " << rows_num << "\n\n";
        std::vector<Float64> results(rows_num, bias);


        for (size_t i = 1; i < arguments.size(); ++i)
        {
            ColumnPtr cur_col = block.getByPosition(arguments[i]).column;
            for (size_t row_num = 0; row_num != rows_num; ++row_num)
            {
                const auto &element = (*cur_col)[row_num];
                if (element.getType() != Field::Types::Float64)
                    throw Exception("Prediction arguments must be values of type Float",
                                    ErrorCodes::BAD_ARGUMENTS);

                results[row_num] += weights[row_num] * element.get<Float64>();
                //            predict_features[i - 1] = element.get<Float64>();
            }
        }

        for (size_t row_num = 0; row_num != rows_num; ++row_num)
        {
            container.emplace_back(results[row_num]);
        }
//        column.getData().push_back(this->data(place).predict(predict_features));
//        column.getData().push_back(this->data(place).predict_for_all());
//        this->data(place).predict_for_all(column.getData(), block, arguments);
    }
};
class LogisticRegression : public IGradientComputer
{
public:
    LogisticRegression(UInt32 sz)
    : IGradientComputer(sz)
    {}

    void compute(const std::vector<Float64> & weights, Float64 bias, Float64 learning_rate,
            Float64 target, const IColumn ** columns, size_t row_num) override
    {
        Float64 derivative = bias;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            derivative += weights[i] * static_cast<const ColumnVector<Float64> &>(*columns[i + 1]).getData()[row_num];;
        }
        derivative *= target;
        derivative = learning_rate * exp(derivative);

        batch_gradient[weights.size()] += target / (derivative + 1);;
        for (size_t i = 0; i < weights.size(); ++i)
        {
            batch_gradient[i] += target / (derivative + 1) * static_cast<const ColumnVector<Float64> &>(*columns[i + 1]).getData()[row_num];
        }
    }
    Float64 predict(const std::vector<Float64> & predict_feature, const std::vector<Float64> & weights, Float64 bias) const override
    {
        /// не обновляем веса при предикте, т.к. это может замедлить предсказание
        /// однако можно например обновлять их при каждом мердже не зависимо от того, сколько элементнов в батче
//        if (cur_batch)
//        {
//            update_weights();
//        }

        Float64 res{0.0};
        for (size_t i = 0; i < predict_feature.size(); ++i)
        {
            res += predict_feature[i] * weights[i];
        }
        res += bias;
        res = 1 / (1 + exp(-res));
        return res;
    }
    void predict_for_all(ColumnVector<Float64>::Container & container, Block & block, const ColumnNumbers & arguments, const std::vector<Float64> & weights, Float64 bias) const override
    {
        std::ignore = container;
        std::ignore = block;
        std::ignore = arguments;
        std::ignore = weights;
        std::ignore = bias;
    }
};

class IWeightsUpdater
{
public:
    virtual ~IWeightsUpdater() = default;

    virtual void update(UInt32 cur_batch, std::vector<Float64> & weights, Float64 & bias, const std::vector<Float64> & gradient) = 0;
    virtual void merge(const std::shared_ptr<IWeightsUpdater>, Float64, Float64) {}
};

class StochasticGradientDescent : public IWeightsUpdater
{
public:
    void update(UInt32 cur_batch, std::vector<Float64> & weights, Float64 & bias, const std::vector<Float64> & batch_gradient) override {
        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] += batch_gradient[i] / cur_batch;
        }
        bias += batch_gradient[weights.size()] / cur_batch;
    }
};
class Momentum : public IWeightsUpdater
{
public:
    Momentum() {}
    Momentum (Float64 alpha) : alpha_(alpha) {}
    void update(UInt32 cur_batch, std::vector<Float64> & weights, Float64 & bias, const std::vector<Float64> & batch_gradient) override {
        if (hk_.size() == 0)
        {
            hk_.resize(batch_gradient.size(), Float64{0.0});
        }
        for (size_t i = 0; i < batch_gradient.size(); ++i)
        {
            hk_[i] = hk_[i] * alpha_ + batch_gradient[i];
        }
        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] += hk_[i] / cur_batch;
        }
        bias += hk_[weights.size()] / cur_batch;
    }
    /// virtual?
    virtual void merge(const std::shared_ptr<IWeightsUpdater> rhs, Float64 frac, Float64 rhs_frac) override {
        auto momentum_rhs = std::dynamic_pointer_cast<Momentum>(rhs);
        for (size_t i = 0; i < hk_.size(); ++i)
        {
            hk_[i] = hk_[i] * frac + momentum_rhs->hk_[i] * rhs_frac;
        }
    }

    Float64 alpha_{0.1};
    std::vector<Float64> hk_;
};

class LinearModelData
{
public:
    LinearModelData()
    {}

    LinearModelData(Float64 learning_rate,
            UInt32 param_num,
            UInt32 batch_size,
            std::shared_ptr<IGradientComputer> gc,
            std::shared_ptr<IWeightsUpdater> wu)
    : learning_rate(learning_rate),
    batch_size(batch_size),
    gradient_computer(std::move(gc)),
    weights_updater(std::move(wu))
    {
        weights.resize(param_num, Float64{0.0});
        cur_batch = 0;
    }


    void add(Float64 target, const IColumn ** columns, size_t row_num)
    {
        gradient_computer->compute(weights, bias, learning_rate, target, columns, row_num);
        ++cur_batch;
        if (cur_batch == batch_size)
        {
            update_state();
        }
    }

    void merge(const LinearModelData & rhs)
    {
        if (iter_num == 0 && rhs.iter_num == 0)
            return;

        update_state();
        /// нельзя обновить из-за константости
//        rhs.update_weights();

        Float64 frac = static_cast<Float64>(iter_num) / (iter_num + rhs.iter_num);
        Float64 rhs_frac = static_cast<Float64>(rhs.iter_num) / (iter_num + rhs.iter_num);

        for (size_t i = 0; i < weights.size(); ++i)
        {
            weights[i] = weights[i] * frac + rhs.weights[i] * rhs_frac;
        }

        bias = bias * frac + rhs.bias * rhs_frac;
        iter_num += rhs.iter_num;
        weights_updater->merge(rhs.weights_updater, frac, rhs_frac);
    }

    void write(WriteBuffer & buf) const
    {
        writeBinary(bias, buf);
        writeBinary(weights, buf);
        writeBinary(iter_num, buf);
        writeBinary(cur_batch, buf);
        gradient_computer->write(buf);
    }

    void read(ReadBuffer & buf)
    {
        readBinary(bias, buf);
        readBinary(weights, buf);
        readBinary(iter_num, buf);
        readBinary(cur_batch, buf);
        gradient_computer->read(buf);
    }

    Float64 predict(const std::vector<Float64> & predict_feature) const
    {
        /// не обновляем веса при предикте, т.к. это может замедлить предсказание
        /// однако можно например обновлять их при каждом мердже не зависимо от того, сколько элементнов в батче
//        if (cur_batch)
//        {
//            update_weights();
//        }

        return gradient_computer->predict(predict_feature, weights, bias);
    }
    void predict_for_all(ColumnVector<Float64>::Container & container, Block & block, const ColumnNumbers & arguments) const
    {
        gradient_computer->predict_for_all(container, block, arguments, weights, bias);
    }

private:
    std::vector<Float64> weights;
    Float64 learning_rate;
    UInt32 batch_size;
    Float64 bias{0.0};
    UInt32 iter_num = 0;
    UInt32 cur_batch;
    std::shared_ptr<IGradientComputer> gradient_computer;
    std::shared_ptr<IWeightsUpdater> weights_updater;

    void update_state()
    {
        if (cur_batch == 0)
            return;

        weights_updater->update(cur_batch, weights, bias, gradient_computer->get());
        cur_batch = 0;
        ++iter_num;
        gradient_computer->reset();
    }
};

template <
        /// Implemented Machine Learning method
        typename Data,
        /// Name of the method
        typename Name
>
class AggregateFunctionMLMethod final : public IAggregateFunctionDataHelper<Data, AggregateFunctionMLMethod<Data, Name>>
{
public:
    String getName() const override { return Name::name; }

    explicit AggregateFunctionMLMethod(UInt32 param_num,
                                        std::shared_ptr<IGradientComputer> gradient_computer,
                                        std::shared_ptr<IWeightsUpdater> weights_updater,
                                        Float64 learning_rate,
                                        UInt32 batch_size)
            : param_num(param_num),
            learning_rate(learning_rate),
            batch_size(batch_size),
            gc(std::move(gradient_computer)),
            wu(std::move(weights_updater))
    {}

    DataTypePtr getReturnType() const override
    {
        return std::make_shared<DataTypeNumber<Float64>>();
    }

    void create(AggregateDataPtr place) const override
    {
        new (place) Data(learning_rate, param_num, batch_size, gc, wu);
    }

    void add(AggregateDataPtr place, const IColumn ** columns, size_t row_num, Arena *) const override
    {
        const auto & target = static_cast<const ColumnVector<Float64> &>(*columns[0]);

        this->data(place).add(target.getData()[row_num], columns, row_num);
    }

    /// хочется не константный rhs
    void merge(AggregateDataPtr place, ConstAggregateDataPtr rhs, Arena *) const override
    {
        this->data(place).merge(this->data(rhs));
    }

    void serialize(ConstAggregateDataPtr place, WriteBuffer & buf) const override
    {
        this->data(place).write(buf);
    }

    void deserialize(AggregateDataPtr place, ReadBuffer & buf, Arena *) const override
    {
        this->data(place).read(buf);
    }

    void predictResultInto(ConstAggregateDataPtr place, IColumn & to, Block & block, size_t row_num, const ColumnNumbers & arguments) const
    {
        std::ignore = row_num;
        std::cout << "\n\n IM CALLED \n\n";

        if (arguments.size() != param_num + 1)
            throw Exception("Predict got incorrect number of arguments. Got: " + std::to_string(arguments.size()) + ". Required: " + std::to_string(param_num + 1),
                            ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH);

        auto &column = dynamic_cast<ColumnVector<Float64> &>(to);

//        std::vector<Float64> predict_features(arguments.size() - 1);
//        for (size_t i = 1; i < arguments.size(); ++i)
//        {
//            const auto& element = (*block.getByPosition(arguments[i]).column)[row_num];
//            if (element.getType() != Field::Types::Float64)
//                throw Exception("Prediction arguments must be values of type Float",
//                        ErrorCodes::BAD_ARGUMENTS);
//
////            predict_features[i - 1] = element.get<Float64>();
//        }
//        column.getData().push_back(this->data(place).predict(predict_features));
//        column.getData().push_back(this->data(place).predict_for_all());
        this->data(place).predict_for_all(column.getData(), block, arguments);
    }

    void insertResultInto(ConstAggregateDataPtr place, IColumn & to) const override
    {
        std::ignore = place;
        std::ignore = to;
        return;
    }

    const char * getHeaderFilePath() const override { return __FILE__; }

private:
    UInt32 param_num;
    Float64 learning_rate;
    UInt32 batch_size;
    std::shared_ptr<IGradientComputer> gc;
    std::shared_ptr<IWeightsUpdater> wu;
};

struct NameLinearRegression { static constexpr auto name = "LinearRegression"; };
struct NameLogisticRegression { static constexpr auto name = "LogisticRegression"; };
}
