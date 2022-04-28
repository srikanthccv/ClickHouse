#pragma once

#include <memory>

namespace DB
{

template <typename T>
struct TrivialWeightFunction
{
    size_t operator()(const T &) const
    {
        return 1;
    }
};

template <typename TKey, typename TMapped, typename HashFunction = std::hash<TKey>, typename WeightFunction = TrivialWeightFunction<TMapped>>
class ICachePolicy
{
public:    
    using Key = TKey;
    using Mapped = TMapped;
    using MappedPtr = std::shared_ptr<Mapped>;
    using OnWeightLossFunction = std::function<void(size_t)>;

    virtual size_t weight() const = 0;
    virtual size_t count() const = 0;
    virtual size_t maxSize() const = 0;

    virtual void reset() = 0;
    virtual void remove(const Key & key) = 0;
    virtual MappedPtr get(const Key & key) = 0;
    virtual void set(const Key & key, const MappedPtr & mapped) = 0;

    virtual ~ICachePolicy() = default;

protected:
    OnWeightLossFunction on_weight_loss_function = [](size_t) {};
};

}
