#include <Common/COWPtr.h>
#include <iostream>


class IColumn : public COWPtr<IColumn>
{
private:
    friend class COWPtr<IColumn>;
    virtual MutablePtr clone() const = 0;

public:
    IColumn() = default;
    IColumn(const IColumn &) = default;
    virtual ~IColumn() = default;

    virtual int get() const = 0;
    virtual void set(int value) = 0;

    virtual MutablePtr mutate() const && { return std::move(*this).template COWPtr<IColumn>::mutate(); }
};

using ColumnPtr = IColumn::Ptr;
using MutableColumnPtr = IColumn::MutablePtr;

class ConcreteColumn : public COWPtrHelper<IColumn, ConcreteColumn>
{
private:
    friend class COWPtrHelper<IColumn, ConcreteColumn>;

    int data;
    ConcreteColumn(int data) : data(data) {}
    ConcreteColumn(const ConcreteColumn &) = default;

public:
    int get() const override { return data; }
    void set(int value) override { data = value; }
};

class ColumnComposition : public COWPtrHelper<IColumn, ColumnComposition>
{
private:
    using Base = COWPtrHelper<IColumn, ColumnComposition>;
    friend class COWPtrHelper<IColumn, ColumnComposition>;

    ConcreteColumn::WrappedPtr wrapped;

    ColumnComposition(int data) : wrapped(ConcreteColumn::create(data)) {}
    ColumnComposition(const ColumnComposition &) = default;

public:
    int get() const override { return wrapped->get(); }
    void set(int value) override { wrapped->set(value); }

    IColumn::MutablePtr mutate() const && override
    {
        std::cerr << "Mutating\n";
        auto res = std::move(*this).Base::mutate();
        static_cast<ColumnComposition *>(res.get())->wrapped = std::move(*wrapped).mutate();
        return res;
    }
};


int main(int, char **)
{
    ColumnPtr x = ColumnComposition::create(1);
    ColumnPtr y = x;

    std::cerr << "values:    " << x->get() << ", " << y->get() << "\n";
    std::cerr << "refcounts: " << x->use_count() << ", " << y->use_count() << "\n";
    std::cerr << "addresses: " << x.get() << ", " << y.get() << "\n";

    {
        MutableColumnPtr mut = std::move(*y).mutate();
        mut->set(2);

        std::cerr << "refcounts: " << x->use_count() << ", " << y->use_count() << ", " << mut->use_count() << "\n";
        std::cerr << "addresses: " << x.get() << ", " << y.get() << ", " << mut.get() << "\n";
        y = std::move(mut);
    }

    std::cerr << "values:    " << x->get() << ", " << y->get() << "\n";
    std::cerr << "refcounts: " << x->use_count() << ", " << y->use_count() << "\n";
    std::cerr << "addresses: " << x.get() << ", " << y.get() << "\n";

    x = ColumnComposition::create(0);

    std::cerr << "values:    " << x->get() << ", " << y->get() << "\n";
    std::cerr << "refcounts: " << x->use_count() << ", " << y->use_count() << "\n";
    std::cerr << "addresses: " << x.get() << ", " << y.get() << "\n";

    {
        MutableColumnPtr mut = std::move(*y).mutate();
        mut->set(3);

        std::cerr << "refcounts: " << x->use_count() << ", " << y->use_count() << ", " << mut->use_count() << "\n";
        std::cerr << "addresses: " << x.get() << ", " << y.get() << ", " << mut.get() << "\n";
        y = std::move(mut);
    }

    std::cerr << "values:    " << x->get() << ", " << y->get() << "\n";
    std::cerr << "refcounts: " << x->use_count() << ", " << y->use_count() << "\n";

    return 0;
}

