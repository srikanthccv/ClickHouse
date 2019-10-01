#pragma once
#include <Processors/ISimpleTransform.h>

namespace DB
{

class ReverseTransform : public ISimpleTransform
{
public:
    explicit ReverseTransform(const Block & header) : ISimpleTransform(header, header, false) {}
    String getName() const override { return "ReverseTransform"; }

protected:
    void transform(Chunk & chunk) override;
};

}
