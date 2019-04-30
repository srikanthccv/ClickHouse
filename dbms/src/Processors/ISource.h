#pragma once

#include <Processors/IProcessor.h>


namespace DB
{

class ISource : public IProcessor
{
protected:
    OutputPort & output;
    bool has_input = false;
    bool finished = false;
    [[maybe_unused]] bool got_exception = false;
    Port::Data current_chunk;

    virtual Chunk generate() = 0;

public:
    ISource(Block header);

    Status prepare() override;
    void work() override;

    OutputPort & getPort() { return output; }
    const OutputPort & getPort() const { return output; }
};

}
