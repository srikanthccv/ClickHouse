#pragma once

#include <Processors/IProcessor.h>


namespace DB
{

class ISource : public IProcessor
{
private:
    ReadProgress read_progress;
    bool read_progress_was_set = false;

protected:
    OutputPort & output;
    bool has_input = false;
    bool finished = false;
    bool got_exception = false;
    Port::Data current_chunk;

    virtual Chunk generate();
    virtual std::optional<Chunk> tryGenerate();

    virtual void progress(size_t read_rows, size_t read_bytes);

public:
    explicit ISource(Block header);

    Status prepare() override;
    void work() override;

    OutputPort & getPort() { return output; }
    const OutputPort & getPort() const { return output; }

    /// Default implementation for all the sources.
    std::optional<ReadProgress> getReadProgress() final;
};

using SourcePtr = std::shared_ptr<ISource>;

}
