#pragma once

#include <Processors/IProcessor.h>
#include <Processors/PipelineResourcesHolder.h>

namespace DB
{

class Chain
{
public:
    Chain() = default;
    Chain(Chain &&) = default;
    Chain(const Chain &) = delete;

    Chain & operator=(Chain &&) = default;
    Chain & operator=(const Chain &) = delete;

    explicit Chain(ProcessorPtr processor);
    explicit Chain(std::list<ProcessorPtr> processors);

    bool empty() const { return processors.empty(); }

    void addSource(ProcessorPtr processor);
    void addSink(ProcessorPtr processor);

    IProcessor & getSource();
    IProcessor & getSink();

    InputPort & getInputPort() const;
    OutputPort & getOutputPort() const;

    const Block & getInputHeader() const { return getInputPort().getHeader(); }
    const Block & getOutputHeader() const { return getOutputPort().getHeader(); }

    const std::list<ProcessorPtr> & getProcessors() const { return processors; }
    static std::list<ProcessorPtr> getProcessors(Chain chain) { return std::move(chain.processors); }

    void addTableLock(TableLockHolder lock) { holder.table_locks.emplace_back(std::move(lock)); }
    void attachResourcesFrom(Chain & other) { holder = std::move(other.holder); }
    PipelineResourcesHolder detachResources() { return std::move(holder); }

private:
    /// -> source -> transform -> ... -> transform -> sink ->
    ///  ^        ->           ->     ->           ->       ^
    ///  input port                               output port
    std::list<ProcessorPtr> processors;
    PipelineResourcesHolder holder;
};

}
