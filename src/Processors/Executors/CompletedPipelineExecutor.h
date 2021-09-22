#pragma once
#include <functional>

namespace DB
{

class QueryPipeline;

class CompletedPipelineExecutor
{
public:
    explicit CompletedPipelineExecutor(QueryPipeline & pipeline_);
    ~CompletedPipelineExecutor();
    void setCancelCallback(std::function<bool()> is_cancelled, size_t interactive_timeout_ms_);
    void execute();
    struct Data;

private:
    QueryPipeline & pipeline;
    std::function<bool()> is_cancelled_callback;
    size_t interactive_timeout_ms = 0;
    std::unique_ptr<Data> data;
};

}
