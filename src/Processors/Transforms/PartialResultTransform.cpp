#include <Processors/Transforms/PartialResultTransform.h>

namespace DB
{

PartialResultTransform::PartialResultTransform(const Block & header, UInt64 partial_result_limit_, UInt64 partial_result_duration_ms_)
    : IProcessor({header}, {header})
    , input(inputs.front())
    , output(outputs.front())
    , partial_result_limit(partial_result_limit_)
    , partial_result_duration_ms(partial_result_duration_ms_)
    , watch(CLOCK_MONOTONIC)
    {}

IProcessor::Status PartialResultTransform::prepare()
{
    if (output.isFinished())
    {
        input.close();
        return Status::Finished;
    }

    if (finished_getting_snapshots)
    {
        output.finish();
        return Status::Finished;
    }

    if (input.hasData())
        partial_result = {input.pull(), SnaphotStatus::Ready};

    /// Send partial result from real processor snapshot or from previous partial result processor if possible
    if (partial_result.snapshot_status == SnaphotStatus::Ready && output.canPush())
    {
        transformPartialResult(partial_result.chunk);
        partial_result.snapshot_status = SnaphotStatus::NotReady;
        if (partial_result.chunk.getNumRows() > 0)
        {
            output.push(std::move(partial_result.chunk));
            return Status::PortFull;
        }
    }

    /// If input data from previous partial result processor is finished then
    /// PartialResultTransform ready to create snapshots and send them as a partial result
    if (input.isFinished())
    {
        return Status::Ready;
    }

    input.setNeeded();
    return Status::NeedData;
}

void PartialResultTransform::work()
{
    if (partial_result_duration_ms < watch.elapsedMilliseconds())
    {
        partial_result = getRealProcessorSnapshot();
        if (partial_result.snapshot_status == SnaphotStatus::Stopped)
            finished_getting_snapshots = true;

        watch.restart();
    }
}

}
