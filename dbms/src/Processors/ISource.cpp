#include <Processors/ISource.h>


namespace DB
{

ISource::ISource(Block header)
    : IProcessor({}, {std::move(header)}), output(outputs.front())
{
}

ISource::Status ISource::prepare()
{
    if (finished)
    {
        output.finish();
        return Status::Finished;
    }

    /// Check can output.
    if (output.isFinished())
        return Status::Finished;

    if (!output.canPush())
        return Status::PortFull;

    if (!has_input)
        return Status::Ready;

    output.pushData(std::move(current_chunk));
    has_input = false;

    if (got_exception)
    {
        finished = true;
        output.finish();
        return Status::Finished;
    }

    /// Now, we pushed to output, and it must be full.
    return Status::PortFull;
}

void ISource::work()
{
    try
    {
        current_chunk = generate();
        if (!std::get<Chunk>(current_chunk))
            finished = true;
        else
            has_input = true;
    }
    catch (...)
    {
        finished = true;
        throw;
    }
//    {
//        current_chunk = std::current_exception();
//        has_input = true;
//        got_exception = true;
//    }
}

}

