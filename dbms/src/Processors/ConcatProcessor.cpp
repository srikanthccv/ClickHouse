#include <Processors/ConcatProcessor.h>


namespace DB
{

ConcatProcessor::Status ConcatProcessor::prepare()
{
    auto & output = outputs[0];

    /// Check can output.

    if (output.isFinished())
    {
        for (; current_input != inputs.end(); ++current_input)
            current_input->close();

        return Status::Finished;
    }

    if (!output.isNeeded())
    {
        if (current_input != inputs.end())
            current_input->setNotNeeded();

        return Status::PortFull;
    }

    if (output.hasData())
        return Status::PortFull;

    /// Check can input.

    if (current_input == inputs.end())
        return Status::Finished;

    if (current_input->isFinished())
    {
        ++current_input;
        if (current_input == inputs.end())
        {
            output.finish();
            return Status::Finished;
        }
    }

    auto & input = *current_input;

    input.setNeeded();

    if (!input.hasData())
        return Status::NeedData;

    /// Move data.
    output.push(input.pull());

    /// Now, we pushed to output, and it must be full.
    return Status::PortFull;
}

}


