#include <Processors/PingPongProcessor.h>

namespace DB
{

static InputPorts createPortsList(const Block & header, const Block & last_header, size_t num_ports)
{
    InputPorts res(num_ports, header);
    res.emplace_back(last_header);
    return res;
}

PingPongProcessor::PingPongProcessor(const Block & header, const Block & aux_header, size_t num_ports, Order order_)
    : IProcessor(createPortsList(header, aux_header, num_ports), OutputPorts(num_ports + 1, header))
    , aux_in_port(inputs.back())
    , aux_out_port(outputs.back())
    , order(order_)
{
    assert(order == First || order == Second);

    port_pairs.resize(num_ports);

    auto input_it = inputs.begin();
    auto output_it = outputs.begin();
    for (size_t i = 0; i < num_ports; ++i)
    {
        port_pairs[i].input_port = &*input_it;
        ++input_it;

        port_pairs[i].output_port = &*output_it;
        ++output_it;
    }
}

void PingPongProcessor::finishPair(PortsPair & pair)
{
    if (!pair.is_finished)
    {
        pair.output_port->finish();
        pair.input_port->close();

        pair.is_finished = true;
        ++num_finished_pairs;
    }
}

bool PingPongProcessor::processPair(PortsPair & pair)
{
    if (pair.output_port->isFinished())
    {
        finishPair(pair);
        return false;
    }

    if (pair.input_port->isFinished())
    {
        finishPair(pair);
        return false;
    }

    if (!pair.output_port->canPush())
    {
        pair.input_port->setNotNeeded();
        return false;
    }

    pair.input_port->setNeeded();
    if (pair.input_port->hasData())
    {
        Chunk chunk = pair.input_port->pull(true);
        ready_to_send = isReady(chunk) || ready_to_send;
        pair.output_port->push(std::move(chunk));
    }

    return true;
}

bool PingPongProcessor::isPairsFinished() const
{
    return num_finished_pairs == port_pairs.size();
}

IProcessor::Status PingPongProcessor::processRegularPorts()
{
    if (isPairsFinished())
        return Status::Finished;

    bool need_data = false;

    for (auto & pair : port_pairs)
        need_data = processPair(pair) || need_data;

    if (isPairsFinished())
        return Status::Finished;

    if (need_data)
        return Status::NeedData;

    return Status::PortFull;
}

bool PingPongProcessor::sendPing()
{
    if (aux_out_port.canPush())
    {
        Chunk chunk(aux_out_port.getHeader().cloneEmpty().getColumns(), 0);
        aux_out_port.push(std::move(chunk));
        is_send = true;
        aux_out_port.finish();
        return true;
    }
    return false;
}

bool PingPongProcessor::recievePing()
{
    if (aux_in_port.hasData())
    {
        aux_in_port.pull();
        is_recieved = true;
        aux_in_port.close();
        return true;
    }
    return false;
}

bool PingPongProcessor::canSend() const
{
    return !is_send && (ready_to_send || isPairsFinished());
}

IProcessor::Status PingPongProcessor::prepare()
{
    if (!set_needed_once && !is_recieved && !aux_in_port.isFinished())
    {
        set_needed_once = true;
        aux_in_port.setNeeded();
    }

    if (order == First || is_send)
    {
        if (!is_recieved)
        {
            bool recieved = recievePing();
            if (!recieved)
            {
                return Status::NeedData;
            }
        }
    }

    if (order == Second || is_recieved)
    {
        if (!is_send && canSend())
        {
            bool sent = sendPing();
            if (!sent)
                return Status::PortFull;
        }
    }

    auto status = processRegularPorts();
    if (status == Status::Finished)
    {
        if (order == First || is_send)
        {
            if (!is_recieved)
            {
                bool recieved = recievePing();
                if (!recieved)
                {
                    return Status::NeedData;
                }
            }
        }

        if (order == Second || is_recieved)
        {
            if (!is_send && canSend())
            {
                bool sent = sendPing();
                if (!sent)
                    return Status::PortFull;
            }
        }
    }
    return status;
}

std::pair<InputPort *, OutputPort *> PingPongProcessor::getAuxPorts()
{
    return std::make_pair(&aux_in_port, &aux_out_port);
}

}
