#include <Processors/Formats/IInputFormat.h>
#include <IO/ReadBuffer.h>

#include <iostream>


namespace DB
{

IInputFormat::IInputFormat(Block header, ReadBuffer & in_)
    : ISource(std::move(header)), in(in_)
{
    column_mapping = std::make_shared<ColumnMapping>();
}

void IInputFormat::resetParser()
{
    std::cerr << "resetParser... in: " << demangle(typeid(in).name()) << "\n";
    in.ignoreAll();
    // those are protected attributes from ISource (I didn't want to propagate resetParser up there)
    finished = false;
    got_exception = false;

    getPort().getInputPort().reopen();
}

void IInputFormat::setReadBuffer(ReadBuffer & in_)
{
    std::cerr << "setReadBuffer... old: " << demangle(typeid(in).name()) << ", new: " << demangle(typeid(in_).name()) << "\n";
    in = in_;
}

}
