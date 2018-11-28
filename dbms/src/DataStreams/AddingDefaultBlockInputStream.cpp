#include <DataStreams/AddingDefaultBlockInputStream.h>
#include <Interpreters/addMissingDefaults.h>


namespace DB
{

AddingDefaultBlockInputStream::AddingDefaultBlockInputStream(
    const BlockInputStreamPtr & input_,
    const Block & header_,
    const ColumnDefaults & column_defaults_,
    const Context & context_)
    : input(input_), header(header_),
      column_defaults(column_defaults_), context(context_)
{
    children.emplace_back(input);
}

Block AddingDefaultBlockInputStream::readImpl()
{
    Block src = children.back()->read();
    if (!src)
        return src;

    return addMissingDefaults(src, header.getNamesAndTypesList(), column_defaults, context);
}

}
