#pragma once

#include <string>
#include <cassandra.h>
#include <Core/Block.h>
#include <DataStreams/IBlockInputStream.h>
#include "ExternalResultDescription.h"


namespace DB
{
/// Allows processing results of a Cassandra query as a sequence of Blocks, simplifies chaining
    class CassandraBlockInputStream final : public IBlockInputStream
    {
    public:
        CassandraBlockInputStream(
                CassSession * session,
                const std::string & query_str,
                const Block & sample_block,
                const size_t max_block_size);
        ~CassandraBlockInputStream() override;

        String getName() const override { return "Cassandra"; }

        Block getHeader() const override { return description.sample_block.cloneEmpty(); }

    private:
        Block readImpl() override;

        CassSession * session,
        const std::string & query_str;
        const size_t max_block_size;
        ExternalResultDescription description;
        const CassResult * result;
        CassIterator * iterator = nullptr;
    };

}
