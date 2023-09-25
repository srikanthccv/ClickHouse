#pragma once

#include <Core/NamesAndAliases.h>
#include <Core/NamesAndTypes.h>
#include <Interpreters/SystemLog.h>

namespace DB
{

struct S3QueueLogElement
{
    time_t event_time{};
    std::string table_uuid;
    std::string file_name;
    size_t rows_processed = 0;

    enum class S3QueueStatus
    {
        Processed,
        Failed,
    };
    S3QueueStatus status;

    static std::string name() { return "S3QueueLog"; }

    static NamesAndTypesList getNamesAndTypes();
    static NamesAndAliases getNamesAndAliases() { return {}; }

    void appendToBlock(MutableColumns & columns) const;
    static const char * getCustomColumnList() { return nullptr; }
};

class S3QueueLog : public SystemLog<S3QueueLogElement>
{
    using SystemLog<S3QueueLogElement>::SystemLog;
};

}
