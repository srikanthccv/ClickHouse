#pragma once
#include <Common/config.h>
#include <memory>
#include <type_traits>
#include <vector>
#include <string>

#if USE_HDFS
#include <hdfs/hdfs.h>
#include <Storages/IStorage.h>

namespace DB
{
namespace detail
{
/* struct HDFSBuilderDeleter */
/* { */
/*     void operator()(hdfsBuilder * builder_ptr) */
/*     { */
/*         hdfsFreeBuilder(builder_ptr); */
/*     } */
/* }; */
struct HDFSFsDeleter
{
    void operator()(hdfsFS fs_ptr)
    {
        hdfsDisconnect(fs_ptr);
    }
};

}

struct HDFSFileInfo
{
    hdfsFileInfo * file_info;
    int length;

    HDFSFileInfo()
        : file_info(nullptr)
        , length(0)
    {
    }
    HDFSFileInfo(const HDFSFileInfo & other) = delete;
    HDFSFileInfo(HDFSFileInfo && other) = default;
    HDFSFileInfo & operator=(const HDFSFileInfo & other) = delete;
    HDFSFileInfo & operator=(HDFSFileInfo && other) = default;

    ~HDFSFileInfo()
    {
        hdfsFreeFileInfo(file_info, length);
    }
};


class HDFSBuilderWrapper
{

    hdfsBuilder * hdfs_builder;
    String hadoop_kerberos_keytab;
    String hadoop_kerberos_principal;

    /*mutable*/ std::vector<std::pair<String, String>> config_stor;

    std::pair<String, String>& keep(const String & k, const String & v)
    {
        return config_stor.emplace_back(std::make_pair(k, v));
    }

    void
    loadFromConfig(const Poco::Util::AbstractConfiguration & config, const String & path);

public:

    static const String CONFIG_PREFIX;

    bool
    needKinit{false};

    String
    getKinitCmd();

    hdfsBuilder *
    get()
    {
        return hdfs_builder;
    }

    HDFSBuilderWrapper()
    {
        hdfs_builder = hdfsNewBuilder();
    }

    ~HDFSBuilderWrapper()
    {
        hdfsFreeBuilder(hdfs_builder);
    };

    HDFSBuilderWrapper(const HDFSBuilderWrapper &) = delete;
    HDFSBuilderWrapper(HDFSBuilderWrapper &&) = default;

    friend HDFSBuilderWrapper createHDFSBuilder(const String & uri_str, const Context & context);
};




/* using HDFSBuilderPtr = std::unique_ptr<hdfsBuilder, detail::HDFSBuilderDeleter>; */
using HDFSFSPtr = std::unique_ptr<std::remove_pointer_t<hdfsFS>, detail::HDFSFsDeleter>;

// set read/connect timeout, default value in libhdfs3 is about 1 hour, and too large
/// TODO Allow to tune from query Settings.
HDFSBuilderWrapper createHDFSBuilder(const String & uri_str, const Context & context);
HDFSFSPtr createHDFSFS(hdfsBuilder * builder);
}
#endif
