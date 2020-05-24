#include <gtest/gtest.h>
#include <Common/config.h>

#if USE_AWS_S3

#    include <IO/S3Common.h>

namespace
{
using namespace DB;

class S3UriTest : public testing::TestWithParam<std::string>
{
};

TEST(S3UriTest, validPatterns)
{
    {
        S3::URI uri(Poco::URI("https://jokserfn.s3.yandexcloud.net/data"));
        ASSERT_EQ("https://s3.yandexcloud.net", uri.endpoint);
        ASSERT_EQ("jokserfn", uri.bucket);
        ASSERT_EQ("data", uri.key);
    }
    {
        S3::URI uri(Poco::URI("https://storage.yandexcloud.net/jokserfn/data"));
        ASSERT_EQ("https://storage.yandexcloud.net", uri.endpoint);
        ASSERT_EQ("jokserfn", uri.bucket);
        ASSERT_EQ("data", uri.key);
    }
}

TEST_P(S3UriTest, invalidPatterns)
{
    ASSERT_ANY_THROW(S3::URI(Poco::URI(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(
    S3,
    S3UriTest,
    testing::Values(
        "https:///",
        "https://jokserfn.s3.yandexcloud.net/",
        "https://.s3.yandexcloud.net/key",
        "https://s3.yandexcloud.net/key",
        "https://jokserfn.s3yandexcloud.net/key",
        "https://s3.yandexcloud.net/key/",
        "https://s3.yandexcloud.net//",
        "https://yandexcloud.net/",
        "https://yandexcloud.net//",
        "https://yandexcloud.net/bucket/",
        "https://yandexcloud.net//key"));

}

#endif
