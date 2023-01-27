#include <IO/S3/getObjectInfo.h>

#if USE_AWS_S3
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectAttributesRequest.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>


namespace ErrorCodes
{
    extern const int S3_ERROR;
}


namespace ProfileEvents
{
    extern const Event S3GetObject;
    extern const Event S3GetObjectAttributes;
    extern const Event S3HeadObject;
    extern const Event DiskS3GetObject;
    extern const Event DiskS3GetObjectAttributes;
    extern const Event DiskS3HeadObject;
}


namespace DB::S3
{

namespace
{
    Aws::S3::Model::HeadObjectOutcome headObject(
        const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id, bool for_disk_s3)
    {
        ProfileEvents::increment(ProfileEvents::S3HeadObject);
        if (for_disk_s3)
            ProfileEvents::increment(ProfileEvents::DiskS3HeadObject);

        Aws::S3::Model::HeadObjectRequest req;
        req.SetBucket(bucket);
        req.SetKey(key);

        if (!version_id.empty())
            req.SetVersionId(version_id);

        return client.HeadObject(req);
    }

    /// Performs a request to get the size and last modification time of an object.
    /// The function performs either HeadObject or GetObjectAttributes request depending on the endpoint.
    std::pair<std::optional<DB::S3::ObjectInfo>, Aws::S3::S3Error> tryGetObjectInfo(
        const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id,
        const S3Settings::RequestSettings & request_settings, bool for_disk_s3)
    {
        if (request_settings.allow_head_object_request)
        {
            auto outcome = headObject(client, bucket, key, version_id, for_disk_s3);
            if (outcome.IsSuccess())
            {
                const auto & result = outcome.GetResult();
                DB::S3::ObjectInfo object_info;
                object_info.size = static_cast<size_t>(result.GetContentLength());
                object_info.last_modification_time = result.GetLastModified().Millis() / 1000;
                return {object_info, {}};
            }

            return {std::nullopt, outcome.GetError()};
        }
        else
        {
            ProfileEvents::increment(ProfileEvents::S3GetObjectAttributes);
            if (for_disk_s3)
                ProfileEvents::increment(ProfileEvents::DiskS3GetObjectAttributes);

            Aws::S3::Model::GetObjectAttributesRequest req;
            req.SetBucket(bucket);
            req.SetKey(key);

            if (!version_id.empty())
                req.SetVersionId(version_id);

            req.SetObjectAttributes({Aws::S3::Model::ObjectAttributes::ObjectSize});

            auto outcome = client.GetObjectAttributes(req);
            if (outcome.IsSuccess())
            {
                const auto & result = outcome.GetResult();
                DB::S3::ObjectInfo object_info;
                object_info.size = static_cast<size_t>(result.GetObjectSize());
                object_info.last_modification_time = result.GetLastModified().Millis() / 1000;
                return {object_info, {}};
            }

            return {std::nullopt, outcome.GetError()};
        }
    }
}


bool isNotFoundError(Aws::S3::S3Errors error)
{
    return error == Aws::S3::S3Errors::RESOURCE_NOT_FOUND || error == Aws::S3::S3Errors::NO_SUCH_KEY;
}

ObjectInfo getObjectInfo(const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id, const S3Settings::RequestSettings & settings, bool for_disk_s3, bool throw_on_error)
{
    auto [object_info, error] = tryGetObjectInfo(client, bucket, key, version_id, settings, for_disk_s3);
    if (object_info)
    {
        return *object_info;
    }
    else if (throw_on_error)
    {
        throw DB::Exception(ErrorCodes::S3_ERROR,
            "Failed to get object attributes: {}. HTTP response code: {}",
            error.GetMessage(), static_cast<size_t>(error.GetResponseCode()));
    }
    return {};
}

size_t getObjectSize(const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id, const S3Settings::RequestSettings & settings, bool for_disk_s3, bool throw_on_error)
{
    return getObjectInfo(client, bucket, key, version_id, settings, for_disk_s3, throw_on_error).size;
}

bool objectExists(const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id, const S3Settings::RequestSettings & settings, bool for_disk_s3)
{
    auto [object_info, error] = tryGetObjectInfo(client, bucket, key, version_id, settings, for_disk_s3);
    if (object_info)
        return true;

    if (isNotFoundError(error.GetErrorType()))
        return false;

    throw S3Exception(error.GetErrorType(),
        "Failed to check existence of key {} in bucket {}: {}",
        key, bucket, error.GetMessage());
}

void checkObjectExists(const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id, const S3Settings::RequestSettings & settings, bool for_disk_s3, std::string_view description)
{
    auto [object_info, error] = tryGetObjectInfo(client, bucket, key, version_id, settings, for_disk_s3);
    if (object_info)
        return;
    throw S3Exception(error.GetErrorType(), "{}Object {} in bucket {} suddenly disappeared: {}",
                        (description.empty() ? "" : (String(description) + ": ")), key, bucket, error.GetMessage());
}

std::map<String, String> getObjectMetadata(const Aws::S3::S3Client & client, const String & bucket, const String & key, const String & version_id, const S3Settings::RequestSettings & request_settings, bool for_disk_s3, bool throw_on_error)
{
    if (request_settings.allow_head_object_request)
    {
        auto outcome = headObject(client, bucket, key, version_id, for_disk_s3);

        if (outcome.IsSuccess())
        {
            return outcome.GetResult().GetMetadata();
        }
        else if (throw_on_error)
        {
            const auto & error = outcome.GetError();
            throw S3Exception(error.GetErrorType(), "Failed to get metadata of key {} in bucket {}: {}", key, bucket, error.GetMessage());
        }
        else
            return {};
    }
    else
    {
        ProfileEvents::increment(ProfileEvents::S3GetObject);
        if (for_disk_s3)
            ProfileEvents::increment(ProfileEvents::DiskS3GetObject);

        Aws::S3::Model::GetObjectRequest req;
        req.SetBucket(bucket);
        req.SetKey(key);

        /// Only the first byte will be read.
        /// We don't need that first byte but the range should be set otherwise the entire object will be read.
        req.SetRange("bytes=0-0");

        if (!version_id.empty())
            req.SetVersionId(version_id);

        auto outcome = client.GetObject(req);

        if (outcome.IsSuccess())
        {
            return outcome.GetResult().GetMetadata();
        }
        else if (throw_on_error)
        {
            const auto & error = outcome.GetError();
            throw S3Exception(error.GetErrorType(),
                "Failed to get metadata of key {} in bucket {}: {}",
                key, bucket, error.GetMessage());
        }
        else
            return {};
    }
}

}

#endif
