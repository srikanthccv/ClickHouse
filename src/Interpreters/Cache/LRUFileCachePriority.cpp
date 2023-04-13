#include <Interpreters/Cache/LRUFileCachePriority.h>
#include <Interpreters/Cache/FileCache.h>
#include <Common/CurrentMetrics.h>
#include <Common/randomSeed.h>
#include <Common/logger_useful.h>

namespace CurrentMetrics
{
    extern const Metric FilesystemCacheSize;
    extern const Metric FilesystemCacheElements;
}

namespace DB
{

namespace ErrorCodes
{
    extern const int LOGICAL_ERROR;
}

IFileCachePriority::Iterator LRUFileCachePriority::add(
    const Key & key,
    size_t offset,
    size_t size,
    KeyMetadataPtr key_metadata,
    const CacheGuard::Lock &)
{
#ifndef NDEBUG
    for (const auto & entry : queue)
    {
        if (entry.key == key && entry.offset == offset)
            throw Exception(
                ErrorCodes::LOGICAL_ERROR,
                "Attempt to add duplicate queue entry to queue. (Key: {}, offset: {}, size: {})",
                entry.key, entry.offset, entry.size);
    }
#endif

    const auto & size_limit = getSizeLimit();
    if (size_limit && current_size + size > size_limit)
    {
        throw Exception(
            ErrorCodes::LOGICAL_ERROR,
            "Not enough space to add {}:{} with size {}: current size: {}/{}",
            key, offset, size, current_size, getSizeLimit());
    }

    current_size += size;

    auto iter = queue.insert(queue.end(), Entry(key, offset, size, key_metadata));

    CurrentMetrics::add(CurrentMetrics::FilesystemCacheSize, size);
    CurrentMetrics::add(CurrentMetrics::FilesystemCacheElements);

    LOG_TEST(log, "Added entry into LRU queue, key: {}, offset: {}", key, offset);

    return std::make_shared<LRUFileCacheIterator>(this, iter);
}

void LRUFileCachePriority::removeAll(const CacheGuard::Lock &)
{
    CurrentMetrics::sub(CurrentMetrics::FilesystemCacheSize, current_size);
    CurrentMetrics::sub(CurrentMetrics::FilesystemCacheElements, queue.size());

    LOG_TEST(log, "Removed all entries from LRU queue");

    queue.clear();
    current_size = 0;
}

void LRUFileCachePriority::pop(const CacheGuard::Lock &)
{
    remove(queue.begin());
}

LRUFileCachePriority::LRUQueueIterator LRUFileCachePriority::remove(LRUQueueIterator it)
{
    current_size -= it->size;

    CurrentMetrics::sub(CurrentMetrics::FilesystemCacheSize, it->size);
    CurrentMetrics::sub(CurrentMetrics::FilesystemCacheElements);

    LOG_TEST(log, "Removed entry from LRU queue, key: {}, offset: {}", it->key, it->offset);
    return queue.erase(it);
}

LRUFileCachePriority::LRUFileCacheIterator::LRUFileCacheIterator(
    LRUFileCachePriority * cache_priority_, LRUFileCachePriority::LRUQueueIterator queue_iter_)
    : cache_priority(cache_priority_), queue_iter(queue_iter_)
{
}

void LRUFileCachePriority::iterate(IterateFunc && func, const CacheGuard::Lock &)
{
    for (auto it = queue.begin(); it != queue.end();)
    {
        auto locked_key = it->key_metadata->lock();
        if (locked_key->getKeyState() != KeyMetadata::KeyState::ACTIVE)
        {
            it = remove(it);
            continue;
        }

        auto result = func(*it, *locked_key);
        switch (result)
        {
            case IterationResult::BREAK:
            {
                return;
            }
            case IterationResult::CONTINUE:
            {
                ++it;
                break;
            }
            case IterationResult::REMOVE_AND_CONTINUE:
            {
                it = remove(it);
                break;
            }
        }
    }
}

LRUFileCachePriority::Iterator LRUFileCachePriority::LRUFileCacheIterator::remove(const CacheGuard::Lock &)
{
    return std::make_shared<LRUFileCacheIterator>(cache_priority, cache_priority->remove(queue_iter));
}

void LRUFileCachePriority::LRUFileCacheIterator::updateSize(ssize_t size)
{
    cache_priority->current_size += size;
    if (size > 0)
        CurrentMetrics::add(CurrentMetrics::FilesystemCacheSize, size);
    else
        CurrentMetrics::sub(CurrentMetrics::FilesystemCacheSize, size);
    queue_iter->size += size;
}

size_t LRUFileCachePriority::LRUFileCacheIterator::use(const CacheGuard::Lock &)
{
    cache_priority->queue.splice(cache_priority->queue.end(), cache_priority->queue, queue_iter);
    return ++queue_iter->hits;
}

};
