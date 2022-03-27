#include "OvercommitTracker.h"

#include <chrono>
#include <mutex>
#include <Interpreters/ProcessList.h>

using namespace std::chrono_literals;

constexpr std::chrono::microseconds ZERO_MICROSEC = 0us;

OvercommitTracker::OvercommitTracker(std::mutex & global_mutex_)
    : max_wait_time(ZERO_MICROSEC)
    , picked_tracker(nullptr)
    , cancelation_state(QueryCancelationState::NONE)
    , global_mutex(global_mutex_)
    , freed_momory(0)
    , required_memory(0)
{}

void OvercommitTracker::setMaxWaitTime(UInt64 wait_time)
{
    std::lock_guard guard(overcommit_m);
    max_wait_time = wait_time * 1us;
}

bool OvercommitTracker::needToStopQuery(MemoryTracker * tracker, Int64 amount)
{
    // NOTE: Do not change the order of locks
    //
    // global_mutex must be acquired before overcommit_m, because
    // method OvercommitTracker::unsubscribe(MemoryTracker *) is
    // always called with already acquired global_mutex in
    // ProcessListEntry::~ProcessListEntry().
    std::unique_lock<std::mutex> global_lock(global_mutex);
    std::unique_lock<std::mutex> lk(overcommit_m);

    if (max_wait_time == ZERO_MICROSEC)
        return true;

    pickQueryToExclude();
    assert(cancelation_state == QueryCancelationState::RUNNING);
    global_lock.unlock();

    // If no query was chosen we need to stop current query.
    // This may happen if no soft limit is set.
    if (picked_tracker == nullptr)
    {
        cancelation_state = QueryCancelationState::NONE;
        return true;
    }
    if (picked_tracker == tracker)
        return true;
    required_memory += amount;
    bool timeout = !cv.wait_for(lk, max_wait_time, [this]()
    {
        return freed_momory >= required_memory || cancelation_state == QueryCancelationState::NONE;
    });

    // If query cancelation is still running, it's possible that other queries will reach
    // hard limit and end up on waiting on condition variable.
    // If so we need to specify that some part of freed memory is acquired at this moment.
    if (!timeout && cancelation_state == QueryCancelationState::RUNNING)
        freed_momory -= amount;
    if (timeout)
        LOG_DEBUG(getLogger(), "Need to stop query because reached waiting timeout");
    else
        LOG_DEBUG(getLogger(), "Memory freed within timeout");
    required_memory -= amount;
    return timeout;
}

void OvercommitTracker::tryContinueQueryExecutionAfterFree(Int64 amount)
{
    std::lock_guard guard(overcommit_m);
    if (cancelation_state != QueryCancelationState::NONE)
    {
        freed_momory += amount;
        if (freed_momory >= required_memory)
            cv.notify_all();
    }
}

void OvercommitTracker::onQueryStop(MemoryTracker * tracker)
{
    std::unique_lock<std::mutex> lk(overcommit_m);
    if (picked_tracker == tracker)
    {
        LOG_DEBUG(getLogger(), "Picked query stopped");

        picked_tracker = nullptr;
        cancelation_state = QueryCancelationState::NONE;
        freed_momory = 0;
        cv.notify_all();
    }
}

UserOvercommitTracker::UserOvercommitTracker(DB::ProcessList * process_list, DB::ProcessListForUser * user_process_list_)
    : OvercommitTracker(process_list->mutex)
    , user_process_list(user_process_list_)
{}

void UserOvercommitTracker::pickQueryToExcludeImpl()
{
    MemoryTracker * query_tracker = nullptr;
    OvercommitRatio current_ratio{0, 0};
    // At this moment query list must be read only.
    // This is guaranteed by locking global_mutex in OvercommitTracker::needToStopQuery.
    auto & queries = user_process_list->queries;
    LOG_DEBUG(logger, "Trying to choose query to stop from {} queries", queries.size());
    for (auto const & query : queries)
    {
        if (query.second->isKilled())
            continue;

        auto * memory_tracker = query.second->getMemoryTracker();
        if (!memory_tracker)
            continue;

        auto ratio = memory_tracker->getOvercommitRatio();
        LOG_DEBUG(logger, "Query has ratio {}/{}", ratio.committed, ratio.soft_limit);
        if (ratio.soft_limit != 0 && current_ratio < ratio)
        {
            query_tracker = memory_tracker;
            current_ratio   = ratio;
        }
    }
    LOG_DEBUG(logger, "Selected to stop query with overcommit ratio {}/{}",
        current_ratio.committed, current_ratio.soft_limit);
    picked_tracker = query_tracker;
}

GlobalOvercommitTracker::GlobalOvercommitTracker(DB::ProcessList * process_list_)
    : OvercommitTracker(process_list_->mutex)
    , process_list(process_list_)
{}

void GlobalOvercommitTracker::pickQueryToExcludeImpl()
{
    MemoryTracker * query_tracker = nullptr;
    OvercommitRatio current_ratio{0, 0};
    // At this moment query list must be read only.
    // This is guaranteed by locking global_mutex in OvercommitTracker::needToStopQuery.
    LOG_DEBUG(logger, "Trying to choose query to stop from {} queries", process_list->size());
    for (auto const & query : process_list->processes)
    {
        if (query.isKilled())
            return;

        Int64 user_soft_limit = 0;
        if (auto const * user_process_list = query.getUserProcessList())
            user_soft_limit = user_process_list->user_memory_tracker.getSoftLimit();
        if (user_soft_limit == 0)
            return;

        auto * memory_tracker = query.getMemoryTracker();
        if (!memory_tracker)
            return;
        auto ratio = memory_tracker->getOvercommitRatio(user_soft_limit);
        LOG_DEBUG(logger, "Query has ratio {}/{}", ratio.committed, ratio.soft_limit);
        if (current_ratio < ratio)
        {
            query_tracker = memory_tracker;
            current_ratio   = ratio;
        }
    }
    LOG_DEBUG(logger, "Selected to stop query with overcommit ratio {}/{}",
        current_ratio.committed, current_ratio.soft_limit);
    picked_tracker = query_tracker;
}
