#include <chrono>
#include <variant>
#include <Interpreters/PreparedSets.h>
#include <Processors/QueryPlan/QueryPlan.h>
#include <Interpreters/InterpreterSelectWithUnionQuery.h>
#include <Interpreters/Set.h>

namespace DB
{

PreparedSetKey PreparedSetKey::forLiteral(const IAST & ast, DataTypes types_)
{
    /// Remove LowCardinality types from type list because Set doesn't support LowCardinality keys now,
    ///   just converts LowCardinality to ordinary types.
    for (auto & type : types_)
        type = recursiveRemoveLowCardinality(type);

    PreparedSetKey key;
    key.ast_hash = ast.getTreeHash();
    key.types = std::move(types_);
    return key;
}

PreparedSetKey PreparedSetKey::forSubquery(const IAST & ast)
{
    PreparedSetKey key;
    key.ast_hash = ast.getTreeHash();
    return key;
}

bool PreparedSetKey::operator==(const PreparedSetKey & other) const
{
    if (ast_hash != other.ast_hash)
        return false;

    if (types.size() != other.types.size())
        return false;

    for (size_t i = 0; i < types.size(); ++i)
    {
        if (!types[i]->equals(*other.types[i]))
            return false;
    }

    return true;
}

SubqueryForSet & PreparedSets::createOrGetSubquery(const String & subquery_id, const PreparedSetKey & key,
                                                   SizeLimits set_size_limit, bool transform_null_in)
{
    SubqueryForSet & subquery = subqueries[subquery_id];

    /// If you already created a Set with the same subquery / table for another ast
    /// In that case several PreparedSetKey would share same subquery and set
    /// Not sure if it's really possible case (maybe for distributed query when set was filled by external table?)
    if (subquery.set.valid())
        sets[key] = subquery.set; // TODO:
    else
    {
        subquery.set_in_progress = std::make_shared<Set>(set_size_limit, false, transform_null_in);
        sets[key] = subquery.promise_to_fill_set.get_future();
    }

    if (!subquery.set_in_progress)
    {
        subquery.key = key;
        subquery.set_in_progress = std::make_shared<Set>(set_size_limit, false, transform_null_in);
    }

    return subquery;
}

/// If the subquery is not associated with any set, create default-constructed SubqueryForSet.
/// It's aimed to fill external table passed to SubqueryForSet::createSource.
SubqueryForSet & PreparedSets::getSubquery(const String & subquery_id) { return subqueries[subquery_id]; }

void PreparedSets::set(const PreparedSetKey & key, SetPtr set_) { sets[key] = makeReadyFutureSet(set_); }

FutureSet PreparedSets::getFuture(const PreparedSetKey & key) const
{
    auto it = sets.find(key);
    if (it == sets.end())// || it->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return {};
    return it->second;
}

SetPtr PreparedSets::get(const PreparedSetKey & key) const
{
    auto it = sets.find(key);
    if (it == sets.end() || it->second.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return nullptr;
    return it->second.get();
}

std::vector<SetPtr> PreparedSets::getByTreeHash(IAST::Hash ast_hash) const
{
    std::vector<SetPtr> res;
    for (const auto & it : this->sets)
    {
        if (it.first.ast_hash == ast_hash)
            res.push_back(it.second.get());
    }
    return res;
}

PreparedSets::SubqueriesForSets PreparedSets::detachSubqueries()
{
    auto res = std::move(subqueries);
    subqueries = SubqueriesForSets();
    return res;
}

bool PreparedSets::empty() const { return sets.empty(); }

void SubqueryForSet::createSource(InterpreterSelectWithUnionQuery & interpreter, StoragePtr table_)
{
    source = std::make_unique<QueryPlan>();
    interpreter.buildQueryPlan(*source);
    if (table_)
        table = table_;
}

bool SubqueryForSet::hasSource() const
{
    return source != nullptr;
}

QueryPlanPtr SubqueryForSet::detachSource()
{
    auto res = std::move(source);
    source = nullptr;
    return res;
}


std::variant<std::promise<SetPtr>, FutureSet> PreparedSetsCache::findOrPromiseToBuild(const PreparedSetKey & key)
{
//    auto* log = &Poco::Logger::get("PreparedSetsCache");

    /// Look for existing entry in the cache.
    {
        std::lock_guard lock(cache_mutex);

        auto it = cache.find(key);
        if (it != cache.end())
        {
            /// If the set is being built, return its future, but if it's ready and is nullptr then we should retry building it.
            /// TODO: consider moving retry logic outside of the cache.
            if (it->second->future.valid() &&
                (it->second->future.wait_for(std::chrono::seconds(0)) != std::future_status::ready || it->second->future.get() != nullptr))
                return it->second->future;
        }

        {
            /// Insert the entry into the cache so that other threads can find it and start waiting for the set.
            std::promise<SetPtr> promise_to_fill_set;
            auto entry = std::make_shared<Entry>();
            entry->future = promise_to_fill_set.get_future();
            cache[key] = entry;
            return promise_to_fill_set;
        }
    }
}


FutureSet makeReadyFutureSet(SetPtr set)
{
    std::promise<SetPtr> promise;
    promise.set_value(set);
    return promise.get_future();
}

};
