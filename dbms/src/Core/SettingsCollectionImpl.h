#pragma once

/**
  * This file implements some functions that are dependent on Field type.
  * Unlike SettingsCollection.h, we only have to include it once for each
  * instantiation of SettingsCollection<>.
  */

#include <Common/SettingsChanges.h>

namespace DB
{
namespace details
{
    struct SettingsCollectionUtils
    {
        static void serializeName(const StringRef & name, WriteBuffer & buf);
        static String deserializeName(ReadBuffer & buf);
        [[noreturn]] static void throwNameNotFound(const StringRef & name);
    };
}


template <class Derived>
size_t SettingsCollection<Derived>::MemberInfos::findIndex(const StringRef & name) const
{
    auto it = by_name_map.find(name);
    if (it == by_name_map.end())
        return static_cast<size_t>(-1); // npos
    return it->second;
}


template <class Derived>
size_t SettingsCollection<Derived>::MemberInfos::findIndexStrict(const StringRef & name) const
{
    auto it = by_name_map.find(name);
    if (it == by_name_map.end())
        details::SettingsCollectionUtils::throwNameNotFound(name);
    return it->second;
}


template <class Derived>
const typename SettingsCollection<Derived>::MemberInfo * SettingsCollection<Derived>::MemberInfos::find(const StringRef & name) const
{
    auto it = by_name_map.find(name);
    if (it == by_name_map.end())
        return nullptr;
    else
        return &infos[it->second];
}


template <class Derived>
const typename SettingsCollection<Derived>::MemberInfo & SettingsCollection<Derived>::MemberInfos::findStrict(const StringRef & name) const
{
    return infos[findIndexStrict(name)];
}


template <class Derived>
void SettingsCollection<Derived>::MemberInfos::add(MemberInfo && member)
{
    size_t index = infos.size();
    infos.emplace_back(member);
    by_name_map.emplace(infos.back().name, index);
}


template <class Derived>
const typename SettingsCollection<Derived>::MemberInfos &
SettingsCollection<Derived>::members()
{
    static const MemberInfos the_instance;
    return the_instance;
}


template <class Derived>
Field SettingsCollection<Derived>::const_reference::getValue() const
{
    return member->get_field(*collection);
}


template <class Derived>
Field SettingsCollection<Derived>::valueToCorrespondingType(size_t index, const Field & value)
{
    return members()[index].value_to_corresponding_type(value);
}


template <class Derived>
Field SettingsCollection<Derived>::valueToCorrespondingType(const StringRef & name, const Field & value)
{
    return members().findStrict(name).value_to_corresponding_type(value);
}


template <class Derived>
typename SettingsCollection<Derived>::iterator SettingsCollection<Derived>::find(const StringRef & name)
{
    const auto * member = members().find(name);
    if (member)
        return iterator(castToDerived(), member);
    return end();
}


template <class Derived>
typename SettingsCollection<Derived>::const_iterator SettingsCollection<Derived>::find(const StringRef & name) const
{
    const auto * member = members().find(name);
    if (member)
        return const_iterator(castToDerived(), member);
    return end();
}


template <class Derived>
typename SettingsCollection<Derived>::iterator SettingsCollection<Derived>::findStrict(const StringRef & name)
{
    return iterator(castToDerived(), &members().findStrict(name));
}


template <class Derived>
typename SettingsCollection<Derived>::const_iterator SettingsCollection<Derived>::findStrict(const StringRef & name) const
{
    return const_iterator(castToDerived(), &members().findStrict(name));
}


template <class Derived>
Field SettingsCollection<Derived>::get(size_t index) const
{
    return (*this)[index].getValue();
}


template <class Derived>
Field SettingsCollection<Derived>::get(const StringRef & name) const
{
    return (*this)[name].getValue();
}


template <class Derived>
bool SettingsCollection<Derived>::tryGet(const StringRef & name, Field & value) const
{
    auto it = find(name);
    if (it == end())
        return false;
    value = it->getValue();
    return true;
}


template <class Derived>
bool SettingsCollection<Derived>::tryGet(const StringRef & name, String & value) const
{
    auto it = find(name);
    if (it == end())
        return false;
    value = it->getValueAsString();
    return true;
}


template <class Derived>
bool SettingsCollection<Derived>::operator ==(const Derived & rhs) const
{
    const auto & the_members = members();
    for (size_t i = 0; i != the_members.size(); ++i)
    {
        const auto & member = the_members[i];
        bool left_changed = member.is_changed(castToDerived());
        bool right_changed = member.is_changed(rhs);
        if (left_changed || right_changed)
        {
            if (left_changed != right_changed)
                return false;
            if (member.get_field(castToDerived()) != member.get_field(rhs))
                return false;
        }
    }
    return true;
}


template <class Derived>
SettingsChanges SettingsCollection<Derived>::changes() const
{
    SettingsChanges found_changes;
    const auto & the_members = members();
    for (size_t i = 0; i != the_members.size(); ++i)
    {
        const auto & member = the_members[i];
        if (member.is_changed(castToDerived()))
            found_changes.push_back({member.name.toString(), member.get_field(castToDerived())});
    }
    return found_changes;
}


template <class Derived>
void SettingsCollection<Derived>::applyChange(const SettingChange & change)
{
    set(change.name, change.value);
}


template <class Derived>
void SettingsCollection<Derived>::applyChanges(const SettingsChanges & changes)
{
    for (const SettingChange & change : changes)
        applyChange(change);
}


template <class Derived>
void SettingsCollection<Derived>::copyChangesFrom(const Derived & src)
{
    const auto & the_members = members();
    for (size_t i = 0; i != the_members.size(); ++i)
    {
        const auto & member = the_members[i];
        if (member.is_changed(src))
            member.set_field(castToDerived(), member.get_field(src));
    }
}


template <class Derived>
void SettingsCollection<Derived>::copyChangesTo(Derived & dest) const
{
    dest.copyChangesFrom(castToDerived());
}


template <class Derived>
void SettingsCollection<Derived>::serialize(WriteBuffer & buf) const
{
    const auto & the_members = members();
    for (size_t i = 0; i != the_members.size(); ++i)
    {
        const auto & member = the_members[i];
        if (member.is_changed(castToDerived()))
        {
            details::SettingsCollectionUtils::serializeName(member.name, buf);
            member.serialize(castToDerived(), buf);
        }
    }
    details::SettingsCollectionUtils::serializeName(StringRef{} /* empty string is a marker of the end of settings */, buf);
}


template <class Derived>
void SettingsCollection<Derived>::deserialize(ReadBuffer & buf)
{
    const auto & the_members = members();
    while (true)
    {
        String name = details::SettingsCollectionUtils::deserializeName(buf);
        if (name.empty() /* empty string is a marker of the end of settings */)
            break;
        auto * member = the_members.find(name);
        if (member)
            member->deserialize(castToDerived(), buf);
        else
            details::SettingsCollectionUtils::throwNameNotFound(name);
    }
}


#define IMPLEMENT_SETTINGS_COLLECTION(DERIVED_CLASS_NAME, LIST_OF_SETTINGS_MACRO) \
    template<> \
    SettingsCollection<DERIVED_CLASS_NAME>::MemberInfos::MemberInfos() \
    { \
        using Derived = DERIVED_CLASS_NAME; \
        struct Functions \
        { \
            LIST_OF_SETTINGS_MACRO(IMPLEMENT_SETTINGS_COLLECTION_DEFINE_FUNCTIONS_HELPER_) \
        }; \
        LIST_OF_SETTINGS_MACRO(IMPLEMENT_SETTINGS_COLLECTION_ADD_MEMBER_INFO_HELPER_) \
    } \
    /** \
      * Instantiation should happen when all method definitions from SettingsCollectionImpl.h \
      * are accessible, so we instantiate explicitly. \
      */ \
    template class SettingsCollection<DERIVED_CLASS_NAME>;


#define IMPLEMENT_SETTINGS_COLLECTION_DEFINE_FUNCTIONS_HELPER_(TYPE, NAME, DEFAULT, DESCRIPTION) \
    static String NAME##_getString(const Derived & collection) { return collection.NAME.toString(); } \
    static Field NAME##_getField(const Derived & collection) { return collection.NAME.toField(); } \
    static void NAME##_setString(Derived & collection, const String & value) { collection.NAME.set(value); } \
    static void NAME##_setField(Derived & collection, const Field & value) { collection.NAME.set(value); } \
    static void NAME##_serialize(const Derived & collection, WriteBuffer & buf) { collection.NAME.serialize(buf); } \
    static void NAME##_deserialize(Derived & collection, ReadBuffer & buf) { collection.NAME.deserialize(buf); } \
    static String NAME##_valueToString(const Field & value) { TYPE temp{DEFAULT}; temp.set(value); return temp.toString(); } \
    static Field NAME##_valueToCorrespondingType(const Field & value) { TYPE temp{DEFAULT}; temp.set(value); return temp.toField(); } \


#define IMPLEMENT_SETTINGS_COLLECTION_ADD_MEMBER_INFO_HELPER_(TYPE, NAME, DEFAULT, DESCRIPTION) \
    add({StringRef(#NAME, strlen(#NAME)), StringRef(DESCRIPTION, strlen(DESCRIPTION)), \
         [](const Derived & d) { return d.NAME.changed; }, \
         &Functions::NAME##_getString, &Functions::NAME##_getField, \
         &Functions::NAME##_setString, &Functions::NAME##_setField, \
         &Functions::NAME##_serialize, &Functions::NAME##_deserialize, \
         &Functions::NAME##_valueToString, &Functions::NAME##_valueToCorrespondingType});
}
