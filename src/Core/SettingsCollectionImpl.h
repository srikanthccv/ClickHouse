#pragma once

/**
  * This file implements some functions that are dependent on Field type.
  * Unlike SettingsCollection.h, we only have to include it once for each
  * instantiation of SettingsCollection<>.
  */

#include <Core/SettingsCollection.h>
#include <Common/SettingsChanges.h>
#include <Common/FieldVisitors.h>
#include <common/StringRef.h>


namespace DB
{
namespace details
{
    struct SettingsCollectionUtils
    {
        static void serializeName(const StringRef & name, WriteBuffer & buf);
        static String deserializeName(ReadBuffer & buf);
        static void serializeFlag(bool flag, WriteBuffer & buf);
        static bool deserializeFlag(ReadBuffer & buf);
        static void skipValue(ReadBuffer & buf);
        static void warningNameNotFound(const StringRef & name);
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
    return member->get_value(*collection);
}


template <class Derived>
Field SettingsCollection<Derived>::castValue(size_t index, const Field & value)
{
    try
    {
        return members()[index].cast_value(value);
    }
    catch (Exception & e)
    {
        e.addMessage(fmt::format("in attempt to set the value of setting to {}",
                                 applyVisitor(FieldVisitorToString(), value)));
        throw;
    }
}


template <class Derived>
Field SettingsCollection<Derived>::castValue(const StringRef & name, const Field & value)
{
    return members().findStrict(name).cast_value(value);
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
bool SettingsCollection<Derived>::operator ==(const SettingsCollection<Derived> & rhs) const
{
    const auto & the_members = members();
    for (size_t i = 0; i != the_members.size(); ++i)
    {
        const auto & member = the_members[i];
        bool left_changed = member.is_changed(castToDerived());
        bool right_changed = member.is_changed(rhs.castToDerived());
        if (left_changed || right_changed)
        {
            if (left_changed != right_changed)
                return false;
            if (member.get_value(castToDerived()) != member.get_value(rhs.castToDerived()))
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
            found_changes.push_back({member.name.toString(), member.get_value(castToDerived())});
    }
    return found_changes;
}


template <class Derived>
std::string SettingsCollection<Derived>::dumpChangesToString() const
{
    std::stringstream ss;
    for (const auto & c : changes())
    {
        ss << c.name << " = "
            << applyVisitor(FieldVisitorToString(), c.value) << "\n";
    }
    return ss.str();
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
            member.set_value(castToDerived(), member.get_value(src));
    }
}


template <class Derived>
void SettingsCollection<Derived>::copyChangesTo(Derived & dest) const
{
    dest.copyChangesFrom(castToDerived());
}


template <class Derived>
void SettingsCollection<Derived>::serialize(WriteBuffer & buf, SettingsBinaryFormat format) const
{
    const auto & the_members = members();
    for (size_t i = 0; i != the_members.size(); ++i)
    {
        const auto & member = the_members[i];
        if (member.is_changed(castToDerived()))
        {
            details::SettingsCollectionUtils::serializeName(member.name, buf);
            if (format >= SettingsBinaryFormat::STRINGS)
            {
                details::SettingsCollectionUtils::serializeFlag(member.is_important, buf);
                details::SettingsCollectionUtils::serializeName(member.get_value_as_string(castToDerived()), buf);
            }
            else
                member.write_binary(castToDerived(), buf);
        }
    }
    details::SettingsCollectionUtils::serializeName(StringRef{} /* empty string is a marker of the end of settings */, buf);
}


template <class Derived>
void SettingsCollection<Derived>::deserialize(ReadBuffer & buf, SettingsBinaryFormat format)
{
    const auto & the_members = members();
    while (true)
    {
        String name = details::SettingsCollectionUtils::deserializeName(buf);
        if (name.empty() /* empty string is a marker of the end of settings */)
            break;
        auto * member = the_members.find(name);
        bool is_important = true;
        if (format >= SettingsBinaryFormat::STRINGS)
            is_important = details::SettingsCollectionUtils::deserializeFlag(buf);
        if (!member)
        {
            if (is_important)
            {
                details::SettingsCollectionUtils::throwNameNotFound(name);
            }
            else
            {
                details::SettingsCollectionUtils::warningNameNotFound(name);
                details::SettingsCollectionUtils::skipValue(buf);
                continue;
            }
        }

        if (format >= SettingsBinaryFormat::STRINGS)
        {
            String value = details::SettingsCollectionUtils::deserializeName(buf);
            member->parse_value_from_string(castToDerived(), value);
        }
        else
            member->read_binary(castToDerived(), buf);
    }
}


//-V:IMPLEMENT_SETTINGS_COLLECTION:501
#define IMPLEMENT_SETTINGS_COLLECTION(DERIVED_CLASS_NAME, LIST_OF_SETTINGS_MACRO) \
    template<> \
    SettingsCollection<DERIVED_CLASS_NAME>::MemberInfos::MemberInfos() \
    { \
        using Derived = DERIVED_CLASS_NAME; \
        struct Functions \
        { \
            LIST_OF_SETTINGS_MACRO(IMPLEMENT_SETTINGS_COLLECTION_DEFINE_FUNCTIONS_HELPER_) \
        }; \
        constexpr int IMPORTANT = 1; \
        UNUSED(IMPORTANT); \
        LIST_OF_SETTINGS_MACRO(IMPLEMENT_SETTINGS_COLLECTION_ADD_MEMBER_INFO_HELPER_) \
    } \
    /** \
      * Instantiation should happen when all method definitions from SettingsCollectionImpl.h \
      * are accessible, so we instantiate explicitly. \
      */ \
    template class SettingsCollection<DERIVED_CLASS_NAME>;


#define IMPLEMENT_SETTINGS_COLLECTION_DEFINE_FUNCTIONS_HELPER_(TYPE, NAME, DEFAULT, DESCRIPTION, FLAGS) \
    static Field NAME##_getValue(const Derived & collection) { return static_cast<Field>(collection.NAME); } \
    static void NAME##_setValue(Derived & collection, const Field & value) { collection.NAME = value; } \
    static String NAME##_toString(const Derived & collection) { return collection.NAME.toString(); } \
    static void NAME##_parseFromString(Derived & collection, const String & value) { collection.NAME.parseFromString(value); } \
    static void NAME##_writeBinary(const Derived & collection, WriteBuffer & buf) { collection.NAME.writeBinary(buf); } \
    static void NAME##_readBinary(Derived & collection, ReadBuffer & buf) { collection.NAME.readBinary(buf); } \
    static Field NAME##_castValue(const Field & value) { return static_cast<Field>(SettingField##TYPE{value}); } \
    static Field NAME##_stringToValue(const String & str) { SettingField##TYPE temp; temp.parseFromString(str); return static_cast<Field>(temp); } \
    static String NAME##_valueToString(const Field & value) { return SettingField##TYPE{value}.toString(); } \


#define IMPLEMENT_SETTINGS_COLLECTION_ADD_MEMBER_INFO_HELPER_(TYPE, NAME, DEFAULT, DESCRIPTION, FLAGS) \
    add({StringRef(#NAME, strlen(#NAME)), \
         StringRef(DESCRIPTION, strlen(DESCRIPTION)), \
         StringRef(#TYPE, strlen(#TYPE)), \
         FLAGS & IMPORTANT, \
         [](const Derived & d) { return d.NAME.changed; }, \
         &Functions::NAME##_getValue, &Functions::NAME##_setValue, \
         &Functions::NAME##_toString, &Functions::NAME##_parseFromString, \
         &Functions::NAME##_writeBinary, &Functions::NAME##_readBinary, \
         &Functions::NAME##_castValue, &Functions::NAME##_stringToValue, &Functions::NAME##_valueToString });
}
