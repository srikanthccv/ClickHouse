#ifndef BOOST_SERIALIZATION_SLIST_HPP
#define BOOST_SERIALIZATION_SLIST_HPP

// MS compatible compilers support #pragma once
#if defined(_MSC_VER)
# pragma once
#endif

/////////1/////////2/////////3/////////4/////////5/////////6/////////7/////////8
// slist.hpp

// (C) Copyright 2002 Robert Ramey - http://www.rrsd.com . 
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

//  See http://www.boost.org for updates, documentation, and revision history.

#include <boost/config.hpp>
#ifdef BOOST_HAS_SLIST
#include BOOST_SLIST_HEADER

#include <boost/serialization/collections_save_imp.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/archive/detail/basic_iarchive.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/collection_size_type.hpp>
#include <boost/serialization/item_version_type.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/detail/stack_constructor.hpp>
#include <boost/serialization/detail/is_default_constructible.hpp>
#include <boost/move/utility_core.hpp>

namespace boost { 
namespace serialization {

template<class Archive, class U, class Allocator>
inline void save(
    Archive & ar,
    const BOOST_STD_EXTENSION_NAMESPACE::slist<U, Allocator> &t,
    const unsigned int file_version
){
    boost::serialization::stl::save_collection<
        Archive,
        BOOST_STD_EXTENSION_NAMESPACE::slist<U, Allocator> 
    >(ar, t);
}

namespace stl {

template<
    class Archive,
    class T,
    class Allocator
>
typename boost::disable_if<
    typename detail::is_default_constructible<
        typename BOOST_STD_EXTENSION_NAMESPACE::slist<T, Allocator>::value_type
    >,
    void
>::type
collection_load_impl(
    Archive & ar,
    BOOST_STD_EXTENSION_NAMESPACE::slist<T, Allocator> &t,
    collection_size_type count,
    item_version_type item_version
){
    t.clear();
    boost::serialization::detail::stack_construct<Archive, T> u(ar, item_version);
    ar >> boost::serialization::make_nvp("item", u.reference());
    t.push_front(boost::move(u.reference()));
    typename BOOST_STD_EXTENSION_NAMESPACE::slist<T, Allocator>::iterator last;
    last = t.begin();
    ar.reset_object_address(&(*t.begin()) , & u.reference());
    while(--count > 0){
        detail::stack_construct<Archive, T> u(ar, item_version);
        ar >> boost::serialization::make_nvp("item", u.reference());
        last = t.insert_after(last, boost::move(u.reference()));
        ar.reset_object_address(&(*last) , & u.reference());
    }
}

} // stl

template<class Archive, class U, class Allocator>
inline void load(
    Archive & ar,
    BOOST_STD_EXTENSION_NAMESPACE::slist<U, Allocator> &t,
    const unsigned int file_version
){
    const boost::archive::library_version_type library_version(
        ar.get_library_version()
    );
    // retrieve number of elements
    item_version_type item_version(0);
    collection_size_type count;
    ar >> BOOST_SERIALIZATION_NVP(count);
    if(boost::archive::library_version_type(3) < library_version){
        ar >> BOOST_SERIALIZATION_NVP(item_version);
    }
    if(detail::is_default_constructible<U>()){
        t.resize(count);
        typename BOOST_STD_EXTENSION_NAMESPACE::slist<U, Allocator>::iterator hint;
        hint = t.begin();
        while(count-- > 0){
            ar >> boost::serialization::make_nvp("item", *hint++);
        }
    }
    else{
        t.clear();
        boost::serialization::detail::stack_construct<Archive, U> u(ar, item_version);
        ar >> boost::serialization::make_nvp("item", u.reference());
        t.push_front(boost::move(u.reference()));
        typename BOOST_STD_EXTENSION_NAMESPACE::slist<U, Allocator>::iterator last;
        last = t.begin();
        ar.reset_object_address(&(*t.begin()) , & u.reference());
        while(--count > 0){
            detail::stack_construct<Archive, U> u(ar, item_version);
            ar >> boost::serialization::make_nvp("item", u.reference());
            last = t.insert_after(last, boost::move(u.reference()));
            ar.reset_object_address(&(*last) , & u.reference());
        }
    }
}

// split non-intrusive serialization function member into separate
// non intrusive save/load member functions
template<class Archive, class U, class Allocator>
inline void serialize(
    Archive & ar,
    BOOST_STD_EXTENSION_NAMESPACE::slist<U, Allocator> &t,
    const unsigned int file_version
){
    boost::serialization::split_free(ar, t, file_version);
}

} // serialization
} // namespace boost

#include <boost/serialization/collection_traits.hpp>

BOOST_SERIALIZATION_COLLECTION_TRAITS(BOOST_STD_EXTENSION_NAMESPACE::slist)

#endif  // BOOST_HAS_SLIST
#endif  // BOOST_SERIALIZATION_SLIST_HPP
