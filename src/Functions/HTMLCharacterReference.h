#pragma once

#include <cstdlib>

// Definition of the class generated by gperf
class HTMLCharacterHash
{
private:
    static inline unsigned int hash(const char * str, size_t len);

public:
    static const struct NameAndGlyph * Lookup(const char * str, size_t len);
};

// Definition of the struct generated by gperf
struct NameAndGlyph
{
    const char * name;
    const char * glyph;
};
