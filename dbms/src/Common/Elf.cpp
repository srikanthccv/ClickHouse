#include <Common/Elf.h>
#include <Common/Exception.h>

#include <iostream>


namespace DB
{

namespace ErrorCodes
{
    extern const int CANNOT_PARSE_ELF;
}


Elf::Elf(const std::string & path)
    : in(path, 0)
{
    std::cerr << "Processing path " << path << "\n";

    /// Check if it's an elf.
    size = in.buffer().size();
    if (size < sizeof(ElfEhdr))
        throw Exception("The size of supposedly ELF file is too small", ErrorCodes::CANNOT_PARSE_ELF);

    mapped = in.buffer().begin();
    header = reinterpret_cast<const ElfEhdr *>(mapped);

    if (memcmp(header->e_ident, "\x7F""ELF", 4) != 0)
        throw Exception("The file is not ELF according to magic", ErrorCodes::CANNOT_PARSE_ELF);

    /// Get section header.
    ElfOff section_header_offset = header->e_shoff;
    uint16_t section_header_num_entries = header->e_shnum;

    if (!section_header_offset
        || !section_header_num_entries
        || section_header_offset + section_header_num_entries * sizeof(ElfShdr) > size)
        throw Exception("The ELF is truncated (section header points after end of file)", ErrorCodes::CANNOT_PARSE_ELF);

    section_headers = reinterpret_cast<const ElfShdr *>(mapped + section_header_offset);

    /// The string table with section names.
    auto section_names_strtab = findSection([&](const Section & section, size_t idx)
    {
        return section.header.sh_type == SHT_STRTAB && header->e_shstrndx == idx;
    });

    if (!section_names_strtab)
        throw Exception("The ELF doesn't have string table with section names", ErrorCodes::CANNOT_PARSE_ELF);

    ElfOff section_names_offset = section_names_strtab->header.sh_offset;
    if (section_names_offset >= size)
        throw Exception("The ELF is truncated (section names string table points after end of file)", ErrorCodes::CANNOT_PARSE_ELF);

    section_names = reinterpret_cast<const char *>(mapped + section_names_offset);
}


Elf::Section::Section(const ElfShdr & header, const Elf & elf)
    : header(header), elf(elf)
{
}


bool Elf::iterateSections(std::function<bool(const Section & section, size_t idx)> && pred) const
{
    for (size_t idx = 0; idx < header->e_shnum; ++idx)
    {
        Section section(section_headers[idx], *this);

        /// Sections spans after end of file.
        if (section.header.sh_offset + section.header.sh_size > size)
            continue;

        if (pred(section, idx))
            return true;
    }
    return false;
}


std::optional<Elf::Section> Elf::findSection(std::function<bool(const Section & section, size_t idx)> && pred) const
{
    std::optional<Elf::Section> result;

    iterateSections([&](const Section & section, size_t idx)
    {
        if (pred(section, idx))
        {
            result.emplace(section);
            return true;
        }
        return false;
    });

    return result;
}


const char * Elf::Section::name() const
{
    if (!elf.section_names)
        throw Exception("Section names are not initialized", ErrorCodes::CANNOT_PARSE_ELF);

    /// TODO buffer overflow is possible, we may need to check strlen.
    return elf.section_names + header.sh_name;
}


const char * Elf::Section::begin() const
{
    return elf.mapped + header.sh_offset;
}

const char * Elf::Section::end() const
{
    return elf.mapped + header.sh_offset + header.sh_size;
}

}
